"""Harbor verifier for property extraction.

Scores predictions against a ground-truth property list by (1) matching
predicted property names to GT property names using embedding similarity
followed by an LLM same-property judge, then (2) scoring values against GT
values using the rubric (0.1% SI, pymatgen, or default exact/categorical).

Follows the pipeline in ``compute_mean_recall_precision``: compute mean
recall (GT → pred direction) and mean precision (pred → GT direction), and
report the F1 of those two as the Harbor reward.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import logging
import re
import sys
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from json import JSONDecoder
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from google import genai
from google.genai import types
from pymatgen.core import Composition
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from llm_utils import (
    Conversation,
    InferenceGenerationConfig,
    LLMChat,
    LLMChatResponse,
    Message,
    get_llm,
)

logger = logging.getLogger(__name__)


#
# Constants
#
EMBEDDING_BATCH_SIZE = 100
EMBEDDING_MODEL_NAME = "gemini-embedding-001"
TOP_K = 3
SERVER = "gemini"
MODEL_NAME = "gemini-2.5-flash"
MAX_OUTPUT_TOKENS = 4096

# Prompt template used by generate_property_name_matches. JSON braces are
# doubled so str.format() leaves them intact.
PROPERTY_MATCHING_PROMPT = """You are an expert condensed matter physicist evaluating whether two property descriptions refer to the same physical measurement.

## Property 1
- Name: "{property_name_1}"
- Context: "{context_1}"

## Property 2
- Name: "{property_name_2}"
- Context: "{context_2}"

## Matching Rules:

**SAME property if:**
- Names are synonymous (e.g., "Tc" = "Critical Temperature" = "Superconducting Transition Temperature")
- Abbreviation differences only (e.g., "Jc" = "Critical Current Density")
- Capitalization/formatting differences

**DIFFERENT properties if:**
- Technically distinct measurements:
  - "Tc onset" ≠ "Tc zero" ≠ "Tc midpoint"
  - "Jc" ≠ "Ic" (density vs absolute current)
  - "lattice constant a" ≠ "lattice constant c"
  - "upper critical field Hc2" ≠ "lower critical field Hc1"
- Different measurement orientations when orientation matters (e.g., "resistivity (c-axis)" ≠ "resistivity (ab-plane)")
- Different conditions not reconcilable (e.g., "Tc at 0 GPa" ≠ "Tc at 10 GPa" unless pressure is tracked separately)

## Response Format:
Return JSON only:
{{
  "is_match": boolean,
  "confidence": "high" | "medium" | "low",
  "reason": "concise explanation",
  "matched_via": "direct" | "synonym" | "abbreviation" | "condition_reconciliation" | null
}}
"""


_SUPERSCRIPT_MAP = str.maketrans(
    {
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "⁺": "+",
        "⁻": "-",
    }
)


#
# Embedding + LLM-based property-name matching
#
def generate_embeddings(property_names: list[str]) -> list[np.ndarray]:
    """Generate embeddings for a list of property names."""
    client = genai.Client()
    embeddings = []
    for i in range(0, len(property_names), EMBEDDING_BATCH_SIZE):
        batch = property_names[i : i + EMBEDDING_BATCH_SIZE]
        result = client.models.embed_content(
            model=EMBEDDING_MODEL_NAME,
            contents=batch,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        )
        embeddings.extend([emb.values for emb in result.embeddings])

    return embeddings


async def check_if_same_property(
    llm: LLMChat,
    inf_gen_config: InferenceGenerationConfig,
    prompt: str,
    property_name_1: str,
    property_name_2: str,
) -> tuple[dict, dict]:
    """Check if two property names are the same using an LLM."""
    if property_name_1.strip() == property_name_2.strip():
        result = {
            "is_match": True,
            "reason": "Property names are identical",
            "confidence": "high",
            "matched_via": "exact",
            "judge": llm.model_name,
            "prompt": None,
        }
        return result, {}

    conv = Conversation(messages=[Message(role="user", content=[prompt])])

    response: LLMChatResponse = await llm.generate_response_async(conv, inf_gen_config)
    if response.pred:
        is_match = response.pred.get("is_match", False)
        reason = response.pred.get("reason", "No reason provided")
        confidence = response.pred.get("confidence")
        matched_via = response.pred.get("matched_via")
    else:
        is_match = False
        reason = "Empty response from LLM"
        confidence = None
        matched_via = None

    result = {
        "is_match": is_match,
        "reason": reason,
        "confidence": confidence,
        "matched_via": matched_via,
        "judge": llm.model_name,
        "prompt": prompt,
    }

    return result, {**response.model_dump(), "judge": llm.model_name}


async def generate_property_name_matches(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    llm: LLMChat,
    inf_gen_config: InferenceGenerationConfig,
    prompt_template: str,
    top_k: int = TOP_K,
    left_on: list[str] = ["property_name", "context"],
    right_on: list[str] = ["property_name", "context"],
    left_suffix: str = "_x",
    right_suffix: str = "_y",
) -> pd.DataFrame:
    """For each row in df1, find the top-k matches in df2 using embeddings + LLM."""
    Y = df2.drop_duplicates(subset=["property_name"])
    similarity_matrix = cosine_similarity(
        np.vstack(df1["embedding"].values),
        np.vstack(Y["embedding"].values),
    )
    top_k_matches_indices = np.argsort(similarity_matrix, axis=1)[:, ::-1][:, :top_k]

    matches = []
    tasks = OrderedDict()
    idx_to_task_id = {}
    for i in tqdm(range(len(df1)), desc="Processing df1"):
        x = df1.iloc[i].to_dict()
        top_k_matches = Y.iloc[top_k_matches_indices[i]]["property_name"].tolist()
        df2_top_k = df2[df2["property_name"].isin(top_k_matches)]
        logger.debug(f"Found {len(df2_top_k)} matches for {x['property_name']}")
        for idx, y in df2_top_k.iterrows():
            x_variables = {k + "_1": x[k] for k in left_on}
            y_variables = {k + "_2": y[k] for k in right_on}
            task_id = (json.dumps(x_variables), json.dumps(y_variables))
            idx_to_task_id[(i, idx)] = task_id
            if task_id not in tasks:
                prompt = prompt_template.format(
                    **x_variables,
                    **y_variables,
                )
                task = check_if_same_property(
                    llm, inf_gen_config, prompt, x["property_name"], y["property_name"]
                )
                tasks[task_id] = task

    batch_size = len(tasks)  # run all at once
    results_data = []
    for i in tqdm(range(0, len(tasks), batch_size), desc="Calling LLM API in batches"):
        batch_tasks = {k: tasks[k] for k in list(tasks.keys())[i : i + batch_size]}
        batch_results = await asyncio.gather(*batch_tasks.values())
        results_data.extend(batch_results)
    results = {
        task_id: result for task_id, (result, _) in zip(tasks.keys(), results_data)
    }

    for i in tqdm(range(len(df1)), desc="Processing df1"):
        x = df1.iloc[i].to_dict()
        top_k_matches = Y.iloc[top_k_matches_indices[i]]["property_name"].tolist()
        df2_top_k = df2[df2["property_name"].isin(top_k_matches)]
        for idx, y in df2_top_k.iterrows():
            result = results[idx_to_task_id[(i, idx)]]
            matches.append(
                {
                    **x,
                    **result,
                    "y_id": idx,
                }
            )
    df_matches = pd.DataFrame(matches)
    df_matches = df_matches.merge(
        df2,
        left_on="y_id",
        right_index=True,
        how="left",
        suffixes=(left_suffix, right_suffix),
    )

    responses = [response for _, response in results_data]
    df_responses = pd.json_normalize(responses)

    return df_matches, df_responses


def is_material_name_same(material1: str, material2: str) -> bool:
    """Check if two material names are the same."""
    return material1 == material2


#
# String normalization helpers (inlined from pbench_eval.utils)
#
def normalize_ws(text: str) -> str:
    """Collapse whitespace for more robust string matching."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_unicode(text: str) -> str:
    """Normalize common unicode variants (dashes, superscripts, delta)."""
    s = str(text or "")
    s = (
        s.replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
    )
    s = s.replace("δ", "d").replace("Δ", "d")
    s = s.translate(_SUPERSCRIPT_MAP)
    return s


def parse_numeric_candidates(value: str) -> list[tuple[float, str | None]]:
    """Extract (value, unit) tuples from a free-form value string."""
    if value is None:
        return []

    value_str = normalize_unicode(str(value)).strip()
    if value_str.upper() == "NOT_FOUND" or value_str == "":
        return []

    value_str = re.sub(r"\(\d+\)", "", value_str)

    candidates: list[tuple[float, str | None, int]] = []
    sci_notation_positions: set[tuple[int, int]] = set()

    sci_pattern = re.compile(
        r"(?P<base>[-+]?\d*\.?\d+)\s*(?:(?:x|×)\s*10(?:\s*\^)?|[eE])\s*(?P<exp>[-+]?\d+)\s*(?P<unit>[a-zA-Z0-9/°%.]+)?",
        re.IGNORECASE,
    )
    for match in sci_pattern.finditer(value_str):
        try:
            base = float(match.group("base"))
            exp = int(match.group("exp"))
            unit = match.group("unit")
            candidates.append((base * (10**exp), unit, match.end()))
            sci_notation_positions.add((match.start(), match.end()))
        except Exception:
            continue

    num_pattern = re.compile(
        r"(?P<num>[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)\s*(?P<unit>[a-zA-Z0-9/°%.]+)?"
    )
    for match in num_pattern.finditer(value_str):
        try:
            match_start = match.start()
            match_end = match.end()
            overlaps = any(
                not (match_end <= sci_start or match_start >= sci_end)
                for sci_start, sci_end in sci_notation_positions
            )
            if overlaps:
                continue

            num = float(match.group("num"))
            unit = match.group("unit")
            candidates.append((num, unit, match.end()))
        except Exception:
            continue

    seen: set[str] = set()
    unique: list[tuple[float, str | None]] = []
    for num, unit, end_pos in candidates:
        key = f"{num:.12g}_{unit}_{end_pos}"
        if key in seen:
            continue
        seen.add(key)
        unique.append((num, unit))

    return unique


def normalize_formula(formula: str) -> str:
    """Normalize a chemical formula to alphabetical element order."""
    comp = Composition(formula)
    return comp.alphabetical_formula


#
# Formula normalization (inlined from pbench_eval.normalize_material)
#
def strip_formula(text: str) -> tuple[str, dict[str, float]]:
    """Extract chemical formula and variable values from text."""
    text = text.strip()
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"(\d|\))\s+([A-Z])", r"\1\2", text)

    variable_values = {}

    var_pattern = (
        r"\(\s*([xyzδ])\s*=\s*([0-9.]+)\s*(?:,\s*([xyzδ])\s*=\s*([0-9.]+)\s*)*\)"
    )
    var_match = re.search(var_pattern, text)
    if var_match:
        var_str = var_match.group(0)
        pairs = re.findall(r"([xyzδ])\s*=\s*([0-9.]+)", var_str)
        for var, val in pairs:
            variable_values[var] = float(val)
        text = text[: var_match.start()] + text[var_match.end() :]
        text = text.strip()

    formula_pattern = r"^([A-Z][a-zA-Z0-9()\-+δ.,/]*)"

    match = re.match(formula_pattern, text)
    if match:
        return match.group(1).rstrip(".,"), variable_values

    return text, variable_values


def classify_and_normalize(
    formula: str, variable_values: dict | None = None
) -> tuple[str, str, list[str]]:
    """Classify formula type and normalize generic/templated formulas."""
    if variable_values is None:
        variable_values = {}

    oxygen_defaults = {"x": 1, "X": 1, "z": 0, "δ": 0, "delta": 0}
    param_defaults = {"x": 0, "y": 0, "z": 0, "δ": 0, "delta": 0}

    oxygen_defaults.update(variable_values)
    param_defaults.update(variable_values)

    has_oxygen_variable = bool(
        re.search(r"O[xXzδ]|O\([^)]+\)|O\d+[+\-][xyzδ]", formula)
    )
    has_parameter = bool(
        re.search(
            r"[A-Z][a-z]?\([^)]*[xyz][^)]*\)[A-Z]|"
            r"[A-Z][a-z]?[xyz][A-Z]|"
            r"[A-BD-NP-Z][a-z]?\d+[+\-][xyz](?=[A-Z]|$)",
            formula,
        )
    )
    has_mixed_site = bool(re.search(r"\([A-Z][a-z]?,", formula))

    normalized = formula
    notes: list[str] = []

    if has_oxygen_variable:
        normalized = re.sub(
            r"O([xXzδ])\b",
            lambda m: f"O{oxygen_defaults.get(m.group(1), 1)}",
            normalized,
        )

        def eval_oxygen_expr(expr: str) -> str | None:
            for var, val in oxygen_defaults.items():
                expr = re.sub(rf"\b{re.escape(var)}\b", str(val), expr)
            try:
                return str(eval(expr))
            except Exception:
                return None

        def eval_oxygen_paren(match: re.Match) -> str | None:
            expr = match.group(1)
            result = eval_oxygen_expr(expr)
            return f"O{result}" if result else match.group(0)

        normalized = re.sub(r"O\(([^)]+)\)", eval_oxygen_paren, normalized)

        def eval_oxygen_no_paren(match: re.Match) -> str | None:
            expr = match.group(1)
            result = eval_oxygen_expr(expr)
            return f"O{result}" if result else match.group(0)

        normalized = re.sub(r"O(\d+[+\-][xyzδ])", eval_oxygen_no_paren, normalized)

        notes.append(f"Oxygen variable normalized with defaults: {oxygen_defaults}")

    if has_mixed_site:

        def split_mixed_site(match: re.Match) -> str:
            elements_str = match.group(1)
            total = float(match.group(2))
            elements = [e.strip() for e in elements_str.split(",")]
            per_element = total / len(elements)
            return "".join([f"{elem}{per_element}" for elem in elements])

        normalized = re.sub(
            r"\(([A-Z][a-z]?(?:\s*,\s*[A-Z][a-z]?)+)\)(\d+(?:\.\d+)?)",
            split_mixed_site,
            normalized,
        )
        notes.append("Mixed site normalized with equal distribution")

    if has_parameter and variable_values:

        def eval_param_expr(expr: str) -> float | None:
            for var, val in param_defaults.items():
                expr = re.sub(rf"\b{re.escape(var)}\b", str(val), expr)
            try:
                return eval(expr)
            except Exception:
                return None

        def eval_element_paren(match: re.Match) -> str | None:
            element = match.group(1)
            expr = match.group(2)
            result = eval_param_expr(expr)
            return f"{element}{result}" if result is not None else match.group(0)

        normalized = re.sub(r"([A-Z][a-z]?)\(([^)]+)\)", eval_element_paren, normalized)

        def eval_element_no_paren(match: re.Match) -> str | None:
            element = match.group(1)
            expr = match.group(2)
            result = eval_param_expr(expr)
            return f"{element}{result}" if result is not None else match.group(0)

        normalized = re.sub(
            r"([A-Z][a-z]?)(\d+[+\-][xyz])(?=[A-Z]|$)",
            eval_element_no_paren,
            normalized,
        )

        for var, val in param_defaults.items():
            normalized = re.sub(
                rf"([A-Z][a-z]?){re.escape(var)}(?![a-z])", rf"\g<1>{val}", normalized
            )

        notes.append(f"Compositional parameters substituted with: {variable_values}")

    if has_parameter and "x" not in variable_values:
        formula_type = "PARAMETER_FORMULA"
        notes.append(
            "WARNING: Contains compositional parameters - cannot fully normalize without specific x value"
        )
    elif re.search(r"[a-z]\)", normalized) or re.search(r"[δδ]", normalized):
        formula_type = "PARTIAL_NORMALIZATION"
        notes.append("Partially normalized - some expressions remain")
    else:
        try:
            Composition(normalized)
            formula_type = "STOICHIOMETRIC"
        except Exception:
            formula_type = "INVALID"
            notes.append("Normalization produced invalid formula")

    return normalized, formula_type, notes


#
# Scorers (inlined from pbench_eval.utils)
#
def scorer_pymatgen(pred: str, answer: str) -> bool:
    """Check if pred composition is close to answer composition."""
    assert isinstance(pred, str), "pred must be a string"
    assert isinstance(answer, str), "answer must be a string"

    pred, pred_vars = strip_formula(pred)
    answer, answer_vars = strip_formula(answer)

    pred_formula, pred_formula_type, pred_notes = classify_and_normalize(
        pred, pred_vars
    )
    answer_formula, answer_formula_type, answer_notes = classify_and_normalize(
        answer, answer_vars
    )

    unparseable_types = {"INVALID", "PARAMETER_FORMULA", "PARTIAL_NORMALIZATION"}
    if (
        pred_formula_type in unparseable_types
        or answer_formula_type in unparseable_types
    ):
        logger.warning(
            f"Unparseable formula detected pred: '{pred}' ({pred_formula_type}, notes: {pred_notes}), "
            f"answer: '{answer}' ({answer_formula_type}, notes: {answer_notes})"
        )
        return False

    pred_comp = Composition(pred_formula)
    answer_comp = Composition(answer_formula)
    return pred_comp.almost_equals(answer_comp)


def scorer_si(
    pred_num: float,
    pred_unit: str | None,
    answer_num: float,
    answer_unit: str | None,
    rel_tol: float = 0.001,
    conversion_df: pd.DataFrame | None = None,
) -> bool:
    """Check if pred is within rel_tol of answer, converting units via conversion_df when needed."""
    logger.debug(
        f"Scoring SI: pred={pred_num} {pred_unit}, answer={answer_num} {answer_unit}, rel_tol={rel_tol}"
    )
    pred_unit_norm = pred_unit.strip() if pred_unit else None
    answer_unit_norm = answer_unit.strip() if answer_unit else None

    if pred_unit_norm == answer_unit_norm:
        if answer_num == 0:
            return pred_num == 0
        return abs(pred_num - answer_num) / abs(answer_num) <= rel_tol

    if conversion_df is not None and pred_unit_norm and answer_unit_norm:
        if "property_unit" in conversion_df.columns:
            conversion_lookup = conversion_df.set_index("property_unit")
        else:
            conversion_lookup = conversion_df

        pred_factor = None
        answer_factor = None
        pred_comment = None
        answer_comment = None

        try:
            if pred_unit_norm in conversion_lookup.index:
                pred_factor = conversion_lookup.loc[pred_unit_norm, "conversion_factor"]
                if "comments" in conversion_lookup.columns:
                    pred_comment = conversion_lookup.loc[pred_unit_norm, "comments"]
        except Exception:
            pass

        try:
            if answer_unit_norm in conversion_lookup.index:
                answer_factor = conversion_lookup.loc[
                    answer_unit_norm, "conversion_factor"
                ]
                if "comments" in conversion_lookup.columns:
                    answer_comment = conversion_lookup.loc[answer_unit_norm, "comments"]
        except Exception:
            pass

        if pd.notna(pred_factor) and pd.notna(answer_factor):
            pred_si = pred_num * float(pred_factor)
            answer_si = answer_num * float(answer_factor)

            if answer_si == 0:
                return pred_si == 0
            return abs(pred_si - answer_si) / abs(answer_si) <= rel_tol

        if pd.isna(pred_factor) and pred_unit_norm in conversion_lookup.index:
            comment = (
                pred_comment
                if pd.notna(pred_comment)
                else "No conversion factor available"
            )
            logger.warning(f"Cannot convert unit '{pred_unit_norm}' to SI: {comment}")

        if pd.isna(answer_factor) and answer_unit_norm in conversion_lookup.index:
            comment = (
                answer_comment
                if pd.notna(answer_comment)
                else "No conversion factor available"
            )
            logger.warning(f"Cannot convert unit '{answer_unit_norm}' to SI: {comment}")

    if answer_num == 0:
        return pred_num == 0
    return abs(pred_num - answer_num) / abs(answer_num) <= rel_tol


def scorer_exact_match(
    pred: str, answer: str, mapping: dict[str, str] | None = None
) -> bool:
    """Exact / substring match after trim (case-insensitive for substring)."""
    assert isinstance(pred, str), "pred must be a string"
    assert isinstance(answer, str), "answer must be a string"

    pred_str = pred.strip()
    answer_str = answer.strip()

    if pred_str == answer_str:
        return True

    p_lower = pred_str.lower()
    a_lower = answer_str.lower()

    if p_lower in a_lower or a_lower in p_lower:
        return True

    return False


def score_value(
    pred_value: str,
    answer_value: str,
    rubric: str,
    mapping: dict[str, str] | None = None,
    conversion_df: pd.DataFrame | None = None,
) -> float:
    """Master scoring function (0.0 to 1.0) routing by rubric."""
    assert isinstance(pred_value, str), "pred_value must be a string"
    assert isinstance(answer_value, str), "answer_value must be a string"
    assert isinstance(rubric, str), "rubric must be a string"

    logger.debug(
        f"Scoring pred_value='{pred_value}' vs answer_value='{answer_value}' using rubric='{rubric}'"
    )
    match rubric:
        case "0.1% SI":
            answer_nums = parse_numeric_candidates(answer_value)
            if not answer_nums:
                return 0.0
            if len(answer_nums) > 1:
                logger.warning(
                    f"Multiple numeric candidates found in answer_value '{answer_value}'. Using the first one: {answer_nums[0][0]}"
                )
            answer_num, answer_unit = answer_nums[0]
            for pred_num, pred_unit in parse_numeric_candidates(pred_value):
                if scorer_si(
                    pred_num,
                    pred_unit,
                    answer_num,
                    answer_unit,
                    conversion_df=conversion_df,
                ):
                    return 1.0
            return 0.0

        case "pymatgen":
            pv = normalize_unicode(pred_value).strip()
            av = normalize_unicode(answer_value).strip()
            return 1.0 if scorer_pymatgen(pv, av) else 0.0

        case _:
            return 1.0 if scorer_exact_match(pred_value, answer_value) else 0.0


def score_evidence(
    evidence_pred: str,
    evidence_gt: str,
    method: Literal["sequence_matcher"] = "sequence_matcher",
) -> float:
    """Score evidence similarity via difflib.SequenceMatcher."""
    if pd.isna(evidence_pred) or pd.isna(evidence_gt):
        return 0.0

    evidence_pred = str(evidence_pred).strip()
    evidence_gt = str(evidence_gt).strip()

    if not evidence_pred or not evidence_gt:
        return 0.0

    return difflib.SequenceMatcher(None, evidence_pred, evidence_gt).ratio()


#
# Property-extraction scoring pipeline (from metrics.py)
#
def get_conditions_for_property(
    property_name: str,
    rubric_df: pd.DataFrame,
    important_only: bool = True,
) -> list[str]:
    """Get the condition column names for a given property from the rubric."""
    property_conditions = rubric_df[
        (rubric_df["property_name"] == property_name)
        & (rubric_df["condition_name"].notna())
        & (rubric_df["condition_name"] != "")
    ]

    if important_only and "not_important" in rubric_df.columns:
        property_conditions = property_conditions[
            property_conditions["not_important"] != "not_important"
        ]

    return property_conditions["condition_name"].tolist()


def check_conditions_match(
    row: pd.Series,
    condition_columns: list[str],
    gt_suffix: str = "_gt",
    pred_suffix: str = "_pred",
) -> bool:
    """Check if all condition columns match between ground truth and prediction."""
    for cond in condition_columns:
        gt_col = f"{cond}{gt_suffix}"
        pred_col = f"{cond}{pred_suffix}"

        if gt_col not in row.index or pred_col not in row.index:
            continue

        gt_val = row.get(gt_col)
        pred_val = row.get(pred_col)

        if pd.isna(gt_val) and pd.isna(pred_val):
            continue

        if pd.isna(gt_val) or pd.isna(pred_val):
            return False

        if str(gt_val).strip().lower() != str(pred_val).strip().lower():
            return False

    return True


def construct_context(row: pd.Series) -> str:
    """Construct the context from a ground-truth property row (value_unit only)."""
    return json.dumps({k: v for k, v in row.items() if k == "value_unit"})


def compute_recall_per_material_property(
    df: pd.DataFrame,
    conversion_df: pd.DataFrame | None = None,
    matching_mode: Literal["material", "conditions"] = "material",
    material_column: str = "material_or_system",
    rubric_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Score recall for a dataframe of predicted-ground truth pairs."""
    if matching_mode == "conditions" and rubric_df is None:
        raise ValueError("rubric_df is required when matching_mode='conditions'")

    grouped = df.groupby(["refno", "agent", "model", "id_gt"], dropna=False)
    logger.info(f"Processing {len(grouped)} unique (material, property) pairs...")
    results = []

    gt_material_col = f"{material_column}_gt"
    pred_material_col = f"{material_column}_pred"

    for (refno, agent, model, id_gt), group in grouped:
        matching_rows = []
        property_name = group["property_name_gt"].iloc[0]

        for idx, row in group.iterrows():
            if not row["is_match"]:
                continue

            if matching_mode == "material":
                if pd.notna(row.get(gt_material_col)) and pd.notna(
                    row.get(pred_material_col)
                ):
                    if scorer_pymatgen(
                        str(row[gt_material_col]),
                        str(row[pred_material_col]),
                    ):
                        matching_rows.append(row)
            else:
                condition_columns = get_conditions_for_property(
                    property_name, rubric_df, important_only=True
                )
                if check_conditions_match(row, condition_columns, "_gt", "_pred"):
                    matching_rows.append(row)

        num_matches = len(matching_rows)

        if num_matches == 0:
            recall_score = 0.0
            evidence_score_val = 0.0
        else:
            scores = []
            evidence_scores = []

            for row in matching_rows:
                ev_score = score_evidence(
                    evidence_pred=row.get("location.evidence_pred", ""),
                    evidence_gt=row.get("location.evidence_gt", ""),
                )
                evidence_scores.append(ev_score)

                if (
                    pd.isna(row["value_string_pred"])
                    or pd.isna(row["value_string_gt"])
                    or pd.isna(row["rubric"])
                ):
                    continue

                score = score_value(
                    pred_value=row["value_string_pred"],
                    answer_value=row["value_string_gt"],
                    rubric=row["rubric"],
                    conversion_df=conversion_df,
                )
                scores.append(score)

            recall_score = max(scores) if scores else 0.0
            evidence_score_val = max(evidence_scores) if evidence_scores else 0.0

        result = {
            "refno": refno,
            "agent": agent,
            "model": model,
            "id_gt": id_gt,
            "property_name_gt": property_name,
            "value_string_gt": ", ".join(
                list(set([str(row["value_string_gt"]) for _, row in group.iterrows()]))
            ),
            "num_property_matches": len(group),
            "num_property_material_matches": num_matches,
            "has_property_material_match": int(num_matches > 0),
            "recall_score": recall_score,
            "evidence_score": evidence_score_val,
            "matches": ", ".join(
                [
                    f"{row['property_name_pred']}: {row['value_string_pred']}"
                    for row in matching_rows
                ]
            ),
            "answers": ", ".join(
                [
                    f"{row['value_string_pred']}"
                    for _, row in group.iterrows()
                    if row["is_match"]
                ]
            ),
        }

        if matching_mode == "material" and gt_material_col in group.columns:
            result["material_or_system_gt"] = group[gt_material_col].iloc[0]
            result["material_or_system_pred"] = ", ".join(
                list(
                    set(
                        [
                            str(row[pred_material_col])
                            for _, row in group.iterrows()
                            if pred_material_col in row.index
                        ]
                    )
                )
            )

        results.append(result)

    return pd.DataFrame(results)


def compute_precision_per_material_property(
    df: pd.DataFrame,
    conversion_df: pd.DataFrame | None = None,
    matching_mode: Literal["material", "conditions"] = "material",
    material_column: str = "material_or_system",
    rubric_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Score precision for a dataframe of predicted-ground truth pairs."""
    if matching_mode == "conditions" and rubric_df is None:
        raise ValueError("rubric_df is required when matching_mode='conditions'")

    grouped = df.groupby(["refno", "agent", "model", "id_pred"], dropna=False)
    logger.info(f"Processing {len(grouped)} unique predicted materials...")
    results = []

    gt_material_col = f"{material_column}_gt"
    pred_material_col = f"{material_column}_pred"

    for (refno, agent, model, id_pred), group in grouped:
        matching_rows = []
        property_name = group["property_name_pred"].iloc[0]

        for idx, row in group.iterrows():
            if not row["is_match"]:
                continue

            if matching_mode == "material":
                if pd.notna(row.get(pred_material_col)) and pd.notna(
                    row.get(gt_material_col)
                ):
                    if scorer_pymatgen(
                        str(row[pred_material_col]),
                        str(row[gt_material_col]),
                    ):
                        matching_rows.append(row)
            else:
                condition_columns = get_conditions_for_property(
                    property_name, rubric_df, important_only=True
                )
                if check_conditions_match(row, condition_columns, "_gt", "_pred"):
                    matching_rows.append(row)

        num_matches = len(matching_rows)

        if num_matches == 0:
            precision_score = 0.0
            evidence_score_val = 0.0
        else:
            scores = []
            evidence_scores = []

            for row in matching_rows:
                ev_score = score_evidence(
                    evidence_pred=row.get("location.evidence_pred", ""),
                    evidence_gt=row.get("location.evidence_gt", ""),
                )
                evidence_scores.append(ev_score)

                if (
                    pd.isna(row["value_string_pred"])
                    or pd.isna(row["value_string_gt"])
                    or pd.isna(row["rubric"])
                ):
                    continue

                score = score_value(
                    pred_value=row["value_string_pred"],
                    answer_value=row["value_string_gt"],
                    rubric=row["rubric"],
                    conversion_df=conversion_df,
                )
                scores.append(score)

            precision_score = max(scores) if scores else 0.0
            evidence_score_val = max(evidence_scores) if evidence_scores else 0.0

        result = {
            "refno": refno,
            "agent": agent,
            "model": model,
            "id_pred": id_pred,
            "property_name_pred": property_name,
            "value_string_pred": ", ".join(
                list(
                    set([str(row["value_string_pred"]) for _, row in group.iterrows()])
                )
            ),
            "num_property_matches": len(group),
            "num_property_material_matches": num_matches,
            "has_property_material_match": int(num_matches > 0),
            "precision_score": precision_score,
            "evidence_score": evidence_score_val,
            "matches": ", ".join(
                [
                    f"{row['property_name_gt']}: {row['value_string_gt']}"
                    for row in matching_rows
                ]
            ),
            "answers": ", ".join(
                [
                    f"{row['value_string_gt']}"
                    for _, row in group.iterrows()
                    if row["is_match"]
                ]
            ),
        }

        if matching_mode == "material" and pred_material_col in group.columns:
            result["material_or_system_pred"] = group[pred_material_col].iloc[0]
            result["material_or_system_gt"] = ", ".join(
                list(
                    set(
                        [
                            str(row[gt_material_col])
                            for _, row in group.iterrows()
                            if gt_material_col in row.index
                        ]
                    )
                )
            )

        results.append(result)

    return pd.DataFrame(results)


def add_property_name_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """Add Gemini embeddings for unique property names (requires GOOGLE_API_KEY)."""
    unique_property_names = list(set(df["property_name"].tolist()))
    embeddings = generate_embeddings(unique_property_names)
    df_embeddings = pd.DataFrame(
        {
            "property_name": unique_property_names,
            "embedding": embeddings,
        }
    )
    return df.merge(df_embeddings, on="property_name", how="left")


async def compute_mean_recall_precision(
    df_pred: pd.DataFrame,
    df_gt: pd.DataFrame,
    property_matching_prompt_template: str,
    conversion_df: pd.DataFrame | None,
) -> tuple[float, float]:
    """Calculate mean recall and precision for a single task (single refno)."""
    assert len(df_pred["refno"].unique()) == 1, "Expected only one refno per file"
    assert len(df_gt["refno"].unique()) == 1, "Expected only one refno per file"
    assert "rubric" in df_gt.columns, (
        "Ground truth dataframe must contain a 'rubric' column"
    )

    llm = get_llm(SERVER, MODEL_NAME)

    inf_gen_config = InferenceGenerationConfig(
        max_output_tokens=MAX_OUTPUT_TOKENS,
        output_format="json",
    )

    df_pred = add_property_name_embeddings(df_pred)
    df_gt = add_property_name_embeddings(df_gt)
    df_pred["context"] = df_pred["location.evidence"]
    df_gt["context"] = df_gt.apply(construct_context, axis=1)

    df_pred_matches, _ = await generate_property_name_matches(
        df_pred,
        df_gt,
        llm,
        inf_gen_config,
        property_matching_prompt_template,
        left_on=["property_name", "context"],
        right_on=["property_name", "context"],
        left_suffix="_pred",
        right_suffix="_gt",
    )
    df_recall = compute_recall_per_material_property(df_pred_matches, conversion_df)

    df_gt_matches, _ = await generate_property_name_matches(
        df_gt,
        df_pred,
        llm,
        inf_gen_config,
        property_matching_prompt_template,
        left_on=["property_name", "context"],
        right_on=["property_name", "context"],
        left_suffix="_gt",
        right_suffix="_pred",
    )
    df_precision = compute_precision_per_material_property(df_gt_matches, conversion_df)

    mean_recall = df_recall["recall_score"].mean()
    mean_precision = df_precision["precision_score"].mean()

    return mean_recall, mean_precision


#
# Harbor verifier I/O helpers
#
@dataclass(frozen=True)
class Prediction:
    """Normalized view of one predicted property record."""

    material: str
    property_name: str
    value_string: str
    value_unit: str
    location_evidence: str
    raw: dict[str, Any]


def _load_json(path: Path) -> Any:
    """Load JSON from disk."""
    with path.open() as f:
        return json.load(f)


def _extract_first_json_array(text: str) -> list[dict[str, Any]] | None:
    """Extract the first JSON array from mixed text (handles fenced blocks)."""

    def looks_like_predictions_array(obj: Any) -> bool:
        if not isinstance(obj, list) or not obj:
            return False
        first = obj[0]
        if not isinstance(first, dict):
            return False
        return "property_name" in first and (
            "material" in first or "material_or_system" in first
        )

    fence_match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text, re.IGNORECASE)
    if fence_match:
        try:
            obj = json.loads(fence_match.group(1))
            if looks_like_predictions_array(obj):
                return obj
        except Exception:
            pass

    decoder = JSONDecoder()
    for match in re.finditer(r"\[", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start() :])
        except Exception:
            continue
        if looks_like_predictions_array(obj):
            return obj
    return None


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from mixed text (handles fenced blocks)."""

    def looks_like_properties_object(obj: Any) -> bool:
        if not isinstance(obj, dict):
            return False
        props = obj.get("properties")
        if not isinstance(props, list) or not props:
            return False
        first = props[0]
        if not isinstance(first, dict):
            return False
        return "property_name" in first and (
            "material_or_system" in first or "material" in first
        )

    fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    if fence_match:
        try:
            obj = json.loads(fence_match.group(1))
            if looks_like_properties_object(obj):
                return obj
        except Exception:
            pass

    decoder = JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start() :])
        except Exception:
            continue
        if looks_like_properties_object(obj):
            return obj
    return None


def _extract_text_from_jsonlines_log(text: str) -> str | None:
    """Decode assistant output embedded in JSONL agent logs (e.g., Claude Code)."""
    decoded_parts: list[str] = []
    any_json = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not line.startswith("{"):
            return None
        try:
            obj = json.loads(line)
        except Exception:
            return None
        any_json = True

        if isinstance(obj, dict):
            message = obj.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and isinstance(
                            block.get("text"), str
                        ):
                            decoded_parts.append(block["text"])

            result_text = obj.get("result")
            if isinstance(result_text, str):
                decoded_parts.append(result_text)

    if not any_json:
        return None

    combined = "\n\n".join(
        part.strip() for part in decoded_parts if part and part.strip()
    )
    return combined or None


def _coerce_predictions_payload(payload: Any) -> list[dict[str, Any]]:
    """Coerce multiple supported prediction JSON shapes into a list[dict]."""
    if isinstance(payload, dict) and isinstance(payload.get("properties"), list):
        return [p for p in payload["properties"] if isinstance(p, dict)]
    if isinstance(payload, list):
        return [p for p in payload if isinstance(p, dict)]
    if isinstance(payload, dict):
        values = list(payload.values())
        if values and all(isinstance(v, dict) for v in values):
            return [v for v in values if isinstance(v, dict)]
    raise ValueError("Unrecognized predictions JSON format")


def _as_prediction(raw: dict[str, Any]) -> Prediction:
    """Map a raw property dict to a normalized Prediction."""
    material = raw.get("material")
    if material is None:
        material = (
            raw.get("material_or_system")
            or raw.get("material_system")
            or raw.get("system")
        )
    property_name = raw.get("property_name") or raw.get("name")

    value_string = (
        raw.get("value_string")
        or raw.get("value")
        or raw.get("pred_value")
        or raw.get("property_value")
        or ""
    )
    value_unit = (
        raw.get("value_unit")
        or raw.get("pred_unit")
        or raw.get("unit")
        or raw.get("property_unit")
        or ""
    )
    location_evidence = ""
    location = raw.get("location")
    if isinstance(location, dict):
        location_evidence = str(location.get("evidence") or "")
    elif "location.evidence" in raw:
        location_evidence = str(raw.get("location.evidence") or "")

    return Prediction(
        material=str(material or ""),
        property_name=str(property_name or ""),
        value_string=str(value_string or ""),
        value_unit=str(value_unit or ""),
        location_evidence=location_evidence,
        raw=raw,
    )


def load_predictions(predictions_path: Path) -> list[Prediction]:
    """Load predictions from disk or fall back to parsing agent logs."""
    if predictions_path.exists():
        payload = _load_json(predictions_path)
        return [_as_prediction(p) for p in _coerce_predictions_payload(payload)]

    agent_logs_dir = Path("/logs/agent")
    if agent_logs_dir.exists():
        for log_path in sorted(agent_logs_dir.glob("*.txt")):
            try:
                content = log_path.read_text()
            except Exception:
                continue
            decoded = _extract_text_from_jsonlines_log(content)
            text = decoded or content

            extracted_obj = _extract_first_json_object(text)
            if extracted_obj is not None:
                return [
                    _as_prediction(p)
                    for p in _coerce_predictions_payload(extracted_obj)
                ]

            extracted_arr = _extract_first_json_array(text)
            if extracted_arr is not None:
                return [
                    _as_prediction(p)
                    for p in _coerce_predictions_payload(extracted_arr)
                ]

    raise FileNotFoundError(
        f"Missing predictions file at {predictions_path} and could not parse JSON from /logs/agent/*.txt"
    )


#
# Harbor verifier main
#
def _ground_truth_to_df(expected: dict[str, Any]) -> pd.DataFrame:
    """Build df_gt from the expected.json ground_truth list."""
    refno = str(expected.get("refno") or "")
    rows = []
    for truth in expected.get("ground_truth") or []:
        if not isinstance(truth, dict):
            continue
        rows.append(
            {
                "refno": refno,
                "agent": "verifier",
                "model": "verifier",
                "id_gt": truth.get("id") or "",
                "property_name": truth.get("property_name") or "",
                "value_string": truth.get("value_string") or "",
                "value_unit": truth.get("value_unit") or "",
                "material_or_system": truth.get("material_or_system")
                or truth.get("material")
                or "",
                "rubric": truth.get("rubric") or "",
                "location.evidence": (
                    (truth.get("location") or {}).get("evidence", "")
                    if isinstance(truth.get("location"), dict)
                    else ""
                ),
            }
        )
    return pd.DataFrame(rows)


def _predictions_to_df(predictions: list[Prediction], refno: str) -> pd.DataFrame:
    """Build df_pred from a list of Prediction records."""
    rows = []
    for i, pred in enumerate(predictions):
        rows.append(
            {
                "refno": refno,
                "agent": "verifier",
                "model": "verifier",
                "id_pred": i,
                "property_name": pred.property_name,
                "value_string": pred.value_string,
                "value_unit": pred.value_unit,
                "material_or_system": pred.material,
                "location.evidence": pred.location_evidence,
            }
        )
    return pd.DataFrame(rows)


async def _async_main(args: argparse.Namespace) -> None:
    """Async body of main — loads inputs, runs the pipeline, writes outputs."""
    expected_path = Path(args.expected)
    predictions_path = Path(args.predictions)
    reward_path = Path(args.reward)
    details_path = Path(args.details)

    expected = _load_json(expected_path)
    refno = str(expected.get("refno") or "")

    df_gt = _ground_truth_to_df(expected)
    if df_gt.empty:
        raise ValueError("expected.json contains no ground_truth entries")

    predictions = load_predictions(predictions_path)
    df_pred = _predictions_to_df(predictions, refno)
    if df_pred.empty:
        raise ValueError("predictions.json contains no usable prediction entries")

    mean_recall, mean_precision = await compute_mean_recall_precision(
        df_pred,
        df_gt,
        PROPERTY_MATCHING_PROMPT,
        conversion_df=None,
    )

    reward = float(
        2.0 * mean_recall * mean_precision / (mean_recall + mean_precision + 1e-8)
    )

    reward_path.parent.mkdir(parents=True, exist_ok=True)
    reward_path.write_text(str(reward))

    details_path.write_text(
        json.dumps(
            {
                "reward": reward,
                "mean_recall": float(mean_recall),
                "mean_precision": float(mean_precision),
                "task": expected.get("task"),
                "refno": expected.get("refno"),
                "n_predictions": len(predictions),
                "n_ground_truth": len(df_gt),
            },
            indent=2,
        )
    )

    if reward < 1.0:
        print(
            f"Prediction check completed. reward={reward:.4f} "
            f"(recall={mean_recall:.4f}, precision={mean_precision:.4f})"
        )
        sys.exit(1)

    print("All predictions correct.")


def main() -> None:
    """Entry point: load expected + predictions, score, write reward/details."""
    parser = argparse.ArgumentParser(
        description="Harbor verifier for property extraction."
    )
    parser.add_argument("--expected", type=str, default="/tests/expected.json")
    parser.add_argument("--predictions", type=str, default="/app/predictions.json")
    parser.add_argument("--reward", type=str, default="/logs/verifier/reward.txt")
    parser.add_argument("--details", type=str, default="/logs/verifier/details.json")
    args = parser.parse_args()

    reward_path = Path(args.reward)
    details_path = Path(args.details)

    try:
        asyncio.run(_async_main(args))
    except Exception as exc:
        reward_path.parent.mkdir(parents=True, exist_ok=True)
        reward_path.write_text("0.0")
        details_path.write_text(
            json.dumps(
                {
                    "reward": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
                indent=2,
            )
        )
        print(f"Verifier error: {type(exc).__name__}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
