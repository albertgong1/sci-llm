from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import traceback
from dataclasses import dataclass
from json import JSONDecoder
from pathlib import Path
from typing import Any

import pandas as pd
from pymatgen.core import Composition


# --- Inlined from pbench_eval.utils ---

logger = logging.getLogger(__name__)


# String normalization helpers

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


def normalize_ws(text: str) -> str:
    """Collapse whitespace for more robust string matching."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_unicode(text: str) -> str:
    """Normalize common unicode variants (dashes, superscripts, delta)."""
    s = str(text or "")
    s = (
        s.replace("‐", "-")
        .replace("‑", "-")
        .replace("‒", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
    )
    s = s.replace("δ", "d").replace("Δ", "d")
    s = s.translate(_SUPERSCRIPT_MAP)
    return s


# Stubs for missing module deps (container has no .normalize_material /
# .space_groups_normalized). These mirror the patches the previous vendoring
# step applied; pymatgen-rubric and space-group-rubric scoring degrade to no-op.

SPACE_GROUPS: dict[str, int] = {}


def classify_and_normalize(*args, **kwargs):
    return None, None, None


def strip_formula(s):
    return s, {}


# Numeric parsing


def parse_numeric_candidates(value: str) -> list[tuple[float, str | None]]:
    """Extract numeric candidates with units from a free-form value string."""
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


# Scorers


def scorer_pymatgen(pred: str, answer: str) -> bool:
    """Check if pred is a valid pymatgen composition and is close to answer."""
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
    """Check if pred is within 0.1% of answer."""
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


def scorer_space_group(pred: str, answer: str) -> bool:
    """Score space group predictions."""

    def get_norm_and_id(val: str) -> tuple[str, str | None]:
        if not isinstance(val, str):
            val = str(val)
        cleaned = re.sub(r"[^a-zA-Z0-9/\-]", "", val)
        norm = cleaned.lower()
        return norm, SPACE_GROUPS.get(norm)

    pred_norm, pred_id = get_norm_and_id(pred)
    answer_norm, answer_id = get_norm_and_id(answer)

    if pred_id is None:
        logger.warning(
            f"Predicted space group '{pred}' (clean: '{pred_norm}') not found in allowed keys."
        )
        return False

    if answer_id is None:
        logger.warning(
            f"Answer space group '{answer}' (clean: '{answer_norm}') not found in allowed keys."
        )
        return False

    return pred_id == answer_id


def scorer_exact_match(
    pred: str, answer: str, mapping: dict[str, str] | None = None
) -> bool:
    """Scores categorical properties (exact match or case-insensitive substring)."""
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


# Master dispatcher


def score_value(
    pred_value: str,
    answer_value: str,
    rubric: str,
    mapping: dict[str, str] | None = None,
    conversion_df: pd.DataFrame | None = None,
) -> float:
    """Master scoring function (0.0 to 1.0).

    rubric: "0.1% SI", "pymatgen", "space_group", or anything else (treated as categorical).
    """
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

        case "space_group":
            return 1.0 if scorer_space_group(pred_value, answer_value) else 0.0

        case _:
            return 1.0 if scorer_exact_match(pred_value, answer_value) else 0.0


# --- End inlined from pbench_eval.utils ---


@dataclass(frozen=True)
class RowKey:
    """Key used to group ground-truth rows by (material, property_name)."""

    material: str
    property_name: str

    @staticmethod
    def from_strings(material: str, property_name: str) -> "RowKey":
        """Build a normalized key from raw strings."""
        return RowKey(
            str(material or "").strip().lower(),
            str(property_name or "").strip().lower(),
        )


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


@dataclass(frozen=True)
class Prediction:
    """Normalized view of one predicted property record."""

    material: str
    property_name: str
    pred_value: str
    pred_unit: str
    raw: dict[str, Any]


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

    pred_value = (
        raw.get("pred_value")
        or raw.get("value")
        or raw.get("value_string")
        or raw.get("property_value")
        or ""
    )
    pred_unit = (
        raw.get("pred_unit") or raw.get("unit") or raw.get("property_unit") or ""
    )

    return Prediction(
        material=str(material or ""),
        property_name=str(property_name or ""),
        pred_value=str(pred_value or ""),
        pred_unit=str(pred_unit or ""),
        raw=raw,
    )


def load_predictions(predictions_path: Path) -> list[Prediction]:
    """Load predictions from disk or fall back to parsing agent logs."""
    payload = None
    if predictions_path.exists():
        payload = _load_json(predictions_path)
    else:
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
                    payload = extracted_obj
                    break

                extracted_arr = _extract_first_json_array(text)
                if extracted_arr is not None:
                    payload = extracted_arr
                    break

    if payload is None:
        raise FileNotFoundError(
            f"Missing predictions file at {predictions_path} and could not parse JSON from /logs/agent/*.txt"
        )

    # Convert to standard list of Predictions (new schema has no 'values' list)
    raw_list = _coerce_predictions_payload(payload)
    return [_as_prediction(r) for r in raw_list]


_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "sample",
    "the",
    "this",
    "to",
}


def _tokens(text: str) -> set[str]:
    """Tokenize text for fuzzy property-name matching."""
    # Use normalize_categorical from utils/local for consistency if needed,
    # but since this is just internal fuzzy matching, lower() is fine or we use normalize_unicode
    # Let's rely on standard logic
    return {
        t
        for t in re.findall(r"[a-z0-9]+", normalize_unicode(normalize_ws(text)).lower())
        if t not in _STOPWORDS
    }


def _is_tc_like_truth(truth_property_name: str, task_name: str | None) -> bool:
    """Detect whether the ground-truth property is Tc-like (critical temperature)."""
    if task_name and task_name.strip().lower() == "tc":
        return True
    toks = _tokens(truth_property_name)
    return "tc" in toks or "t_c" in toks


def _property_name_match(
    *, truth_property_name: str, pred_property_name: str, task_name: str | None
) -> bool:
    """Return True if a prediction's property_name matches the ground-truth name."""
    truth_norm = normalize_unicode(normalize_ws(truth_property_name)).lower()
    pred_norm = normalize_unicode(normalize_ws(pred_property_name)).lower()
    if not pred_norm:
        return False

    if truth_norm == pred_norm:
        return True

    # Removed loose substring match (truth in pred or pred in truth)
    # because short keys like "tc" match "tcn" and break scoring.

    if _is_tc_like_truth(truth_property_name, task_name):
        if re.search(r"\btc\b", pred_norm) or re.search(
            r"\bt\s*[_-]?\s*c\b", pred_norm
        ):
            return True
        if "critical temperature" in pred_norm:
            return True
        if "transition temperature" in pred_norm:
            return True
        if "superconduct" in pred_norm and "temperature" in pred_norm:
            return True
        return False

    if truth_norm == pred_norm:
        return True

    # Strict token overlap for other properties
    truth_toks = _tokens(truth_property_name)
    pred_toks = _tokens(pred_property_name)
    if not truth_toks or not pred_toks:
        return False
    overlap = len(truth_toks & pred_toks) / max(1, len(truth_toks))
    return overlap >= 0.6


def _normalize_material(material: str) -> str:
    """Normalize a material/system string for loose matching."""
    s = normalize_unicode(normalize_ws(material)).lower()
    s = s.replace(" ", "")
    s = re.sub(r"([a-z\)])1(?=([a-z\(\)\-]|$))", r"\1", s)
    return s


def _is_generic_material(material: str) -> bool:
    """Return True for generic/non-identifying material strings."""
    norm = _normalize_material(material)
    return norm in {
        "",
        "material",
        "sample",
        "specimen",
        "film",
        "thinfilm",
        "thinfilms",
        "crystal",
        "singlecrystal",
        "polycrystal",
        "superconductor",
    }


def _material_match(*, truth_material: str, pred_material: str) -> bool:
    """Loose material match (substring match after normalization)."""
    if _is_generic_material(pred_material):
        return False
    truth_norm = _normalize_material(truth_material)
    pred_norm = _normalize_material(pred_material)
    if not truth_norm or not pred_norm:
        return False
    return truth_norm in pred_norm or pred_norm in truth_norm


def main() -> None:
    """Entry point: load expected + predictions, score, write reward/details."""
    parser = argparse.ArgumentParser(
        description="Harbor verifier for property extraction."
    )
    parser.add_argument("--expected", type=str, default="/tests/expected.json")
    parser.add_argument(
        "--predictions", type=str, default="/app/output/predictions.json"
    )
    parser.add_argument("--reward", type=str, default="/logs/verifier/reward.txt")
    parser.add_argument("--details", type=str, default="/logs/verifier/details.json")
    args = parser.parse_args()

    expected_path = Path(args.expected)
    predictions_path = Path(args.predictions)
    reward_path = Path(args.reward)
    details_path = Path(args.details)

    try:
        expected = _load_json(expected_path)
        ground_truth = list(expected.get("ground_truth") or [])
        if not isinstance(ground_truth, list):
            raise TypeError("expected.json ground_truth must be a list")

        task_name = str(expected.get("task") or "").strip().lower() or None
        unique_truth_materials = sorted(
            {str(t.get("material") or "") for t in ground_truth}
        )
        require_material_match = len([m for m in unique_truth_materials if m]) > 1

        predictions = load_predictions(predictions_path)

        grouped_truth: dict[RowKey, list[dict[str, Any]]] = {}
        for truth in ground_truth:
            key = RowKey.from_strings(truth.get("material"), truth.get("property_name"))
            grouped_truth.setdefault(key, []).append(truth)

        results: list[dict[str, Any]] = []
        total = 0
        correct = 0

        for key, truths in grouped_truth.items():
            truth_material = str(truths[0].get("material") or "")
            truth_prop_name = str(truths[0].get("property_name") or "")
            rubric = truths[0].get("rubric")

            candidate_preds = [
                pred
                for pred in predictions
                if _property_name_match(
                    truth_property_name=truth_prop_name,
                    pred_property_name=pred.property_name,
                    task_name=task_name,
                )
                and (
                    not require_material_match
                    or _material_match(
                        truth_material=truth_material, pred_material=pred.material
                    )
                )
            ]

            pool: list[dict[str, Any]] = []
            # We don't need to pool specific parsed values anymore,
            # just the prediction objects, because score_value handles the parsing
            for pred in candidate_preds:
                pool.append({"pred": pred})

            for truth in truths:
                answer_value = str(truth.get("property_value") or "")
                answer_unit = str(truth.get("property_unit") or "")
                rubric_row = truth.get("rubric")

                chosen_pred: Prediction | None = None
                chosen_value: str = ""
                chosen_score = 0.0

                best_idx: int | None = None
                best_score = -1.0

                # Simple greedy match logic reusing the robust score_value
                if not pool:
                    # If no predictions were found for this property, check if the answer is N/A
                    if answer_value in {"N/A", "nan", ""}:
                        chosen_score = 1.0
                        chosen_value = "N/A (Implicit)"
                else:
                    for idx, item in enumerate(pool):
                        pred = item["pred"]
                        score = score_value(pred.pred_value, answer_value, rubric_row)

                        # If perfect score, take it immediately (break early optimization)
                        if score > best_score:
                            best_score = score
                            best_idx = idx
                            chosen_pred = pred
                            chosen_value = pred.pred_value
                            chosen_score = float(score)
                            if best_score == 1.0:
                                break

                if best_idx is not None and chosen_score == 1.0:
                    pool.pop(best_idx)

                results.append(
                    {
                        "material": truth_material,
                        "property_name": truth_prop_name,
                        "rubric": rubric_row,
                        "answer_value": answer_value,
                        "answer_unit": answer_unit,
                        "pred_value": chosen_value,
                        "pred_unit": chosen_pred.pred_unit if chosen_pred else "",
                        "pred_property_name": chosen_pred.property_name
                        if chosen_pred
                        else "",
                        "pred_material": chosen_pred.material if chosen_pred else "",
                        "pred_raw": chosen_pred.raw if chosen_pred else None,
                        "score": chosen_score,
                    }
                )

                total += 1
                correct += int(chosen_score == 1.0)

        reward = (correct / total) if total else 0.0
        reward_path.parent.mkdir(parents=True, exist_ok=True)
        reward_path.write_text(str(reward))

        details_path.write_text(
            json.dumps(
                {
                    "reward": reward,
                    "correct": correct,
                    "total": total,
                    "n_predictions": len(predictions),
                    "task": expected.get("task"),
                    "refno": expected.get("refno"),
                    "require_material_match": require_material_match,
                    "rows": results,
                },
                indent=2,
            )
        )

        if reward < 1.0:
            print("Prediction check completed with mismatches.")
            for row in results:
                if row["score"] != 1.0:
                    print(
                        f"- {row['material']} {row['property_name']}: "
                        f"pred='{row['pred_value']}' answer='{row['answer_value']}' rubric='{row['rubric']}'"
                    )
            sys.exit(1)

        print("All predictions correct.")
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
