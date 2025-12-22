from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from dataclasses import dataclass
from json import JSONDecoder
from pathlib import Path
from typing import Any


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


def _normalize_ws(text: str) -> str:
    """Collapse whitespace for more robust string matching."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _normalize_unicode(text: str) -> str:
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


def _normalize_categorical(value: str) -> str:
    """Normalize a categorical string for comparison."""
    return _normalize_unicode(_normalize_ws(value)).lower()


def _load_json(path: Path) -> Any:
    """Load JSON from disk."""
    with path.open() as f:
        return json.load(f)


def parse_numeric_candidates(value: str) -> list[float]:
    """Extract numeric candidates from a free-form value string (units allowed)."""
    if value is None:
        return []

    value_str = _normalize_unicode(str(value)).strip()
    if value_str.upper() == "NOT_FOUND" or value_str == "":
        return []

    value_str = re.sub(r"\(\d+\)", "", value_str)

    candidates: list[float] = []

    sci_pattern = re.compile(
        r"(?P<base>[-+]?\d*\.?\d+)\s*(?:x|×)\s*10(?:\s*\^)?\s*(?P<exp>[-+]?\d+)",
        re.IGNORECASE,
    )
    for match in sci_pattern.finditer(value_str):
        try:
            base = float(match.group("base"))
            exp = int(match.group("exp"))
            candidates.append(base * (10**exp))
        except Exception:
            continue

    num_pattern = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
    for match in num_pattern.finditer(value_str):
        try:
            candidates.append(float(match.group(0)))
        except Exception:
            continue

    seen: set[str] = set()
    unique: list[float] = []
    for num in candidates:
        key = f"{num:.12g}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(num)

    return unique


def _si_match(
    pred_num: float, answer_num: float, *, rel_tol: float = 0.001
) -> tuple[float, float]:
    """Return (score, rel_error) for a numeric match under relative tolerance."""
    if answer_num == 0:
        return (1.0, 0.0) if pred_num == 0 else (0.0, float("inf"))
    rel_err = abs(pred_num - answer_num) / abs(answer_num)
    return (1.0, rel_err) if rel_err <= rel_tol else (0.0, rel_err)


def _strip_purity_annotations(text: str) -> str:
    """Strip purity suffixes like '(4N)' from element tokens."""
    return re.sub(r"\(\s*\d+\s*N\s*\)", "", text, flags=re.IGNORECASE).strip()


def _parse_simple_formula(formula: str) -> dict[str, float] | None:
    """Parse a simple chemical formula into element->count (conservative)."""
    cleaned = _normalize_unicode(formula).strip()
    cleaned = _strip_purity_annotations(cleaned)
    cleaned = cleaned.replace(" ", "")
    if not cleaned:
        return None
    if "," in cleaned or "(" in cleaned or ")" in cleaned:
        return None

    pos = 0
    comp: dict[str, float] = {}
    token = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")
    while pos < len(cleaned):
        match = token.match(cleaned, pos)
        if not match:
            return None
        element = match.group(1)
        count_str = match.group(2)
        count = float(count_str) if count_str else 1.0
        comp[element] = comp.get(element, 0.0) + count
        pos = match.end()
    return comp


def _normalize_formula(formula: str) -> str | None:
    """Convert a parsed formula into a canonical alphabetical string."""
    comp = _parse_simple_formula(formula)
    if comp is None:
        return None
    parts: list[str] = []
    for element in sorted(comp.keys()):
        count = comp[element]
        if abs(count - 1.0) < 1e-12:
            parts.append(element)
        elif abs(count - round(count)) < 1e-12:
            parts.append(f"{element}{int(round(count))}")
        else:
            parts.append(f"{element}{count:g}")
    return "".join(parts)


def _normalize_formula_set(value: str) -> set[str] | None:
    """Normalize a single formula or a delimiter-separated list of formulas."""
    raw = _normalize_unicode(_normalize_ws(value))
    raw = _strip_purity_annotations(raw)
    if not raw:
        return None

    parts = [p.strip() for p in re.split(r"[;,]", raw) if p.strip()]
    if not parts:
        return None

    normalized: set[str] = set()
    for part in parts:
        norm = _normalize_formula(part)
        if norm is None:
            return None
        normalized.add(norm)
    return normalized


def score_value(pred_value: str, answer_value: str, rubric: str | None) -> float:
    """Score one predicted value against one ground-truth value using a rubric."""
    if rubric == "0.1% SI":
        answer_nums = parse_numeric_candidates(answer_value)
        if not answer_nums:
            return 0.0
        answer_num = answer_nums[0]
        for num in parse_numeric_candidates(pred_value):
            score, _ = _si_match(num, answer_num)
            if score == 1.0:
                return 1.0
        return 0.0

    if rubric == "pymatgen":
        pred_norm = _normalize_formula_set(pred_value)
        ans_norm = _normalize_formula_set(answer_value)
        if pred_norm is not None and ans_norm is not None:
            return 1.0 if pred_norm == ans_norm else 0.0
        return (
            1.0
            if _normalize_categorical(pred_value)
            == _normalize_categorical(answer_value)
            else 0.0
        )

    return (
        1.0
        if _normalize_categorical(pred_value) == _normalize_categorical(answer_value)
        else 0.0
    )


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
    return {
        t
        for t in re.findall(r"[a-z0-9]+", _normalize_categorical(text))
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
    truth_norm = _normalize_categorical(truth_property_name)
    pred_norm = _normalize_categorical(pred_property_name)
    if not pred_norm:
        return False

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
    if truth_norm and (truth_norm in pred_norm or pred_norm in truth_norm):
        return True

    truth_toks = _tokens(truth_property_name)
    pred_toks = _tokens(pred_property_name)
    if not truth_toks or not pred_toks:
        return False
    overlap = len(truth_toks & pred_toks) / max(1, len(truth_toks))
    return overlap >= 0.6


def _normalize_material(material: str) -> str:
    """Normalize a material/system string for loose matching."""
    s = _normalize_unicode(_normalize_ws(material)).lower()
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
            if rubric == "0.1% SI":
                for pred in candidate_preds:
                    for num in parse_numeric_candidates(pred.pred_value):
                        pool.append({"pred": pred, "num": num})
            elif rubric == "pymatgen":
                for pred in candidate_preds:
                    pool.append(
                        {"pred": pred, "norm": _normalize_formula_set(pred.pred_value)}
                    )
            else:
                for pred in candidate_preds:
                    pool.append(
                        {"pred": pred, "norm": _normalize_categorical(pred.pred_value)}
                    )

            for truth in truths:
                answer_value = str(truth.get("property_value") or "")
                answer_unit = str(truth.get("property_unit") or "")
                rubric_row = truth.get("rubric")

                chosen_pred: Prediction | None = None
                chosen_value: str = ""
                chosen_score = 0.0

                best_idx: int | None = None
                best_score = -1.0
                best_tie: float = float("inf")

                if rubric_row == "0.1% SI":
                    answer_nums = parse_numeric_candidates(answer_value)
                    answer_num = answer_nums[0] if answer_nums else None
                    for idx, item in enumerate(pool):
                        pred = item["pred"]
                        pred_num = item.get("num")
                        if answer_num is None or pred_num is None:
                            continue
                        score, tie = _si_match(float(pred_num), float(answer_num))
                        if score > best_score or (
                            score == best_score and tie < best_tie
                        ):
                            best_score = score
                            best_tie = tie
                            best_idx = idx
                            chosen_pred = pred
                            chosen_value = str(pred_num)
                            chosen_score = float(score)
                            if best_score == 1.0 and best_tie == 0.0:
                                break
                elif rubric_row == "pymatgen":
                    ans_norm = _normalize_formula_set(answer_value)
                    for idx, item in enumerate(pool):
                        pred = item["pred"]
                        pred_norm = item.get("norm")
                        score = 0.0
                        if pred_norm is not None and ans_norm is not None:
                            score = 1.0 if pred_norm == ans_norm else 0.0
                        else:
                            score = (
                                1.0
                                if _normalize_categorical(pred.pred_value)
                                == _normalize_categorical(answer_value)
                                else 0.0
                            )
                        if score > best_score:
                            best_score = score
                            best_idx = idx
                            chosen_pred = pred
                            chosen_value = pred.pred_value
                            chosen_score = float(score)
                            if best_score == 1.0:
                                break
                else:
                    ans_norm = _normalize_categorical(answer_value)
                    for idx, item in enumerate(pool):
                        pred = item["pred"]
                        pred_norm = item.get("norm")
                        score = 1.0 if pred_norm == ans_norm else 0.0
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
