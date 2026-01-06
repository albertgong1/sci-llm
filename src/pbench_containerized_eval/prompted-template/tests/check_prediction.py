from __future__ import annotations

import json
import re
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from json import JSONDecoder
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RowKey:
    material: str
    property_name: str

    @staticmethod
    def from_strings(material: str, property_name: str) -> "RowKey":
        """Docstring for from_strings

        :param material: Description
        :type material: str
        :param property_name: Description
        :type property_name: str
        :return: Description
        :rtype: RowKey
        """
        return RowKey(material.strip().lower(), property_name.strip().lower())


def parse_numeric_value(value: str) -> float | None:
    """Parse a numeric value from a string, handling scientific notation and uncertainties."""
    if value is None:
        return None

    value_str = str(value).strip()
    if value_str.upper() == "NOT_FOUND" or value_str == "":
        return None

    value_str = re.sub(r"\(\d+\)", "", value_str)
    try:
        return float(value_str)
    except ValueError:
        return None


def score_value(pred_value: str, answer_value: str, rubric: str | None) -> float:
    """Return 1.0 for correct, 0.0 for incorrect (mirrors accuracy-style scoring)."""
    if rubric == "0.1% SI":
        pred_num = parse_numeric_value(pred_value)
        answer_num = parse_numeric_value(answer_value)
        if pred_num is None or answer_num is None:
            return 0.0
        if answer_num == 0:
            return 1.0 if pred_num == 0 else 0.0
        return 1.0 if abs(pred_num - answer_num) / abs(answer_num) <= 0.001 else 0.0

    if rubric == "pymatgen":
        try:
            from pymatgen.core import Composition  # type: ignore
        except Exception:
            return 0.0
        try:
            return (
                1.0
                if Composition(str(pred_value)).almost_equals(
                    Composition(str(answer_value))
                )
                else 0.0
            )
        except Exception:
            return 0.0

    # Default categorical: case-insensitive equality.
    return (
        1.0
        if str(pred_value).strip().lower() == str(answer_value).strip().lower()
        else 0.0
    )


def _load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def _extract_first_json_array(text: str) -> list[dict[str, Any]] | None:
    """Extract the first JSON array from text, allowing surrounding prose/code fences."""

    def looks_like_predictions_array(obj: Any) -> bool:
        if not isinstance(obj, list) or not obj:
            return False
        first = obj[0]
        if not isinstance(first, dict):
            return False
        return "material" in first and "property_name" in first

    # Prefer fenced blocks if present.
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


def _extract_text_from_jsonlines_log(text: str) -> str | None:
    """Extract decoded text content from a JSONL/stream-json agent log.

    Claude Code writes stream-json logs where the assistant's response is nested in a JSON
    object and the actual text is *escaped* as a JSON string. This helper parses each
    JSON line and concatenates any discovered text blocks so downstream JSON parsing can
    run on real, decoded text.
    """
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

            # Some tools may surface the final answer as `result`.
            result_text = obj.get("result")
            if isinstance(result_text, str):
                decoded_parts.append(result_text)

    if not any_json:
        return None

    combined = "\n\n".join(
        part.strip() for part in decoded_parts if part and part.strip()
    )
    return combined or None


def load_predictions(predictions_path: Path) -> list[dict[str, Any]]:
    """Load predictions from /app/output or fall back to parsing /logs/agent/*.txt."""
    if predictions_path.exists():
        data = _load_json(predictions_path)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            records: list[dict[str, Any]] = []
            for key, value in data.items():
                if isinstance(value, dict):
                    record = dict(value)
                    if "qid" not in record:
                        try:
                            record["qid"] = int(str(key))
                        except Exception:
                            pass
                    records.append(record)
            if records:
                return records
            return list(data.values())
        raise ValueError(
            "predictions.json must be a list or dict of prediction entries"
        )

    agent_logs_dir = Path("/logs/agent")
    if agent_logs_dir.exists():
        for log_path in sorted(agent_logs_dir.glob("*.txt")):
            try:
                content = log_path.read_text()
            except Exception:
                continue
            decoded = _extract_text_from_jsonlines_log(content)
            extracted = _extract_first_json_array(decoded or content)
            if extracted is not None:
                return extracted

    raise FileNotFoundError(
        f"Missing predictions file at {predictions_path} and could not parse JSON from /logs/agent/*.txt"
    )


def main() -> None:
    """Docstring for main"""
    expected_path = Path("/tests/expected.json")
    predictions_path = Path("/app/output/predictions.json")
    reward_path = Path("/logs/verifier/reward.txt")
    details_path = Path("/logs/verifier/details.json")

    try:
        expected = _load_json(expected_path)
        ground_truth = expected["ground_truth"]

        predictions = load_predictions(predictions_path)
        pred_map: dict[RowKey, list[dict[str, Any]]] = defaultdict(list)
        for row in predictions:
            if not isinstance(row, dict):
                raise TypeError("Predictions JSON array must contain objects (dicts).")
            key = RowKey.from_strings(
                str(row.get("material", "")), str(row.get("property_name", ""))
            )
            pred_map[key].append(row)

        results: list[dict[str, Any]] = []
        total = 0
        correct = 0

        for truth in ground_truth:
            key = RowKey.from_strings(truth["material"], truth["property_name"])
            answer_value = str(truth.get("property_value") or "")
            rubric = truth.get("rubric")

            candidates = pred_map.get(key, [])
            chosen: dict[str, Any] | None = None
            chosen_score = 0.0
            if candidates:
                best_idx: int | None = None
                best_score = -1.0
                for idx, cand in enumerate(candidates):
                    cand_value = str(cand.get("pred_value") or cand.get("value") or "")
                    cand_score = score_value(cand_value, answer_value, rubric)
                    if cand_score > best_score:
                        best_score = cand_score
                        best_idx = idx
                        if best_score == 1.0:
                            break
                if best_idx is not None:
                    chosen = candidates.pop(best_idx)
                    chosen_score = float(best_score)

            pred_value = str(
                (chosen or {}).get("pred_value") or (chosen or {}).get("value") or ""
            )

            score = chosen_score
            results.append(
                {
                    "material": truth["material"],
                    "property_name": truth["property_name"],
                    "rubric": rubric,
                    "answer_value": answer_value,
                    "pred_value": pred_value,
                    "score": score,
                }
            )

            total += 1
            correct += int(score == 1.0)

        reward = (correct / total) if total else 0.0
        reward_path.parent.mkdir(parents=True, exist_ok=True)
        reward_path.write_text(str(reward))
        details_path.write_text(
            json.dumps(
                {"reward": reward, "correct": correct, "total": total, "rows": results},
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
