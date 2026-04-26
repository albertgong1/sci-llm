"""Utility functions for loading predictions from Harbor jobs.

Usage:
    from pbench_eval.harbor_utils import get_harbor_data
    df = get_harbor_data(jobs_dir)
"""

import json
import logging
import re
import uuid
from json import JSONDecoder
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)
_VALID_JSON_ESCAPE_STARTS: set[str] = {
    '"',
    "\\",
    "/",
    "b",
    "f",
    "n",
    "r",
    "t",
    "u",
}


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


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from mixed text (handles fenced blocks)."""
    fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    if fence_match:
        try:
            obj = json.loads(fence_match.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    decoder = JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start() :])
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def _extract_first_json_array(text: str) -> list[dict[str, Any]] | None:
    """Extract the first JSON array from mixed text (handles fenced blocks)."""
    fence_match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text, re.IGNORECASE)
    if fence_match:
        try:
            obj = json.loads(fence_match.group(1))
            if isinstance(obj, list):
                return obj
        except Exception:
            pass

    decoder = JSONDecoder()
    for match in re.finditer(r"\[", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start() :])
            if isinstance(obj, list):
                return obj
        except Exception:
            continue
    return None


def _escape_invalid_json_backslashes(text: str) -> str:
    """Escape bare backslashes so slightly malformed model JSON can still parse."""
    escaped_chars: list[str] = []
    idx = 0
    while idx < len(text):
        char = text[idx]
        if char != "\\":
            escaped_chars.append(char)
            idx += 1
            continue

        if idx + 1 >= len(text):
            escaped_chars.append("\\\\")
            idx += 1
            continue

        next_char = text[idx + 1]
        if next_char in _VALID_JSON_ESCAPE_STARTS:
            escaped_chars.append(char)
            escaped_chars.append(next_char)
            idx += 2
            continue

        escaped_chars.append("\\\\")
        idx += 1

    return "".join(escaped_chars)


def _parse_json_payload(text: str) -> dict[str, Any] | list[dict[str, Any]] | None:
    """Parse a direct JSON payload, repairing invalid backslash escapes if needed."""
    for candidate in (text, _escape_invalid_json_backslashes(text)):
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, (dict, list)):
            return payload
    return None


def _load_prediction_candidates(trial_dir: Path) -> list[Path]:
    """Return ordered prediction sources for a Harbor trial."""
    candidates: list[Path] = []

    for predictions_path in (
        trial_dir / "verifier" / "predictions.json",
        trial_dir / "verifier" / "app_output" / "predictions.json",
    ):
        if predictions_path.exists():
            candidates.append(predictions_path)

    agent_dir = trial_dir / "agent"
    if not agent_dir.exists():
        return candidates

    for preferred_name in ("gemini-cli.txt", "codex.txt"):
        preferred_path = agent_dir / preferred_name
        if preferred_path.exists():
            candidates.append(preferred_path)

    for log_path in sorted(agent_dir.glob("*.txt")):
        if log_path not in candidates:
            candidates.append(log_path)

    return candidates


def _normalize_predictions(
    predictions: dict[str, Any] | list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Normalize supported prediction payload variants to {'properties': [...]}."""
    property_keys = {
        "id",
        "material_or_system",
        "property_name",
        "value_string",
        "value_unit",
        "location",
        "conditions",
        "category",
        "notes",
        "method",
    }

    if isinstance(predictions, list):
        if all(
            isinstance(prop, dict) and bool(property_keys.intersection(prop))
            for prop in predictions
        ):
            return {"properties": predictions}
        return None

    properties = predictions.get("properties")
    if not isinstance(properties, list):
        return None
    if not all(isinstance(prop, dict) for prop in properties):
        return None
    return predictions


def _is_prediction_payload(
    predictions: dict[str, Any] | list[dict[str, Any]],
) -> bool:
    """Return whether a parsed payload matches the Harbor prediction schema."""
    normalized = _normalize_predictions(predictions)
    if normalized is None:
        return False
    return isinstance(normalized.get("properties"), list)


def _load_trial_predictions(
    trial_dir: Path,
) -> dict[str, Any] | list[dict[str, Any]] | None:
    """Load JSON predictions from predictions.json file in a single trial directory.

    Args:
        trial_dir: Path to the Harbor trial directory

    Returns:
        Parsed JSON data (either dict or list), or None if not found

    """
    for log_path in _load_prediction_candidates(trial_dir):
        try:
            content = log_path.read_text()
        except Exception:
            continue

        parsed_payload = _parse_json_payload(content)
        if parsed_payload is not None and _is_prediction_payload(parsed_payload):
            return parsed_payload

        # JSONL agent logs need to be decoded back into assistant text first.
        decoded = _extract_text_from_jsonlines_log(content)
        text = decoded or content

        parsed_payload = _parse_json_payload(text)
        if parsed_payload is not None and _is_prediction_payload(parsed_payload):
            return parsed_payload

        extracted_obj = _extract_first_json_object(text)
        if extracted_obj is not None and _is_prediction_payload(extracted_obj):
            return extracted_obj

        extracted_arr = _extract_first_json_array(text)
        if extracted_arr is not None and _is_prediction_payload(extracted_arr):
            return extracted_arr

    return None


def count_trials_per_agent_model(jobs_dir: Path) -> pd.DataFrame:
    """Count the number of trials per agent/model combination in a Harbor jobs directory.

    Args:
        jobs_dir: Path to the Harbor jobs directory containing batch subdirectories

    Returns:
        DataFrame with columns: agent, model, num_trials

    """
    jobs_dir = jobs_dir.resolve()
    if not jobs_dir.exists():
        raise FileNotFoundError(f"Jobs directory not found: {jobs_dir}")

    counts: dict[tuple[str | None, str | None], int] = {}
    for batch_dir in sorted(jobs_dir.iterdir()):
        if not batch_dir.is_dir():
            continue
        # get the agent and model name from the batch_dir config.json
        agent, model = None, None
        config_path = batch_dir / "config.json"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text())
                if config.get("agents"):
                    agent = config["agents"][0].get("name")
                    model = config["agents"][0].get("model_name")
            except Exception:
                pass

        key = (agent, model)
        for trial_dir in sorted(batch_dir.iterdir()):
            if not trial_dir.is_dir():
                continue
            counts[key] = counts.get(key, 0) + 1

    rows = [
        {"agent": agent, "model": model, "num_trials": count}
        for (agent, model), count in counts.items()
    ]
    return pd.DataFrame(rows)


def get_harbor_data(jobs_dir: Path) -> pd.DataFrame:
    """Load predictions from all trials in a Harbor jobs directory.

    Iterates through batches and trials in the jobs directory structure:
    jobs_dir/
      batch_1/
        trial_1/verifier/predictions.json
        trial_2/verifier/predictions.json
      batch_2/
        ...

    Args:
        jobs_dir: Path to the Harbor jobs directory containing batch subdirectories

    Returns:
        DataFrame containing:
        - batch: batch directory name
        - trial_id: trial directory name
        - refno: reference number (if available in trial data)
        - exploded predictions: parsed JSON data from the trial

    Raises:
        FileNotFoundError: If jobs_dir doesn't exist
        ValueError: If no valid trials found

    """
    jobs_dir = jobs_dir.resolve()
    if not jobs_dir.exists():
        raise FileNotFoundError(f"Jobs directory not found: {jobs_dir}")

    dfs = []
    for batch_dir in sorted(jobs_dir.iterdir()):
        if not batch_dir.is_dir():
            continue
        # get the agent and model name from the batch_dir config.json
        agent, model = None, None
        config_path = batch_dir / "config.json"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text())
                if config.get("agents"):
                    agent = config["agents"][0].get("name")
                    model = config["agents"][0].get("model_name")
            except Exception:
                pass

        for trial_dir in sorted(batch_dir.iterdir()):
            if not trial_dir.is_dir():
                continue
            predictions = _load_trial_predictions(trial_dir)
            if predictions is None:
                logger.warning(f"No valid predictions found in trial: {trial_dir}")
                continue
            predictions = _normalize_predictions(predictions)
            if predictions is None:
                logger.warning(
                    f"Unsupported prediction payload found in trial: {trial_dir}"
                )
                continue
            if "properties" not in predictions:
                logger.warning(
                    f"'properties' key not found in predictions for trial: {trial_dir}"
                )
                continue
            if len(predictions["properties"]) == 0:
                logger.warning(
                    f"No properties found in predictions for trial: {trial_dir}"
                )
                continue
            # HACK: if "id" key is missing from any property in the predictions list,
            # then assign a dummy id to each property based on its index using uuid
            for prop in predictions["properties"]:
                if "id" not in prop:
                    prop["id"] = f"prop_{uuid.uuid4()}"
            # Get refno from trial_dir name (e.g., "epl0330153__4QUtrB2")
            refno, _ = trial_dir.name.split("__")

            df = pd.DataFrame(
                data={
                    "agent": agent,
                    "model": model,
                    "batch": batch_dir.name,
                    "trial_id": trial_dir.name,
                    "refno": refno,
                    "properties": predictions,
                }
            )
            # explode predictions into separate rows
            df = df.explode(column="properties").reset_index(drop=True)
            df_properties = pd.json_normalize(df["properties"])
            df = pd.concat([df.drop(columns=["properties"]), df_properties], axis=1)
            dfs.append(df)
    if not dfs:
        raise ValueError(f"No valid trials found in jobs directory: {jobs_dir}")
    df = pd.concat(dfs, ignore_index=True)
    return df
