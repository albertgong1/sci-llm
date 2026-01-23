"""Utility constants and functions for supercon-extraction scripts.

Reference for SuperCon integer codes: https://mdr.nims.go.jp/filesets/28e52e3f-8f92-4656-abb6-354e4cef11a1/download
"""

import json
import re
import uuid
from json import JSONDecoder
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Dataset configuration
HF_DATASET_NAME = "kilian-group/supercon-extraction"
HF_DATASET_REVISION = "main"  # "v0.0.0"
HF_DATASET_SPLIT = "full"
GT_EMBEDDINGS_PATH = Path("scoring/gt_property_name_gemini-embedding-001.json")

RUBRIC_PATH = Path("scoring") / "rubric_4.csv"

# Crystal structure symmetry mapping
CRYSTAL_SYMMETRY: dict[int, str] = {
    1: "cubic",
    2: "tetragonal",
    3: "orthorhombic",
    4: "monoclinic",
    5: "triclinic",
    6: "trigonal",
    7: "hexagonal",
}

# Shape mapping
# NOTE: we insert space before parentheses for formatting consistency
SHAPE: dict[int, str] = {
    1: "single phase (bulk)",
    2: "multi phase (bulk)",
    3: "single crystal (bulk)",
    4: "film",
    5: "film (single)",
}

# Method mapping
METHOD: dict[int, str] = {
    1: "powder sintering method",
    2: "doctor blade method",
    3: "screen printing method",  # typo in original source "screen printing metod"
    4: "extrusion method",
    5: "flux method",
    6: "Top Seeded Solution Growth method",
    7: "floating zone method",
    8: "Liquid Phase epitaxy",
    9: "melt-quench method",
    10: "Bridgeman",
    11: "sol-gel method",
    12: "organic acid base method",
    13: "suspension method",
    14: "spray coating method",
    15: "plasma spray method",
    16: "sputter deposition",
    17: "vacuum deposition",
    18: "CVD method",
    19: "Metal-Organic Chemical Vapor Deposition",
    20: "Vapor Growth method",
    21: "Molecular Beam Epitaxy method",
}

# Analysis method mapping
ANALM: dict[int, str] = {
    1: "X-ray crystallography",
    2: "Neutron crystallography",
    3: "Powder x-ray diffraction",
    4: "Powder neutron diffraction",
    5: "Electron diffraction",
}

# Gap measurement method mapping
GAPMETH: dict[int, str] = {
    1: "tunneling",
    2: "infrared spectroscopy",
    3: "thermal conductivity",
    4: "Raman spectroscopy",
    5: "AC susceptibility",
    6: "nuclear magnetic resonance",
    7: "surface impedance",
    8: "neutron diffraction",
    9: "ultraviolet photoemission spectroscopy",
    10: "microwave transmission",
}

# TC measurement method mapping
TC_MEASUREMENT_METHOD: dict[int, str] = {
    1: "magnetization",
    2: "ac susceptibility",
    3: "resistivity",
    4: "heat capacity",
    5: "tunneling",
    6: "infrared spectroscopy",
    7: "thermal conductivity",
    8: "Raman spectroscopy",
    9: "nuclear magnetic resonance",
    10: "surface impedance",
    11: "neutron diffraction",
    12: "photoemission spectroscopy",
    13: "microwave transmission",
    14: "Others",
}


#
# Helper functions to load predictions from Harbor jobs
#


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


def _load_trial_predictions(
    trial_dir: Path,
) -> dict[str, Any] | list[dict[str, Any]] | None:
    """Load JSON predictions from predictions.json file in a single trial directory.

    Args:
        trial_dir: Path to the Harbor trial directory

    Returns:
        Parsed JSON data (either dict or list), or None if not found

    Raises:
        ValueError: If JSON parsing fails

    """
    # Try to load from predictions.json first
    log_path = trial_dir / "verifier" / "app_output" / "predictions.json"
    if not log_path.exists():
        log_path = trial_dir / "agent" / "gemini-cli.txt"

    try:
        content = log_path.read_text()
    except Exception:
        return None
    # if trial_dir.name == "prb076212504__2s8uVnW":
    # import pdb; pdb.set_trace()
    # First try to extract text from JSONL format
    decoded = _extract_text_from_jsonlines_log(content)
    text = decoded or content

    # Try to extract JSON object first
    extracted_obj = _extract_first_json_object(text)
    if extracted_obj is not None:
        return extracted_obj

    # Fall back to extracting JSON array
    extracted_arr = _extract_first_json_array(text)
    if extracted_arr is not None:
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
        trial_1/verifier/app_output/predictions.json
        trial_2/verifier/app_output/predictions.json
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
    refnos = []
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
            # if trial_dir == Path("/Users/ag2435/sci_llm/src/sci-llm/examples/supercon-extraction/jobs-0119-2/bn4-bs10-gemini-cli-gemini-3-flash-preview-s1/epl10417003__N7A2Qhh"):
            #     import pdb; pdb.set_trace()
            predictions = _load_trial_predictions(trial_dir)
            if predictions is None:
                logger.warning(f"No valid predictions found in trial: {trial_dir}")
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
            refnos.append(refno)
    if not dfs:
        raise ValueError(f"No valid trials found in jobs directory: {jobs_dir}")
    df = pd.concat(dfs, ignore_index=True)
    return df


def sem(x: list, n: int) -> float:
    """Standard error of the mean.

    Args:
        x: List of sample values
        n: Total number of samples (including missing)

    Returns:
        Standard error of the mean

    """
    return np.std(np.concatenate((x, np.zeros(n - len(x)))), ddof=1) / (n**0.5)


def mean_sem_with_n(x: list, n: int) -> str:
    """Format mean ± SEM as a string.

    Args:
        x: List of sample values
        n: Total number of samples (including missing)

    Returns:
        Formatted string "mean ± sem"

    """
    return f"{sum(x) / n:.2f} ± {sem(x, n):.2f}"
