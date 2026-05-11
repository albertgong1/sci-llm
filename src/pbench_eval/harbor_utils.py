"""Utility functions for loading predictions from Harbor jobs.

Usage:
    from pbench_eval.harbor_utils import get_harbor_data
    df = get_harbor_data(jobs_dir)

Each row corresponds to one trial and carries its full `properties` list
(no per-property explode). Consumers that want per-property rows should
iterate df.itertuples() and flatten downstream.
"""

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _load_trial_predictions(trial_dir: Path) -> list[dict] | None:
    """Load `properties` from a trial's predictions.json (strict schema)."""
    path = trial_dir / "verifier" / "predictions.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)["properties"]


def _load_trial_refno(trial_dir: Path) -> str | None:
    """Read refno from the task_meta.json copied into verifier/ by test.sh."""
    path = trial_dir / "verifier" / "task_meta.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)["refno"]


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
        trial_1/verifier/task_meta.json
        trial_2/verifier/predictions.json
        trial_2/verifier/task_meta.json
      batch_2/
        ...

    Args:
        jobs_dir: Path to the Harbor jobs directory containing batch subdirectories

    Returns:
        DataFrame with one row per trial and columns:
        - agent, model (from the batch's config.json)
        - batch: batch directory name
        - trial_id: trial directory name
        - refno: read from trial/verifier/task_meta.json
        - properties: the raw list[dict] from predictions.json (not exploded)

    Raises:
        FileNotFoundError: If jobs_dir doesn't exist
        ValueError: If no valid trials found

    """
    jobs_dir = jobs_dir.resolve()
    if not jobs_dir.exists():
        raise FileNotFoundError(f"Jobs directory not found: {jobs_dir}")

    rows: list[dict] = []
    for batch_dir in sorted(jobs_dir.iterdir()):
        if not batch_dir.is_dir():
            continue
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

            refno = _load_trial_refno(trial_dir)
            if refno is None:
                logger.warning(f"No task_meta.json found in trial: {trial_dir}")
                continue

            properties = _load_trial_predictions(trial_dir)
            if properties is None:
                logger.warning(f"No predictions.json found in trial: {trial_dir}")
                continue
            if len(properties) == 0:
                logger.warning(f"Empty properties list in trial: {trial_dir}")
                continue

            rows.append(
                {
                    "agent": agent,
                    "model": model,
                    "batch": batch_dir.name,
                    "trial_id": trial_dir.name,
                    "refno": refno,
                    "properties": properties,
                }
            )

    if not rows:
        raise ValueError(f"No valid trials found in jobs directory: {jobs_dir}")

    return pd.DataFrame(rows)
