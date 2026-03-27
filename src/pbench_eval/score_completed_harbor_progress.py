"""Score completed Harbor batches and write rolling F1 summaries.

Usage:
    uv run python src/pbench_eval/score_completed_harbor_progress.py \
        --jobs_dir JOBS_DIR --output_dir OUTPUT_DIR
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

import pbench
from pbench_eval.harbor_utils import get_harbor_data

logger = logging.getLogger(__name__)


def _load_batch_config(batch_dir: Path) -> tuple[str, str]:
    """Return the agent and model configured for a Harbor batch."""
    config_path = batch_dir / "config.json"
    if not config_path.exists():
        return "unknown", "unknown"

    try:
        config = json.loads(config_path.read_text())
    except Exception:
        return "unknown", "unknown"

    agents = config.get("agents") or []
    if not agents:
        return "unknown", "unknown"

    agent = str(agents[0].get("name") or "unknown")
    model = str(agents[0].get("model_name") or "unknown")
    return agent, model


def _is_completed_batch(batch_dir: Path) -> bool:
    """Return whether a Harbor batch has finished and written its result.json."""
    result_path = batch_dir / "result.json"
    if not result_path.exists():
        return False

    try:
        result = json.loads(result_path.read_text())
    except Exception:
        return False

    stats = result.get("stats") or {}
    n_total_trials = result.get("n_total_trials")
    n_trials = stats.get("n_trials")
    finished_at = result.get("finished_at")
    return (
        bool(finished_at) and n_total_trials is not None and n_trials == n_total_trials
    )


def _remove_path(path: Path) -> None:
    """Delete a file, symlink, or directory."""
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    for child in path.iterdir():
        _remove_path(child)
    path.rmdir()


def _sync_completed_batch_snapshot(jobs_dir: Path, snapshot_dir: Path) -> list[Path]:
    """Create a symlink snapshot containing only completed Harbor batches."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    completed_batches = sorted(
        batch_dir
        for batch_dir in jobs_dir.iterdir()
        if batch_dir.is_dir() and _is_completed_batch(batch_dir)
    )
    desired_links = {
        batch_dir.name: batch_dir.resolve() for batch_dir in completed_batches
    }

    for existing_path in snapshot_dir.iterdir():
        desired_target = desired_links.get(existing_path.name)
        if desired_target is None:
            _remove_path(existing_path)
            continue
        if existing_path.is_symlink() and existing_path.resolve() == desired_target:
            continue
        _remove_path(existing_path)

    for link_name, target_path in desired_links.items():
        link_path = snapshot_dir / link_name
        if link_path.exists() or link_path.is_symlink():
            continue
        link_path.symlink_to(target_path, target_is_directory=True)

    metadata: list[dict[str, Any]] = []
    for batch_dir in completed_batches:
        agent, model = _load_batch_config(batch_dir)
        trial_refnos = sorted(
            {
                trial_dir.name.split("__", 1)[0]
                for trial_dir in batch_dir.iterdir()
                if trial_dir.is_dir() and "__" in trial_dir.name
            }
        )
        metadata.append(
            {
                "batch": batch_dir.name,
                "path": str(batch_dir.resolve()),
                "agent": agent,
                "model": model,
                "n_refnos": len(trial_refnos),
                "refnos": trial_refnos,
            }
        )
    (snapshot_dir / "completed_batches.json").write_text(json.dumps(metadata, indent=2))
    return completed_batches


def _run_module(module_name: str, args: list[str]) -> None:
    """Run a Python module in the current environment."""
    cmd = [sys.executable, "-m", module_name, *args]
    logger.info("Executing: %s", " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, check=True)


def _count_trials(snapshot_dir: Path) -> dict[tuple[str, str], int]:
    """Count unique task refnos across the completed-batch snapshot."""
    counts: dict[tuple[str, str], int] = {}
    for batch_dir in sorted(snapshot_dir.iterdir()):
        if not batch_dir.is_dir():
            continue
        agent, model = _load_batch_config(batch_dir)
        refnos = {
            trial_dir.name.split("__", 1)[0]
            for trial_dir in batch_dir.iterdir()
            if trial_dir.is_dir() and "__" in trial_dir.name
        }
        key = (agent, model)
        counts[key] = counts.get(key, 0) + len(refnos)
    return counts


def _write_current_f1_summary(
    output_dir: Path,
    snapshot_dir: Path,
    source: str | None,
) -> None:
    """Write a compact rolling F1 summary CSV for the current completed snapshot."""
    f1_by_refno_path = output_dir / "scores" / "f1_by_refno.csv"
    summary_path = output_dir / "current_f1_summary.csv"
    if not f1_by_refno_path.exists():
        empty = pd.DataFrame(
            columns=[
                "source",
                "agent",
                "model",
                "num_trials",
                "scored_refnos",
                "avg_precision",
                "avg_recall",
                "avg_f1",
            ]
        )
        empty.to_csv(summary_path, index=False)
        return

    f1_by_refno = pd.read_csv(f1_by_refno_path)
    if f1_by_refno.empty:
        empty = pd.DataFrame(
            columns=[
                "source",
                "agent",
                "model",
                "num_trials",
                "scored_refnos",
                "avg_precision",
                "avg_recall",
                "avg_f1",
            ]
        )
        empty.to_csv(summary_path, index=False)
        return

    trial_counts = _count_trials(snapshot_dir)
    summary = (
        f1_by_refno.groupby(["agent", "model"], dropna=False)
        .agg(
            scored_refnos=("refno", "nunique"),
            avg_precision=("precision_score", "mean"),
            avg_recall=("recall_score", "mean"),
            avg_f1=("f1_score", "mean"),
        )
        .reset_index()
    )
    summary["num_trials"] = summary.apply(
        lambda row: trial_counts.get((row["agent"], row["model"]), 0),
        axis=1,
    )
    if source is not None:
        summary.insert(0, "source", source)
    else:
        summary.insert(0, "source", "")

    summary = summary[
        [
            "source",
            "agent",
            "model",
            "num_trials",
            "scored_refnos",
            "avg_precision",
            "avg_recall",
            "avg_f1",
        ]
    ]
    summary.to_csv(summary_path, index=False)
    logger.info("Saved rolling F1 summary to %s", summary_path)


def _write_progress_metadata(
    output_dir: Path,
    completed_batches: list[Path],
    snapshot_dir: Path,
    source: str | None,
) -> None:
    """Persist the current completed-batch state alongside the rolling scores."""
    payload = {
        "source": source,
        "jobs_dir": str(snapshot_dir.parent.resolve()),
        "snapshot_dir": str(snapshot_dir.resolve()),
        "n_completed_batches": len(completed_batches),
        "completed_batches": [batch_dir.name for batch_dir in completed_batches],
    }
    (output_dir / "progress_status.json").write_text(json.dumps(payload, indent=2))


def _snapshot_has_valid_trials(snapshot_dir: Path) -> bool:
    """Return whether the completed-batch snapshot contains any valid predictions."""
    try:
        harbor_df = get_harbor_data(snapshot_dir)
    except ValueError:
        return False
    return not harbor_df.empty


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Score completed Harbor batches and write rolling F1 summaries."
    )
    parser = pbench.add_base_args(parser)
    parser.set_defaults(
        model_name=os.environ.get("SUPERCON_SCORE_MATCH_MODEL", "gemini-2.5-flash"),
        hf_repo=os.environ.get(
            "SUPERCON_SCORE_HF_REPO", "kilian-group/supercon-extraction"
        ),
        hf_split=os.environ.get("SUPERCON_SCORE_HF_SPLIT", "full"),
        hf_revision=os.environ.get("SUPERCON_SCORE_HF_REVISION", "v0.2.1"),
    )
    parser.add_argument(
        "--snapshot_dir",
        type=Path,
        default=None,
        help="Directory for the completed-batch symlink snapshot.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Optional paper source label to include in the summary CSV.",
    )
    parser.add_argument(
        "--prompt_path",
        type=Path,
        default=Path(
            os.environ.get(
                "SUPERCON_SCORE_PROMPT_PATH", "prompts/property_matching_prompt.md"
            )
        ),
        help="Prompt path for property matching.",
    )
    parser.add_argument(
        "--rubric_path",
        type=Path,
        default=Path(
            os.environ.get("SUPERCON_SCORE_RUBRIC_PATH", "scoring/rubric_4.csv")
        ),
        help="Rubric CSV path.",
    )
    parser.add_argument(
        "--conversion_factors_path",
        type=Path,
        default=Path(
            os.environ.get(
                "SUPERCON_SCORE_CONVERSION_FACTORS_PATH",
                "scoring/si_conversion_factors.csv",
            )
        ),
        help="Conversion-factors CSV path.",
    )
    parser.add_argument(
        "--matching_mode",
        type=str,
        default=os.environ.get("SUPERCON_SCORE_MATCHING_MODE", "material"),
        choices=["material", "conditions"],
        help="Matching mode for precision/recall/F1.",
    )
    args = parser.parse_args()
    if args.jobs_dir is None:
        raise SystemExit("--jobs_dir is required.")
    if args.output_dir is None:
        raise SystemExit("--output_dir is required.")
    if args.hf_repo is None or args.hf_split is None or args.hf_revision is None:
        raise SystemExit("--hf_repo, --hf_split, and --hf_revision must be set.")
    return args


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    pbench.setup_logging(args.log_level)

    jobs_dir: Path = args.jobs_dir.resolve()
    output_dir: Path = args.output_dir.resolve()
    snapshot_dir: Path = (
        args.snapshot_dir.resolve()
        if args.snapshot_dir is not None
        else (output_dir / "completed_jobs_snapshot").resolve()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    completed_batches = _sync_completed_batch_snapshot(jobs_dir, snapshot_dir)
    _write_progress_metadata(output_dir, completed_batches, snapshot_dir, args.source)

    if not completed_batches:
        logger.warning("No completed Harbor batches found under %s", jobs_dir)
        _write_current_f1_summary(output_dir, snapshot_dir, args.source)
        return

    if not _snapshot_has_valid_trials(snapshot_dir):
        logger.warning(
            "No valid Harbor trials found in completed snapshot under %s", snapshot_dir
        )
        _write_current_f1_summary(output_dir, snapshot_dir, args.source)
        return

    force_args = ["--force"] if args.force else []
    log_args = ["--log_level", args.log_level]

    _run_module(
        "pbench_eval.generate_pred_embeddings",
        [
            "-jd",
            str(snapshot_dir),
            "-od",
            str(output_dir),
            *force_args,
            *log_args,
        ],
    )
    _run_module(
        "pbench_eval.generate_matches_cli",
        [
            "-jd",
            str(snapshot_dir),
            "-od",
            str(output_dir),
            "-m",
            args.model_name,
            "--hf_repo",
            args.hf_repo,
            "--hf_split",
            args.hf_split,
            "--hf_revision",
            args.hf_revision,
            "--prompt_path",
            str(args.prompt_path),
            *force_args,
            *log_args,
        ],
    )
    _run_module(
        "pbench_eval.score_precision_cli",
        [
            "-jd",
            str(snapshot_dir),
            "-od",
            str(output_dir),
            "-m",
            args.model_name,
            "--rubric_path",
            str(args.rubric_path),
            "--conversion_factors_path",
            str(args.conversion_factors_path),
            "--matching_mode",
            args.matching_mode,
            *force_args,
            *log_args,
        ],
    )
    _run_module(
        "pbench_eval.score_recall_cli",
        [
            "-jd",
            str(snapshot_dir),
            "-od",
            str(output_dir),
            "-m",
            args.model_name,
            "--rubric_path",
            str(args.rubric_path),
            "--conversion_factors_path",
            str(args.conversion_factors_path),
            "--matching_mode",
            args.matching_mode,
            *force_args,
            *log_args,
        ],
    )
    _run_module(
        "pbench_eval.score_f1_cli",
        [
            "-jd",
            str(snapshot_dir),
            "-od",
            str(output_dir),
            "-m",
            args.model_name,
            "--rubric_path",
            str(args.rubric_path),
            "--conversion_factors_path",
            str(args.conversion_factors_path),
            "--matching_mode",
            args.matching_mode,
            *force_args,
            *log_args,
        ],
    )
    _write_current_f1_summary(output_dir, snapshot_dir, args.source)


if __name__ == "__main__":
    main()
