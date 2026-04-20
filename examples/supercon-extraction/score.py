"""Score all trials in a Harbor jobs dir (or zeroshot output dir) concurrently.

Runs ``compute_mean_recall_precision`` from check_prediction.py per trial
and writes one CSV row per (agent, model, refno) into
``<output_dir>/scores.csv``.

Usage:
    # Harbor jobs dir
    uv run python score.py -jd jobs-test -od out-test-0419 \
        --hf_repo kilian-group/supercon-extraction --hf_split full \
        --hf_revision v0.2.1 --max_concurrent 5

    # Zeroshot output dir (after `pbench-eval`)
    uv run python score.py -od out-test-0419 \
        --hf_repo kilian-group/supercon-extraction --hf_split full \
        --hf_revision v0.2.1
"""

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from tabulate import tabulate

import pbench
from pbench_eval.harbor_utils import get_harbor_data

from check_prediction import _properties_to_df, compute_mean_recall_precision

logger = logging.getLogger(__name__)

MAX_CONCURRENT_TASKS = 5


parser = argparse.ArgumentParser(
    description=(
        "Score all trials via compute_mean_recall_precision and write a CSV "
        "of per-trial F1/recall/precision."
    )
)
parser = pbench.add_base_args(parser)
parser.add_argument(
    "--conversion_factors_path",
    type=Path,
    default=Path("scoring/si_conversion_factors.csv"),
    help=(
        "Optional SI unit-conversion CSV (indexed by property_unit with a "
        "`conversion_factor` column). Silently skipped if the file is missing."
    ),
)
parser.add_argument(
    "--max_concurrent",
    type=int,
    default=MAX_CONCURRENT_TASKS,
    help=(
        f"Max concurrent trials scored in parallel (default: {MAX_CONCURRENT_TASKS}). "
        "Each trial fans out multiple LLM judge calls internally."
    ),
)
args = parser.parse_args()

pbench.setup_logging(args.log_level)
logging.getLogger("google_genai.models").setLevel(logging.WARNING)
load_dotenv()

if args.output_dir is None:
    parser.error("--output_dir is required")
if args.hf_repo is None or args.hf_split is None:
    parser.error("--hf_repo and --hf_split are required")
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY not found in environment.")


def _load_trials_df() -> pd.DataFrame:
    """Load one row per trial from either Harbor jobs dir or zeroshot JSON dir."""
    if args.jobs_dir is not None:
        logger.info(f"Loading Harbor trials from {args.jobs_dir}")
        return get_harbor_data(args.jobs_dir)

    preds_dir = args.output_dir / args.preds_dirname
    files = list(preds_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {preds_dir}")
    logger.info(f"Loading {len(files)} zeroshot trial JSONs from {preds_dir}")
    trials = []
    for f in files:
        with f.open() as fp:
            payload = json.load(fp)
        trials.append(
            {
                "agent": payload.get("agent"),
                "model": payload.get("model"),
                "refno": payload["refno"],
                "properties": payload["properties"],
            }
        )
    return pd.DataFrame(trials)


def _load_gt_by_refno() -> dict[str, list[dict]]:
    """Load HF ground truth and return {refno: list[property dict]}."""
    logger.info(
        f"Loading GT dataset: {args.hf_repo} "
        f"(split={args.hf_split}, revision={args.hf_revision or 'main'})"
    )
    ds = load_dataset(args.hf_repo, split=args.hf_split, revision=args.hf_revision)
    df_gt = ds.to_pandas()
    return {row["refno"]: list(row["properties"]) for _, row in df_gt.iterrows()}


def _load_conversion_df() -> pd.DataFrame | None:
    """Load conversion factors CSV if present; silently skip otherwise."""
    if args.conversion_factors_path.exists():
        logger.info(f"Loading conversion factors from {args.conversion_factors_path}")
        return pd.read_csv(args.conversion_factors_path, index_col=0)
    logger.info(
        f"Conversion factors file not found at {args.conversion_factors_path}; "
        "skipping unit conversion"
    )
    return None


async def score_one_trial(
    trial_row: pd.Series,
    gt_by_refno: dict[str, list[dict]],
    conversion_df: pd.DataFrame | None,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Run compute_mean_recall_precision for one trial; return a result row."""
    async with semaphore:
        refno = trial_row["refno"]
        agent = trial_row.get("agent") or ""
        model = trial_row.get("model") or ""
        logger.info(f"Scoring {agent=} {model=} {refno=}...")

        gt_props = gt_by_refno.get(refno)
        if gt_props is None:
            logger.warning(f"No ground truth for refno={refno}; skipping")
            return {"agent": agent, "model": model, "refno": refno, "error": "no_gt"}

        df_pred = _properties_to_df(
            trial_row["properties"], refno=refno, id_column="id_pred"
        )
        df_gt = _properties_to_df(gt_props, refno=refno, id_column="id_gt")

        try:
            mean_recall, mean_precision = await compute_mean_recall_precision(
                df_pred, df_gt, conversion_df=conversion_df
            )
        except Exception as exc:
            logger.error(
                f"{agent=} {model=} {refno=} failed: {type(exc).__name__}: {exc}",
                exc_info=exc,
            )
            return {
                "agent": agent,
                "model": model,
                "refno": refno,
                "error": f"{type(exc).__name__}: {exc}",
            }

        f1 = 2.0 * mean_recall * mean_precision / (mean_recall + mean_precision + 1e-8)
        return {
            "agent": agent,
            "model": model,
            "refno": refno,
            "mean_recall": float(mean_recall),
            "mean_precision": float(mean_precision),
            "f1": float(f1),
        }


async def _score_all() -> None:
    df_trials = _load_trials_df()
    logger.info(f"Loaded {len(df_trials)} trial(s)")

    gt_by_refno = _load_gt_by_refno()
    conversion_df = _load_conversion_df()

    semaphore = asyncio.Semaphore(args.max_concurrent)
    tasks = [
        score_one_trial(row, gt_by_refno, conversion_df, semaphore)
        for _, row in df_trials.iterrows()
    ]
    logger.info(
        f"Scoring {len(tasks)} trials with max {args.max_concurrent} concurrent..."
    )
    results = await asyncio.gather(*tasks)

    df_results = pd.DataFrame(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "scores.csv"
    df_results.to_csv(out_path, index=False)
    logger.info(f"Wrote {len(df_results)} rows to {out_path}")

    print(tabulate(df_results, headers="keys", tablefmt="github", floatfmt=".4f"))


asyncio.run(_score_all())
