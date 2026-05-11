"""Generate embeddings for predicted property names from Harbor jobs or CSV files.

This script loads predictions from Harbor job directories or CSV files, generates
embeddings for unique property names using the Gemini embedding model, and saves
them to parquet files.

Usage:
    # From Harbor jobs directory
    uv run python generate_pred_embeddings.py -jd JOBS_DIR -od OUTPUT_DIR

    # From CSV files
    uv run python generate_pred_embeddings.py -od OUTPUT_DIR -pd preds
"""

import argparse
import json
import logging

import pandas as pd
from dotenv import load_dotenv
from slugify import slugify

import pbench
from pbench_eval.harbor_utils import get_harbor_data

from check_prediction import generate_embeddings

load_dotenv()

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description="Generate embeddings for predicted property names."
)
parser = pbench.add_base_args(parser)
parser.add_argument(
    "--agent",
    "-a",
    type=str,
    default="gemini-cli",
    help="Agent name used in Harbor jobs (default: gemini-cli)",
)
args = parser.parse_args()
pbench.setup_logging(args.log_level)

jobs_dir = args.jobs_dir
force = args.force
preds_dirname = args.preds_dirname

if jobs_dir is not None:
    df = get_harbor_data(jobs_dir)
else:
    if args.output_dir is None:
        parser.error("--output_dir is required when not using --jobs_dir")
    preds_dir = args.output_dir / preds_dirname
    preds_files = list(preds_dir.glob("*.json"))
    if not preds_files:
        raise FileNotFoundError(f"No JSON files found in {preds_dir}")
    trials = []
    for file in preds_files:
        with file.open() as f:
            payload = json.load(f)
        trials.append(
            {
                "agent": payload.get("agent"),
                "model": payload.get("model"),
                "refno": payload["refno"],
                "properties": payload["properties"],
            }
        )
    df = pd.DataFrame(trials)

if args.output_dir is None:
    parser.error("--output_dir is required")

embeddings_dir = args.output_dir / "pred_embeddings"
embeddings_dir.mkdir(parents=True, exist_ok=True)

for _, row in df.iterrows():
    # Coerce None → "" so slugify doesn't choke (oracle batches have no model_name).
    agent = row["agent"] or ""
    model = row["model"] or ""
    refno = row["refno"]
    save_path = embeddings_dir / f"{slugify(agent)}_{slugify(model)}_{refno}.parquet"
    if save_path.exists() and not force:
        logger.info(
            f"Embeddings already exist for {agent=} {model=} {refno=}, skipping..."
        )
        continue

    logger.info(f"Generating embeddings for {agent=} {model=} {refno=}...")
    property_names = [
        p["property_name"] for p in row["properties"] if p.get("property_name")
    ]
    unique_property_names = list(set(property_names))
    embeddings = generate_embeddings(unique_property_names)
    embeddings_df = pd.DataFrame(
        {
            "refno": [refno] * len(unique_property_names),
            "property_name": unique_property_names,
            "embedding": embeddings,
            "agent": [agent] * len(unique_property_names),
            "model": [model] * len(unique_property_names),
        }
    )
    embeddings_df.to_parquet(save_path)
    logger.info(f"Saved embeddings to {save_path} with {len(embeddings_df)} rows")
