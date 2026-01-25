"""Generate embeddings for ground-truth property names from a HuggingFace dataset.

This script loads a ground-truth dataset from HuggingFace, extracts unique property
names, generates embeddings using the Gemini embedding model, and saves them to JSON.

Usage:
    uv run pbench-gt-embeddings --hf_repo kilian-group/biosurfactants-extraction \
        --hf_revision v0.0.0 --hf_split full
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from slugify import slugify

import pbench
from pbench_eval.match import generate_embeddings

load_dotenv()

logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings for ground-truth property names from a HuggingFace dataset."
    )
    parser = pbench.add_base_args(parser)
    args = parser.parse_args()
    pbench.setup_logging(args.log_level)

    # Validate required arguments
    if not args.hf_repo:
        parser.error("--hf_repo is required")
    if not args.hf_revision:
        parser.error("--hf_revision is required")
    if not args.hf_split:
        parser.error("--hf_split is required")

    repo_name = args.hf_repo
    revision = args.hf_revision
    split = args.hf_split

    # Generate output path from repo_name, split, and revision
    output_filename = f"embeddings_{slugify(f'{repo_name}_{split}_{revision}')}.json"
    output_path = Path("scoring") / output_filename

    # Check if output already exists
    if output_path.exists() and not args.force:
        logger.info(
            f"Output file already exists: {output_path}. Use --force to overwrite."
        )
        return

    # Load ground truth dataset
    logger.info(f"Loading dataset {repo_name} (split={split}, revision={revision})...")
    ds = load_dataset(repo_name, split=split, revision=revision)
    df = ds.to_pandas()

    # Get unique property_names
    df = df.explode(column="properties").reset_index(drop=True)
    df = pd.concat(
        [df.drop(columns=["properties"]), pd.json_normalize(df["properties"])], axis=1
    )
    unique_property_names = df["property_name"].dropna().unique().tolist()
    logger.info(f"Found {len(unique_property_names)} unique property names")

    # Generate embeddings
    logger.info("Generating embeddings...")
    embeddings = generate_embeddings(unique_property_names)

    # Save embeddings to JSON
    data = []
    for name, embedding in zip(unique_property_names, embeddings):
        data.append(
            {
                "property_name": name,
                "embedding": embedding,
            }
        )

    logger.info(f"Saving {len(data)} ground truth embeddings to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Done.")


if __name__ == "__main__":
    main()
