"""Script to match property names to ground truth labels

Below is our two-stage approach:
1. We first use the embeddings to obtain the top-k matches for each property name.
2. We then query an LLM to check if the property names are the same.

Example usage:
```bash
uv run python generate_property_name_matches.py -m gemini-3-pro-preview -od out/
```
"""

# standard imports
import argparse
import json
import os
import logging
import pandas as pd
from dotenv import load_dotenv
import asyncio
from datasets import load_dataset
from pathlib import Path

# llm imports
from llm_utils import (
    get_llm,
    InferenceGenerationConfig,
)


# pbench imports
import pbench
from pbench_eval.match import generate_property_name_matches

# local imports
from utils import (
    HF_DATASET_NAME,
    HF_DATASET_REVISION,
    HF_DATASET_SPLIT,
    get_harbor_data,
    GT_EMBEDDINGS_PATH,
)

logger = logging.getLogger(__name__)


def construct_context(row: pd.Series) -> str:
    """Construct the context from the ground-truth property row.

    Args:
        row: Ground-truth property row

    Returns:
        Context string

    """
    return json.dumps({k: v for k, v in row.items() if k == "value_unit"})


async def main(args: argparse.Namespace) -> None:
    """Main function to verify alias candidates using Gemini LLM."""
    output_dir = args.output_dir
    model_name = args.model_name
    top_k = args.top_k
    force = args.force
    jobs_dir = args.jobs_dir

    # Load prompt template from markdown file
    if True:
        prompt_path = Path("prompts") / "property_matching_prompt.md"
    else:
        # NOTE: this prompt lead to less reliable results with gemini-2.5-flash-lite on Refno JAC2980051
        prompt_path = Path("prompts") / "property_matching_prompt_cache_friendly.md"
    with open(prompt_path, "r") as f:
        prompt_template = f.read()

    #
    # Load embeddings
    #
    pred_embeddings_dir = output_dir / "pred_embeddings"

    # Get embeddings for predicted property names
    pred_embeddings_files = list(pred_embeddings_dir.glob("*.parquet"))
    if not pred_embeddings_files:
        raise FileNotFoundError(f"No embeddings files found in {pred_embeddings_dir}")

    logger.info(f"Loading ground truth embeddings from {GT_EMBEDDINGS_PATH}...")
    df_gt_embeddings = pd.read_json(GT_EMBEDDINGS_PATH)

    #
    # Load extracted properties
    #
    if jobs_dir is not None:
        # Load predictions from Harbor jobs directory
        df = get_harbor_data(jobs_dir)
    else:
        # Load predictions from CSV files
        pred_properties_dir = output_dir / "unsupervised_llm_extraction"
        pred_properties_files = list(pred_properties_dir.glob("*.csv"))
        if not pred_properties_files:
            raise FileNotFoundError(
                f"No properties files found in {pred_properties_dir}"
            )
        dfs = []
        for file in pred_properties_files:
            df = pd.read_csv(file)
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)

    #
    # Load ground truth properties
    #
    hf_dataset_name = args.hf_repo or HF_DATASET_NAME
    hf_split_name = args.hf_split or HF_DATASET_SPLIT
    hf_revision = args.hf_revision or HF_DATASET_REVISION
    logger.info(
        f"Loading dataset from HuggingFace: {hf_dataset_name} "
        f"(revision={hf_revision}, split={hf_split_name})"
    )
    dataset = load_dataset(hf_dataset_name, split=hf_split_name, revision=hf_revision)
    df_gt: pd.DataFrame = dataset.to_pandas()
    # get unique property_names
    df_gt = df_gt.explode(column="properties").reset_index(drop=True)
    df_gt = pd.concat(
        [df_gt[["refno"]], pd.json_normalize(df_gt["properties"])], axis=1
    )
    # Merge GT properties with their embeddings
    df_gt = df_gt.merge(
        df_gt_embeddings[["property_name", "embedding"]],
        on="property_name",
        how="left",
    )
    # construct the context from conditions and value_unit
    df_gt["context"] = df_gt.apply(lambda row: construct_context(row), axis=1)

    #
    # Main Loop
    #
    pred_matches_dir = output_dir / "pred_matches"
    pred_matches_dir.mkdir(parents=True, exist_ok=True)
    pred_responses_dir = output_dir / "pred_responses"
    pred_responses_dir.mkdir(parents=True, exist_ok=True)
    gt_matches_dir = output_dir / "gt_matches"
    gt_matches_dir.mkdir(parents=True, exist_ok=True)
    gt_responses_dir = output_dir / "gt_responses"
    gt_responses_dir.mkdir(parents=True, exist_ok=True)

    # Initialize LLM
    llm = get_llm(args.server, model_name)

    # Create inference config
    inf_gen_config = InferenceGenerationConfig(
        max_output_tokens=args.max_output_tokens,
        output_format="json",
    )

    for refno in df["refno"].unique():
        if args.refno is not None and refno != args.refno:
            continue
        logger.info(f"Processing refno {refno}...")
        # Get predicted properties for this refno
        df_pred = df[df["refno"] == refno]
        # Obtain the embeddings for the predicted properties
        df_pred_embeddings = pd.read_parquet(pred_embeddings_dir / f"{refno}.parquet")
        # Join df_pred and df_pred_embeddings on property_name
        df_pred = df_pred.merge(
            df_pred_embeddings[["property_name", "embedding"]],
            on="property_name",
            how="left",
        )
        # construct the context from location.evidence
        df_pred["context"] = df_pred["location.evidence"]
        # Obtain the ground truth properties for this refno
        # import pdb; pdb.set_trace()
        df_gt_refno = df_gt[df_gt["refno"].str.lower() == refno.lower()].drop(
            columns=["refno"]
        )

        # -- Get top-k matches for each predicted property name (to compute precision) --
        pred_matches_path = (
            pred_matches_dir / f"pred_matches_{refno}_{model_name}_k{top_k}.csv"
        )
        pred_responses_path = (
            pred_responses_dir / f"pred_responses_{refno}_{model_name}_k{top_k}.csv"
        )
        if pred_matches_path.exists() and not force:
            logger.info(f"Skipping refno {refno} because pred matches already exist")
        else:
            df_pred_matches, df_pred_responses = await generate_property_name_matches(
                df_pred,
                df_gt_refno,
                llm,
                inf_gen_config,
                prompt_template,
                top_k=top_k,
                left_on=["property_name", "context"],
                right_on=["property_name", "context"],
                left_suffix="_pred",
                right_suffix="_gt",
            )
            # import pdb; pdb.set_trace()
            df_pred_matches.drop(columns=["embedding_pred", "embedding_gt"]).to_csv(
                pred_matches_path, index=False
            )
            logger.info(f"Saved pred matches to {pred_matches_path}")
            df_pred_responses.to_csv(pred_responses_path, index=False)
            logger.info(f"Saved pred responses to {pred_responses_path}")

        # -- Get top-k matches for each ground truth property name (to compute recall) --
        gt_matches_path = (
            gt_matches_dir / f"gt_matches_{refno}_{model_name}_k{top_k}.csv"
        )
        gt_responses_path = (
            gt_responses_dir / f"gt_responses_{refno}_{model_name}_k{top_k}.csv"
        )
        if gt_matches_path.exists() and not force:
            logger.info(f"Skipping refno {refno} because gt matches already exist")
        else:
            df_gt_matches, df_gt_responses = await generate_property_name_matches(
                df_gt_refno,
                df_pred,
                llm,
                inf_gen_config,
                prompt_template,
                top_k=top_k,
                left_on=["property_name", "context"],
                right_on=["property_name", "context"],
                left_suffix="_gt",
                right_suffix="_pred",
            )
            df_gt_matches.drop(columns=["embedding_gt", "embedding_pred"]).to_csv(
                gt_matches_path, index=False
            )
            logger.info(f"Saved gt matches to {gt_matches_path}")
            df_gt_responses.to_csv(gt_responses_path, index=False)
            logger.info(f"Saved gt responses to {gt_responses_path}")


if __name__ == "__main__":
    #
    # Parse arguments
    #
    parser = argparse.ArgumentParser(
        description="Verify alias candidates using Gemini LLM."
    )
    parser = pbench.add_base_args(parser)
    parser.add_argument(
        "--refno",
        type=str,
        default=None,
        help="Refno to process. If None, process all refnos",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of top matches to return (default: 3)",
    )

    # LLM generation arguments
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=4096,
        help="Maximum number of output tokens for LLM response (default: 4096)",
    )

    args = parser.parse_args()

    # Setup logging
    pbench.setup_logging(args.log_level)
    # Check env
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("GOOGLE_API_KEY not found in environment.")
    # Load env variables
    load_dotenv()

    # Run main
    asyncio.run(main(args))
