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
from slugify import slugify

# llm imports
from llm_utils import (
    get_llm,
    InferenceGenerationConfig,
)


# pbench imports
from llm_utils.common import LLMChat
import pbench
from pbench_eval.match import generate_property_name_matches

# local imports
from utils import (
    HF_DATASET_NAME,
    HF_DATASET_REVISION,
    HF_DATASET_SPLIT,
    get_harbor_data,
)

logger = logging.getLogger(__name__)

# Concurrency limit for parallel processing
MAX_CONCURRENT_TASKS = 10


async def process_single_group(
    agent: str,
    model: str,
    refno: str,
    group: pd.DataFrame,
    df_gt: pd.DataFrame,
    pred_embeddings_dir: Path,
    pred_matches_dir: Path,
    pred_responses_dir: Path,
    gt_matches_dir: Path,
    gt_responses_dir: Path,
    llm: LLMChat,
    inf_gen_config: InferenceGenerationConfig,
    prompt_template: str,
    model_name: str,
    top_k: int,
    force: bool,
    semaphore: asyncio.Semaphore,
) -> None:
    """Process a single (agent, model, refno) group."""
    async with semaphore:
        logger.info(f"Processing {agent=} {model=} {refno=}...")

        # Prepare predicted data
        df_pred = group
        df_pred_embeddings = pd.read_parquet(
            pred_embeddings_dir / f"{slugify(agent)}_{slugify(model)}_{refno}.parquet"
        )
        df_pred = df_pred.merge(
            df_pred_embeddings[["property_name", "embedding"]],
            on="property_name",
            how="left",
        )
        df_pred["context"] = df_pred["location.evidence"]

        # Prepare ground truth data for this refno
        df_gt_refno = df_gt[df_gt["refno"].str.lower() == refno.lower()].drop(
            columns=["refno"]
        )

        # Define paths
        pred_matches_path = (
            pred_matches_dir
            / f"pred_matches_{slugify(agent)}_{slugify(model)}_{refno}_judge={model_name}_k={top_k}.csv"
        )
        pred_responses_path = (
            pred_responses_dir
            / f"pred_responses_{slugify(agent)}_{slugify(model)}_{refno}_judge={model_name}_k={top_k}.csv"
        )
        gt_matches_path = (
            gt_matches_dir
            / f"gt_matches_{slugify(agent)}_{slugify(model)}_{refno}_judge={model_name}_k={top_k}.csv"
        )
        gt_responses_path = (
            gt_responses_dir
            / f"gt_responses_{slugify(agent)}_{slugify(model)}_{refno}_judge={model_name}_k={top_k}.csv"
        )

        # Build tasks for pred and gt matches (run in parallel within this group)
        tasks: list[tuple[str, any]] = []

        if not pred_matches_path.exists() or force:
            tasks.append(
                (
                    "pred",
                    generate_property_name_matches(
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
                    ),
                )
            )
        else:
            logger.info(
                f"Skipping {agent=} {model=} {refno=} pred matches (already exist)"
            )

        if not gt_matches_path.exists() or force:
            tasks.append(
                (
                    "gt",
                    generate_property_name_matches(
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
                    ),
                )
            )
        else:
            logger.info(
                f"Skipping {agent=} {model=} {refno=} gt matches (already exist)"
            )

        if not tasks:
            return

        # Run pred and gt matches in parallel
        results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)

        for (task_type, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                logger.error(
                    f"Error processing {agent=} {model=} {refno=} {task_type}: {result}"
                )
                continue

            if task_type == "pred":
                df_pred_matches, df_pred_responses = result
                df_pred_matches.drop(columns=["embedding_pred", "embedding_gt"]).to_csv(
                    pred_matches_path, index=False
                )
                logger.info(f"Saved pred matches to {pred_matches_path}")
                df_pred_responses.to_csv(pred_responses_path, index=False)
                logger.info(f"Saved pred responses to {pred_responses_path}")
            else:
                df_gt_matches, df_gt_responses = result
                df_gt_matches.drop(columns=["embedding_gt", "embedding_pred"]).to_csv(
                    gt_matches_path, index=False
                )
                logger.info(f"Saved gt matches to {gt_matches_path}")
                df_gt_responses.to_csv(gt_responses_path, index=False)
                logger.info(f"Saved gt responses to {gt_responses_path}")


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
    preds_dirname = args.preds_dirname

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

    # Construct GT embeddings path from HF dataset parameters
    hf_dataset_name = args.hf_repo or HF_DATASET_NAME
    hf_split_name = args.hf_split or HF_DATASET_SPLIT
    hf_revision = args.hf_revision or HF_DATASET_REVISION
    gt_embeddings_filename = (
        f"embeddings_{slugify(f'{hf_dataset_name}_{hf_split_name}_{hf_revision}')}.json"
    )
    gt_embeddings_path = Path("scoring") / gt_embeddings_filename

    logger.info(f"Loading ground truth embeddings from {gt_embeddings_path}...")
    df_gt_embeddings = pd.read_json(gt_embeddings_path)

    #
    # Load extracted properties
    #
    if jobs_dir is not None:
        # Load predictions from Harbor jobs directory
        df = get_harbor_data(jobs_dir)
    else:
        # Load predictions from CSV files
        pred_properties_dir = output_dir / preds_dirname
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

    # Create semaphore to limit concurrency
    max_concurrent = args.max_concurrent
    semaphore = asyncio.Semaphore(max_concurrent)

    # Build list of tasks for all groups
    tasks = []
    for (agent, model, refno), group in df.groupby(["agent", "model", "refno"]):
        if args.refno is not None and refno != args.refno:
            continue
        tasks.append(
            process_single_group(
                agent=agent,
                model=model,
                refno=refno,
                group=group,
                df_gt=df_gt,
                pred_embeddings_dir=pred_embeddings_dir,
                pred_matches_dir=pred_matches_dir,
                pred_responses_dir=pred_responses_dir,
                gt_matches_dir=gt_matches_dir,
                gt_responses_dir=gt_responses_dir,
                llm=llm,
                inf_gen_config=inf_gen_config,
                prompt_template=prompt_template,
                model_name=model_name,
                top_k=top_k,
                force=force,
                semaphore=semaphore,
            )
        )

    logger.info(
        f"Processing {len(tasks)} groups with max {max_concurrent} concurrent tasks..."
    )
    await asyncio.gather(*tasks, return_exceptions=True)


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
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=MAX_CONCURRENT_TASKS,
        help=f"Maximum number of concurrent tasks (default: {MAX_CONCURRENT_TASKS})",
    )

    args = parser.parse_args()

    # Setup logging
    pbench.setup_logging(args.log_level)
    # Load env variables
    load_dotenv()
    # Check env
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("GOOGLE_API_KEY not found in environment.")

    # Run main
    asyncio.run(main(args))
