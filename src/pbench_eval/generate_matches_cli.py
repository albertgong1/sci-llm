"""Domain-agnostic CLI to match property names to ground truth labels.

This script uses a two-stage approach:
1. First use embeddings to obtain the top-k matches for each property name.
2. Then query an LLM to check if the property names are the same.

Example usage:
```bash
uv run pbench-generate-matches \
    --output_dir ./out \
    --hf_repo kilian-group/supercon-extraction \
    --hf_split full \
    --prompt_path prompts/property_matching_prompt.md \
    --model_name gemini-3-pro-preview
```
"""

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

from llm_utils import get_llm, InferenceGenerationConfig
from llm_utils.common import LLMChat
import pbench
from pbench_eval.match import generate_property_name_matches
from pbench_eval.harbor_utils import get_harbor_data

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
    context_column: str,
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
        # Use configurable context column
        if context_column in df_pred.columns:
            df_pred["context"] = df_pred[context_column]
        else:
            df_pred["context"] = ""

        # Prepare ground truth data for this refno
        if False:
            df_gt_refno = df_gt[df_gt["refno"].str.lower() == refno.lower()].drop(
                columns=["refno"]
            )
        else:
            # NOTE: For Harbor evaluation, the refno for predictions is inferred from the trial dirname,
            # which is slugified. The refno in the GT is not slugified, so we need to slugify it for matching.
            df_gt_refno = df_gt[
                df_gt["refno"].str.lower().apply(lambda x: slugify(x))
                == slugify(refno.lower())
            ].drop(columns=["refno"])

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
    """Main function to match property names using LLM."""
    output_dir = args.output_dir
    model_name = args.model_name
    top_k = args.top_k
    force = args.force
    jobs_dir = args.jobs_dir
    preds_dirname = args.preds_dirname
    context_column = args.context_column

    # Load prompt template from markdown file
    prompt_path = Path(args.prompt_path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
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

    # Load or compute GT embeddings path
    if args.gt_embeddings_path:
        gt_embeddings_path = Path(args.gt_embeddings_path)
    else:
        # Construct GT embeddings path from HF dataset parameters
        hf_dataset_name = args.hf_repo
        hf_split_name = args.hf_split
        hf_revision = args.hf_revision or "main"
        gt_embeddings_filename = f"embeddings_{slugify(f'{hf_dataset_name}_{hf_split_name}_{hf_revision}')}.json"
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
    hf_dataset_name = args.hf_repo
    hf_split_name = args.hf_split
    hf_revision = args.hf_revision or "main"

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
                context_column=context_column,
            )
        )

    logger.info(
        f"Processing {len(tasks)} groups with max {max_concurrent} concurrent tasks..."
    )
    await asyncio.gather(*tasks, return_exceptions=True)


def cli_main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Match property names to ground truth labels using LLM."
    )
    parser = pbench.add_base_args(parser)

    # Required arguments
    parser.add_argument(
        "--prompt_path",
        type=str,
        required=True,
        help="Path to the property matching prompt template (markdown file)",
    )

    # Optional arguments
    parser.add_argument(
        "--gt_embeddings_path",
        type=str,
        default=None,
        help="Path to ground truth embeddings JSON file. If not provided, will be auto-computed from HF params.",
    )
    parser.add_argument(
        "--context_column",
        type=str,
        default="location.evidence",
        help="Column name for context extraction from predictions (default: location.evidence)",
    )
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

    # Validate required arguments
    if args.hf_repo is None:
        parser.error("--hf_repo is required")
    if args.hf_split is None:
        parser.error("--hf_split is required")

    # Setup logging
    pbench.setup_logging(args.log_level)
    # Suppress verbose google_genai logs
    logging.getLogger("google_genai.models").setLevel(logging.WARNING)
    # Load env variables
    load_dotenv()
    # Check env
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("GOOGLE_API_KEY not found in environment.")

    # Run main
    asyncio.run(main(args))


if __name__ == "__main__":
    cli_main()
