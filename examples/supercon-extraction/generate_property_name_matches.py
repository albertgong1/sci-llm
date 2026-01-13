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
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import asyncio
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from collections import OrderedDict
from pathlib import Path

# llm imports
from llm_utils import (
    get_llm,
    LLMChat,
    InferenceGenerationConfig,
    Conversation,
    Message,
    LLMChatResponse,
)


# pbench imports
import pbench
from pbench_eval.match import generate_property_name_matches

logger = logging.getLogger(__name__)


def construct_context(row: pd.Series) -> str:
    """Construct the context from the ground-truth property row.

    Args:
        row: Ground-truth property row

    Returns:
        Context string

    """
    return json.dumps(
        {
            k: v
            for k, v in row.items()
            if k.startswith("conditions.") or k == "value_unit"
        }
    )


PROMPT_TEMPLATE_2 = """\
You are an expert condensed matter physicist evaluating whether two property descriptions refer to the same physical measurement.

## Property 1
- Name: "{name1}"
- Context: "{context1}"

## Property 2
- Name: "{name2}"
- Context: "{context2}"

## Matching Rules:

**SAME property if:**
- Names are synonymous (e.g., "Tc" = "Critical Temperature" = "Superconducting Transition Temperature")
- Abbreviation differences only (e.g., "Jc" = "Critical Current Density")
- Capitalization/formatting differences

**DIFFERENT properties if:**
- Technically distinct measurements:
  - "Tc onset" ≠ "Tc zero" ≠ "Tc midpoint"
  - "Jc" ≠ "Ic" (density vs absolute current)
  - "lattice constant a" ≠ "lattice constant c"
  - "upper critical field Hc2" ≠ "lower critical field Hc1"
- Different measurement orientations when orientation matters (e.g., "resistivity (c-axis)" ≠ "resistivity (ab-plane)")
- Different conditions not reconcilable (e.g., "Tc at 0 GPa" ≠ "Tc at 10 GPa" unless pressure is tracked separately)

## Response Format:
Return JSON only:
{{
  "is_match": boolean,
  "confidence": "high" | "medium" | "low",
  "reason": "concise explanation",
  "matched_via": "direct" | "synonym" | "abbreviation" | "condition_reconciliation" | null
}}
"""


async def check_if_same_property(
    llm: LLMChat,
    inf_gen_config: InferenceGenerationConfig,
    name1: str,
    context1: str,
    name2: str,
    context2: str,
) -> dict:
    """Check if two property names are the same using an LLM.

    Args:
        llm: LLM instance
        inf_gen_config: Inference generation configuration
        name1: name of the first property
        context1: context of the first property
        name2: name of the second property
        context2: context of the second property

    Returns:
        dict: Dictionary containing the result of the check

    """
    # Build conversation
    prompt = PROMPT_TEMPLATE_2.format(
        name1=name1,
        context1=context1,  # use the evidence field as the context
        name2=name2,
        context2=context2,
    )
    conv = Conversation(messages=[Message(role="user", content=[prompt])])

    # Generate response
    response: LLMChatResponse = await llm.generate_response_async(conv, inf_gen_config)
    if response.pred:
        is_match = response.pred.get("is_match", False)
        reason = response.pred.get("reason", "No reason provided")
        confidence = response.pred.get("confidence")
        matched_via = response.pred.get("matched_via")
    else:
        is_match = False
        reason = "Empty response from LLM"
        confidence = None
        matched_via = None

    result = {
        "is_match": is_match,
        "reason": reason,
        "confidence": confidence,
        "matched_via": matched_via,
        "model": llm.model_name,
        "prompt": prompt,
    }

    return result


async def main(args: argparse.Namespace) -> None:
    """Main function to verify alias candidates using Gemini LLM."""
    output_dir = args.output_dir
    model_name = args.model_name
    top_k = 3
    force = args.force

    # Load prompt template from markdown file
    prompt_path = Path("prompts") / "property_matching_prompt.md"
    with open(prompt_path, "r") as f:
        prompt_template = f.read()

    #
    # Load embeddings
    #
    pred_embeddings_dir = output_dir / "pred_embeddings"
    gt_embeddings_dir = output_dir / "gt_embeddings"

    # Get embeddings for predicted property names
    pred_embeddings_files = list(pred_embeddings_dir.glob("*.parquet"))
    if not pred_embeddings_files:
        raise FileNotFoundError(f"No embeddings files found in {pred_embeddings_dir}")

    #
    # Load extracted properties
    #
    pred_properties_dir = output_dir / "unsupervised_llm_extraction"
    pred_properties_files = list(pred_properties_dir.glob("*.csv"))
    if not pred_properties_files:
        raise FileNotFoundError(f"No properties files found in {pred_properties_dir}")

    #
    # Load ground truth properties
    #
    if False:
        hf_dataset_name = pbench.DOMAIN2HF_DATASET_NAME["supercon"]
        logger.info(
            f"Loading dataset from HuggingFace: {hf_dataset_name} (split={args.split})"
        )
        dataset = load_dataset(hf_dataset_name, split=args.split, revision="v2.0.1")
    else:
        dataset = load_from_disk("out-0111/dataset")
    df_gt: pd.DataFrame = dataset.to_pandas()
    # make the refno column the index
    df_gt.set_index("refno", inplace=True)
    logger.info(f"Loaded {len(df_gt)} rows")

    #
    # Main Loop
    #
    pred_matches_dir = output_dir / "pred_matches"
    pred_matches_dir.mkdir(parents=True, exist_ok=True)
    gt_matches_dir = output_dir / "gt_matches"
    gt_matches_dir.mkdir(parents=True, exist_ok=True)

    # Initialize LLM
    llm = get_llm(args.server, model_name)

    # Create inference config
    inf_gen_config = InferenceGenerationConfig(
        max_output_tokens=args.max_output_tokens,
        output_format="json",
    )

    for i, pred_properties_file in enumerate(pred_properties_files):
        if args.file_no is not None and i != args.file_no - 1:
            continue
        df_pred = pd.read_csv(pred_properties_file)
        assert len(df_pred["refno"].unique()) == 1, "Expected only one refno per file"
        refno = df_pred["refno"].unique()[0]
        pred_matches_path = (
            pred_matches_dir / f"pred_matches_{refno}_{model_name}_k{top_k}.csv"
        )
        gt_matches_path = (
            gt_matches_dir / f"gt_matches_{refno}_{model_name}_k{top_k}.csv"
        )
        if pred_matches_path.exists() and gt_matches_path.exists() and not force:
            logger.info(f"Skipping refno {refno} because pred matches already exist")
            continue

        # Obtain the ground truth properties for this refno
        df_gt_refno = pd.DataFrame(df_gt.loc[refno]["properties"].tolist())
        # Obtain the embeddings for the predicted and ground truth properties
        df_pred_embeddings = pd.read_parquet(
            pred_embeddings_dir / f"{pred_properties_file.stem}.parquet"
        )
        df_gt_embeddings = pd.read_parquet(gt_embeddings_dir / f"{refno}.parquet")
        # Join df_pred and df_pred_embeddings on property_name
        df_pred = df_pred.merge(
            df_pred_embeddings[["property_name", "embedding"]],
            on="property_name",
            how="left",
        )
        # construct the context from location.evidence
        df_pred["context"] = df_pred["location.evidence"]
        df_gt_refno = df_gt_refno.merge(
            df_gt_embeddings[["property_name", "embedding"]],
            on="property_name",
            how="left",
        )
        # construct the context from conditions and value_unit
        df_gt_refno["context"] = df_gt_refno.apply(
            lambda row: construct_context(row), axis=1
        )

        if True:
            # -- Get top-k matches for each predicted property name (to compute precision) --
            # Compute the similarity matrix between the predicted and ground truth properties
            similarity_matrix = cosine_similarity(
                np.vstack(df_pred["embedding"].values),
                np.vstack(df_gt_embeddings["embedding"].values),
            )
            top_k_matches_indices = np.argsort(similarity_matrix, axis=1)[:, ::-1][
                :, :top_k
            ]
            pred_matches = []
            for i in tqdm(range(len(df_pred)), desc=f"Processing refno {refno}"):
                pred = df_pred.iloc[i].to_dict()
                # get the top-k matches from the ground truth embeddings
                top_k_matches = df_gt_embeddings.iloc[top_k_matches_indices[i]][
                    "property_name"
                ].tolist()
                df_gt_top_k = df_gt_refno[
                    df_gt_refno["property_name"].isin(top_k_matches)
                ]
                # Create async tasks for all unique matches
                tasks = OrderedDict()
                idx_to_task_id = {}
                for idx, gt in df_gt_top_k.iterrows():
                    name1 = pred["property_name"]
                    context1 = pred["location.evidence"]
                    name2 = gt["property_name"]
                    context2 = json.dumps(
                        {
                            k: v
                            for k, v in gt.items()
                            if k.startswith("conditions.") or k == "value_unit"
                        }
                    )
                    task_id = (name1, context1, name2, context2)
                    idx_to_task_id[idx] = task_id
                    if task_id not in tasks:
                        task = check_if_same_property(
                            llm,
                            inf_gen_config,
                            name1,
                            context1,
                            name2,
                            context2,
                        )
                        tasks[task_id] = task
                # Execute all tasks concurrently
                results = await asyncio.gather(*tasks.values())
                results = {
                    task_id: result for task_id, result in zip(tasks.keys(), results)
                }
                # Combine the results with the predicted and ground truth properties
                for idx, gt in df_gt_top_k.iterrows():
                    result = results[idx_to_task_id[idx]]
                    pred_matches.append(
                        {
                            **pred,
                            **result,
                            "gt_id": idx,  # use this to join the results with the ground truth properties
                        }
                    )
            # save pred_matches to csv
            df_pred_matches = pd.DataFrame(pred_matches)
            df_pred_matches = df_pred_matches.merge(
                df_gt_refno,
                left_on="gt_id",
                right_index=True,
                how="left",
                suffixes=("_pred", "_gt"),
            )
        else:
            df_pred_matches = await generate_property_name_matches(
                df_pred,
                df_gt_refno,
                llm,
                inf_gen_config,
                prompt_template,
                left_on=["property_name", "context"],
                right_on=["property_name", "context"],
            )
        df_pred_matches.to_csv(pred_matches_path, index=False)
        logger.info(f"Saved pred matches to {pred_matches_path}")

        if True:
            # -- Get top-k matches for each ground truth property name (to compute recall) --
            # Compute the similarity matrix between the ground truth and predicted properties
            similarity_matrix = cosine_similarity(
                np.vstack(df_gt_refno["embedding"].values),
                np.vstack(df_pred_embeddings["embedding"].values),
            )
            top_k_matches_indices = np.argsort(similarity_matrix, axis=1)[:, ::-1][
                :, :top_k
            ]
            gt_matches = []
            for i in tqdm(
                range(len(df_gt_refno)), desc=f"Processing refno {refno} (gt)"
            ):
                gt = df_gt_refno.iloc[i].to_dict()
                # get the top-k matches from the predicted embeddings
                top_k_matches = df_pred_embeddings.iloc[top_k_matches_indices[i]][
                    "property_name"
                ].tolist()
                df_pred_top_k = df_pred[df_pred["property_name"].isin(top_k_matches)]
                # Create async tasks for all unique matches
                tasks = OrderedDict()
                idx_to_task_id = {}
                for idx, pred in df_pred_top_k.iterrows():
                    # NOTE: use the same order of property 1 and property 2 as in the previous code block
                    # TODO (Albert): does this have an impact on the results (even though the classification task should be symmetric)?
                    name1 = pred["property_name"]
                    context1 = pred["location.evidence"]
                    name2 = gt["property_name"]
                    context2 = json.dumps(
                        {
                            k: v
                            for k, v in gt.items()
                            if k.startswith("conditions.") or k == "value_unit"
                        }
                    )
                    task_id = (name1, context1, name2, context2)
                    idx_to_task_id[idx] = task_id
                    if task_id not in tasks:
                        task = check_if_same_property(
                            llm,
                            inf_gen_config,
                            name1,
                            context1,
                            name2,
                            context2,
                        )
                        tasks[task_id] = task
                # Execute all tasks concurrently
                results = await asyncio.gather(*tasks.values())
                results = {
                    task_id: result for task_id, result in zip(tasks.keys(), results)
                }
                # Combine the results with the ground truth and predicted properties
                for idx, pred in df_pred_top_k.iterrows():
                    result = results[idx_to_task_id[idx]]
                    gt_matches.append(
                        {
                            **gt,
                            **result,
                            "pred_id": idx,  # use this to join the results with the predicted properties
                        }
                    )
                # save gt_matches to csv
                df_gt_matches = pd.DataFrame(gt_matches)
                df_gt_matches = df_gt_matches.merge(
                    df_pred,
                    left_on="pred_id",
                    right_index=True,
                    how="left",
                    suffixes=("_gt", "_pred"),
                )
        else:
            df_gt_matches = await generate_property_name_matches(
                df_gt_refno,
                df_pred,
                llm,
                inf_gen_config,
                prompt_template,
                left_on=["property_name", "context"],
                right_on=["property_name", "context"],
            )
        df_gt_matches.to_csv(gt_matches_path, index=False)
        logger.info(f"Saved gt matches to {gt_matches_path}")


if __name__ == "__main__":
    #
    # Parse arguments
    #
    parser = argparse.ArgumentParser(
        description="Verify alias candidates using Gemini LLM."
    )
    parser = pbench.add_base_args(parser)
    parser.add_argument(
        "--file_no",
        type=int,
        default=None,
        help="File number to process (1-indexed). If None, process all files",
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
