#!/usr/bin/env -S uv run --env-file=.env -- python
"""Extract properties from PDF files using LLMs.

Usage:
```bash
./src/pbench_eval/extract_properties.py \
    --domain supercon \
    --task tc \
    --server gemini \
    --model_name gemini-2.5-flash \
    -od out/
```

For this example, the results will be saved in `out/supercon/preds/*.json`.
"""

import argparse
import re
import logging
import math
from pathlib import Path
import json

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

import llm_utils
from llm_utils.common import Conversation, File, LLMChatResponse, Message
from pbench_eval import constants, utils

logger = logging.getLogger(__name__)

# Prompt template for property extraction
# Use as PROMPT.format(material=material, property_name=property_name, definition=definition)
PROMPT = """
You are given the following paper.

You will be provided a question about extracting a material property from the paper. Your task is to provide an answer according to these instructions:
- Identify the material by its chemical formula or name
- Locate the property name, value, and unit in the text
- Ensure the property value corresponds to the correct material
- Extract exact values as they appear in the paper
- The output must be in the format: <property>PROPERTY</property><value>PROPERTY_VALUE</value><unit>PROPERTY_UNIT</unit>
- If a property is not found, respond with: <property>PROPERTY</property><value>NOT_FOUND</value><unit>N/A</unit>
- DO NOT include any additional information beyond the requested format.

Question: What is the {property_name} recommended for {material}? Here, "{property_name}" is defined as "{definition}".
Answer:
"""


def extract_property_from_response(response_text: str) -> tuple[str, str, str]:
    """Parse the Gemini response to extract property, value, and unit.

    Args:
        response_text: Response text from Gemini API

    Returns:
        Tuple of (property, value, unit)

    """
    # Extract property
    property_match = re.search(r"<property>(.*?)</property>", response_text, re.DOTALL)
    property_value = property_match.group(1).strip() if property_match else "NOT_FOUND"

    # Extract value
    value_match = re.search(r"<value>(.*?)</value>", response_text, re.DOTALL)
    value = value_match.group(1).strip() if value_match else "NOT_FOUND"

    # Extract unit
    unit_match = re.search(r"<unit>(.*?)</unit>", response_text, re.DOTALL)
    unit = unit_match.group(1).strip() if unit_match else "N/A"

    return property_value, value, unit


def process_paper(
    paper_path: Path,
    df_paper: pd.DataFrame,
    llm: llm_utils.LLMChat,
    inf_gen_config: llm_utils.InferenceGenerationConfig,
) -> list[dict]:
    """Process a single paper and extract properties for all rows.

    Args:
        paper_path: Path to the PDF file
        df_paper: DataFrame containing all rows for this paper
        llm: LLMChat instance
        inf_gen_config: InferenceGenerationConfig instance
        batch_results: List to store results (modified in-place)

    Returns:
        List of results (true values, predicted values, response, etc.)
        for each row in the paper

    """
    # Initialize file object so that the upload is cached on the first call to the LLM
    file = File(path=paper_path)

    paper_results = []

    # Process each row for this paper
    for _, row in df_paper.iterrows():
        refno = row["refno"]
        material = row["material"]
        definition = row["definition"]
        property_name = row["property_name"]
        true_value = row["property_value"]
        true_unit = row["property_unit"]

        # Format prompt with property name and definition
        formatted_prompt = PROMPT.format(
            material=material, property_name=property_name, definition=definition
        )

        # Create a conversation with PDF file attachment and prompt
        conv = Conversation(
            messages=[
                Message(role="user", content=[file, formatted_prompt]),
            ]
        )

        try:
            response: LLMChatResponse = llm.generate_response(conv, inf_gen_config)
        except Exception as e:
            response = LLMChatResponse(pred="", usage={}, error=str(e))

        # Parse and store extracted values
        _, pred_value, pred_unit = extract_property_from_response(response.pred)

        # TODO: Make this a pydantic model so that score_task.py can use it
        result = {
            "refno": refno,
            "material": material,
            "property_name": property_name,
            "true": {
                "value": true_value,
                "unit": true_unit,
            },
            "pred": {
                "value": pred_value,
                "unit": pred_unit,
            },
            "response": response.model_dump(),
            "inf_gen_config": inf_gen_config.model_dump(),
        }
        paper_results.append(result)

    # Remove the file from the LLM server
    llm.delete_file(file)

    return paper_results


def main(args: argparse.Namespace) -> None:
    """Main function to extract properties from PDF files using LLMs."""
    # Load dataset from HuggingFace
    hf_dataset_name = constants.DOMAIN2HF_DATASET_NAME[args.domain]
    logger.info(
        f"Loading dataset from HuggingFace: {hf_dataset_name} (task={args.task})"
    )
    dataset = load_dataset(hf_dataset_name, name=args.task, split=args.split)
    df: pd.DataFrame = dataset.to_pandas()
    logger.info(f"Loaded {len(df)} rows")

    # Validate batch_number if specified
    num_rows = len(df)
    bs = args.batch_size
    if args.batch_number is not None:
        assert args.batch_number >= 1, "Batch number must be >= 1"
        assert args.batch_number <= math.ceil(num_rows / bs), (
            f"Batch number must be <= {math.ceil(num_rows / bs)}"
        )

    # Setup paths
    paper_dir = args.data_dir / args.domain / "Paper_DB"
    preds_dir = args.output_dir / args.domain / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)

    llm = llm_utils.get_llm(args.server, args.model_name)

    # TODO: Load inference generation config from yaml file
    inf_gen_config = llm_utils.InferenceGenerationConfig()

    # Process batches
    for bn in range(1, math.ceil(num_rows / bs) + 1):
        # Skip if specific batch_number is requested and this isn't it
        if (args.batch_number is not None) and (bn != args.batch_number):
            continue

        # Construct preds path
        preds_path = preds_dir / (
            f"task={args.task}__split={args.split}__model={args.model_name.replace('/', '--')}__bs={bs}__bn={bn}.json"
        )

        # Skip if output file already exists and --force is not set
        if preds_path.exists() and not args.force:
            logger.info(
                f"Skipping batch {bn} as {preds_path} already exists. Use --force to overwrite."
            )
            continue

        # Get batch
        batch_start_idx = (bn - 1) * bs
        batch_end_idx = batch_start_idx + bs
        logger.info(
            f"Processing batch {bn}: rows [{batch_start_idx}, {batch_end_idx}) out of {num_rows}"
        )
        batch_df = df.iloc[batch_start_idx:batch_end_idx]

        # Group batch by refno and process each paper
        batch_grouped = batch_df.groupby("refno")
        batch_n_papers = len(batch_grouped)
        logger.info(f"Batch contains {batch_n_papers} unique papers")

        batch_results = []

        # Loop over each paper and process it
        for refno, df_paper in tqdm(
            batch_grouped, total=batch_n_papers, desc=f"Processing papers in batch {bn}"
        ):
            paper_path = paper_dir / f"{refno}.pdf"
            if not paper_path.exists():
                logger.warning(f"PDF not found at {paper_path}, skipping...")
                continue

            paper_results: list[dict] = process_paper(
                paper_path, df_paper, llm, inf_gen_config
            )
            batch_results.extend(paper_results)

        if len(batch_results) > 0:
            with open(preds_path, "w") as f:
                json.dump(batch_results, f, indent=4)
            logger.info(f"Batch {bn} results saved to {preds_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract properties from PDF files using LLMs"
    )
    # Task-specific arguments
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        choices=constants.SUPPORTED_DOMAINS,
        help="Domain to perform extraction on",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="HuggingFace dataset configuration name, depending on the domain (e.g., 'Tc', 'gap')",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of the dataset to use",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default="data/",
        help="Directory containing Paper_DB folder with PDFs",
    )
    parser.add_argument(
        "--output_dir",
        "-od",
        type=Path,
        default="out/",
        help="Output directory for results",
    )

    # LLM arguments
    parser.add_argument(
        "--server",
        type=str,
        default="gemini",
        choices=llm_utils.SUPPORTED_SERVERS,
        help="LLM server to use",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemini-2.5-flash",
        help="Name of the LLM to use for extraction",
    )

    # Batch processing arguments
    parser.add_argument(
        "--batch_size",
        "-bs",
        type=int,
        default=10,
        help="Number of rows to process per batch",
    )
    parser.add_argument(
        "--batch_number",
        "-bn",
        type=int,
        default=None,
        help="Specific batch number to process (1-indexed). If None, process all batches",
    )
    parser.add_argument(
        "--force", "-f", action="store_true", help="Overwrite existing output files"
    )
    parser.add_argument(
        "--log_level", type=int, default=logging.INFO, help="Logging level"
    )

    args = parser.parse_args()
    utils.setup_logging(args.log_level)

    assert args.domain == "supercon", "Only supercon domain is supported for now"

    main(args)
