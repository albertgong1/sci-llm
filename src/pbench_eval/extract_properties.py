#!/usr/bin/env -S uv run --env-file=.env -- python
"""Extract properties from PDF files using LLMs.
Currently supported:
- Properties from supercon
- gemini LLM server

This script processes a dataset of superconductor properties, extracts information
from PDF papers using Google's Gemini API, and stores the results.

Usage:
```bash
./src/pbench_eval/extract_properties.py \
    --task supercon \
    --server gemini \
    --model_name gemini-2.5-flash \
    -od out/
```

For this example, the results will be saved in `out/supercon/preds__model=gemini-2.5-flash.csv`.
"""

import argparse
import re
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import llm_utils
from llm_utils.common import Conversation, File, LLMChatResponse, Message
from pbench_eval import constants

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
) -> pd.DataFrame:
    """Process a single paper and extract properties for all rows.

    Args:
        paper_path: Path to the PDF file
        df_paper: DataFrame containing all rows for this paper
        llm: LLMChat instance

    Returns:
        DataFrame with added LLM response columns

    """
    # Initialize file object so that the upload is cached on the first call to the LLM
    file = File(path=paper_path)

    # Initialize result columns
    df_paper[f"{llm.model_name}"] = ""
    df_paper[f"{llm.model_name}-pred-value"] = ""
    df_paper[f"{llm.model_name}-pred-unit"] = ""

    inf_gen_config = llm_utils.InferenceGenerationConfig()

    # Process each row for this paper
    for idx, row in df_paper.iterrows():
        material = row["material"]
        property_name = row["property_name"]
        definition = row["definition"]

        print(f"Processing row {idx}: {material} {property_name}")

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
            # Generate response
            response: LLMChatResponse = llm.generate_response(conv, inf_gen_config)

            if response.error:
                response_text = f"<error>{response.error}</error>"
            else:
                response_text = response.pred

            # TODO: Save the full LLMChatResponse in a json file
            # Store full response
            df_paper.at[idx, f"{llm.model_name}"] = response_text

            # Parse and store extracted values
            _, value, unit = extract_property_from_response(response_text)
            df_paper.at[idx, f"{llm.model_name}-pred-value"] = value
            df_paper.at[idx, f"{llm.model_name}-pred-unit"] = unit

            print(f"  Extracted: {material} {property_name} = {value} {unit}")

        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            raise e
            df_paper.at[idx, f"{llm.model_name}"] = f"<error>{str(e)}</error>"
            df_paper.at[idx, f"{llm.model_name}-pred-value"] = (
                f"<error>{str(e)}</error>"
            )
            df_paper.at[idx, f"{llm.model_name}-pred-unit"] = f"<error>{str(e)}</error>"

        # Rate limiting - add delay between requests
        time.sleep(0.5)

    llm.delete_file(file)

    return df_paper


def main(args: argparse.Namespace) -> None:
    """Main function to extract properties from PDF files using LLMs."""
    output_csv_fname = f"preds__model={args.model_name}.csv"
    output_csv_path = args.output_task_dir / output_csv_fname

    # Load dataset
    dataset_csv_path = constants.ASSETS_DIR / args.task / "dataset.csv"
    print(f"Loading dataset from {dataset_csv_path}")
    df = pd.read_csv(dataset_csv_path)
    print(f"Loaded {len(df)} rows")

    # Group by paper and process each
    grouped = df.groupby("paper")
    total_papers = len(grouped)

    results = []

    # Loop over each paper and process it
    for paper, df_paper in tqdm(grouped, total=total_papers, desc="Processing papers"):
        paper_path = constants.ASSETS_DIR / args.task / str(paper)

        df_processed = process_paper(paper_path, df_paper, args.llm)
        results.append(df_processed)

    # Save results as a pandas DataFrame
    results_df = pd.concat(results, ignore_index=True)
    results_df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Task-specific arguments
    parser.add_argument(
        "--task",
        type=str,
        default="supercon",
        choices=constants.SUPPORTED_TASKS,
        help="Task to perform",
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
        help="Name of the Gemini model to use",
    )

    parser.add_argument(
        "--output_dir",
        "-od",
        type=str,
        default="out",
        help="Directory to save the results csv file",
    )
    args = parser.parse_args()

    assert args.task == "supercon", "Only supercon task is supported for now"

    args.output_dir = Path(args.output_dir)
    args.output_task_dir: Path = args.output_dir / args.task
    args.output_task_dir.mkdir(parents=True, exist_ok=True)

    args.llm = llm_utils.get_llm(args.server, args.model_name)

    main(args)
