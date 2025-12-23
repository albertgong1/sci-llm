"""Extract superconductor properties from PDF files using Gemini 2.5 Flash API.

This script loads datasets from HuggingFace (kilian-group/supercon-mini),
processes them in batches, extracts information from PDF papers using Google's
Gemini API, and stores the results as JSON files.
"""

import argparse
import json
import math
import os
import re
import time
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from google import genai
from google.genai import types as genai_types
from tqdm import tqdm

# Configure Gemini API
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

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


def upload_pdf_to_gemini(pdf_path: Path, client: genai.Client) -> genai_types.File:
    """Upload a PDF file to Gemini API for processing.

    Args:
        pdf_path: Path to the PDF file
        client: Gemini Client instance

    Returns:
        File object from Gemini API

    """
    print(f"Uploading {pdf_path}...")
    google_pdf_file = client.files.upload(file=pdf_path)
    return google_pdf_file


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
    client: genai.Client,
    model_name: str,
    batch_results: dict,
) -> None:
    """Process a single paper and extract properties for all rows.

    Args:
        paper_path: Path to the PDF file
        df_paper: DataFrame containing all rows for this paper
        client: Gemini Client instance
        model_name: Name of the Gemini model to use
        batch_results: Dictionary to store results (modified in-place)

    """
    # Upload PDF to Gemini
    pdf_file = upload_pdf_to_gemini(paper_path, client)
    if pdf_file is None:
        raise Exception(f"Failed to upload PDF to Gemini: {paper_path}")
    time.sleep(1)  # wait for file to be processed

    # Process each row for this paper
    for idx, row in df_paper.iterrows():
        material = row["material"]
        property_name = row["property_name"]
        definition = row["definition"]
        refno = row["refno"]
        property_value = row["property_value"]
        property_unit = row["property_unit"]

        # Format prompt with property name and definition
        formatted_prompt = PROMPT.format(
            material=material, property_name=property_name, definition=definition
        )

        try:
            # Generate response from Gemini
            print(f"Processing row {idx}: {material} {property_name}")
            response = client.models.generate_content(
                model=model_name, contents=[pdf_file, formatted_prompt]
            )
            if response and response.text:
                response_text = response.text
            else:
                response_text = "ERROR: No response from Gemini"

            # Parse and store extracted values
            _, pred_value, pred_unit = extract_property_from_response(response_text)

            print(f"  Extracted: {material} {property_name} = {pred_value} {pred_unit}")

        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            response_text = f"ERROR: {str(e)}"
            pred_value = f"ERROR: {str(e)}"
            pred_unit = f"ERROR: {str(e)}"

        # Store result in batch_results dictionary
        batch_results[idx] = {
            "refno": refno,
            "material": material,
            "property_name": property_name,
            "property_value": property_value,
            "property_unit": property_unit,
            "pred_value": pred_value,
            "pred_unit": pred_unit,
            "response": response_text,
        }

        # Rate limiting - add delay between requests
        time.sleep(0.5)

    # Remove the file from the client
    client.files.delete(name=pdf_file.name)  # type: ignore
    print(f"Deleted uploaded file: {pdf_file.name}")  # type: ignore


def main(args: argparse.Namespace) -> None:
    """Generate predictions using Gemini API for the given task.

    Args:
        args: command line arguments

    Returns:
        None

    """
    # Load dataset from HuggingFace
    print(
        f"Loading dataset from HuggingFace: kilian-group/supercon-mini (task={args.task})"
    )
    dataset = load_dataset("kilian-group/supercon-mini", name=args.task, split="test")
    df = dataset.to_pandas()
    print(f"Loaded {len(df)} rows")

    # Validate batch_number if specified
    num_rows = len(df)
    batch_size = args.batch_size
    if args.batch_number is not None:
        assert args.batch_number >= 1, "Batch number must be >= 1"
        assert args.batch_number <= math.ceil(num_rows / batch_size), (
            f"Batch number must be <= {math.ceil(num_rows / batch_size)}"
        )

    # Setup paths
    paper_dir = Path(args.data_dir) / "Paper_DB"
    output_preds_dir = Path(args.output_dir) / "preds"
    output_preds_dir.mkdir(parents=True, exist_ok=True)

    # Process batches
    for batch_number in range(1, math.ceil(num_rows / batch_size) + 1):
        # Skip if specific batch_number is requested and this isn't it
        if (args.batch_number is not None) and (batch_number != args.batch_number):
            continue

        # Construct output path
        pred_path = (
            output_preds_dir
            / f"task={args.task}__model={args.model_name.replace('/', '--')}__bs={batch_size}__bn={batch_number}.json"
        )

        # Skip if output file already exists and --force is not set
        if pred_path.exists() and not args.force:
            print(
                f"Skipping batch {batch_number} as {pred_path} already exists. Use --force to overwrite."
            )
            continue

        # Get batch
        batch_start_idx = (batch_number - 1) * batch_size
        batch_end_idx = batch_start_idx + batch_size
        print(
            f"\nProcessing batch {batch_number}: rows [{batch_start_idx}, {batch_end_idx}) out of {num_rows}"
        )
        batch_df = df.iloc[batch_start_idx:batch_end_idx]

        # Group batch by refno and process each paper
        batch_grouped = batch_df.groupby("refno")
        total_papers = len(batch_grouped)
        print(f"  Batch contains {total_papers} unique papers")

        # Dictionary to store batch results
        batch_results = {}

        # Loop over each paper and process it
        for refno, df_paper in tqdm(
            batch_grouped,
            total=total_papers,
            desc=f"Processing papers in batch {batch_number}",
        ):
            paper_path = paper_dir / f"{refno}.pdf"

            if not paper_path.exists():
                print(f"Warning: PDF not found at {paper_path}, skipping...")
                continue

            process_paper(
                paper_path, df_paper, args.client, args.model_name, batch_results
            )

        # Add metadata to each result
        for idx in batch_results:
            batch_results[idx]["metadata"] = {
                "batch_size": batch_size,
                "batch_number": batch_number,
                "model": args.model_name,
                "task": args.task,
            }

        # Save batch results to JSON
        with open(pred_path, "w") as f:
            json.dump(batch_results, f, indent=4)
        print(f"Batch {batch_number} results saved to {pred_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract superconductor properties from PDFs using Gemini API"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="HuggingFace dataset configuration name (e.g., 'Tc', 'gap')",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing Paper_DB folder with PDFs",
    )
    parser.add_argument(
        "--output_dir", type=str, default="out", help="Output directory for results"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemini-2.5-flash",
        help="Name of the Gemini model to use",
    )
    parser.add_argument(
        "--batch_size", type=int, default=10, help="Number of rows to process per batch"
    )
    parser.add_argument(
        "--batch_number",
        type=int,
        default=None,
        help="Specific batch number to process (1-indexed). If None, process all batches",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing output files"
    )
    args = parser.parse_args()
    args.client = genai.Client(api_key=GOOGLE_API_KEY)
    main(args)
