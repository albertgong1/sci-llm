"""
Extract superconductor properties from PDF files using Gemini 2.5 Flash API.

This script processes a dataset of superconductor properties, extracts information
from PDF papers using Google's Gemini API, and stores the results.
"""

import argparse
import os
import re
import time
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import types as genai_types
from tqdm import tqdm

# Configure Gemini API
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

# Prompt template for property extraction
# Use as PROMPT.format(material=material, property_name=property_name, definition=definition)
PROMPT = f"""
You are given the following paper.

You will be provided a question about extracting a material property from the paper. Your task is to provide an answer according to these instructions:
- Identify the material by its chemical formula or name
- Locate the property name, value, and unit in the text
- Ensure the property value corresponds to the correct material
- Extract exact values as they appear in the paper
- The output must be in the format: <property>PROPERTY</property><value>PROPERTY_VALUE</value><unit>PROPERTY_UNIT</unit>
- If a property is not found, respond with: <property>PROPERTY</property><value>NOT_FOUND</value><unit>N/A</unit>
- DO NOT include any additional information beyond the requested format.

Question: What is the {{property_name}} recommended for {{material}}? Here, "{{property_name}}" is defined as "{{definition}}".
Answer:
"""


def upload_pdf_to_gemini(pdf_path: Path, client: genai.Client) -> genai_types.File:
    """
    Upload a PDF file to Gemini API for processing.

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
    """
    Parse the Gemini response to extract property, value, and unit.

    Args:
        response_text: Response text from Gemini API

    Returns:
        Tuple of (property, value, unit)
    """
    # Extract property
    property_match = re.search(r'<property>(.*?)</property>', response_text, re.DOTALL)
    property_value = property_match.group(1).strip() if property_match else "NOT_FOUND"

    # Extract value
    value_match = re.search(r'<value>(.*?)</value>', response_text, re.DOTALL)
    value = value_match.group(1).strip() if value_match else "NOT_FOUND"

    # Extract unit
    unit_match = re.search(r'<unit>(.*?)</unit>', response_text, re.DOTALL)
    unit = unit_match.group(1).strip() if unit_match else "N/A"

    return property_value, value, unit


def process_paper(paper_path: Path, df_paper: pd.DataFrame, client: genai.Client) -> pd.DataFrame:
    """
    Process a single paper and extract properties for all rows.

    Args:
        paper_path: Path to the PDF file
        df_paper: DataFrame containing all rows for this paper
        client: Gemini Client instance

    Returns:
        DataFrame with added Gemini response columns
    """
    # Upload PDF to Gemini
    pdf_file = upload_pdf_to_gemini(paper_path, client)
    if pdf_file is None:
        raise Exception(f"Failed to upload PDF to Gemini: {paper_path}")
    time.sleep(2) # wait for file to be processed

    # Initialize result columns
    df_paper[f'{args.model_name}'] = ""
    df_paper[f'{args.model_name}-pred-value'] = ""
    df_paper[f'{args.model_name}-pred-unit'] = ""

    # Process each row for this paper
    for idx, row in df_paper.iterrows():
        material = row['material']
        property_name = row['property_name']
        definition = row['definition']

        # Format prompt with property name and definition
        formatted_prompt = PROMPT.format(
            material=material,
            property_name=property_name,
            definition=definition
        )

        try:
            # Generate response from Gemini
            print(f"Processing row {idx}: {material} {property_name}")
            response = client.models.generate_content(
                model=args.model_name,
                contents=[pdf_file, formatted_prompt]
            )
            if response and response.text:
                response_text = response.text
            else:
                response_text = "ERROR: No response from Gemini"

            # Store full response
            df_paper.at[idx, f'{args.model_name}'] = response_text

            # Parse and store extracted values
            _, value, unit = extract_property_from_response(response_text)
            df_paper.at[idx, f'{args.model_name}-pred-value'] = value
            df_paper.at[idx, f'{args.model_name}-pred-unit'] = unit

            print(f"  Extracted: {material} {property_name} = {value} {unit}")

        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            df_paper.at[idx, f'{args.model_name}'] = f"ERROR: {str(e)}"
            df_paper.at[idx, f'{args.model_name}-pred-value'] = f"ERROR: {str(e)}"
            df_paper.at[idx, f'{args.model_name}-pred-unit'] = f"ERROR: {str(e)}"

        # Rate limiting - add delay between requests
        time.sleep(1)

    # Remove the file from the client
    client.files.delete(name=pdf_file.name) # type: ignore
    print(f"Deleted uploaded file: {pdf_file.name}") # type: ignore

    return df_paper


def main(args):
    # Paths
    base_dir = Path("assets")
    csv_path = base_dir / "dataset.csv"
    pdf_dir = base_dir # note that dataset.csv paper column contains Paper_DB/..., so we don't need to add it here

    save_results_csv_dir = Path(args.save_results_csv_dir)
    save_results_csv_dir.mkdir(parents=True, exist_ok=True)
    save_results_csv_path = save_results_csv_dir / f"preds__{args.model_name}.csv"

    # Load dataset
    print(f"Loading dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")

    # Group by paper and process each
    grouped = df.groupby("paper")
    total_papers = len(grouped)
    print(grouped)

    results = []

    # Loop over each paper and process it
    for paper, df_paper in tqdm(grouped, total=total_papers, desc="Processing papers"):
        paper_path = pdf_dir / Path(str(paper))

        df_processed = process_paper(paper_path, df_paper, args.client)
        results.append(df_processed)

    # Save results as a pandas DataFrame
    results_df = pd.concat(results, ignore_index=True)
    results_df.to_csv(save_results_csv_path, index=False)
    print(f"Results saved to {save_results_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_results_csv_dir", type=str, default="results", help="Directory to save the results csv file")
    parser.add_argument("--model_name", type=str, default="gemini-2.5-flash", help="Name of the Gemini model to use")
    args = parser.parse_args()
    args.client = genai.Client(api_key=GOOGLE_API_KEY)
    main(args)
