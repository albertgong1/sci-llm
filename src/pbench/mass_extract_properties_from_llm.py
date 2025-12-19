#!/usr/bin/env -S uv run --env-file=.env -- python
"""Mass extract properties from PDFs using unsupervised LLM extraction.

This script processes all PDF papers and extracts comprehensive property data
using an LLM with an unsupervised extraction prompt.

Usage:
```bash
# Process all PDFs
./src/pbench/mass_extract_properties_from_llm.py \
    --domain supercon \
    --server gemini \
    --model_name gemini-2.5-flash \
    -od out/
```

For this example, the results will be saved in `out/supercon/unsupervised_llm_extraction/*.csv`.
"""

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import llm_utils
from llm_utils.common import Conversation, File, LLMChatResponse, Message
import pbench

logger = logging.getLogger(__name__)


def load_prompt(prompt_path: Path) -> str:
    """Load the extraction prompt from a markdown file.

    Args:
        prompt_path: Path to the prompt markdown file

    Returns:
        The prompt text as a string

    """
    with open(prompt_path, "r") as f:
        return f.read()


def parse_json_response(response_text: str) -> dict | None:
    """Parse JSON from LLM response, handling markdown code blocks.

    Args:
        response_text: Raw response text from LLM

    Returns:
        Parsed JSON dict or None if parsing fails

    """
    print(f"Parsing JSON response: {response_text}")
    # Try direct JSON parsing first
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Try to extract from markdown code blocks
    json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find any JSON object in the text
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def map_conditions(json_conditions: dict) -> dict:
    """Map JSON conditions object to CSV condition columns.

    Args:
        json_conditions: Conditions dict from JSON

    Returns:
        Dict with CSV condition column names and values

    """
    csv_conditions = {
        "conditions.field_orientation": "",
        "conditions.current_direction": "",
        "conditions.field": "",
        "conditions.temperature": "",
        "conditions.field_range": "",
        "conditions.temperature_range": "",
    }

    if not json_conditions:
        return csv_conditions

    # Map orientation to field_orientation
    if orientation := json_conditions.get("orientation", ""):
        # if "orientation" in json_conditions and json_conditions["orientation"]:
        csv_conditions["conditions.field_orientation"] = orientation

    # Map field
    if field_val := json_conditions.get("field", ""):
        # if "field" in json_conditions and json_conditions["field"]:
        # field_val = json_conditions["field"]
        csv_conditions["conditions.field"] = field_val
        # Check if it's a range
        if any(indicator in str(field_val) for indicator in ["–", "-", "to", "~"]):
            csv_conditions["conditions.field_range"] = field_val
            csv_conditions["conditions.field"] = ""

    # Map temperature
    if temp_val := json_conditions.get("temperature", ""):
        # if "temperature" in json_conditions and json_conditions["temperature"]:
        csv_conditions["conditions.temperature"] = temp_val
        # Check if it's a range
        if any(
            indicator in str(temp_val) for indicator in ["–", "-", "to", "~", "<", ">"]
        ):
            csv_conditions["conditions.temperature_range"] = temp_val
            csv_conditions["conditions.temperature"] = ""

    return csv_conditions


def json_property_to_csv_row(prop: dict) -> pd.Series:
    """Convert a JSON property object to a CSV row as pandas Series.

    Args:
        prop: Property dict from JSON response

    Returns:
        pandas Series with all 27 CSV columns

    """
    # Map conditions
    conditions = map_conditions(prop.get("conditions", {}))

    # Get location info
    location = prop.get("location", {})

    # Build Series
    return pd.Series(
        {
            # "id": "", # NOTE: will be automatically assigned when writing to CSV
            # "refno": refno, # NOTE: will be automatically assigned when writing to CSV
            "material_or_system": prop.get("material_or_system", ""),
            "sample_label": prop.get("sample_label", ""),
            "property_name": prop.get("property_name", ""),
            "category": prop.get("category", ""),
            "value_string": prop.get("value_string", ""),
            "value_number": "",  # TODO: how to extract value number from value_string?
            "units": "",  # TODO: how to extract units from value_string?
            "method": prop.get("method", ""),
            "notes": prop.get("notes", ""),
            "location.page": location.get("page", ""),
            "location.section": location.get("section", ""),
            "location.source_type": location.get("source_type", ""),
            "location.evidence": location.get("evidence", ""),
            "location.figure_or_table": location.get("figure_or_table", ""),
            "conditions.field_orientation": conditions["conditions.field_orientation"],
            "conditions.current_direction": conditions["conditions.current_direction"],
            "conditions.field": conditions["conditions.field"],
            "conditions.temperature": conditions["conditions.temperature"],
            "conditions.field_range": conditions["conditions.field_range"],
            "conditions.temperature_range": conditions["conditions.temperature_range"],
            # NOTE: values below will be populated later by the validator app
            # "paper_pdf_path": "",
            # "validated": False,
            # "validator_name": "",
            # "validation_date": "",
            # "flagged": False,
        }
    )


def process_paper(
    paper_path: Path,
    prompt: str,
    llm: llm_utils.LLMChat,
    inf_gen_config: llm_utils.InferenceGenerationConfig,
) -> list[pd.Series]:
    """Process a single paper and extract all properties.

    Args:
        paper_path: Path to the PDF file
        prompt: Extraction prompt text
        llm: LLMChat instance
        inf_gen_config: InferenceGenerationConfig instance

    Returns:
        List of pandas Series (one per extracted property)

    """
    # Extract refno from filename
    refno = paper_path.stem
    paper_pdf_path = str(paper_path)

    # Create file object
    file = File(path=paper_path)

    # Build conversation
    # TODO: Maybe process 1 page of the PDF at a time
    conv = Conversation(
        messages=[
            Message(role="user", content=[file, prompt]),
        ]
    )

    try:
        # Get LLM response
        response: LLMChatResponse = llm.generate_response(conv, inf_gen_config)

        # Check for errors
        if response.error:
            logger.error(f"LLM error for {refno}: {response.error}")
            return []

        # Parse JSON from response
        json_data = parse_json_response(response.pred)
        if json_data is None:
            logger.error(f"Failed to parse JSON from response for {refno}")
            logger.debug(f"Response text: {response.pred[:500]}...")
            return []

        # Check for properties array
        if "properties" not in json_data:
            logger.error(f"No 'properties' key in JSON for {refno}")
            return []

        properties = json_data["properties"]
        if not isinstance(properties, list):
            logger.error(f"'properties' is not a list for {refno}")
            return []

        logger.info(f"Extracted {len(properties)} properties from {refno}")

        # Convert each property to CSV row (pandas Series)
        rows: list[pd.Series] = []
        for prop_idx, prop in enumerate(properties):
            try:
                row_series: pd.Series = json_property_to_csv_row(prop)

                # Assign metadata and values that will be populated later by the validator app
                row_series["id"] = f"prop_{prop_idx:03d}"
                row_series["refno"] = refno
                row_series["paper_pdf_path"] = paper_pdf_path
                row_series["validated"] = False
                row_series["validator_name"] = ""
                row_series["validation_date"] = ""
                row_series["flagged"] = False

                rows.append(row_series)
            except Exception as e:
                logger.warning(f"Failed to convert property {prop} from {refno}: {e}")
                continue

        return rows

    except Exception as e:
        logger.error(f"Error processing {refno}: {e}")
        return []
    finally:
        llm.delete_file(file)


def main(args: argparse.Namespace) -> None:
    """Main function to extract properties from all PDFs."""
    # Load extraction prompt
    prompt_path = (
        pbench.ASSETS_DIR / args.domain / args.unsupervised_extraction_prompt_filename
    )
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    logger.info(f"Loading prompt from {prompt_path}")
    prompt = load_prompt(prompt_path)

    # Get list of PDFs
    paper_dir = args.data_dir / args.domain / "Paper_DB"
    if not paper_dir.exists():
        raise FileNotFoundError(f"Paper directory not found: {paper_dir}")

    pdf_files = sorted(paper_dir.glob("*.pdf"))
    num_pdfs = len(pdf_files)
    logger.info(f"Found {num_pdfs} PDF files in {paper_dir}")

    if num_pdfs == 0:
        logger.warning("No PDF files found to process")
        return

    # Initialize LLM
    llm = llm_utils.get_llm(args.server, args.model_name)

    # Create inference config
    inf_gen_config = llm_utils.InferenceGenerationConfig(
        max_output_tokens=args.max_output_tokens,
    )

    # Setup output directory
    preds_dir = args.output_dir / args.domain / "unsupervised_llm_extraction"
    preds_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize model name for filename
    model_name_safe = args.model_name.replace("/", "--")

    # Process each PDF
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        refno = pdf_path.stem

        # Construct output path
        output_filename = (
            f"extracted_properties__model={model_name_safe}__refno={refno}.csv"
        )
        output_path = preds_dir / output_filename

        # Skip if output file already exists and --force is not set
        if output_path.exists() and not args.force:
            logger.info(
                f"Skipping {refno} as {output_path} already exists. Use --force to overwrite."
            )
            continue

        # Process the paper
        rows = process_paper(pdf_path, prompt, llm, inf_gen_config)

        # Save to CSV
        if len(rows) > 0:
            # Create DataFrame from list of Series
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(rows)} properties from {refno} to {output_path}")
        else:
            logger.warning(f"No properties extracted from {refno}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mass extract properties from PDF files using unsupervised LLM extraction"
    )
    parser = pbench.add_base_args(parser)
    parser.add_argument(
        "--unsupervised_extraction_prompt_filename",
        type=str,
        default="unsupervised_extraction_prompt.md",
        help="Filename of the unsupervised extraction prompt (default: unsupervised_extraction_prompt.md)",
    )

    # LLM generation arguments
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=65536,
        help="Maximum number of output tokens for LLM response (default: 65536)",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite existing output files",
    )

    args = parser.parse_args()
    pbench.setup_logging(args.log_level)

    assert args.domain == "supercon", "Only supercon domain is supported for now"

    main(args)
