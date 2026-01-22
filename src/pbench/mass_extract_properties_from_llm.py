#!/usr/bin/env -S uv run --env-file=.env -- python
"""Mass extract properties from PDFs using unsupervised LLM extraction.

This script processes all PDF papers and extracts comprehensive property data
using an LLM with an unsupervised extraction prompt.

The script expects to be run from a directory containing:
- prompts/unsupervised_extraction_prompt.md (or custom prompt path)
- data/Paper_DB/ (directory containing the PDF files to process)

Usage:
```bash
# From examples/biosurfactants-extraction/ directory
uv run pbench-extract \
    --server gemini \
    --model_name gemini-3-pro-preview \
    -od out/ \
    -pp prompts/unsupervised_extraction_prompt.md \
    -dd data/ \
    -log_level INFO
```

For this example, the results will be saved in `out/unsupervised_llm_extraction/*.csv`.
"""

import argparse
import asyncio
import json
import logging
import re
from pathlib import Path

import pandas as pd
from tqdm.asyncio import tqdm
import fitz  # PyMuPDF

import llm_utils
from llm_utils.common import Conversation, File, LLMChatResponse, Message
import pbench

logger = logging.getLogger(__name__)

# Maximum number of conditions to store in CSV
MAX_CONDITIONS = 10


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


def json_property_to_csv_row(prop: dict) -> pd.Series:
    """Convert a JSON property object to a CSV row as pandas Series.

    Args:
        prop: Property dict from JSON response

    Returns:
        pandas Series with CSV columns

    """
    # Get conditions and flatten them
    conditions = prop.get("conditions") or prop.get("critical_matrix") or {}

    # Create a dict to hold the flattened condition columns
    condition_cols = {}
    for i in range(1, MAX_CONDITIONS + 1):
        condition_cols[f"condition{i}_name"] = ""
        condition_cols[f"condition{i}_value"] = ""

    # Fill in the conditions that exist
    if isinstance(conditions, dict):
        for idx, (cond_name, cond_value) in enumerate(conditions.items(), start=1):
            if idx <= MAX_CONDITIONS:
                condition_cols[f"condition{idx}_name"] = str(cond_name)
                condition_cols[f"condition{idx}_value"] = str(cond_value)

    # Get location info
    location = prop.get("location", {})

    # Build Series
    row_data = {
        # "id": "", # NOTE: will be automatically assigned when writing to CSV
        # "refno": refno, # NOTE: will be automatically assigned when writing to CSV
        "material_or_system": prop.get("material_or_system", ""),
        "sample_label": prop.get("sample_label", ""),
        "property_name": prop.get("property_name", ""),
        "category": prop.get("category", ""),
        "value_string": prop.get("value_string", ""),
        "value_number": "",
        "units": "",
        "method": prop.get("method", ""),
        "notes": prop.get("notes", ""),
        "location.page": location.get("page", ""),
        "location.section": location.get("section", ""),
        "location.source_type": location.get("source_type", ""),
        "location.evidence": location.get("evidence", ""),
        "location.figure_or_table": location.get("figure_or_table", ""),
    }

    # Add the flattened condition columns
    row_data.update(condition_cols)

    # NOTE: values below will be populated later by the validator app
    # "paper_pdf_path": "",
    # "validated": False,
    # "validator_name": "",
    # "validation_date": "",
    # "flagged": False,

    return pd.Series(row_data)


async def process_single_page(
    page_num: int,
    file: File,
    prompt: str,
    refno: str,
    llm: llm_utils.LLMChat,
    inf_gen_config: llm_utils.InferenceGenerationConfig,
) -> list[dict]:
    """Process a single page and extract properties.

    Args:
        page_num: Page number to process
        file: File object for the paper
        prompt: Extraction prompt text
        refno: Reference number of the paper
        llm: LLMChat instance
        inf_gen_config: InferenceGenerationConfig instance

    Returns:
        List of property dicts (with page_num included for sorting later)

    """
    # Build conversation
    page_prompt = f"""
\n\n
------FINAL INSTRUCTIONS------
Now extract all properties in the research article that are on page number {page_num}.
It is important to ONLY extract properties that are on page number {page_num}.
DO NOT extract properties that are on other pages.
--------------------------------
    """
    conv = Conversation(
        messages=[
            Message(role="user", content=[file, prompt, page_prompt]),
        ]
    )

    try:
        # Get LLM response
        response: LLMChatResponse = await llm.generate_response_async(
            conv, inf_gen_config
        )

        # Check for errors
        if response.error:
            logger.error(f"LLM error for {refno} page {page_num}: {response.error}")
            return []

        # Parse JSON from response
        json_data = parse_json_response(response.pred)
        if json_data is None:
            logger.warning(f"Failed to parse JSON from {refno} page {page_num}")
            return []

        # Check for properties array
        if "properties" not in json_data:
            logger.warning(f"No 'properties' key in JSON for {refno} page {page_num}")
            return []

        properties = json_data["properties"]
        if not isinstance(properties, list):
            logger.warning(f"'properties' is not a list for {refno} page {page_num}")
            return []

        # Add page_num to each property for later sorting
        for prop in properties:
            prop["_page_num"] = page_num

        return properties

    except Exception as e:
        logger.error(f"Error processing {refno} page {page_num}: {e}")
        return []


async def process_paper(
    paper_path: Path,
    prompt: str,
    llm: llm_utils.LLMChat,
    inf_gen_config: llm_utils.InferenceGenerationConfig,
    model_name: str,
) -> list[pd.Series]:
    """Process a single paper by prompting the LLM one page at a time to extract properties.

    Args:
        paper_path: Path to the PDF file
        prompt: Extraction prompt text
        llm: LLMChat instance
        inf_gen_config: InferenceGenerationConfig instance
        model_name: Name of the LLM model used for extraction

    Returns:
        List of pandas Series (one per extracted property)

    """
    # Extract refno from filename and create file object for the paper
    refno = paper_path.stem
    file = File(path=paper_path)

    # Get number of pages in the paper, we will upload the entire paper to the LLM
    # server at once, but prompt one page at a time.
    try:
        doc = fitz.open(paper_path)
        num_pages = len(doc)
        doc.close()
    except Exception as e:
        logger.error(f"Failed to get number of pages from {paper_path}: {e}")
        return []

    # Create tasks for all pages
    tasks = [
        process_single_page(page_num, file, prompt, refno, llm, inf_gen_config)
        for page_num in range(1, num_pages + 1)
    ]

    # Process all pages concurrently with progress bar
    total_properties = 0
    pbar = tqdm(total=len(tasks), desc=f"Processing {refno} (0 props)")

    page_results = []
    for coro in asyncio.as_completed(tasks):
        result = await coro
        page_results.append(result)
        total_properties += len(result)
        pbar.set_description(f"Processing {refno} ({total_properties} props)")
        pbar.update(1)

    pbar.close()

    # Flatten results and convert to CSV rows
    all_rows: list[pd.Series] = []
    property_counter = 0

    for properties in page_results:
        for prop in properties:
            try:
                # Remove the temporary page_num field before converting
                prop.pop("_page_num", None)
                row_series: pd.Series = json_property_to_csv_row(prop)

                # Assign metadata
                row_series["id"] = f"prop_{property_counter:03d}"
                row_series["refno"] = refno
                row_series["paper_pdf_path"] = str(paper_path)
                row_series["agent"] = "zeroshot"
                row_series["model"] = model_name
                row_series["validated"] = None
                row_series["validator_name"] = ""
                row_series["validation_date"] = ""
                row_series["flagged"] = False

                all_rows.append(row_series)
                property_counter += 1

            except Exception as e:
                logger.warning(f"Failed to convert property from {refno}: {e}")
                continue

    logger.info(f"Total properties extracted from {refno}: {len(all_rows)}")
    llm.delete_file(file)
    return all_rows


async def extract_properties(args: argparse.Namespace) -> None:
    """Main function to extract properties from all PDFs."""
    # Load extraction prompt (relative to current working directory)
    prompt_path = args.prompt_path
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    logger.info(f"Loading prompt from {prompt_path}")
    prompt = load_prompt(prompt_path)

    # Get list of PDFs (relative to current working directory)
    paper_dir = args.data_dir / "Paper_DB"
    if not paper_dir.exists():
        raise FileNotFoundError(f"Paper directory not found: {paper_dir}")

    pdf_files = sorted(paper_dir.glob("*.pdf"))

    # Filter out excluded files if exclude list is provided
    if args.exclude_list and args.exclude_list.exists():
        logger.info(f"Loading exclude list from {args.exclude_list}")
        with open(args.exclude_list, "r") as f:
            excluded_filenames = {line.strip() for line in f if line.strip()}

        original_count = len(pdf_files)
        pdf_files = [pdf for pdf in pdf_files if pdf.name not in excluded_filenames]
        excluded_count = original_count - len(pdf_files)

        if excluded_count > 0:
            logger.info(f"Excluded {excluded_count} PDF file(s) based on exclude list")

    # Default ordering is by filename
    refnos_ordering: list[str] = [pdf.stem for pdf in pdf_files]
    if args.harbor_task_ordering_registry_path is not None:
        logger.info(
            f"Loading harbor task ordering from {args.harbor_task_ordering_registry_path}"
        )
        with open(args.harbor_task_ordering_registry_path, "r") as f:
            harbor_task_ordering = json.load(f)

        # Load the refnos from harbor_task_ordering[0]["tasks"][:]["name"]
        refnos_ordering = [
            task["name"].strip().upper() for task in harbor_task_ordering[0]["tasks"]
        ]

    if args.max_num_papers is not None:
        refnos_ordering = refnos_ordering[: args.max_num_papers]

    # Reorder the pdf_files based on the refnos_ordering
    reordered_pdf_files = []
    for refno in refnos_ordering:
        for pdf in pdf_files:
            if pdf.stem == refno:
                reordered_pdf_files.append(pdf)
                break

    num_pdfs = len(reordered_pdf_files)
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
    preds_dir = args.output_dir / "unsupervised_llm_extraction"
    preds_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize model name for filename
    model_name_safe = args.model_name.replace("/", "--")

    # Process each PDF
    for pdf_path in tqdm(reordered_pdf_files, desc="Processing PDFs"):
        refno = pdf_path.stem

        # Skip if refno is specified and this isn't it
        if args.refno is not None and refno != args.refno:
            continue

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
        rows = await process_paper(
            pdf_path, prompt, llm, inf_gen_config, args.model_name
        )

        # Save to CSV
        if len(rows) > 0:
            # Create DataFrame from list of Series
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(rows)} properties from {refno} to {output_path}")
        else:
            logger.warning(f"No properties extracted from {refno}")


def main() -> None:
    """CLI entry point for console script."""
    parser = argparse.ArgumentParser(
        description="Mass extract properties from PDF files using unsupervised LLM extraction"
    )
    parser = pbench.add_base_args(parser)
    parser.add_argument(
        "--prompt_path",
        "-pp",
        type=Path,
        default=Path("prompts/unsupervised_extraction_prompt.md"),
        help="Path to the unsupervised extraction prompt (default: prompts/unsupervised_extraction_prompt.md)",
    )
    # parser.add_argument(
    #     "--file_no",
    #     "-fn",
    #     type=int,
    #     default=None,
    #     help="Specific file number to process (1-indexed). If None, process all files",
    # )
    parser.add_argument(
        "--refno",
        type=str,
        default=None,
        help="Refno to process. If None, process all refnos",
    )
    parser.add_argument(
        "--exclude_list",
        "-el",
        type=Path,
        default=None,
        help="Path to file containing list of PDF filenames to exclude (one per line)",
    )
    parser.add_argument(
        "--harbor_task_ordering_registry_path",
        type=Path,
        default=None,
        help="Path to the registry_data.json file that defines the ordering of the papers to process (default: None)",
    )
    parser.add_argument(
        "--max_num_papers",
        type=int,
        default=None,
        help="Maximum number of papers to process (default: None)",
    )
    # LLM generation arguments
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=65536,
        help="Maximum number of output tokens for LLM response (default: 65536)",
    )

    args = parser.parse_args()
    pbench.setup_logging(args.log_level)

    asyncio.run(extract_properties(args))


if __name__ == "__main__":
    main()
