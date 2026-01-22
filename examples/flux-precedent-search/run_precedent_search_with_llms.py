"""Run precedent search for flux growth in materials using LLMs with web search.

This script queries LLMs (with web search grounding) to determine whether materials
have been reported grown with flux growth.

Usage:
```bash
cd examples/flux-precedent-search/
uv run python run_precedent_search_with_llms.py \
    --csv flux_materials-devset.csv \
    --server gemini \
    -m gemini-3-pro-preview \
    -od out/ \
    --use_web_search
```
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path

import pandas as pd
from tqdm.asyncio import tqdm

import llm_utils
from llm_utils.common import Conversation, LLMChatResponse, Message

import pbench

logger = logging.getLogger(__name__)


def load_template(template_path: Path) -> str:
    """Load the instruction template from a markdown file.

    Args:
        template_path: Path to the template markdown file.

    Returns:
        The template text as a string.

    """
    with open(template_path, "r") as f:
        return f.read()


def render_template(template: str, material: str) -> str:
    """Render the template with the given material.

    Args:
        template: The instruction template.
        material: The material name to substitute.

    Returns:
        The rendered instruction.

    """
    rendered = template.replace("{material}", material)
    return rendered


def extract_predictions_from_json(json_data: dict, material: str) -> dict:
    """Extract predictions from the JSON response.

    Args:
        json_data: Parsed JSON response.
        material: The material name.

    Returns:
        Dict with extracted predictions.

    """
    result = {
        "material": material,
        "is_grown_with_flux": None,  # "Yes", "No", or "Unknown"
        "sources": [],  # List of source objects with title, authors, year, doi, quoted_span
        "missing_or_notable_information": None,
    }

    properties = json_data.get("properties", [])
    for prop in properties:
        prop_name = prop.get("property_name", "")
        value = prop.get("value_string", "")
        sources = prop.get("source_dois", [])

        if prop_name == "is_grown_with_flux":
            result["is_grown_with_flux"] = value
            result["sources"].extend(sources)

    # Extract other fields
    result["missing_or_notable_information"] = json_data.get(
        "missing_or_notable_information", ""
    )

    # Deduplicate sources by DOI
    seen_dois = set()
    unique_sources = []
    for source in result["sources"]:
        doi = source.get("doi") if isinstance(source, dict) else source
        if doi and doi not in seen_dois:
            seen_dois.add(doi)
            unique_sources.append(source)
    result["sources"] = unique_sources

    return result


async def process_material(
    material: str,
    instruction: str,
    llm: llm_utils.LLMChat,
    inf_gen_config: llm_utils.InferenceGenerationConfig,
) -> dict:
    """Process a single material query.

    Args:
        material: The material to search for.
        instruction: The rendered instruction prompt.
        llm: LLMChat instance.
        inf_gen_config: InferenceGenerationConfig instance.

    Returns:
        Dict with results including predictions and metadata.

    """
    conv = Conversation(
        messages=[
            Message(role="user", content=[instruction]),
        ]
    )

    result = {
        "material": material,
        "is_grown_with_flux": None,
        "sources": None,
        "missing_or_notable_information": None,
        "web_search_queries": None,
        "web_search_uris": None,
        "raw_response": None,
        "error": None,
    }

    try:
        response: LLMChatResponse = await llm.generate_response_async(
            conv, inf_gen_config
        )

        if response.error:
            result["error"] = response.error
            logger.error(f"Error processing {material}: {response.error}")
            return result

        result["raw_response"] = response.pred

        # Extract web search metadata if available
        if web_search_metadata := response.web_search_metadata:
            result["web_search_queries"] = json.dumps(web_search_metadata.queries)
            result["web_search_uris"] = json.dumps(web_search_metadata.uris)

        # Extract predictions
        predictions = extract_predictions_from_json(response.pred, material)
        result["is_grown_with_flux"] = predictions["is_grown_with_flux"]
        result["sources"] = json.dumps(predictions["sources"])
        result["missing_or_notable_information"] = predictions[
            "missing_or_notable_information"
        ]

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Error processing {material}: {e}")

    return result


async def run_precedent_search(args: argparse.Namespace) -> None:
    """Main function to run precedent search for all materials."""
    # Load CSV with materials
    csv_path = args.csv
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    logger.info(f"Loading materials from {csv_path}")
    df = pd.read_csv(csv_path)
    # Load materials from material_id column and remove spaces
    materials = df["material_id"].apply(lambda x: x.replace(" ", "")).tolist()

    if args.limit:
        materials = materials[: args.limit]

    logger.info(f"Processing {len(materials)} materials")

    # Load instruction template
    template_path = args.template_path
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    logger.info(f"Loading template from {template_path}")
    template = load_template(template_path)

    # Initialize LLM
    llm = llm_utils.get_llm(args.server, args.model_name)

    # Create inference config with web search enabled
    inf_gen_config = llm_utils.InferenceGenerationConfig(
        max_output_tokens=args.max_output_tokens,
        output_format="json",
        use_web_search=args.use_web_search,
    )

    # Setup output directory
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize model name for filename
    model_name_safe = args.model_name.replace("/", "--")
    web_search_suffix = "__websearch" if args.use_web_search else ""
    output_filename = f"flux_precedent_search__model={model_name_safe}{web_search_suffix}.csv"
    output_path = output_dir / output_filename

    # Check if output already exists
    if output_path.exists() and not args.force:
        logger.info(f"Output file exists: {output_path}. Use --force to overwrite.")
        return

    # Process materials with concurrency control
    semaphore = asyncio.Semaphore(args.max_concurrent)

    async def process_with_semaphore(material: str) -> dict:
        async with semaphore:
            instruction = render_template(template, material)
            return await process_material(material, instruction, llm, inf_gen_config)

    # Create tasks
    tasks = [process_with_semaphore(m) for m in materials]

    # Process all materials with progress bar
    results = []
    for coro in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc="Processing materials",
    ):
        result = await coro
        results.append(result)

    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)

    # Reorder columns for clarity
    column_order = [
        "material",
        "is_grown_with_flux",
        "sources",
        "missing_or_notable_information",
        "web_search_queries",
        "web_search_uris",
        "raw_response",
        "error",
    ]
    results_df = results_df[[c for c in column_order if c in results_df.columns]]

    results_df.to_csv(output_path, index=False)
    logger.info(f"Saved results to {output_path}")

    # Print summary
    n_success = results_df["error"].isna().sum()
    n_errors = results_df["error"].notna().sum()
    logger.info(f"Completed: {n_success} successful, {n_errors} errors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run precedent search for flux growth in materials using LLMs with web search"
    )
    parser = pbench.add_base_args(parser)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("flux_materials-devset.csv"),
        help="Path to CSV file with materials (default: flux_materials-devset.csv)",
    )
    parser.add_argument(
        "--template_path",
        "-tp",
        type=Path,
        default=Path("search-template/instruction.md.template"),
        help="Path to the instruction template (default: search-template/instruction.md.template)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of materials to process (default: None = all)",
    )
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=8192,
        help="Maximum number of output tokens for LLM response (default: 8192)",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=5,
        help="Maximum number of concurrent API calls (default: 5)",
    )
    parser.add_argument(
        "--use_web_search",
        action="store_true",
        default=True,
        help="Enable web search grounding (default: True)",
    )

    args = parser.parse_args()
    pbench.setup_logging(args.log_level)

    asyncio.run(run_precedent_search(args))
