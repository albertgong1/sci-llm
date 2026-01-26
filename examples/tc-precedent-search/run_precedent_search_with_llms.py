"""Run precedent search for superconducting materials using LLMs with web search.

This script queries LLMs (with web search grounding) to determine whether materials
have been reported as superconducting, extracting Tc values and related information.

Usage:
```bash
cd examples/tc-precedent-search/
uv run python run_precedent_search_with_llms.py \
    --csv SuperCon_Tc_Tcn_dev-set.csv \
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
import time
from pathlib import Path

import pandas as pd
from tqdm.asyncio import tqdm

import llm_utils
from llm_utils.common import Conversation, LLMChatResponse, Message, parse_json_response

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
    # Remove or replace {paper_at_command} since we're using web search instead of papers
    rendered = template.replace("{paper_at_command}", "")
    rendered = rendered.replace("{material}", material)
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
        "is_superconducting": None,  # "Yes", "No", or "Unknown"
        "tc_values": [],
        "tcn_values": [],
        "sources": [],  # List of source objects with title, authors, year, doi, quoted_span
        "superconductors_mentioned": [],
        "electronic_or_magnetic_phases": [],
        "missing_or_notable_information": None,
    }

    properties = json_data.get("properties", [])
    for prop in properties:
        prop_name = prop.get("property_name", "")
        value = prop.get("value_string", "")
        sources = prop.get("source_dois", [])

        if prop_name == "is_superconducting":
            result["is_superconducting"] = value
            result["sources"].extend(sources)
        elif prop_name == "tc":
            if value:
                result["tc_values"].append(value)
            result["sources"].extend(sources)
        elif prop_name == "tcn":
            if value:
                result["tcn_values"].append(value)
            result["sources"].extend(sources)

    # Extract other fields
    result["superconductors_mentioned"] = json_data.get("superconductors_mentioned", [])
    result["electronic_or_magnetic_phases"] = json_data.get(
        "electronic_or_magnetic_phases", []
    )
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
        "is_superconducting": None,
        "tc_values": None,
        "tcn_values": None,
        "sources": None,
        "superconductors_mentioned": None,
        "electronic_or_magnetic_phases": None,
        "missing_or_notable_information": None,
        "web_search_queries": None,
        "web_search_uris": None,
        "web_search_num_tool_calls": None,
        "raw_response": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "cached_tokens": None,
        "thinking_tokens": None,
        "time_taken_seconds": None,
        "finish_reason": None,
        "web_search_titles": None,
        "web_search_grounding_supports": None,
        "error": None,
    }

    try:
        start_time = time.time()
        response: LLMChatResponse = await llm.generate_response_async(
            conv, inf_gen_config
        )
        end_time = time.time()
        result["time_taken_seconds"] = end_time - start_time
        
        # Extract usage
        if response.usage:
            result["prompt_tokens"] = response.usage.get("prompt_tokens")
            result["completion_tokens"] = response.usage.get("completion_tokens")
            result["total_tokens"] = response.usage.get("total_tokens")
            result["cached_tokens"] = response.usage.get("cached_tokens")
            result["thinking_tokens"] = response.usage.get("thinking_tokens")

        if response.error:
            result["error"] = response.error
            logger.error(f"Error processing {material}: {response.error}")
            return result

        if str(getattr(response, "finish_reason", "")):
             result["finish_reason"] = response.finish_reason

        result["raw_response"] = response.pred

        # Extract web search metadata if available
        if web_search_metadata := response.web_search_metadata:
            result["web_search_queries"] = json.dumps(web_search_metadata.queries)
            result["web_search_uris"] = json.dumps(web_search_metadata.uris)
            result["web_search_titles"] = json.dumps(web_search_metadata.titles)
            result["web_search_grounding_supports"] = json.dumps(web_search_metadata.grounding_supports)
            result["web_search_num_tool_calls"] = web_search_metadata.num_tool_calls

        # Extract predictions
        json_data = parse_json_response(response.pred)
        predictions = extract_predictions_from_json(json_data, material)
        result["is_superconducting"] = predictions["is_superconducting"]
        result["tc_values"] = json.dumps(predictions["tc_values"])
        result["tcn_values"] = json.dumps(predictions["tcn_values"])
        result["sources"] = json.dumps(predictions["sources"])
        result["superconductors_mentioned"] = json.dumps(
            predictions["superconductors_mentioned"]
        )
        result["electronic_or_magnetic_phases"] = json.dumps(
            predictions["electronic_or_magnetic_phases"]
        )
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
    materials = df["material"].tolist()

    if args.limit:
        materials = materials[: args.limit]

    if args.material:
        if args.material in materials:
            materials = [args.material]
        else:
            logger.warning(f"Material {args.material} not found in dataset")
            materials = []

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
    # NOTE: OpenAI does not support web_search+JSON output format. So if the server is openai and web search is enabled, the output format will be text.
    if args.server == "openai" and args.use_web_search:
        output_format = "text"
    else:
        output_format = "json"

    inf_gen_config = llm_utils.InferenceGenerationConfig(
        max_output_tokens=args.max_output_tokens,
        output_format=output_format,
        use_web_search=args.use_web_search,
        reasoning_effort=args.reasoning_effort,
    )

    # Setup output directory
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize model name for filename
    model_name_safe = args.model_name.replace("/", "--")
    web_search_suffix = "__websearch" if args.use_web_search else ""
    run_suffix = args.run
    output_filename = f"precedent_search__model={model_name_safe}{web_search_suffix}_{run_suffix}.csv"
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

    # Merge with ground truth from original CSV for comparison
    ground_truth_df = df[
        [
            "material",
            "Has material been reported to be superconducting?",
            "What is the highest measured Tc?",
            "What is the lowest temp for measurement at which material was not superconducting?",
        ]
    ].rename(
        columns={
            "Has material been reported to be superconducting?": "gt_is_superconducting",
            "What is the highest measured Tc?": "gt_highest_tc",
            "What is the lowest temp for measurement at which material was not superconducting?": "gt_lowest_tcn",
        }
    )

    # Merge predictions with ground truth
    merged_df = results_df.merge(ground_truth_df, on="material", how="left")

    # Reorder columns for clarity
    column_order = [
        "material",
        "is_superconducting",
        "gt_is_superconducting",
        "tc_values",
        "gt_highest_tc",
        "tcn_values",
        "gt_lowest_tcn",
        "sources",
        "superconductors_mentioned",
        "electronic_or_magnetic_phases",
        "missing_or_notable_information",
        "web_search_queries",
        "web_search_uris",
        "web_search_titles",
        "web_search_grounding_supports",
        "web_search_num_tool_calls",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_tokens",
        "thinking_tokens",
        "time_taken_seconds",
        "finish_reason",
        "raw_response",
        "error",
    ]
    merged_df = merged_df[[c for c in column_order if c in merged_df.columns]]

    merged_df.to_csv(output_path, index=False)
    logger.info(f"Saved results to {output_path}")

    # Print summary
    n_success = merged_df["error"].isna().sum()
    n_errors = merged_df["error"].notna().sum()
    logger.info(f"Completed: {n_success} successful, {n_errors} errors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run precedent search for superconducting materials using LLMs with web search"
    )
    parser = pbench.add_base_args(parser)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("SuperCon_Tc_Tcn_dev-set.csv"),
        help="Path to CSV file with materials (default: SuperCon_Tc_Tcn_dev-set.csv)",
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
        "--material",
        type=str,
        default=None,
        help="Specific material to process (default: None)",
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
        default=False,
        help="Enable web search grounding (default: False)",
    )
    parser.add_argument(
        "--run",
        type=str,
        default="",
        help="Suffix to add to the output filename (default: '')",
    )

    args = parser.parse_args()
    pbench.setup_logging(args.log_level)

    asyncio.run(run_precedent_search(args))
