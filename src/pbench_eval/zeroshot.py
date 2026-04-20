#!/usr/bin/env -S uv run --env-file=.env -- python
"""Mass extract properties from PDFs using unsupervised LLM extraction.

This script processes all PDF papers and extracts comprehensive property data
using an LLM with an unsupervised extraction prompt. Predictions are saved as
per-paper JSON files in the shape consumed by the Harbor verifier:

    {
      "refno": "...",
      "agent": "zeroshot",
      "model": "...",
      "paper_pdf_path": "...",
      "properties": [{"id": "prop_000", "material_or_system": "...", ...}, ...]
    }

Usage:
```bash
uv run pbench-eval \
    --server gemini \
    --model_name gemini-3-pro-preview \
    -od out/ \
    -pp prompts/unsupervised_extraction_prompt.md \
    -dd data/ \
    -log_level INFO
```

Predictions land in `out/preds/*.json`.
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path

from slugify import slugify
from datasets import load_dataset
from dotenv import load_dotenv

import llm_utils
from llm_utils.common import Conversation, File, LLMChatResponse, Message
import pbench

logger = logging.getLogger(__name__)

# Default concurrency limit for parallel processing
MAX_CONCURRENT_TASKS = 20


def load_prompt(prompt_path: Path) -> str:
    """Load the extraction prompt from a markdown file."""
    with open(prompt_path, "r") as f:
        return f.read()


def load_hf_refnos(hf_repo: str, hf_split: str, hf_revision: str | None) -> set[str]:
    """Load refnos from a HuggingFace dataset."""
    dataset = load_dataset(hf_repo, split=hf_split, revision=hf_revision)
    return {row["refno"] for row in dataset}


async def process_paper(
    paper_path: Path,
    prompt: str,
    llm: llm_utils.LLMChat,
    inf_gen_config: llm_utils.InferenceGenerationConfig,
) -> tuple[list[dict], LLMChatResponse | None]:
    """Extract properties from a single PDF. Returns (properties, raw response)."""
    refno = paper_path.stem
    file = File(path=paper_path)

    conv = Conversation(messages=[Message(role="user", content=[file, prompt])])
    response = await llm.generate_response_async(conv, inf_gen_config)
    if response.error:
        logger.warning(f"LLM error for {refno}: {response.error}")

    # NOTE: with output_format="json", response.pred is already parsed JSON.
    properties: list[dict] = []
    try:
        json_data = response.pred or {}
        if "properties" not in json_data:
            logger.warning(f"No 'properties' key in JSON for {refno}")
        props = json_data.get("properties", [])
        if not isinstance(props, list):
            logger.warning(f"'properties' is not a list for {refno}")
            props = []
        properties = [p for p in props if isinstance(p, dict)]
    except Exception as e:
        logger.error(f"Error processing {refno}: {e}")

    # Stamp a stable id on each property; drop the temporary _page_num field
    # from per-page extraction modes if present.
    for i, prop in enumerate(properties):
        prop.pop("_page_num", None)
        prop.setdefault("id", f"prop_{i:03d}")

    logger.info(f"Total properties extracted from {refno}: {len(properties)}")
    llm.delete_file(file)
    return properties, response


async def process_single_paper_task(
    pdf_path: Path,
    prompt: str,
    llm: llm_utils.LLMChat,
    inf_gen_config: llm_utils.InferenceGenerationConfig,
    model_name: str,
    model_name_safe: str,
    preds_dir: Path,
    trajectory_dir: Path,
    force: bool,
    semaphore: asyncio.Semaphore,
) -> None:
    """Process a single paper with semaphore-controlled concurrency."""
    async with semaphore:
        refno = pdf_path.stem
        logger.info(f"Processing {refno}...")

        reasoning_effort = inf_gen_config.reasoning_effort
        reasoning_suffix = (
            f"__reasoning_effort={reasoning_effort}" if reasoning_effort else ""
        )
        output_filename = f"extracted_properties__model={model_name_safe}{reasoning_suffix}__refno={refno}.json"
        output_path = preds_dir / output_filename

        if output_path.exists() and not force:
            logger.info(
                f"Skipping {refno} as {output_path} already exists. Use --force to overwrite."
            )
            return

        properties, response = await process_paper(
            pdf_path, prompt, llm, inf_gen_config
        )

        if properties:
            payload = {
                "refno": refno,
                "agent": "zeroshot",
                "model": model_name,
                "paper_pdf_path": str(pdf_path),
                "properties": properties,
            }
            with open(output_path, "w") as f:
                json.dump(payload, f, indent=2)
            logger.info(
                f"Saved {len(properties)} properties from {refno} to {output_path}"
            )
        else:
            logger.warning(f"No properties extracted from {refno}")

        trajectory_filename = f"trajectory__agent=zeroshot__model={model_name_safe}{reasoning_suffix}__refno={refno}.json"
        trajectory_path = trajectory_dir / trajectory_filename
        trajectory_data = {
            "prompt": prompt,
            "inf_gen_config": inf_gen_config.model_dump(),
            "llm_response": response.model_dump() if response else None,
        }
        with open(trajectory_path, "w") as f:
            json.dump(trajectory_data, f, indent=2)
        logger.info(f"Saved trajectory for {refno} to {trajectory_path}")


async def extract_properties(args: argparse.Namespace) -> None:
    """Main function to extract properties from all PDFs."""
    prompt_path = args.prompt_path
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    logger.info(f"Loading prompt from {prompt_path}")
    prompt = load_prompt(prompt_path)

    paper_dir = args.data_dir / "Paper_DB"
    if not paper_dir.exists():
        raise FileNotFoundError(f"Paper directory not found: {paper_dir}")

    pdf_files = sorted(paper_dir.glob("*.pdf"))

    if args.exclude_list and args.exclude_list.exists():
        logger.info(f"Loading exclude list from {args.exclude_list}")
        with open(args.exclude_list, "r") as f:
            excluded_filenames = {line.strip() for line in f if line.strip()}

        original_count = len(pdf_files)
        pdf_files = [pdf for pdf in pdf_files if pdf.name not in excluded_filenames]
        excluded_count = original_count - len(pdf_files)

        if excluded_count > 0:
            logger.info(f"Excluded {excluded_count} PDF file(s) based on exclude list")

    refnos_ordering: list[str] = [pdf.stem for pdf in pdf_files]
    if args.harbor_task_ordering_registry_path is not None:
        logger.info(
            f"Loading harbor task ordering from {args.harbor_task_ordering_registry_path}"
        )
        with open(args.harbor_task_ordering_registry_path, "r") as f:
            harbor_task_ordering = json.load(f)

        refnos_ordering = [task["name"] for task in harbor_task_ordering[0]["tasks"]]

    if args.max_num_papers is not None:
        refnos_ordering = refnos_ordering[: args.max_num_papers]

    # Reorder the pdf_files based on refnos_ordering. Slugify both sides because
    # Harbor registry refnos are slugified.
    reordered_pdf_files = []
    for refno in refnos_ordering:
        for pdf in pdf_files:
            if slugify(pdf.stem) == slugify(refno):
                reordered_pdf_files.append(pdf)
                break

    if args.refno is not None:
        reordered_pdf_files = [
            pdf for pdf in reordered_pdf_files if pdf.stem == args.refno
        ]

    if args.hf_repo is not None and args.hf_split is not None:
        logger.info(
            f"Loading refnos from HuggingFace dataset: {args.hf_repo} "
            f"(split={args.hf_split}, revision={args.hf_revision})"
        )
        hf_refnos = load_hf_refnos(args.hf_repo, args.hf_split, args.hf_revision)
        original_count = len(reordered_pdf_files)
        reordered_pdf_files = [
            pdf for pdf in reordered_pdf_files if pdf.stem in hf_refnos
        ]
        filtered_count = original_count - len(reordered_pdf_files)
        if filtered_count > 0:
            logger.info(
                f"Filtered out {filtered_count} PDF(s) not in HuggingFace dataset"
            )

    num_pdfs = len(reordered_pdf_files)
    logger.info(f"Found {num_pdfs} PDF files in {paper_dir}")

    if num_pdfs == 0:
        logger.warning("No PDF files found to process")
        return

    llm = llm_utils.get_llm(args.server, args.model_name)
    inf_gen_config = llm_utils.InferenceGenerationConfig(
        max_output_tokens=args.max_output_tokens,
        output_format="json",
        reasoning_effort=args.openai_reasoning_effort,
    )

    preds_dir = args.output_dir / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir = args.output_dir / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    model_name_safe = args.model_name.replace("/", "--")
    semaphore = asyncio.Semaphore(args.max_concurrent)

    tasks = [
        process_single_paper_task(
            pdf_path=pdf_path,
            prompt=prompt,
            llm=llm,
            inf_gen_config=inf_gen_config,
            model_name=args.model_name,
            model_name_safe=model_name_safe,
            preds_dir=preds_dir,
            trajectory_dir=trajectory_dir,
            force=args.force,
            semaphore=semaphore,
        )
        for pdf_path in reordered_pdf_files
    ]

    logger.info(
        f"Processing {len(tasks)} papers with max {args.max_concurrent} concurrent tasks..."
    )
    await asyncio.gather(*tasks, return_exceptions=True)


def main() -> None:
    """CLI entry point for console script."""
    load_dotenv()

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
        help="Path to the registry_data.json file that defines the ordering of papers (default: None)",
    )
    parser.add_argument(
        "--max_num_papers",
        type=int,
        default=None,
        help="Maximum number of papers to process (default: None)",
    )
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=65536,
        help="Maximum number of output tokens for LLM response (default: 65536)",
    )
    parser.add_argument(
        "--openai_reasoning_effort",
        type=str,
        default=None,
        help="Reasoning effort for OpenAI models (default: None)",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=MAX_CONCURRENT_TASKS,
        help=f"Maximum number of concurrent tasks (default: {MAX_CONCURRENT_TASKS})",
    )

    args = parser.parse_args()
    pbench.setup_logging(args.log_level)

    asyncio.run(extract_properties(args))


if __name__ == "__main__":
    main()
