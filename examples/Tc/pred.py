#!/usr/bin/env python3
"""
Query materials from combined_predictions.csv using Google Gemini API.

This script processes the CSV file in batches, querying the Gemini API to identify
related superconductor materials mentioned in the formatted_answer column.

Based on phantom-wiki's __main__.py structure with async batch processing.
"""

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

import pandas as pd

try:
    from gemini_utils import GeminiChat, InferenceGenerationConfig, LLMChatResponse
except ImportError:
    from .gemini_utils import GeminiChat, InferenceGenerationConfig, LLMChatResponse


RELATED_MATERIALS_PROMPT_TEMPLATE = """\
You are a careful condensed-matter physicist.
Your task is to analyze the provided MATERIAL DESCRIPTION and extract highly specific, factual information *only if it explicitly appears in the description itself*.
Do NOT infer, guess, or use any external knowledge.
-------------------------------------
### MATERIAL DESCRIPTION
{{formatted_answer}}
-------------------------------------
### TASKS
1. *Superconductor Mention Check*
   Identify all materials mentioned in the description that are:
   - closely related to the main material (parent compound, doped variant, structural analogue, neighboring stoichiometry, etc.), AND
   - explicitly stated to be superconductors.
   For EACH superconducting related material extract:
     • The material name/formula
     • The reported critical temperature Tc (exactly as stated)
2. *Electronic / Magnetic Ground State Extraction*
   Identify all explicit statements about electronic or magnetic phases for:
   - the main material
   - any related materials
   Extract a separate entry for each material-phase pair.
   Recognized phases include (but are not limited to):
   - band gap / semiconductor
   - Mott insulator
   - antiferromagnet
   - ferromagnet
   - charge density wave
   - spin glass
   - metallic state
   - paramagnetic / diamagnetic behavior
   - any explicitly named low-temperature phase
   For EACH phase extract:
     • the material it refers to
     • the phase name
     • the temperature at which the phase is measured, if stated
     • whether the temperature refers to a transition temperature or a measurement temperature
3. *Descriptive Note (Missing or Notable Information)*
   Provide a short (1–3 sentence) note summarizing:
   - Important information in the description that does not fit the structured fields
   - Ambiguities that might limit extraction
   - Extra details about materials, synthesis, structure, or properties that cannot be placed into the JSON categories
   This note must still be grounded *only in the text explicitly provided*.
4. *STRICT CONSTRAINTS*
   - Use ONLY information explicitly present in the description.
   - If a Tc, phase, or temperature is not stated, return null.
   - Do NOT infer superconductivity or phases even if they are typical for the material class.
   - If multiple materials or phases appear, list them all independently.
5. *OUTPUT FORMAT*
   Return a JSON object with the following exact structure:
{
  "superconductors_mentioned": [
    {
      "material": "...",
      "Tc": "..."     // numeric value + units exactly as written
    }
  ],
  "electronic_or_magnetic_phases": [
    {
      "material": "...",
      "phase": "...",
      "temperature": "...",          // null if not provided
      "temperature_type": "transition | measurement | null"
    }
  ],
  "missing_or_notable_information": "..."
}
If no items exist for a category, return an empty list for that category.
6. *DO NOT add explanations outside the JSON.*
"""

NOT_SUPERCONDUCTOR_PROMPT_TEMPLATE = """\
You are a careful condensed-matter physicist.
Your task is to answer the provided QUESTION using the DESCRIPTION.
Do NOT infer, guess, or use any external knowledge.
Think step by step.
-------------------------------------
### EXAMPLE
QUESTION: What is the lowest temperature that the material NdPb3 has been tested for superconductivity? If mentioned, please put your answer in \\boxed{}.
DESCRIPTION: We investigated the rare-earth lead intermetallic NdPb3 (AuCu3-type, L12) within the RPb3 family. Across magnetization, transport, and de Haas–van Alphen (dHvA) studies, NdPb3 is consistently reported as an antiferromagnet with Néel temperature around 2.7 K, and no superconducting transition has been observed down to the experimental floors of these measurements (~0.4 K).
ANSWER: \\boxed{0.4 K}
-------------------------------------
### TASK
QUESTION: What is the lowest temperature that the material {{reduced_formula}} has been tested for superconductivity? If mentioned, please put your answer in \\boxed{}.
DESCRIPTION: {{formatted_answer}}
ANSWER:
"""

if False:
    def create_prompt(reduced_formula: str, formatted_answer: str) -> str:
        """Create prompt for querying about related superconductors."""
        # return (
        #     "Does this description mention a closely related material that is a "
        #     "known superconductor? If so, name the material and the associated Tc "
        #     "using only the information in the description\n\n"
        #     f"Description: {formatted_answer}"
        # )
        # Use replace() instead of format() to avoid issues with curly braces in formatted_answer
        return RELATED_MATERIALS_PROMPT_TEMPLATE.replace("{{formatted_answer}}", formatted_answer)
else:
    # for doping insulators, can you find out which ones are magnetic/antiferromagnetic? I think what you want to dope are Mott insulators. Can you check which ones are reported to be insulators but calculated to be metallic for example?
    def create_prompt(reduced_formula: str, formatted_answer: str) -> str:
        """Create prompt for querying about insulating superconductors."""
        return NOT_SUPERCONDUCTOR_PROMPT_TEMPLATE.replace("{{reduced_formula}}", reduced_formula).replace("{{formatted_answer}}", formatted_answer)

# Based on the provided description, is material {reduced_formula} magnetic/antiferromagnetic?
# Description: {formatted_answer}
# Answer: """

# Basedon the provided description, is material {reduced_formula} metallic?
# Description: {formatted_answer}
# Answer: """


def save_preds(
    pred_path: Path,
    batch_df: pd.DataFrame,
    responses: list[LLMChatResponse],
    args: argparse.Namespace,
    batch_number: int,
) -> None:
    """
    Save predictions to JSON file as a list of dictionaries.

    Matches the format from query_materials_with_edison.py for compatibility
    with combine_predictions.py.
    """
    preds = []
    batch_size = len(batch_df)

    for i, (row_idx, row) in enumerate(batch_df.iterrows()):
        reduced_formula = str(row.get("reduced_formula", ""))
        formatted_answer = str(row.get("formatted_answer", ""))

        pred_dict = {
            "icsd_id": row.get("icsd_id", None),
            "reduced_formula": reduced_formula,
            "batch_number": row.get("batch_number", None),
            "query": row.get("query", None),
            "formatted_answer": formatted_answer,
            "prompt": create_prompt(reduced_formula, formatted_answer),
            "gemini_response": responses[i].pred,
            "error": responses[i].error,
            "status": "success" if responses[i].error is None else "error",
            "metadata": {
                "model": args.model_name,
                "batch_size": batch_size,
                "batch_number": batch_number,
                "row_index": str(row_idx),  # Store as string for JSON compatibility
            },
            "usage": responses[i].usage,
        }
        preds.append(pred_dict)

    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pred_path, "w") as f:
        json.dump(preds, f, indent=2, default=str)
        f.flush()

    print(f"✓ Saved predictions to {pred_path}")


async def process_batch(
    llm_chat: GeminiChat,
    batch_df: pd.DataFrame,
    inf_gen_config: InferenceGenerationConfig,
    args: argparse.Namespace,
    batch_number: int,
    batch_size: int,
) -> None:
    """
    Process a single batch of rows.

    Uses asyncio.gather to process rows in parallel.
    """
    # Create output path matching the edison pattern: gemini_batch={N}__bs={size}.json
    run_name = f"gemini_batch={batch_number}__bs={batch_size}"
    pred_path = Path(args.output_dir) / f"{run_name}.json"

    # Skip if output already exists and --force is not set
    if pred_path.exists() and not args.force:
        print(f"⚠ Skipping {pred_path} as it already exists. Use --force to overwrite.")
        return

    print(f"\n{'='*60}")
    print(f"Processing batch {batch_number}")
    print(f"Rows: {len(batch_df)}")
    print(f"{'='*60}")

    # Process all rows in batch using asyncio.gather
    print(f"Querying Gemini API for {len(batch_df)} rows...")
    responses: list[LLMChatResponse] = await asyncio.gather(
        *[
            llm_chat.chat_async(
                create_prompt(
                    str(row.get("reduced_formula", "")),
                    str(row.get("formatted_answer", "")),
                ),
                inf_gen_config,
            )
            for _, row in batch_df.iterrows()
        ]
    )

    # Count successes and errors
    num_success = sum(1 for r in responses if r.error is None)
    num_errors = len(responses) - num_success
    print(f"✓ Completed: {num_success} successful, {num_errors} errors")

    # Save predictions
    save_preds(pred_path, batch_df, responses, args, batch_number)


async def main(args: argparse.Namespace) -> None:
    """Main function to run the script."""
    # Load data
    print(f"Loading data from {args.input}...")
    try:
        df = pd.read_csv(args.input)
        print(f"✓ Loaded {len(df)} rows")
    except FileNotFoundError:
        print(f"✗ Error: Input file not found: {args.input}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        sys.exit(1)

    # Initialize Gemini client
    print(f"Initializing Gemini client with model '{args.model_name}'...")
    try:
        llm_chat = GeminiChat(model_name=args.model_name)
        print("✓ Gemini client initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing Gemini client: {e}")
        sys.exit(1)

    # Setup inference configuration
    inf_gen_config = InferenceGenerationConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        max_retries=args.max_retries,
        wait_seconds=args.wait_seconds,
    )

    # Process batches
    num_rows = len(df)
    batch_size = args.batch_size
    num_batches = math.ceil(num_rows / batch_size)

    print(f"\n{'='*60}")
    print(f"Processing Configuration")
    print(f"{'='*60}")
    print(f"Total rows: {num_rows}")
    print(f"Batch size: {batch_size}")
    print(f"Total batches: {num_batches}")
    print(f"{'='*60}")

    for batch_number in range(1, num_batches + 1):
        # Skip if specific batch number is requested and this isn't it
        if args.batch_number is not None and batch_number != args.batch_number:
            continue

        # Calculate batch indices
        batch_start_idx = (batch_number - 1) * batch_size
        batch_end_idx = min(batch_start_idx + batch_size, num_rows)

        print(
            f"\nBatch {batch_number}/{num_batches}: rows [{batch_start_idx}, {batch_end_idx})"
        )

        # Get batch data
        batch_df = df.iloc[batch_start_idx:batch_end_idx]

        # Process batch
        try:
            await process_batch(
                llm_chat, batch_df, inf_gen_config, args, batch_number, batch_size
            )
        except Exception as e:
            print(f"✗ Error processing batch {batch_number}: {e}")
            import traceback

            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print("All batches complete!")
    print(f"{'='*60}")


def get_parser() -> argparse.ArgumentParser:
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        description="Query materials using Google Gemini API"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "combined_predictions.csv",
        help="Input CSV file with materials (default: combined_predictions.csv)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "preds",
        help="Output directory for results (default: preds)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of rows to process in each batch (default: 100)",
    )

    parser.add_argument(
        "--batch-number",
        type=int,
        default=None,
        help="Specific batch number to process (1-indexed). If not set, processes all batches",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model name (default: gemini-2.5-flash)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing results",
    )

    # Inference configuration
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Maximum tokens in response (default: 1024)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation (default: 0.7)",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-k sampling (default: 40)",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling (default: 0.95)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries on API errors (default: 3)",
    )

    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=1.0,
        help="Initial wait time between retries in seconds (default: 1.0)",
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    asyncio.run(main(args))
