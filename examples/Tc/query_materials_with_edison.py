#!/usr/bin/env python3
"""Query materials from filtered_dataset.csv using Edison Platform API.

This script submits batch queries to the Edison Platform using the PRECEDENT job type
to determine if materials are known superconductors and their reported Tc values.
"""

import argparse
import asyncio
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

try:
    from edison_client import EdisonClient, JobNames
except ImportError:
    print("Error: edison-client not installed. Please install it with:")
    print("  uv pip install edison-client")
    sys.exit(1)


def load_materials(csv_path: Path, limit: int | None = None) -> pd.DataFrame:
    """Load materials from CSV file."""
    print(f"Loading materials from {csv_path}...")
    df = pd.read_csv(csv_path)

    if limit:
        df = df.head(limit)
        print(f"Limited to first {limit} materials")

    print(f"Loaded {len(df)} materials")
    return df


def create_query_for_material(formula: str) -> str:
    """Create a query for a given material formula."""
    return f"Is {formula} a known superconductor? If so, what is the highest reported Tc and what is the paper source?"


async def submit_batch_queries(
    client: EdisonClient,
    materials_df: pd.DataFrame,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Submit batch queries to Edison Platform API.

    Args:
        client: EdisonClient instance
        materials_df: DataFrame with materials data
        verbose: Enable verbose output

    Returns:
        List of result dictionaries

    """
    # Prepare task data for all materials
    tasks_data = []
    for idx, row in materials_df.iterrows():
        formula = row["reduced_formula"]
        query = create_query_for_material(formula)

        task = {
            "name": JobNames.PRECEDENT,
            "query": query,
        }
        tasks_data.append(
            {
                "task": task,
                "icsd_id": row["icsd_id"],
                "reduced_formula": formula,
                "query": query,
            }
        )

    # Submit all tasks as a batch
    print(f"\nSubmitting {len(tasks_data)} tasks to Edison Platform API...")
    print("This may take a while as we wait for all results...")

    # Extract just the task dictionaries for submission
    task_requests = [td["task"] for td in tasks_data]

    try:
        # Submit batch and wait for results
        task_responses = await client.arun_tasks_until_done(
            task_requests, verbose=verbose
        )

        # Combine responses with material data
        results = []
        for task_data, response in zip(tasks_data, task_responses):
            result = {
                "icsd_id": task_data["icsd_id"],
                "reduced_formula": task_data["reduced_formula"],
                "query": task_data["query"],
                "task_id": getattr(response, "id", None),
                "answer": getattr(response, "answer", None),
                "formatted_answer": getattr(response, "formatted_answer", None),
                "has_successful_answer": getattr(
                    response, "has_successful_answer", None
                ),
                "status": "success",
                "error": None,
            }

            # Store full response for JSON output
            result["_full_response"] = response
            results.append(result)

        print(f"\n✓ Successfully completed {len(results)} queries")
        return results

    except Exception as e:
        print(f"\n✗ Error during batch submission: {e}")
        # Return partial results with error information
        results = []
        for task_data in tasks_data:
            results.append(
                {
                    "icsd_id": task_data["icsd_id"],
                    "reduced_formula": task_data["reduced_formula"],
                    "query": task_data["query"],
                    "task_id": None,
                    "answer": None,
                    "formatted_answer": None,
                    "has_successful_answer": False,
                    "status": "error",
                    "error": str(e),
                }
            )
        return results


async def main(args: argparse.Namespace) -> None:
    """Main function to run the script.

    Args:
        args: ArgumentParser namespace

    """
    # Check for API key
    api_key = os.environ.get("EDISON_API_KEY")
    if not api_key:
        print("Error: EDISON_API_KEY environment variable not set")
        print("\nTo set the API key:")
        print("  conda env config vars set EDISON_API_KEY=your_api_key_here")
        print("  conda activate sci-llm  # Reactivate to apply")
        print("\nOr for current session only:")
        print("  export EDISON_API_KEY='your_api_key_here'")
        sys.exit(1)

    # Load materials
    try:
        materials_df = load_materials(args.input, limit=None)
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading materials: {e}")
        sys.exit(1)

    # Initialize Edison client
    try:
        client = EdisonClient(api_key=api_key)
        print("✓ Edison client initialized successfully")
    except Exception as e:
        print(f"Error initializing Edison client: {e}")
        sys.exit(1)

    try:
        # Process materials in batches
        num_materials = len(materials_df)
        batch_size = args.batch_size
        num_batches = math.ceil(num_materials / batch_size)

        print(f"\nTotal materials: {num_materials}")
        print(f"Batch size: {batch_size}")
        print(f"Total batches: {num_batches}")

        for batch_number in range(1, num_batches + 1):
            # Skip if specific batch number is requested and this isn't it
            if args.batch_number is not None and batch_number != args.batch_number:
                continue

            # Calculate batch indices
            batch_start_idx = (batch_number - 1) * batch_size
            batch_end_idx = min(batch_start_idx + batch_size, num_materials)

            print(f"\n{'=' * 60}")
            print(f"Processing batch {batch_number}/{num_batches}")
            print(
                f"Materials [{batch_start_idx}, {batch_end_idx}) out of {num_materials}"
            )
            print(f"{'=' * 60}")

            # Get batch data
            batch_df = materials_df.iloc[batch_start_idx:batch_end_idx]

            # Create filename with batch info
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"batch={batch_number}__bs={batch_size}__ts={timestamp}"

            # Check if output already exists
            output_csv = args.output_dir / f"edison_precedent_{run_name}.csv"
            if output_csv.exists():
                print(f"⚠ Output file already exists: {output_csv}")
                print("Skipping this batch. Delete the file to reprocess.")
                continue

            # Submit queries for this batch
            try:
                results = await submit_batch_queries(
                    client, batch_df, verbose=args.verbose
                )
            except Exception as e:
                print(f"Error during query submission: {e}")
                import traceback

                traceback.print_exc()
                continue

            # Save results
            if results:
                try:
                    # Create custom save with batch info
                    args.output_dir.mkdir(parents=True, exist_ok=True)

                    # Save CSV
                    # csv_data = [
                    #     {k: v for k, v in r.items() if k != "_full_response"}
                    #     for r in results
                    # ]
                    # df_results = pd.DataFrame(csv_data)
                    # df_results.to_csv(output_csv, index=False)
                    # print(f"\n✓ Saved CSV results to: {output_csv}")

                    # Save JSON
                    output_json = args.output_dir / f"edison_precedent_{run_name}.json"
                    json_data = []
                    for result in results:
                        json_result = {
                            k: v for k, v in result.items() if k != "_full_response"
                        }
                        if (
                            "_full_response" in result
                            and result["_full_response"] is not None
                        ):
                            try:
                                response_obj = result["_full_response"]
                                json_result["full_response"] = {
                                    attr: getattr(response_obj, attr, None)
                                    for attr in dir(response_obj)
                                    if not attr.startswith("_")
                                }
                            except Exception:
                                pass
                        json_data.append(json_result)

                    with open(output_json, "w") as f:
                        json.dump(json_data, f, indent=2, default=str)
                    print(f"✓ Saved JSON results to: {output_json}")

                    print(
                        f"\n✓ Batch {batch_number} complete: {len(results)} materials processed"
                    )
                except Exception as e:
                    print(f"Error saving results: {e}")
                    import traceback

                    traceback.print_exc()

        print(f"\n{'=' * 60}")
        print("All batches complete!")
        print(f"{'=' * 60}")

    finally:
        # Clean up client resources to prevent "Unclosed client session" warnings
        if client is not None:
            try:
                # Try to close any aiohttp sessions in the client
                import inspect

                if hasattr(client, "client"):
                    inner_client = client.client
                    if hasattr(inner_client, "close"):
                        result = inner_client.close()
                        if inspect.iscoroutine(result):
                            await result
                # Also try closing the client itself
                if hasattr(client, "close"):
                    result = client.close()
                    if inspect.iscoroutine(result):
                        await result
            except Exception:
                # Suppress cleanup errors - these are just warnings
                pass


def get_parser() -> argparse.ArgumentParser:
    """Get argument parser for query materials with Edison Platform API."""
    parser = argparse.ArgumentParser(
        description="Query materials using Edison Platform API"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "data" / "filtered_dataset.csv",
        help="Input CSV file with materials (default: data/filtered_dataset.csv)",
    )

    parser.add_argument(
        "--output-dir",
        "-od",
        type=Path,
        default="out",
        help="Output directory for results (default: out)",
    )

    parser.add_argument(
        "--batch-size",
        "-bs",
        type=int,
        default=10,
        help="Number of materials to process in each batch (default: 100)",
    )

    parser.add_argument(
        "--batch-number",
        "-bn",
        type=int,
        default=None,
        help="Specific batch number to process (1-indexed). If not set, processes all batches",
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output from Edison API"
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    asyncio.run(main(args))
