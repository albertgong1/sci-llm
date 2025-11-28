#!/usr/bin/env python3
"""Query materials from filtered_dataset.csv using Edison Platform API.

This script submits batch queries to the Edison Platform using the PRECEDENT job type
to determine if materials are known superconductors and their reported Tc values.
"""

import argparse
import asyncio
import json
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
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """Submit batch queries to Edison Platform API.

    Args:
        client: EdisonClient instance
        materials_df: DataFrame with materials data
        verbose: Enable verbose output
        dry_run: If True, only print queries without submitting

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

    if dry_run:
        print("\n=== DRY RUN MODE ===")
        print(f"Would submit {len(tasks_data)} queries:")
        for i, task_data in enumerate(tasks_data[:5], 1):  # Show first 5
            print(f"{i}. {task_data['query']}")
        if len(tasks_data) > 5:
            print(f"... and {len(tasks_data) - 5} more")
        return []

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


def save_results(
    results: List[Dict[str, Any]], output_dir: Path, timestamp: str
) -> tuple[Path, Path]:
    """Save results to CSV and JSON files.

    Args:
        results: List of result dictionaries
        output_dir: Directory to save output files
        timestamp: Timestamp string for filename

    Returns:
        Tuple of (csv_path, json_path)

    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare CSV data (exclude full response object)
    csv_data = []
    for result in results:
        csv_row = {k: v for k, v in result.items() if k != "_full_response"}
        csv_data.append(csv_row)

    # Save to CSV
    csv_path = output_dir / f"edison_precedent_results_{timestamp}.csv"
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved CSV results to: {csv_path}")

    # Prepare JSON data (convert response objects to dict)
    json_data = []
    for result in results:
        json_result = {k: v for k, v in result.items() if k != "_full_response"}

        # Try to serialize the full response if available
        if "_full_response" in result and result["_full_response"] is not None:
            try:
                response_obj = result["_full_response"]
                # Convert response object to dict by extracting attributes
                json_result["full_response"] = {
                    attr: getattr(response_obj, attr, None)
                    for attr in dir(response_obj)
                    if not attr.startswith("_")
                }
            except Exception as e:
                json_result["full_response_error"] = str(e)

        json_data.append(json_result)

    # Save to JSON
    json_path = output_dir / f"edison_precedent_results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"✓ Saved JSON results to: {json_path}")

    return csv_path, json_path


async def main() -> None:
    """Main function to run the script."""
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
        default=Path(__file__).parent / "out",
        help="Output directory for results (default: out)",
    )

    parser.add_argument(
        "--limit", type=int, help="Limit number of materials to query (for testing)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview queries without submitting to API",
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output from Edison API"
    )

    args = parser.parse_args()

    # Check for API key (not needed for dry-run)
    api_key = os.environ.get("EDISON_API_KEY")
    if not api_key and not args.dry_run:
        print("Error: EDISON_API_KEY environment variable not set")
        print("\nTo set the API key:")
        print("  conda env config vars set EDISON_API_KEY=your_api_key_here")
        print("  conda activate sci-llm  # Reactivate to apply")
        print("\nOr for current session only:")
        print("  export EDISON_API_KEY='your_api_key_here'")
        sys.exit(1)

    # Load materials
    try:
        materials_df = load_materials(args.input, args.limit)
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading materials: {e}")
        sys.exit(1)

    # Initialize Edison client
    client = None
    if not args.dry_run:
        try:
            client = EdisonClient(api_key=api_key)
            print("✓ Edison client initialized successfully")
        except Exception as e:
            print(f"Error initializing Edison client: {e}")
            sys.exit(1)

    try:
        # Submit queries
        try:
            results = await submit_batch_queries(
                client, materials_df, verbose=args.verbose, dry_run=args.dry_run
            )
        except Exception as e:
            print(f"Error during query submission: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

        # Save results (skip if dry run)
        if not args.dry_run and results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            try:
                csv_path, json_path = save_results(results, args.output_dir, timestamp)
                print(f"\n{'=' * 60}")
                print("Processing complete!")
                print(f"{'=' * 60}")
                print(f"Total materials processed: {len(results)}")
                print(f"Output directory: {args.output_dir}")
            except Exception as e:
                print(f"Error saving results: {e}")
                import traceback

                traceback.print_exc()
                sys.exit(1)
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


if __name__ == "__main__":
    asyncio.run(main())
