"""Collect Harbor verifier outputs into PBench-style prediction JSON files.

Usage:
  uv run python src/pbench_containerized_eval/collect_harbor_results.py \
    --trials-dir trials \
    --output-dir out/harbor/precedent-search/preds

This script reads `verifier/details.json` from each trial and writes one JSON file
per trial with PBench-style records (refno/material/property_name/true/pred/response).
"""

import argparse
import json
from pathlib import Path


def main() -> int:
    """Collect trial results and write one JSON file per trial."""
    parser = argparse.ArgumentParser(
        description="Collect Harbor results into PBench format."
    )
    parser.add_argument("--trials-dir", type=Path, default="trials")
    parser.add_argument(
        "--output-dir", type=Path, default="out/harbor/precedent-search/preds"
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for trial_dir in args.trials_dir.iterdir():
        if not trial_dir.is_dir():
            continue

        details_path = trial_dir / "verifier/details.json"
        if not details_path.exists():
            continue

        try:
            with open(details_path) as f:
                details = json.load(f)
        except Exception as e:
            print(f"Skipping {trial_dir}: {e}")
            continue

        # Extract info
        rows = details.get("rows", [])
        if not rows:
            continue

        # Convert to pbench format list-of-dicts
        # Need: refno, material, property_name, true{value, unit}, pred{value, unit}, response{pred}
        # In details.json, we have this flat in 'rows'

        pbench_records = []
        for row in rows:
            record = {
                "refno": details.get("refno", "unknown"),
                "material": row["material"],
                "property_name": row["property_name"],
                "true": {"value": row["answer_value"], "unit": row["answer_unit"]},
                "pred": {"value": row["pred_value"], "unit": row["pred_unit"]},
                "rubric": row.get("rubric"),
                "response": {
                    "pred": row.get("pred_raw", {}).get("value_string", ""),
                    "source_doi": row.get("pred_raw", {}).get("source_doi"),
                    "conditions": row.get("pred_raw", {}).get("conditions"),
                    "related_materials": row.get("pred_raw", {}).get(
                        "related_materials"
                    ),
                },
                # Add metadata to enable filtering in score_task.py
                "metadata": {
                    "task": details.get("task", "precedent-search"),
                    "trial_id": trial_dir.name,
                },
            }
            pbench_records.append(record)

        # Save one JSON per trial
        out_name = f"{trial_dir.name}.json"
        out_path = args.output_dir / out_name
        with open(out_path, "w") as f:
            json.dump(pbench_records, f, indent=2)

        count += 1

    print(f"Collected results from {count} trials to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
