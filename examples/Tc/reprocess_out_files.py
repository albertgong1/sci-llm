import argparse
import json

from pathlib import Path

def reprocess_out_files(args: argparse.Namespace):
    """Reprocess the out files in the input directory and save the results to the output directory."""
    output_dir = Path(str(args.input_dir) + "__reprocessed")
    output_dir.mkdir(parents=True, exist_ok=True)
    for file in args.input_dir.glob(args.pattern):
        with open(file, "r") as f:
            data = json.load(f)
        for item in data:
            response = item["full_response"]["environment_frame"]["state"]["state"]["response"]["answer"]
            item["answer"] = response["raw_answer"]
            item["formatted_answer"] = response["answer"]
            item["has_successful_answer"] = response["has_successful_answer"]
        with open(output_dir / file.name, "w") as f:
            json.dump(data, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default="out")
    parser.add_argument("--pattern", type=str, default="edison_precedent_*.json")
    args = parser.parse_args()
    reprocess_out_files(args)