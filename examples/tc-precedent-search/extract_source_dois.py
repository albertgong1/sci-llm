#!/usr/bin/env -S uv run --env-file=.env -- python
"""Extract source DOI information from agent outputs and generate CSV.

Usage:
  python examples/harbor-workspace/extract_source_dois.py

This script:
1. Reads all CSV files from examples/harbor-workspace/out/harbor/precedent-search/analysis/final_results/
2. For each row, finds the corresponding agent output file based on job_id and metadata_trial_id
3. Extracts JSON from the agent output
4. Creates CSV files in the same directory with citation_info- prefix,
   preserving original columns and adding citation_info- prefixed columns
"""

import csv
import json
import re
from pathlib import Path
from typing import Optional, Union, Dict, List, Any

def get_agent_type(csv_filename: str) -> str:
    """Determine agent type from CSV filename."""
    filename_lower = csv_filename.lower()
    if "gemini_cli" in filename_lower:
        return "gemini_cli"
    elif "codex" in filename_lower:
        return "codex"
    elif "qwen_coder" in filename_lower or "qwen" in filename_lower:
        return "qwen_coder"
    elif "terminus" in filename_lower:
        return "terminus"
    else:
        raise ValueError(f"Unknown agent type in filename: {csv_filename}")


def get_agent_output_path(
    jobs_dir: Path, job_id: str, trial_id: str, agent_type: str
) -> Optional[Path]:
    """Get the path to the agent output file."""
    # Handle job_id variations (some have agent suffix like _gemini-cli)
    job_dirs = list(jobs_dir.glob(f"{job_id}*"))
    if not job_dirs:
        return None
    job_dir = job_dirs[0]

    trial_dir = job_dir / trial_id
    if not trial_dir.exists():
        return None

    agent_dir = trial_dir / "agent"
    if not agent_dir.exists():
        return None

    if agent_type == "codex":
        output_file = agent_dir / "codex.txt"
    elif agent_type == "gemini_cli":
        output_file = agent_dir / "gemini-cli.txt"
    elif agent_type == "qwen_coder":
        output_file = agent_dir / "qwen-code.txt"
    elif agent_type == "terminus":
        output_file = agent_dir / "terminus_2.pane"
        if not output_file.exists():
             # Fallback to episode-*/response.txt
            episode_dirs = list(agent_dir.glob("episode-*"))
            def get_episode_num(path):
                try:
                    return int(path.name.split("-")[-1])
                except ValueError:
                    return -1
            episode_dirs.sort(key=get_episode_num)
            if episode_dirs:
                output_file = episode_dirs[-1] / "response.txt"
        return None

    return output_file if output_file.exists() else None


def extract_json_from_jsonl(text: str) -> Optional[dict]:
    """Extract JSON from JSON Lines format (codex output).

    Codex outputs are in JSON Lines format where the final output JSON
    is embedded in an agent_message item's "text" field.
    """
    # Parse each line as JSON and look for agent_message with properties
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
            if item.get("type") == "item.completed":
                inner_item = item.get("item", {})
                if inner_item.get("type") == "agent_message":
                    message_text = inner_item.get("text", "")
                    # Try to parse the text as JSON
                    try:
                        data = json.loads(message_text)
                        if "properties" in data:
                            return data
                    except json.JSONDecodeError:
                        # Try to find JSON in the text
                        extracted = extract_json_from_plain_text(message_text)
                        if extracted:
                            return extracted
        except json.JSONDecodeError:
            continue
    return None


def extract_json_from_plain_text(text: str) -> Optional[dict]:
    """Extract JSON from plain text (gemini-cli, terminus outputs)."""
    # Try to find JSON in markdown code blocks first
    json_pattern = r"```json\s*([\s\S]*?)\s*```"
    matches = re.findall(json_pattern, text)

    for match in matches:
        try:
            data = json.loads(match)
            if "properties" in data:
                return data
        except json.JSONDecodeError:
            continue

    # Try to find standalone JSON with "properties" key
    # More robust approach: find JSON starting with {"properties"
    start_idx = text.find('{"properties"')
    if start_idx == -1:
        start_idx = text.find('{\n  "properties"')
    if start_idx == -1:
        start_idx = text.find('{\r\n  "properties"')

    if start_idx != -1:
        # Try to parse JSON from this position
        depth = 0
        end_idx = start_idx
        in_string = False
        escape_next = False

        for i, char in enumerate(text[start_idx:], start_idx):
            if escape_next:
                escape_next = False
                continue
            if char == "\\" and in_string:
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break

        if end_idx > start_idx:
            json_str = text[start_idx:end_idx]
            # Clean up escape sequences that might be in terminal output
            json_str = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", json_str)
            json_str = re.sub(r"\[\?2004[hl]", "", json_str)
            try:
                data = json.loads(json_str)
                if "properties" in data:
                    return data
            except json.JSONDecodeError:
                pass

    return None


def extract_json_from_text(text: str, agent_type: str) -> Optional[dict]:
    """Extract JSON from agent output text based on agent type."""
    if agent_type == "codex":
        # Codex uses JSON Lines format
        result = extract_json_from_jsonl(text)
        if result:
            return result
        # Fallback to plain text extraction
        return extract_json_from_plain_text(text)
    elif agent_type == "qwen_coder":
        # Qwen coder may have JSON Lines or plain text
        result = extract_json_from_jsonl(text)
        if result:
            return result
        return extract_json_from_plain_text(text)
    else:
        # gemini_cli, terminus use plain text
        return extract_json_from_plain_text(text)


def extract_source_doi_info(source_doi: dict) -> dict:
    """Extract relevant fields from a source_doi entry."""
    if isinstance(source_doi, str):
        # Sometimes source_dois is just a DOI string
        return {
            "title": "",
            "authors": "",
            "year": "",
            "doi": source_doi,
            "quoted_span": "",
        }
    return {
        "title": source_doi.get("title", ""),
        "authors": (
            "; ".join(source_doi.get("authors", []))
            if isinstance(source_doi.get("authors"), list)
            else source_doi.get("authors", "")
        ),
        "year": str(source_doi.get("year", "")),
        "doi": source_doi.get("doi", ""),
        "quoted_span": source_doi.get("quoted_span", ""),
    }


def flatten_source_dois(
    source_dois: list, property_name: str, max_sources: int = 3
) -> dict:
    """Flatten source_dois list into columns."""
    result = {}
    for i in range(max_sources):
        prefix = f"{property_name}_source_{i + 1}"
        if i < len(source_dois):
            info = extract_source_doi_info(source_dois[i])
            result[f"{prefix}_title"] = info["title"]
            result[f"{prefix}_authors"] = info["authors"]
            result[f"{prefix}_year"] = info["year"]
            result[f"{prefix}_doi"] = info["doi"]
            result[f"{prefix}_quoted_span"] = info["quoted_span"]
        else:
            result[f"{prefix}_title"] = ""
            result[f"{prefix}_authors"] = ""
            result[f"{prefix}_year"] = ""
            result[f"{prefix}_doi"] = ""
            result[f"{prefix}_quoted_span"] = ""
    return result


def process_json_data(data: dict) -> dict:
    """Process extracted JSON data and return flattened source_doi info."""
    result = {}
    properties = data.get("properties", [])

    # Group properties by property_name
    property_map: dict[str, list] = {"is_superconducting": [], "tc": [], "tcn": []}

    for prop in properties:
        prop_name = prop.get("property_name", "")
        if prop_name in property_map:
            source_dois = prop.get("source_dois", [])
            if source_dois:
                property_map[prop_name].extend(source_dois)

    # Flatten each property's source_dois
    for prop_name, source_dois in property_map.items():
        flattened = flatten_source_dois(source_dois, prop_name)
        result.update(flattened)

    return result


def process_csv_file(input_path: Path, jobs_dir: Path, output_dir: Path) -> None:
    """Process a single CSV file and extract citation info."""
    print(f"\nProcessing: {input_path.name}")

    # Determine agent type from filename
    try:
        agent_type = get_agent_type(input_path.name)
    except ValueError as e:
        print(f"  Skipping: {e}")
        return

    print(f"  Agent type: {agent_type}")

    # Read input CSV and preserve original columns
    rows = []
    original_fieldnames = []
    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)
        original_fieldnames = reader.fieldnames or []
        for row in reader:
            rows.append(row)

    print(f"  Rows: {len(rows)}")

    # Define citation columns
    citation_columns = []
    for prop_name in ["is_superconducting", "tc", "tcn"]:
        for i in range(1, 4):  # Up to 3 sources per property
            prefix = f"{prop_name}_source_{i}"
            citation_columns.extend(
                [
                    f"{prefix}_title",
                    f"{prefix}_authors",
                    f"{prefix}_year",
                    f"{prefix}_doi",
                    f"{prefix}_quoted_span",
                ]
            )

    # Combined fieldnames: original columns + citation columns
    all_fieldnames = list(original_fieldnames) + citation_columns

    output_rows = []
    success_count = 0
    error_count = 0

    for row in rows:
        job_id = row.get("job_id", "")
        trial_id = row.get("metadata_trial_id", "")

        # Start with original row data
        output_row = dict(row)

        # Initialize empty citation columns
        for col in citation_columns:
            output_row[col] = ""

        # Find agent output file
        output_path = get_agent_output_path(jobs_dir, job_id, trial_id, agent_type)

        if output_path is None:
            error_count += 1
        else:
            try:
                text = output_path.read_text(errors="ignore")
                data = extract_json_from_text(text, agent_type)

                if data is None:
                    error_count += 1
                else:
                    source_info = process_json_data(data)
                    output_row.update(source_info)
                    success_count += 1
            except Exception:
                error_count += 1

        output_rows.append(output_row)

    # Write output CSV
    output_filename = "citation_info-" + input_path.name
    out_path = output_dir / output_filename

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"  Result: {success_count} success, {error_count} errors")
    print(f"  Output: {out_path}")


def main() -> int:
    """Process all CSV files in final_results directory."""
    # Configuration
    input_dir = Path("examples/harbor-workspace/out/harbor/precedent-search/analysis/final_results")
    jobs_dir = Path("examples/harbor-workspace/jobs")
    output_dir = input_dir / "citation_info"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate directories
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")
    if not jobs_dir.exists():
        raise SystemExit(f"Jobs directory not found: {jobs_dir}")

    # Find all CSV files (exclude already processed ones with citation_info- prefix)
    csv_files = sorted(
        f for f in input_dir.glob("*.csv") if not f.name.startswith("citation_info-")
    )
    if not csv_files:
        raise SystemExit(f"No CSV files found in: {input_dir}")

    print(f"Found {len(csv_files)} CSV files in {input_dir}")
    print(f"Output directory: {output_dir}")

    # Process each CSV file
    for csv_file in csv_files:
        process_csv_file(csv_file, jobs_dir, output_dir)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
