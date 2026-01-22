import argparse
import json
import re
from pathlib import Path

from typing import Optional

def parse_codex_log(log_path: Path) -> Optional[dict]:
    """
    Parse the agent/codex.txt file to find the last valid JSON agent message.
    Expected format in log lines:
    {"type":"agent_message","text":"{\n  \"properties\": ... }"}
    """
    if not log_path.exists():
        return None

    # Read lines in reverse or just read all and find the last one
    try:
        with open(log_path, "r", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return None

    # Look for the last agent_message
    last_json = None
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            # The log wrapper wraps the actual message in a JSON object
            wrapper = json.loads(line)
            
            # Direct agent_message
            if wrapper.get("type") == "agent_message" and "text" in wrapper:
                content = wrapper["text"]
                last_json = json.loads(content)
                break
                
            # Wrapped in item.completed
            if wrapper.get("type") == "item.completed":
                item = wrapper.get("item", {})
                if item.get("type") == "agent_message" and "text" in item:
                    content = item["text"]
                    last_json = json.loads(content)
                    break
                    
        except json.JSONDecodeError:
            continue
    
    return last_json

def get_task_ground_truth(task_path: Path) -> Optional[dict]:
    """Load ground truth from expected.json in the task directory."""
    expected_path = task_path / "tests" / "expected.json"
    if not expected_path.exists():
        return None
    
    try:
        with open(expected_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {expected_path}: {e}")
        return None

def normalize_value(val):
    if val is None:
        return ""
    return str(val).strip()

def main():
    parser = argparse.ArgumentParser(description="Recover Codex results from logs.")
    parser.add_argument("--trials-dir", type=Path, required=True, help="Directory containing trial runs.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save recovered JSONs.")
    parser.add_argument("--limit", type=int, default=None, help="Max trials to process (for testing).")
    args = parser.parse_args()

    if not args.trials_dir.exists():
        print(f"Trials directory not found: {args.trials_dir}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    success_count = 0
    
    trials = sorted([p for p in args.trials_dir.iterdir() if p.is_dir()])
    
    print(f"Found {len(trials)} trial directories.")

    for trial_dir in trials:
        if args.limit and count >= args.limit:
            break
            
        count += 1
        trial_name = trial_dir.name
        
        # 1. Locate logs
        codex_log = trial_dir / "agent" / "codex.txt"
        
        # 2. Parse Log
        prediction_data = parse_codex_log(codex_log)
        if not prediction_data:
            print(f"[{trial_name}] No valid JSON found in {codex_log}")
            continue

        # 3. Locate Config to find Task
        config_path = trial_dir / "config.json"
        if not config_path.exists():
            print(f"[{trial_name}] No config.json found")
            continue
            
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except Exception:
            print(f"[{trial_name}] Failed to parse config.json")
            continue
            
        # The task path in config is relative to repo root, usually "out/harbor/..."
        # We need to resolve it relative to the current CWD (repo root)
        # config["task"]["path"] might be absolute or relative
        task_rel_path = config.get("task", {}).get("path")
        if not task_rel_path:
            print(f"[{trial_name}] No task path in config")
            continue
            
        # Try resolving relative to workspace
        workspace_root = Path("examples/harbor-workspace")
        task_path = (workspace_root / task_rel_path).resolve()
        
        if not task_path.exists():
             # Fallback: maybe it is relative to repo root?
             repo_root_path = Path(task_rel_path).resolve()
             if repo_root_path.exists():
                 task_path = repo_root_path
             else:
                 print(f"[{trial_name}] Task path does not exist: {task_path}")
                 continue

        # 4. Get Ground Truth
        ground_truth_data = get_task_ground_truth(task_path)
        if not ground_truth_data:
             print(f"[{trial_name}] No expected.json found at {task_path}")
             continue
             
        # 5. Match Predictions to Ground Truth
        # Structure of expected.json: {"ground_truth": [ {"material":..., "property_name":..., "property_value":...}, ... ]}
        # Structure of prediction_data: {"properties": [ {"material":..., "property_name":..., "value_string":...}, ... ]}
        
        gt_list = ground_truth_data.get("ground_truth", [])
        pred_list = prediction_data.get("properties", [])
        
        # We create one PBench record per GT item, looking for a matching Prediction
        pbench_records = []
        
        for gt_item in gt_list:
            mat = gt_item.get("material")
            prop = gt_item.get("property_name")
            
            # Find matching prediction
            # Simple match by material and property_name
            # Note: Material names might need normalization? Assuming exact match for now based on logs.
            
            match = None
            for p in pred_list:
                if p.get("material") == mat and p.get("property_name") == prop:
                    match = p
                    break
            
            pred_value = ""
            pred_unit = "" 
            response_details = {}
            
            if match:
                pred_value = match.get("value_string") or match.get("value") or ""
                # Some extraction output puts unit in a separate field, some in value string.
                # The simple 'value_string' schema usually implies the model puts it there.
                # If there's a unit field, grab it.
                pred_unit = match.get("unit", "")
                
            # Gather other details for the 'response' block
            response_details = {
                "pred": pred_value,
                "source_doi": match.get("source_dois") if match else [],
                "conditions": match.get("conditions") if match else {},
                "related_materials": match.get("related_materials") if match else []
            }
            
            record = {
                "refno": ground_truth_data.get("refno", "unknown"),
                "material": mat,
                "property_name": prop,
                "true": {
                    "value": gt_item.get("property_value"),
                    "unit": gt_item.get("property_unit", "") # GT usually doesn't have separated unit in this schema
                },
                "pred": {
                    "value": pred_value,
                    "unit": pred_unit
                },
                "rubric": gt_item.get("rubric"),
                "response": response_details,
                "metadata": {
                    "task": ground_truth_data.get("task", "precedent-search"),
                    "trial_id": trial_name,
                    "recovery_method": "log_mining"
                }

            }
            pbench_records.append(record)
            
        # 6. Save Result
        output_file = args.output_dir / f"{trial_name}__codex__gpt-5.1.json"
        
        try:
            with open(output_file, "w") as f:
                json.dump(pbench_records, f, indent=2)
            success_count += 1
            print(f"[{trial_name}] Recovered {len(pbench_records)} records -> {output_file.name}")
        except Exception as e:
             print(f"[{trial_name}] Error writing output: {e}")

    print(f"Finished. Recovered results for {success_count} / {count} processed trials.")

if __name__ == "__main__":
    main()
