import argparse
import csv
import json
import os
from pathlib import Path

def get_token_counts(trial_id, job_id, jobs_root):
    # Try constructing the path directly if job_id is available
    if job_id:
        trial_path = jobs_root / job_id / trial_id / 'result.json'
        if trial_path.exists():
            tokens = extract_tokens_from_json(trial_path)
            return tokens + (job_id,) # Return job_id we used
    
    # If not found or job_id missing, search recent job directories
    # Get all job directories, sort by name (which includes date/time) descending
    job_dirs = sorted([d for d in jobs_root.iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)
    
    for job_dir in job_dirs:
        trial_path = job_dir / trial_id / 'result.json'
        if trial_path.exists():
            tokens = extract_tokens_from_json(trial_path)
            # Found it in this job
            return tokens + (job_dir.name,)
            
    return None, None, None, None

def extract_tokens_from_json(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            agent_result = data.get('agent_result', {})
            return (
                agent_result.get('n_input_tokens'),
                agent_result.get('n_cache_tokens'),
                agent_result.get('n_output_tokens')
            )
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return None, None, None

def main():
    parser = argparse.ArgumentParser(description='Add token counts to scored results CSV.')
    parser.add_argument('--input', required=True, help='Path to input CSV file')
    parser.add_argument('--output', required=True, help='Path to output CSV file')
    parser.add_argument('--jobs-dir', default='examples/harbor-workspace/jobs', help='Path to jobs directory')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    jobs_root = Path(args.jobs_dir).resolve() # Use absolute path for safety
    
    if not jobs_root.exists():
        # fallback to trying relative to current working directory if generic default was used but not found
        if not jobs_root.is_absolute():
             jobs_root = Path(os.getcwd()) / args.jobs_dir
    
    print(f"Reading from {input_path}")
    print(f"Searching for jobs in {jobs_root}")

    new_columns = ['n_input_tokens', 'n_cache_tokens', 'n_output_tokens', 'job_id']
    
    rows = []
    fieldnames = []
    
    with open(input_path, 'r', newline='') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        # Check if columns already exist to avoid duplication if run multiple times
        for col in new_columns:
            if col not in fieldnames:
                fieldnames.append(col)
        
        for row in reader:
            trial_id = row.get('metadata_trial_id')
            job_id = row.get('job_id')
            
            if trial_id:
                # Initialize with None/Empty if not found
                n_input, n_cache, n_output, found_job_id = None, None, None, None
                
                # Only look up if we don't already have data (or overwriting? let's overwrite to be safe/fresh)
                n_input, n_cache, n_output, found_job_id = get_token_counts(trial_id, job_id, jobs_root)
                
                if n_input is not None:
                    row['n_input_tokens'] = n_input
                    row['n_cache_tokens'] = n_cache
                    row['n_output_tokens'] = n_output
                    row['job_id'] = found_job_id
                else:
                    print(f"Warning: Could not find result.json for trial {trial_id}")
            
            rows.append(row)

    print(f"Writing to {output_path}")
    with open(output_path, 'w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Done.")

if __name__ == "__main__":
    main()
