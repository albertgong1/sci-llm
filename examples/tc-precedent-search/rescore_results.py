import pandas as pd
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Rescore results using 0-1-2 state-dependent logic.')
    parser.add_argument('--input', required=True, help='Path to input enriched CSV')
    parser.add_argument('--output', required=True, help='Path to output rescored CSV')
    
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    print(f"Reading from {input_path}")
    df = pd.read_csv(input_path)
    
    # Ensure scores are numeric
    df['score'] = pd.to_numeric(df['score'], errors='coerce').fillna(0.0)
    
    # Group by trial
    # We need to process each group to determine the final score
    
    rescored_rows = []
    
    for trial_id, group in df.groupby('metadata_trial_id'):
        # Extract basic info from the first row of the group (assuming consistent per trial)
        first_row = group.iloc[0]
        material = first_row.get('material', 'Unknown')
        job_id = first_row.get('job_id', '') # Preserve job_id
        
        # Token counts (first row is fine as they are enriched per trial)
        n_input = first_row.get('n_input_tokens', 0)
        n_cache = first_row.get('n_cache_tokens', 0)
        n_output = first_row.get('n_output_tokens', 0)
        
        total_tokens = float(n_input) + float(n_cache) + float(n_output) if pd.notna(n_input) and pd.notna(n_output) else 0
        
        # Scoring Logic
        # 1. Check is_superconducting score
        sc_row = group[group['property_name'] == 'is_superconducting']
        
        final_score = 0
        score_category = "0 - Wrong Class"
        
        if sc_row.empty:
            # Should not happen typically, but if classification row missing, assume fail
            print(f"Warning: Trial {trial_id} missing is_superconducting row.")
            final_score = 0
        else:
            sc_score = sc_row.iloc[0]['score']
            sc_gt = str(sc_row.iloc[0]['property_value']).strip() # "Yes" or "No"
            # Check prediction for 'unknown' (using response field which holds the pred string)
            pred_response = str(sc_row.iloc[0].get('response', '')).strip()
            
            if pred_response.lower() == 'unknown':
                final_score = 0
                score_category = "0 - Unknown"
            elif sc_score < 0.99: # Allowing float tol, basically if not 1.0
                final_score = 0
                score_category = "0 - Wrong Class"
            else:
                # Classification Correct. Check value.
                if sc_gt.lower() == 'yes':
                    # Check 'tc'
                    tc_row = group[group['property_name'] == 'tc']
                    if tc_row.empty:
                        # Missing Tc row -> Treat as wrong value (1)
                        final_score = 1
                        score_category = "1 - Right Class, Missing Value"
                    else:
                        tc_score = tc_row.iloc[0]['score']
                        if tc_score > 0.99:
                            final_score = 2
                            score_category = "2 - Perfect"
                        else:
                            final_score = 1
                            score_category = "1 - Right Class, Wrong Value"
                            
                elif sc_gt.lower() == 'no':
                    # Check 'tcn'
                    tcn_row = group[group['property_name'] == 'tcn']
                    if tcn_row.empty:
                        # Missing Tcn row -> Treat as wrong value (1)
                        final_score = 1
                        score_category = "1 - Right Class, Missing Value"
                    else:
                        tcn_score = tcn_row.iloc[0]['score']
                        if tcn_score > 0.99:
                            final_score = 2
                            score_category = "2 - Perfect"
                        else:
                            final_score = 1
                            score_category = "1 - Right Class, Wrong Value"
                else:
                    print(f"Warning: Unexpected GT for is_superconducting: {sc_gt} in trial {trial_id}")
                    final_score = 0
                    
        rescored_rows.append({
            'metadata_trial_id': trial_id,
            'job_id': job_id,
            'material': material,
            'score_012': final_score,
            'score_category': score_category,
            'score_category': score_category,
            'n_input_tokens': n_input,
            'n_cached_tokens': n_cache,
            'n_output_tokens': n_output,
            'total_tokens': total_tokens,
        })
        
    # Create DataFrame and save
    df_rescored = pd.DataFrame(rescored_rows)
    
    # Reorder columns
    expected_columns = [
        'metadata_trial_id', 'job_id', 'material', 'score_012', 'score_category', 
        'n_input_tokens', 'n_cached_tokens', 'n_output_tokens', 'total_tokens'
    ]
    # Ensure all expected columns are present (some might be missing logic if bugs exist, but here we constructed them)
    # Also keep any other columns that might have been added? 
    # The user asked for specific order for these. Let's just select these for now as per "final scored 012 csv" usually implies strictly these.
    # But to be safe, let's just make sure these are first.
    
    cols = expected_columns + [c for c in df_rescored.columns if c not in expected_columns]
    df_rescored = df_rescored[cols]
    print(f"Rescored {len(df_rescored)} trials.")
    
    print(f"Writing to {output_path}")
    df_rescored.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
