"""
Combined script to score web search llm-formatted CSVs.
Combines logic from `convert_gemini_format.py` (web search -> Enriched) and `rescore_results.py` (Enriched -> Final Score).

Usage:
    # Batch mode (scores all CSVs in input_dir):
    python score_web_search_csv_format.py 

    # Single file mode:
    python score_web_search_csv_format.py --input <input_csv> --output <output_csv>
"""

import csv
import json
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tabulate import tabulate
import re
from collections import defaultdict

def parse_list_string(s):
    try:
        s = str(s).strip()
        if not s:
            return []
        if not s.startswith('['):
            return [s]
        return json.loads(s)
    except:
        return [s]

def extract_source_doi_info(source_doi: dict | str) -> dict[str, str]:
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

def score_single_file(input_path: Path, output_path: Path, output_detailed_path: Path | None = None):
    print(f"Reading from {input_path}")

    # --- Step 1: Conversion / Enrichment Logic ---
    enriched_rows = []
    
    # Use pandas to read the CSV - more robust to quoting/encoding issues
    try:
        df_in = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading CSV with pandas: {e}")
        return

    # Normalize columns to lower case/strip
    df_in.columns = [str(c).strip().lower() for c in df_in.columns]
    
    for row in df_in.itertuples(index=False):
        # Access fields safely
        try:
             # Basic fields - assume standard names exist or are close
             material = getattr(row, 'material', '')
             if pd.isna(material): continue
             
             pred_sc = str(getattr(row, 'is_superconducting', ''))
             if pd.isna(pred_sc): pred_sc = ''
             
             gt_sc = str(getattr(row, 'gt_is_superconducting', ''))
             if pd.isna(gt_sc): gt_sc = ''
             
             pred_tc = str(getattr(row, 'tc_values', ''))
             if pd.isna(pred_tc): pred_tc = ''
             
             gt_tc = str(getattr(row, 'gt_highest_tc', ''))
             if pd.isna(gt_tc): gt_tc = ''
             
             pred_tcn = str(getattr(row, 'tcn_values', ''))
             if pd.isna(pred_tcn): pred_tcn = ''
             
             gt_tcn = str(getattr(row, 'gt_lowest_tcn', ''))
             if pd.isna(gt_tcn): gt_tcn = ''

             error_val = str(getattr(row, 'error', ''))
             if pd.isna(error_val) or error_val.lower() == 'nan': error_val = ''

             sources_str = str(getattr(row, 'sources', '[]'))
             if pd.isna(sources_str): sources_str = '[]'
             
             queries_str = str(getattr(row, 'web_search_queries', '[]'))
             if pd.isna(queries_str): queries_str = '[]'
             
             # Metrics
             def get_metric(name, default=0):
                 val = getattr(row, name, default)
                 if pd.isna(val) or val == '': return default
                 if default is None:
                     try: return float(val)
                     except: return default
                 try: return type(default)(val)
                 except: return default

             web_search_num_tool_calls = get_metric('web_search_num_tool_calls', None)
             total_tokens = get_metric('total_tokens', 0)
             time_taken_seconds = get_metric('time_taken_seconds', 0.0)
             prompt_tokens = get_metric('prompt_tokens', 0)
             completion_tokens = get_metric('completion_tokens', 0)
             cached_tokens = get_metric('cached_tokens', 0)
             thinking_tokens = get_metric('thinking_tokens', 0)
             
             sources_list = parse_list_string(sources_str)
             queries_list = parse_list_string(queries_str)
             
             num_web_queries = len(queries_list)
             # Keep tool_calls aliased for the CSV output, but it might be None now if web_search_num_tool_calls is None
             tool_calls = web_search_num_tool_calls if web_search_num_tool_calls is not None else 0
             
        except Exception as e:
            print(f"Error processing row: {e}")
            continue

        # Generate unique ID for grouping
        trial_id = material
        job_id = "web-search"
        
        # Helper to generate source columns
        source_cols = {}
        source_cols.update(flatten_source_dois(sources_list, "is_superconducting"))
        source_cols.update(flatten_source_dois(sources_list, "tc"))
        source_cols.update(flatten_source_dois(sources_list, "tcn"))

        # 1. is_superconducting score
        sc_score = 1.0 if pred_sc.strip().lower() == gt_sc.strip().lower() else 0.0
        
        row_dict = {
            'metadata_trial_id': trial_id,
            'job_id': job_id,
            'material': material,
            'property_name': 'is_superconducting',
            'property_value': gt_sc,
            'response': pred_sc,
            'score': sc_score,
            'tool_calls': tool_calls,
            'web_search_num_tool_calls': web_search_num_tool_calls,
            'num_web_queries': num_web_queries,
            'total_tokens': total_tokens,
            'time_taken_seconds': time_taken_seconds,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'cached_tokens': cached_tokens,
            'thinking_tokens': thinking_tokens,
            'error': error_val
        }
        row_dict.update(source_cols)
        enriched_rows.append(row_dict)

        # 2. Tc
        if gt_sc.strip().lower() == 'yes':
            clean_gt_tc = gt_tc.strip()
            clean_pred_tc_list = parse_list_string(pred_tc)
            
            tc_score = 0.0
            for p in clean_pred_tc_list:
                if str(p).strip() == clean_gt_tc:
                    tc_score = 1.0
                    break
                try:
                    val_p = float(p)
                    val_gt = float(clean_gt_tc) if clean_gt_tc else 0.0
                    max_val = max(abs(val_p), abs(val_gt))
                    if max_val == 0:
                        tc_score = 1.0
                        break
                    relative_diff = abs(val_p - val_gt) / max_val
                    if relative_diff <= 0.001: 
                            tc_score = 1.0
                            break
                except:
                    pass
            
            row_dict = {
                'metadata_trial_id': trial_id,
                'job_id': job_id,
                'material': material,
                'property_name': 'tc',
                'property_value': clean_gt_tc,
                'response': pred_tc,
                'score': tc_score,
                'tool_calls': tool_calls,
                'web_search_num_tool_calls': web_search_num_tool_calls,
                'num_web_queries': num_web_queries,
                'total_tokens': total_tokens,
                'time_taken_seconds': time_taken_seconds,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'cached_tokens': cached_tokens,
                'thinking_tokens': thinking_tokens,
                'error': error_val
            }
            row_dict.update(source_cols)
            enriched_rows.append(row_dict)

        # 3. Tcn
        elif gt_sc.strip().lower() == 'no':
            clean_gt_tcn = gt_tcn.strip()
            clean_pred_tcn_list = parse_list_string(pred_tcn)
            
            tcn_score = 0.0
            for p in clean_pred_tcn_list:
                if str(p).strip() == clean_gt_tcn:
                    tcn_score = 1.0
                    break
                try:
                    val_p = float(p)
                    val_gt = float(clean_gt_tcn) if clean_gt_tcn else 0.0
                    max_val = max(abs(val_p), abs(val_gt))
                    if max_val == 0:
                        tcn_score = 1.0
                        break
                    relative_diff = abs(val_p - val_gt) / max_val
                    if relative_diff <= 0.001: 
                            tcn_score = 1.0
                            break
                except:
                    pass
            
            row_dict = {
                'metadata_trial_id': trial_id,
                'job_id': job_id,
                'material': material,
                'property_name': 'tcn',
                'property_value': clean_gt_tcn,
                'response': pred_tcn,
                'score': tcn_score,
                'tool_calls': tool_calls,
                'web_search_num_tool_calls': web_search_num_tool_calls,
                'num_web_queries': num_web_queries,
                'total_tokens': total_tokens,
                'time_taken_seconds': time_taken_seconds,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'cached_tokens': cached_tokens,
                'thinking_tokens': thinking_tokens,
                'error': error_val
            }
            row_dict.update(source_cols)
            enriched_rows.append(row_dict)

    print(f"Processed {len(enriched_rows)} enriched properties.")
    
    # --- Step 2: Aggregation / Rescoring Logic ---
    df = pd.DataFrame(enriched_rows)
    # Ensure scores are numeric
    df['score'] = pd.to_numeric(df['score'], errors='coerce').fillna(0.0)

    # Save Detailed CSV if requested
    if output_detailed_path:
        df.to_csv(output_detailed_path, index=False)
        print(f"Saved detailed results to {output_detailed_path}")
    
    final_rows = []
    
    for trial_id, group in df.groupby('metadata_trial_id'):
        first_row = group.iloc[0]
        material = first_row['material']
        job_id = first_row['job_id']
        
        # Check for error first
        error_val = ""
        if 'error' in first_row:
             error_val = str(first_row['error']).strip()
             if error_val.lower() == 'nan': error_val = ""
        
        # Scoring Logic
        sc_row = group[group['property_name'] == 'is_superconducting']
        
        final_score = 0
        score_category = "0 - Incorrect"
        
        if error_val:
            final_score = 0
            # User requested "0 - Error" specifically for errors
            score_category = "0 - Error"
            
        elif sc_row.empty:
            final_score = 0
            score_category = "0 - Error"
        else:
            sc_score = sc_row.iloc[0]['score']
            sc_gt = str(sc_row.iloc[0]['property_value']).strip()
            pred_response = str(sc_row.iloc[0].get('response', '')).strip()
            
            if pred_response.lower() == 'unknown':
                final_score = 0
                score_category = "0 - Unknown"
            elif sc_score < 0.99:
                final_score = 0
                score_category = "0 - Wrong Class"
            else:
                # Classification Correct
                if sc_gt.lower() == 'yes':
                    tc_row = group[group['property_name'] == 'tc']
                    if tc_row.empty:
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
                    tcn_row = group[group['property_name'] == 'tcn']
                    if tcn_row.empty:
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
                    final_score = 0
        
        # Build Final Record
        final_record = {
            'metadata_trial_id': trial_id,
            'job_id': job_id,
            'material': material,
            'score_012': final_score,
            'score_category': score_category,
            'error': error_val,
            # Metrics placeholders (compat with compute_metrics.py)
            'total_tokens': first_row.get('total_tokens', 0),
            'tool_calls': first_row.get('tool_calls', 0),
            'web_search_num_tool_calls': first_row.get('web_search_num_tool_calls', None), # Keep as None if originally None
            'num_web_queries': first_row.get('num_web_queries', 0),
            'time_taken_seconds': first_row.get('time_taken_seconds', 0.0),
            'prompt_tokens': first_row.get('prompt_tokens', 0),
            'completion_tokens': first_row.get('completion_tokens', 0),
            'cached_tokens': first_row.get('cached_tokens', 0),
            'thinking_tokens': first_row.get('thinking_tokens', 0)
        }
        
        # Add citation columns from first_row (they are the same for the whole trial)
        for col in first_row.index:
            if '_source_' in col:
                final_record[col] = first_row[col]

        final_rows.append(final_record)

    # Save Final CSV
    df_final = pd.DataFrame(final_rows)
    print(f"Scored {len(df_final)} trials.")
    df_final.to_csv(output_path, index=False)
    print(f"Saved final results to {output_path}")

    print(f"Saved final results to {output_path}")

    # Calculate metrics
    summary_stats = compute_and_print_metrics(df_final, input_path.name)
    return summary_stats, df_final

def compute_and_print_metrics(df_final, label_name):
    """
    Calculates distributions, averages, SEM, coverage, precision, accuracy.
    Prints tables to stdout.
    Returns summary_stats dict.
    """
    summary_stats = {'filename': label_name}

    # Print tables
    if 'score_012' in df_final.columns:
        print("\nScore Distribution (0-1-2):")
        dist = df_final['score_012'].value_counts().sort_index().reset_index()
        dist.columns = ['Score', 'Count']
        print(tabulate(dist, headers='keys', tablefmt='pretty', showindex=False))
        
        # Add to summary
        for index, row in dist.iterrows():
            summary_stats[f"score_{row['Score']}_count"] = row['Count']
    
    if 'score_category' in df_final.columns:
        print("\nScore Category Breakdown:")
        cat_dist = df_final['score_category'].value_counts().sort_index().reset_index()
        cat_dist.columns = ['Category', 'Count']
        print(tabulate(cat_dist, headers='keys', tablefmt='pretty', showindex=False))

        # Add to summary - Ensure all categories exist even if count is 0
        all_categories = [
            '0 - Error', '0 - Unknown', '0 - Wrong Class',
            '1 - Right Class, Missing Value', '1 - Right Class, Wrong Value',
            '2 - Perfect'
        ]
        for cat in all_categories:
            count = len(df_final[df_final['score_category'] == cat])
            summary_stats[f"category_{cat}_count"] = count
        
    # Calculate Averages
    token_metrics = [
        'total_tokens', 'prompt_tokens', 'completion_tokens', 
        'cached_tokens', 'thinking_tokens'
    ]
    other_metrics = [
        'web_search_num_tool_calls', 'num_web_queries', 
        'time_taken_seconds'
    ]
    
    # Helper to print transposed table AND collect stats
    def print_and_collect_metric_table(metric_list, title):
        headers = []
        values = []
        for metric in metric_list:
            if metric in df_final.columns:
                try:
                    # mean() and sem() ignore NaNs by default
                    series = df_final[metric]
                    mean_val = series.mean()
                    sem_val = series.sem()
                    
                    if pd.notna(mean_val):
                        headers.append(metric)
                        if pd.notna(sem_val):
                            values.append(f"{mean_val:.2f} ± {sem_val:.2f}")
                            summary_stats[f"sem_{metric}"] = sem_val
                        else:
                            values.append(f"{mean_val:.2f}")
                            summary_stats[f"sem_{metric}"] = 0.0 # Single value or constant has 0 SEM
                            
                        summary_stats[f"avg_{metric}"] = mean_val
                except:
                    pass
        
        if headers:
            print(f"\n{title}:")
            print(tabulate([values], headers=headers, tablefmt='pretty'))

    print_and_collect_metric_table(token_metrics, "Average Token Metrics (± SEM)")
    print_and_collect_metric_table(other_metrics, "Average Tool & Time Metrics (± SEM)")

    # --- New Metrics: Success Rate, Coverage & Precision ---
    total_trials = len(df_final)
    # Count specific categories
    # Identify errors by the error column OR the category '0 - Error'
    # Ensure error column is treated as string
    df_final['error'] = df_final['error'].fillna('').astype(str)

    # Rows are errors if they have an error message OR are categorized as '0 - Error'
    is_error = (df_final['error'].str.strip() != '') | (df_final['score_category'] == '0 - Error')
    num_errors = len(df_final[is_error])

    # Success Rate: Percentage of trials without errors
    success_rate = 0.0
    sem_success_rate = 0.0
    if total_trials > 0:
        success_rate = (total_trials - num_errors) / total_trials
        # Success rate SEM = sqrt(p(1-p)/n)
        sem_success_rate = (success_rate * (1 - success_rate) / total_trials) ** 0.5

    summary_stats['success_rate'] = success_rate
    summary_stats['sem_success_rate'] = sem_success_rate

    print(f"\nSuccess Rate: {success_rate:.2%} ± {sem_success_rate:.2%} ({total_trials - num_errors}/{total_trials})")

    num_unknowns = len(df_final[df_final['score_category'] == '0 - Unknown'])

    valid_denom = total_trials - num_errors
    answered_count = valid_denom - num_unknowns
    
    coverage = 0.0
    if valid_denom > 0:
        coverage = answered_count / valid_denom
    
    summary_stats['coverage'] = coverage
    
    # Coverage SEM = sqrt(p(1-p)/n)
    sem_coverage = 0.0
    if valid_denom > 0:
        sem_coverage = (coverage * (1 - coverage) / valid_denom) ** 0.5
    summary_stats['sem_coverage'] = sem_coverage
    
    print(f"\nCoverage: {coverage:.2%} ± {sem_coverage:.2%} ({answered_count}/{valid_denom})")

    # Accuracy: Correct / Valid Trials (Unknowns count as 0)
    # Strict: Score 2
    # Loose: Score 1 or 2
    num_strict = len(df_final[df_final['score_012'] == 2])
    num_loose = len(df_final[df_final['score_012'] >= 1])
    
    accuracy_strict = 0.0
    accuracy_loose = 0.0
    sem_accuracy_strict = 0.0
    sem_accuracy_loose = 0.0
    
    if valid_denom > 0:
        accuracy_strict = num_strict / valid_denom
        accuracy_loose = num_loose / valid_denom
        
        # Accuracy SEM = sqrt(p(1-p)/n)
        sem_accuracy_strict = (accuracy_strict * (1 - accuracy_strict) / valid_denom) ** 0.5
        sem_accuracy_loose = (accuracy_loose * (1 - accuracy_loose) / valid_denom) ** 0.5
        
    summary_stats['accuracy_strict'] = accuracy_strict
    summary_stats['sem_accuracy_strict'] = sem_accuracy_strict
    
    summary_stats['accuracy_loose'] = accuracy_loose
    summary_stats['sem_accuracy_loose'] = sem_accuracy_loose
    
    print(f"Accuracy (Strict): {accuracy_strict:.2%} ± {sem_accuracy_strict:.2%} ({num_strict}/{valid_denom})")
    print(f"Accuracy (Loose):  {accuracy_loose:.2%} ± {sem_accuracy_loose:.2%} ({num_loose}/{valid_denom})")

    # Precision: Score distribution on answered trials only
    # Answered = No Error AND Not Unknown
    answered_df = df_final[
        (~is_error) & 
        (df_final['score_category'] != '0 - Unknown')
    ]
    
    if not answered_df.empty and 'score_012' in answered_df.columns:
        print("\nPrecision Distribution (Answered Trials Only):")
        prec_dist = answered_df['score_012'].value_counts().sort_index().reset_index()
        prec_dist.columns = ['Score', 'Count']
        print(tabulate(prec_dist, headers='keys', tablefmt='pretty', showindex=False))
        
        # Add to summary - Ensure 0, 1, 2 keys exist even if count is 0 for consistency
        for s in [0, 1, 2]:
            count = len(answered_df[answered_df['score_012'] == s])
            summary_stats[f"precision_score_{s}_count"] = count
    
    return summary_stats

def compute_aggregate_metrics_with_run_sem(batch_stats: list[dict], model_name: str, agg_label: str) -> dict:
    """
    Compute aggregate statistics with run-to-run SEM.

    Extracts metrics from individual runs for the specified model,
    then computes mean and SEM across runs.
    """
    # Find all runs for this model
    run_stats = [s for s in batch_stats if model_name in s.get('filename', '')]

    if not run_stats:
        return {'filename': agg_label}

    agg_stats = {'filename': agg_label}

    # Metrics to aggregate - map from run-level key to aggregate key
    metrics_to_agg = {
        'success_rate': 'success_rate',
        'coverage': 'coverage',
        'accuracy_strict': 'accuracy_strict',
        'accuracy_loose': 'accuracy_loose',
        'avg_total_tokens': 'avg_total_tokens',
        'avg_prompt_tokens': 'avg_prompt_tokens',
        'avg_completion_tokens': 'avg_completion_tokens',
        'avg_cached_tokens': 'avg_cached_tokens',
        'avg_thinking_tokens': 'avg_thinking_tokens',
        'avg_web_search_num_tool_calls': 'avg_web_search_num_tool_calls',
        'avg_num_web_queries': 'avg_num_web_queries',
        'avg_time_taken_seconds': 'avg_time_taken_seconds'
    }

    # Map to determine SEM column name (strip "avg_" prefix for token metrics)
    sem_column_map = {
        'success_rate': 'sem_success_rate',
        'coverage': 'sem_coverage',
        'accuracy_strict': 'sem_accuracy_strict',
        'accuracy_loose': 'sem_accuracy_loose',
        'avg_total_tokens': 'sem_total_tokens',
        'avg_prompt_tokens': 'sem_prompt_tokens',
        'avg_completion_tokens': 'sem_completion_tokens',
        'avg_cached_tokens': 'sem_cached_tokens',
        'avg_thinking_tokens': 'sem_thinking_tokens',
        'avg_web_search_num_tool_calls': 'sem_web_search_num_tool_calls',
        'avg_num_web_queries': 'sem_num_web_queries',
        'avg_time_taken_seconds': 'sem_time_taken_seconds'
    }

    for run_metric, agg_metric in metrics_to_agg.items():
        values = [s[run_metric] for s in run_stats if run_metric in s and pd.notna(s[run_metric])]

        if values:
            mean_val = np.mean(values)
            if len(values) > 1:
                # SEM = std / sqrt(n) with Bessel's correction (ddof=1)
                sem_val = np.std(values, ddof=1) / np.sqrt(len(values))
            else:
                sem_val = 0.0

            agg_stats[agg_metric] = mean_val
            agg_stats[sem_column_map[run_metric]] = sem_val

            print(f"{agg_metric}: {mean_val:.4f} ± {sem_val:.4f}")

    # Also aggregate score counts (but don't add sem_ columns to avoid duplication)
    # These are counts, so we just average them across runs
    count_keys = [k for k in run_stats[0].keys() if k.startswith(('score_', 'category_', 'precision_score_'))]
    for key in count_keys:
        values = [s[key] for s in run_stats if key in s and pd.notna(s[key])]
        if values:
            mean_val = np.mean(values)
            agg_stats[key] = mean_val

    # Calculate and print Precision Percentages (0, 1, 2)
    # Based on precision_score_X_count
    score_keys = [0, 1, 2]
    
    print("\nAggregated Precision Percentages (Answered Trials Only):")
    for s in score_keys:
        key_count = f"precision_score_{s}_count"
        pct_values = []
        
        for run in run_stats:
            # Reconstruct total answered for this run
            total_answered = sum(run.get(f"precision_score_{k}_count", 0) for k in score_keys)
            if total_answered > 0:
                count = run.get(key_count, 0)
                pct_values.append(count / total_answered)
        
        if pct_values:
            mean_val = np.mean(pct_values)
            if len(pct_values) > 1:
                sem_val = np.std(pct_values, ddof=1) / np.sqrt(len(pct_values))
            else:
                sem_val = 0.0
            
            agg_stats[f"precision_score_{s}_pct"] = mean_val
            agg_stats[f"sem_precision_score_{s}_pct"] = sem_val
            print(f"Score {s}: {mean_val:.2%} ± {sem_val:.2%}")

    return agg_stats


def main():
    script_dir = Path(__file__).parent.absolute()
    
    parser = argparse.ArgumentParser(description='Score Anmol-formatted CSVs directly to final results.')
    parser.add_argument('--input', help='Path to input CSV or directory containing CSVs.')
    parser.add_argument('--output', help='Path to output final scored CSV. Only used if --input is a file.')
    parser.add_argument('--output-detailed', help='Path to output detailed scored CSV. Optional.')
    parser.add_argument('--input-dir', default='examples/harbor-workspace/jobs/llm-api',
                        help='Directory to scan for input CSVs if --input is not provided.')
    parser.add_argument('--output-dir', default='examples/harbor-workspace/out/harbor/precedent-search/scores',
                        help='Directory to save output CSVs if --input is not provided or --input is a directory.')
    
    args = parser.parse_args()

    # Resolve paths relative to script_dir if they are relative
    def resolve_path(p):
        if not p: return None
        path = Path(p)
        if path.is_absolute():
            return path
        return script_dir / path

    input_path = resolve_path(args.input)
    
    if input_path and input_path.is_file():
        # Single File Mode
        if not args.output:
            print("Error: --output is required when --input is a file.")
            sys.exit(1)
        output_path = resolve_path(args.output)
        output_detailed = resolve_path(args.output_detailed)
        score_single_file(input_path, output_path, output_detailed)
    
    else:
        # Batch Mode (either via directory in --input or via --input-dir)
        if input_path and input_path.is_dir():
            input_dir = input_path
        else:
            input_dir = resolve_path(args.input_dir)
            
        output_dir = resolve_path(args.output_dir)
        
        if not input_dir.exists():
            print(f"Error: Input directory {input_dir} does not exist.")
            sys.exit(1)
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        csv_files = list(input_dir.glob('*.csv'))
        if not csv_files:
            print(f"No CSV files found in {input_dir}")
            return

        print(f"Found {len(csv_files)} CSV files in {input_dir}")
        
        batch_stats = []
        model_groups = defaultdict(list)
        
        for csv_file in csv_files:
            output_filename = f"{csv_file.stem}_scored.csv"
            output_detailed_filename = f"{csv_file.stem}_scored_detailed.csv"
            
            output_path = output_dir / output_filename
            output_detailed_path = output_dir / output_detailed_filename
            
            print(f"\nProcessing {csv_file.name} -> {output_filename}")
            try:
                stats, df_scored = score_single_file(csv_file, output_path, output_detailed_path)
                if stats:
                    batch_stats.append(stats)
                    
                    # Extract model name for grouping
                    # Expecting pattern: ...model=MODELNAME__...
                    match = re.search(r'model=(.+?)__', csv_file.name)
                    if match:
                        model_name = match.group(1)
                        model_groups[model_name].append(df_scored)
            except Exception as e:
                print(f"Error processing {csv_file.name}: {e}")
                import traceback
                traceback.print_exc()

        # Process Aggregates
        print("\n--- Computing Aggregates ---")
        for model_name, dfs in model_groups.items():
            if len(dfs) > 1:
                print(f"\nAggregating {len(dfs)} runs for model: {model_name}")

                # Compute run-to-run SEM
                agg_label = f"AGGREGATE_model={model_name}"
                agg_stats = compute_aggregate_metrics_with_run_sem(
                    batch_stats, model_name, agg_label
                )
                batch_stats.append(agg_stats)

        if batch_stats:
            summary_df = pd.DataFrame(batch_stats)
            # Reorder columns to make filename first
            cols = ['filename'] + [c for c in summary_df.columns if c != 'filename']
            summary_df = summary_df[cols]
            
            summary_output_path = output_dir / "batch_summary.csv"
            summary_df.to_csv(summary_output_path, index=False)
            print(f"\nSaved batch summary to {summary_output_path}")

if __name__ == "__main__":
    main()
