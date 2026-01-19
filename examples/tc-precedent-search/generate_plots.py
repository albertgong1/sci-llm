import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

import argparse

def main():
    parser = argparse.ArgumentParser(description='Generate plots from scored results.')
    parser.add_argument('--tag', type=str, default='012', help='Tag suffix for input/output files (default: 012)')
    parser.add_argument('--base-dir', type=Path, default=Path('examples/harbor-workspace/out/harbor/precedent-search'), help='Base directory for output')
    
    args = parser.parse_args()
    
    tag = args.tag
    base_dir = args.base_dir
    
    # Construct filenames based on tag
    # Input: scored_results_{tag}.csv
    # Output: plots/scatter_tokens_vs_score_{tag}.png, etc.
    
    scores_filename = f'scored_results_{tag}.csv'
    scores_path = base_dir / 'scores' / scores_filename
    output_dir = base_dir / 'analysis/plots'
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Data
    print(f"Loading {scores_path}...")
    try:
        df_trials = pd.read_csv(scores_path)
    except FileNotFoundError:
        print(f"Error: Could not find {scores_path}")
        return

    num_trials = len(df_trials)
    print(f"Loaded {num_trials} trials.")

    # Set style
    sns.set_theme(style="whitegrid")
    
    # Check if 'score_012' exists, if not maybe compute it or fallback to 'score'
    # The user example used 'score_012', assuming the input csv has it. 
    # score_task.py outputs 'score' (float usually). 
    # If the user script relies on 'score_012', we might need to verify if that column exists. 
    # Standard score_task.py outputs 'score'. 
    # But wait, the user's "scored_results_012.csv" might have been special.
    # Let's assume standard 'score' column is what we have from score_task.py.
    # But if the user wants 0-1-2 buckets, we might need a transformer or check bounds.
    # Re-reading generate_plots.py original code:
    # sns.scatterplot(data=df_trials, x='total_tokens', y='score_012', ...)
    # It seems the previous '012' file has a specific column 'score_012'.
    # If score_task.py produces just 'score', we should use that.
    
    y_col = 'score'
    if 'score_012' in df_trials.columns:
        y_col = 'score_012'
    
    # Also total_tokens might need to be enriched first?
    # The user's workflow implies: score -> enrich (add tokens) -> plot.
    # So we should assume input has 'total_tokens'.
    
    if 'total_tokens' not in df_trials.columns and 'n_input_tokens' in df_trials.columns:
         # Calculate total tokens if separate columns exist
         df_trials['total_tokens'] = df_trials['n_input_tokens'] + df_trials['n_cache_tokens'] + df_trials.get('n_output_tokens', 0)

    plotting_possible = True
    if 'total_tokens' not in df_trials.columns:
        print("Warning: 'total_tokens' column missing. Skipping plots involving tokens.")
        plotting_possible = False
        
    # --- Plot 1: Scatter Plot of Trial Tokens vs Trial Score (0-2) ---
    if plotting_possible:
        plt.figure(figsize=(10, 6))

        # Add some jitter to x and y to see points better if they overlap
        # We jitter Y manually because they are integers 0, 1, 2
        # But seaborn scatterplot handles 'jitter' natively only in stripping plots usually,
        # and we want true scatter.
        # Let's just plot it standard, maybe adjust alpha.

        sns.scatterplot(
            data=df_trials,
            x='total_tokens',
            y=y_col,
            alpha=0.6,
            s=100
        )

        # Set consistent axes for comparability across plots
        plt.xlim(0, 10_000_000)
        plt.ylim(-0.2, 2.2)  # Scores are 0, 1, 2

        plt.title(f'Token Cost vs. Score (N={num_trials})', fontsize=15)
        plt.xlabel('Total Tokens', fontsize=12)
        plt.ylabel('Score', fontsize=12)

        plt.tight_layout()
        output_file_scatter = output_dir / f'scatter_tokens_vs_score_{tag}.png'
        plt.savefig(output_file_scatter)
        print(f"Saved {output_file_scatter}")
        plt.close()

        # --- Plot 2: Histogram of Token Cost ---
        plt.figure(figsize=(10, 6))
        ax = sns.histplot(df_trials['total_tokens'], kde=False, bins=20, binrange=(0, 10_000_000))

        # Set consistent x-axis for comparability across plots
        plt.xlim(0, 10_000_000)

        plt.title(f'Distribution of Token Cost (N={num_trials})', fontsize=15)
        plt.xlabel('Total Tokens', fontsize=12)
        plt.ylabel('Count', fontsize=12)

        # Add count labels to top of bars
        for container in ax.containers:
            ax.bar_label(container)

        plt.tight_layout()
        output_file_hist_tokens = output_dir / f'hist_token_cost_{tag}.png'
        plt.savefig(output_file_hist_tokens)
        print(f"Saved {output_file_hist_tokens}")
        plt.close()

    # --- Plot 3: Distribution of Scores ---
    plt.figure(figsize=(10, 6))

    # Check distinct values
    # If standard score [0,1], treat as categorical or bins?
    # If 0,1,2, verify unique values

    ax = sns.countplot(data=df_trials, x=y_col, color='steelblue', order=[0, 1, 2])

    # Set consistent x-axis for comparability across plots (scores 0, 1, 2)
    plt.xlim(-0.5, 2.5)

    plt.title(f'Distribution of Scores (N={num_trials})', fontsize=15)
    plt.xlabel('Score Category', fontsize=12)
    plt.ylabel('Number of Trials', fontsize=12)

    # Add count labels
    for container in ax.containers:
        ax.bar_label(container)

    plt.tight_layout()
    output_file_hist_scores = output_dir / f'hist_scores_{tag}.png'
    plt.savefig(output_file_hist_scores)
    print(f"Saved {output_file_hist_scores}")
    plt.close()
    
    print("All plots generated successfully.")

if __name__ == "__main__":
    main()
