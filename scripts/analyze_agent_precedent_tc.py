f"""Analyze and visualize stats of LLM agent trajectory results for precedent search.

Loads scored results CSVs and extracts tool use counts from trajectory.json files.
Outputs summary statistics including mean ± stderr of tool use counts across runs.

Usage:
    uv run python scripts/analyze_agent_precedent_tc.py \
        precedent-search-agent-metadata/scored_results_tc-gemini-cli-run-{1,2,3}_detailed.csv \
        precedent-search-agent-metadata/scored_results_tc-codex-run-{1,2,3}_detailed.csv \
        precedent-search-agent-metadata/scored_results_tc-terminus-gemini-run-{1,2,3}_detailed.csv \
        precedent-search-agent-metadata/scored_results_tc-terminus-gpt-run-{1,2,3}_detailed.csv

"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate


# Agent log file mapping for log mining recovery methods
AGENT_LOG_FILES: dict[str, str] = {
    "gemini-cli": "gemini-cli.txt",
    "codex": "codex.txt",
    "terminus-gemini": None,  # uses episode-*/response.txt
    "terminus-gpt": None,  # uses episode-*/response.txt
}

# Base directory for agent metadata
METADATA_BASE_DIR = Path("precedent-search-agent-metadata")


def parse_agent_from_csv_path(csv_path: Path) -> str:
    """Extract agent name from CSV filename.

    Example: scored_results_tc-gemini-cli-run-1_detailed.csv -> gemini-cli
    """
    match = re.search(r"scored_results_tc-([a-z0-9-]+)-run-\d+", csv_path.stem)
    if match:
        return match.group(1)
    raise ValueError(f"Could not parse agent name from {csv_path}")


def get_task_run_dir(agent: str, job_id: str, metadata_trial_id: str) -> Path:
    """Construct the task run directory path."""
    return METADATA_BASE_DIR / f"tc-{agent}" / job_id / metadata_trial_id


def load_tool_use_counts_from_trajectory(trajectory_path: Path) -> int | None:
    """Load agent step count from trajectory.json.

    Counts len(tool_calls) in trajectory.json.
    Returns None if trajectory file doesn't exist or is malformed.
    """
    if not trajectory_path.exists():
        return None

    try:
        with open(trajectory_path) as f:
            data = json.load(f)

        # Count tool calls in trajectory.json
        # Each step may have multiple tool calls, so we sum them up
        tool_use_counts = sum(len(step.get("tool_calls", [])) for step in data["steps"])
        return tool_use_counts

    except (json.JSONDecodeError, KeyError):
        return None


def load_json_agent_completed_run(
    task_run_dir: Path, recovery_method: str, agent: str
) -> dict | None:
    """Load the agent's output for a completed run.

    Args:
        task_run_dir: Path to the task run directory
        recovery_method: How to find the agent's output (predictions_json, gemini_log_mining, etc.)
        agent: Agent name

    Returns:
        Parsed JSON output or None if not found
    """
    if recovery_method == "predictions_json":
        predictions_path = task_run_dir / "verifier" / "app_output" / "predictions.json"
        if predictions_path.exists():
            with open(predictions_path) as f:
                return json.load(f)

    elif recovery_method == "gemini_log_mining":
        log_path = task_run_dir / "agent" / "gemini-cli.txt"
        if log_path.exists():
            return {"log_path": str(log_path)}

    elif recovery_method == "codex_log_mining":
        log_path = task_run_dir / "agent" / "codex.txt"
        if log_path.exists():
            return {"log_path": str(log_path)}

    elif recovery_method == "terminus_log_mining":
        # Look at episode-*/response.txt starting from latest episode
        agent_dir = task_run_dir / "agent"
        if agent_dir.exists():
            episode_dirs = sorted(
                [d for d in agent_dir.iterdir() if d.is_dir() and d.name.startswith("episode-")],
                key=lambda x: int(x.name.split("-")[1]),
                reverse=True,
            )
            for episode_dir in episode_dirs:
                response_path = episode_dir / "response.txt"
                if response_path.exists():
                    return {"log_path": str(response_path)}

    return None


def add_tool_counts_to_df(df: pd.DataFrame, agent: str) -> pd.DataFrame:
    """Add n_tool_use_counts column to dataframe by reading trajectory.json files.

    Args:
        df: DataFrame with metadata_trial_id and job_id columns
        agent: Agent name

    Returns:
        DataFrame with n_tool_use_counts column added
    """
    # Get unique (job_id, metadata_trial_id) combinations
    # Each material has 3 rows (3 property_names), so we deduplicate
    unique_trials = df[["job_id", "metadata_trial_id"]].drop_duplicates()

    # Build mapping from (job_id, trial_id) -> n_tool_use_counts
    tool_use_counts_map: dict[tuple[str, str], int | None] = {}

    for _, row in unique_trials.iterrows():
        job_id = row["job_id"]
        trial_id = row["metadata_trial_id"]

        task_run_dir = get_task_run_dir(agent, job_id, trial_id)
        trajectory_path = task_run_dir / "agent" / "trajectory.json"

        n_tool_use_counts = load_tool_use_counts_from_trajectory(trajectory_path)
        tool_use_counts_map[(job_id, trial_id)] = n_tool_use_counts

    # Apply mapping to create new column
    df["n_tool_use_counts"] = df.apply(
        lambda row: tool_use_counts_map.get((row["job_id"], row["metadata_trial_id"])),
        axis=1,
    )

    return df


def compute_tool_use_counts_summary(dfs: list[pd.DataFrame], agents: list[str]) -> pd.DataFrame:
    """Compute mean ± stderr of tool use counts across job_ids for each agent.

    Args:
        dfs: List of DataFrames (one per CSV file)
        agents: List of agent names corresponding to each DataFrame

    Returns:
        DataFrame with agent step statistics
    """
    results: list[dict] = []

    # Group DataFrames by agent
    agent_dfs: dict[str, list[pd.DataFrame]] = {}
    for df, agent in zip(dfs, agents):
        if agent not in agent_dfs:
            agent_dfs[agent] = []
        agent_dfs[agent].append(df)

    for agent, agent_df_list in agent_dfs.items():
        # Get mean tool use counts per job_id (average across materials within each job)
        job_means: list[float] = []

        for df in agent_df_list:
            # Get unique materials and their step counts
            material_tool_use_counts = df.groupby("material")["n_tool_use_counts"].first()
            valid_tool_use_counts = material_tool_use_counts.dropna()

            if len(valid_tool_use_counts) > 0:
                job_means.append(valid_tool_use_counts.mean())

        if len(job_means) == 0:
            results.append({
                "agent": agent,
                "mean_tool_use_counts": None,
                "stderr_tool_use_counts": None,
                "n_jobs": 0,
            })
        else:
            mean_val = np.mean(job_means)
            stderr_val = np.std(job_means, ddof=1) / np.sqrt(len(job_means)) if len(job_means) > 1 else 0

            results.append({
                "agent": agent,
                "mean_tool_use_counts": mean_val,
                "stderr_tool_use_counts": stderr_val,
                "n_jobs": len(job_means),
            })

    return pd.DataFrame(results)


def format_mean_stderr(mean: float | None, stderr: float | None) -> str:
    """Format mean ± stderr as string."""
    if mean is None:
        return "N/A"
    if stderr is None or stderr == 0:
        return f"{mean:.1f}"
    return f"{mean:.1f} ± {stderr:.1f}"


def main() -> None:
    """Main entry point for the script."""
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Analyze LLM agent trajectory results for precedent search tasks"
    )
    parser.add_argument(
        "csv_paths",
        nargs="+",
        type=Path,
        help="Paths to scored results CSV files (e.g., scored_results_tc-gemini-cli-run-1_detailed.csv)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to save augmented CSV with n_tool_use_counts column",
    )
    args = parser.parse_args()

    # Load and process each CSV
    all_dfs: list[pd.DataFrame] = []
    all_agents: list[str] = []

    for csv_path in args.csv_paths:
        agent = parse_agent_from_csv_path(csv_path)
        print(f"Processing {csv_path.name} (agent: {agent})")

        df = pd.read_csv(csv_path)
        df = add_tool_counts_to_df(df, agent)

        all_dfs.append(df)
        all_agents.append(agent)

        # Optionally save augmented CSV
        if args.output_csv:
            output_path = args.output_csv.parent / f"{csv_path.stem}_with_steps.csv"
            df.to_csv(output_path, index=False)
            print(f"  Saved augmented CSV to {output_path}")

    # Compute and display summary statistics
    summary_df = compute_tool_use_counts_summary(all_dfs, all_agents)

    # Format for display
    display_df = pd.DataFrame({
        "Agent": summary_df["agent"],
        "Tool Calls (mean ± stderr)": summary_df.apply(
            lambda row: format_mean_stderr(row["mean_tool_use_counts"], row["stderr_tool_use_counts"]),
            axis=1,
        ),
    })

    print("\n## Tool Calls Summary\n")
    print(tabulate(display_df, headers="keys", tablefmt="github", showindex=False))


if __name__ == "__main__":
    main()
