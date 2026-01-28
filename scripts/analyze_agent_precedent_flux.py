"""Analyze and visualize stats of LLM agent trajectory results for flux precedent search.

Loads scored results CSVs and extracts tool use counts from trajectory.json files.
Outputs summary statistics including mean ± stderr of tool use counts across runs.

Usage:
    uv run python scripts/analyze_agent_precedent_flux.py \
        precedent-search-agent-metadata/scored_results_flux-gemini-cli-run-{1,2,3}_detailed.csv \
        precedent-search-agent-metadata/scored_results_flux-codex-run-{1,2,3}_detailed.csv \
        precedent-search-agent-metadata/scored_results_flux-terminus-gemini-run-{1,2,3}_detailed.csv \
        precedent-search-agent-metadata/scored_results_flux-terminus-gpt-run-{1,2,3}_detailed.csv

"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

# Category labels for correctness breakdown
CATEGORY_INCORRECT = "Incorrect"
CATEGORY_CORRECT = "Correct"


# Base directory for agent metadata
METADATA_BASE_DIR = Path("precedent-search-agent-metadata")


def parse_agent_from_csv_path(csv_path: Path) -> str:
    """Extract agent name from CSV filename.

    Example: scored_results_flux-gemini-cli-run-1_detailed.csv -> gemini-cli
    """
    match = re.search(r"scored_results_flux-([a-z0-9-]+)-run-\d+", csv_path.stem)
    if match:
        return match.group(1)
    raise ValueError(f"Could not parse agent name from {csv_path}")


def get_task_run_dir(agent: str, job_id: str, metadata_trial_id: str) -> Path:
    """Construct the task run directory path."""
    return METADATA_BASE_DIR / f"flux-{agent}" / job_id / metadata_trial_id


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


def add_tool_counts_to_df(df: pd.DataFrame, agent: str) -> pd.DataFrame:
    """Add n_tool_use_counts column to dataframe by reading trajectory.json files.

    Args:
        df: DataFrame with metadata_trial_id and job_id columns
        agent: Agent name

    Returns:
        DataFrame with n_tool_use_counts column added
    """
    # Get unique (job_id, metadata_trial_id) combinations
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


def categorize_material(material_df: pd.DataFrame) -> str:
    """Categorize a material based on classification correctness.

    Args:
        material_df: DataFrame containing the row(s) for a single material

    Returns:
        One of: CATEGORY_INCORRECT, CATEGORY_CORRECT
    """
    # Get the is_grown_with_flux row
    is_flux_row = material_df[material_df["property_name"] == "is_grown_with_flux"]
    if is_flux_row.empty:
        return CATEGORY_INCORRECT

    is_flux_score = is_flux_row["score"].iloc[0]

    # Check classification correctness
    if is_flux_score == 1:
        return CATEGORY_CORRECT
    else:
        return CATEGORY_INCORRECT


def add_category_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add correctness_category column to dataframe.

    Args:
        df: DataFrame with material, property_name, score columns

    Returns:
        DataFrame with correctness_category column added
    """
    # Build mapping from material -> category
    category_map: dict[str, str] = {}

    for material in df["material"].unique():
        material_df = df[df["material"] == material]
        category_map[material] = categorize_material(material_df)

    df["correctness_category"] = df["material"].map(category_map)
    return df


def compute_breakdown_summary(
    dfs: list[pd.DataFrame], agents: list[str]
) -> pd.DataFrame:
    """Compute mean ± stderr of tool calls by correctness category for each agent.

    Args:
        dfs: List of DataFrames (one per CSV file)
        agents: List of agent names corresponding to each DataFrame

    Returns:
        DataFrame with columns: agent, category, mean_tool_calls, stderr_tool_calls, n_materials
    """
    results: list[dict] = []

    # Group DataFrames by agent
    agent_dfs: dict[str, list[pd.DataFrame]] = {}
    for df, agent in zip(dfs, agents):
        if agent not in agent_dfs:
            agent_dfs[agent] = []
        agent_dfs[agent].append(df)

    categories = [CATEGORY_INCORRECT, CATEGORY_CORRECT]

    for agent, agent_df_list in agent_dfs.items():
        for category in categories:
            # Collect tool counts for this category across all jobs
            all_tool_counts: list[float] = []

            for df in agent_df_list:
                # Get unique materials in this category
                cat_df = df[df["correctness_category"] == category]
                material_counts = cat_df.groupby("material")["n_tool_use_counts"].first()
                valid_counts = material_counts.dropna()
                all_tool_counts.extend(valid_counts.tolist())

            if len(all_tool_counts) == 0:
                results.append({
                    "agent": agent,
                    "category": category,
                    "mean_tool_calls": None,
                    "stderr_tool_calls": None,
                    "n_materials": 0,
                })
            else:
                mean_val = np.mean(all_tool_counts)
                stderr_val = (
                    np.std(all_tool_counts, ddof=1) / np.sqrt(len(all_tool_counts))
                    if len(all_tool_counts) > 1
                    else 0
                )
                results.append({
                    "agent": agent,
                    "category": category,
                    "mean_tool_calls": mean_val,
                    "stderr_tool_calls": stderr_val,
                    "n_materials": len(all_tool_counts),
                })

    return pd.DataFrame(results)


def plot_breakdown_bar_chart(breakdown_df: pd.DataFrame, output_path: Path) -> None:
    """Generate grouped bar chart of tool calls by correctness category.

    Args:
        breakdown_df: DataFrame from compute_breakdown_summary
        output_path: Path to save the chart
    """
    agents = list(breakdown_df["agent"].unique())
    categories = [CATEGORY_INCORRECT, CATEGORY_CORRECT]

    # Colors for each category
    category_colors = {
        CATEGORY_INCORRECT: "red",
        CATEGORY_CORRECT: "blue",
    }

    # Short labels for legend
    category_short = {
        CATEGORY_INCORRECT: "Incorrect",
        CATEGORY_CORRECT: "Correct",
    }

    x = np.arange(len(agents))
    width = 0.35
    n_categories = len(categories)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, cat in enumerate(categories):
        means = []
        stderrs = []
        for agent in agents:
            row = breakdown_df[(breakdown_df["agent"] == agent) & (breakdown_df["category"] == cat)]
            if row.empty or pd.isna(row["mean_tool_calls"].iloc[0]):
                means.append(0)
                stderrs.append(0)
            else:
                means.append(row["mean_tool_calls"].iloc[0])
                stderrs.append(row["stderr_tool_calls"].iloc[0])

        offset = (i - n_categories / 2 + 0.5) * width
        ax.bar(
            x + offset,
            means,
            width,
            label=category_short[cat],
            color=category_colors[cat],
            yerr=stderrs,
            capsize=3,
        )

    ax.set_xlabel("Agent")
    ax.set_ylabel("Tool Calls (mean ± stderr)")
    ax.set_title("Tool Calls by Correctness Category per Agent (Flux)")
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.legend(title="Category", loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved breakdown chart to {output_path}")


def compute_yes_no_comparison(
    dfs: list[pd.DataFrame], agents: list[str]
) -> pd.DataFrame:
    """Compare tool calls for correctly classified Yes vs No materials.

    Args:
        dfs: List of DataFrames (one per CSV file)
        agents: List of agent names corresponding to each DataFrame

    Returns:
        DataFrame with columns: agent, ground_truth, mean_tool_calls, stderr_tool_calls, n_materials
    """
    results: list[dict] = []

    # Group DataFrames by agent
    agent_dfs: dict[str, list[pd.DataFrame]] = {}
    for df, agent in zip(dfs, agents):
        if agent not in agent_dfs:
            agent_dfs[agent] = []
        agent_dfs[agent].append(df)

    for agent, agent_df_list in agent_dfs.items():
        for ground_truth in ["Yes", "No"]:
            all_tool_counts: list[float] = []

            for df in agent_df_list:
                # Filter to correct classification + is_grown_with_flux rows with matching ground truth
                mask = (
                    (df["correctness_category"] == CATEGORY_CORRECT)
                    & (df["property_name"] == "is_grown_with_flux")
                    & (df["property_value"] == ground_truth)
                )
                filtered = df[mask]
                valid_counts = filtered["n_tool_use_counts"].dropna()
                all_tool_counts.extend(valid_counts.tolist())

            if len(all_tool_counts) == 0:
                results.append({
                    "agent": agent,
                    "ground_truth": ground_truth,
                    "mean_tool_calls": None,
                    "stderr_tool_calls": None,
                    "n_materials": 0,
                })
            else:
                mean_val = np.mean(all_tool_counts)
                stderr_val = (
                    np.std(all_tool_counts, ddof=1) / np.sqrt(len(all_tool_counts))
                    if len(all_tool_counts) > 1
                    else 0
                )
                results.append({
                    "agent": agent,
                    "ground_truth": ground_truth,
                    "mean_tool_calls": mean_val,
                    "stderr_tool_calls": stderr_val,
                    "n_materials": len(all_tool_counts),
                })

    return pd.DataFrame(results)


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
        description="Analyze LLM agent trajectory results for flux precedent search tasks"
    )
    parser.add_argument(
        "csv_paths",
        nargs="+",
        type=Path,
        help="Paths to scored results CSV files (e.g., scored_results_flux-gemini-cli-run-1_detailed.csv)",
    )
    parser.add_argument(
        "--output-dir",
        "-od",
        type=Path,
        default=Path("out"),
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
        df = add_category_to_df(df)

        all_dfs.append(df)
        all_agents.append(agent)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / f"{csv_path.stem}_with_steps.csv"
        df.to_csv(output_path, index=False)
        print(f"  Saved augmented CSV to {output_path}")

    # 1. Compute and display summary statistics
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

    # 2. Compute and display breakdown by correctness category
    breakdown_df = compute_breakdown_summary(all_dfs, all_agents)

    # Format breakdown table
    categories = [CATEGORY_INCORRECT, CATEGORY_CORRECT]
    breakdown_display_rows: list[dict] = []

    for agent in breakdown_df["agent"].unique():
        row_data = {"Agent": agent}
        for cat in categories:
            cat_row = breakdown_df[(breakdown_df["agent"] == agent) & (breakdown_df["category"] == cat)]
            if cat_row.empty:
                row_data[cat] = "N/A"
            else:
                mean_val = cat_row["mean_tool_calls"].iloc[0]
                stderr_val = cat_row["stderr_tool_calls"].iloc[0]
                n_mat = cat_row["n_materials"].iloc[0]
                row_data[cat] = f"{format_mean_stderr(mean_val, stderr_val)} (n={n_mat})"
        breakdown_display_rows.append(row_data)

    breakdown_display_df = pd.DataFrame(breakdown_display_rows)

    print("\n## Tool Calls by Correctness Category\n")
    print(tabulate(breakdown_display_df, headers="keys", tablefmt="github", showindex=False))

    # Generate grouped bar chart
    chart_path = args.output_dir / "tool_calls_breakdown_by_correctness_category_flux.png"
    plot_breakdown_bar_chart(breakdown_df, chart_path)

    # 3. Yes vs No comparison for fully correct materials
    yes_no_df = compute_yes_no_comparison(all_dfs, all_agents)

    # Format table: Agent | Correct Yes | Correct No | Δ (No - Yes)
    yes_no_display_rows: list[dict] = []

    for agent in yes_no_df["agent"].unique():
        agent_data = yes_no_df[yes_no_df["agent"] == agent]
        yes_row = agent_data[agent_data["ground_truth"] == "Yes"]
        no_row = agent_data[agent_data["ground_truth"] == "No"]

        yes_mean = yes_row["mean_tool_calls"].iloc[0] if not yes_row.empty else None
        yes_stderr = yes_row["stderr_tool_calls"].iloc[0] if not yes_row.empty else None
        yes_n = yes_row["n_materials"].iloc[0] if not yes_row.empty else 0

        no_mean = no_row["mean_tool_calls"].iloc[0] if not no_row.empty else None
        no_stderr = no_row["stderr_tool_calls"].iloc[0] if not no_row.empty else None
        no_n = no_row["n_materials"].iloc[0] if not no_row.empty else 0

        # Compute delta
        if yes_mean is not None and no_mean is not None:
            delta = no_mean - yes_mean
            delta_str = f"{delta:+.1f}"
        else:
            delta_str = "N/A"

        yes_no_display_rows.append({
            "Agent": agent,
            "Correct Yes (TP)": f"{format_mean_stderr(yes_mean, yes_stderr)} (n={yes_n})",
            "Correct No (TN)": f"{format_mean_stderr(no_mean, no_stderr)} (n={no_n})",
            "Δ (No - Yes)": delta_str,
        })

    yes_no_display_df = pd.DataFrame(yes_no_display_rows)

    print("\n## Tool Calls: Correct Yes vs Correct No (Fully Correct)\n")
    print(tabulate(yes_no_display_df, headers="keys", tablefmt="github", showindex=False))


if __name__ == "__main__":
    main()
