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
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tldextract
from tabulate import tabulate

# Set font family to Times New Roman
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
})

# URL extraction regex pattern
URL_PATTERN = re.compile(r'https?://[^\s\'"<>\\]+')

SKIP_DOMAINS: list[str] = ["w3.org"]

AGENT2LATEX_MACRO: dict[str, str] = {
    "gemini-cli": "\\geminicli",
    "codex": "\\codex",
    "terminus-gemini": "\\terminusgemini",
    "terminus-gpt": "\\terminusgpt",
}

AGENT2AGENT_NAME: dict[str, str] = {
    "gemini-cli": "gemini-cli",
    "codex": "codex",
    "terminus-gemini": "T2-gemini",
    "terminus-gpt": "T2-gpt",
}

# Agent log file mapping for log mining recovery methods
AGENT_LOG_FILES: dict[str, str] = {
    "gemini-cli": "gemini-cli.txt",
    "codex": "codex.txt",
    "terminus-gemini": None,  # uses episode-*/response.txt
    "terminus-gpt": None,  # uses episode-*/response.txt
}

# Category labels for correctness breakdown
CATEGORY_INCORRECT_CLASS = "Incorrect"
CATEGORY_CORRECT_WRONG_VAL = "Partial"
CATEGORY_FULLY_CORRECT = "Correct"

CATEGORIES = [CATEGORY_INCORRECT_CLASS, CATEGORY_CORRECT_WRONG_VAL, CATEGORY_FULLY_CORRECT]

# Colors for each category
CATEGORY2COLOR: dict[str, str] = {
    CATEGORY_INCORRECT_CLASS: "#FFCC80",
    CATEGORY_CORRECT_WRONG_VAL: "#C6F1D6",
    CATEGORY_FULLY_CORRECT: "#88D8B0",
}

NOT_ATTEMPTED_COLOR = "#E0E0E0"

# Base directory for agent metadata
METADATA_BASE_DIR = Path("precedent-search-agent-metadata")


def extract_domains_from_text(text: str) -> list[str]:
    """Extract registered domains from URLs found in text.

    Uses tldextract to normalize domains (e.g., api.crossref.org -> crossref.org).

    Args:
        text: String that may contain URLs

    Returns:
        List of registered domains (e.g., ["crossref.org", "duckduckgo.com"])
    """
    urls = URL_PATTERN.findall(text)
    domains: list[str] = []
    for url in urls:
        try:
            extracted = tldextract.extract(url)
            if extracted.domain and extracted.suffix:
                registered_domain = f"{extracted.domain}.{extracted.suffix}"
                if registered_domain not in SKIP_DOMAINS:
                    domains.append(registered_domain)
        except Exception:
            continue
    return domains


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


def _extract_items_codex(tool_call: dict) -> list[str]:
    """Extract domains or function name from a codex tool call.

    For exec_command, extracts URL domains from the cmd argument.
    If not exec_command, returns empty list.
    """
    fn_name = tool_call.get("function_name", "")
    if fn_name != "exec_command":
        return []
        # return [fn_name] if fn_name else []

    args = tool_call.get("arguments", {})
    cmd = args.get("cmd", "")
    if isinstance(cmd, str):
        domains = extract_domains_from_text(cmd)
        if domains:
            return domains
    return []
    # return ["exec_command (local)"]


def _extract_items_terminus(tool_call: dict) -> list[str]:
    """Extract domains or function name from a terminus tool call.

    For bash_command, extracts URL domains from the keystrokes argument.
    If not bash_command, returns empty list.
    """
    fn_name = tool_call.get("function_name", "")
    if fn_name != "bash_command":
        return []
        # return [fn_name] if fn_name else []

    args = tool_call.get("arguments", {})
    keystrokes = args.get("keystrokes", "")
    if isinstance(keystrokes, str):
        domains = extract_domains_from_text(keystrokes)
        if domains:
            return domains
    return []
    # return ["bash_command (local)"]


def _extract_items_default(tool_call: dict) -> list[str]:
    """Extract function name from a tool call (default behavior)."""
    fn_name = tool_call.get("function_name", "")
    return [fn_name] if fn_name else []


def load_tool_call_names_from_trajectory(
    trajectory_path: Path, agent: str | None = None
) -> Counter[str]:
    """Load tool call items from trajectory.json.

    For codex and terminus agents, extracts URL domains from command arguments.
    For other agents, extracts function names.

    Args:
        trajectory_path: Path to trajectory.json file
        agent: Agent name (used to determine extraction strategy)

    Returns:
        Counter of function_name or domain occurrences.
    """
    if not trajectory_path.exists():
        return Counter()

    # Select extraction function based on agent
    if agent == "codex":
        extract_fn = _extract_items_codex
    elif agent and agent.startswith("terminus-"):
        extract_fn = _extract_items_terminus
    else:
        extract_fn = _extract_items_default

    try:
        with open(trajectory_path) as f:
            data = json.load(f)

        items: list[str] = []
        for step in data["steps"]:
            for tool_call in step.get("tool_calls", []):
                items.extend(extract_fn(tool_call))

        return Counter(items)

    except (json.JSONDecodeError, KeyError):
        return Counter()


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


def categorize_material(material_df: pd.DataFrame) -> str:
    """Categorize a material based on classification and value extraction correctness.

    Args:
        material_df: DataFrame containing the 3 rows for a single material
                    (is_superconducting, tc, tcn)

    Returns:
        One of: CATEGORY_INCORRECT_CLASS, CATEGORY_CORRECT_WRONG_VAL, CATEGORY_FULLY_CORRECT
    """
    # Get the is_superconducting row
    is_sc_row = material_df[material_df["property_name"] == "is_superconducting"]
    if is_sc_row.empty:
        return CATEGORY_INCORRECT_CLASS

    is_sc_score = is_sc_row["score"].iloc[0]
    is_sc_value = is_sc_row["property_value"].iloc[0]

    # Check classification correctness
    if is_sc_score == 0:
        return CATEGORY_INCORRECT_CLASS

    # Classification is correct, now check value extraction
    if is_sc_value == "Yes":
        # Look at tc row
        tc_row = material_df[material_df["property_name"] == "tc"]
        if tc_row.empty:
            return CATEGORY_CORRECT_WRONG_VAL
        value_score = tc_row["score"].iloc[0]
    elif is_sc_value == "No":
        # Look at tcn row
        tcn_row = material_df[material_df["property_name"] == "tcn"]
        if tcn_row.empty:
            return CATEGORY_CORRECT_WRONG_VAL
        value_score = tcn_row["score"].iloc[0]
    else:
        # Unknown or other value - treat as wrong value
        return CATEGORY_CORRECT_WRONG_VAL

    if value_score == 1:
        return CATEGORY_FULLY_CORRECT
    else:
        return CATEGORY_CORRECT_WRONG_VAL


def add_category_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add correctness_category column to dataframe.

    Args:
        df: DataFrame with material, property_name, property_value, score columns

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

    for agent, agent_df_list in agent_dfs.items():
        for category in CATEGORIES:
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
    FONT_SIZE = 24
    agents = list(breakdown_df["agent"].unique())

    x = np.arange(len(agents))
    width = 0.25
    n_categories = len(CATEGORIES)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, cat in enumerate(CATEGORIES):
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
            label=cat,
            color=CATEGORY2COLOR[cat],
            yerr=stderrs,
            capsize=3,
        )

    ax.set_ylabel("# Tool Calls", fontsize=FONT_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels([AGENT2AGENT_NAME[agent] for agent in agents])
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.legend(loc="upper right", fontsize=FONT_SIZE)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved breakdown chart to {output_path}")


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


def compute_tool_call_distribution(
    dfs: list[pd.DataFrame], agents: list[str]
) -> dict[str, dict[str, float]]:
    """Compute normalized distribution of tool call function names per agent.

    Args:
        dfs: List of DataFrames (one per CSV file)
        agents: List of agent names corresponding to each DataFrame

    Returns:
        Dict mapping agent -> {function_name: normalized_fraction}
    """
    # Group DataFrames by agent
    agent_dfs: dict[str, list[pd.DataFrame]] = {}
    for df, agent in zip(dfs, agents):
        if agent not in agent_dfs:
            agent_dfs[agent] = []
        agent_dfs[agent].append(df)

    results: dict[str, dict[str, float]] = {}

    for agent, agent_df_list in agent_dfs.items():
        # Aggregate tool call counts across all trajectories for this agent
        total_counter: Counter[str] = Counter()

        for df in agent_df_list:
            # Get unique (job_id, metadata_trial_id) combinations
            unique_trials = df[["job_id", "metadata_trial_id"]].drop_duplicates()

            for _, row in unique_trials.iterrows():
                job_id = row["job_id"]
                trial_id = row["metadata_trial_id"]

                task_run_dir = get_task_run_dir(agent, job_id, trial_id)
                trajectory_path = task_run_dir / "agent" / "trajectory.json"

                counter = load_tool_call_names_from_trajectory(trajectory_path, agent)
                total_counter.update(counter)

        # Normalize by total count
        total_calls = sum(total_counter.values())
        if total_calls > 0:
            results[agent] = {
                fn_name: count / total_calls
                for fn_name, count in total_counter.most_common()
            }
        else:
            results[agent] = {}

    return results


def compute_tool_call_distribution_by_category(
    dfs: list[pd.DataFrame], agents: list[str]
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute normalized URL distribution per agent per correctness category.

    Args:
        dfs: List of DataFrames (one per CSV file)
        agents: List of agent names corresponding to each DataFrame

    Returns:
        Dict mapping agent -> category -> {domain: normalized_fraction}
    """
    # Group DataFrames by agent
    agent_dfs: dict[str, list[pd.DataFrame]] = {}
    for df, agent in zip(dfs, agents):
        if agent not in agent_dfs:
            agent_dfs[agent] = []
        agent_dfs[agent].append(df)

    results: dict[str, dict[str, dict[str, float]]] = {}

    for agent, agent_df_list in agent_dfs.items():
        results[agent] = {}

        for category in CATEGORIES:
            # Aggregate tool calls for materials in this category
            total_counter: Counter[str] = Counter()

            for df in agent_df_list:
                # Get unique materials in this category
                cat_materials = df[df["correctness_category"] == category]["material"].unique()

                for material in cat_materials:
                    # Get the trial info for this material
                    material_row = df[df["material"] == material].iloc[0]
                    job_id = material_row["job_id"]
                    trial_id = material_row["metadata_trial_id"]

                    task_run_dir = get_task_run_dir(agent, job_id, trial_id)
                    trajectory_path = task_run_dir / "agent" / "trajectory.json"

                    counter = load_tool_call_names_from_trajectory(trajectory_path, agent)
                    total_counter.update(counter)

            # Normalize by total count
            total_calls = sum(total_counter.values())
            if total_calls > 0:
                results[agent][category] = {
                    name: count / total_calls
                    for name, count in total_counter.most_common()
                }
            else:
                results[agent][category] = {}

    return results


def plot_url_distribution_stacked_horizontal(
    dist_by_cat: dict[str, dict[str, dict[str, float]]],
    output_dir: Path,
    top_n: int = 8,
) -> None:
    """Plot stacked horizontal bar chart of URL distribution by correctness category.

    Creates one plot per agent. Each bar represents a category, with stacked segments
    showing domain proportions within that category.

    Args:
        dist_by_cat: Dict from compute_tool_call_distribution_by_category
        output_dir: Directory to save the charts
        top_n: Number of top domains to show (others grouped as "Other")
    """
    agents = list(dist_by_cat.keys())
    category_short = {
        CATEGORY_INCORRECT_CLASS: "Incorrect Class",
        CATEGORY_CORRECT_WRONG_VAL: "Correct + Wrong Val",
        CATEGORY_FULLY_CORRECT: "Fully Correct",
    }

    for agent in agents:
        fig, ax = plt.subplots(figsize=(10, 4))

        # Build color palette for domains for this agent
        all_domains: set[str] = set()
        for cat in CATEGORIES:
            cat_dist = dist_by_cat[agent].get(cat, {})
            top_items = sorted(cat_dist.items(), key=lambda x: x[1], reverse=True)[:top_n]
            all_domains.update(d for d, _ in top_items)

        if not all_domains:
            plt.close()
            print(f"No data for agent {agent}, skipping stacked horizontal chart")
            continue

        domain_list = sorted(all_domains)
        cmap = plt.cm.get_cmap("tab20", len(domain_list) + 1)
        domain_colors = {d: cmap(i) for i, d in enumerate(domain_list)}
        domain_colors["Other"] = cmap(len(domain_list))

        y_positions = np.arange(len(CATEGORIES))
        bar_height = 0.6

        for y_idx, cat in enumerate(CATEGORIES):
            cat_dist = dist_by_cat[agent].get(cat, {})
            if not cat_dist:
                continue

            # Get top N domains + "Other"
            sorted_items = sorted(cat_dist.items(), key=lambda x: x[1], reverse=True)
            top_items = sorted_items[:top_n]
            other_frac = sum(frac for _, frac in sorted_items[top_n:])

            # Draw stacked bars
            left = 0
            for domain, frac in top_items:
                ax.barh(
                    y_idx,
                    frac,
                    height=bar_height,
                    left=left,
                    color=domain_colors.get(domain, "gray"),
                    edgecolor="white",
                    linewidth=0.5,
                )
                left += frac

            if other_frac > 0:
                ax.barh(
                    y_idx,
                    other_frac,
                    height=bar_height,
                    left=left,
                    color=domain_colors["Other"],
                    edgecolor="white",
                    linewidth=0.5,
                )

        ax.set_yticks(y_positions)
        ax.set_yticklabels([c for c in CATEGORIES])
        ax.set_xlabel("Fraction")
        ax.set_title(f"{agent} - URL Distribution by Correctness Category")
        ax.set_xlim(0, 1)
        ax.grid(axis="x", alpha=0.3)

        # Create legend with all domains
        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=domain_colors[d], edgecolor="white")
            for d in domain_list
        ]
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, facecolor=domain_colors["Other"], edgecolor="white")
        )
        legend_labels = domain_list + ["Other"]

        ax.legend(
            legend_handles,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            fontsize=9,
            title="Domain",
        )

        plt.tight_layout()
        output_path = output_dir / f"url_distribution_stacked_horizontal_{agent}.pdf"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved stacked horizontal chart to {output_path}")


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

    ### 1. Compute summary statistics
    summary_df = compute_tool_use_counts_summary(all_dfs, all_agents)

    ### 2. Compute breakdown by correctness category
    breakdown_df = compute_breakdown_summary(all_dfs, all_agents)

    # Build combined table: Agent | Total
    combined_rows: list[dict] = []

    for agent in summary_df["agent"].unique():
        # Get overall stats
        agent_summary = summary_df[summary_df["agent"] == agent].iloc[0]
        overall_mean = agent_summary["mean_tool_use_counts"]
        overall_stderr = agent_summary["stderr_tool_use_counts"]

        row_data = {
            "Agent": agent,
            "Total": format_mean_stderr(overall_mean, overall_stderr),
        }

        combined_rows.append(row_data)

    combined_df = pd.DataFrame(combined_rows)
    # Reorder the columns to "Agent", "Total"
    combined_df = combined_df[["Agent", "Total"]]

    # Print markdown table
    print("\n## Tool Calls Summary\n")
    print(tabulate(combined_df, headers="keys", tablefmt="github", showindex=False))

    # Print LaTeX table
    print("\n## LaTeX Table (for Overleaf)\n")
    combined_df_latex = combined_df.copy()
    combined_df_latex["Agent"] = combined_df_latex["Agent"].map(AGENT2LATEX_MACRO)
    print(combined_df_latex.to_latex(index=False, escape=False))

    # Generate grouped bar chart
    chart_path = args.output_dir / "tool_calls_breakdown_by_correctness_category.pdf"
    plot_breakdown_bar_chart(breakdown_df, chart_path)

    ### 3. Tool call function name distribution (top 10 per agent)
    tool_dist = compute_tool_call_distribution(all_dfs, all_agents)

    print("\n## Tool Call Distribution (Top 10 per Agent)\n")
    for agent in tool_dist:
        print(f"\n### {agent}\n")
        # Sort by fraction descending and take top 10
        sorted_items = sorted(tool_dist[agent].items(), key=lambda x: x[1], reverse=True)[:10]
        if not sorted_items:
            print("No tool calls found.\n")
            continue

        top_rows = [{"Tool/Domain": name, "Fraction": f"{frac:.1%}"} for name, frac in sorted_items]
        print(tabulate(top_rows, headers="keys", tablefmt="github", showindex=False))

    ### 4. URL distribution by correctness category (top 5 per agent per category)
    tool_dist_by_cat = compute_tool_call_distribution_by_category(all_dfs, all_agents)

    # print("\n## URL Distribution by Correctness Category (Top 10 per Agent)\n")
    # for agent in tool_dist_by_cat:
    #     print(f"\n### {agent}\n")
    #     for category in CATEGORIES:
    #         cat_dist = tool_dist_by_cat[agent].get(category, {})
    #         sorted_items = sorted(cat_dist.items(), key=lambda x: x[1], reverse=True)[:10]

    #         print(f"\n**{category}**\n")
    #         if not sorted_items:
    #             print("No URLs found.\n")
    #             continue

    #         top_rows = [{"Domain": name, "Fraction": f"{frac:.1%}"} for name, frac in sorted_items]
    #         print(tabulate(top_rows, headers="keys", tablefmt="github", showindex=False))

    # Generate URL distribution visualizations (one plot per agent)
    plot_url_distribution_stacked_horizontal(tool_dist_by_cat, args.output_dir)


if __name__ == "__main__":
    main()
