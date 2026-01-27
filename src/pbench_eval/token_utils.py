"""Token usage collection and formatting utilities.

Usage:
    from pbench_eval.token_utils import (
        collect_harbor_token_usage,
        collect_zeroshot_token_usage,
        format_token_statistics,
        count_trials_per_group,
    )
"""

import json
import re
from pathlib import Path
from typing import TypedDict

import pandas as pd
from tqdm import tqdm

from llm_utils import calculate_cost
from pbench_eval.stats import mean_sem_with_n


class TokenUsageRecord(TypedDict, total=False):
    """Single token usage record from a trial or zeroshot run."""

    batch: str
    trial_id: str
    agent: str
    model_name: str
    reasoning_effort: str
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cached_tokens: int
    total_thinking_tokens: int
    total_steps: int
    total_cost_usd: float | None


def collect_harbor_token_usage(
    jobs_dir: Path,
    include_reasoning_effort: bool = False,
) -> list[TokenUsageRecord]:
    """Collect token usage from all trials in a Harbor jobs directory.

    Args:
        jobs_dir: Path to the Harbor jobs directory
        include_reasoning_effort: If True, extract reasoning_effort from trial config

    Returns:
        List of TokenUsageRecord dicts

    """
    jobs_dir = jobs_dir.resolve()
    if not jobs_dir.exists():
        raise FileNotFoundError(f"Jobs directory not found: {jobs_dir}")

    results: list[TokenUsageRecord] = []

    for batch_dir in tqdm(sorted(jobs_dir.iterdir()), desc="Batches"):
        if not batch_dir.is_dir():
            continue

        # Get agent and model name from batch config.json
        agent_name, model_name = "unknown", "unknown"
        config_path = batch_dir / "config.json"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text())
                if config.get("agents"):
                    agent_name = config["agents"][0].get("name", "unknown")
                    model_name = config["agents"][0].get("model_name", "unknown")
            except Exception:
                pass

        for trial_dir in sorted(batch_dir.iterdir()):
            if not trial_dir.is_dir():
                continue
            # Skip non-trial directories (e.g., config.json, result.json)
            if not (trial_dir / "agent").exists():
                continue

            trajectory_path = trial_dir / "agent/trajectory.json"
            if not trajectory_path.exists():
                continue

            try:
                with open(trajectory_path) as f:
                    trajectory = json.load(f)
            except Exception as e:
                print(f"Skipping {trial_dir}: {e}")
                continue

            # Get reasoning_effort from trial config.json if requested
            reasoning_effort = ""
            if include_reasoning_effort:
                trial_config_path = trial_dir / "config.json"
                if trial_config_path.exists():
                    try:
                        trial_config = json.loads(trial_config_path.read_text())
                        reasoning_effort = (
                            trial_config.get("agent", {})
                            .get("kwargs", {})
                            .get("reasoning_effort", "")
                        )
                    except Exception:
                        pass

            # Extract final metrics
            final_metrics = trajectory.get("final_metrics", {})
            extra_metrics = final_metrics.get("extra", {})

            # Get step count: use "steps" field length for terminus-2, otherwise use final_metrics
            if agent_name == "terminus-2":
                total_steps = len(trajectory.get("steps", []))
            else:
                total_steps = final_metrics.get("total_steps", 0)

            # Get thinking tokens: check extra.reasoning_output_tokens (OpenAI) first
            thinking_tokens = extra_metrics.get(
                "reasoning_output_tokens",
                final_metrics.get("total_thinking_tokens", 0),
            )

            # Get cost: use total_cost_usd if available, otherwise estimate from tokens
            total_cost = final_metrics.get("total_cost_usd")
            if total_cost is None:
                total_cost = calculate_cost(
                    model=model_name,
                    prompt_tokens=final_metrics.get("total_prompt_tokens", 0),
                    completion_tokens=final_metrics.get("total_completion_tokens", 0),
                    cached_tokens=final_metrics.get("total_cached_tokens", 0),
                )

            record: TokenUsageRecord = {
                "batch": batch_dir.name,
                "trial_id": trial_dir.name,
                "agent": agent_name,
                "model_name": model_name,
                "total_prompt_tokens": final_metrics.get("total_prompt_tokens", 0),
                "total_completion_tokens": final_metrics.get(
                    "total_completion_tokens", 0
                ),
                "total_cached_tokens": final_metrics.get("total_cached_tokens", 0),
                "total_thinking_tokens": thinking_tokens,
                "total_steps": total_steps,
                "total_cost_usd": total_cost,
            }
            if include_reasoning_effort:
                record["reasoning_effort"] = reasoning_effort

            results.append(record)

    return results


def _collect_from_usage_dir(output_dir: Path) -> list[TokenUsageRecord]:
    """Collect token usage from usage/ directory (supercon format).

    Args:
        output_dir: Path to the output directory containing usage/ subdirectory

    Returns:
        List of TokenUsageRecord dicts

    """
    usage_dir = output_dir / "usage"
    if not usage_dir.exists():
        raise FileNotFoundError(f"Usage directory not found: {usage_dir}")

    # Parse usage JSON files: usage__model=<model>__refno=<refno>.json
    # or usage__agent=<agent>__model=<model>__refno=<refno>.json
    usage_pattern = re.compile(r"usage__(?:agent=.+__)?model=(.+?)__refno=(.+)\.json")

    results: list[TokenUsageRecord] = []

    for usage_file in tqdm(sorted(usage_dir.glob("*.json")), desc="Usage files"):
        match = usage_pattern.match(usage_file.name)
        if not match:
            print(f"Skipping unrecognized file: {usage_file.name}")
            continue

        model_name = match.group(1)
        refno = match.group(2)

        try:
            with open(usage_file) as f:
                usage = json.load(f)
        except Exception as e:
            print(f"Skipping {usage_file}: {e}")
            continue

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        cached_tokens = usage.get("cached_tokens", 0)
        thinking_tokens = usage.get("thinking_tokens", 0)

        # Calculate cost from tokens
        total_cost = calculate_cost(
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
        )

        results.append(
            {
                "batch": "zeroshot",
                "trial_id": refno,
                "agent": "zeroshot",
                "model_name": model_name,
                "total_prompt_tokens": prompt_tokens,
                "total_completion_tokens": completion_tokens,
                "total_cached_tokens": cached_tokens,
                "total_thinking_tokens": thinking_tokens,
                "total_steps": 1,
                "total_cost_usd": total_cost,
            }
        )

    return results


def _collect_from_trajectories_dir(
    output_dir: Path,
    include_reasoning_effort: bool = False,
) -> list[TokenUsageRecord]:
    """Collect token usage from trajectories/ directory (biosurfactants format).

    Args:
        output_dir: Path to the output directory containing trajectories/ subdirectory
        include_reasoning_effort: If True, extract reasoning_effort from inf_gen_config

    Returns:
        List of TokenUsageRecord dicts

    """
    trajectory_dir = output_dir / "trajectories"
    if not trajectory_dir.exists():
        raise FileNotFoundError(f"Trajectories directory not found: {trajectory_dir}")

    # Parse trajectory JSON files: trajectory__agent={agent}__model={model}__refno={refno}.json
    trajectory_pattern = re.compile(
        r"trajectory__agent=([^_]+)__model=([^_]+)__refno=(.+)\.json"
    )

    results: list[TokenUsageRecord] = []

    for traj_file in tqdm(
        sorted(trajectory_dir.glob("*.json")), desc="Trajectory files"
    ):
        match = trajectory_pattern.match(traj_file.name)
        if not match:
            print(f"Skipping unrecognized file: {traj_file.name}")
            continue

        agent = match.group(1)
        model_name = match.group(2).replace("--", "/")  # Convert back from safe name
        refno = match.group(3)

        try:
            with open(traj_file) as f:
                trajectory = json.load(f)
        except Exception as e:
            print(f"Skipping {traj_file}: {e}")
            continue

        # Get reasoning_effort from inf_gen_config
        reasoning_effort = ""
        if include_reasoning_effort:
            inf_gen_config = trajectory.get("inf_gen_config", {})
            reasoning_effort = inf_gen_config.get("reasoning_effort", "")
            if reasoning_effort is None:
                reasoning_effort = ""

        # Extract usage from llm_response
        llm_response = trajectory.get("llm_response")
        if llm_response is None:
            print(f"No llm_response in {traj_file.name}")
            continue

        usage = llm_response.get("usage", {})
        if not usage:
            print(f"No usage data in {traj_file.name}")
            continue

        # Normalize token field names between OpenAI and Gemini
        prompt_tokens = usage.get("prompt_tokens", 0)
        cached_tokens = usage.get("cached_tokens", 0)

        # OpenAI uses output_tokens (includes reasoning_tokens)
        # Gemini uses completion_tokens (excludes thinking_tokens)
        if "output_tokens" in usage:
            # OpenAI format
            completion_tokens = usage["output_tokens"]
            thinking_tokens = usage.get("reasoning_tokens", 0)
            thinking_included_in_completion = True
        else:
            # Gemini format
            completion_tokens = usage.get("completion_tokens", 0)
            thinking_tokens = usage.get("thinking_tokens", 0)
            thinking_included_in_completion = False

        # Calculate cost
        total_cost = calculate_cost(
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            thinking_tokens=thinking_tokens,
            thinking_included_in_completion=thinking_included_in_completion,
        )

        record: TokenUsageRecord = {
            "trial_id": refno,
            "agent": agent,
            "model_name": model_name,
            "total_prompt_tokens": prompt_tokens,
            "total_completion_tokens": completion_tokens,
            "total_cached_tokens": cached_tokens,
            "total_thinking_tokens": thinking_tokens,
            "total_steps": 1,  # zeroshot is always 1 step
            "total_cost_usd": total_cost,
        }
        if include_reasoning_effort:
            record["reasoning_effort"] = reasoning_effort

        results.append(record)

    return results


def collect_zeroshot_token_usage(
    output_dir: Path,
    include_reasoning_effort: bool = False,
) -> list[TokenUsageRecord]:
    """Auto-detect and collect token usage from zeroshot output directory.

    Checks for trajectories/ directory first (biosurfactants format),
    then usage/ directory (supercon format).

    Args:
        output_dir: Path to the zeroshot output directory
        include_reasoning_effort: If True, extract reasoning_effort

    Returns:
        List of TokenUsageRecord dicts

    Raises:
        FileNotFoundError: If neither usage/ nor trajectories/ directory found

    """
    output_dir = output_dir.resolve()

    # Try trajectories/ first (has more metadata)
    if (output_dir / "trajectories").exists():
        return _collect_from_trajectories_dir(output_dir, include_reasoning_effort)

    # Fall back to usage/
    if (output_dir / "usage").exists():
        return _collect_from_usage_dir(output_dir)

    raise FileNotFoundError(
        f"Neither trajectories/ nor usage/ directory found in: {output_dir}"
    )


def count_trials_per_group(
    jobs_dir: Path,
    include_reasoning_effort: bool = False,
) -> dict[tuple, int]:
    """Count trials per agent/model (optionally with reasoning_effort).

    Args:
        jobs_dir: Path to the Harbor jobs directory
        include_reasoning_effort: If True, group by (agent, model, reasoning_effort)

    Returns:
        Dict mapping group tuple to trial count

    """
    jobs_dir = jobs_dir.resolve()
    if not jobs_dir.exists():
        raise FileNotFoundError(f"Jobs directory not found: {jobs_dir}")

    counts: dict[tuple, int] = {}

    for batch_dir in sorted(jobs_dir.iterdir()):
        if not batch_dir.is_dir():
            continue

        # Get agent and model name from batch config.json
        agent, model = None, None
        config_path = batch_dir / "config.json"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text())
                if config.get("agents"):
                    agent = config["agents"][0].get("name")
                    model = config["agents"][0].get("model_name")
            except Exception:
                pass

        for trial_dir in sorted(batch_dir.iterdir()):
            if not trial_dir.is_dir():
                continue

            if include_reasoning_effort:
                # Extract reasoning_effort from trial config.json
                reasoning_effort = ""
                trial_config_path = trial_dir / "config.json"
                if trial_config_path.exists():
                    try:
                        trial_config = json.loads(trial_config_path.read_text())
                        reasoning_effort = (
                            trial_config.get("agent", {})
                            .get("kwargs", {})
                            .get("reasoning_effort", "")
                        )
                    except Exception:
                        pass
                key = (agent, model, reasoning_effort)
            else:
                key = (agent, model)

            counts[key] = counts.get(key, 0) + 1

    return counts


def count_zeroshot_trials_per_group(
    output_dir: Path,
    include_reasoning_effort: bool = False,
) -> dict[tuple, int]:
    """Count trials per agent/model from trajectory files in output directory.

    Args:
        output_dir: Path to the zeroshot output directory containing trajectories/
        include_reasoning_effort: If True, group by (agent, model, reasoning_effort)

    Returns:
        Dict mapping group tuple to trial count

    """
    output_dir = output_dir.resolve()
    trajectory_dir = output_dir / "trajectories"
    if not trajectory_dir.exists():
        raise FileNotFoundError(f"Trajectories directory not found: {trajectory_dir}")

    # Parse trajectory JSON files: trajectory__agent={agent}__model={model}__refno={refno}.json
    trajectory_pattern = re.compile(
        r"trajectory__agent=([^_]+)__model=([^_]+)__refno=(.+)\.json"
    )

    counts: dict[tuple, int] = {}

    for traj_file in sorted(trajectory_dir.glob("*.json")):
        match = trajectory_pattern.match(traj_file.name)
        if not match:
            continue

        agent = match.group(1)
        model_name = match.group(2).replace("--", "/")  # Convert back from safe name

        if include_reasoning_effort:
            # Need to read the file to get reasoning_effort
            reasoning_effort = ""
            try:
                with open(traj_file) as f:
                    trajectory = json.load(f)
                inf_gen_config = trajectory.get("inf_gen_config", {})
                reasoning_effort = inf_gen_config.get("reasoning_effort", "")
                if reasoning_effort is None:
                    reasoning_effort = ""
            except Exception:
                pass
            key = (agent, model_name, reasoning_effort)
        else:
            key = (agent, model_name)

        counts[key] = counts.get(key, 0) + 1

    return counts


def format_token_statistics(
    records: list[TokenUsageRecord],
    group_cols: list[str],
    trials_lookup: dict[tuple, int] | None = None,
    scale_factor: float = 1e6,
    scale_suffix: str = "M",
) -> pd.DataFrame:
    """Compute and format token usage statistics.

    Args:
        records: List of TokenUsageRecord dicts
        group_cols: Columns to group by (e.g., ["agent", "model_name"])
        trials_lookup: Optional dict mapping group keys to trial counts.
                       If None, counts unique trial_ids per group.
        scale_factor: Factor to divide token counts by (default: 1e6 for millions)
        scale_suffix: Suffix for scaled columns (default: "M")

    Returns:
        DataFrame with mean +/- SEM statistics per group

    """
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["total_tokens"] = df["total_prompt_tokens"] + df["total_completion_tokens"]

    # Fill missing thinking tokens with 0
    if "total_thinking_tokens" not in df.columns:
        df["total_thinking_tokens"] = 0
    df["total_thinking_tokens"] = df["total_thinking_tokens"].fillna(0)

    # Get trial counts for normalization
    if trials_lookup is None:
        trials_lookup = df.groupby(group_cols)["trial_id"].nunique().to_dict()

    # Metric columns and their display names
    metric_cols = [
        "total_prompt_tokens",
        "total_completion_tokens",
        "total_cached_tokens",
        "total_thinking_tokens",
        "total_tokens",
        "total_steps",
        "total_cost_usd",
    ]

    # Scale token columns (except steps and cost)
    scale_cols = [
        "total_prompt_tokens",
        "total_completion_tokens",
        "total_cached_tokens",
        "total_thinking_tokens",
        "total_tokens",
    ]
    df[scale_cols] = df[scale_cols] / scale_factor

    # Display column names
    display_cols = [
        f"Prompt ({scale_suffix})",
        f"Completion ({scale_suffix})",
        f"Cached ({scale_suffix})",
        f"Thinking ({scale_suffix})",
        f"Total ({scale_suffix})",
        "Steps",
        "Cost ($)",
    ]

    def compute_stats(g: pd.DataFrame) -> pd.Series:
        """Compute mean ± SEM for each metric column."""
        key = g.name  # tuple from groupby
        n_trials = trials_lookup.get(key, len(g))

        stats = {
            new_col: mean_sem_with_n(g[old_col].tolist(), n_trials)
            for old_col, new_col in zip(metric_cols, display_cols)
        }
        stats["successful_count"] = len(g)
        stats["num_trials"] = n_trials
        return pd.Series(stats)

    result = (
        df.groupby(group_cols).apply(compute_stats, include_groups=False).reset_index()
    )

    return result
