"""Script to format token usage for the property matching step."""

from argparse import ArgumentParser
import pandas as pd
from tabulate import tabulate
import yaml

import pbench

parser = ArgumentParser(description="Format property match tokens")
parser = pbench.add_base_args(parser)
args = parser.parse_args()
pbench.setup_logging(args.log_level)

# Load property match results
pred_responses_dir = args.output_dir / "pred_responses"
# get all CSV files in the directory
csv_files = list(pred_responses_dir.glob("*.csv"))
dfs = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(df)} total rows from {len(csv_files)} files")

# Compute average group by model_name
mean_sem = lambda x: f"{x.mean():.2f} ± {x.sem():.2f}"  # noqa: E731
df_token_usage = (
    df.groupby("model")
    .agg(
        avg_prompt_tokens=pd.NamedAgg(column="usage.prompt_tokens", aggfunc=mean_sem),
        avg_cached_tokens=pd.NamedAgg(column="usage.cached_tokens", aggfunc=mean_sem),
        avg_completion_tokens=pd.NamedAgg(
            column="usage.completion_tokens", aggfunc=mean_sem
        ),
        avg_thinking_tokens=pd.NamedAgg(
            column="usage.thinking_tokens", aggfunc=mean_sem
        ),
        avg_total_tokens=pd.NamedAgg(column="usage.total_tokens", aggfunc=mean_sem),
        count=pd.NamedAgg(column="model", aggfunc="count"),
    )
    .reset_index()
)

# print as table using the tabulate library with 'github' format
print(tabulate(df_token_usage, headers="keys", tablefmt="github", showindex=False))

# Load pricing data
pricing_file = "gemini_pricing.yaml"
with open(pricing_file, "r") as f:
    pricing_map = yaml.safe_load(f)


# Calculate cost per request for each model
def calculate_cost(row: pd.Series) -> float | None:
    """Calculate dollar cost per request based on token usage.

    Mapping:
    - prompt_tokens -> input pricing (per 1M tokens)
    - completion_tokens -> output pricing (per 1M tokens)
    - thinking_tokens -> output pricing (per 1M tokens)
    """
    model = row["model"]

    if model not in pricing_map:
        return None

    prices = pricing_map[model]
    if prices is None or not isinstance(prices, dict):
        return None

    input_price = prices.get("input")  # USD per 1M tokens
    cache_price = prices.get("context_cache_read")  # USD per 1M tokens
    output_price = prices.get("output")  # USD per 1M tokens

    if input_price is None or output_price is None:
        return None

    # Calculate cost (divide by 1M since prices are per 1M tokens)
    prompt_cost = (
        (row["usage.prompt_tokens"] - row["usage.cached_tokens"])
        * input_price
        / 1_000_000
    )
    cache_cost = row["usage.cached_tokens"] * cache_price / 1_000_000
    completion_cost = row["usage.completion_tokens"] * output_price / 1_000_000

    total_cost = prompt_cost + cache_cost + completion_cost
    return total_cost


# Add cost column to df
df["cost_usd"] = df.apply(calculate_cost, axis=1)

# Compute cost statistics by model
mean_sem_cost = lambda x: f"${x.mean():.6f} ± ${x.sem():.6f}"  # noqa: E731
df_cost_usage = (
    df.groupby("model")
    .agg(
        avg_cost_per_request=pd.NamedAgg(column="cost_usd", aggfunc=mean_sem_cost),
        total_cost=pd.NamedAgg(column="cost_usd", aggfunc=lambda x: f"${x.sum():.2f}"),
        count=pd.NamedAgg(column="model", aggfunc="count"),
    )
    .reset_index()
)

print("\n--- Cost Estimation ---")
print(tabulate(df_cost_usage, headers="keys", tablefmt="github", showindex=False))
