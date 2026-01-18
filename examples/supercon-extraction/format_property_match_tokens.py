"""Script to format token usage for the property matching step."""

from argparse import ArgumentParser
import pandas as pd
import json
from tabulate import tabulate
import yaml

import pbench

parser = ArgumentParser(description="Format property match tokens")
parser = pbench.add_base_args(parser)
args = parser.parse_args()
pbench.setup_logging(args.log_level)

# Load property match results
pred_matches_dir = args.output_dir / "pred_matches"
# get all CSV files in the directory
csv_files = list(pred_matches_dir.glob("*.csv"))
dfs = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    dfs.append(df)
df_matches = pd.concat(dfs, ignore_index=True)

df_matches["response_data"] = df_matches["serialized_response"].apply(
    lambda x: json.loads(x)
)
df_matches["prompt_tokens"] = df_matches["response_data"].apply(
    lambda x: x["usage"]["prompt_tokens"]
)
df_matches["completion_tokens"] = df_matches["response_data"].apply(
    lambda x: x["usage"]["completion_tokens"]
)
df_matches["thinking_tokens"] = df_matches["response_data"].apply(
    lambda x: x["usage"]["thinking_tokens"]
)
df_matches["total_tokens"] = df_matches["response_data"].apply(
    lambda x: x["usage"]["total_tokens"]
)

# Compute average group by model_name
mean_sem = lambda x: f"{x.mean():.2f} ± {x.sem():.2f}"  # noqa: E731
df_token_usage = (
    df_matches.groupby("model")
    .agg(
        avg_prompt_tokens=pd.NamedAgg(column="prompt_tokens", aggfunc=mean_sem),
        avg_completion_tokens=pd.NamedAgg(column="completion_tokens", aggfunc=mean_sem),
        avg_thinking_tokens=pd.NamedAgg(column="thinking_tokens", aggfunc=mean_sem),
        avg_total_tokens=pd.NamedAgg(column="total_tokens", aggfunc=mean_sem),
        count=pd.NamedAgg(column="model", aggfunc="count"),
    )
    .reset_index()
)

# print as table using the tabulate library with 'github' format
print(tabulate(df_token_usage, headers="keys", tablefmt="github", showindex=False))

# Load pricing data
pricing_file = "gemini_pricing.yaml"
with open(pricing_file, "r") as f:
    pricing_data = yaml.safe_load(f)

# Flatten pricing data for easier lookup
pricing_map = {}
for model_name, prices in pricing_data.items():
    pricing_map[model_name] = prices


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
    input_price = prices["input"]  # USD per 1M tokens
    output_price = prices["output"]  # USD per 1M tokens

    # Calculate cost (divide by 1M since prices are per 1M tokens)
    prompt_cost = row["prompt_tokens"] * input_price / 1_000_000
    completion_cost = row["completion_tokens"] * output_price / 1_000_000
    thinking_cost = row["thinking_tokens"] * output_price / 1_000_000

    total_cost = prompt_cost + completion_cost + thinking_cost
    return total_cost


# Add cost column to df_matches
df_matches["cost_usd"] = df_matches.apply(calculate_cost, axis=1)

# Compute cost statistics by model
mean_sem_cost = lambda x: f"${x.mean():.6f} ± ${x.sem():.6f}"  # noqa: E731
df_cost_usage = (
    df_matches.groupby("model")
    .agg(
        avg_cost_per_request=pd.NamedAgg(column="cost_usd", aggfunc=mean_sem_cost),
        total_cost=pd.NamedAgg(column="cost_usd", aggfunc=lambda x: f"${x.sum():.2f}"),
        count=pd.NamedAgg(column="model", aggfunc="count"),
    )
    .reset_index()
)

print("\n--- Cost Estimation ---")
print(tabulate(df_cost_usage, headers="keys", tablefmt="github", showindex=False))
