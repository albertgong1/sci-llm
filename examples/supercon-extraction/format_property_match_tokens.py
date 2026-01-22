"""Script to format token usage for the property matching step."""

from argparse import ArgumentParser
import pandas as pd
from tabulate import tabulate
import pbench

# Add src to path so we can import llm_utils
# Resolves to: .../sci-llm/src
# sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from llm_utils import calculate_cost

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

# Add cost column to df
df["cost_usd"] = df.apply(
    lambda row: calculate_cost(
        model=row["model"],
        prompt_tokens=row["usage.prompt_tokens"],
        completion_tokens=row["usage.completion_tokens"],
        cached_tokens=row["usage.cached_tokens"],
        thinking_tokens=row["usage.thinking_tokens"],
    ),
    axis=1,
)

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
