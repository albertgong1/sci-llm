# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: sci-llm
#     language: python
#     name: python3
# ---

# %%
"""Script to create a HuggingFace dataset from the Biosurfactants dataset.

NOTE: skip the --repo_name argument if you don't want to push to HuggingFace Hub.

Each row contains the following information:
{
    "refno": "...",
    "properties": [
        {
            "id": "prop_001",
            "material_or_system": "...",
            "sample_label": "...",
            "property_name": "...",
            "category": "...",
            "value_string": "...",
            "method": "...",
            "notes": "...",
            "conditions": {
                "condition1_name": "condition1_value",
                "condition2_name": "condition2_value",
                ...
            },
            "location": {
                "page": 1,
                "section": "...",
                "figure_or_table": "...",
                "source_type": "text",
                "evidence": "..."
            }
        },
    ]
}

Usage:
```bash
# To filter out rows where paper PDF is not found at data_dir/Paper_DB
uv run python create_huggingface_dataset.py \
    --output_dir out-0113-for-jiashuo \
    --repo_name REPO_NAME \
    --filter_pdf

# To use all rows (including rows where paper PDF is not found at data_dir/Paper_DB)
uv run python create_huggingface_dataset.py \
    --output_dir out-0113-for-jiashuo \
    --repo_name REPO_NAME
```
"""

# %%
import logging
from argparse import ArgumentParser
from pathlib import Path

import huggingface_hub
import pandas as pd
from datasets import Dataset

import pbench

logger = logging.getLogger(__name__)

# %%
parser = ArgumentParser(
    description="Create a HuggingFace dataset from the Biosurfactants dataset."
)
parser = pbench.add_base_args(parser)
parser.add_argument(
    "--repo_name",
    type=str,
    default=None,
    help="Name of the HuggingFace repository to push to",
)
parser.add_argument(
    "--tag_name",
    type=str,
    default=None,
    help="Name of the tag to apply to the dataset",
)
parser.add_argument(
    "--filter_pdf",
    action="store_true",
    help="If true, filter out rows where paper PDF is not found at data_dir/Paper_DB",
)
parser.add_argument(
    "--split",
    type=str,
    default="test",
    help="Name of the split to use for the dataset (default: test)",
)
args = parser.parse_args()
pbench.setup_logging(args.log_level)

paper_dir = args.data_dir / "Paper_DB"
args.output_dir.mkdir(parents=True, exist_ok=True)

# %%
# Load the validated candidates CSV
data_path = (
    args.output_dir / "validated_candidates" / "extracted_properties_combined.csv"
)
logger.info(f"Loading dataset from {data_path}...")
df = pd.read_csv(data_path, dtype=str)
logger.info(f"Loaded {len(df)} rows")

# Filter out rows where validated is not True
df = df[df["validated"].str.lower() == "true"]
logger.info(f"After filtering for validated=True: {len(df)} rows")


def process_row(row: pd.Series) -> dict:
    """Process a single row of the Biosurfactants dataset.

    Args:
        row: a single row of the Biosurfactants dataset

    Returns:
        a processed property dict

    """
    # Build conditions dict from condition{N}_name/condition{N}_value pairs
    conditions = {}
    for i in range(1, 11):  # condition1 through condition10
        name_col = f"condition{i}_name"
        value_col = f"condition{i}_value"
        if name_col in row.index and value_col in row.index:
            name = row[name_col]
            value = row[value_col]
            if pd.notna(name) and str(name).strip():
                conditions[str(name)] = str(value) if pd.notna(value) else None

    # Build location dict
    location = {}
    location_fields = ["page", "section", "source_type", "evidence", "figure_or_table"]
    for field in location_fields:
        col = f"location.{field}"
        if col in row.index and pd.notna(row[col]):
            location[field] = str(row[col])

    return {
        "id": row["id"] if pd.notna(row.get("id")) else None,
        "material_or_system": str(row["material_or_system"])
        if pd.notna(row.get("material_or_system"))
        else None,
        "sample_label": str(row["sample_label"])
        if pd.notna(row.get("sample_label"))
        else None,
        "property_name": str(row["property_name"])
        if pd.notna(row.get("property_name"))
        else None,
        "category": str(row["category"]) if pd.notna(row.get("category")) else None,
        "value_string": str(row["value_string"])
        if pd.notna(row.get("value_string"))
        else None,
        "method": str(row["method"]) if pd.notna(row.get("method")) else None,
        "notes": str(row["notes"]) if pd.notna(row.get("notes")) else None,
        "conditions": conditions if conditions else None,
        "location": location if location else None,
    }


# Process each row
results = []
for _, row in df.iterrows():
    prop = process_row(row)
    results.append({"refno": row["refno"], "property": prop})

df_processed = pd.DataFrame(results)

# Group by refno and combine properties
df_grouped = df_processed.groupby("refno", as_index=False).agg(
    {"property": lambda x: list(x)}
)
df_grouped = df_grouped.rename(columns={"property": "properties"})

if args.filter_pdf:
    logger.info("Filtering out rows where paper PDF is not found")
    df_grouped = df_grouped[
        df_grouped["refno"].apply(lambda x: Path(paper_dir / f"{x}.pdf").exists())
    ]
    logger.info(f"{len(df_grouped)} rows have paper PDF")

logger.info("Saving dataset to CSV...")
save_path = args.output_dir / f"{args.split}.csv"
save_path.parent.mkdir(parents=True, exist_ok=True)
df_grouped.to_csv(save_path, index=False)
logger.info(f"Dataset saved to {save_path}")

dataset = Dataset.from_pandas(df_grouped)
dataset.save_to_disk(args.output_dir / f"{args.split}")
logger.info(f"Dataset saved to {args.output_dir / f'{args.split}'}")

# Load the dataset from disk and print the first row
loaded_dataset = Dataset.load_from_disk(args.output_dir / f"{args.split}")
logger.info("Loading first row from saved dataset:")
first_row = loaded_dataset[0]
print("\n" + "=" * 80)
print("First row of the saved dataset:")
print("=" * 80)
for key, value in first_row.items():
    print(f"\n{key}:")
    print(f"  {value}")
print("=" * 80 + "\n")

# %%
# Push to HuggingFace Hub (requires authentication)
# Make sure you're logged in: hf auth login
if args.repo_name is not None:
    logger.info(f"Pushing dataset to HuggingFace Hub: {args.repo_name}")
    logger.info(f"Uploading {len(df_grouped)} rows...")
    dataset = Dataset.from_pandas(df_grouped)
    dataset.push_to_hub(args.repo_name, private=False, split=args.split)
    logger.info(f"All {len(df_grouped)} rows pushed to {args.repo_name}")

    # Tag the dataset so that we can easily refer to different versions of the dataset
    # Ref: https://github.com/huggingface/datasets/discussions/5370
    if args.tag_name is not None:
        huggingface_hub.create_tag(
            args.repo_name, tag=args.tag_name, repo_type="dataset"
        )
        logger.info(f"Dataset tagged with {args.tag_name}")
