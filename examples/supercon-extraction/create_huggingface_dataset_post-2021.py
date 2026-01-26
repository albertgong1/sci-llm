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
"""Script to create a HuggingFace dataset from the post-2021 SuperCon dataset.

This script reads from data/combined_validation_results.csv which contains
validated extraction results in a flat CSV format with condition1_name/condition1_value
style columns.

NOTE: skip the --hf_repo argument if you don't want to push to HuggingFace Hub.

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
            "value_unit": "...",
            "qualifier": "...",
            "value_detail": "...",
            "conditions": {
                "temperature": "...",
                "pressure": "...",
                "field": "...",
                "frequency": "...",
                "orientation": "...",
                "environment": "...",
                "sample_state": "...",
                "other_conditions": "..."
            },
            "method": "...",
            "model_or_fit": "...",
            "location": {
                "page": 1,
                "section": "...",
                "figure_or_table": "...",
                "source_type": "text",
                "evidence": "..."
            },
            "notes": "..."
        }
    ]
}

Usage:
```bash
# To filter out rows where paper PDF is not found at data_dir/Paper_DB
uv run python create_huggingface_dataset_post-2021.py \
    --data_dir data \
    --hf_repo REPO_NAME \
    --filter_pdf

# To use all rows (including rows where paper PDF is not found at data_dir/Paper_DB)
uv run python create_huggingface_dataset_post-2021.py \
    --data_dir data \
    --hf_repo REPO_NAME
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
    description="Create a HuggingFace dataset from the SuperCon dataset."
)
parser = pbench.add_base_args(parser)
parser.add_argument(
    "--filter_pdf",
    action="store_true",
    help="If true, filter out rows where paper PDF is not found at data_dir/Paper_DB",
)
args = parser.parse_args()
pbench.setup_logging(args.log_level)

paper_dir = args.data_dir / "Paper_DB"
args.output_dir.mkdir(parents=True, exist_ok=True)

# %%
# Load the validated candidates CSV
data_path = args.data_dir / "combined_validation_results.csv"
logger.info(f"Loading dataset from {data_path}...")
df = pd.read_csv(data_path, dtype=str)
logger.info(f"Loaded {len(df)} rows")
# Filter out rows where validated is not True
df = df[df["validated_resolved"].str.lower() == "true"]
logger.info(f"After filtering for validated_resolved=True: {len(df)} rows")


def process_row(row: pd.Series) -> dict:
    """Process a single row of the SuperCon dataset.

    Args:
        row: a single row of the SuperCon dataset

    Returns:
        a processed property dict

    """
    # Build conditions dict from condition{N}_name/condition{N}_value pairs
    # Map to the standard condition keys when possible
    standard_condition_keys = {
        "temperature",
        "pressure",
        "field",
        "frequency",
        "orientation",
        "environment",
        "sample_state",
    }
    conditions = {
        "temperature": "",
        "pressure": "",
        "field": "",
        "frequency": "",
        "orientation": "",
        "environment": "",
        "sample_state": "",
        "other_conditions": "",
    }
    other_conditions_list = []

    for i in range(1, 11):  # condition1 through condition10
        name_col = f"condition{i}_name"
        value_col = f"condition{i}_value"
        if name_col in row.index and value_col in row.index:
            name = row[name_col]
            value = row[value_col]
            if pd.notna(name) and str(name).strip():
                name_str = str(name).strip().lower()
                value_str = str(value) if pd.notna(value) else ""
                # Map to standard keys if possible
                if name_str in standard_condition_keys:
                    conditions[name_str] = value_str
                else:
                    # Add to other_conditions
                    other_conditions_list.append(f"{name}: {value_str}")

    if other_conditions_list:
        conditions["other_conditions"] = "; ".join(other_conditions_list)

    # Build location dict
    location = {}
    location_fields = ["page", "section", "source_type", "evidence", "figure_or_table"]
    for field in location_fields:
        col = f"location.{field}"
        if col in row.index and pd.notna(row[col]):
            location[field] = str(row[col])

    # Get value_unit from the 'units' column if present
    value_unit = str(row["units"]) if pd.notna(row.get("units")) else None

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
        "value_unit": value_unit,
        "qualifier": None,
        "value_detail": None,
        "conditions": conditions,
        "method": str(row["method"]) if pd.notna(row.get("method")) else None,
        "model_or_fit": None,
        "location": location if location else None,
        "notes": str(row["notes"]) if pd.notna(row.get("notes")) else None,
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
save_path = args.output_dir / f"{args.hf_split}.csv"
save_path.parent.mkdir(parents=True, exist_ok=True)
df_grouped.to_csv(save_path, index=False)
logger.info(f"Dataset saved to {save_path}")

if True:
    # FOR DEBUGGING PURPOSES ONLY:
    # save exploded version of the dataset for easier inspection
    df_exploded = df_grouped.explode("properties").reset_index(drop=True)
    # expand the properties dict into separate columns
    properties_df = pd.json_normalize(df_exploded["properties"])
    # add back the refno column
    properties_df.insert(0, "refno", df_exploded["refno"].values)
    # rename columns for clarity
    properties_df = properties_df.rename(
        columns={"value_string": "property_value", "value_unit": "property_unit"}
    )
    # save properties_df to csv
    exploded_save_path = args.output_dir / f"{args.hf_split}_exploded.csv"
    logger.info(f"Saving exploded dataset to {exploded_save_path}...")
    properties_df.to_csv(exploded_save_path, index=False)

dataset = Dataset.from_pandas(df_grouped)
dataset.save_to_disk(args.output_dir / f"{args.hf_split}")
logger.info(f"Dataset saved to {args.output_dir / f'{args.hf_split}'}")

# Load the dataset from disk and print the first row
loaded_dataset = Dataset.load_from_disk(args.output_dir / f"{args.hf_split}")
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
if args.hf_repo is not None:
    logger.info(f"Pushing dataset to HuggingFace Hub: {args.hf_repo}...")
    logger.info(f"Uploading {len(df_grouped)} rows...")
    dataset = Dataset.from_pandas(df_grouped)
    # Note: If schema changes, you may need to delete existing data first:
    # huggingface_hub.HfApi().delete_folder(repo_id=args.hf_repo, path_in_repo="data", repo_type="dataset")
    dataset.push_to_hub(args.hf_repo, private=False, split=args.hf_split)
    logger.info(f"✓ All {len(df_grouped)} rows pushed to {args.hf_repo}")

    # Tag the dataset so that we can easily refer to different versions of the dataset
    # Ref: https://github.com/huggingface/datasets/discussions/5370
    if args.hf_revision is not None:
        huggingface_hub.create_tag(
            args.hf_repo, tag=args.hf_revision, repo_type="dataset"
        )
        logger.info(f"✓ Dataset tagged with {args.hf_revision}")
