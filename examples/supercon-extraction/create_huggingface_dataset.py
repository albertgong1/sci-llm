#! /usr/bin/env -S uv run --env-file=.env -- python
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
"""Script to create a HuggingFace dataset from the SuperCon dataset.

NOTE: we deduplicates rows that have the same material, property_name, property_value, and property_unit.
NOTE: skip the --repo_name argument if you don't want to push to HuggingFace Hub.

Each row contains the following information:
- paper (PDFplumber object): the PDF file of the paper (or path to the PDF file).
- material (string): the chemical formula of the material
- property_name (string): the name of the property
- property_value (string): the value of the property
- property_unit (string): the unit of the property
- definition (string): the definition of the property

Usage:
```bash
# To filter out rows where paper PDF is not found at data_dir/supercon/Paper_DB
./src/pbench/create_supercon_hf_dataset.py \
    --data_dir data \
    --repo_name REPO_NAME \
    --filter_pdf

# To use all rows (including rows where paper PDF is not found at data_dir/supercon/Paper_DB)
./src/pbench/create_supercon_hf_dataset.py \
    --data_dir data \
    --repo_name REPO_NAME
```
"""

# %%
import pandas as pd
from pathlib import Path
from datasets import Dataset, load_from_disk
from argparse import ArgumentParser
import logging
from joblib import Parallel, delayed

import pbench

logger = logging.getLogger(__name__)

# %%
parser = ArgumentParser(
    description="Create a HuggingFace dataset from the SuperCon dataset."
)
parser = pbench.add_base_args(parser)
parser.add_argument(
    "--repo_name",
    type=str,
    default=None,
    help="Name of the HuggingFace repository to push to",
)
parser.add_argument(
    "--filter_pdf",
    action="store_true",
    help="If true, filter out rows where paper PDF is not found at data_dir/Paper_DB",
)
args = parser.parse_args()
pbench.setup_logging(args.log_level)

paper_dir = args.data_dir / "Paper_DB"
args.output_dir.mkdir(parents=True, exist_ok=True)

# Load the glossary of properties
glossary_path = "properties-oxide-metal-glossary.csv"
df_glossary = pd.read_csv(
    glossary_path,
    index_col=0,
)
# set index to db and rename index to "order"
df_glossary = df_glossary.reset_index(names="order").set_index("db")
# import pdb; pdb.set_trace()
# load units
units_path = "property_unit_mappings.csv"
df_units = pd.read_csv(units_path, index_col=0)
# import pdb; pdb.set_trace()

# %%
data_path = args.data_dir / "SuperCon.csv"
logger.info(f"Loading dataset from {data_path}...")
# Use the second row as the header
df = pd.read_csv(data_path, header=2, dtype=str)
logger.info(f"Loaded {len(df)} rows")
# drop "year.1" as it's duplicates of "year"
df = df.drop(columns=["year.1"])


def process_row(row: pd.Series) -> pd.Series:
    """Process a single row of the SuperCon dataset.

    Args:
        row: a single row of the SuperCon dataset

    Returns:
        a processed row of the SuperCon dataset

    """
    # element corresponds to the label "chemical formula"
    keys = []
    values = []
    units = []
    for col in df.columns:
        if col in ["refno", "element"]:
            continue
        if col in df_units["unit"].values:
            # Skip properties that are units
            continue
        keys.append(col)
        values.append(row[col])
        # get the unit for the property
        if col in df_units.index:
            unit_key = df_units.loc[col, "unit"]
            unit_value = row[unit_key]
            units.append(unit_value)
        else:
            units.append(None)
    return pd.Series(
        {
            "refno": row["refno"],
            "material": row["element"],
            "property_name": keys,
            "property_value": values,
            "property_unit": units,
        }
    )


results = Parallel(n_jobs=-1, batch_size=100, verbose=1)(
    delayed(process_row)(row) for _, row in df.iterrows()
)
df = pd.DataFrame(results)
# explode the property_name and property_value columns
df = df.explode(["property_name", "property_value", "property_unit"])
# drop rows where property_value is None
df = df[df["property_value"].notna()]
# join the property_name and property_value columns with the glossary of properties
df = df.merge(
    df_glossary, left_on="property_name", right_on="db", how="left", sort=False
)
# drop the rows that are not physically relevant
df = df[df["Physically_Relevant"]]

# Drop duplicate (material, property_name, property_value) combinations
# This removes duplicates when the same material appears in multiple SuperCon rows
df = df.drop_duplicates(
    subset=["material", "property_name", "property_value", "property_unit"],
    keep="first",
)
# rename label to property_name
# we will use the original property_name as the task in Harbor and the config in HuggingFace
df = df.rename(columns={"property_name": "task", "label": "property_name"})

if args.filter_pdf:
    logger.info("Filtering out rows where paper PDF is not found")
    df = df[df["refno"].apply(lambda x: Path(paper_dir / f"{x}.pdf").exists())]
    logger.info(f"{len(df)} rows have paper PDF")

logger.info("Saving dataset to CSV...")
df_copy = (
    df.copy()
    .sort_values(["refno", "material", "order"], kind="stable")
    .reset_index(drop=True)[
        [
            "refno",
            "material",
            "task",
            "property_name",
            "property_value",
            "property_unit",
            "definition",
        ]
    ]
)
save_path = args.output_dir / "property_extraction_dataset" / "dataset.csv"
save_path.parent.mkdir(parents=True, exist_ok=True)
df_copy.to_csv(save_path, index=False)
logger.info(f"Dataset saved to {save_path}")

# %%
logger.info("Creating HuggingFace dataset...")
# Group by task and create separate configs for each task
COLUMNS = [
    "refno",
    "material",
    "task",
    "property_value",
    "property_unit",
    "property_name",
    "definition",
    "category",
    "data_type",
]

# Group dataframe by task
grouped_by_task = df[COLUMNS].groupby("task")
logger.info(f"Found {len(grouped_by_task)} unique tasks:")
for task_name, task_df in grouped_by_task:
    logger.info(f"  - {task_name}: {len(task_df)} examples")

# %%
# Create and save datasets for each task
datasets_dict = {}
for task_name, task_df in grouped_by_task:
    task_dataset = Dataset.from_pandas(task_df, preserve_index=False)
    datasets_dict[task_name] = task_dataset

    # Save each task config locally
    task_dataset_path = args.output_dir / "property_extraction_dataset" / task_name
    task_dataset.save_to_disk(task_dataset_path)
    logger.info(f"Saved {task_name} config to {task_dataset_path}")

# %%
# Load one dataset back to verify it works
first_task = list(datasets_dict.keys())[0]
first_task_path = args.output_dir / "property_extraction_dataset" / first_task
loaded_dataset = load_from_disk(first_task_path)

print(f"\n✓ Dataset loaded successfully (task: {first_task})!")
print(f"Number of examples: {len(loaded_dataset)}")
print(f"Features: {loaded_dataset.features}")
print("\nFirst example:")
print(f"  material: {loaded_dataset[0]['material']}")
print(f"  property_name: {loaded_dataset[0]['property_name']}")
print(f"  property_value: {loaded_dataset[0]['property_value']}")
print(f"  task: {loaded_dataset[0]['task']}")
print(f"  refno: {loaded_dataset[0]['refno']}")

# %%
# Push to HuggingFace Hub (requires authentication)
# Make sure you're logged in: hf auth login
if args.repo_name is not None:
    logger.info(f"Pushing dataset to HuggingFace Hub: {args.repo_name}")
    logger.info(f"Uploading {len(datasets_dict)} configs...")

    for task_name, task_dataset in datasets_dict.items():
        logger.info(f"  Pushing config: {task_name} ({len(task_dataset)} examples)")
        task_dataset.push_to_hub(
            args.repo_name,
            config_name=task_name,
            private=True,  # Set to True if you want a private dataset
            split="test",
        )
        logger.info(f"  ✓ Config {task_name} pushed successfully")

    logger.info(f"✓ All {len(datasets_dict)} configs pushed to {args.repo_name}")
