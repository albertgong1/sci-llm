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
from datasets import Dataset
from argparse import ArgumentParser
import logging
from joblib import Parallel, delayed
import huggingface_hub

import pbench

from utils import (
    ANALM,
    CRYSTAL_SYMMETRY,
    GAPMETH,
    SHAPE,
    METHOD,
    TC_MEASUREMENT_METHOD,
)

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
# create a lookup table for db -> property name
db_to_property_name_lookup = df_glossary["label"].to_dict()
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
    properties = []
    prop_id = 0
    for col in df.columns:
        if col in ["refno", "element"]:
            continue
        if col in df_units["unit"].values:
            # Skip properties that are units
            continue
        # Skip properties that are not physically relevant
        if col not in df_units.index:
            continue
        if pd.isna(value := row[col]):
            continue
        # get the unit for the property
        if pd.notna(df_units.loc[col, "unit"]):
            unit_value = row[df_units.loc[col, "unit"]]
            value_string = f"{value} {unit_value}"
        elif col == "shape":  # shape
            unit_value = None
            value_string = SHAPE.get(int(value), "")
        elif col == "str1":  # crystal symmetry
            unit_value = None
            value_string = CRYSTAL_SYMMETRY.get(int(value), "")
        elif col == "method":  # preparation method
            unit_value = None
            value_string = METHOD.get(int(value), "")
        else:
            unit_value = None
            value_string = str(value) if pd.notna(value) else ""

        # Build conditions dict
        conditions = {}
        # Get temperature condition
        temp_col = df_units.loc[col, "conditions.temperature"]
        if pd.notna(temp_col) and temp_col in row.index and pd.notna(row[temp_col]):
            conditions["temperature"] = str(row[temp_col])
        else:
            conditions["temperature"] = ""

        # Get field condition
        field_col = df_units.loc[col, "conditions.field"]
        if pd.notna(field_col) and field_col in row.index and pd.notna(row[field_col]):
            conditions["field"] = str(row[field_col])
        else:
            conditions["field"] = ""

        # Build location dict
        location = {}
        # Get figure or table reference
        fig_table_col = df_units.loc[col, "location.figure_or_table"]
        if (
            pd.notna(fig_table_col)
            and fig_table_col in row.index
            and pd.notna(row[fig_table_col])
        ):
            location["figure_or_table"] = str(row[fig_table_col])

        # Get method
        method = None
        method_col = df_units.loc[col, "methods"]
        if (
            pd.notna(method_col)
            and method_col in row.index
            and pd.notna(row[method_col])
        ):
            if method_col == "analm":  # *method of analysis for structure
                method = ANALM.get(int(row[method_col]), "")
            elif method_col == "gapmeth":  # method of measuring energy gap
                method = GAPMETH.get(int(row[method_col]), "")
            elif method_col == "tcmeth":  # TC measurement method
                # import pdb; pdb.set_trace()
                method = TC_MEASUREMENT_METHOD.get(int(row[method_col]), "")
            else:
                method = str(row[method_col])

        properties.append(
            {
                "id": f"prop_{prop_id:03d}",
                "material_or_system": row[
                    "element"
                ],  # TODO (Albert): normalize using pymatgen
                "sample_label": None,
                "property_name": db_to_property_name_lookup.get(col, col),
                "category": None,
                "value_string": value_string,
                "value_unit": unit_value,
                "qualifier": None,
                "value_detail": None,
                "conditions": conditions if conditions else None,
                "method": method,
                "model_or_fit": None,
                "location": location if location else None,
                "notes": None,
            }
        )
        prop_id += 1
    return pd.Series(
        {
            "refno": row["refno"],
            "properties": properties,
        }
    )


results = Parallel(n_jobs=-1, batch_size=100, verbose=1)(
    delayed(process_row)(row) for _, row in df.iterrows()
)
df = pd.DataFrame(results)

# Group by refno and combine properties
df = df.groupby("refno", as_index=False).agg(
    {"properties": lambda x: [prop for props_list in x for prop in props_list]}
)

if args.filter_pdf:
    logger.info("Filtering out rows where paper PDF is not found")
    df = df[df["refno"].apply(lambda x: Path(paper_dir / f"{x}.pdf").exists())]
    logger.info(f"{len(df)} rows have paper PDF")

logger.info("Saving dataset to CSV...")
save_path = args.output_dir / f"{args.split}.csv"
save_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(save_path, index=False)
logger.info(f"Dataset saved to {save_path}")

if False:
    # FOR DEBUGGING PURPOSES ONLY:
    # save exploded version of the dataset for easier inspection
    df_exploded = df.explode("properties").reset_index(drop=True)
    # expand the properties dict into separate columns
    properties_df = pd.json_normalize(df_exploded["properties"])
    # add back the refno column
    properties_df.insert(0, "refno", df_exploded["refno"].values)
    # rename columns for clarity
    properties_df = properties_df.rename(
        columns={"value_string": "property_value", "value_unit": "property_unit"}
    )
    # create reverse lookup from property name to db label
    property_name_to_db = {v: k for k, v in db_to_property_name_lookup.items()}
    # merge db label from glossary and insert before property_name
    db_label_values = properties_df["property_name"].map(property_name_to_db)
    property_name_idx = properties_df.columns.get_loc("property_name")
    properties_df.insert(property_name_idx, "db_label", db_label_values)
    # import pdb; pdb.set_trace()
    # save properties_df to csv
    exploded_save_path = args.output_dir / f"{args.split}_exploded.csv"
    logger.info(f"Saving exploded dataset to {exploded_save_path}...")
    properties_df.to_csv(exploded_save_path, index=False)
    exit(0)

dataset = Dataset.from_pandas(df)
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
    logger.info(f"Uploading {len(df)} rows...")
    dataset = Dataset.from_pandas(df)
    # Note: If schema changes, you may need to delete existing data first:
    # huggingface_hub.HfApi().delete_folder(repo_id=args.repo_name, path_in_repo="data", repo_type="dataset")
    dataset.push_to_hub(args.repo_name, private=False, split=args.split)
    logger.info(f"✓ All {len(df)} rows pushed to {args.repo_name}")

    # Tag the dataset so that we can easily refer to different versions of the dataset
    # Ref: https://github.com/huggingface/datasets/discussions/5370
    if args.tag_name is not None:
        huggingface_hub.create_tag(
            args.repo_name, tag=args.tag_name, repo_type="dataset"
        )
        logger.info(f"✓ Dataset tagged with {args.tag_name}")
