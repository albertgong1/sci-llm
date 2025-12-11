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

Each row contains the following information:
- paper (PDFplumber object): the PDF file of the paper (or path to the PDF file).
- material (string): the chemical formula of the material
- property_name (string): the name of the property
- property_value (string): the value of the property
- property_unit (string): the unit of the property
- definition (string): the definition of the property

Example usage:
```bash
python create_huggingface_dataset.py --repo_name REPO_NAME
```
"""

# %%
import pandas as pd
from pathlib import Path
from datasets import Dataset, load_from_disk
from argparse import ArgumentParser
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %%
parser = ArgumentParser()
parser.add_argument(
    "--paper_dir", type=str, default="Paper_DB", help="Directory containing the papers"
)
parser.add_argument(
    "--properties_dir",
    type=str,
    default="data",
    help="Directory containing the properties",
)
parser.add_argument(
    "--repo_name",
    type=str,
    default=None,
    help="Name of the HuggingFace repository to push to",
)
parser.add_argument(
    "--output_dir",
    "-od",
    type=str,
    default="out",
    help="Directory to save the dataset csv file",
)
args = parser.parse_args()

paper_dir = args.paper_dir
properties_dir = args.properties_dir
repo_name = args.repo_name
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

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
data_path = Path(properties_dir) / "SuperCon.csv"
# Use the second row as the header
df = pd.read_csv(data_path, header=2, dtype=str)
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


df = df.apply(process_row, axis=1)
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

if True:
    df_copy = df.copy()
    # Filter to only include papers that have PDFs (from curated_filtered_properties.csv)
    curated_path = Path(properties_dir) / "curated_filtered_properties.csv"
    df_curated = pd.read_csv(curated_path)
    # Create mapping from Refno to Paper name
    refno_to_paper = dict(zip(df_curated["Refno"], df_curated["Paper"]))
    # Filter df to only include refnos that have PDFs
    df_copy = df_copy[df_copy["refno"].isin(refno_to_paper.keys())]

    # Add paper column with path to PDF
    df_copy["paper"] = (
        df_copy["refno"]
        .map(refno_to_paper)
        .apply(lambda x: Path(paper_dir) / f"{x}.pdf")
    )
    # Convert Path objects to strings for HuggingFace
    df_copy["paper"] = df_copy["paper"].astype(str)
    # sort the dataframe by paper, but keep the order within each paper
    df_copy = df_copy.sort_values(["paper", "material", "order"], kind="stable")
    df_copy = df_copy.reset_index(drop=True)

    # only keep columns paper, material, property_name, property_value, property_unit, definition
    df_copy = df_copy[
        [
            "paper",
            "material",
            "task",
            "property_name",
            "property_value",
            "property_unit",
            "definition",
        ]
    ]

    print(df_copy)
    save_path = Path(output_dir) / "dataset.csv"
    df_copy.to_csv(save_path, index=False)
    print(f"Dataset saved to {save_path}")

# %%
# Create HuggingFace dataset with proper PDF feature type
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
dataset = Dataset.from_pandas(df[COLUMNS], preserve_index=False)

# Cast the paper column to use the Pdf feature type
# decode=True will allow the dataset to load PDFs as pdfplumber objects
# decode=False will keep them as {"path": ..., "bytes": ...} dictionaries
# dataset = dataset.cast_column("paper", Pdf(decode=True))

# %%
print(dataset)
print(f"\nDataset features: {dataset.features}")
print(f"Number of examples: {len(dataset)}")

# %%
# Save the dataset locally
dataset_path = Path(output_dir) / "property_extraction_dataset"
dataset.save_to_disk(dataset_path)

# %%
# Load the dataset back to verify it works
loaded_dataset = load_from_disk(dataset_path)

print("✓ Dataset loaded successfully!")
print(f"Number of examples: {len(loaded_dataset)}")
print(f"Features: {loaded_dataset.features}")
print("\nFirst example:")
print(f"  material: {loaded_dataset[0]['material']}")
print(f"  property_name: {loaded_dataset[0]['property_name']}")
print(f"  property_value: {loaded_dataset[0]['property_value']}")
# print(f"  paper: {loaded_dataset[0]['paper']}")
print(f"  refno: {loaded_dataset[0]['refno']}")

# %%
# Test that PDF loading works
# print("\n✓ Testing PDF loading...")
# pdf_obj = loaded_dataset[0]['paper']
# print(f"PDF object type: {type(pdf_obj)}")
# print(f"PDF has {len(pdf_obj.pages)} pages")

# %%
# Push to HuggingFace Hub (requires authentication)
# Make sure you're logged in: huggingface-cli login
if repo_name is not None:
    print(f"Pushing dataset to HuggingFace Hub: {repo_name}")
    dataset.push_to_hub(
        repo_name,
        private=True,  # Set to True if you want a private dataset
        split="test",
    )
