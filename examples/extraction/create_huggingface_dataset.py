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
"""Script to create a HuggingFace dataset for the property extraction task

Each row contains the following information:
- Refno (string): the reference number of the paper
- paper (PDFplumber object): the PDF file of the paper.
- property_name (string): the name of the property
- property_value (string): the value of the property

Example usage:
```bash
python create_huggingface_dataset.py --repo_name REPO_NAME
```
"""

# %%
import pandas as pd
from pathlib import Path
from datasets import Dataset, Pdf, load_from_disk
from argparse import ArgumentParser
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %%
parser = ArgumentParser()
parser.add_argument("--paper_dir", type=str, default="Paper_DB", help="Directory containing the papers")
parser.add_argument("--properties_dir", type=str, default="data", help="Directory containing the properties")
parser.add_argument("--repo_name", type=str, default=None, help="Name of the HuggingFace repository to push to")
parser.add_argument("--output_dir", type=str, default="out", help="Directory to save the dataset csv file")
args = parser.parse_args()

paper_dir = args.paper_dir
properties_dir = args.properties_dir
repo_name = args.repo_name
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# Load the glossary of properties
glossary_path = Path(properties_dir) / "properties-oxide-metal-glossary.csv"
df_glossary = pd.read_csv(glossary_path, index_col=0)
# construct a dictionary of property name -> definition
definitions = {}
for index, row in df_glossary.iterrows():
    if row['label'] in definitions:
        logger.warning(f"Warning: property name {row['label']} already exists in the definitions")
        logger.warning(f"The old definition is: {definitions[row['label']]}")
        logger.warning(f"The new definition is: {row['definition']}")
    definitions[row['label']] = row['definition']

# %%
answer_path = Path(properties_dir) / "curated_filtered_properties.csv"
# Read Properties as Python objects (avoid needing eval later)
df_answer = pd.read_csv(answer_path, converters={"Properties": eval})
# each row contains multiple properties
# explode the rows into multiple rows
df_answer = df_answer.explode('Properties')
# df_answer['num_properties'] = df_answer['Properties'].apply(len)
df_answer['paper'] = df_answer['Paper'].apply(lambda x: Path(paper_dir) / f"{x}.pdf")

# the Properties column contains a dictionary of property name -> property value pairs
# we want a separate row for each property name -> property value pair
# note that we need a new column for the property value
# After exploding the list of dictionaries, explode each dictionary into separate rows
# Convert each dictionary to a Series with items(), then explode
df_answer = df_answer.apply(lambda row: pd.Series({
    'paper': row['paper'],
    'material': row['Properties']["common formula of materials"],
    'property_name': list(row['Properties'].keys()),
    'property_value': list(row['Properties'].values())
}), axis=1).explode(['property_name', 'property_value'])
df_answer = df_answer[df_answer['property_name'] != 'common formula of materials']
# Convert Paper Path objects to strings for HuggingFace
df_answer['paper'] = df_answer['paper'].astype(str)
# Convert property_value to string to handle mixed types (float, str, etc.)
df_answer['property_value'] = df_answer['property_value'].astype(str)

# Fix unicode character issues in property names
unicode_fixes = {
    'Å€': '⊥',  # Fix perpendicular symbol
    'Ã—': '×',  # Fix multiplication sign if needed
    'Î"': 'Δ',  # Fix delta if needed
}
for bad_char, good_char in unicode_fixes.items():
    df_answer['property_name'] = df_answer['property_name'].str.replace(bad_char, good_char, regex=False)

# Add the definition of the property to the dataframe
df_answer['definition'] = df_answer['property_name'].map(definitions)
df_answer = df_answer.reset_index(drop=True)

# %%
print(df_answer)
import pdb; pdb.set_trace()
save_path = Path(output_dir) / "dataset.csv"
df_answer.to_csv(save_path, index=False)
print(f"Dataset saved to {save_path}")

# %%
# Create HuggingFace dataset with proper PDF feature type
dataset = Dataset.from_pandas(df_answer)

# Cast the paper column to use the Pdf feature type
# decode=True will allow the dataset to load PDFs as pdfplumber objects
# decode=False will keep them as {"path": ..., "bytes": ...} dictionaries
dataset = dataset.cast_column("paper", Pdf(decode=True))

# %%
print(dataset)
print(f"\nDataset features: {dataset.features}")
print(f"Number of examples: {len(dataset)}")

# %%
# Save the dataset locally
dataset.save_to_disk("property_extraction_dataset")

# %%
# Load the dataset back to verify it works
loaded_dataset = load_from_disk("property_extraction_dataset")

print("✓ Dataset loaded successfully!")
print(f"Number of examples: {len(loaded_dataset)}")
print(f"Features: {loaded_dataset.features}")
print("\nFirst example:")
print(f"  material: {loaded_dataset[0]['material']}")
print(f"  property_name: {loaded_dataset[0]['property_name']}")
print(f"  property_value: {loaded_dataset[0]['property_value']}")
print(f"  paper: {loaded_dataset[0]['paper']}")

# %%
# Test that PDF loading works
print("\n✓ Testing PDF loading...")
pdf_obj = loaded_dataset[0]['paper']
print(f"PDF object type: {type(pdf_obj)}")
print(f"PDF has {len(pdf_obj.pages)} pages")

# %%
# Push to HuggingFace Hub (requires authentication)
# Make sure you're logged in: huggingface-cli login
if repo_name is not None:
    print(f"Pushing dataset to HuggingFace Hub: {repo_name}")
    dataset.push_to_hub(
        repo_name,
        private=False,  # Set to True if you want a private dataset
        split="test",
    )
