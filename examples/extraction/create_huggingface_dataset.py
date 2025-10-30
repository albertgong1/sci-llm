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

The model will be provided the following information:
- Refno (string): the reference number of the paper
- Paper (PDF object): the PDF object of the paper
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

# %%
parser = ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data")
parser.add_argument("--repo_name", type=str, default=None)
args = parser.parse_args()

data_dir = args.data_dir
repo_name = args.repo_name

# TODO: get the list of all properties
# properties_path = Path(data_dir) / "properties-oxide-metal.csv"
# df_properties = pd.read_csv(properties_path, index_col=0)

# %%
answer_path = Path(data_dir) / "curated_filtered_properties.csv"
# Read Properties as Python objects (avoid needing eval later)
df_answer = pd.read_csv(answer_path, converters={"Properties": eval})
# each row contains multiple properties
# explode the rows into multiple rows
df_answer = df_answer.explode('Properties')
# df_answer['num_properties'] = df_answer['Properties'].apply(len)
df_answer['Paper'] = df_answer['Paper'].apply(lambda x: Path(data_dir) / "Paper_DB" / f"{x}.pdf")

# the Properties column contains a dictionary of property name -> property value pairs
# we want a separate row for each property name -> property value pair
# note that we need a new column for the property value
# After exploding the list of dictionaries, explode each dictionary into separate rows
# Convert each dictionary to a Series with items(), then explode
df_answer = df_answer.apply(lambda row: pd.Series({
    'Paper': row['Paper'],
    'Refno': row['Refno'],
    'property_name': list(row['Properties'].keys()),
    'property_value': list(row['Properties'].values())
}), axis=1).explode(['property_name', 'property_value'])

# %%
print(df_answer)

# %%
# Convert Paper Path objects to strings for HuggingFace
df_answer['paper'] = df_answer['Paper'].astype(str)

# Drop the Path object column
df_answer = df_answer.drop(columns=['Paper'])

# Convert property_value to string to handle mixed types (float, str, etc.)
df_answer['property_value'] = df_answer['property_value'].astype(str)

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
print(f"  Refno: {loaded_dataset[0]['Refno']}")
print(f"  Property name: {loaded_dataset[0]['property_name']}")
print(f"  Property value: {loaded_dataset[0]['property_value']}")
print(f"  Paper: {loaded_dataset[0]['paper']}")

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
    )
