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
"""Script to create a HuggingFace dataset for property extraction

The model will be provided the following information:
- Paper path
- Chemical formula
- Property to extract

There will be a column for the answer.

Example usage:
```bash
python create_huggingface_dataset.py
```
"""

# %%
import pandas as pd
from pathlib import Path

# %%
data_dir = "data"
properties_path = Path(data_dir) / "properties-oxide-metal.csv"
df_properties = pd.read_csv(properties_path, index_col=0)


# %%
df_properties

# %%
for row in df_properties.itertuples():
    print(row.label)

# %%
