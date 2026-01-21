"""Script to format SuperCon extraction prompt with targeted properties from the SuperCon database.

Formats the properties as:

### Category 1
- Property Name 1
- Property Name 2
...

### Category 2
- Property Name 3
- Property Name 4
...
...
and so on for all categories in the CSV file.
"""

import pandas as pd

df = pd.read_csv("scoring/rubric_3.csv")

for category, group in df.groupby("category"):
    property_names = group["property_name"].tolist()
    print(f"### {category}")
    for prop in property_names:
        print(f"- {prop}")
    print()  # Add a blank line between categories
