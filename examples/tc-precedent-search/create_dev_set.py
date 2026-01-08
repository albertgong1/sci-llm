from pathlib import Path
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

repo_root = Path(__file__).resolve().parents[2]
input_path = repo_root / "examples/tc-precedent-search/SuperCon_Tc_Tcn - no NA.csv"
output_path = repo_root / "examples/tc-precedent-search/SuperCon_Tc_Tcn_dev-set.csv"

print(f"Loading {input_path}...")
df = pd.read_csv(input_path)

# Filter by superconductivity status
yes_df = df[df['Has material been reported to be superconducting?'] == 'Yes']
no_df = df[df['Has material been reported to be superconducting?'] == 'No']

print(f"Total 'Yes' rows: {len(yes_df)}")
print(f"Total 'No' rows: {len(no_df)}")

# Check if we have enough data
if len(yes_df) < 100 or len(no_df) < 100:
    print("Warning: Not enough rows to sample 100 from each category.")
    # We should have enough based on previous steps (21k rows) but just a sanity check

# Randomly sample 100 from each
print("Sampling 100 rows from each category...")
sample_yes = yes_df.sample(n=100, random_state=42)
sample_no = no_df.sample(n=100, random_state=42)

# Combine
dev_set = pd.concat([sample_yes, sample_no])

# Shuffle the combined set so they are mixed (optional, but good for inspection)
dev_set = dev_set.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Dev set created with {len(dev_set)} rows.")
print(dev_set['Has material been reported to be superconducting?'].value_counts())

print(f"Saving to {output_path}...")
dev_set.to_csv(output_path, index=False)
print("Done.")
