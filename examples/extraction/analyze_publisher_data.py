"""Script to analyze publisher data by joining SuperConDOI verdicts with journal publisher information.

This script:
1. Filters SuperConDOI_LLM_verdict.csv for rows where llm_verdict is True
2. Performs a left join with journal_publishers_consolidated.csv
3. Summarizes missing publisher information
4. Saves the results to a CSV file
"""

import pandas as pd
from pathlib import Path

# Define file paths
data_dir = Path(__file__).parent / "data"
supercon_file = data_dir / "SuperConDOI_LLM_verdict.csv"
publishers_file = data_dir / "journal_publishers_consolidated.csv"
output_file = data_dir / "supercon_with_publishers.csv"

# Read the SuperConDOI file
print("Reading SuperConDOI_LLM_verdict.csv...")
supercon_df = pd.read_csv(supercon_file)
print(f"Total rows in SuperConDOI file: {len(supercon_df)}")

# Filter for llm_verdict == True
print("\nFiltering for rows where llm_verdict is True...")
filtered_df = supercon_df[supercon_df["llm_verdict"]]
print(f"Rows with llm_verdict=True: {len(filtered_df)}")

# Apply aliases to output_journal column to improve matching
print("\nApplying journal name aliases...")
journal_aliases = {
    "Physica C: Superconductivity and its Applications": "Physica C: Superconductivity",
    "Physica B: Condensed Matter": "Physica B",
    "EPL (Europhysics Letters)": "Europhysics Letters (EPL)",
    "physica status solidi (b)": "Physica Status Solidi (b)",
    "Zeitschrift f�r Physik": "Zeitschrift fur Physik",
    "Zeitschrift f�r Physik B Condensed Matter": "Zeitschrift fur Physik",
    "Reviews of Modern Physics": "Review of Modern Physics",
    "Journal of Materials Science Letters": "Journal of Materials Science",
    "Zeitschrift für Physik A Hadrons and nuclei": "Zeitschrift fur Physik",
    "Zeitschrift für Physik": "Zeitschrift fur Physik",
    "Physics Letters": "Physics Letters A",
    "Materials Science and Engineering: B": "Materials Science and Engineering",
    "Applied Physics A": "Applied Physics",
    "Zeitschrift f�r Physik B Condensed Matter and Quanta": "Zeitschrift fur Physik",
    "Materials Science and Engineering: A": "Materials Science and Engineering",
    "physica status solidi c": "Physica Status Solidi (c)",
}
filtered_df = filtered_df.copy()
filtered_df["output_journal"] = filtered_df["output_journal"].replace(journal_aliases)  # type: ignore
journals_aliased = sum(
    supercon_df[supercon_df["llm_verdict"]]["output_journal"].isin(
        list(journal_aliases.keys())
    )  # type: ignore
)
print(f"Applied aliases to {journals_aliased} journal entries")

# Read the publishers file
print("\nReading journal_publishers_consolidated.csv...")
publishers_df = pd.read_csv(publishers_file)
print(f"Total journal publishers: {len(publishers_df)}")

# Perform left join
print("\nPerforming left join on output_journal and journal_name...")
merged_df = filtered_df.merge(
    publishers_df, left_on="output_journal", right_on="journal_name", how="left"
)
print(f"Total rows after join: {len(merged_df)}")

# Analyze missing publisher info
print("\n" + "=" * 60)
print("SUMMARY OF MISSING PUBLISHER INFORMATION")
print("=" * 60)

# Count rows with missing publisher info (where journal_name is NaN after join)
missing_publisher = merged_df["journal_name"].isna()
num_missing = missing_publisher.sum()
num_with_publisher = len(merged_df) - num_missing

print(f"\nTotal rows (llm_verdict=True): {len(merged_df)}")
print(
    f"Rows with publisher info: {num_with_publisher} ({100 * num_with_publisher / len(merged_df):.2f}%)"
)
print(
    f"Rows missing publisher info: {num_missing} ({100 * num_missing / len(merged_df):.2f}%)"
)

# Show unique journals without publisher info
if num_missing > 0:
    print("\nUnique journals without publisher information:")
    missing_df = merged_df[missing_publisher]
    missing_journals = missing_df["output_journal"].unique()  # type: ignore
    print(f"Number of unique journals: {len(missing_journals)}")
    print("\nAll missing journals:")
    missing_journal_counts = missing_df["output_journal"].value_counts()  # type: ignore
    for journal, count in missing_journal_counts.items():
        print(f"  - {journal}: {count} occurrences")

# Save the merged data
print(f"\nSaving merged data to {output_file}...")
merged_df.to_csv(output_file, index=False)
print("Done!")
