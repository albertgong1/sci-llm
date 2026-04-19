from datasets import load_dataset
from argparse import ArgumentParser
import logging
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import fitz  # PyMuPDF

import pbench

from utils import (
    HF_DATASET_NAME,
    HF_DATASET_REVISION,
    HF_DATASET_SPLIT,
)

parser = ArgumentParser(
    description="Plot dataset statistics for SuperCon Extraction dataset"
)
parser = pbench.add_base_args(parser)
args = parser.parse_args()
pbench.setup_logging(args.log_level)

logger = logging.getLogger(__name__)

# get GT dataset
hf_dataset_name = args.hf_repo or HF_DATASET_NAME
hf_split_name = args.hf_split or HF_DATASET_SPLIT
hf_revision = args.hf_revision or HF_DATASET_REVISION
logger.info(
    f"Loading dataset from HuggingFace: {hf_dataset_name} "
    f"(revision={hf_revision}, split={hf_split_name})"
)
dataset = load_dataset(hf_dataset_name, split=hf_split_name, revision=hf_revision)
df = dataset.to_pandas()

#
# Compute page statistics
#
paper_dir = args.data_dir / "Paper_DB"


def get_page_count(refno: str) -> int | None:
    """Get page count for a paper given its refno.

    Args:
        refno: reference number of the paper (without .pdf extension)

    Returns:
        Number of pages in the paper, or None if the paper is not found

    """
    paper_path = paper_dir / f"{refno}.pdf"
    if not paper_path.exists():
        return None
    doc = fitz.open(paper_path)
    num_pages = len(doc)
    doc.close()
    return num_pages


df["pages"] = df["refno"].apply(get_page_count)
print(df["pages"].describe())

#
# Compute property statistics
#
df["num_properties"] = df["properties"].apply(lambda x: len(x))
print(df["num_properties"].describe())

#
# Compute proportion of categories
#
# NOTE: use the "category" column from the rubric
df = df.explode(column="properties").reset_index(drop=True)
df = pd.concat(
    [df.drop(columns=["properties"]), pd.json_normalize(df["properties"])], axis=1
)
# drop category from GT datasets since we will re-assign from rubric
df = df.drop(columns=["category"])
# Load rubric
rubric_path = Path("scoring") / "rubric_2.csv"
logger.info(f"Loading rubric from {rubric_path}")
df_rubric = pd.read_csv(rubric_path)
logger.info(f"Loaded {len(df_rubric)} rows from rubric")
# Join matches with rubric to get scoring method
logger.info("Joining matches with rubric...")
df = df.merge(
    df_rubric[["property_name", "category", "rubric"]],
    on="property_name",
    how="left",
)
category_counts = df["category"].value_counts(normalize=True)
# save category counts as a CSV
category_counts_path = args.output_dir / "dataset_category_distribution.csv"
category_counts.to_csv(category_counts_path, header=["proportion"])
# plot pie chart
fig_dir = args.output_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(6.75, 2))
plt.pie(
    category_counts.values,
    labels=category_counts.index,
    autopct="%1.1f%%",
    startangle=90,
)
plt.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
fig_path = fig_dir / "dataset_category_distribution.pdf"
plt.savefig(fig_path)
