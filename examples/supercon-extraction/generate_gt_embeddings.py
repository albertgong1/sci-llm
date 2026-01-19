"""Script to generate embeddings from ground truth property names.

Reference: https://ai.google.dev/gemini-api/docs/embeddings#task-types
"""

# standard imports
import logging
import pandas as pd
from argparse import ArgumentParser
from datasets import load_dataset
import json

# pbench imports
import pbench
from pbench_eval.match import generate_embeddings

# local imports
from utils import (
    HF_DATASET_NAME,
    HF_DATASET_REVISION,
    HF_DATASET_SPLIT,
    GT_EMBEDDINGS_PATH,
)

parser = ArgumentParser(description="Generate embeddings from property names.")
parser = pbench.add_base_args(parser)
args = parser.parse_args()
pbench.setup_logging(args.log_level)
logger = logging.getLogger(__name__)

#
# Load ground truth dataset
#
ds = load_dataset(HF_DATASET_NAME, split=HF_DATASET_SPLIT, revision=HF_DATASET_REVISION)
df = ds.to_pandas()
# get unique property_names
df = df.explode(column="properties").reset_index(drop=True)
df = pd.concat(
    [df.drop(columns=["properties"]), pd.json_normalize(df["properties"])], axis=1
)
unique_property_names = df["property_name"].dropna().unique().tolist()

embeddings = generate_embeddings(unique_property_names)
# save embeddings to parquet
data = []
for name, embedding in zip(unique_property_names, embeddings):
    data.append(
        {
            "property_name": name,
            "embedding": embedding,
        }
    )

logger.info(f"Saving {len(data)} ground truth embeddings to {GT_EMBEDDINGS_PATH}...")
with open(GT_EMBEDDINGS_PATH, "w") as f:
    json.dump(data, f, indent=2)
logger.info("Done.")
