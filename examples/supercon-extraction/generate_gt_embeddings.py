"""Script to generate embeddings from ground truth property names.

Reference: https://ai.google.dev/gemini-api/docs/embeddings#task-types
"""

# standard imports
import logging
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
from datasets import load_dataset
import json

# third-party imports
from dotenv import load_dotenv
from slugify import slugify

# pbench imports
import pbench
from pbench_eval.match import generate_embeddings

# local imports
from utils import (
    HF_DATASET_NAME,
    HF_DATASET_REVISION,
    HF_DATASET_SPLIT,
)

# Load env variables
load_dotenv()

parser = ArgumentParser(description="Generate embeddings from property names.")
parser = pbench.add_base_args(parser)
args = parser.parse_args()
pbench.setup_logging(args.log_level)
logger = logging.getLogger(__name__)

repo_name = args.hf_repo or HF_DATASET_NAME
revision = args.hf_revision or HF_DATASET_REVISION
split = args.hf_split or HF_DATASET_SPLIT

# Generate output path from repo_name, split, and revision
output_filename = f"embeddings_{slugify(f'{repo_name}_{split}_{revision}')}.json"
output_path = Path("scoring") / output_filename

#
# Load ground truth dataset
#
ds = load_dataset(repo_name, split=split, revision=revision)
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

logger.info(f"Saving {len(data)} ground truth embeddings to {output_path}...")
with open(output_path, "w") as f:
    json.dump(data, f, indent=2)
logger.info("Done.")
