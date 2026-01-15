"""Script to generate embeddings from ground truth property names.

Reference: https://ai.google.dev/gemini-api/docs/embeddings#task-types
"""

# standard imports
import logging
import pandas as pd
from argparse import ArgumentParser
from datasets import load_dataset

# pbench imports
import pbench
from pbench_eval.match import generate_embeddings

# local imports
from utils import HF_DATASET_NAME, HF_DATASET_REVISION, HF_DATASET_SPLIT

parser = ArgumentParser(description="Generate embeddings from property names.")
parser = pbench.add_base_args(parser)
args = parser.parse_args()
pbench.setup_logging(args.log_level)
logger = logging.getLogger(__name__)

#
# Load ground truth dataset
#
ds = load_dataset(HF_DATASET_NAME, split=HF_DATASET_SPLIT, revision=HF_DATASET_REVISION)
gt_df = ds.to_pandas()

embeddings_dir = args.output_dir / "gt_embeddings"
embeddings_dir.mkdir(parents=True, exist_ok=True)

for i, row in gt_df.iterrows():
    refno = row["refno"]
    logger.info(f"Processing refno {refno}...")
    print(f"Generating embeddings for {refno}...")
    properties = row["properties"]
    # get unique property names
    unique_property_names = list(set([prop["property_name"] for prop in properties]))
    # generate embeddings
    embeddings = generate_embeddings(unique_property_names)
    # save embeddings to parquet
    df = pd.DataFrame(
        {
            "refno": [refno] * len(unique_property_names),
            "property_name": unique_property_names,
            "embedding": embeddings,
        }
    )
    save_path = embeddings_dir / f"{refno}.parquet"
    df.to_parquet(save_path)
    print(f"Saved embeddings to {save_path} with {len(df)} rows")
