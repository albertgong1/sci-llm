"""Script to generate embeddings from ground truth property names.

Reference: https://ai.google.dev/gemini-api/docs/embeddings#task-types
"""

# standard imports
import logging
import pandas as pd
from argparse import ArgumentParser
from datasets import load_dataset

# llm imports
from google import genai
from google.genai import types

# pbench imports
import pbench

BATCH_SIZE = 100

parser = ArgumentParser(description="Generate embeddings from property names.")
parser = pbench.add_base_args(parser)
args = parser.parse_args()
pbench.setup_logging(args.log_level)
logger = logging.getLogger(__name__)

#
# Load ground truth dataset
#
ds = load_dataset(args.dataset, split=args.split, revision="v2.0.1")
gt_df = ds.to_pandas()

embeddings_dir = args.output_dir / "gt_embeddings"
embeddings_dir.mkdir(parents=True, exist_ok=True)

client = genai.Client()

for i, row in gt_df.iterrows():
    refno = row["refno"]
    logger.info(f"Processing refno {refno}...")
    print(f"Generating embeddings for {refno}...")
    properties = row["properties"]
    # get unique property names
    unique_property_names = list(set([prop["property_name"] for prop in properties]))
    # generate embeddings in batches
    embeddings = []
    for i in range(0, len(unique_property_names), BATCH_SIZE):
        batch = unique_property_names[i : i + BATCH_SIZE]
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=batch,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        )
        embeddings.extend([emb.values for emb in result.embeddings])

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
