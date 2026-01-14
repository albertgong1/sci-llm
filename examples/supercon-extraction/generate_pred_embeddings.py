"""Script to generate embeddings from property names.

Reference: https://ai.google.dev/gemini-api/docs/embeddings#task-types
"""

# standard imports
import pandas as pd
from argparse import ArgumentParser

# pbench imports
import pbench
from pbench_eval.match import generate_embeddings

parser = ArgumentParser(description="Generate embeddings from property names.")
parser = pbench.add_base_args(parser)
args = parser.parse_args()
pbench.setup_logging(args.log_level)

preds_dir = args.output_dir / "unsupervised_llm_extraction"
preds_files = list(preds_dir.glob("*.csv"))
if not preds_files:
    raise FileNotFoundError(f"No CSV files found in {preds_dir}")

embeddings_dir = args.output_dir / "pred_embeddings"
embeddings_dir.mkdir(parents=True, exist_ok=True)

for file in preds_files:
    print(f"Generating embeddings for {file.stem}...")
    preds_df = pd.read_csv(file)
    property_names = preds_df["property_name"].tolist()
    # get unique property names
    unique_property_names = list(set(property_names))
    # generate embeddings
    embeddings = generate_embeddings(unique_property_names)
    # save embeddings to parquet
    assert len(preds_df["refno"].unique()) == 1, "Expected only one refno per file"
    refno = preds_df["refno"].unique()[0]
    df = pd.DataFrame(
        {
            "refno": [refno] * len(unique_property_names),
            "property_name": unique_property_names,
            "embedding": embeddings,
        }
    )
    save_path = embeddings_dir / f"{file.stem}.parquet"
    df.to_parquet(save_path)
    print(f"Saved embeddings to {save_path} with {len(df)} rows")
