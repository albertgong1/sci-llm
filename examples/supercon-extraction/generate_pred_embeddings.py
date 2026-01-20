"""Script to generate embeddings from property names.

Reference: https://ai.google.dev/gemini-api/docs/embeddings#task-types
"""

# standard imports
import pandas as pd
from argparse import ArgumentParser
import re
import logging

# pbench imports
import pbench
from pbench_eval.match import generate_embeddings
from slugify import slugify

# local imports
from utils import get_harbor_data

logger = logging.getLogger(__name__)


parser = ArgumentParser(description="Generate embeddings from property names.")
parser = pbench.add_base_args(parser)
parser.add_argument(
    "--agent",
    "-a",
    type=str,
    default="gemini-cli",
    help="Agent name used in Harbor jobs (default: gemini-cli)",
)
args = parser.parse_args()
pbench.setup_logging(args.log_level)
jobs_dir = args.jobs_dir
force = args.force
# model_name = args.model_name
# agent = args.agent

if jobs_dir is not None:
    # Load predictions from Harbor jobs directory
    df = get_harbor_data(jobs_dir)
    # # filter by agent and model name
    # df = df[(df["agent"] == agent) & (df["model"] == model_name)]
else:
    # Load predictions from CSV files
    preds_dir = args.output_dir / "unsupervised_llm_extraction"
    preds_files = list(preds_dir.glob("*.csv"))
    if not preds_files:
        raise FileNotFoundError(f"No CSV files found in {preds_dir}")
    dfs = []
    for file in preds_files:
        df = pd.read_csv(file)
        if True:
            logger.warning(
                "Inferring refno from filename using regex pattern: refno=<value>"
            )
            df["refno"] = re.search(r"refno=([^.]+)", str(file)).group(1)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)


embeddings_dir = args.output_dir / "pred_embeddings"
embeddings_dir.mkdir(parents=True, exist_ok=True)

for (agent, model), group in df.groupby(["agent", "model"]):
    for refno in group["refno"].unique():
        save_path = (
            embeddings_dir / f"{slugify(agent)}_{slugify(model)}_{refno}.parquet"
        )
        if save_path.exists() and not force:
            logger.info(
                f"Embeddings already exist for {agent=} {model=} {refno=}, skipping..."
            )
            continue

        print(f"Generating embeddings for {agent=} {model=} {refno=}...")
        preds_df = group[group["refno"] == refno]
        property_names = preds_df["property_name"].dropna().tolist()
        # get unique property names
        unique_property_names = list(set(property_names))
        # generate embeddings
        embeddings = generate_embeddings(unique_property_names)
        # save embeddings to parquet
        assert len(preds_df["refno"].unique()) == 1, "Expected only one refno per file"
        refno = preds_df["refno"].unique()[0]
        embeddings_df = pd.DataFrame(
            {
                "refno": [refno] * len(unique_property_names),
                "property_name": unique_property_names,
                "embedding": embeddings,
                "agent": [agent] * len(unique_property_names),
                "model": [model] * len(unique_property_names),
            }
        )
        embeddings_df.to_parquet(save_path)
        print(f"Saved embeddings to {save_path} with {len(embeddings_df)} rows")
