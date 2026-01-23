from argparse import ArgumentParser
import logging
import sys
import pandas as pd

import pbench

logger = logging.getLogger(__name__)

parser = ArgumentParser(
    description="Validate formatting accuracy of material property extraction"
)
parser = pbench.add_base_args(parser)
args = parser.parse_args()
pbench.setup_logging(args.log_level)


validated_candidates_dir = args.output_dir / "validated_candidates"
csv_files = list(validated_candidates_dir.glob("*.csv"))
if not csv_files:
    logger.error(f"No CSV files found in {validated_candidates_dir}")
    sys.exit(1)

logger.info(f"Found {len(csv_files)} CSV file(s) in {validated_candidates_dir}")
dfs = []
for csv_file in csv_files:
    logger.debug(f"Loading {csv_file.name}")
    df = pd.read_csv(csv_file)
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
logger.info(f"Loaded {len(df)} total rows from validated candidates")
# HACK: Filter for specific refnos that are known to have been validated by both annotators
df = df[
    df["refno"].isin(["10.1002--advs.202506089", "10.1016--j.intermet.2025.108748"])
]

# Filter for rows where "validated" is not nan
df_validated = df[~df["validated"].isna()]

# Micro-average accuracy
micro_accuracy = (df_validated["validated"] == True).mean()  # noqa: E712
micro_sem = (df_validated["validated"] == True).sem()  # noqa: E712
print(f"Average precision across all records: {micro_accuracy:.4f} ± {micro_sem:.4f}")

# Macro-average accuracy
per_refno_accuracy = df_validated.groupby("refno")["validated"].mean()
macro_accuracy = per_refno_accuracy.mean()
macro_sem = per_refno_accuracy.sem()
print(f"Average precision across papers: {macro_accuracy:.4f} ± {macro_sem:.4f}")
