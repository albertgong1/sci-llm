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

# load from args.data_dir / "combined_validation_results.csv"
combined_validation_results_path = args.data_dir / "combined_validation_results.csv"
if not combined_validation_results_path.exists():
    logger.error(f"File not found: {combined_validation_results_path}")
    sys.exit(1)
df = pd.read_csv(combined_validation_results_path)
VALIDATED_COL = "validated_resolved"

# Filter for rows where "validated" is not nan
df_validated = df[df[VALIDATED_COL].isin(["TRUE", "FALSE"])]
# can you convert to boolean?
df_validated.loc[:, VALIDATED_COL] = df_validated[VALIDATED_COL] == "TRUE"

# Micro-average accuracy
micro_accuracy = (df_validated[VALIDATED_COL] == True).mean()  # noqa: E712
micro_sem = (df_validated[VALIDATED_COL] == True).sem()  # noqa: E712
print(
    f"Average precision across all records (n={len(df_validated)}): {micro_accuracy:.4f} ± {micro_sem:.4f}"
)

# Macro-average accuracy
per_refno_accuracy = df_validated.groupby("refno")[VALIDATED_COL].mean()
macro_accuracy = per_refno_accuracy.mean()
macro_sem = per_refno_accuracy.sem()
print(
    f"Average precision across papers (n={len(per_refno_accuracy)}): {macro_accuracy:.4f} ± {macro_sem:.4f}"
)
