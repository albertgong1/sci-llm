from argparse import ArgumentParser
from pathlib import Path
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def load_rubric(rubric_path: Path) -> pd.DataFrame:
    """Load and filter rubric CSV.

    For biosurfactants-style rubrics with condition rows, filter to only property rows
    (where condition_name is empty/NaN).

    Args:
        rubric_path: Path to rubric CSV file

    Returns:
        DataFrame with rubric data (property rows only for biosurfactants-style)

    """
    df_rubric = pd.read_csv(rubric_path)

    # If this is a biosurfactants-style rubric with condition_name column,
    # filter to property rows only (where condition_name is empty)
    if "condition_name" in df_rubric.columns:
        df_property_rubric = df_rubric[
            df_rubric["condition_name"].isna() | (df_rubric["condition_name"] == "")
        ]
        logger.info(
            f"Filtered rubric from {len(df_rubric)} rows to {len(df_property_rubric)} property rows"
        )
        return df_property_rubric

    return df_rubric


def add_scoring_args(parser: ArgumentParser) -> ArgumentParser:
    """Add command-line arguments for scoring functionality."""
    # Required arguments
    parser.add_argument(
        "--rubric_path",
        type=Path,
        required=True,
        help="Path to rubric CSV file",
    )

    # Optional arguments
    parser.add_argument(
        "--conversion_factors_path",
        type=Path,
        default=None,
        help="Path to SI conversion factors CSV file (optional)",
    )
    parser.add_argument(
        "--matching_mode",
        type=str,
        choices=["material", "conditions"],
        default="material",
        help="Matching mode: 'material' for supercon-style, 'conditions' for biosurfactants-style (default: material)",
    )
    parser.add_argument(
        "--material_column",
        type=str,
        default="material_or_system",
        help="Column name for material matching (default: material_or_system)",
    )
    return parser
