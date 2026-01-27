from argparse import ArgumentParser
from pathlib import Path


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
