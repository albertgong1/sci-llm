"""Constants and base parsers for the pbench project."""

import argparse
import logging
from pathlib import Path

import llm_utils

# NOTE: Assets are in the root directory of the project, change the path
# if the project structure changes.
# ASSETS_DIR = Path(__file__).parent.parent.parent / "assets"
# DATA_DIR = Path(__file__).parent.parent.parent / "data"

SUPPORTED_DOMAINS: list[str] = ["supercon", "precedent-search"]

DOMAIN2HF_DATASET_NAME: dict[str, str] = {
    "supercon": "kilian-group/supercon-mini-v2",
}


def add_base_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add base arguments to the argument parser."""
    # Domain args
    parser.add_argument(
        "--domain",
        type=str,
        # required=True,
        choices=SUPPORTED_DOMAINS,
        help="Material science domain",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="kilian-group/supercon-mini-v2",
        help="Path to Ground Truth CSV or Hugging Face dataset name",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="HuggingFace dataset configuration name, depending on the domain (e.g., 'tc', 'gap')",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of the dataset to use",
    )

    # Paths args
    parser.add_argument(
        "--data_dir",
        "-dd",
        type=Path,
        default="data",
        help="Directory containing the papers and properties",
    )
    parser.add_argument(
        "--output_dir",
        "-od",
        type=Path,
        default="out/",
        help="Output directory (default: out)",
    )

    # LLM args
    parser.add_argument(
        "--server",
        type=str,
        default="gemini",
        choices=llm_utils.SUPPORTED_SERVERS,
        help="LLM server to use",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemini-3-flash-preview",
        help="Model name (e.g., 'gemini-3-flash-preview').",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite existing output files",
    )

    # Logging args
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    return parser


def setup_logging(log_level: int = logging.INFO) -> None:
    """Setup logging for the script.

    Args:
        log_level: Logging level (default: `logging.INFO`)

    """
    # Suppress logging from httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
