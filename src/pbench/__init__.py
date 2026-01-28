"""Constants and base parsers for the pbench project."""

import argparse
import logging
from pathlib import Path

import llm_utils


def add_base_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add base arguments to the argument parser."""
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
        default=None,  # Path("out"),
        help="Output directory (default: out)",
    )
    parser.add_argument(
        "--jobs_dir",
        "-jd",
        type=Path,
        default=None,  # Path("jobs"),
        help="Jobs directory for Harbor runs (default: jobs)",
    )
    parser.add_argument(
        "--preds_dirname",
        "-pd",
        type=str,
        default="preds",
        help="Directory name for predictions (default: preds)",
    )

    # Dataset args
    parser.add_argument(
        "--hf_repo",
        type=str,
        default=None,
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--hf_split",
        type=str,
        default=None,
        help="HuggingFace dataset split",
    )
    parser.add_argument(
        "--hf_revision",
        type=str,
        default=None,
        help="HuggingFace dataset revision",
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
        "-m",
        type=str,
        default="gemini-3-flash-preview",
        help="Model name (e.g., 'gemini-3-flash-preview').",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default=None,
        choices=["none", "low", "medium", "high", "xhigh"],
        help="Reasoning effort for OpenAI models (default: None)",
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


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging for the script.

    Args:
        log_level: Logging level (default: "INFO")

    """
    # Suppress logging from httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
