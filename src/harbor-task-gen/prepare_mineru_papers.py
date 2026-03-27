#!/usr/bin/env -S uv run -- python
"""Precompute normalized MinerU bundles for a PDF directory.

Usage: uv run python src/harbor-task-gen/prepare_mineru_papers.py --pdf-dir PDFs --output-dir OUT
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from mineru import MinerUConfig, ensure_mineru_bundle


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Precompute normalized MinerU bundles for a directory of PDFs."
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        required=True,
        help="Directory containing PDF files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where normalized MinerU bundles will be written.",
    )
    parser.add_argument(
        "--refno",
        action="append",
        default=None,
        help="Only preprocess the given PDF stem(s). Repeat as needed.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of PDFs to preprocess.",
    )
    parser.add_argument(
        "--mineru-binary",
        type=str,
        default="mineru",
        help="MinerU CLI binary name (default: mineru).",
    )
    parser.add_argument(
        "--mineru-backend",
        type=str,
        default="hybrid-auto-engine",
        choices=[
            "pipeline",
            "hybrid-auto-engine",
            "hybrid-http-client",
            "vlm-auto-engine",
            "vlm-http-client",
        ],
        help="MinerU backend (default: hybrid-auto-engine).",
    )
    parser.add_argument(
        "--mineru-method",
        type=str,
        default="auto",
        choices=["auto", "txt", "ocr"],
        help="MinerU parsing method (default: auto).",
    )
    parser.add_argument(
        "--mineru-lang",
        type=str,
        default=None,
        help="Optional MinerU OCR language hint.",
    )
    parser.add_argument(
        "--mineru-source",
        type=str,
        default=None,
        choices=["huggingface", "modelscope", "local"],
        help="Optional MinerU model source override.",
    )
    parser.add_argument(
        "--mineru-device",
        type=str,
        default=None,
        help="Optional MinerU device override such as cpu, mps, or cuda:0.",
    )
    parser.add_argument(
        "--mineru-extra-arg",
        action="append",
        default=None,
        help="Extra raw CLI arg to pass through to MinerU. Repeat as needed.",
    )
    parser.add_argument(
        "--mineru-formula",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable MinerU formula parsing (default: enabled).",
    )
    parser.add_argument(
        "--mineru-table",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable MinerU table parsing (default: enabled).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild bundles even if they already exist.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (default: INFO).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    pdf_dir = args.pdf_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = MinerUConfig(
        binary=args.mineru_binary,
        backend=args.mineru_backend,
        method=args.mineru_method,
        formula=bool(args.mineru_formula),
        table=bool(args.mineru_table),
        lang=args.mineru_lang,
        source=args.mineru_source,
        device=args.mineru_device,
        extra_args=list(args.mineru_extra_arg or []),
        force=bool(args.force),
    )

    pdf_paths = sorted(path for path in pdf_dir.glob("*.pdf") if path.is_file())
    if args.refno:
        requested = {value.strip() for value in args.refno if value and value.strip()}
        pdf_paths = [path for path in pdf_paths if path.stem in requested]
    if args.limit is not None:
        pdf_paths = pdf_paths[: args.limit]

    if not pdf_paths:
        logger.warning("No PDFs matched the requested inputs.")
        return

    for pdf_path in pdf_paths:
        bundle = ensure_mineru_bundle(
            pdf_path=pdf_path, bundle_root=output_dir, config=config
        )
        logger.info("Prepared %s -> %s", pdf_path.name, bundle.bundle_dir)


if __name__ == "__main__":
    main()
