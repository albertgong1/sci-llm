"""Rename arxiv PDFs from arxiv IDs to paper_ids (refnos) based on CSV mapping.

This script:
1. Makes a copy of data-arxiv/Paper_DB to data-arxiv/Paper_DB_renamed
2. Renames PDFs using the mapping from the CSV file
3. Skips PDFs with parentheses in names (duplicates) and saves them to a log file

Usage:
    uv run python examples/supercon-extraction/rename_arxiv_pdfs.py [--dry-run]
"""

import argparse
import logging
import re
import shutil
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def has_parentheses_suffix(filename: str) -> bool:
    """Check if filename has a parentheses suffix like (1), (2), etc."""
    # Match pattern like "0710.1413(1).pdf"
    return bool(re.search(r"\(\d+\)\.pdf$", filename))


def main() -> None:
    """Rename arxiv PDFs from arxiv IDs to paper_ids (refnos) based on CSV mapping."""
    parser = argparse.ArgumentParser(
        description="Rename arxiv PDFs from arxiv IDs to paper_ids (refnos)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without actually doing it",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "data-arxiv",
        help="Path to the data-arxiv directory",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    source_dir = data_dir / "SuperCon Arxiv Papers"
    target_dir = data_dir / "Paper_DB"
    csv_path = data_dir / "SuperCon Property Extraction Dataset - Arxiv.csv"
    skipped_log_path = data_dir / "skipped_duplicate_pdfs.csv"

    # Validate paths
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Step 1: Copy the directory
    if not args.dry_run:
        if target_dir.exists():
            logger.warning(f"Target directory already exists, removing: {target_dir}")
            shutil.rmtree(target_dir)
        logger.info(f"Copying {source_dir} to {target_dir}...")
        shutil.copytree(source_dir, target_dir)
        logger.info("Copy complete")
    else:
        logger.info(f"[DRY RUN] Would copy {source_dir} to {target_dir}")

    # Step 2: Load the CSV and create mapping
    logger.info(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Filter rows with non-empty file_path
    df_with_file = df[df["file_path"].notna() & (df["file_path"] != "")].copy()
    logger.info(f"Found {len(df_with_file)} rows with file_path")

    # Create mapping from file_path to paper_id
    # Handle duplicates: same file_path mapping to multiple paper_ids
    file_to_paperids: dict[str, list[str]] = {}
    for _, row in df_with_file.iterrows():
        file_path = row["file_path"]
        paper_id = row["paper_id"]
        if file_path not in file_to_paperids:
            file_to_paperids[file_path] = []
        file_to_paperids[file_path].append(paper_id)

    # Step 3: Process PDFs
    # Track skipped files (duplicates with parentheses)
    skipped_parentheses: list[dict[str, str]] = []
    # Track successful renames
    renamed_count = 0
    # Track copies made for multiple refnos
    copied_for_multiple_refnos = 0
    # Track files not in CSV
    not_in_csv: list[str] = []

    pdf_files = (
        list(target_dir.glob("*.pdf"))
        if not args.dry_run
        else list(source_dir.glob("*.pdf"))
    )
    logger.info(f"Processing {len(pdf_files)} PDF files...")

    for pdf_path in pdf_files:
        filename = pdf_path.name

        # Check for parentheses suffix (duplicates)
        if has_parentheses_suffix(filename):
            # Extract the base name without parentheses suffix
            base_name = re.sub(r"\(\d+\)(\.pdf)$", r"\1", filename)
            skipped_parentheses.append(
                {
                    "original_filename": filename,
                    "base_filename": base_name,
                    "reason": "duplicate_with_parentheses",
                }
            )
            logger.debug(f"Skipping duplicate: {filename}")
            continue

        # Look up the paper_id for this file
        if filename not in file_to_paperids:
            not_in_csv.append(filename)
            logger.debug(f"Not in CSV: {filename}")
            continue

        paper_ids = file_to_paperids[filename]

        # Handle multiple refnos by copying the file for each
        if len(paper_ids) > 1:
            for paper_id in paper_ids:
                new_filename = f"{paper_id}.pdf"
                new_path = pdf_path.parent / new_filename
                if not args.dry_run:
                    shutil.copy2(pdf_path, new_path)
                    copied_for_multiple_refnos += 1
                else:
                    logger.info(f"[DRY RUN] Would copy: {filename} -> {new_filename}")
                    copied_for_multiple_refnos += 1
            # Remove the original file after copying
            if not args.dry_run:
                pdf_path.unlink()
            else:
                logger.info(f"[DRY RUN] Would remove original: {filename}")
            continue

        # Rename the file (single refno case)
        paper_id = paper_ids[0]
        new_filename = f"{paper_id}.pdf"
        new_path = pdf_path.parent / new_filename

        if not args.dry_run:
            pdf_path.rename(new_path)
            renamed_count += 1
        else:
            logger.info(f"[DRY RUN] Would rename: {filename} -> {new_filename}")
            renamed_count += 1

    # Step 4: Save skipped files log (only parentheses duplicates now)
    if skipped_parentheses:
        df_skipped = pd.DataFrame(skipped_parentheses)
        if not args.dry_run:
            df_skipped.to_csv(skipped_log_path, index=False)
            logger.info(
                f"Saved {len(skipped_parentheses)} skipped files to {skipped_log_path}"
            )
        else:
            logger.info(
                f"[DRY RUN] Would save {len(skipped_parentheses)} skipped files to {skipped_log_path}"
            )
            logger.info(f"Skipped files:\n{df_skipped.to_string()}")

    # Summary
    logger.info("=" * 50)
    logger.info("Summary:")
    logger.info(f"  Total PDFs processed: {len(pdf_files)}")
    logger.info(f"  Successfully renamed: {renamed_count}")
    logger.info(f"  Copied for multiple refnos: {copied_for_multiple_refnos}")
    logger.info(f"  Skipped (parentheses duplicates): {len(skipped_parentheses)}")
    logger.info(f"  Not in CSV mapping: {len(not_in_csv)}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
