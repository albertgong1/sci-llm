#!/usr/bin/env -S uv run --env-file=.env -- python
"""Analyze citation quality by comparing agent citations with ground truth DOIs/titles.

Usage:
  python examples/harbor-workspace/analyze_citation_quality.py

This script:
1. Loads ground truth data from dev-set_with_dois.csv
2. For each citation_info CSV file, checks if GT DOI or title matches agent citations
3. Generates a detailed report and updates CSVs with GT columns
"""

import csv
import re
import difflib
from pathlib import Path
from collections import defaultdict


def normalize_doi(doi: str) -> str:
    """Normalize DOI for comparison."""
    if not doi:
        return ""
    doi = doi.lower().strip()
    # Remove common prefixes
    doi = re.sub(r"^https?://", "", doi)
    doi = re.sub(r"^doi\.org/", "", doi)
    doi = re.sub(r"^dx\.doi\.org/", "", doi)
    doi = re.sub(r"^doi:", "", doi)
    return doi


def normalize_title(title: str) -> str:
    """Normalize title for comparison."""
    if not title:
        return ""
    title = title.lower().strip()
    # Remove HTML/XML tags
    title = re.sub(r"<[^>]+>", "", title)
    # Remove special characters and extra whitespace
    title = re.sub(r"[^\w\s]", " ", title)
    title = re.sub(r"\s+", " ", title)
    return title.strip()


def normalize_material(material: str) -> str:
    """Normalize material name for matching."""
    if not material:
        return ""
    return material.lower().strip().replace(" ", "").replace("-", "")


def load_ground_truth(gt_path: Path) -> dict:
    """Load ground truth data keyed by normalized material name."""
    gt_data = {}
    with open(gt_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            material = normalize_material(row.get("material", ""))
            if material:
                gt_data[material] = {
                    "refno": row.get("refno", ""),
                    "material_original": row.get("material", ""),
                    "doi": row.get("doi", ""),
                    "true_doi": row.get("true_doi", ""),
                    "title": row.get("title", ""),
                    "paper_title": row.get("paper_title", ""),
                    "arxiv_title": row.get("arxiv_title", ""),
                    "year": row.get("year", ""),
                    "published_year": row.get("published_year", ""),
                }
    return gt_data


def get_all_agent_dois(row: dict) -> list[str]:
    """Extract all DOIs from agent citations."""
    dois = []
    for prop in ["is_superconducting", "tc", "tcn"]:
        for i in range(1, 4):
            doi = row.get(f"{prop}_source_{i}_doi", "")
            if doi:
                dois.append(normalize_doi(doi))
    return dois


def get_all_agent_titles(row: dict) -> list[str]:
    """Extract all titles from agent citations."""
    titles = []
    for prop in ["is_superconducting", "tc", "tcn"]:
        for i in range(1, 4):
            title = row.get(f"{prop}_source_{i}_title", "")
            if title:
                titles.append(normalize_title(title))
    return titles


def check_doi_match_detailed(gt_dois: list[str], row: dict) -> list[dict]:
    """Check DOI matches and return details about which property/source matched."""
    matches = []
    for gt_doi in gt_dois:
        if not gt_doi:
            continue
        for prop in ["is_superconducting", "tc", "tcn"]:
            for i in range(1, 4):
                agent_doi = normalize_doi(row.get(f"{prop}_source_{i}_doi", ""))
                if not agent_doi:
                    continue
                if gt_doi in agent_doi or agent_doi in gt_doi:
                    matches.append({
                        "match_type": "doi",
                        "property": prop,
                        "source_num": i,
                        "gt_doi": gt_doi,
                        "agent_doi": agent_doi,
                        "agent_title": row.get(f"{prop}_source_{i}_title", ""),
                    })
    return matches


def check_title_match_detailed(gt_titles: list[str], row: dict) -> list[dict]:
    """Check title matches and return details about which property/source matched."""
    matches = []
    for gt_title in gt_titles:
        if not gt_title or len(gt_title) < 10:
            continue
        
        t1 = normalize_title(gt_title)
        
        for prop in ["is_superconducting", "tc", "tcn"]:
            for i in range(1, 4):
                agent_title = normalize_title(row.get(f"{prop}_source_{i}_title", ""))
                if not agent_title or len(agent_title) < 10:
                    continue
                
                # Use SequenceMatcher for better fuzzy matching
                ratio = difflib.SequenceMatcher(None, t1, agent_title).ratio()
                
                if ratio > 0.8:  # 80% similarity threshold
                    matches.append({
                        "match_type": "title",
                        "property": prop,
                        "source_num": i,
                        "gt_title": gt_title[:100],
                        "agent_title": row.get(f"{prop}_source_{i}_title", ""),
                        "agent_doi": row.get(f"{prop}_source_{i}_doi", ""),
                        "score": ratio
                    })
    
    # Sort by score descending
    matches.sort(key=lambda x: x.get("score", 0), reverse=True)
    return matches


def check_doi_match(gt_dois: list[str], agent_dois: list[str]) -> bool:
    """Check if any GT DOI matches any agent DOI."""
    for gt_doi in gt_dois:
        if not gt_doi:
            continue
        for agent_doi in agent_dois:
            if not agent_doi:
                continue
            if gt_doi in agent_doi or agent_doi in gt_doi:
                return True
    return False


def check_title_match(gt_titles: list[str], agent_titles: list[str]) -> bool:
    """Check if any GT title matches any agent title (fuzzy match)."""
    for gt_title in gt_titles:
        if not gt_title or len(gt_title) < 10:
            continue
            
        t1 = normalize_title(gt_title)
        
        for agent_title in agent_titles:
            if not agent_title or len(agent_title) < 10:
                continue
                
            t2 = normalize_title(agent_title)
            ratio = difflib.SequenceMatcher(None, t1, t2).ratio()
            
            if ratio > 0.8:
                return True
    return False


def analyze_citation_file(
    citation_path: Path, gt_data: dict
) -> tuple[dict, list[dict], list[str], list[dict], list[dict]]:
    """Analyze a single citation_info CSV file."""
    stats = {
        "total": 0,
        "has_gt": 0,
        "doi_match": 0,
        "title_match_only": 0,
        "no_match": 0,
        "no_gt_data": 0,
        "details": [],
    }

    updated_rows = []
    detailed_matches = []  # For the detailed CSV
    high_score_no_matches = [] # For the high score no match CSV

    with open(citation_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])

        for row in reader:
            stats["total"] += 1
            material = normalize_material(row.get("material", ""))

            # Add GT columns
            gt = gt_data.get(material, {})
            row["GT_doi"] = gt.get("doi", "")
            row["GT_true_doi"] = gt.get("true_doi", "")
            row["GT_title"] = gt.get("title", "")
            row["GT_paper_title"] = gt.get("paper_title", "")
            row["GT_arxiv_title"] = gt.get("arxiv_title", "")
            row["GT_year"] = gt.get("year", "")

            if not gt:
                stats["no_gt_data"] += 1
                row["citation_match_status"] = "no_gt_data"
                updated_rows.append(row)
                continue

            stats["has_gt"] += 1

            # Get GT DOIs and titles
            gt_dois = [
                normalize_doi(gt.get("doi", "")),
                normalize_doi(gt.get("true_doi", "")),
            ]
            gt_dois = [d for d in gt_dois if d]

            gt_titles = [
                normalize_title(gt.get("title", "")),
                normalize_title(gt.get("paper_title", "")),
                normalize_title(gt.get("arxiv_title", "")),
            ]
            gt_titles = [t for t in gt_titles if t]

            # Get agent DOIs and titles
            agent_dois = get_all_agent_dois(row)
            agent_titles = get_all_agent_titles(row)

            # Check matches with details
            doi_matches = check_doi_match_detailed(gt_dois, row)
            title_matches = check_title_match_detailed(gt_titles, row)

            doi_matched = len(doi_matches) > 0
            title_matched = len(title_matches) > 0

            detail = {
                "material": row.get("material", ""),
                "gt_dois": gt_dois,
                "gt_titles": [gt.get("title", ""), gt.get("arxiv_title", "")],
                "agent_dois": agent_dois[:3],
                "agent_titles": agent_titles[:3],
                "doi_match": doi_matched,
                "title_match": title_matched,
            }

            # Build detailed match record
            match_record = {
                "source_file": citation_path.name,
                "material": row.get("material", ""),
                "job_id": row.get("job_id", ""),
                "metadata_trial_id": row.get("metadata_trial_id", ""),
                "score_012": row.get("score_012", ""),
                "score_category": row.get("score_category", ""),
                "match_status": "",
                "match_type": "",
                "gt_doi": gt.get("doi", ""),
                "gt_title": gt.get("title", "")[:100] if gt.get("title") else "",
            }
            
            # Initialize match columns
            for i in range(1, 4):
                match_record[f"matched_property_{i}"] = ""
                match_record[f"matched_source_num_{i}"] = ""
                match_record[f"matched_doi_{i}"] = ""
                match_record[f"matched_title_{i}"] = ""

            all_matches = []
            if doi_matched:
                stats["doi_match"] += 1
                row["citation_match_status"] = "doi_match"
                match_record["match_status"] = "doi_match"
                match_record["match_type"] = "doi"
                all_matches.extend(doi_matches)
            elif title_matched:
                stats["title_match_only"] += 1
                row["citation_match_status"] = "title_match"
                match_record["match_status"] = "title_match"
                match_record["match_type"] = "title"
                all_matches.extend(title_matches)
            else:
                stats["no_match"] += 1
                row["citation_match_status"] = "no_match"
                match_record["match_status"] = "no_match"
                detail["no_match_reason"] = "Neither DOI nor title matched"
                
                # Check for high score no match case
                if str(row.get("score_012", "")) in ["1", "2"]:
                    hs_record = {
                        "source_file": citation_path.name,
                        "material": row.get("material", ""),
                        "job_id": row.get("job_id", ""),
                        "metadata_trial_id": row.get("metadata_trial_id", ""),
                        "score_012": row.get("score_012", ""),
                        "score_category": row.get("score_category", ""),
                        "gt_doi": gt.get("doi", ""),
                        "gt_title": gt.get("title", "")[:100] if gt.get("title") else "",
                    }
                    
                    # split agent citations by property
                    for prop in ["is_superconducting", "tc", "tcn"]:
                        prop_dois = []
                        prop_titles = []
                        for i in range(1, 4):
                             d = row.get(f"{prop}_source_{i}_doi", "")
                             t = row.get(f"{prop}_source_{i}_title", "")
                             if d: prop_dois.append(d)
                             if t: prop_titles.append(t)
                        
                        hs_record[f"agent_dois_{prop}"] = "; ".join(prop_dois)
                        hs_record[f"agent_titles_{prop}"] = "; ".join(prop_titles)

                    high_score_no_matches.append(hs_record)

            # Fill in up to 3 matches
            # Deduplicate matches based on (agent_doi, agent_title)
            # Aggregate properties and source_nums for duplicates
            unique_matches_map = {}
            
            for m in all_matches:
                # Create a unique key for the paper
                # Use agent_doi if present, otherwise agent_title
                key = (m.get("agent_doi", ""), m.get("agent_title", ""))
                
                if key not in unique_matches_map:
                    unique_matches_map[key] = {
                        "agent_doi": m.get("agent_doi", ""),
                        "agent_title": m.get("agent_title", ""),
                        "properties": [],
                        "source_nums": [],
                        "score": m.get("score", 0) # Keep highest score if available
                    }
                
                # Add property and source_num if not already present for this key
                # (Ideally they differ, but we handle strictly)
                entry = unique_matches_map[key]
                if m["property"] not in entry["properties"] or m["source_num"] not in entry["source_nums"]:
                     entry["properties"].append(m["property"])
                     entry["source_nums"].append(str(m["source_num"]))
                     if m.get("score", 0) > entry["score"]:
                         entry["score"] = m.get("score", 0)

            # Convert map to list and sort
            # Sort by presence of DOI (often more reliable) or Score
            unique_matches = list(unique_matches_map.values())
            # Simple heuristic sort: prefer matches that have a DOI
            unique_matches.sort(key=lambda x: (bool(x["agent_doi"]), x["score"]), reverse=True)
            
            for i, match in enumerate(unique_matches[:3]):
                match_record[f"matched_property_{i+1}"] = "; ".join(match["properties"])
                match_record[f"matched_source_num_{i+1}"] = "; ".join(match["source_nums"])
                match_record[f"matched_doi_{i+1}"] = match.get("agent_doi", "")
                match_record[f"matched_title_{i+1}"] = match.get("agent_title", "")

            detailed_matches.append(match_record)
            stats["details"].append(detail)
            updated_rows.append(row)

    # Add new columns to fieldnames
    new_cols = [
        "GT_doi",
        "GT_true_doi",
        "GT_title",
        "GT_paper_title",
        "GT_arxiv_title",
        "GT_year",
        "citation_match_status",
    ]
    for col in new_cols:
        if col not in fieldnames:
            fieldnames.append(col)

    for col in new_cols:
        if col not in fieldnames:
            fieldnames.append(col)

    return stats, updated_rows, fieldnames, detailed_matches, high_score_no_matches


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    gt_path = script_dir.parent / "tc-precedent-search" / "dev-set_with_dois.csv"
    citation_dir = script_dir / "out" / "harbor" / "precedent-search" / "analysis" / "final_results" / "citation_info"

    if not gt_path.exists():
        raise SystemExit(f"Ground truth file not found: {gt_path}")
    if not citation_dir.exists():
        raise SystemExit(f"Citation info directory not found: {citation_dir}")

    print(f"Loading ground truth from: {gt_path}")
    gt_data = load_ground_truth(gt_path)
    print(f"Loaded {len(gt_data)} materials from ground truth\n")

    citation_files = sorted(citation_dir.glob("citation_info-*.csv"))
    if not citation_files:
        raise SystemExit(f"No citation_info CSV files found in: {citation_dir}")

    print(f"Found {len(citation_files)} citation_info files\n")

    all_stats = {}
    all_detailed_matches = []
    all_high_score_no_matches = []

    for citation_file in citation_files:
        print(f"Processing: {citation_file.name}")
        stats, updated_rows, fieldnames, detailed_matches, high_score_no_matches = analyze_citation_file(citation_file, gt_data)
        all_stats[citation_file.name] = stats
        all_detailed_matches.extend(detailed_matches)
        all_high_score_no_matches.extend(high_score_no_matches)

        # Write updated CSV with GT columns
        with open(citation_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated_rows)

        print(f"  Updated with GT columns")
        print(f"  DOI matches: {stats['doi_match']}, Title matches: {stats['title_match_only']}, No match: {stats['no_match']}")


    # Write detailed matches CSV
    detailed_csv_path = citation_dir / "detailed_matches.csv"
    detailed_fieldnames = [
        "source_file",
        "material",
        "job_id",
        "metadata_trial_id",
        "score_012",
        "score_category",
        "match_status",
        "match_type",
        "gt_doi",
        "gt_title",
        "matched_property_1",
        "matched_source_num_1",
        "matched_doi_1",
        "matched_title_1",
        "matched_property_2",
        "matched_source_num_2",
        "matched_doi_2",
        "matched_title_2",
        "matched_property_3",
        "matched_source_num_3",
        "matched_doi_3",
        "matched_title_3",
    ]
    with open(detailed_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detailed_fieldnames)
        writer.writeheader()
        writer.writerows(all_detailed_matches)
    print(f"Detailed matches saved to: {detailed_csv_path}")

    # Generate title_or_doi_matches.csv (Filtered version)
    filtered_csv_path = citation_dir / "title_or_doi_matches.csv"
    filtered_matches = [
        m for m in all_detailed_matches 
        if m["match_status"] in ["title_match", "doi_match"]
    ]
    with open(filtered_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detailed_fieldnames)
        writer.writeheader()
        writer.writerows(filtered_matches)
    print(f"Filtered matches saved to: {filtered_csv_path}")

    # Generate high_score_no_citation_match.csv
    # Rows where score_012 is 1 or 2 (good result) but no citation match (hallucination risk or missing GT)
    high_score_no_match_path = citation_dir / "high_score_no_citation_match.csv"
    
    hs_fieldnames = [
        "source_file",
        "material",
        "job_id",
        "metadata_trial_id",
        "score_012",
        "score_category",
        "gt_doi",
        "gt_title",
        "agent_dois_is_superconducting",
        "agent_titles_is_superconducting",
        "agent_dois_tc",
        "agent_titles_tc",
        "agent_dois_tcn",
        "agent_titles_tcn",
    ]
    
    with open(high_score_no_match_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=hs_fieldnames)
        writer.writeheader()
        writer.writerows(all_high_score_no_matches)
    print(f"High score no match saved to: {high_score_no_match_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
