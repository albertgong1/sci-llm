"""
Batch PDF to CSV Preprocessing for ChatExtract
Processes all PDFs in a directory and combines into a single CSV

Usage:
    # Process single PDF
    python batch_pdf_to_csv.py --pdf_path "paper.pdf" --output_csv "output.csv"
    
    # Process entire directory
    python batch_pdf_to_csv.py --pdf_dir "pdfs/" --output_csv "all_papers.csv"
    
    # Process directory with filtering
    python batch_pdf_to_csv.py --pdf_dir "pdfs/" --output_csv "filtered.csv" --min_length 20
"""

import fitz  # PyMuPDF
import re
from nltk.tokenize import sent_tokenize
import nltk
import argparse
import csv
from pathlib import Path
from typing import List, Dict, Tuple

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')


def extract_text_from_pdf(pdf_path: str) -> Tuple[str, str]:
    """Extract text from PDF using PyMuPDF"""
    doc = fitz.open(pdf_path)
    
    # Extract title (usually from first page, first few lines)
    first_page = doc[0]
    first_page_text = first_page.get_text()
    lines = first_page_text.split('\n')
    # Heuristic: title is usually in first few non-empty lines
    title = ' '.join([l.strip() for l in lines[:5] if l.strip()])
    
    # Extract all text from all pages
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    
    doc.close()
    return title, full_text


def preprocess_pdf_for_extraction(pdf_path: str, min_sentence_length: int = 10) -> Tuple[str, List[str]]:
    """
    Complete preprocessing pipeline for PDF
    Returns: title, list of sentences with context
    """
    # 1. Extract text from PDF
    title, full_text = extract_text_from_pdf(pdf_path)
    
    # 2. Basic cleaning
    # Remove excessive whitespace
    full_text = re.sub(r'\s+', ' ', full_text)
    
    # 3. Split into sentences
    sentences = sent_tokenize(full_text)
    
    # 4. Filter: keep only sentences with numbers
    # (following their recommendation for minimal prior knowledge)
    sentences_with_numbers = [s.strip() for s in sentences 
                              if re.search(r'\d', s) and len(s.strip()) > min_sentence_length]
    
    return title, sentences_with_numbers


def create_passages(title: str, sentences: List[str], doi: str) -> List[Dict]:
    """
    Create passages: title + preceding_sentence + target_sentence
    Following ChatExtract methodology
    """
    passages = []
    
    for i, sentence in enumerate(sentences):
        # Get preceding sentence (or empty string if first sentence)
        preceding = sentences[i-1] if i > 0 else ""
        
        # Create passage
        passage = f"{title} {preceding} {sentence}"
        
        passages.append({
            'sentence': sentence,
            'passage': passage,
            'doi': doi
        })
    
    return passages


def process_single_pdf(pdf_path: str, min_sentence_length: int = 10) -> List[Dict]:
    """
    Process a single PDF and return passages
    
    Args:
        pdf_path: Path to PDF file
        min_sentence_length: Minimum sentence length to keep
        
    Returns:
        List of passage dictionaries
    """
    # Extract paper title from PDF path for DOI field
    paper_title = Path(pdf_path).stem
    
    print(f"Processing: {pdf_path}")
    
    # Preprocess
    title, sentences = preprocess_pdf_for_extraction(pdf_path, min_sentence_length)
    
    print(f"  Title: {title[:80]}...")
    print(f"  Sentences with numbers: {len(sentences)}")
    
    # Create passages
    passages = create_passages(title, sentences, paper_title)
    
    return passages


def process_pdf_directory(pdf_dir: str, min_sentence_length: int = 10) -> List[Dict]:
    """
    Process all PDFs in a directory
    
    Args:
        pdf_dir: Directory containing PDF files
        min_sentence_length: Minimum sentence length to keep
        
    Returns:
        Combined list of all passages from all PDFs
    """
    pdf_dir_path = Path(pdf_dir)
    pdf_files = list(pdf_dir_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return []
    
    print(f"Found {len(pdf_files)} PDF files")
    print("="*80)
    
    all_passages = []
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] ", end="")
        try:
            passages = process_single_pdf(str(pdf_file), min_sentence_length)
            all_passages.extend(passages)
        except Exception as e:
            print(f"  Error processing {pdf_file}: {e}")
    
    print("\n" + "="*80)
    print(f"Total passages extracted: {len(all_passages)}")
    
    return all_passages


def save_passages_to_csv(passages: List[Dict], output_csv: str):
    """
    Save passages to CSV file in ChatExtract format
    
    Args:
        passages: List of passage dictionaries
        output_csv: Output CSV file path
    """
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['sentence', 'passage', 'doi'])
        writer.writeheader()
        for p in passages:
            writer.writerow({
                'sentence': p['sentence'],
                'passage': p['passage'],
                'doi': p['doi']
            })
    
    print(f"Saved {len(passages)} passages to {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch preprocess PDFs for ChatExtract",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single PDF
  python batch_pdf_to_csv.py --pdf_path "paper.pdf" --output_csv "output.csv"
  
  # Process entire directory
  python batch_pdf_to_csv.py --pdf_dir "pdfs/" --output_csv "all_papers.csv"
  
  # Process with custom minimum sentence length
  python batch_pdf_to_csv.py --pdf_dir "pdfs/" --output_csv "filtered.csv" --min_length 20
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--pdf_path', 
        type=str, 
        help='Path to a single PDF file to preprocess'
    )
    input_group.add_argument(
        '--pdf_dir', 
        type=str, 
        help='Directory containing PDF files to batch process'
    )
    
    # Output and options
    parser.add_argument(
        '--output_csv', 
        type=str, 
        default='preprocessed_output.csv', 
        help='Output CSV file path (default: preprocessed_output.csv)'
    )
    parser.add_argument(
        '--min_length', 
        type=int, 
        default=10, 
        help='Minimum sentence length to keep (default: 10)'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("PDF to CSV Preprocessing for ChatExtract")
    print("="*80)
    print()
    
    # Process PDFs
    if args.pdf_path:
        # Single PDF
        passages = process_single_pdf(args.pdf_path, args.min_length)
    else:
        # Directory of PDFs
        passages = process_pdf_directory(args.pdf_dir, args.min_length)
    
    # Save to CSV
    if passages:
        print()
        save_passages_to_csv(passages, args.output_csv)
        print()
        print("="*80)
        print("Preprocessing complete!")
        print(f"Ready to run: python ChatExtract_fixed.py {args.output_csv} \"your_property\"")
        print("="*80)
    else:
        print("\nNo passages extracted. Check your PDF files.")


if __name__ == "__main__":
    main()