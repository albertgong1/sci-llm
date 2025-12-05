import fitz  # PyMuPDF
import re
from nltk.tokenize import sent_tokenize
import nltk
import argparse

# Download required NLTK data (run once)
nltk.download('punkt_tab')

def extract_text_from_pdf(pdf_path):
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

def preprocess_pdf_for_extraction(pdf_path):
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
                              if re.search(r'\d', s) and len(s.strip()) > 10]
    
    return title, sentences_with_numbers

def create_passages(title, sentences):
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
            'title': title,
            'preceding': preceding
        })
    
    return passages

# Example usage:
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Preprocess PDF for ChatExtract")
    parser.add_argument('--pdf_path', type=str, help='Path to the PDF file to preprocess')
    parser.add_argument('--output_csv', type=str, default='output.csv', help='Output CSV file path')
    args = parser.parse_args()


    pdf_path = args.pdf_path
    # Extract paper title from PDF path for logging, paper title is the last part from path
    paper_title = pdf_path.split('/')[-1].replace('.pdf', '')

    # Preprocess
    title, sentences = preprocess_pdf_for_extraction(pdf_path)
    
    print(f"Title: {title}")
    print(f"Number of sentences with numbers: {len(sentences)}")
    
    # Create passages for ChatExtract
    passages = create_passages(title, sentences)
    
    # Create CSV as required by ChatExtract code
    import csv
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['sentence', 'passage', 'doi'])
        writer.writeheader()
        for p in passages:
            writer.writerow({
                'sentence': p['sentence'],
                'passage': p['passage'],
                'doi': paper_title  # or actual DOI if available
            })