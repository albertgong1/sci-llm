import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import fitz  # PyMuPDF
import os
from datetime import datetime
import re
import argparse
import unicodedata
from google import genai
from google.genai import types

# Property Validator App
#
# A Streamlit application for validating extracted property data from scientific papers against their source PDFs.
#
# Usage:
#     streamlit run validator_app.py
#
# CLI Arguments:
#     --csv_path (str): Path to the initial CSV file to validate.
#     --pdf_path (str): Default PDF path to use if not specified in CSV.
#     --csv_folder (str): Folder containing validation CSVs (for the dropdown).
#
# Example:
#     streamlit run validator_app.py -- --csv_path "data/my_properties.csv"/extraction/assets/validate_csv/Ru7B3_full_properties_rearraged.csv"

# Page configuration
st.set_page_config(layout="wide", page_title="Property Validator")

# Parse command line arguments
try:
    parser = argparse.ArgumentParser(description="Validator App Configuration")
    parser.add_argument("--csv_path", type=str, default=None, help="Path to the initial CSV file")
    parser.add_argument("--pdf_path", type=str, default=None, help="Optional initial PDF path override.")
    parser.add_argument("--csv_folder", type=str, default="examples/extraction/assets/validate_csv/", help="Folder containing validation CSVs")
    parser.add_argument("--paper_folder", type=str, default="examples/extraction/assets/Paper_DB/", help="Folder containing PDFs")
    
    args, unknown = parser.parse_known_args()
    
    CSV_PATH = args.csv_path
    INITIAL_PDF_PATH = args.pdf_path
    CSV_FOLDER = args.csv_folder
    PAPER_FOLDER = args.paper_folder
except SystemExit:
    pass
except Exception as e:
    print(f"Error parsing args: {e}")
    # Fallback to no defaults
    CSV_PATH = None
    INITIAL_PDF_PATH = None
    CSV_FOLDER = "examples/extraction/assets/validate_csv/"
    PAPER_FOLDER = "examples/extraction/assets/Paper_DB/" 

# --- AI Finder (For when we don't have "property evidence" from CSV) ---
def search_with_ai(pdf_path, value, prop_name, unit="", material_or_system=""):
    """
    Uses Gemini to find a value in a PDF.
    Returns: (page_num, evidence_text, confidence)
    """
    if not os.path.exists(pdf_path):
        return None, "PDF not found", 0.0

    match = re.search(r"Paper_(\d+)\.pdf", pdf_path)
    paper_id = match.group(1) if match else "Unknown"

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    
    try:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        # Handle missing/nan value case
        val_str = str(value).strip()
        if val_str.lower() == 'nan' or val_str == '':
            prompt_text = f"I am looking for the value of the property '{prop_name}' in this paper (Paper ID: {paper_id})."
        else:
            prompt_text = f"I am looking for the value '{val_str}' for the property '{prop_name}' in this paper (Paper ID: {paper_id})."

        if material_or_system:
            prompt_text += f" for material '{material_or_system}'"

        if unit:
            prompt_text += f", with unit '{unit}'."

        prompt = f"""
        {prompt_text}
        
        Please find exactly where this value appears.
        Return ONLY a JSON object with this format:
        {{
            "page_number": <int>,
            "evidence_sentence": "<string>",
            "confidence_score": <float between 0.0 and 1.0>
        }}
        If you cannot find it, return null for fields.
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                        types.Part.from_text(text=prompt),
                    ],
                ),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        
        import json
        text = response.text.strip()
        # Clean markdown code blocks if present
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
            
        result = json.loads(text)
        
        # Handle case where Gemini returns a list of objects instead of a single object
        if isinstance(result, list):
            if len(result) > 0:
                result = result[0]
            else:
                return None, "AI returned empty list", 0.0
        
        if result and isinstance(result, dict) and result.get("page_number"):
            return result.get("page_number"), result.get("evidence_sentence"), result.get("confidence_score", 0.0)
        else:
            return None, "Not found by Gemini", 0.0
            
    except Exception as e:
        return None, f"Gemini Error: {str(e)}", 0.0

# Helper Functions

def get_available_csv_files():
    """Returns a list of CSV files in the validate_csv folder."""
    if not os.path.exists(CSV_FOLDER):
        return []

    csv_files = [f for f in os.listdir(CSV_FOLDER) if f.endswith('.csv')]
    return sorted(csv_files)

def get_available_pdf_files():
    """Returns a list of PDF files in the paper folder."""
    if not os.path.exists(PAPER_FOLDER):
        return []
    pdf_files = [f for f in os.listdir(PAPER_FOLDER) if f.endswith('.pdf')]
    return sorted(pdf_files)

def preprocess_dataframe(df):
    """Ensures required columns exist in the dataframe."""
    # Ensure validation columns
    if "validated" not in df.columns:
        df["validated"] = None # None/NaN implies not yet reviewed
    if "validator_name" not in df.columns:
        df["validator_name"] = ""
    if "validation_date" not in df.columns:
        df["validation_date"] = ""

    # Convert validated to nullable boolean
    if "validated" in df.columns:
         # Best practice for Streamlit/Arrow: use bool or pd.NA
         df["validated"] = df["validated"].astype(object).where(df["validated"].notna(), None)

    # Ensure flagged column exists
    if "flagged" not in df.columns:
        df["flagged"] = False # Default to False
    else:
        # Ensure boolean type (fill NaN with False)
        df["flagged"] = df["flagged"].fillna(False).astype(bool)

    for text_col in ["validator_name", "validation_date"]:
        # Force object dtype so assigning strings doesn't clash with floats
        df[text_col] = df[text_col].astype(object).fillna("")
        
    # Ensure paper_pdf_path column exists
    if "paper_pdf_path" not in df.columns:
        df["paper_pdf_path"] = None

    # Ensure location columns exist (crucial for extraction pipeline compatibility)
    if "location.page" not in df.columns:
        df["location.page"] = None # Default to None so we can auto-discover it
    if "location.section" not in df.columns:
        df["location.section"] = ""
    if "location.evidence" not in df.columns:
        df["location.evidence"] = ""

    # Ensure location columns exist (crucial for extraction pipeline compatibility)
    if "location.page" not in df.columns:
        df["location.page"] = None # Default to None so we can auto-discover it
    if "location.section" not in df.columns:
        df["location.section"] = ""
    if "location.evidence" not in df.columns:
        df["location.evidence"] = ""

    # Reverted the forced overwrite here. 
    # We will handle PDF path updating dynamically in the UI when the user selects a PDF.
        
    return df

def load_data(csv_path):
    """Loads the CSV and ensures required columns exist."""
    if not os.path.exists(csv_path):
        st.error(f"CSV not found at {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    return preprocess_dataframe(df)

def save_data(df, csv_path):
    """Saves the dataframe back to CSV."""
    df.to_csv(csv_path, index=False)

# --- Search Logic ---

def normalize(text):
    """Normalize text: NFKD decomposition, replace non-alphanumeric with space (except ...), lower case."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFKD', text)
    # Preserve '...' as a unique token
    if '...' in text:
        text = text.replace('...', ' wildcardtoken ') 
    
    # Remove non-alphanumeric but replace with space
    # We want to split 17.2 -> 17 2, so dots correspond to space
    # underscores also replaced by space
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text).lower()
    return text.strip()

def is_valid_page(page_num):
    """
    Checks if page_num is a valid integer string or number.
    Returns: (is_valid, integer_value)
    """
    if pd.isna(page_num) or page_num == "":
        return False, None
    try:
        # Handle "1.0" or 1.0 -> 1
        val = int(float(page_num))
        return True, val
    except ValueError:
        return False, None

def get_tokens(text):
    """Split text into alphanumeric tokens."""
    if not isinstance(text, str): return []
    text = normalize(text)
    return text.split()

def find_best_match_on_page(page, evidence_text):
    """
    Searches a single PyMuPDF page for the evidence text.
    Returns: (score, list_of_rects)
    score: 0.0 to 1.0 (1.0 = perfect match)
    """
    if not evidence_text or not isinstance(evidence_text, str) or evidence_text == "nan":
        return 0.0, []

    # Strategy 1: Exact Match (Fastest)
    text_instances = page.search_for(evidence_text)
    if text_instances:
        # If we find exact matches, we consider it a score of 1.0
        return 1.0, text_instances

    # Strategy 2: Fuzzy Sequence Match
    # 1. Get all words from page with their bboxes
    # page.get_text("words") returns list of (x0, y0, x1, y1, "text", block_no, line_no, word_no)
    words = page.get_text("words") 
    
    if not words:
        return 0.0, []

    # 2. Tokenize Evidence and Page Words
    evidence_tokens = get_tokens(evidence_text)
    if not evidence_tokens:
        return 0.0, []

    page_tokens = [] # list of (token, word_idx)
    for w_idx, w in enumerate(words):
        raw_text = w[4]
        # One "word" in PDF might contain multiple tokens if we split by special chars?
        # maximize granular matching
        w_tokens_list = get_tokens(raw_text)
        for t in w_tokens_list:
            page_tokens.append((t, w_idx))

    if not page_tokens:
        return 0.0, []

    # 3. Find Best Sequence Match
    best_match_indices = []
    best_match_score = 0.0
    
    start_token = evidence_tokens[0]
    # Find all occurrences of the first token to start checking sequences
    possible_starts = [i for i, pt in enumerate(page_tokens) if pt[0] == start_token]
    
    # If first token not found, try finding substring match for first token? 
    # Or just loop all? For efficiency, let's stick to exact start match or substring start match.
    # To be more robust, let's allow substring start for the first token too.
    if not possible_starts:
         possible_starts = [i for i, pt in enumerate(page_tokens) if start_token in pt[0]]

    # Optimization: if evidence is long, maybe try to match rare tokens? 
    # For now, stick to left-to-right scan.

    for start_idx in possible_starts:
        current_match_word_indices = []
        p_idx = start_idx
        e_idx = 0
        
        # Align sequence allowing for skips
        while p_idx < len(page_tokens) and e_idx < len(evidence_tokens):
            p_token = page_tokens[p_idx][0]
            e_token = evidence_tokens[e_idx]
            
            # 1. Handle Ellipsis Wildcard
            if "wildcardtoken" in e_token: 
                 # Skip ahead in PDF until we find the NEXT evidence token
                if e_idx + 1 < len(evidence_tokens):
                    next_e_token = evidence_tokens[e_idx + 1]
                    # Search forward in PDF for next_e_token
                    found_next_start = False
                    # Limit lookahead to avoid false positives across the whole page
                    limit = 200 # words
                    for search_p_idx in range(p_idx, min(p_idx + limit, len(page_tokens))):
                        # fast forward
                        cand = page_tokens[search_p_idx][0]
                        if cand == next_e_token or (len(next_e_token) > 2 and next_e_token in cand):
                            p_idx = search_p_idx # Jump to it
                            e_idx += 1 # Consumed the wildcard
                            found_next_start = True
                            break
                    
                    if found_next_start:
                        continue # Continue outer loop from new p_idx
                    else:
                        break # Failed to find resume point
                else:
                    # Wildcard is last token - we matched everything up to here
                    e_idx += 1
                    break 
            
            # 2. Exact Match
            if p_token == e_token:
                current_match_word_indices.append(page_tokens[p_idx][1])
                p_idx += 1
                e_idx += 1
                continue

            # 3. Substring Match (e.g. 001 inside 2001)
            if len(e_token) > 1 and e_token in p_token:
                current_match_word_indices.append(page_tokens[p_idx][1])
                p_idx += 1
                e_idx += 1
                continue

            # 4. Greedy Merge Match (Handles split formulas like P6 + 3 + mc -> P63mc)
            # Try to merge next K tokens as long as they build up the target token
            merged_text = p_token
            # Look ahead up to 10 tokens (arbitrary limit for sanity)
            found_merge = False
            
            # Start loop looking at next token
            temp_idx = p_idx + 1
            while temp_idx < len(page_tokens) and (temp_idx - p_idx) <= 10:
                next_part = page_tokens[temp_idx][0]
                potential_merge = merged_text + next_part
                
                # If perfect match
                if potential_merge == e_token:
                    # Found it! Record all indices involved
                    for k in range(p_idx, temp_idx + 1):
                         current_match_word_indices.append(page_tokens[k][1])
                    p_idx = temp_idx + 1
                    e_idx += 1
                    found_merge = True
                    break
                
                # If it's a valid prefix, keep going (e.g. "P6" + "3" = "P63", which is start of "P63mc")
                if e_token.startswith(potential_merge):
                    merged_text = potential_merge
                    temp_idx += 1
                else:
                    # Not a prefix, so this path is dead. Stop trying to merge.
                    break
            
            if found_merge:
                continue

            # 5. Skip Mismatch (Lookahead/Noise tolerance)
            found_next = False
            lookahead = 5
            for look in range(1, lookahead + 1):
                if p_idx + look < len(page_tokens):
                    cand = page_tokens[p_idx + look][0]
                    # Check if the next evidence token matches a future page token
                    if cand == e_token or (len(e_token) > 2 and e_token in cand):
                        p_idx += look
                        found_next = True
                        break
            
            if found_next:
                continue
            else:
                # Sequence broken
                break
        
        # Calculate score for this start position
        # simple coverage score
        score = e_idx / len(evidence_tokens)
        
        if score > best_match_score:
            best_match_score = score
            best_match_indices = current_match_word_indices
            
        # Optimization: Early exit if perfect score?
        if best_match_score >= 1.0:
            break

    # Convert word indices to rects
    rects = []
    if best_match_indices:
        unique_indices = sorted(list(set(best_match_indices)))
        for idx in unique_indices:
            # words[idx] -> (x0, y0, x1, y1, text, ...)
            r = fitz.Rect(words[idx][:4])
            rects.append(r)
            
    return best_match_score, rects

def find_evidence_in_pdf(pdf_path, evidence_text, suggested_page_num=None, scan_all_pages=True):
    """
    Search for evidence in the PDF.
    
    Args:
        pdf_path: Path to PDF file.
        evidence_text: The text string to find.
        suggested_page_num: 1-based page number to check first.
        scan_all_pages: If False, only check suggested_page_num.
        
    Returns:
        (best_page_num, best_score, best_rects)
        best_page_num is 1-based.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF {pdf_path}: {e}")
        return None, 0.0, []

    best_global_score = 0.0
    best_global_page = None
    best_global_rects = []
    
    # Check suggested page first
    if suggested_page_num is not None:
        p_idx = int(suggested_page_num) - 1
        if 0 <= p_idx < len(doc):
            score, rects = find_best_match_on_page(doc[p_idx], evidence_text)
            # If good enough, return immediately
            if score > 0.85:
                doc.close()
                return suggested_page_num, score, rects
            
            best_global_score = score
            best_global_page = suggested_page_num
            best_global_rects = rects

    # If we are strictly checking only the suggested page, return what we found there
    if not scan_all_pages:
        doc.close()
        # Only return if we found something reasonable? 
        # Actually user wants "no highlighting" if not found, so we return whatever matches we got (even if specific score is low, rects might be empty or partial).
        # find_best_match_on_page returns best local match.
        return best_global_page, best_global_score, best_global_rects

    # Global search (if strict check failed or wasn't requested)
    # Only scan if we didn't get a perfect match already
    step = 1# if len(doc) < 20 else 2 # Optimization?
    for p_idx in range(0, len(doc), step):
        # Skip suggested page as we already checked it
        if suggested_page_num is not None and p_idx == (int(suggested_page_num) - 1):
            continue
            
        score, rects = find_best_match_on_page(doc[p_idx], evidence_text)
        
        if score > best_global_score:
            best_global_score = score
            best_global_page = p_idx + 1
            best_global_rects = rects
            
            if best_global_score > 0.9:
                break
                
    doc.close()
    return best_global_page, best_global_score, best_global_rects

    # If match is poor, scan all pages
    for i, page in enumerate(doc):
        # Skip the suggested page as we already checked it
        current_page_num = i + 1
        if current_page_num == suggested_page_num:
            continue
            
        score, rects = find_best_match_on_page(page, evidence_text)
        
        if score > best_global_score:
            best_global_score = score
            best_global_page = current_page_num
            best_global_rects = rects
            
            # If we find a near perfect match, stop scanning?
            if best_global_score > 0.95:
                break
                
    doc.close()
    return best_global_page, best_global_score, best_global_rects

def render_page_image(pdf_path, page_num, highlight_rects=None):
    """
    Renders a PDF page as an image with optional highlighting.
    page_num is 1-based.
    highlight_rects is a list of fitz.Rect objects.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        return None, f"Error opening PDF: {e}"

    if page_num < 1 or page_num > len(doc):
        return None, f"Page {page_num} out of range (1-{len(doc)})"
        
    page_idx = int(page_num) - 1
    page = doc[page_idx]
    
    # Highlight rects if provided
    if highlight_rects:
        for rect in highlight_rects:
            page.add_highlight_annot(rect)

    # Render page to image
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) # 2x zoom for better quality
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes, None

# --- Main App Logic ---

def main():
    st.title("Property Validator")

    # Remove legacy widget state keys to avoid Streamlit conflicts
    st.session_state.pop("property_selector", None)
    
    # Initialize df in session state if not present
    if "df" not in st.session_state:
        st.session_state.df = None

    # Sidebar for Validator Input
    with st.sidebar:
        st.header("Extraction Pipeline")

        # CSV File Selector
        st.subheader("Data Selection")
        available_csv_files = get_available_csv_files()

        if not available_csv_files:
            st.error(f"No CSV files found in {CSV_FOLDER}")
            st.stop()

        # Initialize selected CSV in session state
        # Only set default if provided via CLI, otherwise None
        if "selected_csv" not in st.session_state:
            if CSV_PATH and os.path.basename(CSV_PATH) in available_csv_files:
                st.session_state.selected_csv = os.path.basename(CSV_PATH)
            else:
                st.session_state.selected_csv = None

        # Determine index for selectbox
        csv_index = None
        if st.session_state.selected_csv in available_csv_files:
            csv_index = available_csv_files.index(st.session_state.selected_csv)

        selected_csv = st.selectbox(
            "Select CSV File",
            available_csv_files,
            index=csv_index,
            placeholder="Choose a CSV file...",
        )

        # Update session state if changed
        if selected_csv != st.session_state.selected_csv:
            st.session_state.selected_csv = selected_csv
            st.session_state.df = None  # Force reload
            st.rerun()

        # Load data immediately if CSV selected (for PDF inference)
        csv_path = None
        if st.session_state.selected_csv:
            csv_path = os.path.join(CSV_FOLDER, st.session_state.selected_csv)
            if "df" not in st.session_state or st.session_state.df is None:
                if os.path.exists(csv_path):
                    st.session_state.df = load_data(csv_path)

        # PDF File Selector
        st.subheader("PDF Selection") # Header close to CSV
        available_pdf_files = get_available_pdf_files()
        
        if not available_pdf_files:
             st.warning(f"No PDF files found in {PAPER_FOLDER}")
        
        # Initialize selected PDF in session state
        if "selected_pdf" not in st.session_state:
            if INITIAL_PDF_PATH and os.path.basename(INITIAL_PDF_PATH) in available_pdf_files:
                 st.session_state.selected_pdf = os.path.basename(INITIAL_PDF_PATH)
            else:
                 current_pdf_path_in_df = None
                 # Safe check for df presence and columns
                 if st.session_state.df is not None and not st.session_state.df.empty and "paper_pdf_path" in st.session_state.df.columns:
                    first_path = st.session_state.df.iloc[0]["paper_pdf_path"]
                    if first_path and isinstance(first_path, str):
                        current_pdf_path_in_df = os.path.basename(first_path)
                 
                 if current_pdf_path_in_df and current_pdf_path_in_df in available_pdf_files:
                     st.session_state.selected_pdf = current_pdf_path_in_df
                 else:
                     st.session_state.selected_pdf = None

        pdf_index = None
        if st.session_state.selected_pdf in available_pdf_files:
            pdf_index = available_pdf_files.index(st.session_state.selected_pdf)

        selected_pdf = st.selectbox(
            "Select PDF File",
            available_pdf_files,
            index=pdf_index,
            placeholder="Choose a PDF file...",
        )

        if selected_pdf:
             # Always sync session state
             st.session_state.selected_pdf = selected_pdf

             full_pdf_path = os.path.join(PAPER_FOLDER, selected_pdf)
             need_update = False
             if st.session_state.df is not None:
                 if "paper_pdf_path" not in st.session_state.df.columns:
                     need_update = True
                 else:
                     current_val = st.session_state.df.iloc[0].get("paper_pdf_path")
                     if current_val != full_pdf_path:
                         need_update = True
             
             if need_update and st.session_state.df is not None:
                 with st.spinner(f"Updating all records to use {selected_pdf}..."):
                     st.session_state.df["paper_pdf_path"] = full_pdf_path
                     save_data(st.session_state.df, csv_path)
                     st.toast(f"Linked {selected_pdf} to this CSV!", icon="🔗")
                     st.rerun()
        else:
            st.session_state.selected_pdf = None

        st.divider()

        # Validator Name
        validator_name = st.text_input("Validator Name", value=st.session_state.get("validator_name", ""))
        st.session_state.validator_name = validator_name

    # Unified Stop Condition
    if not st.session_state.selected_csv or not st.session_state.selected_pdf:
        st.info("Please select a CSV file and corresponding PDF file from the sidebar to continue.")
        st.stop()

    csv_path = os.path.join(CSV_FOLDER, st.session_state.selected_csv)

    # Ensure Session State is ready (redundant but safe)
    if "df" not in st.session_state or st.session_state.df is None:
        st.session_state.df = load_data(csv_path)

    if st.session_state.df.empty:
        st.stop()


    # Continue with sidebar metrics
    with st.sidebar:
        
        st.metric("Total Properties", len(st.session_state.df))
        validated_count = st.session_state.df["validated"].notna().sum()
        st.metric("Validated", validated_count)
        
        # Filter controls
        filter_status = st.radio("Show", ["All", "Pending", "Valid", "Invalid", "Flagged"], index=1)

    # Filter Data based on selection
    df = st.session_state.df
    if filter_status == "Pending":
        filtered_indices = df[df["validated"].isna()].index
    elif filter_status == "Valid":
        filtered_indices = df[df["validated"] == True].index
    elif filter_status == "Invalid":
        filtered_indices = df[df["validated"] == False].index
    elif filter_status == "Flagged":
        filtered_indices = df[df["flagged"] == True].index
    else:
        filtered_indices = df.index

    if len(filtered_indices) == 0:
        st.info("No properties found for this filter.")
        return

    # Work with a list for stable ordering and easier indexing
    filtered_indices = list(filtered_indices)

    # Select Property Logic
    # We use a selectbox to navigate, but we want it to be smart.
    # Default to the first item in the filtered list if no selection.

    # Initialize current index in session state
    if "current_property_index" not in st.session_state:
        st.session_state.current_property_index = 0

    # Helper to keep dropdown selection + index synced
    def select_property(position):
        position = max(0, min(len(filtered_indices) - 1, position))
        st.session_state.current_property_index = position

    # Make sure the current index stays within bounds when filters change
    st.session_state.current_property_index = max(
        0,
        min(st.session_state.current_property_index, len(filtered_indices) - 1)
    )

    # Ensure current index is within bounds of filtered list
    if st.session_state.current_property_index >= len(filtered_indices):
        st.session_state.current_property_index = 0

    # Create a nice label for the dropdown
    def get_label(idx):
        row = df.loc[idx]
        val_status = "❓"
        if row["validated"] == True: val_status = "✅"
        elif row["validated"] == False: val_status = "❌"
        
        flag_status = "🚩 " if row["flagged"] else ""
        return f"{flag_status}{val_status} {row['id']} - {row['property_name']} ({row['value_string']})"

    # Show dropdown for property selection first
    dropdown_index = st.selectbox(
        "Select Property",
        filtered_indices,
        format_func=get_label,
        index=st.session_state.current_property_index
    )

    # Navigation buttons with keyboard shortcuts (below the dropdown)
    nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])

    with nav_col1:
        if st.button("⬅️ Previous Property", width="stretch", key="prev_btn"):
            if st.session_state.current_property_index > 0:
                select_property(st.session_state.current_property_index - 1)
                st.rerun()

    with nav_col2:
        st.markdown(
            f"<div style='text-align: center; padding: 8px;'>Property {st.session_state.current_property_index + 1} of {len(filtered_indices)}</div>",
            unsafe_allow_html=True
        )

    with nav_col3:
        if st.button("Next Property ➡️", width="stretch", key="next_btn"):
            if st.session_state.current_property_index < len(filtered_indices) - 1:
                select_property(st.session_state.current_property_index + 1)
                st.rerun()

    # Update current index if dropdown changed (check after buttons so button clicks work first)
    # Convert the selected df index to position in filtered_indices
    new_position = filtered_indices.index(dropdown_index)
    if new_position != st.session_state.current_property_index:
        select_property(new_position)
        st.rerun()

    # Add keyboard navigation using components.html
    keyboard_js = """
    <script>
    const streamlitDoc = window.parent.document;

    function findAndClickButton(searchText) {
        const buttons = streamlitDoc.querySelectorAll('button');
        for (let btn of buttons) {
            if (btn.textContent && btn.textContent.includes(searchText)) {
                btn.click();
                return true;
            }
        }
        return false;
    }

    if (streamlitDoc.navKeyHandler) {
        streamlitDoc.removeEventListener('keydown', streamlitDoc.navKeyHandler);
    }

    streamlitDoc.navKeyHandler = function(e) {
        const activeElement = streamlitDoc.activeElement;
        const isTyping = activeElement && (
            activeElement.tagName === 'INPUT' ||
            activeElement.tagName === 'TEXTAREA' ||
            activeElement.isContentEditable
        );

        if (isTyping) return;

        if (e.key === 'ArrowLeft') {
            e.preventDefault();
            findAndClickButton('Previous Property');
        } else if (e.key === 'ArrowRight') {
            e.preventDefault();
            findAndClickButton('Next Property');
        }
    };

    streamlitDoc.addEventListener('keydown', streamlitDoc.navKeyHandler);
    </script>
    """
    components.html(keyboard_js, height=0)

    # Get selected index from current position
    selected_index = filtered_indices[st.session_state.current_property_index]

    row = df.loc[selected_index]

    # Layout
    col1, col2 = st.columns([1, 1.5]) # Left panel (info + list), Right panel (PDF)

    with col1:
        # Top Left: Validation Card
        st.subheader("Validation")
        
        container = st.container(border=True)
        with container:
            st.markdown(f"**Property:** `{row['property_name']}`")
            st.markdown(f"**Value:** `{row['value_string']}`")
            st.markdown(f"**Unit:** `{row['units']}`")
            st.markdown(f"**Material:** `{row['material_or_system']}`")
            st.markdown(f"**Location:** Page {row['location.page']}, {row['location.section']}")
            st.markdown(f"**Evidence:** _{row['location.evidence']}_")

            st.write("---")
            st.write("**Is this valid?**")

            # Check if validator name is provided
            name_provided = validator_name and validator_name.strip() != ""

            if not name_provided:
                st.warning("Please enter your name in the sidebar to begin validation.")

            # Validation Controls
            # Give more weight to Valid/Invalid to prevent wrapping
            col_valid, col_invalid, col_flag, col_reset = st.columns([1.5, 1.5, 1.1, 1.1])

            is_validated = row["validated"]
            is_flagged = row["flagged"]
            
            # Determine button styles based on state
            valid_type = "primary" if is_validated is True else "secondary"
            invalid_type = "primary" if is_validated is False else "secondary"
            flag_type = "primary" if is_flagged else "secondary"

            with col_valid:
                # Modifying Valid Button to Handle Pending AI Matches
                if st.button("✅ Valid", type=valid_type, width="stretch", disabled=not name_provided):
                    # Check if there is a pending AI suggestion for this row
                    if 'pending_ai_match' in st.session_state and st.session_state.pending_ai_match.get('index') == selected_index:
                         suggestion = st.session_state.pending_ai_match
                         df.at[selected_index, 'location.page'] = suggestion['page']
                         df.at[selected_index, 'location.evidence'] = suggestion['evidence']
                         df.at[selected_index, 'location.section'] = "AI Discovered"
                         df.at[selected_index, 'location.source_type'] = "text"
                         
                         # Clear the pending match after accepting
                         del st.session_state.pending_ai_match
                    
                    df.at[selected_index, "validated"] = True
                    df.at[selected_index, "validator_name"] = st.session_state.validator_name
                    df.at[selected_index, "validation_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    save_data(df, csv_path)
                    st.session_state.df = df
                    st.rerun()

            with col_invalid:
                if st.button("❌ Invalid", type=invalid_type, width="stretch", disabled=not name_provided):
                    df.at[selected_index, "validated"] = False
                    df.at[selected_index, "validator_name"] = st.session_state.validator_name
                    df.at[selected_index, "validation_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Also clear pending AI match if invalidated? Maybe user rejected the AI suggestion implicitly.
                    if 'pending_ai_match' in st.session_state and st.session_state.pending_ai_match.get('index') == selected_index:
                        del st.session_state.pending_ai_match

                    save_data(df, csv_path)
                    st.session_state.df = df
                    st.rerun()

            with col_flag:
                flag_label = "Unflag" if is_flagged else "Flag"
                if st.button(flag_label, type=flag_type, width="stretch"):
                    new_flag_state = not is_flagged
                    df.at[selected_index, "flagged"] = new_flag_state
                    save_data(df, csv_path)
                    st.session_state.df = df
                    st.rerun()
                    
            with col_reset:
                if st.button("Reset", type="secondary", width="stretch"):
                    df.at[selected_index, "validated"] = None
                    df.at[selected_index, "validator_name"] = ""
                    df.at[selected_index, "validation_date"] = ""
                    
                    # Clear pending match
                    if 'pending_ai_match' in st.session_state and st.session_state.pending_ai_match.get('index') == selected_index:
                        del st.session_state.pending_ai_match

                    save_data(df, csv_path)
                    st.session_state.df = df
                    st.rerun()
                    
            if is_flagged:
                st.warning("This property is flagged for review.")

        # Bottom Left: Full Data Table (Read-only view of context)
        st.subheader("Property List")
        # Highlight currently selected row
        def highlight_selected(r):
            if r.name == selected_index:
                return ['background-color: #ffffb3'] * len(r)
            return [''] * len(r)

        # Show a subset of columns
        display_cols = ["id", "property_name", "value_string", "validated"]
        st.dataframe(
            df[display_cols].loc[filtered_indices], # Only show filtered
            width="stretch",
            height=400,
            hide_index=False
        )

    with col2:
        # Right Panel: PDF Viewer
        st.subheader("Evidence Viewer")

        pdf_path = row["paper_pdf_path"]

        # Check if PDF path is missing or empty
        if pd.isna(pdf_path) or pdf_path == "" or pdf_path is None:
            st.warning("No PDF path specified for this property.")
        elif not os.path.exists(pdf_path):
            # Try resolving relative path
            if os.path.exists(pdf_path.replace("../", "")):
                pdf_path = pdf_path.replace("../", "")
            else:
                st.warning(f"PDF file not found at path: {pdf_path}")
                pdf_path = None

        page_num = row['location.page']
        evidence_text = row['location.evidence']

        # Check for pending AI match for this row
        pending_match = None
        if 'pending_ai_match' in st.session_state and st.session_state.pending_ai_match.get('index') == selected_index:
             pending_match = st.session_state.pending_ai_match
             page_num = pending_match['page'] # Override for viewing
             evidence_text = pending_match['evidence'] # Override for viewing

        if pdf_path:
            # Check effective page num
            valid_page, p_num = is_valid_page(page_num)
            
            img_bytes = None
            caption = ""
            
            if valid_page:
                rects = []
                score = 0.0
                if pending_match:
                     # Highlight pending AI match (strict on the found AI page)
                     # For AI match, we trust the AI found the text, but our finding logic might fail if text extraction is bad.
                     # We still check score to avoid random highlights.
                     _, score, rects = find_evidence_in_pdf(pdf_path, evidence_text, suggested_page_num=p_num, scan_all_pages=False)
                elif evidence_text and str(evidence_text).lower() != 'nan' and str(evidence_text).strip() != "":
                     # Highlight existing CSV evidence (User Request: Strict check on THIS page only)
                     _, score, rects = find_evidence_in_pdf(pdf_path, evidence_text, suggested_page_num=p_num, scan_all_pages=False)
                
                # Filter out low confidence matches (User: "If there is no good match, there just wont be any highlghting")
                if score < 0.6: 
                     rects = []

                img_bytes, error = render_page_image(pdf_path, p_num, highlight_rects=rects)
                
                if img_bytes:
                    caption = f"Page {p_num}"
                    if pending_match: caption += " (AI Suggestion - Pending Validation)"
                else:
                    st.error(error)

            # If not valid page AND no pending match, show search
            # If pending match exists, we fall through to the image renderer above (since p_num is valid from AI)
            elif (pd.isna(page_num) or page_num == ""):
                 st.info("No page number specified.")
                 
                 # AI Search Button
                 if st.button("🤖 Find with AI", key="ai_search_btn"):
                     with st.spinner("Asking Gemini to find the value in the paper..."):
                         ai_page, ai_evidence, ai_conf = search_with_ai(
                             pdf_path, 
                             value=str(row['value_string']) or str(row['value_number']), 
                             prop_name=row['property_name'],
                             unit=str(row['units']) if pd.notna(row['units']) else "",
                             material_or_system=str(row['material_or_system']) if "material_or_system" in row and pd.notna(row['material_or_system']) else ""
                         )
                     
                     if ai_page:
                         st.success(f"Found Match! (Confidence: {ai_conf})")
                         
                         # Verify logic: AI sometimes hallucinates the page number but gets the text right.
                         # We blindly trust the text, but let's verify the page.
                         # Use scan_all_pages=True to find where this text REALLY is.
                         found_page, found_score, _ = find_evidence_in_pdf(
                             pdf_path, 
                             ai_evidence, 
                             suggested_page_num=ai_page,
                             scan_all_pages=True
                         )
                         
                         if found_page and found_score > 0.8:
                             # If we found looking on the "AI Page" or "Any Page" with high confidence, use THAT page.
                             ai_page = found_page

                         # Save to session state as pending match
                         st.session_state.pending_ai_match = {
                             'index': selected_index,
                             'page': ai_page,
                             'evidence': ai_evidence,
                             'confidence': ai_conf
                         }
                         st.rerun() # Rerun to render the found page immediately
                     else:
                        if "AI returned empty list" in ai_evidence:
                           st.error("Gemini returned an empty list.")
                        else:
                           st.error(f"Gemini could not find the value. ({ai_evidence})")
            else:
                 st.warning(f"Invalid page number: {page_num}")

            if img_bytes:
                st.image(img_bytes, caption=caption, width="stretch")
            elif pdf_path and not img_bytes and (pd.isna(page_num) or page_num == ""):
                 # Clean "Preview not available"
                 pass

if __name__ == "__main__":
    main()
