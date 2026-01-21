"""Streamlit Property Validator App

An interactive web application for human validation of property data extracted from scientific papers.
Displays extracted properties alongside their source PDF evidence, enabling validators to confirm or reject
each extraction and flag entries for review.

Features:
- Side-by-side view of extracted data and PDF source with highlighted evidence
- AI-powered evidence search using Gemini for properties missing location information
- Fuzzy text matching to locate evidence across PDF pages
- Navigation with keyboard shortcuts and table selection
- Automatic page correction when evidence is found on different pages
- Validation tracking with validator names, timestamps, and notes
- Progress tracking with filter options (Pending, Valid, Invalid, Flagged)

Keyboard Shortcuts:
- ← (Left Arrow): Previous property
- → (Right Arrow): Next property
- V: Accept/Valid (marks property as valid and advances to next)
- X: Reject/Invalid (marks property as invalid and advances to next)
- F: Toggle flag/unflag and advance to next

Usage:
```bash
uv run streamlit run src/pbench_validator_app/app.py -- \
    --output_dir /path/to/output/ \
    --data_dir /path/to/data/
```

The app expects:
- CSVs in: `{output_dir}/candidates/*.csv`
- PDFs in: `{data_dir}/Paper_DB/*.pdf`
- Validated CSVs saved to: `{output_dir}/validated_candidates/*.csv`

CLI Arguments (via pbench.add_base_args):
    --output_dir (Path): Output directory containing the 'candidates' folder with CSV files
    --data_dir (Path): Data directory containing the 'Paper_DB' folder with PDF files
    --log_level (str): Logging level (default: INFO)

Environment Variables:
    GOOGLE_API_KEY: Required for AI-powered evidence search functionality
"""

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
import json
from pathlib import Path

# pbench imports
import pbench

# Page configuration
st.set_page_config(layout="wide", page_title="Property Validator")


# Parse command line arguments
parser = argparse.ArgumentParser(description="Validator App Configuration")
parser = pbench.add_base_args(parser)
args, unknown = parser.parse_known_args()
pbench.setup_logging(args.log_level)

CSV_FOLDER = args.output_dir / "candidates"
PAPER_FOLDER = args.data_dir / "Paper_DB"


# --- AI Finder (For when we don't have "property evidence" from CSV) ---
def search_with_ai(
    pdf_path: Path,
    value: str,
    prop_name: str,
    unit: str = "",
    material_or_system: str = "",
) -> tuple[int | None, str, float]:
    """Uses Gemini to find a value in a PDF.

    Args:
        pdf_path: Path to the PDF file
        value: The value to search for
        prop_name: The name of the property
        unit: The unit of the property
        material_or_system: The material or system of the property

    Returns:
        page_num
        evidence_text
        confidence

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
        if val_str.lower() == "nan" or val_str == "":
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
                        types.Part.from_bytes(
                            data=pdf_bytes, mime_type="application/pdf"
                        ),
                        types.Part.from_text(text=prompt),
                    ],
                ),
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )

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
            return (
                result.get("page_number"),
                result.get("evidence_sentence"),
                result.get("confidence_score", 0.0),
            )
        else:
            return None, "Not found by Gemini", 0.0

    except Exception as e:
        return None, f"Gemini Error: {str(e)}", 0.0


# Helper Functions


def get_available_csv_files() -> list[str]:
    """Returns a list of CSV files in the validate_csv folder."""
    if not os.path.exists(CSV_FOLDER):
        return []

    csv_files = [f for f in os.listdir(CSV_FOLDER) if f.endswith(".csv")]
    return sorted(csv_files)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures required columns exist in the dataframe.

    Args:
        df: The dataframe to preprocess
    Returns:
        The preprocessed dataframe

    """
    # Ensure validation columns
    if "validated" not in df.columns:
        df["validated"] = None  # None/NaN implies not yet reviewed
    if "validator_name" not in df.columns:
        df["validator_name"] = ""
    if "validation_date" not in df.columns:
        df["validation_date"] = ""

    # Convert validated to nullable boolean
    if "validated" in df.columns:
        # Best practice for Streamlit/Arrow: use bool or pd.NA
        df["validated"] = (
            df["validated"].astype(object).where(df["validated"].notna(), None)
        )

    # Ensure flagged column exists
    if "flagged" not in df.columns:
        df["flagged"] = False  # Default to False
    else:
        # Ensure boolean type (fill NaN with False)
        df["flagged"] = df["flagged"].fillna(False).astype(bool)

    for text_col in ["validator_name", "validation_date", "validator_note"]:
        if text_col not in df.columns:
            df[text_col] = ""
        # Force object dtype so assigning strings doesn't clash with floats
        df[text_col] = df[text_col].astype(object).fillna("")

    # Ensure paper_pdf_path column exists
    if "paper_pdf_path" not in df.columns:
        df["paper_pdf_path"] = None

    # Ensure location columns exist (crucial for extraction pipeline compatibility)
    if "location.page" not in df.columns:
        df["location.page"] = None  # Default to None so we can auto-discover it
    # Force object type for page (it can be int or str or None)
    df["location.page"] = df["location.page"].astype(object)

    if "location.section" not in df.columns:
        df["location.section"] = ""
    # Force object type for section (text)
    df["location.section"] = df["location.section"].astype(str)

    if "location.evidence" not in df.columns:
        df["location.evidence"] = ""
    # Force object type for evidence (text)
    df["location.evidence"] = df["location.evidence"].astype(str)

    return df


def load_data(csv_path: Path) -> pd.DataFrame:
    """Loads the CSV. Implements 'Shadow Loader' pattern:
    1. Checks for a validated copy in validated_candidates directory.
    2. If found, loads that instead.
    3. If not found, CREATES it (Copy-on-Load) and loads it.
    This ensures we never edit the original file, but preserve all original data (including PDF paths).

    Args:
        csv_path: Path to the CSV file
    Returns:
        The loaded dataframe

    """
    if not os.path.exists(csv_path):
        st.error(f"CSV not found at {csv_path}")
        return pd.DataFrame()

    # Determine validated path in validated_candidates directory
    csv_path_obj = Path(csv_path)
    validated_dir = args.output_dir / "validated_candidates"
    validated_path = validated_dir / csv_path_obj.name

    # Copy-on-Load Logic
    if not os.path.exists(validated_path):
        # Create validated_candidates directory if it doesn't exist
        validated_dir.mkdir(parents=True, exist_ok=True)
        # Read original
        original_df = pd.read_csv(csv_path)
        # Save immediately to validated path
        original_df.to_csv(validated_path, index=False)
        st.toast(f"Created working copy: {validated_path}", icon="🆕")

    # Always load from the validated path (which now guaranteed exists)
    df = pd.read_csv(validated_path)

    # Store the WORKING path in session state so we know where to save
    st.session_state.current_working_csv_path = str(validated_path)

    return preprocess_dataframe(df)


def save_data(df: pd.DataFrame) -> None:
    """Saves the dataframe to the validated CSV path.

    Args:
        df: dataframe to save
        original_csv_path_ignored: Legacy arg, ignored. We use st.session_state.current_working_csv_path

    Args:
        df: The dataframe to save

    """
    if "current_working_csv_path" in st.session_state:
        df.to_csv(st.session_state.current_working_csv_path, index=False)
    else:
        st.error("Cannot save: No working CSV path found in session state.")


# --- Search Logic ---


def normalize(text: str) -> str:
    """Normalize text: NFKD decomposition, replace non-alphanumeric with space (except ...), lower case."""
    if not isinstance(text, str):
        return ""

    # Remove newlines and carriage returns early - they're just formatting artifacts in PDFs
    # This helps match text that spans multiple lines in the PDF
    text = text.replace("\n", " ").replace("\r", " ")

    # CRITICAL: Handle context-dependent patterns FIRST before simple replacements
    # In this PDF, ð (U+00F0) can mean either ξ (xi) or ( depending on context
    # Pattern 1: ð followed by digit and Þ means ξ(...)
    # Example: ð0Þ -> ξ(0) which should tokenize as "xi 0"
    text = re.sub(r"ð(\d+)Þ", r"xi \1 ", text)  # ð0Þ -> "xi 0 "
    text = re.sub(r"ð(\d+)þ", r"xi \1 ", text)  # Alternative with lowercase thorn

    # Pattern 2: \tðHÞ or \tðHþ means γ(H)
    # Tab character followed by ð/Þ or ð/þ is gamma in mathematical context
    text = re.sub(r"\tð([A-Za-z]+)[Þþ]", r"gamma \1 ", text)  # \tðHÞ -> "gamma H"

    # CRITICAL: Apply PDF-specific character replacements BEFORE NFKD normalization
    # because NFKD will convert things like ½ (U+00BD) into "1/2" which breaks our logic
    pdf_char_replacements = {
        # Custom font brackets (common in mathematical PDFs)
        "\xf0": "(",  # ð -> ( (only for remaining ð not matched by pattern above)
        "\xfe": ")",  # þ -> )
        "\xfd": ")",  # Alternative closing bracket
        "\xde": ")",  # Þ (capital thorn) -> )
        "\xbd": "[",  # ½ -> [ (MUST be before NFKD which converts to "1/2")
        "\xbc": "]",  # ¼ -> ]
        "\x02": "]",  # STX control character -> ]
        "\x03": "]",  # ETX control character -> ]
        "\x04": "]",  # EOT control character -> ]
        # Mathematical symbols
        "≃": "~",  # approximately equal
        "≈": "~",  # approximately equal
        "∥": "||",  # parallel to
        "⊥": "_|_",  # perpendicular
        "×": "x",  # multiplication
        "·": "",  # middle dot (often used in units)
    }

    for old, new in pdf_char_replacements.items():
        text = text.replace(old, new)

    # Now apply NFKD normalization
    text = unicodedata.normalize("NFKD", text)

    # Normalize Greek letters by mapping to their name equivalents
    # This handles both actual Greek unicode AND common PDF encodings
    greek_replacements = {
        "α": "alpha",
        "β": "beta",
        "γ": "gamma",
        "δ": "delta",
        "ε": "epsilon",
        "ζ": "zeta",
        "η": "eta",
        "θ": "theta",
        "κ": "kappa",
        "λ": "lambda",
        "μ": "mu",
        "ν": "nu",
        "ξ": "xi",
        "π": "pi",
        "ρ": "rho",
        "σ": "sigma",
        "τ": "tau",
        "φ": "phi",
        "χ": "chi",
        "ψ": "psi",
        "ω": "omega",
        "Δ": "Delta",
        "Θ": "Theta",
        "Λ": "Lambda",
        "Ξ": "Xi",
        "Π": "Pi",
        "Σ": "Sigma",
        "Φ": "Phi",
        "Ψ": "Psi",
        "Ω": "Omega",
    }

    for old, new in greek_replacements.items():
        text = text.replace(old, new)

    # Handle common PDF encoding where tab character represents gamma
    # in mathematical contexts like γ(H) -> \t(H)
    # Must do AFTER pdf_char_replacements converted ð to (
    text = re.sub(r"\t(\(|\[)", r"gamma\1", text)

    # Normalize subscripts: remove underscores before letters/numbers
    # This handles cases like ξ(0)_GL vs ξ(0)GL
    text = re.sub(r"_([A-Za-z0-9])", r"\1", text)

    # Normalize Angstrom symbol variations
    # Å (precomposed) -> angstrom
    text = text.replace("Å", "angstrom")
    # A˚ (A + combining ring) -> angstrom (after NFKD, may have space: "A ̊")
    text = re.sub(
        r"A\s*[\u0300-\u036f]+", "angstrom", text
    )  # A with optional space and combining diacritics
    # Also handle lowercase after processing
    text = re.sub(r"a\s*[\u0300-\u036f]+", "angstrom", text)

    # Preserve '...' as a unique token (handle after symbol replacements)
    if "..." in text:
        text = text.replace("...", " ELLIPSIS ")

    # Handle hyphenated line breaks (e.g., "super-\nconductor" or "super- conductor")
    # This is common in PDFs where words are split across lines
    # Pattern: word characters, hyphen, optional whitespace (including newlines), word characters
    text = re.sub(
        r"(\w+)-\s+(\w+)", r"\1\2", text
    )  # "super- conductor" -> "superconductor"

    # Remove non-alphanumeric (but keep unicode letters) by replacing with space
    # \w matches any unicode word character
    text = re.sub(r"[^\w]", " ", text).lower()
    return text.strip()


def is_valid_page(page_num: str | int | float | None) -> tuple[bool, int | None]:
    """Checks if page_num is a valid integer string or number.

    Args:
        page_num: The page number to check
    Returns:
        is_valid: True if the page number is valid, False otherwise
        integer_value: The integer value of the page number if valid, None otherwise

    """
    if pd.isna(page_num) or page_num == "":
        return False, None
    try:
        # Handle "1.0" or 1.0 -> 1
        val = int(float(page_num))
        return True, val
    except ValueError:
        return False, None


def get_tokens(text: str) -> list[str]:
    """Split text into alphanumeric tokens, filtering out ellipsis tokens.

    Args:
        text: The text to split into tokens
    Returns:
        The list of tokens

    """
    if not isinstance(text, str):
        return []
    text = normalize(text)
    tokens = text.split()
    # Filter out ellipsis tokens - we'll handle them separately in matching
    return [t for t in tokens if t != "ellipsis"]


def find_best_match_on_page(
    page: fitz.Page, evidence_text: str
) -> tuple[float, list[fitz.Rect]]:
    """Searches a single PyMuPDF page for the evidence text.

    Args:
        page: The PyMuPDF page to search
        evidence_text: The text to search for
    Returns:
        score: 0.0 to 1.0 (1.0 = perfect match)
        rects: The list of rectangles of the evidence text

    """
    if (
        not evidence_text
        or not isinstance(evidence_text, str)
        or evidence_text == "nan"
    ):
        return 0.0, []

    # Strategy 1: Exact Match (Fastest)
    text_instances = page.search_for(evidence_text)
    if text_instances:
        # If we find exact matches, we consider it a score of 1.0
        return 1.0, text_instances

    # Strategy 2: Fuzzy Sequence Match
    # 1. Get all words from page with their bboxes
    words = page.get_text("words")

    if not words:
        return 0.0, []

    # 2. Tokenize Evidence and Page Words
    evidence_tokens = get_tokens(evidence_text)
    if not evidence_tokens:
        return 0.0, []

    page_tokens = []  # list of (token, word_idx)
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

    # Allow starting from the first FEW tokens to handle missing prefixes (like Greek chars stripped)
    possible_starts = []
    start_tokens_to_check = evidence_tokens[:3]  # Check first 3 tokens

    for s_idx, s_token in enumerate(start_tokens_to_check):
        # find original index in evidence_tokens
        full_e_idx = evidence_tokens.index(s_token)

        matches = [
            i
            for i, pt in enumerate(page_tokens)
            if pt[0] == s_token or (len(s_token) > 2 and s_token in pt[0])
        ]
        for m in matches:
            possible_starts.append((m, full_e_idx))

    # Optimization: if evidence is long, maybe try to match rare tokens?
    # For now, stick to left-to-right scan.

    for start_p_idx, start_e_idx in possible_starts:
        current_match_word_indices = []
        p_idx = start_p_idx
        e_idx = start_e_idx

        # Align sequence allowing for skips
        # Note: ellipsis tokens are already filtered out by get_tokens(),
        # so we just match the actual content tokens flexibly
        while p_idx < len(page_tokens) and e_idx < len(evidence_tokens):
            p_token = page_tokens[p_idx][0]
            e_token = evidence_tokens[e_idx]

            # 1. Exact Match
            if p_token == e_token:
                current_match_word_indices.append(page_tokens[p_idx][1])
                p_idx += 1
                e_idx += 1
                continue

            # 2. Substring Match (e.g. 001 inside 2001)
            if len(e_token) > 1 and e_token in p_token:
                current_match_word_indices.append(page_tokens[p_idx][1])
                p_idx += 1
                e_idx += 1
                continue

            # 3. Skip Mismatch (Lookahead in PDF - "Extra words in PDF")
            # Increased lookahead to handle evidence with ellipses (tokens far apart)
            # Some evidence strings have tokens 30+ words apart (e.g., "coherence length ... 138")
            found_next_pdf = False
            lookahead = 20  # Allow matching across larger gaps
            for look in range(1, lookahead + 1):
                if p_idx + look < len(page_tokens):
                    cand = page_tokens[p_idx + look][0]
                    # Check if the next evidence token matches a future page token
                    if cand == e_token or (len(e_token) > 2 and e_token in cand):
                        p_idx += look
                        found_next_pdf = True
                        break

            if found_next_pdf:
                continue

            # 4. Skip Mismatch (Lookahead in Evidence - "Missing words in PDF")
            # If the current page token doesn't match current evidence,
            # maybe it matches the NEXT evidence token? (i.e. we skip one word in evidence)
            if e_idx + 1 < len(evidence_tokens):
                next_e_token = evidence_tokens[e_idx + 1]
                # Check if current page token matches NEXT evidence token
                if p_token == next_e_token or (
                    len(next_e_token) > 2 and next_e_token in p_token
                ):
                    e_idx += 1  # Skip the current failing evidence token
                    continue

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


def find_evidence_in_pdf(
    pdf_path: Path,
    evidence_text: str,
    suggested_page_num: int | None = None,
    scan_all_pages: bool = True,
) -> tuple[int | None, float, list[fitz.Rect]]:
    """Search for evidence in the PDF.

    Args:
        pdf_path: Path to PDF file.
        evidence_text: The text string to find.
        suggested_page_num: 1-based page number to check first.
        scan_all_pages: If False, only check suggested_page_num.

    Returns:
        best_page_num: 1-based page number of the best match
        best_score: 0.0 to 1.0 (1.0 = perfect match)
        best_rects: The list of rectangles of the best match

        best_page_num is 1-based.

    """
    best_global_score = 0.0
    best_global_page = None
    best_global_rects = []

    try:
        with fitz.open(pdf_path) as doc:
            # Check suggested page first
            if suggested_page_num is not None:
                p_idx = int(suggested_page_num) - 1
                if 0 <= p_idx < len(doc):
                    score, rects = find_best_match_on_page(doc[p_idx], evidence_text)
                    # If good enough, return immediately
                    if score > 0.85:
                        return suggested_page_num, score, rects

                    best_global_score = score
                    best_global_page = suggested_page_num
                    best_global_rects = rects

            # If we are strictly checking only the suggested page, return what we found there
            if not scan_all_pages:
                return best_global_page, best_global_score, best_global_rects

            # Global search (if strict check failed or wasn't requested)
            # Only scan if we didn't get a perfect match already
            step = 1
            for p_idx in range(0, len(doc), step):
                # Skip suggested page as we already checked it
                if suggested_page_num is not None and p_idx == (
                    int(suggested_page_num) - 1
                ):
                    continue

                score, rects = find_best_match_on_page(doc[p_idx], evidence_text)

                if score > best_global_score:
                    best_global_score = score
                    best_global_page = p_idx + 1
                    best_global_rects = rects

                    # Early exit if we found a very good match
                    if best_global_score > 0.9:
                        break

    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        return None, 0.0, []

    return best_global_page, best_global_score, best_global_rects


# --- Main App Logic ---


def get_pdf_images(
    pdf_path: Path,
    highlight_rects: list[fitz.Rect] | None = None,
    page_num: int | None = None,
) -> list[tuple[bytes, str]] | tuple[list[tuple[bytes, str]], str]:
    """Generates a list of images for each page in the PDF.
    Target page will have highlights applied.

    Args:
        pdf_path: Path to the PDF file
        highlight_rects: The list of rectangles to highlight (optional)
        page_num: The 1-based page number to highlight (optional)

    Returns:
        list of (img_bytes, page_caption): The list of images and page captions
        error_message: The error message if there is an error

    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        return [], f"Error opening PDF: {e}"

    images = []
    # Handle page_num as 0-indexed integer for internal usage
    target_page_idx = int(page_num) - 1 if page_num else -1

    for i, page in enumerate(doc):
        # Apply highlights only to the target page
        if highlight_rects and i == target_page_idx:
            for rect in highlight_rects:
                page.add_highlight_annot(rect)

        # Render page
        # Matrix(2, 2) = 2x zoom for clear text
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_bytes = pix.tobytes("png")

        caption = f"Page {i + 1}"
        if i == target_page_idx:
            caption += " (Evidence)"

        images.append((img_bytes, caption))

    doc.close()
    return images, None


def main() -> None:
    """Main function to run the app."""
    st.title("Property Validator")

    # Custom CSS to make the PDF viewer fill available viewport height
    st.markdown(
        """
        <style>
        /* Target the PDF container and make it fill viewport */
        [data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stImage"]) {
            height: calc(100vh - 220px) !important;
            max-height: calc(100vh - 220px) !important;
            overflow-y: auto !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

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

        # PDF Filter - allows filtering properties by PDF file
        st.divider()
        st.subheader("PDF Filter")

        # Get available PDFs from the dataframe
        available_pdfs = []
        if (
            st.session_state.df is not None
            and not st.session_state.df.empty
            and "paper_pdf_path" in st.session_state.df.columns
        ):
            # Extract unique PDF filenames from the dataframe
            pdf_paths = st.session_state.df["paper_pdf_path"].dropna().unique()
            available_pdfs = sorted(
                [os.path.basename(p) for p in pdf_paths if p and isinstance(p, str)]
            )

        # Initialize selected PDF in session state
        if "selected_pdf" not in st.session_state:
            st.session_state.selected_pdf = None

        # Reset PDF selection when CSV changes
        if "last_csv_for_pdf" not in st.session_state:
            st.session_state.last_csv_for_pdf = st.session_state.selected_csv
        elif st.session_state.last_csv_for_pdf != st.session_state.selected_csv:
            st.session_state.selected_pdf = None
            st.session_state.last_csv_for_pdf = st.session_state.selected_csv

        # PDF selector dropdown
        pdf_index = None
        if (
            st.session_state.selected_pdf
            and st.session_state.selected_pdf in available_pdfs
        ):
            pdf_index = available_pdfs.index(st.session_state.selected_pdf)

        selected_pdf = st.selectbox(
            "Filter by PDF",
            options=[None] + available_pdfs,
            index=0 if pdf_index is None else pdf_index + 1,
            format_func=lambda x: "All PDFs" if x is None else x,
            placeholder="Choose a PDF file...",
        )

        # Update session state if changed
        if selected_pdf != st.session_state.selected_pdf:
            st.session_state.selected_pdf = selected_pdf
            st.session_state.current_property_index = 0  # Reset to first property
            st.rerun()

        st.divider()

        # Validator Name
        validator_name = st.text_input(
            "Validator Name", value=st.session_state.get("validator_name", "")
        )
        st.session_state.validator_name = validator_name

    # Unified Stop Condition
    if not st.session_state.selected_csv:
        st.info("Please select a CSV file from the sidebar to continue.")
        st.stop()

    csv_path = os.path.join(CSV_FOLDER, st.session_state.selected_csv)

    # Ensure Session State is ready (redundant but safe)
    if "df" not in st.session_state or st.session_state.df is None:
        st.session_state.df = load_data(csv_path)

    if st.session_state.df.empty:
        st.stop()

    # Continue with sidebar metrics
    with st.sidebar:
        # Calculate metrics based on PDF filter if applied
        metrics_df = st.session_state.df
        if st.session_state.get("selected_pdf"):
            selected_pdf_basename = st.session_state.selected_pdf
            pdf_mask = metrics_df["paper_pdf_path"].apply(
                lambda x: os.path.basename(str(x)) == selected_pdf_basename
                if pd.notna(x)
                else False
            )
            metrics_df = metrics_df[pdf_mask]
            st.caption(f"Filtered by: {selected_pdf_basename}")

        st.metric("Total Properties", len(metrics_df))
        validated_count = metrics_df["validated"].notna().sum()
        st.metric("Validated", validated_count)

        # Filter controls
        filter_status = st.radio(
            "Show", ["All", "Pending", "Valid", "Invalid", "Flagged"], index=1
        )

    # Filter Data based on selection
    # We will display using the masked dataframe but save on the original dataframe (st.session_state.df)
    df_selected_pdf = st.session_state.df

    # First, filter by PDF if one is selected
    if st.session_state.get("selected_pdf"):
        selected_pdf_basename = st.session_state.selected_pdf
        # Filter to only rows where paper_pdf_path ends with the selected PDF filename
        pdf_mask = df_selected_pdf["paper_pdf_path"].apply(
            lambda x: os.path.basename(str(x)) == selected_pdf_basename
            if pd.notna(x)
            else False
        )
        df_selected_pdf = df_selected_pdf[pdf_mask]

    # Then filter by validation status
    if filter_status == "Pending":
        filtered_indices = df_selected_pdf[df_selected_pdf["validated"].isna()].index
    elif filter_status == "Valid":
        filtered_indices = df_selected_pdf[df_selected_pdf["validated"] == True].index  # noqa: E712
    elif filter_status == "Invalid":
        filtered_indices = df_selected_pdf[df_selected_pdf["validated"] == False].index  # noqa: E712
    elif filter_status == "Flagged":
        filtered_indices = df_selected_pdf[df_selected_pdf["flagged"] == True].index  # noqa: E712
    else:
        filtered_indices = df_selected_pdf.index

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
    def select_property(position: int) -> None:
        position = max(0, min(len(filtered_indices) - 1, position))
        st.session_state.current_property_index = position

    # Make sure the current index stays within bounds when filters change
    st.session_state.current_property_index = max(
        0, min(st.session_state.current_property_index, len(filtered_indices) - 1)
    )

    # Ensure current index is within bounds of filtered list
    if st.session_state.current_property_index >= len(filtered_indices):
        st.session_state.current_property_index = 0

    # Create a nice label for the dropdown
    def get_label(idx: int) -> str:
        row = df_selected_pdf.loc[idx]
        val_status = "❓"
        if row["validated"]:
            val_status = "✅"
        elif not row["validated"]:
            val_status = "❌"

        flag_status = "🚩 " if row["flagged"] else ""
        return f"{flag_status}{val_status} {row['id']} - {row['property_name']} ({row['value_string']})"

    # Show dropdown for property selection first
    dropdown_index = st.selectbox(
        "Select Property",
        filtered_indices,
        format_func=get_label,
        index=st.session_state.current_property_index,
    )

    # Navigation buttons with keyboard shortcuts (below the dropdown)
    nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])

    with nav_col1:
        if st.button("⬅️ Previous (←)", width="stretch", key="prev_btn"):
            if st.session_state.current_property_index > 0:
                select_property(st.session_state.current_property_index - 1)
                st.rerun()

    with nav_col2:
        st.markdown(
            f"<div style='text-align: center; padding: 8px;'>Property {st.session_state.current_property_index + 1} of {len(filtered_indices)}</div>",
            unsafe_allow_html=True,
        )

    with nav_col3:
        if st.button("Next (→) ➡️", width="stretch", key="next_btn"):
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

        // Navigation shortcuts
        if (e.key === 'ArrowLeft') {
            e.preventDefault();
            findAndClickButton('Previous');
        } else if (e.key === 'ArrowRight') {
            e.preventDefault();
            findAndClickButton('Next');
        }
        // Validation shortcuts
        else if (e.key === 'v' || e.key === 'V') {
            e.preventDefault();
            findAndClickButton('Valid');
        }
        else if (e.key === 'x' || e.key === 'X') {
            e.preventDefault();
            findAndClickButton('Invalid');
        }
    };

    streamlitDoc.addEventListener('keydown', streamlitDoc.navKeyHandler);
    </script>
    """
    components.html(keyboard_js, height=0)

    # Get selected index from current position
    selected_index = filtered_indices[st.session_state.current_property_index]

    row = df_selected_pdf.loc[selected_index]

    # Layout
    col1, col2 = st.columns([1, 1.5])  # Left panel (info + list), Right panel (PDF)

    with col1:
        # Top Left: Validation Card
        st.subheader("Validation")
        # st.write("**Is this valid?**")

        # Check if validator name is provided
        name_provided = validator_name and validator_name.strip() != ""

        if not name_provided:
            st.warning("Please enter your name in the sidebar to begin validation.")

        is_validated = row["validated"]
        is_flagged = row["flagged"]

        if is_flagged:
            st.warning("This property is flagged for review.")

        container = st.container(border=True)
        with container:
            st.markdown(f"**Material:** `{row['material_or_system']}`")
            st.markdown(f"**Property:** `{row['property_name']}`")
            st.markdown(f"**Value String:** `{row['value_string']}`")
            st.markdown(f"**Value Number:** `{row['value_number']}`")
            st.markdown(f"**Unit:** `{row['units']}`")
            st.markdown(
                f"**Location:** Page {row['location.page']}, {row['location.section']}"
            )
            st.markdown(f"**Evidence:** _{row['location.evidence']}_")

            with st.expander("More info", expanded=True):
                # Define columns to exclude (displayed above + system cols)
                displayed_cols = {
                    "property_name",
                    "value_string",
                    "value_number",
                    "units",
                    "material_or_system",
                    "location.page",
                    "location.section",
                    "location.evidence",
                }
                system_cols = {
                    "validated",
                    "validator_name",
                    "validation_date",
                    "flagged",
                    "paper_pdf_path",
                    "validator_note",
                }
                exclude_cols = displayed_cols.union(system_cols)

                # Iterate and display
                for col_name, val in row.items():
                    if (
                        col_name not in exclude_cols
                        and pd.notna(val)
                        and str(val).strip() != ""
                    ):
                        st.markdown(f"**{col_name}:** `{val}`")

            # Validation Controls
            # Give more weight to Valid/Invalid to prevent wrapping
            col_valid, col_invalid, col_flag, col_reset = st.columns(
                [1.5, 1.5, 1.1, 1.1]
            )

            # Determine button styles based on state
            valid_type = "primary" if is_validated is True else "secondary"
            invalid_type = "primary" if is_validated is False else "secondary"
            flag_type = "primary" if is_flagged else "secondary"

            # Alias the original dataframe to df for easy access
            df_for_saving = st.session_state.df

            with col_valid:
                # Modifying Valid Button to Handle Pending AI Matches
                if st.button(
                    "✅ Valid (V)",
                    type=valid_type,
                    width="stretch",
                    disabled=not name_provided,
                ):
                    # Check if there is a pending AI suggestion for this row
                    if (
                        "pending_ai_match" in st.session_state
                        and st.session_state.pending_ai_match.get("index")
                        == selected_index
                    ):
                        suggestion = st.session_state.pending_ai_match
                        df_for_saving.at[selected_index, "location.page"] = suggestion["page"]
                        df_for_saving.at[selected_index, "location.evidence"] = suggestion[
                            "evidence"
                        ]
                        df_for_saving.at[selected_index, "location.section"] = "AI Discovered"
                        df_for_saving.at[selected_index, "location.source_type"] = "text"

                        # Clear the pending match after accepting
                        del st.session_state.pending_ai_match

                    df_for_saving.at[selected_index, "validated"] = True
                    df_for_saving.at[selected_index, "validator_name"] = (
                        st.session_state.validator_name
                    )
                    df_for_saving.at[selected_index, "validation_date"] = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    # Auto-advance if filter preserves the item (All, Flagged, etc)
                    if filter_status in ["All", "Flagged"]:
                        current_pos = st.session_state.current_property_index
                        for i in range(current_pos + 1, len(filtered_indices)):
                            idx = filtered_indices[i]
                            if pd.isna(df_for_saving.at[idx, "validated"]):
                                st.session_state.current_property_index = i
                                break

                    save_data(df_for_saving)
                    st.session_state.df = df_for_saving
                    st.rerun()

            with col_invalid:
                if st.button(
                    "❌ Invalid (X)",
                    type=invalid_type,
                    width="stretch",
                    disabled=not name_provided,
                ):
                    df_for_saving.at[selected_index, "validated"] = False
                    df_for_saving.at[selected_index, "validator_name"] = (
                        st.session_state.validator_name
                    )
                    df_for_saving.at[selected_index, "validation_date"] = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

                    # Also clear pending AI match if invalidated
                    if (
                        "pending_ai_match" in st.session_state
                        and st.session_state.pending_ai_match.get("index")
                        == selected_index
                    ):
                        del st.session_state.pending_ai_match

                    # Auto-advance if filter preserves the item (All, Flagged, etc)
                    if filter_status in ["All", "Flagged"]:
                        current_pos = st.session_state.current_property_index
                        for i in range(current_pos + 1, len(filtered_indices)):
                            idx = filtered_indices[i]
                            if pd.isna(df_for_saving.at[idx, "validated"]):
                                st.session_state.current_property_index = i
                                break

                    save_data(df_for_saving)
                    st.session_state.df = df_for_saving
                    st.rerun()

            with col_flag:
                flag_label = "Unflag (F)" if is_flagged else "Flag (F)"
                if st.button(flag_label, type=flag_type, width="stretch"):
                    new_flag_state = not is_flagged
                    df_for_saving.at[selected_index, "flagged"] = new_flag_state

                    # Auto-advance to next property
                    if (
                        st.session_state.current_property_index
                        < len(filtered_indices) - 1
                    ):
                        st.session_state.current_property_index += 1

                    save_data(df_for_saving)
                    st.session_state.df = df_for_saving
                    st.rerun()

            with col_reset:
                if st.button("Reset", type="secondary", width="stretch"):
                    df_for_saving.at[selected_index, "validated"] = None
                    df_for_saving.at[selected_index, "validator_name"] = ""
                    df_for_saving.at[selected_index, "validation_date"] = ""

                    # Clear pending match
                    if (
                        "pending_ai_match" in st.session_state
                        and st.session_state.pending_ai_match.get("index")
                        == selected_index
                    ):
                        del st.session_state.pending_ai_match

                    save_data(df_for_saving)
                    st.session_state.df = df_for_saving
                    st.rerun()

            # Notes Section
            current_note = row.get("validator_note", "")
            if pd.isna(current_note):
                current_note = ""
            current_note = str(current_note)

            new_note = st.text_area(
                "Add notes:",
                value=current_note,
                key=f"note_input_{selected_index}",
                height=100,
                placeholder="Enter notes here...",
            )

            if new_note != current_note:
                df_for_saving.at[selected_index, "validator_note"] = str(new_note)
                save_data(df_for_saving)
                st.session_state.df = df_for_saving
                st.toast("Note saved!")

        # Prepare display dataframe (just for data access)
        # 1. Start index at 1
        # display_df = df[display_cols].loc[filtered_indices].copy()

        # DataFrame Implementation with Sticky "Status | ID"

        # Create a display copy to manipulate
        display_df = df_selected_pdf.loc[filtered_indices].copy()

        # 1. Helper to get icons
        def get_status_icon(val: bool) -> str:
            if val is True:
                return "✅"
            if val is False:
                return "❌"
            return "❓"

        # 2. Create Merged "Status | ID" column
        # e.g. "✅ prop_001"
        display_df["status_icon"] = display_df["validated"].apply(get_status_icon)
        display_df["flag_icon"] = display_df["flagged"].apply(
            lambda x: "🚩 " if x else ""
        )

        display_df["status_id"] = (
            display_df["flag_icon"]
            + display_df["status_icon"]
            + " "
            + display_df["id"].astype(str)
        )

        # 3. Select Columns to Display
        cols_to_show = [
            "status_id",
            "material_or_system",
            "property_name",
            "value_string",
            "value_number",
            "units",
        ]
        final_display_df = display_df[cols_to_show].copy()

        # Back to Dataframe: Best option for "Row = 1 Click" + "Spreadsheet Layout" + "Small Text"

        # 4. 0-Based RangeIndex for Display
        # This aligns Row 0 with Index 0, avoiding off-by-one confusion if IDs are 0-based
        final_display_df.index = range(0, len(final_display_df))

        event = st.dataframe(
            final_display_df,
            column_config={
                "status_id": st.column_config.TextColumn(
                    "Status | ID",
                    help="Validation Status and Property ID",
                    width="stretch",
                    pinned=True,
                ),
                "material_or_system": st.column_config.TextColumn(
                    "Material/System", width="stretch"
                ),
                "property_name": st.column_config.TextColumn(
                    "Property", width="medium"
                ),
                "value_string": st.column_config.TextColumn(
                    "Value String", width="medium"
                ),
                "value_number": st.column_config.TextColumn("Value Num", width="small"),
                "units": st.column_config.TextColumn("Unit", width="small"),
            },
            width="stretch",
            height=400,
            hide_index=False,  # Show the 0-based index
            on_select="rerun",
            selection_mode="single-row",
        )

        if len(event.selection.rows):
            # With RangeIndex(0..N), selection returns the integer label
            selected_idx_label = event.selection.rows[0]
            # print(f"DEBUG: Selected Label: {selected_idx_label}")

            # Simple math: 0-based numeric label -> 0-based list index
            # Label 0 -> Index 0
            if isinstance(selected_idx_label, int):
                selected_list_position = selected_idx_label

                # Check bounds just in case
                if 0 <= selected_list_position < len(filtered_indices):
                    # Update state only if changed
                    if (
                        st.session_state.current_property_index
                        != selected_list_position
                    ):
                        st.session_state.current_property_index = selected_list_position
                        st.rerun()
            else:
                # Fallback if something weird happens (e.g. string label)
                pass

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

        page_num = row["location.page"]
        evidence_text = row["location.evidence"]

        # Check for pending AI match for this row
        pending_match = None
        if (
            "pending_ai_match" in st.session_state
            and st.session_state.pending_ai_match.get("index") == selected_index
        ):
            pending_match = st.session_state.pending_ai_match
            page_num = pending_match["page"]  # Override for viewing
            evidence_text = pending_match["evidence"]  # Override for viewing

        if pdf_path:
            # Check effective page num
            valid_page, p_num = is_valid_page(page_num)

            pdf_images = []

            if valid_page:
                rects = []
                score = 0.0
                if pending_match:
                    # Highlight pending AI match (strict on the found AI page)
                    _, score, rects = find_evidence_in_pdf(
                        pdf_path,
                        evidence_text,
                        suggested_page_num=p_num,
                        scan_all_pages=False,
                    )
                elif (
                    evidence_text
                    and str(evidence_text).lower() != "nan"
                    and str(evidence_text).strip() != ""
                ):
                    # Highlight existing CSV evidence (User Request: Strict check on THIS page only)
                    _, score, rects = find_evidence_in_pdf(
                        pdf_path,
                        evidence_text,
                        suggested_page_num=p_num,
                        scan_all_pages=False,
                    )

                    # 2. If Strict Search failed (low score), try Global Search
                    if score < 0.8:
                        best_page, best_score, best_rects = find_evidence_in_pdf(
                            pdf_path,
                            evidence_text,
                            suggested_page_num=p_num,
                            scan_all_pages=True,
                        )

                        # If Global Search found a strong match
                        if best_score > 0.8:
                            # Check if page changed
                            old_page_int = (
                                int(p_num)
                                if pd.notna(p_num)
                                and str(p_num).replace(".", "", 1).isdigit()
                                else -1
                            )
                            new_page_int = int(best_page)

                            # Always use the better rects/score from global search
                            p_num = new_page_int
                            rects = best_rects
                            score = best_score

                            if new_page_int != old_page_int:
                                # Auto-Correction event
                                df_for_saving = st.session_state.df
                                df_for_saving.at[selected_index, "location.page"] = p_num
                                save_data(df_for_saving)
                                st.session_state.df = df_for_saving
                                st.toast(f"Page auto-corrected to {p_num}", icon="🔄")

                # Filter out low confidence matches
                if score < 0.6:
                    rects = []

                # Get all images
                pdf_images, error = get_pdf_images(
                    pdf_path, highlight_rects=rects, page_num=p_num
                )

                if not pdf_images:
                    st.error(error)

            # If not valid page AND no pending match, show search
            # If pending match exists, we fall through to the image renderer above (since p_num is valid from AI)
            elif pd.isna(page_num) or page_num == "":
                st.info("No page number specified.")

                # Show full PDF without highlights if no page specified?
                # User requested "just convert each pdf page to an image and then display this".
                # So we should probably show it even if no page num, just no highlights.
                pdf_images, _ = get_pdf_images(pdf_path)

                # AI Search Button
                st.divider()
                if st.button("🤖 Find with AI", key="ai_search_btn"):
                    with st.spinner("Asking Gemini to find the value in the paper..."):
                        ai_page, ai_evidence, ai_conf = search_with_ai(
                            pdf_path,
                            value=str(row["value_string"]) or str(row["value_number"]),
                            prop_name=row["property_name"],
                            unit=str(row["units"]) if pd.notna(row["units"]) else "",
                            material_or_system=str(row["material_or_system"])
                            if "material_or_system" in row
                            and pd.notna(row["material_or_system"])
                            else "",
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
                            scan_all_pages=True,
                        )

                        if found_page and found_score > 0.8:
                            # If we found looking on the "AI Page" or "Any Page" with high confidence, use THAT page.
                            ai_page = found_page

                        # Save to session state as pending match
                        st.session_state.pending_ai_match = {
                            "index": selected_index,
                            "page": ai_page,
                            "evidence": ai_evidence,
                            "confidence": ai_conf,
                        }
                        st.rerun()  # Rerun to render the found page immediately
                    else:
                        if "AI returned empty list" in ai_evidence:
                            st.error("Gemini returned an empty list.")
                        else:
                            st.error(
                                f"Gemini could not find the value. ({ai_evidence})"
                            )
            else:
                st.warning(f"Invalid page number: {page_num}")

            # Render Images
            if pdf_images:
                # Create a scrollable container
                with st.container(height=1500):
                    for i, (img_bytes, caption) in enumerate(pdf_images):
                        # Add an anchor for Javascript to target
                        # The page_num is user-facing (1-based), so we use that index logic
                        # caption "Page X (Evidence)" or "Page X"

                        # Check if this is the target page to scroll to
                        is_target = False
                        if valid_page and (i + 1) == p_num:
                            is_target = True

                        # Inject a small script to scroll to this element if it is the target
                        # We use a unique ID based on the loop index to ensure specificity
                        if is_target:
                            # We insert a dummy element to target - BEFORE the image so we scroll to top
                            st.markdown(
                                '<div id="target_page_scroll_marker"></div>',
                                unsafe_allow_html=True,
                            )

                            # Javascript to scroll this marker into view
                            # We must access window.parent to break out of the component iframe
                            js = """
                             <script>
                                 setTimeout(function() {
                                     try {
                                         const element = window.parent.document.getElementById("target_page_scroll_marker");
                                         if (element) {
                                             element.scrollIntoView({behavior: "smooth", block: "start"});
                                         }
                                     } catch (e) {
                                         console.log("Auto-scroll failed:", e);
                                     }
                                 }, 500);
                             </script>
                             """
                            components.html(js, height=0)

                        st.image(img_bytes, caption=caption, width="stretch")
            elif pdf_path and not pdf_images and (pd.isna(page_num) or page_num == ""):
                # Fallback if get_pdf_images failed or returned empty but we have a path
                # Try one more time without highlights
                pdf_images, _ = get_pdf_images(pdf_path)
                if pdf_images:
                    with st.container(height=1500):
                        for img_bytes, caption in pdf_images:
                            st.image(img_bytes, caption=caption, width="stretch")


if __name__ == "__main__":
    main()
