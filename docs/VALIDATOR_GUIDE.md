# Validator App User Guide

This guide explains how to use the Property Validator App for two main workflows: validating existing extracted data and finding missing data using AI.

## Setup & Configuration

1. Following the setup instructions in [README.md](../README.md#setup-instructions)

2. Install additional dependencies:

```bash
uv sync --group validator
```

3. Setup API keys for **AI Search** feature (optional): create a file named `.env` in the root directory and add

```bash
GOOGLE_API_KEY=your_api_key_here
```

## Getting Started

Run the app using Streamlit:

```bash
./src/pbench_validator_app/app.py
```

By default, the app looks for files in:
*   **CSVs**: `examples/extraction/assets/validate_csv/`
*   **PDFs**: `examples/extraction/assets/Paper_DB/`

You can customize these folders using CLI flags:

```bash
./src/pbench_validator_app/app.py --csv_folder /path/to/unsupervised_llm_extraction/ --paper_folder /path/to/Paper_DB/
```

**Workflow**:
1.  **Launch**: When the app starts, no files are selected. You will see a message: `"Please select a CSV file and corresponding PDF file from the sidebar to continue."`
2.  **Select CSV**: Choose your target CSV file from the sidebar.
3.  **Select PDF**: Immediately below the CSV selector, choose the corresponding PDF file.
  - Only after BOTH are selected will the validation interface appear.

> **Note**: Selecting a PDF automatically updates the `paper_pdf_path` column in your CSV to link them together.


---

## Workflow 1: Validating Existing Data

This workflow is for when your CSV already contains `location.page` and `location.evidence`.

### CSV Requirements
Your CSV **must** contain the following columns to enable validation:
*   **`location.page`**: The page number where the evidence is located.
*   **`location.evidence`**: The specific text quote that supports the property value.
*   **`property_name`**: The name of the property.
*   **`value_string`** OR **`value_number`**: The value being validated.
*   **`units`** (Optional): The unit of the value.
*   **`paper_pdf_path`** (Optional but recommended): Path to the PDF file. If missing, you will need to select it manually in the app.

1.  **Select a Property**: Use the dropdown or "Previous/Next" buttons to navigate through the properties.
2.  **Review Evidence**:
    *   The app will automatically locate the page specified in `location.page`.
    *   It will attempt to **highlight** the text specified in `location.evidence`.
    *   If the text is found on that page, it will be highlighted in **yellow**.
    *   If the text is *not* found on that page, **no highlights** will appear. The app strictly searches only the specified page to avoid showing incorrect matches from other pages.
3.  **Validate**:
    *   **✅ Valid**: Marks the property as valid ONLY if property name, value, evidence are all correct.
    *   **❌ Invalid**: Marks the property as invalid if any of the property name, value, evidence are incorrect.
    *   **Flag**: Flags the property for further review.
    *   **Reset**: Resets the validation status to "Not Validated".

---

## Workflow 2: Finding Missing Data with AI

This workflow is for when your CSV has missing `location.page` (or just `nan`) but you want to find the evidence in the PDF.

### CSV Requirements for AI
To use this feature effectively, your CSV **must** contain the following columns:
*   **`property_name`**: The specific property to look for (e.g., "superconducting transition temperature Tc").
*   **`value_string`** OR **`value_number`**: The target value to find.
*   **`units`** (Optional): Helps refine the search (e.g., "K", "meV").
*   **`material_or_system`** (Optional): Provides context to the AI.

1.  **Find with AI**:
    *   Click the **🤖 Find with AI** button in the Evidence Viewer.
    *   The app uses Gemini to search the PDF for the property value, property name, and unit associated with the material.
2.  **Review the Match**:
    *   **AI Suggestion**: The app will display the page the AI found.
    *   **Auto-Correction**: If the AI suggests a page (e.g., Page 5) but the text is actually on Page 4, the app automatically detects this and displays the correct page (Page 4).
    *   **Visual Confirmation**: The evidence text found by the AI will be highlighted in yellow. The caption will indicate "(AI Suggestion - Pending Validation)".
3.  **Accept & Validate**:
    *   Click **✅ Valid**.
    *   This automatically **saves** the AI-discovered page number (location.page), evidence text (location.evidence), and section (location.section) into your CSV file.
    *   The match is confirmed and the property is marked as validated.

---

## Troubleshooting highlights

-   **"No highlights on the page"**:
    -   This means the exact text in `location.evidence` (or the AI result) could not be found *on that specific page* with high confidence (>60% match).
    -   This is a feature, not a bug, to prevent misleading "ghost" highlights from other parts of the document.

-   **"P63mc is not highlighting"**:
    -   The app uses a "Greedy Merge" algorithm to handle complex chemical formulas (e.g., `P63mc`) that PDFs often split into multiple tokens (`P6` + `3` + `mc`). It should highlight correctly.
