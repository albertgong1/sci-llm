"""Cluster unique property values using Gemini.

This script reads unique materials science property values from a CSV file,
uses the Gemini LLM to cluster them into canonical categories based on scientific
definitions, and saves the mapping to a JSON file.

Example usage:
    python examples/extraction/cluster_properties.py
    python examples/extraction/cluster_properties.py --model gemini-2.5-flash
    python examples/extraction/cluster_properties.py --model gemini-3-pro --output property_clusters.json
    python examples/extraction/cluster_properties.py \
        --unique_values assets/SuperCon Property Extraction Dataset - Unique Property Values.csv \
        --glossary assets/SuperCon Property Extraction Dataset - Glossary.csv \
        --model gemini-3-pro-preview
"""

import os
import json
import pandas as pd
# import google.generativeai as genai
import google.genai as genai
from typing import Dict, List
import time
import logging
import argparse 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
# Defaults (can be overridden by CLI args)
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "assets")
DEFAULT_UNIQUE_VALUES_CSV = os.path.join(DEFAULT_DATA_DIR, "SuperCon Property Extraction Dataset - Unique Property Values.csv")
DEFAULT_GLOSSARY_FILE = os.path.join(DEFAULT_DATA_DIR, "SuperCon Property Extraction Dataset - Glossary.csv")
DEFAULT_MODEL = "gemini-3-pro-preview"

# Target Properties to Cluster (using exact names from CSV)
TARGET_PROPERTIES = [
    "method of Hc1 derivation",
    "method of Hc2 derivation",
    "method of dHc2/dT derivation",
    "method of COHERE derivation",
    "method of PENET derivation",
    "method for derivation of Debye temperature",
    "*method of analysis for structure",
]

def load_glossary_definitions(glossary_path: str) -> Dict[str, str]:
    """
    Loads property definitions from the glossary CSV.
    Uses 'label' as the key and 'definition' as the value.
    This glossary file provides the semantic context needed for the LLM 
    to understand what each property (e.g., 'method of Hc1') actually means physically.
    """
    if not os.path.exists(glossary_path):
        raise FileNotFoundError(f"Glossary file not found at {glossary_path}")
        
    df = pd.read_csv(glossary_path)
    definitions = {}
    for _, row in df.iterrows():
        # Clean up column names just in case, but assume standard format
        if 'label' in row and 'definition' in row and pd.notna(row['label']) and pd.notna(row['definition']):
             definitions[row['label'].strip()] = row['definition']
    return definitions

def cluster_values_with_gemini(property_name: str, values: List[str], definition: str, model_name: str) -> Dict[str, str]:
    """
    Uses Gemini to cluster a list of property values into canonical categories.
    Returns a dictionary mapping original value -> canonical category.
    """
    logging.info(f"Clustering {len(values)} values for property: {property_name}")

    if not values:
        return {}

    # Configure Gemini
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(model_name)

    prompt = f"""
    You are an expert materials scientist assisting with data standardization.
    
    Task: Group the following raw values for the property '{property_name}' into canonical categories.
    Definition of property: {definition}
    
    The raw values contain specific experimental details (temperature, field strength, criteria) which are scientifically important.
    
    GOAL: standardize the Phrasing of these details while preserving the physical Meaning.
    
    Rules for Canonicalization:
    1.  **Preserve Physics**: distinctive conditions like "H//c" (field parallel to c-axis) or "0.5Rn" (resistivity criterion) MUST be kept.
    2.  **Normalize Phrasing**: multiple ways of writing the same thing should map to ONE standard string.
        *   "H//c", "H parallel c", "H || c", "field along c" -> "H//c"
        *   "0.5Rn", "50% Rn", "R=0.5Rn", "midpoint" -> "Resistivity (0.5Rn)"
        *   "onset", "R_onset", "onset of transition" -> "Resistivity (onset)"
        *   "zero resistance", "R=0", "zero offset" -> "Resistivity (zero)"
        *   "M-H", "M vs H", "Magnetization curve" -> "Magnetization"
    3.  **Generalize where appropriate**: If no specific parameters are given, map to the broad method.
    
    Output Format:
    Return a VALID JSON object where keys are the raw values and values are the canonical category.
    Map unreadable/garbage values to "Unknown".
    
    Input Values:
    {json.dumps(values)}
    
    Output JSON:
    """

    try:
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        result = json.loads(response.text)
        return result
    except Exception as e:
        logging.error(f"Error clustering {property_name}: {e}")
        return {v: v for v in values}

def extract_keys_from_clusters(clusters_file: str, output_file: str):
    """
    Extracts unique canonical categories from the generated clusters JSON.
    """
    if not os.path.exists(clusters_file):
        logging.error(f"Clusters file not found at {clusters_file}")
        return

    with open(clusters_file, 'r') as f:
        data = json.load(f)

    unique_keys = {}
    for property_name, mapping in data.items():
        # Get all values (canonical names), remove duplicates, and sort
        canonical_values = sorted(list(set(mapping.values())))
        unique_keys[property_name] = {
            "count": len(canonical_values),
            "categories": canonical_values
        }
    
    with open(output_file, 'w') as f:
        json.dump(unique_keys, f, indent=2)
    
    logging.info(f"Unique cluster keys saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Cluster unique property values using Gemini LLM.")
    parser.add_argument(
        "--unique_values",
        default=DEFAULT_UNIQUE_VALUES_CSV,
        help="Path to the CSV file containing unique property values."
    )
    parser.add_argument(
        "--glossary",
        default=DEFAULT_GLOSSARY_FILE,
        help="Path to the CSV file containing property definitions (glossary)."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model name to use for clustering (default: {DEFAULT_MODEL})."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save the output JSON. If not specified, uses property_clusters_<MODEL_NAME>.json"
    )
    parser.add_argument(
        "--output_keys",
        default=None,
        help="Path to save the cluster keys summary JSON. If not specified, uses property_cluster_keys_<MODEL_NAME>.json"
    )

    args = parser.parse_args()

    unique_values_csv = args.unique_values
    glossary_file = args.glossary
    model_name = args.model

    # Generate output filenames based on model name if not provided
    output_file = args.output or os.path.join(DEFAULT_DATA_DIR, f"property_clusters_{model_name}.json")
    output_keys_file = args.output_keys or os.path.join(DEFAULT_DATA_DIR, f"property_cluster_keys_{model_name}.json")

    if not os.path.exists(unique_values_csv):
        raise FileNotFoundError(f"Unique values CSV not found at {unique_values_csv}")

    # Load data
    logging.info(f"Loading data from {unique_values_csv}")
    df_values = pd.read_csv(unique_values_csv)
    
    # Load definitions
    logging.info(f"Loading definitions from {glossary_file}")
    definitions = load_glossary_definitions(glossary_file)

    all_clusters = {}

    for prop_name in TARGET_PROPERTIES:
        if prop_name not in df_values.columns:
            logging.warning(f"Column '{prop_name}' not found in dataset. Skipping.")
            continue
            
        # Extract unique non-null values
        unique_vals = df_values[prop_name].dropna().unique().tolist()
        unique_vals = [str(v).strip() for v in unique_vals if str(v).strip()]
        
        if not unique_vals:
            logging.warning(f"No values found for '{prop_name}'. Skipping.")
            continue
            
        definition = definitions.get(prop_name, "No definition available.")
        if definition == "No definition available.":
            logging.warning(f"Definition for '{prop_name}' not found in glossary.")
        
        # Cluster
        clusters = cluster_values_with_gemini(prop_name, unique_vals, definition, model_name)
        all_clusters[prop_name] = clusters
        
        # Rate limiting (for Gemini)
        time.sleep(1) 

    # Save to JSON
    logging.info(f"Saving clusters to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(all_clusters, f, indent=2)

    logging.info("Clustering complete.")
    
    # Extract keys
    extract_keys_from_clusters(output_file, output_keys_file)

if __name__ == "__main__":
    main()
