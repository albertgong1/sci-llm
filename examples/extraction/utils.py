"""Helper functions"""

import json
import re
from pathlib import Path
from pymatgen.core import Composition

# Load normalized space groups
try:
    # Assuming this file is in the same directory as this script (examples/extraction)
    # and assets is a subdirectory (examples/extraction/assets)
    ASSETS_DIR = Path(__file__).parent / "assets"
    SPACE_GROUPS_PATH = ASSETS_DIR / "space_groups_normalized.json"
    
    with open(SPACE_GROUPS_PATH, "r") as f:
        SPACE_GROUPS = json.load(f)
except Exception as e:
    print(f"Warning: Could not load space_groups_normalized.json: {e}")
    SPACE_GROUPS = {}


def normalize_formula(formula: str) -> str:
    """Normalize a chemical formula to alphabetical element order.

    Examples:
        >>> normalize_formula("BiPtPb3")
        'BiPb3Pt'
        >>> normalize_formula("PtBiPb3")
        'BiPb3Pt'
    """
    comp = Composition(formula)
    return comp.alphabetical_formula

def scorer_pymatgen(pred: str, answer: str) -> bool:
    """Check if pred is a valid pymatgen composition and is close to answer.
    """
    try:
        pred_comp = Composition(pred)
        answer_comp = Composition(answer)
        return pred_comp.almost_equals(answer_comp)
    except Exception:
        return False

def scorer_si(pred: float, answer: float, rel_tol: float = 0.001) -> bool:
    """Check if pred is within 0.1% of answer.

    Args:
        pred: The predicted value.
        answer: The reference/ground truth value.
        rel_tol: Relative tolerance (default 0.001 = 0.1%).

    Returns:
        True if pred is within rel_tol of answer.

    Examples:
        >>> scorer_si(100.0, 100.05)
        True
        >>> scorer_si(100.0, 100.2)
        False
        >>> scorer_si(0.0, 0.0)
        True
    """
    if answer == 0:
        return pred == 0
    return abs(pred - answer) / abs(answer) <= rel_tol

def scorer_categorical(pred: str, answer: str) -> bool:
    """Check if pred is a valid categorical answer and is close to answer.
    """
    return False

def scorer_space_group(pred: str, answer: str) -> bool:
    """
    Score space group predictions.
    The space group alphabet is {letters, numbers, /, -}.
    
    1. Clean input (keep only {letters, numbers, /, -} and lowercase).
    2. Map to ID.
    3. Compare IDs.
    """
    if not SPACE_GROUPS:
        return False
        
    def get_norm_and_id(val):
        if not isinstance(val, str):
            val = str(val)
        cleaned = re.sub(r"[^a-zA-Z0-9/\-]", "", val)
        norm = cleaned.lower()
        return norm, SPACE_GROUPS.get(norm)

    pred_norm, pred_id = get_norm_and_id(pred)
    answer_norm, answer_id = get_norm_and_id(answer)
    
    # Adding these two checks in case there's some alias we missed or haven't heard of
    if pred_id is None:
        print(f"Warning: Predicted space group '{pred}' (clean: '{pred_norm}') not found in allowed keys.")
        return False
        
    if answer_id is None:
        print(f"Warning: Answer space group '{answer}' (clean: '{answer_norm}') not found in allowed keys.")
        return False
        
    return pred_id == answer_id