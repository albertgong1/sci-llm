"""Helper functions"""

from pymatgen.core import Composition


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
    """Check if pred is a valid pymatgen composition and is close to answer."""
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
        >>> is_close_si_unit(100.0, 100.05)
        True
        >>> is_close_si_unit(100.0, 100.2)
        False
        >>> is_close_si_unit(0.0, 0.0)
        True

    """
    if answer == 0:
        return pred == 0
    return abs(pred - answer) / abs(answer) <= rel_tol


def scorer_categorical(pred: str, answer: str, mapping: dict[str, str] | None = None) -> bool:
    """
    Scores categorical ("method of X") properties.
    If a mapping is provided, normalizes both pred and answer to their canonical categories.
    Returns True if:
    1. Exact match (after normalization)
    2. Substring match (one canonical category contains the other)
    """
    pred_str = str(pred).strip()
    answer_str = str(answer).strip()

    if mapping:
        # Normalize to canonical category if available in the map
        pred_norm = mapping.get(pred_str, pred_str)
        answer_norm = mapping.get(answer_str, answer_str)
    else:
        # If not in map, keep original string
        pred_norm = pred_str
        answer_norm = answer_str

    # 1. Exact Match
    if pred_norm == answer_norm:
        return True

    # 2. Relaxed Substring Match (Case-insensitive)
    p_lower = pred_norm.lower()
    a_lower = answer_norm.lower()

    if p_lower in a_lower or a_lower in p_lower:
        return True

    return False
