"""Helper functions"""

import re
from pathlib import Path
from pymatgen.core import Composition

# Load normalized space groups
import json

# Assuming this file is in the same directory as this script (examples/extraction)
# and assets is a subdirectory (examples/extraction/assets)
ASSETS_DIR = Path(__file__).parent.parent.parent / "assets"
SPACE_GROUPS_PATH = ASSETS_DIR / "hard" / "space_groups_normalized.json"

with open(SPACE_GROUPS_PATH, "r") as f:
    SPACE_GROUPS = json.load(f)


_SUPERSCRIPT_MAP = str.maketrans(
    {
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "⁺": "+",
        "⁻": "-",
    }
)


def normalize_ws(text: str) -> str:
    """Collapse whitespace for more robust string matching."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_unicode(text: str) -> str:
    """Normalize common unicode variants (dashes, superscripts, delta)."""
    s = str(text or "")
    s = (
        s.replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
    )
    s = s.replace("δ", "d").replace("Δ", "d")
    s = s.translate(_SUPERSCRIPT_MAP)
    return s


def parse_numeric_candidates(value: str) -> list[float]:
    """Extract numeric candidates from a free-form value string (units allowed)."""
    if value is None:
        return []

    value_str = normalize_unicode(str(value)).strip()
    if value_str.upper() == "NOT_FOUND" or value_str == "":
        return []

    value_str = re.sub(r"\(\d+\)", "", value_str)

    candidates: list[float] = []

    # Scientific notation: 1.2 x 10^3
    sci_pattern = re.compile(
        r"(?P<base>[-+]?\d*\.?\d+)\s*(?:x|×)\s*10(?:\s*\^)?\s*(?P<exp>[-+]?\d+)",
        re.IGNORECASE,
    )
    for match in sci_pattern.finditer(value_str):
        try:
            base = float(match.group("base"))
            exp = int(match.group("exp"))
            candidates.append(base * (10**exp))
        except Exception:
            continue

    # Standard float/int: 12.34, .5, 1e5
    num_pattern = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
    for match in num_pattern.finditer(value_str):
        try:
            candidates.append(float(match.group(0)))
        except Exception:
            continue

    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[float] = []
    for num in candidates:
        key = f"{num:.12g}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(num)

    return unique


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


def scorer_space_group(pred: str, answer: str) -> bool:
    """Score space group predictions.
    The space group alphabet is {letters, numbers, /, -}.

    1. Clean input (keep only {letters, numbers, /, -} and lowercase).
    2. Map to ID.
    3. Compare IDs.
    """
    if not SPACE_GROUPS:
        return False

    def get_norm_and_id(val: str) -> tuple[str, str | None]:
        if not isinstance(val, str):
            val = str(val)
        cleaned = re.sub(r"[^a-zA-Z0-9/\-]", "", val)
        norm = cleaned.lower()
        return norm, SPACE_GROUPS.get(norm)

    pred_norm, pred_id = get_norm_and_id(pred)
    answer_norm, answer_id = get_norm_and_id(answer)

    # Adding these two checks in case there's some alias we missed or haven't heard of
    if pred_id is None:
        print(
            f"Warning: Predicted space group '{pred}' (clean: '{pred_norm}') not found in allowed keys."
        )
        return False

    if answer_id is None:
        print(
            f"Warning: Answer space group '{answer}' (clean: '{answer_norm}') not found in allowed keys."
        )
        return False

    return pred_id == answer_id


def scorer_categorical(
    pred: str, answer: str, mapping: dict[str, str] | None = None
) -> bool:
    """Scores categorical ("method of X") properties.
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


def score_value(
    pred_value: str,
    answer_value: str,
    rubric: str,
    mapping: dict[str, str] | None = None,
) -> float:
    """Master scoring function (0.0 to 1.0).

    Args:
        pred_value: The predicted string.
        answer_value: The ground truth string.
        rubric: "0.1% SI", "pymatgen", or "categorical".
        mapping: Optional mapping for categorical clustering.

    """
    if rubric == "0.1% SI":
        # Check if ANY predicted candidate matches the first answer candidate
        answer_nums = parse_numeric_candidates(answer_value)
        if not answer_nums:
            return 0.0
        # Strict: The ground truth should be unambiguous, so we take the first number found.
        answer_num = answer_nums[0]
        for num in parse_numeric_candidates(pred_value):
            if scorer_si(num, answer_num):
                return 1.0
        return 0.0

    elif rubric == "pymatgen":
        # Clean inputs before pymatgen parsing if needed?
        # For now, just pass raw strings as scorer_pymatgen handles robust Composition checks?
        # Actually scorer_pymatgen is basic. Let's make it robust against raw inputs by normalizing unicode.
        pv = normalize_unicode(pred_value).strip()
        av = normalize_unicode(answer_value).strip()
        return 1.0 if scorer_pymatgen(pv, av) else 0.0

    else:
        # Default to categorical
        return (
            1.0
            if scorer_categorical(pred_value, answer_value, mapping=mapping)
            else 0.0
        )
