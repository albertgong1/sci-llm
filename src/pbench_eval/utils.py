"""Helper functions"""

import re
from pymatgen.core import Composition
import logging
import pandas as pd

from .space_groups_normalized import (
    SPACE_GROUPS,
)

logger = logging.getLogger(__name__)


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


def parse_numeric_candidates(value: str) -> list[tuple[float, str | None]]:
    """Extract numeric candidates with units from a free-form value string.

    Returns:
        List of (value, unit) tuples. Unit is None if no unit found after the number.

    Examples:
        >>> parse_numeric_candidates("0.6570 nm")
        [(0.657, 'nm')]
        >>> parse_numeric_candidates("1.2 x 10^3 K")
        [(1200.0, 'K')]
        >>> parse_numeric_candidates("12.34")
        [(12.34, None)]

    """
    if value is None:
        return []

    value_str = normalize_unicode(str(value)).strip()
    if value_str.upper() == "NOT_FOUND" or value_str == "":
        return []

    value_str = re.sub(r"\(\d+\)", "", value_str)

    candidates: list[tuple[float, str | None, int]] = []  # (value, unit, end_position)

    # Scientific notation: 1.2 x 10^3 with optional unit
    sci_pattern = re.compile(
        r"(?P<base>[-+]?\d*\.?\d+)\s*(?:x|×)\s*10(?:\s*\^)?\s*(?P<exp>[-+]?\d+)\s*(?P<unit>[a-zA-Z/°%]+)?",
        re.IGNORECASE,
    )
    for match in sci_pattern.finditer(value_str):
        try:
            base = float(match.group("base"))
            exp = int(match.group("exp"))
            unit = match.group("unit")
            candidates.append((base * (10**exp), unit, match.end()))
        except Exception:
            continue

    # Standard float/int with optional unit: 12.34 nm, .5 K, 1e5 Pa
    num_pattern = re.compile(
        r"(?P<num>[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)\s*(?P<unit>[a-zA-Z/°%]+)?"
    )
    for match in num_pattern.finditer(value_str):
        try:
            num = float(match.group("num"))
            unit = match.group("unit")
            candidates.append((num, unit, match.end()))
        except Exception:
            continue

    # Deduplicate preserving order, using position to avoid duplicates from overlapping patterns
    seen: set[str] = set()
    unique: list[tuple[float, str | None]] = []
    for num, unit, end_pos in candidates:
        # Create key from both number and position to handle same number with different units
        key = f"{num:.12g}_{unit}_{end_pos}"
        if key in seen:
            continue
        seen.add(key)
        unique.append((num, unit))

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


def scorer_si(
    pred_num: float,
    pred_unit: str | None,
    answer_num: float,
    answer_unit: str | None,
    rel_tol: float = 0.001,
    conversion_df: pd.DataFrame | None = None,
) -> bool:
    """Check if pred is within 0.1% of answer.

    Args:
        pred_num: The predicted numerical value.
        pred_unit: The predicted unit (or None if no unit).
        answer_num: The reference/ground truth numerical value.
        answer_unit: The reference/ground truth unit (or None if no unit).
        rel_tol: Relative tolerance (default 0.001 = 0.1%).
        conversion_df: DataFrame for unit conversion with columns:
            - property_unit: unit name (used as index)
            - conversion_factor: factor to convert to SI units
            - comments: notes about the conversion

    Returns:
        True if pred is within rel_tol of answer.

    Examples:
        >>> scorer_si(100.0, "K", 100.05, "K")
        True
        >>> scorer_si(100.0, "K", 100.2, "K")
        False
        >>> scorer_si(0.0, None, 0.0, None)
        True

    """
    logger.debug(
        f"Scoring SI: pred={pred_num} {pred_unit}, answer={answer_num} {answer_unit}, rel_tol={rel_tol}"
    )
    # Normalize units for comparison (strip whitespace)
    pred_unit_norm = pred_unit.strip() if pred_unit else None
    answer_unit_norm = answer_unit.strip() if answer_unit else None

    # If units are the same (after normalization), no conversion needed
    if pred_unit_norm == answer_unit_norm:
        if answer_num == 0:
            return pred_num == 0
        return abs(pred_num - answer_num) / abs(answer_num) <= rel_tol

    # Units are different - attempt conversion if conversion_df is provided
    if conversion_df is not None and pred_unit_norm and answer_unit_norm:
        # Set index to property_unit for easy lookup if not already indexed
        if "property_unit" in conversion_df.columns:
            conversion_lookup = conversion_df.set_index("property_unit")
        else:
            conversion_lookup = conversion_df

        # Get conversion factors for both units
        pred_factor = None
        answer_factor = None
        pred_comment = None
        answer_comment = None

        try:
            if pred_unit_norm in conversion_lookup.index:
                pred_factor = conversion_lookup.loc[pred_unit_norm, "conversion_factor"]
                if "comments" in conversion_lookup.columns:
                    pred_comment = conversion_lookup.loc[pred_unit_norm, "comments"]
        except Exception:
            pass

        try:
            if answer_unit_norm in conversion_lookup.index:
                answer_factor = conversion_lookup.loc[
                    answer_unit_norm, "conversion_factor"
                ]
                if "comments" in conversion_lookup.columns:
                    answer_comment = conversion_lookup.loc[answer_unit_norm, "comments"]
        except Exception:
            pass

        # Check if both conversion factors are non-NaN
        if pd.notna(pred_factor) and pd.notna(answer_factor):
            # Convert both values to SI units
            pred_si = pred_num * float(pred_factor)
            answer_si = answer_num * float(answer_factor)

            # Apply scoring rule on SI values
            if answer_si == 0:
                return pred_si == 0
            return abs(pred_si - answer_si) / abs(answer_si) <= rel_tol

        # If one or both conversion factors are NaN, log warning and use usual scoring
        if pd.isna(pred_factor) and pred_unit_norm in conversion_lookup.index:
            comment = (
                pred_comment
                if pd.notna(pred_comment)
                else "No conversion factor available"
            )
            logger.warning(f"Cannot convert unit '{pred_unit_norm}' to SI: {comment}")

        if pd.isna(answer_factor) and answer_unit_norm in conversion_lookup.index:
            comment = (
                answer_comment
                if pd.notna(answer_comment)
                else "No conversion factor available"
            )
            logger.warning(f"Cannot convert unit '{answer_unit_norm}' to SI: {comment}")

    # Fallback: use usual scoring rule without unit conversion
    if answer_num == 0:
        return pred_num == 0
    return abs(pred_num - answer_num) / abs(answer_num) <= rel_tol


def scorer_space_group(pred: str, answer: str) -> bool:
    """Score space group predictions.
    The space group alphabet is {letters, numbers, /, -}.

    1. Clean input (keep only {letters, numbers, /, -} and lowercase).
    2. Map to ID.
    3. Compare IDs.

    Args:
        pred: Predicted space group string.
        answer: Ground truth space group string.

    Returns:
        True if IDs match, False otherwise.

    """

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


def scorer_exact_match(
    pred: str, answer: str, mapping: dict[str, str] | None = None
) -> bool:
    """Scores categorical ("method of X") properties.
    If a mapping is provided, normalizes both pred and answer to their canonical categories.
    Returns True if:
    1. Exact match (after normalization)
    2. Substring match (one canonical category contains the other)
    """
    assert isinstance(pred, str), "pred must be a string"
    assert isinstance(answer, str), "answer must be a string"

    pred_str = pred.strip()
    answer_str = answer.strip()

    # 1. Exact Match
    if pred_str == answer_str:
        return True

    # 2. Relaxed Substring Match (Case-insensitive)
    p_lower = pred_str.lower()
    a_lower = answer_str.lower()

    if p_lower in a_lower or a_lower in p_lower:
        return True

    return False


def score_value(
    pred_value: str,
    answer_value: str,
    rubric: str,
    mapping: dict[str, str] | None = None,  # not used
    conversion_df: pd.DataFrame | None = None,
) -> float:
    """Master scoring function (0.0 to 1.0).

    Args:
        pred_value: The predicted string.
        answer_value: The ground truth string.
        rubric: "0.1% SI", "pymatgen", or "categorical".
        mapping: Optional mapping for categorical scoring.
        conversion_df: Optional DataFrame for unit conversion.

    """
    assert isinstance(pred_value, str), "pred_value must be a string"
    assert isinstance(answer_value, str), "answer_value must be a string"
    assert isinstance(rubric, str), "rubric must be a string"

    logger.debug(
        f"Scoring pred_value='{pred_value}' vs answer_value='{answer_value}' using rubric='{rubric}'"
    )
    match rubric:
        case "0.1% SI":
            # Check if ANY predicted candidate matches the first answer candidate
            answer_nums = parse_numeric_candidates(
                answer_value
            )  # return (number, units) tuples
            if not answer_nums:
                return 0.0
            # Strict: The ground truth should be unambiguous, so we take the first number found.
            if len(answer_nums) > 1:
                logger.warning(
                    f"Multiple numeric candidates found in answer_value '{answer_value}'. Using the first one: {answer_nums[0][0]}"
                )
            answer_num, answer_unit = answer_nums[0]
            for pred_num, pred_unit in parse_numeric_candidates(pred_value):
                # TODO: implement unit conversion/checking
                if scorer_si(
                    pred_num,
                    pred_unit,
                    answer_num,
                    answer_unit,
                    conversion_df=conversion_df,
                ):
                    return 1.0
            return 0.0

        case "pymatgen":
            # Clean inputs before pymatgen parsing if needed?
            # For now, just pass raw strings as scorer_pymatgen handles robust Composition checks?
            # Actually scorer_pymatgen is basic. Let's make it robust against raw inputs by normalizing unicode.
            pv = normalize_unicode(pred_value).strip()
            av = normalize_unicode(answer_value).strip()
            return 1.0 if scorer_pymatgen(pv, av) else 0.0

        case "space_group":
            return 1.0 if scorer_space_group(pred_value, answer_value) else 0.0

        case _:
            # Default to exact match
            return 1.0 if scorer_exact_match(pred_value, answer_value) else 0.0
