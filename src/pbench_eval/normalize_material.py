import re
from pymatgen.core import Composition


def strip_formula(text: str) -> tuple[str, dict[str, float]]:
    """Extract chemical formula and variable values from text.

    Strips trailing non-formula text like "thin films", "single crystal", etc.
    Also extracts variable assignments like "(x=0.15)" or "(x = 0.15, y = 0.2)".

    Args:
        text: String that may contain a formula plus extra text.
              e.g., "YBa2Cu3O7 thin films", "MgB2 single crystal",
              "La2-xSrxCuO4 (x=0.15)"

    Returns:
        Tuple of (formula_string, variable_values_dict).

    Examples:
        >>> strip_formula("YBa2Cu3O7 thin films")
        ('YBa2Cu3O7', {})
        >>> strip_formula("MgB2 single crystal")
        ('MgB2', {})
        >>> strip_formula("La2-xSrxCuO4 (x=0.15)")
        ('La2-xSrxCuO4', {'x': 0.15})
        >>> strip_formula("Bi2Sr2CaCu2O8+δ (δ = 0.2)")
        ('Bi2Sr2CaCu2O8+δ', {'δ': 0.2})
        >>> strip_formula("YBa2Cu3O7-δ (δ=0.1, z=0.5)")
        ('YBa2Cu3O7-δ', {'δ': 0.1, 'z': 0.5})
        >>> strip_formula("YBa2Cu3 O7−δ thin films")
        ('YBa2Cu3O7-δ', {})

    """
    text = text.strip()

    # Normalize unicode characters:
    # - U+2212 (MINUS SIGN) → U+002D (HYPHEN-MINUS)
    # - U+2013 (EN DASH) → U+002D (HYPHEN-MINUS)
    # - U+2014 (EM DASH) → U+002D (HYPHEN-MINUS)
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")

    # Remove spaces within formula (e.g., "YBa2Cu3 O7" -> "YBa2Cu3O7")
    # Only remove space if it's between formula characters (not before descriptive text)
    # Pattern: space between (letter/number/symbol) and (uppercase letter or O followed by number/variable)
    text = re.sub(r"(\d|\))\s+([A-Z])", r"\1\2", text)

    variable_values = {}

    # Extract variable assignments like (x=0.15) or (x = 0.15, δ = 0.2)
    var_pattern = (
        r"\(\s*([xyzδ])\s*=\s*([0-9.]+)\s*(?:,\s*([xyzδ])\s*=\s*([0-9.]+)\s*)*\)"
    )
    var_match = re.search(var_pattern, text)
    if var_match:
        # Extract all variable assignments from the matched group
        var_str = var_match.group(0)
        # Find all var=value pairs
        pairs = re.findall(r"([xyzδ])\s*=\s*([0-9.]+)", var_str)
        for var, val in pairs:
            variable_values[var] = float(val)
        # Remove the variable assignment from text for formula extraction
        text = text[: var_match.start()] + text[var_match.end() :]
        text = text.strip()

    # Pattern for chemical formula: starts with element, may contain numbers,
    # parentheses, variables (x, y, z, δ), operators (+, -), and subscripts
    # Stop at first space followed by lowercase word (likely descriptive text)
    formula_pattern = r"^([A-Z][a-zA-Z0-9()\-+δ.,/]*)"

    match = re.match(formula_pattern, text)
    if match:
        return match.group(1).rstrip(".,"), variable_values

    return text, variable_values


def classify_and_normalize(
    formula: str, variable_values: dict | None = None
) -> tuple[str, str, list[str]]:
    """Classify formula type and normalize appropriately.

    This function handles generic/template chemical formulas that contain variables
    or placeholders and converts them into stoichiometric formulas that can be
    parsed by pymatgen. It distinguishes between:

    1. **Variables**: Placeholders like x, z, δ in oxygen positions (e.g., "Ox", "O(7-z)")
       - These represent unknown/variable oxygen content
       - Normalized using default values (typically ideal/fully oxygenated state)

    2. **Parameters**: Compositional variables that change material properties (e.g., "La(2-x)Srx")
       - These cannot be normalized without explicit values
       - Require user to provide specific values

    3. **Mixed sites**: Notation like "(Sr,Ca)14" indicating element mixing
       - Normalized by equal distribution if no amounts specified

    Args:
        formula (str): Generic chemical formula string, which may contain:
            - Oxygen variables: "Ox", "OX", "O(7-z)", "O(8+δ)"
            - Mixed sites: "(Sr,Ca)14", "(A,B,C)n"
            - Parameters: "La(2-x)Srx", expressions with compositional variables

        variable_values (dict, optional): Dictionary mapping variable names to specific values.
            Common variables:
            - 'x': Compositional parameter (default: 0 for undoped state)
            - 'z': Oxygen deficiency (default: 0 for fully oxygenated)
            - 'δ' or 'delta': Oxygen excess/deficiency (default: 0)
            - 'X': Direct oxygen content for "OX" notation (default: 7 for YBCO-type)

    Examples:
                {'x': 0.15}  # For La(2-x)Srx with 15% Sr doping
                {'z': 0.5}   # For O(7-z) with oxygen deficiency of 0.5
                {'δ': 0.2}   # For O(8+δ) with oxygen excess of 0.2

    Returns:
        tuple: (normalized_formula, formula_type, notes)
            - normalized_formula (str): Stoichiometric formula (may still contain
              unresolved parameters if values not provided)
            - formula_type (str): One of:
                - "STOICHIOMETRIC": Fully normalized, parseable by pymatgen
                - "PARAMETER_FORMULA": Contains compositional parameters, needs values
                - "PARTIAL_NORMALIZATION": Some normalization done, issues remain
                - "INVALID": Normalization failed, formula is invalid
            - notes (list): List of strings describing normalization steps and warnings

    Examples:
        >>> # Example 1: Oxygen variable (can normalize with defaults)
        >>> formula = "Er1Ba2Cu3Ox"
        >>> normalized, ftype, notes = classify_and_normalize(formula)
        >>> print(normalized)
        'Er1Ba2Cu3O7'
        >>> print(ftype)
        'STOICHIOMETRIC'
        >>> print(notes)
        ['Oxygen variable normalized with defaults: {...}']
        >>> # Can parse with pymatgen:
        >>> from pymatgen.core import Composition
        >>> comp = Composition(normalized)
        >>> print(comp.reduced_formula)
        'ErBa2Cu3O7'

        >>> # Example 2: Oxygen deficiency notation
        >>> formula = "Y1Ba2Cu3O(7-z)"
        >>> normalized, ftype, notes = classify_and_normalize(formula)
        >>> print(normalized)
        'Y1Ba2Cu3O7.0'
        >>> print(ftype)
        'STOICHIOMETRIC'
        >>> # With z=0 (default), oxygen content is 7.0

        >>> # Example 3: Oxygen deficiency with custom value
        >>> formula = "Y1Ba2Cu3O(7-z)"
        >>> normalized, ftype, notes = classify_and_normalize(formula, {'z': 0.5})
        >>> print(normalized)
        'Y1Ba2Cu3O6.5'
        >>> print(ftype)
        'STOICHIOMETRIC'
        >>> # With z=0.5, oxygen content is 6.5

        >>> # Example 4: Mixed site notation
        >>> formula = "(Sr,Ca)14Cu24O41"
        >>> normalized, ftype, notes = classify_and_normalize(formula)
        >>> print(normalized)
        'Sr7.0Ca7.0Cu24O41'
        >>> print(ftype)
        'STOICHIOMETRIC'
        >>> print(notes)
        ['Mixed site normalized with equal distribution']
        >>> # Equal distribution: 14 total → 7 Sr + 7 Ca

        >>> # Example 5: Compositional parameter (CANNOT normalize without value)
        >>> formula = "La(2-x)SrxCuO4"
        >>> normalized, ftype, notes = classify_and_normalize(formula)
        >>> print(normalized)
        'La(2-x)SrxCuO4'
        >>> print(ftype)
        'PARAMETER_FORMULA'
        >>> print(notes)
        ['WARNING: Contains compositional parameters - cannot fully normalize without specific x value']
        >>> # This is NOT parseable by pymatgen without providing x!

        >>> # Example 6: Compositional parameter WITH value
        >>> formula = "La(2-x)SrxCuO4"
        >>> normalized, ftype, notes = classify_and_normalize(formula, {'x': 0.15})
        >>> print(normalized)
        'La1.85Sr0.15CuO4'
        >>> print(ftype)
        'STOICHIOMETRIC'
        >>> # Now it's parseable!
        >>> comp = Composition(normalized)
        >>> print(comp.reduced_formula)
        'La1.85Sr0.15CuO4'

        >>> # Example 7: Oxygen excess notation
        >>> formula = "Bi2Sr2CaCu2O(8+δ)"
        >>> normalized, ftype, notes = classify_and_normalize(formula)
        >>> print(normalized)
        'Bi2Sr2CaCu2O8.0'
        >>> print(ftype)
        'STOICHIOMETRIC'
        >>> # With δ=0 (default), oxygen content is 8.0

        >>> # Example 8: Oxygen excess with custom value
        >>> formula = "Bi2Sr2CaCu2O(8+δ)"
        >>> normalized, ftype, notes = classify_and_normalize(formula, {'δ': 0.2})
        >>> print(normalized)
        'Bi2Sr2CaCu2O8.2'
        >>> print(ftype)
        'STOICHIOMETRIC'
        >>> # With δ=0.2, oxygen content is 8.2

        >>> # Example 9: Multiple variables (oxygen + mixed site)
        >>> formula = "(Sr,Ca)14Cu24O(41-z)"
        >>> normalized, ftype, notes = classify_and_normalize(formula, {'z': 1})
        >>> print(normalized)
        'Sr7.0Ca7.0Cu24O40.0'
        >>> print(ftype)
        'STOICHIOMETRIC'
        >>> print(notes)
        ['Oxygen variable normalized with defaults: {...}',
         'Mixed site normalized with equal distribution']

        >>> # Example 10: Complex case - multiple elements in mixed site
        >>> formula = "(Y,Gd,Er)1Ba2Cu3O7"
        >>> normalized, ftype, notes = classify_and_normalize(formula)
        >>> print(normalized)
        'Y0.333...Gd0.333...Er0.333...Ba2Cu3O7'
        >>> print(ftype)
        'STOICHIOMETRIC'
        >>> # Three elements share position 1: 1/3 each

    Notes:
        - **Oxygen variables** (Ox, O(7-z), etc.) are ALWAYS normalized using defaults
          or provided values. Default for 'X' is 7 (appropriate for YBCO-type cuprates).

        - **Mixed sites** (Sr,Ca)n are normalized by equal distribution by default.
          Currently does not support specifying custom ratios.

        - **Compositional parameters** like x in "La(2-x)Srx" fundamentally change
          the material and CANNOT be normalized without explicit values.

        - The function uses these default values if not provided:
            {'x': 0, 'y': 0, 'z': 0, 'δ': 0, 'delta': 0, 'X': 7}
          Note: X=7 is specific to YBCO-type compounds. Other materials may need
          different defaults (e.g., La2CuO4 would want O4, TiO2 would want O2).

        - For reliable results, always provide explicit variable values when working
          with parameter formulas or non-YBCO materials.

    See Also:
        - smart_normalize_formula(): Simpler function for basic normalization
        - get_default_oxygen_content(): Get context-appropriate oxygen defaults
        - pymatgen.core.Composition: For parsing stoichiometric formulas

    Warnings:
        - Formula normalization is **context-dependent**. The default X=7 is only
          appropriate for YBCO-type (ReBa2Cu3O7) cuprate superconductors.

        - Not all formulas can be normalized! Formulas with compositional parameters
          like "La(2-x)Srx" REQUIRE explicit values for x.

        - Always verify that normalized formulas are chemically reasonable before
          using them in calculations.

    """
    if variable_values is None:
        variable_values = {}

    # Separate defaults for oxygen variables vs compositional parameters
    # Oxygen variables (in O position) default to typical values
    oxygen_defaults = {"x": 1, "X": 1, "z": 0, "δ": 0, "delta": 0}
    # Compositional parameters default to 0 (undoped state)
    param_defaults = {"x": 0, "y": 0, "z": 0, "δ": 0, "delta": 0}

    # User-provided values override both
    oxygen_defaults.update(variable_values)
    param_defaults.update(variable_values)

    # Check formula type
    # Match: Ox, OX, Oδ, O(7-z), O(8+δ), O7-z, O8+δ (with or without parentheses)
    has_oxygen_variable = bool(
        re.search(r"O[xXzδ]|O\([^)]+\)|O\d+[+\-][xyzδ]", formula)
    )
    # Match: Element(expr) like La(2-x), or Elementx like Srx, or Element<num>[+-]<var> like Ba1-x
    # Exclude oxygen patterns (O is handled separately)
    has_parameter = bool(
        re.search(
            r"[A-Z][a-z]?\([^)]*[xyz][^)]*\)[A-Z]|"  # La(2-x)Sr pattern
            r"[A-Z][a-z]?[xyz][A-Z]|"  # SrxCu pattern
            r"[A-BD-NP-Z][a-z]?\d+[+\-][xyz](?=[A-Z]|$)",  # Ba1-x pattern (exclude O)
            formula,
        )
    )
    has_mixed_site = bool(re.search(r"\([A-Z][a-z]?,", formula))

    normalized = formula
    notes = []

    # Handle oxygen variables (always safe to normalize)
    if has_oxygen_variable:
        # "Ox" or "OX" type - use oxygen_defaults (x defaults to 1)
        normalized = re.sub(
            r"O([xXzδ])\b",
            lambda m: f"O{oxygen_defaults.get(m.group(1), 1)}",
            normalized,
        )

        # Helper to evaluate oxygen expressions
        def eval_oxygen_expr(expr: str) -> str | None:
            for var, val in oxygen_defaults.items():
                expr = re.sub(rf"\b{re.escape(var)}\b", str(val), expr)
            try:
                return str(eval(expr))
            except Exception:
                return None

        # "O(7-z)" type - with parentheses
        def eval_oxygen_paren(match: re.Match) -> str | None:
            expr = match.group(1)
            result = eval_oxygen_expr(expr)
            return f"O{result}" if result else match.group(0)

        normalized = re.sub(r"O\(([^)]+)\)", eval_oxygen_paren, normalized)

        # "O7-z" or "O8+δ" type - without parentheses (number followed by +/- and variable)
        def eval_oxygen_no_paren(match: re.Match) -> str | None:
            expr = match.group(1)  # e.g., "7-z" or "8+δ"
            result = eval_oxygen_expr(expr)
            return f"O{result}" if result else match.group(0)

        normalized = re.sub(r"O(\d+[+\-][xyzδ])", eval_oxygen_no_paren, normalized)

        notes.append(f"Oxygen variable normalized with defaults: {oxygen_defaults}")

    # Handle mixed sites (safe to normalize with equal distribution)
    if has_mixed_site:

        def split_mixed_site(match: re.Match) -> str:
            elements_str = match.group(1)
            total = float(match.group(2))
            elements = [e.strip() for e in elements_str.split(",")]
            per_element = total / len(elements)
            return "".join([f"{elem}{per_element}" for elem in elements])

        normalized = re.sub(
            r"\(([A-Z][a-z]?(?:\s*,\s*[A-Z][a-z]?)+)\)(\d+(?:\.\d+)?)",
            split_mixed_site,
            normalized,
        )
        notes.append("Mixed site normalized with equal distribution")

    # Handle compositional parameters (e.g., La(2-x)Srx) when values are provided
    if has_parameter and variable_values:
        # Helper to evaluate parameter expressions
        def eval_param_expr(expr: str) -> float | None:
            for var, val in param_defaults.items():
                expr = re.sub(rf"\b{re.escape(var)}\b", str(val), expr)
            try:
                return eval(expr)
            except Exception:
                return None

        # Substitute element(expr) patterns like La(2-x) -> La1.85
        def eval_element_paren(match: re.Match) -> str | None:
            element = match.group(1)
            expr = match.group(2)
            result = eval_param_expr(expr)
            return f"{element}{result}" if result is not None else match.group(0)

        normalized = re.sub(r"([A-Z][a-z]?)\(([^)]+)\)", eval_element_paren, normalized)

        # Substitute element<num>[+-]<var> patterns like Ba1-x -> Ba0.6 (without parentheses)
        def eval_element_no_paren(match: re.Match) -> str | None:
            element = match.group(1)
            expr = match.group(2)  # e.g., "1-x" or "2+y"
            result = eval_param_expr(expr)
            return f"{element}{result}" if result is not None else match.group(0)

        normalized = re.sub(
            r"([A-Z][a-z]?)(\d+[+\-][xyz])(?=[A-Z]|$)",
            eval_element_no_paren,
            normalized,
        )

        # Substitute element+var patterns like Srx -> Sr0.15
        for var, val in param_defaults.items():
            # Match Element followed by variable, where variable is not followed by lowercase letter
            # (uppercase letter means new element, so that's OK)
            normalized = re.sub(
                rf"([A-Z][a-z]?){re.escape(var)}(?![a-z])", rf"\g<1>{val}", normalized
            )

        notes.append(f"Compositional parameters substituted with: {variable_values}")

    # Check if still has parameters
    if has_parameter and "x" not in variable_values:
        formula_type = "PARAMETER_FORMULA"
        notes.append(
            "WARNING: Contains compositional parameters - cannot fully normalize without specific x value"
        )
    elif re.search(r"[a-z]\)", normalized) or re.search(r"[δδ]", normalized):
        formula_type = "PARTIAL_NORMALIZATION"
        notes.append("Partially normalized - some expressions remain")
    else:
        try:
            Composition(normalized)
            formula_type = "STOICHIOMETRIC"
        except Exception:
            formula_type = "INVALID"
            notes.append("Normalization produced invalid formula")

    return normalized, formula_type, notes
