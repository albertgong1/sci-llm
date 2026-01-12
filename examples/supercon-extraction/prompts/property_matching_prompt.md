You are an expert condensed matter physicist evaluating whether two property descriptions refer to the same physical measurement.

## Property 1
- Name: "{name1}"
- Context: "{context1}"

## Property 2
- Name: "{name2}"
- Context: "{context2}"

## Matching Rules:

**SAME property if:**
- Names are synonymous (e.g., "Tc" = "Critical Temperature" = "Superconducting Transition Temperature")
- Abbreviation differences only (e.g., "Jc" = "Critical Current Density")
- Capitalization/formatting differences

**DIFFERENT properties if:**
- Technically distinct measurements:
  - "Tc onset" ≠ "Tc zero" ≠ "Tc midpoint"
  - "Jc" ≠ "Ic" (density vs absolute current)
  - "lattice constant a" ≠ "lattice constant c"
  - "upper critical field Hc2" ≠ "lower critical field Hc1"
- Different measurement orientations when orientation matters (e.g., "resistivity (c-axis)" ≠ "resistivity (ab-plane)")
- Different conditions not reconcilable (e.g., "Tc at 0 GPa" ≠ "Tc at 10 GPa" unless pressure is tracked separately)

## Response Format:
Return JSON only:
{{
  "is_match": boolean,
  "confidence": "high" | "medium" | "low",
  "reason": "concise explanation",
  "matched_via": "direct" | "synonym" | "abbreviation" | "condition_reconciliation" | null
}}
