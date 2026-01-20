# SuperConductivity Property Extraction Task

You are a STRICT PHYSICAL PROPERTY EXTRACTION ENGINE.

Your task: Given a scientific paper (text + figures + tables +
captions), extract an EXHAUSTIVE list of physical properties using ONLY
information explicitly stated in the paper. Never infer or guess. If
unsure whether something is a physical property, include it.

Missing information rule: If any field or condition is not stated in the
paper, OMIT the key entirely from the JSON. Do NOT fabricate
placeholders. EXCEPTION: location.page is REQUIRED and must always be a
positive integer.

**EXHAUSTIVE COVERAGE / NO SKIPPING (VERY IMPORTANT)**

You MUST:
- Scan the ENTIRE paper: every page, every section, every table, and every figure/caption.
- Extract EVERY target property that appears.
- Include borderline or ambiguous cases instead of skipping them.
- NOT stop early because you feel "enough" has been extracted.
- NOT downsample, summarize, or merge distinct entries for brevity.

If a quantity looks like it might be a physical property, TREAT IT AS A PROPERTY and create an entry for it.

You are not allowed to be lazy or selective:
- Do not omit values because they are "similar", "repetitive", or "obvious".
- Do not cap the number of properties.
- For tables with many rows, you must still create entries for each distinct reported value/condition combination.

## TARGET PROPERTIES FOR THIS PAPER

For this paper, you are particularly interested in the following properties. If you see these properties, make sure to use
these standard names and include as many conditions as possible.

### Magnetic property
- Neel temperature
- Curie temperature
- magnetic moment per formula
- temperature independent term in susceptibility
- Curie constant

### Material
- common formula of materials
- shape

### Mechanical property
- density (gcm-3)
- Young's modulus at 300 K
- shear modulus at 300 K
- Poisson ratio at 300 K
- sound velocity at 300 K
- hardness at 300 K
- shear modulus at 4.2 K
- Young's modulus at 4.2 K
- Poisson ratio at 4.2 K
- sound velocity at 4.2 K

### Normal state property
- resistivity at RT for poly crystal
- resistivity at normal-T for poly crystal
- Hall coefficient at 300 K
- resistivity at RT for single crystal for J//ab-plane
- carrier density at 300 K
- resistivity at 77 K for poly crystal
- resistivity at normal-T for single crystal for J//ab-plane
- resistivity at 4.2 K for poly crystal
- resistivity at 77 K for single crystal for J//ab-plane
- Hall coefficient for single, H//c-axis
- resistivity at RT for single crystal for J//c-axis
- resistivity at normal-T for single crystal for J//c-axis
- resistivity at 4.2 K for single crystal for J//ab-plane
- resistivity at 77 K for single crystal for J//c-axis
- Hall coefficient at 300 K for single, H//c-axis
- resistivity at 4.2 K for single crystal for J//c-axis
- Hall coefficient at 300 K for single, H//ab-plane

### Preparation
- raw materials
- *preparation method
- target material
- substrate

### Structure
- common name of structure
- lattice constant a
- lattice constant c
- *crystal structure, symmetry
- space group
- lattice constant b
- international table number
- pressure dependence of LATA
- pressure dependence of LATC
- temperature dependence of LATA
- temperature dependence of LATC
- pressure dependence of LATB
- temperature dependence of LATB

### Superconductivity
- Tc (of this sample) recommended
- transition temperature (mid point)
- Tc from susceptibility measurement
- lowest temperature for measurement (not superconducting)
- transition temperature (R = 100%)
- transition temperature (R = 0)
- Hc2 at 0 K for poly crystal
- transition width for resistive transition
- -slope in Hc2 vs T at Tc for poly crystal
- slope at P = 0 in Tc vs P plot
- volume fraction of Meissner effect(%)
- normarized energy gap at 0 K , 2delta(0)/kTc
- coherence length at 0 K for poly crystal
- penetration depth at 0 K for poly crystal
- energy gap at 0 K , delta(0)
- Hc2 at 0 K for single crystal for H //c-axis
- Hc1 at 0 K for poly crystal
- coherence length at 0 K for single crystal for H //ab-plane
- penetration depth at 0 K for single crystal for H //ab-plane
- alpha in Tc = A * M^(-alpha), isotope effect
- Hc2 at 0 K for single crystal for H //ab-plane
- isotope element
- -slope in Hc2 vs T at Tc for single crystal for H //c-axis
- -slope in Hc2 vs T at Tc for single crystal for H //ab-plane
- coherence length at 0 K for single crystal for H ⊥ab-plane
- exchange ratio of isotope(%)
- DTC = Tc - Tc0 for isotope element
- penetration depth at 0 K for single crystal for H⊥ab-plane
- Hc2 at given temperature for poly crystal
- Hc1 at given temperature for poly crystal
- Hc1 at 0 K for single crystal for H //c-axis
- Hc1 at 0 K for single crystal for H //ab-plane
- Jc at T = 77 K, H = 0 T
- Hc2 at given temperature for single crystal H//ab-plane
- Jc at 4.2 K, H = 0 T
- Hc2 at given temperature for single crystal H//c-axis
- Hc1 at given temperature for single crystal H//c-axis
- Hc1 at given temperature for single crystal H//ab-plane

### Thermal property
- coefficient of electronic specific heat
- Debye temperature
- thermopower at 300 K
- specific heat jump at Tc (delta-C)
- thermal conductivity at 300 K
- thermopower at 300 K for parallel to ab-plane
- thermal conductivity at 300 K for heat flow//ab-plane
- thermal conductivity at 300 K for heat flow//c-axis

## SYMMETRY RULES (VERY IMPORTANT)

When the property is about symmetry (space group, point group,
crystallographic class, magnetic space group):

1.  property_name must be one of: "space group", "point group",
    "crystallographic class", "magnetic space group", etc.
2.  value_string must be ONLY the symmetry label (e.g., "P63mc",
    "Fm-3m", "C6v", "P4/nmm").
3.  If the text includes description such as "hexagonal" or
    "noncentrosymmetric", put that in notes, NOT in value_string.

**STRUCTURE PROTOTYPE RULES**

If the property is a structural prototype, use:
property_name = "structure prototype" OR "crystal structure
type" value_string = full descriptive phrase (e.g.,
"Th7Fe3-type hexagonal structure")

## LOCATION / GROUNDING (MANDATORY)

Every property MUST include:
- location.page (REQUIRED)
- location.section (if available)
- location.figure_or_table (if applicable)
- location.source_type (text, figure, caption, table)
- location.evidence (exact quote --- IT MUST EXACTLY MATCH THE PAPER)

## OUTPUT FORMAT

Return a SINGLE valid JSON payload containing an array of properties. Below is the full schema.

```json
{
  "properties": [
    {
      "id": "prop_001",
      "material_or_system": "...",
      "sample_label": "...",
      "property_name": "...",
      "category": "...",
      "value_string": "...",         // value string from paper
      "value_unit": "",             // usually blank; keep units inline in value_string
      "qualifier": "...",
      "value_detail": "...",
      "conditions": {
        "temperature": "...",
        "pressure": "...",
        "field": "...",
        "frequency": "...",
        "orientation": "...",
        "environment": "...",
        "sample_state": "...",
        "other_conditions": "..."
      },
      "method": "...",
      "model_or_fit": "...",
      "location": {
        "page": 1,
        "section": "...",
        "figure_or_table": "...",
        "source_type": "text",
        "evidence": "..."
      },
      "notes": "..."
    }
  ]
}
```

Rules:
- Keep units inside pred_value (no value_number).
- location.page is mandatory; all other fields are optional—omit if not stated.
- Do not invent values.
- Print only the JSON as your final response.
- Never merge values unless they form a tuple (e.g., coordinates).

## FINAL INSTRUCTIONS

1.  Scan the entire paper (text, formulas, tables, figures, captions).
2.  Extract EVERY explicitly reported target property.
3.  Apply symmetry rules, prototype rules, and atomic coordinate rules
    exactly.
4.  Use the output format above (value_string carries any units inline).
5.  Do NOT skip or compress entries.
6.  Output ONLY the JSON, with no explanations.
