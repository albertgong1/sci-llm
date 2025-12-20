#! /usr/bin/env -S uv run --env-file=.env -- python
"""Script to generate a CSV file of SuperCon property-unit mappings from the properties-oxide-metal-glossary.csv file.

NOTE: the unit property is usually found before the property it is associated with.

This script will generate a CSV file with the following columns:
- property: the property db
- unit: the unit db
- property_label: the label of the property
- unit_label: the label of the unit

Example usage:
```bash
./src/pbench/generate_supercon_property_unit_mappings.py
```
"""

import csv
import re
import logging
from pathlib import Path

import pbench

logger = logging.getLogger(__name__)

# Known property aliases and their full names
PROPERTY_ALIASES = {
    "TC": {"Tc", "temperature", "critical temperature", "transition temperature"},
    "HC": {"Hc", "field", "critical field", "magnetic field"},
    "JC": {"Jc", "current", "critical current", "current density"},
    "RHO": {"rho", "resistivity", "electrical resistivity"},
    "CONDUCTIVITY": {"conductivity", "electrical conductivity", "sigma"},
    "DENSITY": {"density", "mass density"},
    "PRESSURE": {"pressure", "p"},
    "VOLUME": {"volume", "v"},
    "ENERGY": {"energy", "e"},
    "ENTROPY": {"entropy", "s"},
}


def is_unit_row(row) -> bool:  # noqa: ANN001
    """Check if a row represents a unit definition."""
    label = row.get("label", "").lower()
    description = row.get("description", "").lower()
    return "unit of" in label or "unit of" in description


def extract_property_name_from_unit(row) -> set[str]:  # noqa: ANN001
    """Extract a set of property name aliases that this unit is for."""
    label = row.get("label", "")
    description = row.get("description", "")

    # Try to extract from "unit of X" pattern
    match = re.search(r"unit of\s+(\w+)", label + " " + description, re.IGNORECASE)
    if match:
        property_name = match.group(1).upper()

        # Return aliases if we have them, otherwise return the extracted name and common variations
        if property_name in PROPERTY_ALIASES:
            return PROPERTY_ALIASES[property_name]
        else:
            # Return a set with the property name and its lowercase version
            return {property_name, property_name.lower()}

    return set()


def is_related_property(unit_row, property_row, property_names_from_unit) -> bool:  # noqa: ANN001
    """Check if property_row is related to the unit_row.

    Args:
        unit_row: The unit row
        property_row: The property row to check
        property_names_from_unit: A set of property name aliases

    """
    if not property_row.get("Physically_Relevant") == "TRUE":
        return False

    # Property can only have units if it's Float or Double
    data_type = property_row.get("data_type", "")
    if data_type not in ("Float", "Double"):
        return False

    prop_db = property_row.get("db", "").lower()

    # Skip method properties - they should not have units/values associated
    # Methods are the values in PROPERTY_METHOD_MAPPINGS (e.g., mhc1, mhc2, gap, mdebye)
    method_properties = set(PROPERTY_METHOD_MAPPINGS.values())
    if prop_db in method_properties:
        return False

    prop_label = property_row.get("label", "").lower()
    prop_desc = property_row.get("description", "").lower()

    if property_names_from_unit:
        # Check if any of the aliases appear in the property row
        for search_term in property_names_from_unit:
            search_term_lower = search_term.lower()
            if (
                search_term_lower in prop_label
                or search_term_lower in prop_desc
                or search_term_lower in prop_db
            ):
                return True

    return False


# Define property-to-temperature-condition mappings
PROPERTY_TEMPERATURE_CONDITIONS = {
    "phc1t": "tempc2",
    "nhc1t": "tempc2",
    "hc2t": "tempc2",
    "phc2t": "tempc2",
    "nhc2t": "tempc2",
    "resn": "nort",
    "abresn": "nort",
    "cresn": "nort",
}

# Define property-to-field-condition mappings
PROPERTY_FIELD_CONDITIONS = {
    "rh300": "field",
    "rh300n": "field",
    "rh300p": "field",
    "rhn": "field",
}

# Define property-to-method mappings
PROPERTY_METHOD_MAPPINGS = {
    "hc1zero": "mhc1",
    "phc1zero": "mhc1",
    "nhc1zero": "mhc1",
    "hc1t": "mhc1",
    "phc1t": "mhc1",
    "nhc1t": "mhc1",
    "hc2zero": "mhc2",
    "phc2zeron": "mhc2",
    "nhc2zero": "mhc2",
    "hc2t": "mhc2",
    "phc2t": "mhc2",
    "nhc2t": "mhc2",
    "dhc2dt": "mdhc2dt",
    "pdhc2dt": "mdhc2dt",
    "ndhc2dt": "mdhc2dt",
    "cohere": "mcohere",
    "pcohere": "mcohere",
    "ncohere": "mcohere",
    "penet": "mpenet",
    "ppenet": "mpenet",
    "npenet": "mpenet",
    "gapene": "gap",
    "gapmeth": "gap",
    "debyet": "mdebye",
}

# Define property-to-figure/table mappings
PROPERTY_FIGURE_MAPPINGS = {
    "thc300": "thcfig",
    "thc300n": "thcfig",
    "thc300p": "thcfig",
    "tp300": "tpfig",
    "tp300n": "tpfig",
    "tp300p": "tpfig",
    "rh300": "hallfig",
    "rh300n": "hallfig",
    "rh300p": "hallfig",
    "rhn": "hallfig",
}


# Read the CSV file
input_file = pbench.ASSETS_DIR / "supercon" / "properties-oxide-metal-glossary.csv"
if False:
    output_file = pbench.ASSETS_DIR / "supercon" / "property_unit_mappings.csv"
else:
    output_file = Path("out-1219/property_unit_mappings.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)

mappings = []

with open(input_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Process rows to find unit -> property mappings
i = 0
while i < len(rows):
    row = rows[i]

    if is_unit_row(row):
        unit_db = row.get("db", "")
        property_names = extract_property_name_from_unit(row)

        # Look ahead for related properties
        j = i + 1
        found_properties = []

        # Check next several rows (up to 15) for related properties
        while j < min(i + 15, len(rows)):
            next_row = rows[j]

            # Stop if we hit another unit row
            if is_unit_row(next_row):
                break

            # Check if this is a related property (using the set of property names)
            if is_related_property(row, next_row, property_names):
                found_properties.append(next_row.get("db", ""))

            j += 1

        # Add mappings for all found properties
        for prop_db in found_properties:
            # Get temperature condition if it exists
            temp_condition = PROPERTY_TEMPERATURE_CONDITIONS.get(prop_db.lower(), "")
            # Get field condition if it exists
            field_condition = PROPERTY_FIELD_CONDITIONS.get(prop_db.lower(), "")
            # Get method if it exists
            method = PROPERTY_METHOD_MAPPINGS.get(prop_db.lower(), "")
            # Get figure/table location if it exists
            figure_location = PROPERTY_FIGURE_MAPPINGS.get(prop_db.lower(), "")

            mappings.append(
                {
                    "property": prop_db,
                    "unit": unit_db,
                    "property_label": [
                        r.get("label", "") for r in rows if r.get("db") == prop_db
                    ][0]
                    if prop_db
                    else "",
                    "unit_label": row.get("label", ""),
                    "conditions.temperature": temp_condition,
                    "conditions.field": field_condition,
                    "methods": method,
                    "location.figure_or_table": figure_location,
                }
            )

    i += 1

# Write the mappings to a CSV file
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "property",
            "unit",
            "property_label",
            "unit_label",
            "conditions.temperature",
            "conditions.field",
            "methods",
            "location.figure_or_table",
        ],
    )
    writer.writeheader()
    writer.writerows(mappings)

logger.info(f"Found {len(mappings)} property-unit mappings")
logger.info(f"Output written to: {output_file}")
logger.info("First 10 mappings:")
for mapping in mappings[:10]:
    logger.info(
        f"  {mapping['property']} -> {mapping['unit']} ({mapping['property_label']} -> {mapping['unit_label']})"
    )
