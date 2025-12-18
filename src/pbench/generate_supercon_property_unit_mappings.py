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


# Read the CSV file
input_file = pbench.ASSETS_DIR / "supercon" / "properties-oxide-metal-glossary.csv"
output_file = pbench.ASSETS_DIR / "supercon" / "property_unit_mappings.csv"

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
                }
            )

    i += 1

# hard code some mappings that aren't captured by the programmatic approach above
mappings.append(
    {
        "property": "nort",
        "unit": "utc",
        "property_label": "normal temperature",
        "unit_label": "unit of Tc",
    }
)
mappings.append(
    {
        "property": "tempc1",
        "unit": "utc",
        "property_label": "measuring temperature",
        "unit_label": "unit of Tc",
    }
)

# Write the mappings to a CSV file
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=["property", "unit", "property_label", "unit_label"]
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
