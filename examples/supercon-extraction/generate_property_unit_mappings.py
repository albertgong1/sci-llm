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

import re
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO)
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
    label = str(row.get("label", "")).lower() if pd.notna(row.get("label")) else ""
    description = (
        str(row.get("description", "")).lower()
        if pd.notna(row.get("description"))
        else ""
    )
    return "unit of" in label or "unit of" in description


def extract_property_name_from_unit(row) -> set[str]:  # noqa: ANN001
    """Extract a set of property name aliases that this unit is for."""
    label = row["label"]
    description = row["description"]

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
    # import pdb; pdb.set_trace()
    if not property_row.get("Physically_Relevant", False):
        return False

    # Property can only have units if it's Float or Double
    data_type = property_row["data_type"]
    if data_type not in ("Float", "Double"):
        return False

    prop_db = property_row["db"]

    # Skip method properties - they should not have units/values associated
    # Methods are the values in PROPERTY_METHOD_MAPPINGS (e.g., mhc1, mhc2, gap, mdebye)
    method_properties = set(PROPERTY_METHOD_MAPPINGS.values())
    if prop_db in method_properties:
        return False

    prop_label_val = property_row.get("label", "")
    prop_label = str(prop_label_val).lower() if pd.notna(prop_label_val) else ""
    prop_desc_val = property_row.get("description", "")
    prop_desc = str(prop_desc_val).lower() if pd.notna(prop_desc_val) else ""

    if property_names_from_unit:
        # import pdb; pdb.set_trace()
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
    "hc1t": "tempc1",
    "phc1t": "tempc1",
    "nhc1t": "tempc1",
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
    "tc": "tcmeth",
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
    "gap": "gapmeth",
    "gapene": "gapmeth",
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

# Define property-to-pressure mappings
PROPERTY_PRESSURE_MAPPINGS = {
    "dtcdp": "pmax",
}

# Define explicit property-to-unit mappings
# (for properties that need manual unit assignments)
EXPLICIT_PROPERTY_UNIT_MAPPINGS = {
    "debyet": "utc",
    "curiet": "utc",
    "neelt": "utc",
}


# Read the CSV file
input_file = "properties-oxide-metal-glossary.csv"
output_file = "property_unit_mappings.csv"

mappings = []

# Read the CSV file using pandas
df = pd.read_csv(input_file, encoding="utf-8", index_col=0)
rows = df.to_dict("records")

# Process rows to find unit -> property mappings
i = 0
all_found_properties = set()
while i < len(rows):
    row = rows[i]
    if is_unit_row(row):
        # import pdb; pdb.set_trace()
        unit_db = row["db"]
        # update all found properties with unit
        all_found_properties.add(unit_db)
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
                prop_db = next_row["db"]
                found_properties.append(prop_db)

            j += 1
        # update all found properties with related properties
        all_found_properties.update(found_properties)

        # Add mappings for all found properties
        for prop_db in found_properties:
            # Get temperature condition if it exists
            if temp_condition := PROPERTY_TEMPERATURE_CONDITIONS.get(prop_db):
                all_found_properties.add(temp_condition)
            # Get field condition if it exists
            if field_condition := PROPERTY_FIELD_CONDITIONS.get(prop_db):
                all_found_properties.add(field_condition)
            # Get pressure condition if it exists
            if pressure_condition := PROPERTY_PRESSURE_MAPPINGS.get(prop_db):
                all_found_properties.add(pressure_condition)
            # Get method if it exists
            if method := PROPERTY_METHOD_MAPPINGS.get(prop_db):
                all_found_properties.add(method)
            # Get figure/table location if it exists
            if figure_location := PROPERTY_FIGURE_MAPPINGS.get(prop_db):
                all_found_properties.add(figure_location)

            # Find property label
            prop_label = ""
            for r in rows:
                if r["db"] == prop_db:
                    prop_label = r["label"]
                    break

            unit_label = row["label"]

            mappings.append(
                {
                    "property": prop_db,
                    "unit": unit_db,
                    "property_label": prop_label,
                    "unit_label": unit_label,
                    "conditions.temperature": temp_condition,
                    "conditions.field": field_condition,
                    "conditions.pressure": pressure_condition,
                    "methods": method,
                    "location.figure_or_table": figure_location,
                }
            )

    i += 1

# Add explicit property-unit mappings
for prop_db, unit_db in EXPLICIT_PROPERTY_UNIT_MAPPINGS.items():
    # Skip if this property-unit pair already exists
    if any(m["property"] == prop_db and m["unit"] == unit_db for m in mappings):
        continue

    # Find property and unit labels
    prop_label = ""
    unit_label = ""
    for r in rows:
        if r["db"] == prop_db:
            prop_label = r["label"]
        if r["db"] == unit_db:
            unit_label = r["label"]

    # Get associated conditions and methods
    temp_condition = PROPERTY_TEMPERATURE_CONDITIONS.get(prop_db)
    field_condition = PROPERTY_FIELD_CONDITIONS.get(prop_db)
    pressure_condition = PROPERTY_PRESSURE_MAPPINGS.get(prop_db)
    method = PROPERTY_METHOD_MAPPINGS.get(prop_db)
    figure_location = PROPERTY_FIGURE_MAPPINGS.get(prop_db)

    # Update all found properties
    all_found_properties.add(prop_db)
    all_found_properties.add(unit_db)
    if temp_condition:
        all_found_properties.add(temp_condition)
    if field_condition:
        all_found_properties.add(field_condition)
    if pressure_condition:
        all_found_properties.add(pressure_condition)
    if method:
        all_found_properties.add(method)
    if figure_location:
        all_found_properties.add(figure_location)

    mappings.append(
        {
            "property": prop_db,
            "unit": unit_db,
            "property_label": prop_label,
            "unit_label": unit_label,
            "conditions.temperature": temp_condition,
            "conditions.field": field_condition,
            "conditions.pressure": pressure_condition,
            "methods": method,
            "location.figure_or_table": figure_location,
        }
    )

# Add remaining physically relevant properties as unitless
remaining_physically_relevant_properties = df[
    ~df["db"].isin(all_found_properties) & df["Physically_Relevant"]
]["db"].tolist()
for prop_db in remaining_physically_relevant_properties:
    mappings.append(
        {
            "property": prop_db,
            "unit": None,
        }
    )

# Write the mappings to a CSV file using pandas
mappings_df = pd.DataFrame(
    mappings,
    columns=[
        "property",
        "unit",
        "property_label",
        "unit_label",
        "conditions.temperature",
        "conditions.field",
        "conditions.pressure",
        "methods",
        "location.figure_or_table",
    ],
)
mappings_df.to_csv(output_file, index=False, encoding="utf-8")

logger.info(f"Total number of rows in input file: {len(rows)}")
logger.info(f"Found {len(mappings)} property-unit mappings")
logger.info(f"Linked {len(all_found_properties)} unique properties")
logger.info(f"Output written to: {output_file}")
logger.info("First 10 mappings:")
for mapping in mappings[:10]:
    logger.info(
        f"  {mapping['property']} -> {mapping['unit']} ({mapping['property_label']} -> {mapping['unit_label']})"
    )
