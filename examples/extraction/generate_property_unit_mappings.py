import csv
import re

def is_unit_row(row):
    """Check if a row represents a unit definition."""
    label = row.get('label', '').lower()
    description = row.get('description', '').lower()
    return 'unit of' in label or 'unit of' in description

def extract_property_name_from_unit(row):
    """Extract the property name that this unit is for."""
    label = row.get('label', '')
    description = row.get('description', '')
    
    # Try to extract from "unit of X" pattern
    match = re.search(r'unit of\s+(\w+)', label + ' ' + description, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None

def is_related_property(unit_row, property_row, property_name_from_unit):
    """Check if property_row is related to the unit_row."""
    if not property_row.get('Physically_Relevant') == 'TRUE':
        return False
    
    prop_db = property_row.get('db', '').lower()
    prop_label = property_row.get('label', '').lower()
    prop_desc = property_row.get('description', '').lower()
    
    if property_name_from_unit:
        search_term = property_name_from_unit.lower()
        # Check if the property name appears in the property row
        if (search_term in prop_label or 
            search_term in prop_desc or
            search_term in prop_db):
            return True
    
    return False

# Read the CSV file
input_file = '/Users/ag2435/sci_llm/src/sci-llm/examples/extraction/data/properties-oxide-metal-glossary.csv'
output_file = '/Users/ag2435/sci_llm/src/sci-llm/examples/extraction/property_unit_mappings.csv'

mappings = []

with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Process rows to find unit -> property mappings
i = 0
while i < len(rows):
    row = rows[i]
    
    if is_unit_row(row):
        unit_db = row.get('db', '')
        property_name = extract_property_name_from_unit(row)
        
        # Look ahead for related properties
        j = i + 1
        found_properties = []
        
        # Check next several rows (up to 10) for related properties
        while j < min(i + 15, len(rows)):
            next_row = rows[j]
            
            # Stop if we hit another unit row
            if is_unit_row(next_row):
                break
            
            # Check if this is a related property
            if is_related_property(row, next_row, property_name):
                found_properties.append(next_row.get('db', ''))
            
            j += 1
        
        # Add mappings for all found properties
        for prop_db in found_properties:
            mappings.append({
                'property': prop_db,
                'unit': unit_db,
                'property_label': [r.get('label', '') for r in rows if r.get('db') == prop_db][0] if prop_db else '',
                'unit_label': row.get('label', '')
            })
    
    i += 1

# Write the mappings to a CSV file
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['property', 'unit', 'property_label', 'unit_label'])
    writer.writeheader()
    writer.writerows(mappings)

print(f"Found {len(mappings)} property-unit mappings")
print(f"Output written to: {output_file}")
print("\nFirst 10 mappings:")
for mapping in mappings[:10]:
    print(f"  {mapping['property']} -> {mapping['unit']} ({mapping['property_label']} -> {mapping['unit_label']})")

