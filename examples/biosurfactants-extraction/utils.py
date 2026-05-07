"""Utility constants and functions for biosurfactants-extraction scripts.

NOTE: this may be outdated

"""

from pathlib import Path

# Dataset configuration
HF_DATASET_NAME = "kilian-group/biosurfactants-extraction"
HF_DATASET_REVISION = "main"
HF_DATASET_SPLIT = "lite"
GT_EMBEDDINGS_PATH = Path("scoring/gt_property_name_gemini-embedding-001.json")

# Category aliases (maps underscore names to display names)
CATEGORY_ALIASES: dict[str, str] = {
    "interfacial_and_surface_properties": "Interfacial\n& Surface",
    "physicochemical_and_phase_properties": "Physicochemical\n& Phase",
    "soil_handling_and_electrostatic_properties": "Soil Handling\n& Electrostatic",
    "biological_and_kinetic_properties": "Biological\n& Kinetic",
    "rheological_and_physical_properties": "Rheological\n& Physical",
}
