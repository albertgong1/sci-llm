"""Categorical mappings for integer codes in the SuperCon Database.

Reference: https://mdr.nims.go.jp/filesets/28e52e3f-8f92-4656-abb6-354e4cef11a1/download
"""

# Crystal structure symmetry mapping
CRYSTAL_SYMMETRY: dict[int, str] = {
    1: "cubic",
    2: "tetragonal",
    3: "orthorhombic",
    4: "monoclinic",
    5: "triclinic",
    6: "trigonal",
    7: "hexagonal",
}

# Shape mapping
# NOTE: we insert space before parentheses for formatting consistency
SHAPE: dict[int, str] = {
    1: "single phase (bulk)",
    2: "multi phase (bulk)",
    3: "single crystal (bulk)",
    4: "film",
    5: "film (single)",
}

# Method mapping
METHOD: dict[int, str] = {
    1: "powder sintering method",
    2: "doctor blade method",
    3: "screen printing method",  # typo in original source "screen printing metod"
    4: "extrusion method",
    5: "flux method",
    6: "Top Seeded Solution Growth method",
    7: "floating zone method",
    8: "Liquid Phase epitaxy",
    9: "melt-quench method",
    10: "Bridgeman",
    11: "sol-gel method",
    12: "organic acid base method",
    13: "suspension method",
    14: "spray coating method",
    15: "plasma spray method",
    16: "sputter deposition",
    17: "vacuum deposition",
    18: "CVD method",
    19: "Metal-Organic Chemical Vapor Deposition",
    20: "Vapor Growth method",
    21: "Molecular Beam Epitaxy method",
}

# Analysis method mapping
ANALM: dict[int, str] = {
    1: "X-ray crystallography",
    2: "Neutron crystallography",
    3: "Powder x-ray diffraction",
    4: "Powder neutron diffraction",
    5: "Electron diffraction",
}

# Gap measurement method mapping
GAPMETH: dict[int, str] = {
    1: "tunneling",
    2: "infrared spectroscopy",
    3: "thermal conductivity",
    4: "Raman spectroscopy",
    5: "AC susceptibility",
    6: "nuclear magnetic resonance",
    7: "surface impedance",
    8: "neutron diffraction",
    9: "ultraviolet photoemission spectroscopy",
    10: "microwave transmission",
}

# TC measurement method mapping
TC_MEASUREMENT_METHOD: dict[int, str] = {
    1: "magnetization",
    2: "ac susceptibility",
    3: "resistivity",
    4: "heat capacity",
    5: "tunneling",
    6: "infrared spectroscopy",
    7: "thermal conductivity",
    8: "Raman spectroscopy",
    9: "nuclear magnetic resonance",
    10: "surface impedance",
    11: "neutron diffraction",
    12: "photoemission spectroscopy",
    13: "microwave transmission",
    14: "Others",
}
