from pbench_eval.normalize_material import classify_and_normalize

from pymatgen.core import Composition


def test_classify_and_normalize() -> None:
    """Test all examples from the docstring"""
    print("=" * 80)
    print("TESTING classify_and_normalize() DOCSTRING EXAMPLES")
    print("=" * 80)

    test_cases = [
        # (formula, variable_values, expected_normalized, expected_type, description)
        (
            "Er1Ba2Cu3Ox",
            {},
            "Er1Ba2Cu3O1",
            "STOICHIOMETRIC",
            "Example 1: Oxygen variable with defaults",
        ),
        (
            "Y1Ba2Cu3O(7-z)",
            {},
            "Y1Ba2Cu3O7",
            "STOICHIOMETRIC",
            "Example 2: Oxygen deficiency notation (z=0 default)",
        ),
        (
            "Y1Ba2Cu3O(7-z)",
            {"z": 0.5},
            "Y1Ba2Cu3O6.5",
            "STOICHIOMETRIC",
            "Example 3: Oxygen deficiency with z=0.5",
        ),
        (
            "(Sr,Ca)14Cu24O41",
            {},
            "Sr7.0Ca7.0Cu24O41",
            "STOICHIOMETRIC",
            "Example 4: Mixed site equal distribution",
        ),
        (
            "La(2-x)SrxCuO4",
            {},
            "La(2-x)SrxCuO4",
            "PARAMETER_FORMULA",
            "Example 5: Parameter formula without value",
        ),
        (
            "La(2-x)SrxCuO4",
            {"x": 0.15},
            "La1.85Sr0.15CuO4",
            "STOICHIOMETRIC",
            "Example 6: Parameter formula with x=0.15",
        ),
        (
            "Bi2Sr2CaCu2O(8+δ)",
            {},
            "Bi2Sr2CaCu2O8",
            "STOICHIOMETRIC",
            "Example 7: Oxygen excess with δ=0 default",
        ),
        (
            "Bi2Sr2CaCu2O(8+δ)",
            {"δ": 0.2},
            "Bi2Sr2CaCu2O8.2",
            "STOICHIOMETRIC",
            "Example 8: Oxygen excess with δ=0.2",
        ),
    ]

    all_passed = True

    for i, (formula, var_vals, expected_norm, expected_type, description) in enumerate(
        test_cases, 1
    ):
        print(f"\n{'=' * 80}")
        print(f"Test Case {i}: {description}")
        print(f"{'=' * 80}")
        print(f"Input formula: {formula}")
        if var_vals:
            print(f"Variables: {var_vals}")

        normalized, ftype, notes = classify_and_normalize(formula, var_vals)

        print("\nResults:")
        print(f"  Normalized: {normalized}")
        print(f"  Type: {ftype}")
        print(f"  Notes: {notes}")

        # Check if matches expected
        passed = normalized == expected_norm and ftype == expected_type

        if passed:
            print("\n✓ PASSED")

            # If stoichiometric, verify pymatgen can parse
            if ftype == "STOICHIOMETRIC":
                try:
                    comp = Composition(normalized)
                    print(f"  Pymatgen parsing: ✓ {comp.reduced_formula}")
                except Exception as e:
                    print(f"  Pymatgen parsing: ✗ {e}")
                    passed = False
        else:
            print("\n✗ FAILED")
            print(f"  Expected normalized: {expected_norm}")
            print(f"  Expected type: {expected_type}")
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 80)


# Run tests
if __name__ == "__main__":
    test_classify_and_normalize()
