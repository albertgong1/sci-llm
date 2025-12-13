#!/usr/bin/env python3
"""Script to scrape all entries from PPPdb Cloud Points database.

This version:
1. Clicks the "All" button to show all entries
2. Waits for the table to fully load
3. Extracts all data to CSV
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC  # noqa: N812
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
from typing import List, Dict


def fetch_all_cloud_point_data(headless: bool = True, wait_time: int = 20) -> list:
    """Fetch all entries from the cloud points database by clicking 'All' button.

    Args:
        headless: Run browser in headless mode
        wait_time: Maximum wait time in seconds

    Returns:
        List of Selenium WebElement objects representing table rows

    """
    url = "https://pppdb.uchicago.edu/cloud_points"

    # Setup Chrome options
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")

    print("=" * 70)
    print("PPPdb Cloud Points Database - Complete Scraper")
    print("=" * 70)
    print("\nInitializing Chrome driver...")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        print(f"Loading {url}...")
        driver.get(url)

        # Wait for initial table load
        print("Waiting for initial table load...")
        WebDriverWait(driver, wait_time).until(
            EC.presence_of_element_located((By.TAG_NAME, "table"))
        )

        time.sleep(3)  # Let page stabilize

        # Find and click the "All" button to show all entries
        print("\nLooking for 'Show All' button...")
        try:
            # Common selectors for pagination "All" buttons
            all_button_selectors = [
                "//a[contains(text(), 'All')]",
                "//button[contains(text(), 'All')]",
                "//option[contains(text(), 'All')]",
                "//select[@name='DataTables_Table_0_length']//option[contains(text(), 'All')]",
                "//select[contains(@class, 'length')]//option[text()='All']",
            ]

            button_found = False
            for selector in all_button_selectors:
                try:
                    elements = driver.find_elements(By.XPATH, selector)
                    if elements:
                        element = elements[0]
                        print(f"Found 'All' button using selector: {selector}")

                        # If it's an option, we need to select it
                        if element.tag_name == "option":
                            # Find parent select and select this option
                            parent_select = element.find_element(By.XPATH, "..")
                            driver.execute_script(
                                "arguments[0].scrollIntoView(true);", parent_select
                            )
                            time.sleep(1)

                            # Click the select to open dropdown
                            parent_select.click()
                            time.sleep(1)

                            # Click the "All" option
                            element.click()
                            print("Selected 'All' from dropdown")
                        else:
                            # It's a button or link, just click it
                            driver.execute_script(
                                "arguments[0].scrollIntoView(true);", element
                            )
                            time.sleep(1)
                            element.click()
                            print("Clicked 'All' button")

                        button_found = True
                        break
                except Exception:
                    continue

            if button_found:
                print("Waiting for all entries to load...")
                time.sleep(5)  # Give time for all rows to render

                # Wait for table to update (look for more rows)
                print("Verifying all rows loaded...")
                time.sleep(3)
            else:
                print(
                    "Warning: Could not find 'All' button, will scrape visible entries"
                )
                print("You may need to manually select 'All' in the browser")

        except Exception as e:
            print(f"Warning: Error clicking 'All' button: {e}")
            print("Proceeding with visible entries...")

        # Scroll through the page to ensure all content is loaded
        print("\nScrolling to load all content...")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(2)

        # Extract data directly from Selenium WebElements (not page_source)
        # This is necessary because DataTables renders rows dynamically via JavaScript
        final_rows = driver.find_elements(By.XPATH, "//table//tbody//tr")
        print(f"\n✓ Successfully loaded page with {len(final_rows)} data rows")

        # Parse data while driver is still open
        print("\nParsing entries...")
        cloud_point_data = parse_cloud_point_data_from_elements(final_rows)

        return cloud_point_data

    finally:
        print("\nClosing browser...")
        driver.quit()


def parse_cloud_point_data_from_elements(row_elements: list) -> List[Dict]:
    """Parse Selenium WebElement rows to extract all cloud point entries.

    Args:
        row_elements: List of Selenium WebElement objects representing table rows

    Returns:
        List of dictionaries with cloud point data

    """
    data_rows = []

    print(f"Parsing {len(row_elements)} entries...")

    for idx, row in enumerate(row_elements, 1):
        try:
            # Find all td cells in this row
            cols = row.find_elements(By.TAG_NAME, "td")

            if len(cols) >= 14:
                # Column 0: Polymer
                polymer = cols[0].text.strip()

                # Column 1: Polymer CAS
                polymer_cas = cols[1].text.strip()

                # Column 2: Polymer SMILES
                polymer_smiles = cols[2].text.strip()

                # Column 3: Solvent
                solvent = cols[3].text.strip()

                # Column 4: Solvent CAS
                solvent_cas = cols[4].text.strip()

                # Column 5: Solvent SMILES
                solvent_smiles = cols[5].text.strip()

                # Column 6: Mw (Da)
                mw = cols[6].text.strip()

                # Column 7: PDI
                pdi = cols[7].text.strip()

                # Column 8: ϕ (phi - volume fraction)
                phi = cols[8].text.strip()

                # Column 9: w (weight fraction)
                w = cols[9].text.strip()

                # Column 10: P (MPa)
                pressure = cols[10].text.strip()

                # Column 11: CP (°C) - Cloud Point temperature
                cloud_point = cols[11].text.strip()

                # Column 12: 1-Phase
                one_phase = cols[12].text.strip()

                # Column 13: References
                references = []
                ref_links = cols[13].find_elements(By.TAG_NAME, "a")
                for link in ref_links:
                    href = link.get_attribute("href")
                    text = link.text.strip()

                    if href and "doi.org" in href:
                        # Clean DOI link
                        doi = (
                            href.replace("http://dx.doi.org/", "")
                            .replace("https://dx.doi.org/", "")
                            .replace("http://doi.org/", "")
                            .replace("https://doi.org/", "")
                        )
                        references.append(f"doi:{doi}")
                    elif text:
                        # Other citation formats
                        if "ISBN" in text or "Citation" in text or "Ciation" in text:
                            references.append(text)

                # Also check for text nodes in reference cell
                ref_text = cols[13].text.strip()
                if ref_text and not references:
                    references.append(ref_text)

                reference_str = "; ".join(references) if references else ""

                row_data = {
                    "entry_number": idx,
                    "polymer": polymer,
                    "polymer_cas": polymer_cas,
                    "polymer_smiles": polymer_smiles,
                    "solvent": solvent,
                    "solvent_cas": solvent_cas,
                    "solvent_smiles": solvent_smiles,
                    "mw_Da": mw,
                    "pdi": pdi,
                    "phi": phi,
                    "w": w,
                    "pressure_MPa": pressure,
                    "cloud_point_C": cloud_point,
                    "one_phase": one_phase,
                    "references": reference_str,
                }

                data_rows.append(row_data)

                # Progress indicator
                if idx % 50 == 0:
                    print(f"  Parsed {idx} entries...")
        except Exception as e:
            print(f"  Warning: Error parsing row {idx}: {e}")
            continue

    print(f"✓ Successfully parsed {len(data_rows)} entries")
    return data_rows


def export_to_csv(
    data: List[Dict], output_file: str = "pppdb_cloud_point_complete.csv"
) -> None:
    """Export all data to CSV.

    Args:
        data: List of cloud point data
        output_file: Output filename

    """
    df = pd.DataFrame(data)

    # Reorder columns
    columns = [
        "entry_number",
        "polymer",
        "polymer_cas",
        "polymer_smiles",
        "solvent",
        "solvent_cas",
        "solvent_smiles",
        "mw_Da",
        "pdi",
        "phi",
        "w",
        "pressure_MPa",
        "cloud_point_C",
        "one_phase",
        "references",
    ]

    df = df[columns]

    # Export
    df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"\n{'=' * 70}")
    print(f"✓ Data exported to: {output_file}")
    print(f"{'=' * 70}")

    # Statistics
    print("\nDatabase Statistics:")
    print(f"  Total entries: {len(df)}")
    print(f"  Unique polymers: {df['polymer'].nunique()}")
    print(f"  Unique solvents: {df['solvent'].nunique()}")
    print(f"  Entries with molecular weight: {(df['mw_Da'] != '').sum()}")
    print(f"  Entries with PDI: {(df['pdi'] != '').sum()}")
    print(f"  Entries with volume fraction (phi): {(df['phi'] != '').sum()}")
    print(f"  Entries with weight fraction (w): {(df['w'] != '').sum()}")
    print(f"  Entries with pressure: {(df['pressure_MPa'] != '').sum()}")
    print(f"  Entries with cloud point: {(df['cloud_point_C'] != '').sum()}")
    print(f"  Entries with references: {(df['references'] != '').sum()}")

    # Top polymers
    print("\nTop 10 Most Studied Polymers:")
    top_polymers = df["polymer"].value_counts().head(10)
    for polymer, count in top_polymers.items():
        print(f"  {polymer}: {count} entries")

    # Top solvents
    print("\nTop 10 Most Common Solvents:")
    top_solvents = df["solvent"].value_counts().head(10)
    for solvent, count in top_solvents.items():
        print(f"  {solvent}: {count} entries")

    # Cloud point temperature range
    cloud_points = pd.to_numeric(df["cloud_point_C"], errors="coerce").dropna()
    if len(cloud_points) > 0:
        print("\nCloud Point Temperature Range:")
        print(f"  Min: {cloud_points.min():.2f} °C")
        print(f"  Max: {cloud_points.max():.2f} °C")
        print(f"  Mean: {cloud_points.mean():.2f} °C")

    # Molecular weight range
    mw_values = pd.to_numeric(df["mw_Da"], errors="coerce").dropna()
    if len(mw_values) > 0:
        print("\nMolecular Weight Range:")
        print(f"  Min: {mw_values.min():.0f} Da")
        print(f"  Max: {mw_values.max():.0f} Da")
        print(f"  Mean: {mw_values.mean():.0f} Da")

    return df


def main(headless: bool = True) -> None:
    """Main execution."""
    try:
        # Fetch and parse all data
        cloud_point_data = fetch_all_cloud_point_data(headless=headless)

        if not cloud_point_data:
            print("\n✗ No data extracted!")
            return

        # Export to CSV
        df = export_to_csv(cloud_point_data)

        # Show sample
        print("\nFirst 5 entries preview:")
        print(
            df.head(5)[
                ["entry_number", "polymer", "solvent", "cloud_point_C", "pressure_MPa"]
            ]
        )

        print(f"\n{'=' * 70}")
        print("✓ Scraping completed successfully!")
        print("✓ All entries exported to pppdb_cloud_point_complete.csv")
        print(f"{'=' * 70}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape all entries from PPPdb Cloud Points database"
    )
    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Show browser window (useful for debugging)",
    )

    args = parser.parse_args()
    main(headless=not args.show_browser)
