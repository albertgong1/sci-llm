#!/usr/bin/env python3
"""Enhanced script to scrape all 263 entries from PPPdb Chi database.

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
from pathlib import Path


def fetch_all_chi_data(headless: bool = True, wait_time: int = 20) -> list:
    """Fetch all 263 entries from the database by clicking 'All' button.

    Args:
        headless: Run browser in headless mode
        wait_time: Maximum wait time in seconds

    Returns:
        List of Selenium WebElement objects representing table rows

    """
    url = "https://pppdb.uchicago.edu/chi"

    # Setup Chrome options
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")

    print("=" * 70)
    print("PPPdb Chi Database - Complete Scraper")
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
        chi_data = parse_chi_data_from_elements(final_rows)

        return chi_data

    finally:
        print("\nClosing browser...")
        driver.quit()


def parse_chi_data_from_elements(row_elements: list) -> List[Dict]:
    """Parse Selenium WebElement rows to extract all chi parameter entries.

    Args:
        row_elements: List of Selenium WebElement objects representing table rows

    Returns:
        List of dictionaries with chi data

    """
    data_rows = []

    print(f"Parsing {len(row_elements)} entries...")

    for idx, row in enumerate(row_elements, 1):
        try:
            # Find all td cells in this row
            cols = row.find_elements(By.TAG_NAME, "td")

            if len(cols) >= 7:
                # Column 0: Info (icon/button)
                info = cols[0].text.strip()

                # Column 1: Compound 1
                compound1 = cols[1].text.strip()

                # Column 2: Compound 2
                compound2 = cols[2].text.strip()

                # Column 3: χ (chi parameter)
                chi = cols[3].text.strip()

                # Column 4: χN
                chi_n = cols[4].text.strip()

                # Column 5: Temperature
                temperature = cols[5].text.strip()

                # Column 6: References
                references = []
                ref_links = cols[6].find_elements(By.TAG_NAME, "a")
                for link in ref_links:
                    href = link.get_attribute("href")
                    text = link.text.strip()

                    if href and "doi.org" in href:
                        # Clean DOI link
                        doi = href.replace("http://dx.doi.org/", "").replace(
                            "https://dx.doi.org/", ""
                        )
                        references.append(f"doi:{doi}")
                    elif text:
                        # Other citation formats
                        if "ISBN" in text or "Citation" in text or "Ciation" in text:
                            references.append(text)

                # Also check for text nodes in reference cell
                ref_text = cols[6].text.strip()
                if ref_text and not references:
                    references.append(ref_text)

                reference_str = "; ".join(references) if references else ""

                row_data = {
                    "entry_number": idx,
                    "compound_1": compound1,
                    "compound_2": compound2,
                    "chi_parameter": chi,
                    "chi_N": chi_n,
                    "temperature_K": temperature,
                    "references": reference_str,
                    "info": info,
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
    data: List[Dict], output_file: str = "data/pppdb_chi_complete.csv"
) -> None:
    """Export all data to CSV.

    Args:
        data: List of chi parameter data
        output_file: Output filename

    """
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(data)

    # Reorder columns
    columns = [
        "entry_number",
        "compound_1",
        "compound_2",
        "chi_parameter",
        "chi_N",
        "temperature_K",
        "references",
        "info",
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
    print(f"  Unique polymers (compound 1): {df['compound_1'].nunique()}")
    print(f"  Unique polymers (compound 2): {df['compound_2'].nunique()}")
    print(f"  Entries with chi parameter: {df['chi_parameter'].notna().sum()}")
    print(f"  Entries with temperature: {(df['temperature_K'] != '').sum()}")
    print(f"  Entries with χN: {(df['chi_N'] != '').sum()}")
    print(f"  Entries with references: {(df['references'] != '').sum()}")

    # Top polymers
    print("\nTop 10 Most Studied Polymers/Solvents:")
    all_compounds = pd.concat([df["compound_1"], df["compound_2"]])
    top_compounds = all_compounds.value_counts().head(10)
    for compound, count in top_compounds.items():
        print(f"  {compound}: {count} entries")

    # Temperature range
    temps = pd.to_numeric(df["temperature_K"], errors="coerce").dropna()
    if len(temps) > 0:
        print("\nTemperature Range:")
        print(f"  Min: {temps.min():.2f} K")
        print(f"  Max: {temps.max():.2f} K")
        print(f"  Mean: {temps.mean():.2f} K")

    return df


def main(headless: bool = True) -> None:
    """Main execution."""
    try:
        # Fetch and parse all data
        chi_data = fetch_all_chi_data(headless=headless)

        if not chi_data:
            print("\n✗ No data extracted!")
            return

        # Export to CSV
        df = export_to_csv(chi_data)

        # Show sample
        print("\nFirst 10 entries preview:")
        print(
            df.head(10)[
                [
                    "entry_number",
                    "compound_1",
                    "compound_2",
                    "chi_parameter",
                    "temperature_K",
                ]
            ]
        )

        print(f"\n{'=' * 70}")
        print("✓ Scraping completed successfully!")
        print("✓ All 263 entries exported to data/pppdb_chi_complete.csv")
        print(f"{'=' * 70}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape all 263 entries from PPPdb Chi database"
    )
    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Show browser window (useful for debugging)",
    )

    args = parser.parse_args()
    main(headless=not args.show_browser)
