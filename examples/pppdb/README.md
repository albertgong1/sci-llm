# PPPDB (Polymers Database)

Scripts to scrape data from the [Polymer Property Predicted Database](https://pppdb.uchicago.edu/).

## Setup Instructions

```bash
uv pip install selenium webdriver-manager beautifulsoup4 pandas
```

## Available Scrapers

### Chi (χ) Database
Scrapes Flory-Huggins interaction parameters between polymer pairs.

```bash
uv run python scrape_all_chi_entries.py
```

Outputs: `pppdb_chi_complete.csv`

### Tg Database
Scrapes glass transition temperatures for polymers.

```bash
uv run python scrape_all_tg_entries.py
```

Outputs: `pppdb_tg_complete.csv`

### Cloud Points Database
Scrapes binary polymer solution cloud point data (phase separation temperatures).

```bash
uv run python scrape_all_cloud_point_entries.py
```

Outputs: `pppdb_cloud_point_complete.csv`

## Options

All scripts support a `--show-browser` flag to display the browser window (useful for debugging):

```bash
uv run python scrape_all_cloud_point_entries.py --show-browser
```
