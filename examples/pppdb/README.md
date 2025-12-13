# PPPDB (Polymers Database)

Scripts to scrape data from the [Polymer Property Predicted Database](https://pppdb.uchicago.edu/).

## Setup Instructions

1. Please following the instructions in [README.md](../../README.md#setup-instructions)

<details>
    <summary>Create CSV files from scratch (optional) </summary>

1. Install additional dependencies:

```bash
uv pip install selenium webdriver-manager beautifulsoup4 pandas
```

2. Run Python scripts to scrape the data from the official website:

```bash
uv run python scrape_all_chi_entries.py
uv run python scrape_all_tg_entries.py
uv run python scrape_all_cloud_point_entries.py
```

</details>
