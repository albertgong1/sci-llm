from pathlib import Path

ASSETS_DIR = Path("assets")
SUPPORTED_DOMAINS: list[str] = ["supercon"]

DOMAIN2HF_DATASET_NAME: dict[str, str] = {
    "supercon": "kilian-group/supercon-mini",
}
