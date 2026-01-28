import os
from textwrap import dedent
from pathlib import Path

import pytest

from src.config import REPO_ROOT


@pytest.fixture
def example_doi_url() -> str:
    return "https://doi.org/10.1103/physrevb.102.165125"


@pytest.fixture
def example_doi_dx_url() -> str:
    return "https://dx.doi.org/10.1021/ma702733f"


@pytest.fixture
def example_doi() -> str:
    return "10.1103/PhysRevB.102.165125"


@pytest.fixture
def example_arxiv_pdf_link() -> str:
    return "https://arxiv.org/pdf/1903.05679"


@pytest.fixture(scope="session")
def google_recaptcha_key() -> str:
    # can set this manually if getting blocked often
    return os.getenv("GOOGLE_RECAPTCHA_KEY") or ""


@pytest.fixture(scope="session")
def test_data_path() -> Path:
    return (REPO_ROOT / "tests") / "test_data"
