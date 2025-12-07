from textwrap import dedent

import pytest


@pytest.fixture()
def example_doi_url() -> str:
    return "https://doi.org/10.1103/physrevb.102.165125"


@pytest.fixture()
def example_doi() -> str:
    return "10.1103/PhysRevB.102.165125"


@pytest.fixture
def example_doi_response_html() -> str:
    return dedent(
        """
        <html><head><title>Handle Redirect</title></head>
        <body><a href="https://link.aps.org/doi/10.1103/PhysRevB.102.165125">https://link.aps.org/doi/10.1103/PhysRevB.102.165125</a></body></html>
        """
    ).strip()
