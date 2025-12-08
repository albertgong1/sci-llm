from pathlib import Path

import pytest

from src.arxiv import (
    get_arxiv_search_results_by_dois,
    get_arxiv_paper_detail_result_by_abs_links,
    get_arxiv_pdf_likely_has_supplement,
    get_arxiv_abs_link_from_pdf_link,
)


@pytest.fixture
def arxiv_pdf_path_has_supp_true(test_data_path: Path) -> Path:
    # indeed has a supplement in the pdf
    return test_data_path / "1904.02522v1.pdf"


@pytest.fixture
def arxiv_pdf_path_has_supp_false(test_data_path: Path) -> Path:
    # does NOT have a supplement in the pdf
    return test_data_path / "9908365.pdf"


def test_get_arxiv_search_results_by_dois(
    example_doi: str, google_recaptcha_key: str
) -> None:
    if not google_recaptcha_key:
        pytest.skip("environment variable GOOGLE_RECAPTCHA_KEY is unset, skipping.")

    results = get_arxiv_search_results_by_dois(
        [example_doi], captcha_auth=google_recaptcha_key
    )
    parsed_doi_search_result = results[0][0]
    abstract = parsed_doi_search_result.pop("arxiv_abstract")
    assert isinstance(abstract, str)
    assert abstract.startswith(
        "We report here the properties of single crystals of La$_2$Ni$_2$In. Electrical resistivity and specific heat measurements concur with the results of Density Functional Theory (DFT) calculations, finding that La$_{2}$Ni$_{2}$In is a weakly correlated metal"
    )
    assert parsed_doi_search_result == {
        "arxiv_authors": [
            "Jannis Maiwald",
            "Igor I. Mazin",
            "Alex Gurevich",
            "Meigan Aronson",
        ],
        "arxiv_title": " Superconductivity in La$_2$Ni$_2$In",
        "arxiv_url_abstract": "https://arxiv.org/abs/2008.06104",
        "arxiv_url_html": "https://arxiv.org/format/2008.06104",
        "arxiv_url_pdf": "https://arxiv.org/pdf/2008.06104",
    }


def test_get_arxiv_abs_from_pdf_link(example_arxiv_pdf_link: str) -> None:
    assert (
        get_arxiv_abs_link_from_pdf_link(example_arxiv_pdf_link)
        == "https://arxiv.org/abs/1903.05679"
    )


def test_get_arxiv_paper_detail_result_by_abs_links(
    example_arxiv_pdf_link: str, google_recaptcha_key: str
) -> None:
    if not google_recaptcha_key:
        pytest.skip("environment variable GOOGLE_RECAPTCHA_KEY is unset, skipping.")

    abs_links = [get_arxiv_abs_link_from_pdf_link(example_arxiv_pdf_link)]
    parsed_detail = get_arxiv_paper_detail_result_by_abs_links(
        abs_links, captcha_auth=google_recaptcha_key
    )
    assert len(parsed_detail) == 1
    paper_detail = parsed_detail[0]
    assert paper_detail == {
        "arxiv_comments": "19 pages (including Supplemental Material), 12 figures",
        "arxiv_journal_reference": "Phys. Rev. Research 1, 023011 (2019)",
        # this is the same DOI as the original source
        "arxiv_related_doi": "https://doi.org/10.1103/PhysRevResearch.1.023011",
        "arxiv_url_abstract": "https://arxiv.org/abs/1903.05679",
    }


def test_check_if_pdf_has_supplement(
    arxiv_pdf_path_has_supp_false: Path, arxiv_pdf_path_has_supp_true: Path
) -> None:
    assert get_arxiv_pdf_likely_has_supplement(arxiv_pdf_path_has_supp_true)
    assert not get_arxiv_pdf_likely_has_supplement(arxiv_pdf_path_has_supp_false)
