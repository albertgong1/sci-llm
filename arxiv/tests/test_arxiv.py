from pathlib import Path
from typing import Callable, TypeVar

import pytest

from src.arxiv import (
    get_arxiv_search_results_by_dois,
    get_arxiv_paper_detail_result_by_abs_links,
    get_arxiv_pdf_likely_has_supplement,
    get_arxiv_abs_link_from_pdf_link,
    get_has_supplement_from_arxiv_comment,
)
from src.arxiv.arxiv import ReCaptchaExpiredException

T = TypeVar("T")


@pytest.fixture
def arxiv_pdf_path_has_supp_true(test_data_path: Path) -> Path:
    # indeed has a supplement in the pdf
    return test_data_path / "1904.02522v1.pdf"


@pytest.fixture
def arxiv_pdf_path_has_supp_false(test_data_path: Path) -> Path:
    # does NOT have a supplement in the pdf
    return test_data_path / "9908365.pdf"


def _handle_recaptcha(
    dc_callable: Callable[..., T], google_recaptcha_key: str, **kwargs
) -> T:
    try:
        # try to get results without captcha, works often but not always
        return dc_callable(**kwargs)
    except ReCaptchaExpiredException:
        if google_recaptcha_key:
            return dc_callable(
                **kwargs,
                captcha_auth=google_recaptcha_key,
            )
        else:
            pytest.skip(
                "Skipping because GOOGLE_RECAPTCHA_KEY is unset and got ReCaptcha request."
            )


def test_get_arxiv_search_results_by_dois(
    example_doi: str,
    google_recaptcha_key: str,
) -> None:
    results = _handle_recaptcha(
        get_arxiv_search_results_by_dois, google_recaptcha_key, true_dois=[example_doi]
    )
    parsed_doi_search_result = results[0][0]
    abstract = parsed_doi_search_result.pop("arxiv_abstract")
    assert isinstance(abstract, str)
    assert abstract.startswith(
        "We report here the properties of single crystals of La$_2$Ni$_2$In. Electrical resistivity and specific heat "
        "measurements concur with the results of Density Functional Theory (DFT) calculations, finding that "
        "La$_{2}$Ni$_{2}$In is a weakly correlated metal"
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
    example_arxiv_pdf_link: str,
    google_recaptcha_key: str,
) -> None:
    abs_urls = [get_arxiv_abs_link_from_pdf_link(example_arxiv_pdf_link)]
    parsed_detail = _handle_recaptcha(
        get_arxiv_paper_detail_result_by_abs_links,
        google_recaptcha_key,
        abs_urls=abs_urls,
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


def test_get_arxiv_paper_detail_result_by_abs_links_with_nested_contents(
    example_arxiv_pdf_link: str, google_recaptcha_key: str
) -> None:
    parsed_detail = _handle_recaptcha(
        get_arxiv_paper_detail_result_by_abs_links,
        google_recaptcha_key,
        abs_urls=["https://arxiv.org/abs/0808.3254"],
    )
    assert len(parsed_detail) == 1
    paper_detail = parsed_detail[0]
    assert paper_detail == {
        "arxiv_comments": "5 pages, 5 figures; data added and text revised. A combined version (this paper and arXiv:0807.1304) published in PRB",
        "arxiv_journal_reference": "Phys. Rev. B 79, 054521 (2009)",
        "arxiv_related_doi": "https://doi.org/10.1103/PhysRevB.79.054521",
        "arxiv_url_abstract": "https://arxiv.org/abs/0808.3254",
    }


def test_check_if_pdf_has_supplement(
    arxiv_pdf_path_has_supp_false: Path, arxiv_pdf_path_has_supp_true: Path
) -> None:
    assert get_arxiv_pdf_likely_has_supplement(arxiv_pdf_path_has_supp_true)
    assert not get_arxiv_pdf_likely_has_supplement(arxiv_pdf_path_has_supp_false)


def test_paper_comment_has_supplement() -> None:
    cases = [
        # easy negatives
        ("7 pages, 5 figures. See published version for the latest update", False),
        ("to appear in Physical Review B (2020); 13 pages, 9 figures", False),
        ("18 pages, 14 figures", False),
        # difficult negatives
        (
            "5 + epsilon pages, 7 figures, Supplemental Material on demand; v2 includes additional magnetization and resistivity data; v3 includes additional resistivity data from LaFeSiH single crystal",
            False,
        ),
        ("6 pages, 4 figures, Supplementary Material available", False),
        ("11 pages, 4 figures, supplementary information is not uploaded", False),
        (
            "4 figures, 5 pages (excluding supplementary material). To be published in Phys. Rev.",
            False,
        ),
        (
            "revised (6 pages, 5 figures) - includes additional experimental results",
            False,
        ),
        # easy positives
        ("15 pages, 4 figures, Supplementary material", True),
        (
            "5 pages + Supplemental Material, accepted for publication in J. Phys. Soc. Jpn",
            True,
        ),
        (
            "10 pages, 8 figures, including supplemental materials, chosen as Editors' choice, open access",
            True,
        ),
        ("10 pages, 6 figures, including supplemental material", True),
        ("19 pages (including Supplemental Material), 12 figures", True),
        (
            "34 pages (+ supplemental material), 15 figures (+ 4 in suppl. mat.); Accepted for publication in Physical Review B",
            True,
        ),
        ("4 pages, 7 figures, + supporting information", True),
        # difficult positives
        (
            "(main text - 5 pages, 4 figures; supplementary information - 4 pages, 5 figures, to be published in Physical Review Letters)",
            True,
        ),
        (
            "Main text: 5 pages, 4 figures, Supplemental information: 2 pages, 1 figure, 1 table",
            True,
        ),
        (
            "5 pages main text, 6 pages supplementary information; error in the interpretation of the crystallographic axes of CPSBS in the previous version has been corrected, resulting in updated discussions and conclusion",
            True,
        ),
        (
            "6 pages, 4 figures, 12 supplementary pages, 12 supplementary figures, PRB accepted as Rapid Communication",
            True,
        ),
        (
            "10 pages (including 4 supplementary), 9 figures (including 4 supplementary)",
            True,
        ),
        ("8 pages, 5 figures, main text plus supplemental material", True),
        (
            "9 pages, 5 figures (paper proper) + 2 pages, 3 figures (supporting information). Accepted for publication in Chemistry of Materials",
            True,
        ),
        (
            "5 pages, 5 figures, SUPPLEMENTAL MATERIAL. appears in Physical Review B: Rapid Communications (2015)",
            True,
        ),
    ]
    for input_text, expected_output in cases:
        assert get_has_supplement_from_arxiv_comment(input_text) == expected_output
