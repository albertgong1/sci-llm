from src.arxiv import (
    get_arxiv_search_results_by_dois,
    get_arxiv_paper_detail_result_by_abs_links,
)
from src.arxiv.arxiv import get_arxiv_abs_link_from_pdf_link


def test_get_arxiv_search_results_by_dois(example_doi: str) -> None:
    results = get_arxiv_search_results_by_dois([example_doi])
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
    example_arxiv_pdf_link: str,
) -> None:
    abs_links = [get_arxiv_abs_link_from_pdf_link(example_arxiv_pdf_link)]
    parsed_detail = get_arxiv_paper_detail_result_by_abs_links(abs_links)
    assert len(parsed_detail) == 1
    paper_detail = parsed_detail[0]
    assert paper_detail == {
        "comments": "19 pages (including Supplemental Material), 12 figures",
        "journal_reference": "Phys. Rev. Research 1, 023011 (2019)",
        # this is the same DOI as the original source
        "related_doi": "https://doi.org/10.1103/PhysRevResearch.1.023011",
        "url": "https://arxiv.org/abs/1903.05679",
    }
