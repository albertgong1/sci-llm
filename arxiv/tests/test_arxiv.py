from src.arxiv import get_arxiv_search_results_by_dois


def test_get_arxiv_search_results_by_dois(example_doi: str) -> None:
    results = get_arxiv_search_results_by_dois([example_doi])
    parsed_doi_search_result = results[0][0]
    abstract = parsed_doi_search_result.pop("abstract")
    assert isinstance(abstract, str)
    assert abstract.startswith(
        "We report here the properties of single crystals of La$_2$Ni$_2$In. Electrical resistivity and specific heat measurements concur with the results of Density Functional Theory (DFT) calculations, finding that La$_{2}$Ni$_{2}$In is a weakly correlated metal"
    )
    assert parsed_doi_search_result == {
        "authors": [
            "Jannis Maiwald",
            "Igor I. Mazin",
            "Alex Gurevich",
            "Meigan Aronson",
        ],
        "title": " Superconductivity in La$_2$Ni$_2$In",
        "url_abstract": "https://arxiv.org/abs/2008.06104",
        "url_html": "https://arxiv.org/format/2008.06104",
        "url_pdf": "https://arxiv.org/pdf/2008.06104",
    }
