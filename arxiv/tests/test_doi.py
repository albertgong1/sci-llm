from src.doi import get_proper_doi_from_doi_urls


def test_get_doi_from_doi_url_resp(example_doi_url: str, example_doi: str) -> None:
    parsed_dois = get_proper_doi_from_doi_urls([example_doi_url])
    assert parsed_dois == [example_doi]


def test_get_doi_from_doi_dx_url_resp(example_doi_dx_url: str) -> None:
    parsed_dois = get_proper_doi_from_doi_urls([example_doi_dx_url])
    assert parsed_dois == ["10.1021/ma702733f"]
