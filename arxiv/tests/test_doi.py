from src.main import get_proper_doi_from_doi_url_resp


def test_get_doi_from_doi_url_resp(
    example_doi_url: str, example_doi_response_html: str
) -> None:
    assert (
        get_proper_doi_from_doi_url_resp(example_doi_url, example_doi_response_html)
        == "10.1103/PhysRevB.102.165125"
    )
