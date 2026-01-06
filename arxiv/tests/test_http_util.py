from src.http_util import get_responses


def test_get_responses(example_doi_url: str, example_doi_response_html: str) -> None:
    responses = get_responses([example_doi_url], allow_redirects=False)
    assert len(responses) == 1
    response_text = responses[0]
    assert response_text == example_doi_response_html
