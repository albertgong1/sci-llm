import re

from src.http_util import get_responses as _get_responses


def _get_proper_doi_from_doi_url_resp(orig_doi_url: str, doi_url_resp: str) -> str:
    # test: https://regex101.com
    # link.aps.org style
    pat = r"<a href=\"https:\/\/(.*?)doi\/(.*)\""
    x = re.search(pat, doi_url_resp)
    if x:
        return x.group(2)

    # iopscience style
    pat = r"a href=\"https://iopscience(.*?)/article/(.*)?\""
    x = re.search(pat, doi_url_resp)
    if x:
        return x.group(2)

    b = orig_doi_url.split("doi.org/")[-1].strip()
    return b


def get_proper_doi_from_doi_urls(doi_urls: list[str]) -> list[str]:
    chunk_size = 1_000
    collection = []
    for i in range(0, len(doi_urls), chunk_size):
        doi_url_group = doi_urls[i : i + chunk_size]
        responses = _get_responses(
            doi_url_group, max_concurrent_coro=chunk_size, allow_redirects=False
        )
        parsed_dois = [
            _get_proper_doi_from_doi_url_resp(in_url, x)
            for x, in_url in zip(responses, doi_urls)
        ]
        collection += parsed_dois
    return collection
