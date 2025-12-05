import csv
import re
import pandas as pd

from src.config import DATASET_ROOT
from src.http_util import get_responses
from src.arxiv import get_arxiv_search_results_by_dois


def get_proper_doi_from_doi_url_resp(orig_doi_url: str, doi_url_resp: str) -> str:
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
        responses = get_responses(
            doi_url_group, max_concurrent_coro=chunk_size, allow_redirects=False
        )
        parsed_dois = [
            get_proper_doi_from_doi_url_resp(in_url, x)
            for x, in_url in zip(responses, doi_urls)
        ]
        collection += parsed_dois
    return collection


if __name__ == "__main__":
    SOURCE_CSV = DATASET_ROOT / "SuperConDOI.csv"
    # turn to None for all
    TESTING_LIMIT = None

    parsed = []
    with open(SOURCE_CSV, newline="\n", encoding="utf-8") as f:
        reader = csv.reader(f)
        # skip header
        _ = next(reader)
        for row in reader:
            paper_id, paper_title, doi_link, published_year, publisher_name = row
            parsed.append(
                {
                    "paper_id": paper_id,
                    "paper_title": paper_title,
                    "doi_link": doi_link,
                    "published_year": int(published_year),
                    "publisher_name": publisher_name,
                }
            )

    all_doi_urls = [x["doi_link"] for x in parsed]
    true_doi_values = get_proper_doi_from_doi_urls(all_doi_urls)
    assert len(parsed) == len(true_doi_values)

    # now all entries have their 'true doi'
    for p, true_doi in zip(parsed, true_doi_values):
        p["true_doi"] = true_doi

    print("Done getting true DOIs")
    ordered_papers = list(
        sorted(parsed, key=lambda x: x["published_year"], reverse=True)
    )[:TESTING_LIMIT]

    # using the true dois, search for them
    returned_results = get_arxiv_search_results_by_dois(
        [x["true_doi"] for x in ordered_papers]
    )

    normed = []
    for paper_info, arxiv_search_results in zip(ordered_papers, returned_results):
        for arxiv_search_result in arxiv_search_results:
            to_save = {
                **paper_info,
                **arxiv_search_result,
            }
            normed.append(to_save)

    df = pd.json_normalize(normed)
    df.to_csv(DATASET_ROOT / "supercon_augmented_search_results.csv", index=False)
