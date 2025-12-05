import csv
import re
from pathlib import Path

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


def get_super_con_papers(csv_path: Path = DATASET_ROOT / "SuperConDOI.csv"):
    parsed = []
    with open(csv_path, newline="\n", encoding="utf-8") as f:
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

    return parsed


def get_real_doi_from_base_doi(parsed_objects: list[dict[str, str]]):
    all_doi_urls = [x["doi_link"] for x in parsed_objects]
    true_doi_values = get_proper_doi_from_doi_urls(all_doi_urls)
    assert len(parsed_objects) == len(true_doi_values)

    # now all entries have their 'true doi'
    for p, true_doi in zip(parsed_objects, true_doi_values):
        p["true_doi"] = true_doi

    return parsed_objects


def get_super_con_arxiv_info(testing_limit: int | None):
    super_con_papers = get_super_con_papers()
    parsed = get_real_doi_from_base_doi(super_con_papers)
    ordered_papers = list(
        sorted(parsed, key=lambda x: x["published_year"], reverse=True)
    )[:testing_limit]

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


def get_charge_density_wave_papers(
    csv_path: Path = DATASET_ROOT / "Charge_Density_Wave_Paper_list_Sheet1.csv",
) -> list[dict[str, str]]:
    parsed = []
    with open(csv_path, newline="\n", encoding="utf-8") as f:
        reader = csv.reader(f)
        # skip header
        _ = next(reader)
        for row in reader:
            paper_title, doi_link, published_year = row
            parsed.append(
                {
                    "paper_title": paper_title,
                    "doi_link": doi_link,
                    "published_year": int(published_year),
                }
            )
    return parsed


if __name__ == "__main__":
    # get_super_con_arxiv_info()
    papers = get_charge_density_wave_papers()
    parsed = get_real_doi_from_base_doi(papers)

    TESTING_LIMIT = None
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
    df.to_csv(
        DATASET_ROOT / "charge_density_wave_paper_list_sheet_1_augmented.csv",
        index=False,
    )
