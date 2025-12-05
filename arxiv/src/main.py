import csv
from pathlib import Path

import pandas as pd

from src.arxiv import get_arxiv_search_results_by_dois
from src.config import ARTIFACTS_ROOT
from src.doi import get_proper_doi_from_doi_urls
from src.http_util import get_responses


def get_super_con_papers(
    csv_path: Path = ARTIFACTS_ROOT / "SuperConDOI.csv",
) -> list[dict[str, str]]:
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


def get_real_doi_from_base_doi(
    parsed_objects: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Fix the DOI case insensitivity issue.

    The DOIs given by the CSV seem to have case information removed, e.g.

        `10.1103/physrevb.102.165125` instead of
        `10.1103/PhysRevB.102.165125`.

    This matters because the arXiv advanced search is case-sensitive; writing
    the DOI with incorrect casing yields no results. Bad false negative. To
    resolve this, we can follow the DOI link to the DOI website itself (NOT
    the ultimate resolution URL), and it returns the DOI with proper casing.

    This is fast because there are no rate limits or throttling on the DOI
    website, so >1,000 conns can be open at once.

    :param parsed_objects: List of objects that contain a key named
        `doi_link`, which contains a DOI url formatted like:
        `https://doi.org/X` where X is the DOI.
    :return: return a reference to the original list of dictionaries.
        Technically this mutates the input by reference so output isn't
        necessary to assign but whtever
    """
    all_doi_urls = [x["doi_link"] for x in parsed_objects]
    true_doi_values = get_proper_doi_from_doi_urls(all_doi_urls)
    assert len(parsed_objects) == len(true_doi_values)

    # now all entries have their 'true doi'
    for p, true_doi in zip(parsed_objects, true_doi_values):
        p["true_doi"] = true_doi

    return parsed_objects


def get_super_con_arxiv_info(testing_limit: int | None) -> None:
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
    df.to_csv(ARTIFACTS_ROOT / "supercon_augmented_search_results.csv", index=False)


def get_charge_density_wave_papers(
    csv_path: Path = ARTIFACTS_ROOT / "Charge_Density_Wave_Paper_list_Sheet1.csv",
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


def get_charge_density_wave_arxiv_info(testing_limit: int | None = None) -> None:
    papers = get_charge_density_wave_papers()
    parsed = get_real_doi_from_base_doi(papers)

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
    df.to_csv(
        ARTIFACTS_ROOT / "charge_density_wave_paper_list_sheet_1_augmented.csv",
        index=False,
    )


def get_super_con_arxiv_info_downloads(
    saved_csv_path: Path = ARTIFACTS_ROOT / "supercon_augmented_search_results.csv",
) -> None:
    df = pd.read_csv(saved_csv_path)
    pdf_links = [
        x for x in df["arxiv_url_pdf"].tolist() if isinstance(x, str) and bool(x)
    ]
    save_to = saved_csv_path.parent / saved_csv_path.stem
    save_to.mkdir(exist_ok=True, parents=False)
    get_responses(pdf_links, download_to_folder=save_to)


def get_charge_density_wave_arxiv_downloads(
    saved_csv_path: Path = ARTIFACTS_ROOT
    / "charge_density_wave_paper_list_sheet_1_augmented.csv",
) -> None:
    df = pd.read_csv(saved_csv_path)
    pdf_links = [
        x for x in df["arxiv_url_pdf"].tolist() if isinstance(x, str) and bool(x)
    ]
    save_to = saved_csv_path.parent / saved_csv_path.stem
    save_to.mkdir(exist_ok=True, parents=False)
    get_responses(pdf_links, download_to_folder=save_to)


if __name__ == "__main__":
    # get_super_con_arxiv_info()
    # get_charge_density_wave_arxiv_info()
    # get_super_con_arxiv_info_downloads()
    get_charge_density_wave_arxiv_downloads()
