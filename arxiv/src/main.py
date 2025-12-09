import csv
from pathlib import Path

from tqdm import tqdm
import pandas as pd

from src.arxiv import (
    get_arxiv_search_results_by_dois,
    get_arxiv_paper_detail_result_by_abs_links,
    get_has_supplement_from_arxiv_comment,
    get_arxiv_pdf_likely_has_supplement,
)
from src.config import ARTIFACTS_ROOT, REPO_ROOT
from src.doi import get_proper_doi_from_doi_urls
from src.http_util import get_responses
from src.google_util.sheets import (
    GoogleSheetWithGoogleDriveLinks,
    GoogleSheetFileLinkConfig,
)
from src.util import get_path_with_suffix


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


def get_df_with_filenames_and_supplement_heuristic(
    all_pdfs_path: Path, orig_df: pd.DataFrame
) -> pd.DataFrame:
    lookup_by_id = {}
    for pdf_path in all_pdfs_path.glob("*.pdf"):
        pdf_prefix: str = pdf_path.stem
        lookup_by_id[pdf_prefix] = pdf_path

    base_df = orig_df
    result_list_of_dicts = base_df.to_dict(orient="records")
    for e in tqdm(result_list_of_dicts):
        pdf_url = e["arxiv_url_pdf"]

        # defaults
        e["arxiv_pdf_suggests_has_supplement"] = False
        e["arxiv_id"] = ""
        e["file_path"] = ""

        if not isinstance(pdf_url, str):
            # is NaN
            continue

        arxiv_id: str = pdf_url.split("/")[-1].strip()
        pdf_path_on_disk = lookup_by_id.get(arxiv_id, None)
        if pdf_path_on_disk and isinstance(pdf_path_on_disk, Path):
            pdf_has_supp = get_arxiv_pdf_likely_has_supplement(pdf_path_on_disk)
        else:
            pdf_has_supp = False

        e["arxiv_pdf_suggests_has_supplement"] = pdf_has_supp
        e["arxiv_id"] = arxiv_id
        e["file_path"] = pdf_path_on_disk

    return pd.json_normalize(result_list_of_dicts)


def get_super_con_arxiv_info(testing_limit: int | None = None) -> None:
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
            to_save: dict[str, str] = {
                **paper_info,
                **arxiv_search_result,
            }
            normed.append(to_save)

    # save just in case something goes wrong in the next step
    df_intermediate = pd.json_normalize(normed)
    df_intermediate.to_csv(
        ARTIFACTS_ROOT / "supercon_augmented_search_results.csv", index=False
    )

    for i, paper_result in enumerate(normed):
        paper_details = get_arxiv_paper_detail_result_by_abs_links(
            [paper_result["arxiv_url_abstract"]]
        )[0]
        normed[i] = {
            **paper_result,
            **paper_details,
            "arxiv_comment_suggests_has_supplement": get_has_supplement_from_arxiv_comment(
                paper_details["arxiv_comments"]
            ),
        }

    # will overwrite the intermediate df
    df = pd.json_normalize(normed)
    df.to_csv(ARTIFACTS_ROOT / "supercon_augmented_search_results.csv", index=False)


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
    orig_df_path: Path = ARTIFACTS_ROOT / "SuperConDOI.csv",
) -> Path:
    # for all the PDF urls, download them
    df = pd.read_csv(saved_csv_path)
    pdf_links = [
        x for x in df["arxiv_url_pdf"].tolist() if isinstance(x, str) and bool(x)
    ]
    save_to = saved_csv_path.parent / saved_csv_path.stem
    save_to.mkdir(exist_ok=True, parents=False)
    get_responses(pdf_links, download_to_folder=save_to)

    # now that they are on disk, inspect their contents to determine if
    # they contain supplemental info
    df = get_df_with_filenames_and_supplement_heuristic(save_to, df)

    # load original corpus
    orig_df = pd.read_csv(orig_df_path)
    orig_df.columns = [x.lower() for x in orig_df.columns]

    # join with orig corpus and save search results with new columns as file
    save_to = get_path_with_suffix(saved_csv_path, "google_sheets")
    df = orig_df.merge(df, how="left", on="paper_id")
    df.to_csv(save_to, index=False)
    return save_to


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


def upload_super_con_to_google_sheets(saved_csv_path: Path) -> None:
    df = pd.read_csv(saved_csv_path)
    sheet = GoogleSheetWithGoogleDriveLinks(
        # https://developers.google.com/workspace/sheets/api/quickstart/python?authuser=3#authorize_credentials_for_a_desktop_application
        # obtain the credentials file from google developer console
        credentials_json_file_path=REPO_ROOT / "credentials.json",
        # token.json does not need to exist, but it can
        token_json_file_path=REPO_ROOT / "token.json",
    )
    folder_id = sheet.create_folder_in_google_drive("SuperCon Arxiv Papers")
    spreadsheet_id = sheet.create_google_sheet_from_pandas_df(
        df,
        "SuperCon Arxiv",
        folder_id,
        GoogleSheetFileLinkConfig(
            col_name_of_readable_name="paper_id", col_name_of_local_filepath="file_path"
        ),
    )
    print(f"Drive folder ID: {folder_id}")
    print(f"Sheet URL: https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit")


if __name__ == "__main__":
    pass
    # --- supercon ---
    # get_super_con_arxiv_info()
    # g_search_results = get_super_con_arxiv_info_downloads()
    # upload_super_con_to_google_sheets(g_search_results)

    # --- charge density ---
    # get_charge_density_wave_arxiv_info()
    # get_super_con_arxiv_info_downloads()
    # get_charge_density_wave_arxiv_downloads()
