import pandas as pd

from src.arxiv.text import get_has_supplement_from_arxiv_comment
from src.util import Printer

p = "/Users/admin/Desktop/sci-llm/arxiv/artifacts/supercon_augmented_search_results_enriched.csv"

if __name__ == "__main__":
    df = pd.read_csv(p)
    comments = df["arxiv_comments"].tolist()
    paper_ids = df["arxiv_url_abstract"].tolist()
    for paper_id, comment in zip(paper_ids, comments):
        if not isinstance(comment, str):
            continue

        result = get_has_supplement_from_arxiv_comment(comment)
        t_result = Printer.green("True") if result else Printer.red("False")
        print(comment, ": || :", paper_id)
        print("----")
