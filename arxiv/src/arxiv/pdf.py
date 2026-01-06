"""
Functions for working with PDFs obtained from arxiv
"""

import re
from pathlib import Path

import fitz
import pymupdf

# catches possible variations
# "Supplementary Materials"
# "Supplemental Materials"
# "Supplemental Material"
_R_HAS_SUPP_MATCH = r"(?i)supplement"


def get_arxiv_pdf_likely_has_supplement(
    arxiv_pdf_path: Path, suppress_error_msg: bool = True
) -> bool:
    """This uses heuristics to determine if a given PDF path has Supplemental material within it.

    :param arxiv_pdf_path: Path to the PDF obtained from arxiv
    :param suppress_error_msg: Whether to silence error messages on malformed PDFs,
        true by default.
    :return: True if it likely has a Supplement Section, False otherwise.
    """
    if suppress_error_msg:
        # https://github.com/pymupdf/PyMuPDF/issues/606
        fitz.TOOLS.mupdf_display_errors(on=False)

    assert isinstance(arxiv_pdf_path, Path)
    assert arxiv_pdf_path.exists() and arxiv_pdf_path.is_file()
    assert arxiv_pdf_path.suffix == ".pdf"

    doc = pymupdf.open(arxiv_pdf_path)
    # https://pymupdf.readthedocs.io/en/latest/document.html#Document.get_toc
    # (lvl, title, page)
    toc = doc.get_toc()

    # check the table of contents to see where the cited papers are
    main_content_ends_at_page = float("inf")
    for section in toc:
        _, title, page_num = section
        title = title.lower().strip()
        if "references" in title:
            main_content_ends_at_page = page_num

    # check for text that could define a supplemental material section
    page_num = 1
    total_pages = doc.page_count
    r = {}
    for page in doc:
        text = page.get_text()
        # finds first occurrence of pattern
        x = re.search(_R_HAS_SUPP_MATCH, text)
        if x is not None:
            r[page_num] = x
        page_num += 1

    # turn errors back on
    fitz.TOOLS.mupdf_display_errors(on=True)

    if not r:
        # never found something that seemed to reference a supplemental section
        return False

    # for all possible references for supplement, check if any occur in the pdf
    # after the references page. This is a strong indicator that the pdf contains
    # a supplement. The supplement section should start, however, before the
    # final page. If it starts on the last page, then the supplemental material
    # is very small and could mean it is an external link to something else
    for page_start in r.keys():
        if total_pages > page_start > main_content_ends_at_page:
            return True

    # none of our heuristics would indicate that this has a supplement in the pdf
    return False
