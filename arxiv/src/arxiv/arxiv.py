"""
Functions for working with the website, arxiv.org
"""

from textwrap import dedent
import urllib.parse

from tqdm import tqdm
from lxml import html

from src.scraping.curl_impersonate import get_webpage, CurlResponse


class ReCaptchaExpiredException(Exception):
    pass


def _check_arxiv_recaptcha_page(p_html) -> None:
    e_title = p_html.xpath("//title")
    if e_title:
        title = e_title[0].text
        if title == "arXiv reCAPTCHA":
            raise ReCaptchaExpiredException("arXiv reCAPTCHA requests a refresh.")
    return


def _parse_arxiv_search_results(my_html: str) -> list[dict[str, str]]:
    r = html.fromstring(my_html)

    _check_arxiv_recaptcha_page(r)

    # get the container of search results
    e_list_container = r.xpath('//ol[@class="breathe-horizontal"]')
    if not e_list_container:
        return []

    assert len(e_list_container) == 1
    list_container = e_list_container[0]

    results = []
    for search_item in list_container.xpath('//li[@class="arxiv-result"]'):
        # XPath syntax reminder:
        # ./ immediate children only
        # .// any subtree (any child, allows recursion)
        # // global, do not use on found elements, searches entire doc

        e_arxiv_id = search_item.xpath('.//p[@class="list-title is-inline-block"]//a')
        header_links = [x.attrib["href"] for x in e_arxiv_id]
        if len(header_links) > 3:
            header_links = header_links[:3]
        elif len(header_links) < 3:
            header_links += [None] * (3 - len(header_links))
        abstract_link, pdf_link, html_link = header_links

        e_titles = search_item.xpath('.//p[contains(@class, "title")]')
        # title may have latex / mathjax-like symbols in it, e.g. $_2$, etc.
        title = " ".join([(x.text or "").strip() for x in e_titles])

        # not collecting their search page, is useless
        e_authors = search_item.xpath('.//p[@class="authors"]')
        assert len(e_authors) == 1
        author_names = [x.text for x in e_authors[0].findall("a")]

        # get the abstract that is displayed
        e_abstract = search_item.xpath(
            ".//span[contains(@class, 'abstract') and contains(@style, 'display: inline')]"
        )
        assert len(e_abstract) == 1
        abstract_text = e_abstract[0].text.strip()

        list_elem = {
            "arxiv_title": title,
            "arxiv_authors": author_names,
            "arxiv_abstract": abstract_text,
            "arxiv_url_abstract": abstract_link,
            "arxiv_url_pdf": pdf_link,
            "arxiv_url_html": html_link,
        }
        results.append(list_elem)
    return results


def _get_curl_for_doi(true_doi: str, captcha_auth: str) -> str:
    enc_true_doi = urllib.parse.quote_plus(true_doi)
    # curl impersonate command
    curl_base = f"""
    'https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term={enc_true_doi}&terms-0-field=doi&classification-physics_archives=all&classification-include_cross_list=include&date-filter_by=all_dates&date-year=&date-from_date=&date-to_date=&date-date_type=submitted_date&abstracts=show&size=50&order=-announced_date_first' \
      -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7' \
      -H 'accept-language: en-US,en;q=0.9' \\
      -H 'cache-control: max-age=0' \
      -b 'arxiv-search-parameters="{{\\"order\\": \\"-announced_date_first\\"\\054 \\"size\\": \\"50\\"\\054 \\"abstracts\\": \\"show\\"\\054 \\"date-date_type\\": \\"submitted_date\\"}}"; captchaAuth={captcha_auth}' \
      -H 'priority: u=0, i' \\
      -H 'referer: https://arxiv.org/search/advanced' \\
      -H 'sec-ch-ua: "Chromium";v="142", "Google Chrome";v="142", "Not_A Brand";v="99"' \\
      -H 'sec-ch-ua-mobile: ?0' \\
      -H 'sec-ch-ua-platform: "macOS"' \\
      -H 'sec-fetch-dest: document' \\
      -H 'sec-fetch-mode: navigate' \\
      -H 'sec-fetch-site: same-origin' \\
      -H 'sec-fetch-user: ?1' \\
      -H 'upgrade-insecure-requests: 1' \\
      -H 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36' 
    """.strip()
    return dedent(curl_base)


def _get_arxiv_search_results_by_doi(captcha_auth: str, true_doi: str):
    curl_cmd = _get_curl_for_doi(true_doi, captcha_auth)
    result: CurlResponse = get_webpage(curl_cmd)
    parsed: list[dict[str, str]] = _parse_arxiv_search_results(result.text)
    assert parsed is not None
    return parsed


def get_arxiv_search_results_by_dois(
    true_dois: list[str],
    captcha_auth: str = "",
) -> list[list[dict[str, str]]]:
    results = []
    # captcha only required if you make a large volume of requests
    _captcha_auth = captcha_auth
    success = 0
    for doi in tqdm(true_dois):
        while True:
            try:
                parsed = _get_arxiv_search_results_by_doi(_captcha_auth, doi)
                if len(parsed):
                    success += 1
                    print(success)
                # append even empty list
                results.append(parsed)
                break
            except ReCaptchaExpiredException:
                # captcha expires every ~1,000 entries or so
                print(f"Go to: https://arxiv.org/")
                _captcha_auth = input("Recaptcha expired, enter value: ").strip()
    return results


def get_arxiv_abs_link_from_pdf_link(arxiv_pdf_link: str) -> str:
    assert arxiv_pdf_link and isinstance(arxiv_pdf_link, str)
    assert arxiv_pdf_link.startswith("https://arxiv.org/pdf/")
    arxiv_id = arxiv_pdf_link.split("pdf/")[-1]
    return f"https://arxiv.org/abs/{arxiv_id}"


def _parse_arxiv_detail_to_json(abs_link: str, arxiv_html: str) -> dict[str, str]:
    r = html.fromstring(arxiv_html)
    _check_arxiv_recaptcha_page(r)

    e_metatable = r.xpath('//div[@class="metatable"]')
    assert e_metatable and len(e_metatable) == 1
    metatable = e_metatable[0]

    # Comments
    comments = ""
    e_comments = metatable.xpath('.//td[contains(@class, "tablecell comments")]')
    if e_comments:
        comments = e_comments[0].text_content()

    # Journal Reference
    journal_reference = ""
    e_journal_reference = metatable.xpath('.//td[contains(@class, "tablecell jref")]')
    if e_journal_reference:
        journal_reference = e_journal_reference[0].text_content()

    # Related DOI
    related_doi = ""
    e_related_doi = metatable.xpath('.//td[contains(@class, "tablecell doi")]')
    if e_related_doi:
        related_doi = e_related_doi[0].xpath(".//a")[0].text_content()

    return {
        "arxiv_url_abstract": abs_link,
        # this can contain information on what the PDF contains. Sometimes it is blank.
        # simple string matching to determine if content is or is not present will not work.
        # some examples of what might be written here are:
        # - "15 pages, 4 figures, Supplementary materials"
        # - "5 + epsilon pages, 7 figures, Supplemental Material on demand; v2 includes ..."
        # - "5 pages, 3 figures. The Supplementary Information file is available upon request"
        # - "11 pages, 4 figures, supplementary information is not uploaded"
        "arxiv_comments": comments,
        "arxiv_journal_reference": journal_reference,
        "arxiv_related_doi": related_doi,
    }


def _get_curl_for_detail(abs_url: str, captcha_auth: str) -> str:
    # curl impersonate command
    curl_base = f"""
    '{abs_url}' \\
    -X 'GET' \
    -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' \
    -H 'Sec-Fetch-Site: none' \
    -H 'Cookie: captchaAuth={captcha_auth}; arxiv_labs={{%22sameSite%22:%22strict%22%2C%22expires%22:365%2C%22last_tab%22:%22tabone%22}}; arxiv-search-parameters="{{\\"order\\": \\"-announced_date_first\\"\\054 \\"size\\": \\"50\\"\\054 \\"abstracts\\": \\"show\\"\\054 \\"date-date_type\\": \\"submitted_date\\"}}"; browser=73.186.99.152.1730915620264849' \\
    -H 'Referer: https://arxiv.org/abs/1903.05679' \
    -H 'Sec-Fetch-Mode: navigate' \
    -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.6 Safari/605.1.15' \
    -H 'Accept-Language: en-US,en;q=0.9' \
    -H 'Sec-Fetch-Dest: document' \
    -H 'Accept-Encoding: gzip, deflate, br' \
    -H 'Priority: u=0, i'
    """.strip()

    # user agent here uses Safari, so it technically will be a mismatch between
    # curl impersonate and reported UA. This is fine for now though.
    return dedent(curl_base)


def _get_arxiv_detail_by_abs_url(captcha_auth: str, abs_url: str) -> dict[str, str]:
    curl_cmd = _get_curl_for_detail(abs_url, captcha_auth)
    result: CurlResponse = get_webpage(curl_cmd)
    parsed: dict[str, str] = _parse_arxiv_detail_to_json(abs_url, result.text)
    assert parsed is not None
    return parsed


def get_arxiv_paper_detail_result_by_abs_links(
    abs_urls: list[str],
    captcha_auth: str = "",
) -> list[dict[str, str]]:
    results = []
    # captcha only required if you make a large volume of requests
    _captcha_auth = captcha_auth
    for abs_url in tqdm(abs_urls):
        while True:
            try:
                # expect that, with fresh captcha ALL these requests should succeed
                parsed = _get_arxiv_detail_by_abs_url(_captcha_auth, abs_url)
                assert parsed
                results.append(parsed)
                break
            except ReCaptchaExpiredException:
                # captcha expires every ~1,000 entries or so
                print(f"Go to: https://arxiv.org/")
                _captcha_auth = input("Recaptcha expired, enter value: ").strip()
    return results
