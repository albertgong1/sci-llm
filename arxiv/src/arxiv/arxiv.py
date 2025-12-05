from textwrap import dedent
import urllib.parse
from tqdm import tqdm


from lxml import html

from src.scraping.curl_impersonate import get_webpage, CurlResponse


class ReCaptchaExpiredException(Exception):
    pass


def _parse_arxiv_search_results(my_html: str) -> list[dict[str, str]]:
    r = html.fromstring(my_html)

    e_title = r.xpath("//title")
    if e_title:
        title = e_title[0].text
        if title == "arXiv reCAPTCHA":
            raise ReCaptchaExpiredException("arXiv reCAPTCHA requests a refresh.")

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
) -> list[list[dict[str, str]]]:
    results = []
    captcha_auth = ""
    success = 0
    for doi in tqdm(true_dois):
        while True:
            try:
                parsed = _get_arxiv_search_results_by_doi(captcha_auth, doi)
                if len(parsed):
                    success += 1
                    print(success)
                # append even empty list
                results.append(parsed)
                break
            except ReCaptchaExpiredException:
                # captcha expires every ~1,000 entries or so
                print(f"Go to: https://arxiv.org/")
                captcha_auth = input("Recaptcha expired, enter value: ").strip()
    return results
