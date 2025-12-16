import requests
from lxml import html
import pandas as pd

from src.http_util import get_responses

# Unicode to latex
_SYMBOL_MAP = {
    "χ": r"\chi",
}


def _text_to_latex(text: str) -> str:
    if not text:
        return ""
    return "".join(_SYMBOL_MAP.get(ch, ch) for ch in text)


def inline_html_to_latex(e: html.Element) -> str:
    parts: list[str] = [_text_to_latex(e.text)]
    for child in e:
        cls = (child.get("class") or "").split()

        # parse fractions
        if child.tag == "span" and "fraction" in cls:
            top_texts = child.xpath(
                ".//span[contains(concat(' ', normalize-space(@class), ' '), ' top ')]//text()"
            )
            bottom_texts = child.xpath(
                ".//span[contains(concat(' ', normalize-space(@class), ' '), ' bottom ')]//text()"
            )

            num = "".join(t.strip() for t in top_texts)
            den = "".join(t.strip() for t in bottom_texts)
            parts.append(rf"\frac{{{num}}}{{{den}}}")
            parts.append(_text_to_latex(child.tail))
        else:
            # recurse
            parts.append(inline_html_to_latex(child))
            parts.append(_text_to_latex(child.tail))

    return "".join(parts)


def parse_urls_and_text(e: html.HtmlElement) -> tuple[list[str], str]:
    hrefs = [a.get("href") for a in e.xpath(".//a[@href]")]
    hrefs = [x.replace("http://", "https://") for x in hrefs if x]

    e_copy = html.fromstring(html.tostring(e))
    # ignore stuff we collected
    for a in e_copy.xpath(".//a"):
        a.drop_tree()

    text = e_copy.text_content()
    text = " ".join(text.split())
    return hrefs, text


def _parse_simple_table_from_html(my_html: str) -> list[dict[str, str]]:
    r = html.fromstring(my_html)
    e_top_row = r.xpath("//thead")
    top_row = e_top_row[0]
    header = []
    for header_col in top_row.xpath(".//th"):
        header.append(header_col.text)

    e_table_body = r.xpath("//tbody")
    table_body = e_table_body[0]
    rows = []
    for row in table_body.xpath(".//tr"):
        cur_row = {}
        for col_name, col in zip(header, row.xpath(".//td")):
            urls, text = parse_urls_and_text(col)
            cur_row[f"{col_name}_text"] = text
            cur_row[f"{col_name}_urls"] = urls
        rows.append(cur_row)
    return rows


def parse_chi_val_detail_html(my_html: str) -> dict[str, str]:
    # r_my_html = requests.get("https://pppdb.uchicago.edu/chi/entry/37")
    # my_html = r_my_html.text
    r = html.fromstring(my_html)

    # captures 'components' table and 'measurement details' table
    e_components_table = r.xpath('//table[@class="chi-section-table"]')
    tables = []
    for table in e_components_table:
        rows = {}
        for row in table.xpath(".//tr"):
            cols = [x.text_content() for x in row.xpath(".//td")]
            feature_name = cols[0]
            rows[feature_name] = cols[1:]
        tables.append(pd.DataFrame.from_dict(rows))
    df_component_table, df_method_table = tables

    # find the h4 element with content `Chi` and find next div
    chi_div = r.xpath("//h4[b[normalize-space() = 'Chi']]/following-sibling::div[1]")[0]

    # this is better than the detail page
    latex_chi: str = inline_html_to_latex(chi_div)

    return {
        "chi_parsed": latex_chi,
        # super lazy way to transpose this
        "component_table": df_component_table.to_json(orient="records"),
        "method_table": df_method_table.to_json(orient="records"),
    }


def get_chi_vals_html() -> pd.DataFrame:
    r_chi_html = requests.get("https://pppdb.uchicago.edu/chi")
    chi_html = r_chi_html.text

    base_info = _parse_simple_table_from_html(chi_html)
    base_url = "https://pppdb.uchicago.edu/"
    detail_urls = [base_url + x["Info_urls"][0] for x in base_info]

    # each of these has a detail page for some reason
    resps = get_responses(urls=detail_urls)
    c = []
    for row, detail_html in zip(base_info, resps):
        c.append(
            {
                **row,
                **parse_chi_val_detail_html(detail_html),
            }
        )

    df = pd.json_normalize(c)
    df = df.rename(columns={"Reference_urls": "doi"})
    return df


def get_tg_vals_html() -> pd.DataFrame:
    r_tg_html = requests.get("https://pppdb.uchicago.edu/tg")
    tg_html = r_tg_html.text
    df = pd.json_normalize(_parse_simple_table_from_html(tg_html))
    df = df.rename(columns={"Reference_urls": "doi"})
    return df


def get_cloud_points_html() -> pd.DataFrame:
    r_cloud_points_html = requests.get("https://pppdb.uchicago.edu/cloud_points")
    cloud_points_html = r_cloud_points_html.text
    df = pd.json_normalize(_parse_simple_table_from_html(cloud_points_html))
    df = df.rename(columns={"Ref._urls": "doi"})
    return df


def get_pppdb_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    # m2o relation between property and DOI
    df_chi = get_chi_vals_html()
    df_tg = get_tg_vals_html()
    df_cloud_points = get_cloud_points_html()

    all_dois = df_chi["doi"].tolist()
    all_dois += df_tg["doi"].tolist()
    all_dois += df_cloud_points["doi"].tolist()
    all_dois = list(sorted(set([x for b in all_dois for x in b])))
    return df_chi, df_tg, df_cloud_points, all_dois

