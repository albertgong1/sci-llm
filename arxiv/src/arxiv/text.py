import re

SUPP_PATTERNS = {"supplementary", "supplemental", "supporting information"}
POSITIVE_WORDS = {
    "include",
    "includes",
    "included",
    "provide",
    "provides",
    "provided",
    "list",
    "lists",
    "listed",
    "contain",
    "contains",
    "contained",
    "here",
}

NEGATIVE_PATTERNS = {
    "not included",
    "not uploaded",
    "not provided",
    "without",
    "excluding",
    "available upon request",
    "available",
    "on request only",
    "upon request only",
    "on demand",
}


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_clauses(text: str) -> list[str]:
    # Split on punctuation and some conjunctions
    clauses = re.split(r"[.;]|,(?!\d)|\bbut\b|\bhowever\b", text)
    return [c.strip() for c in clauses if c.strip()]


def _has_any_word(text: str, words: set[str]) -> bool:
    return any(re.search(rf"\b{re.escape(w)}\b", text) for w in words)


def _has_negative(text: str) -> bool:
    return any(p in text for p in NEGATIVE_PATTERNS)


def _has_supp_pattern(text: str) -> bool:
    return any(p in text for p in SUPP_PATTERNS)


def _has_positive(text: str) -> bool:
    return _has_any_word(text, POSITIVE_WORDS)


def get_has_supplement_from_arxiv_comment(comment_text: str) -> bool:
    t = _normalize(comment_text)

    clauses = _split_clauses(t)

    any_positive_spec = False
    any_negative_spec = False

    for c in clauses:
        has_supp = _has_supp_pattern(c)
        if not has_supp:
            continue

        neg = _has_negative(c)

        if neg:
            any_negative_spec = True
        else:
            any_positive_spec = True

    if any_negative_spec:
        return False
    if any_positive_spec:
        return True

    # can't get any information from this
    return False
