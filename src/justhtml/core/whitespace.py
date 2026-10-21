"""HTML/CSS whitespace operations, preserving other Unicode space characters."""

import re

_HTML_TOKEN_RE = re.compile(r"[^\t\n\f\r ]+")


def split_html_whitespace(value: str) -> list[str]:
    """Split on TAB, LF, FF, CR and SPACE, as defined by Infra."""
    # SPACE is the only printable whitespace character. On printable text the
    # native splitter is exact, including text containing non-ASCII letters.
    if value.isprintable():
        return value.split()
    return _HTML_TOKEN_RE.findall(value)
