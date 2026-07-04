from __future__ import annotations

import string

SPACE = " \t\n\f\r"
TAG_NAME_STOP = "\t\n\f />"
# In the HTML attribute-name state, quotes, apostrophes, "<", and NUL are
# parse errors but remain part of the attribute name (NUL is replaced later).
ATTR_NAME_STOP = "\t\n\f />="
ATTR_VALUE_STOP = SPACE + ">"
TAG_END_NAME_STOP = SPACE + "/>"

_ASCII_LOWER_TABLE = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)


def ascii_lower(html: str) -> str:
    """Fold the input for ASCII case-insensitive scanning, index-aligned.

    The folded copy is searched for ASCII needles with positions taken from
    the original string, so the fold must behave like lowercasing only A-Z
    (markup matching is ASCII case-insensitive per §13.2.5) and must be
    length-preserving. str.lower() alone is neither: "İ" (U+0130) lowers to
    two characters, shifting every later index, and U+212A KELVIN SIGN lowers
    to "k", over-matching ASCII needles.

    str.lower() is still used as the fast path: when it preserves the total
    length, every character mapped one-to-one, so indexes stay aligned, and
    with U+212A absent no non-ASCII character can produce an ASCII needle
    character. The strict A-Z table fold only runs for the rare inputs that
    fail those checks.
    """
    if html.isascii():
        return html.lower()
    lowered = html.lower()
    if len(lowered) != len(html) or "\u212a" in html:
        return html.translate(_ASCII_LOWER_TABLE)
    return lowered


def find_tag_end(html: str, pos: int, end: int) -> int:
    quote: str | None = None
    while pos < end:
        ch = html[pos]
        if quote is not None:
            if ch == quote:
                quote = None
        elif ch == '"' or ch == "'":
            quote = ch
        elif ch == ">":
            return pos
        pos += 1
    return -1


def find_rawtext_end_tag(html: str, lower: str, name: str, pos: int, end: int) -> tuple[int | None, int]:
    needle = f"</{name}"
    needle_len = len(needle)
    search = pos
    while True:
        close = lower.find(needle, search, end)
        if close == -1:
            return None, end
        after_name = close + needle_len
        if after_name < end and html[after_name] not in TAG_END_NAME_STOP:
            search = after_name
            continue
        tag_end = find_tag_end(html, after_name, end)
        if tag_end == -1:
            if after_name < end and not html[after_name:end].strip(SPACE + "/"):
                return close, end
            search = after_name
            continue
        return close, tag_end + 1


def find_script_end_tag(html: str, lower: str, pos: int, end: int) -> tuple[int | None, int]:
    search = pos
    escaped = False
    double_escaped = False

    while True:
        close, next_pos = find_rawtext_end_tag(html, lower, "script", search, end)
        if close is None:
            return None, end
        if not escaped:
            comment_start = lower.find("<!--", search, close)
            if comment_start == -1 or close < comment_start:
                return close, next_pos
            escaped = True
            search = comment_start + 4
            continue

        script_start = find_script_start_marker(html, lower, search, close) if not double_escaped else -1
        comment_end = lower.find("-->", search, close)
        if (
            comment_end != -1
            and comment_end < close
            and (double_escaped or script_start == -1 or comment_end < script_start)
        ):
            escaped = False
            double_escaped = False
            search = comment_end + 3
            continue
        if script_start != -1 and script_start < close:
            double_escaped = True
            search = script_start + 7
            continue
        if double_escaped:
            later_end = lower.find("</script", close + 8, next_pos)
            if later_end != -1:
                after_name = later_end + 8
                if after_name >= end or html[after_name] in TAG_END_NAME_STOP:
                    double_escaped = False
                    search = close + 8
                    continue
            double_end = lower.find("</script", search, close)
            if double_end != -1:
                after_name = double_end + 8
                if after_name >= end or html[after_name] in TAG_END_NAME_STOP:
                    double_escaped = False
                    search = after_name
                    continue
            double_escaped = False
            search = next_pos
            continue
        return close, next_pos


def find_script_start_marker(html: str, lower: str, pos: int, end: int) -> int:
    search = pos
    needle = "<script"
    needle_len = len(needle)
    while True:
        start = lower.find(needle, search, end)
        if start == -1:
            return -1
        after_name = start + needle_len
        if after_name >= end or html[after_name] in TAG_END_NAME_STOP:
            return start
        search = after_name
