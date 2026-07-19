from __future__ import annotations

SPACE = " \t\n\f\r"
TAG_NAME_STOP = "\t\n\f />"
# In the HTML attribute-name state, quotes, apostrophes, "<", and NUL are
# parse errors but remain part of the attribute name (NUL is replaced later).
ATTR_NAME_STOP = "\t\n\f />="
ATTR_VALUE_STOP = SPACE + ">"
TAG_END_NAME_STOP = SPACE + "/>"


def ascii_startswith(html: str, needle: str, start: int, end: int) -> bool:
    """Return whether an ASCII needle matches case-insensitively at start."""
    stop = start + len(needle)
    if stop > end:
        return False
    candidate = html[start:stop]
    return candidate == needle or (candidate.isascii() and candidate.lower() == needle.lower())


def ascii_find(html: str, needle: str, start: int, end: int) -> int:
    """Find an ASCII needle case-insensitively without folding all of html."""
    first = needle[0]
    needle_lower = needle.lower()
    needle_len = len(needle)
    if "a" <= first <= "z":
        alternate_first = first.upper()
    elif "A" <= first <= "Z":
        alternate_first = first.lower()
    else:
        alternate_first = first
    while True:
        found = html.find(first, start, end)
        if alternate_first != first:
            alternate_found = html.find(alternate_first, start, end)
            if found == -1 or (alternate_found != -1 and alternate_found < found):
                found = alternate_found
        if found == -1:
            return -1
        match_end = found + needle_len
        if match_end > end:
            return -1
        candidate = html[found:match_end]
        if candidate == needle or (candidate.isascii() and candidate.lower() == needle_lower):
            return found
        start = found + 1


def ascii_rfind(html: str, needle: str, start: int, end: int) -> int:
    """Find the last ASCII needle case-insensitively without a folded copy."""
    first = needle[0]
    if "a" <= first <= "z":
        alternate_first = first.upper()
    elif "A" <= first <= "Z":
        alternate_first = first.lower()
    else:
        alternate_first = first
    search_end = end
    while True:
        found = html.rfind(first, start, search_end)
        if alternate_first != first:
            alternate_found = html.rfind(alternate_first, start, search_end)
            if alternate_found > found:
                found = alternate_found
        if found == -1:
            return -1
        if ascii_startswith(html, needle, found, end):
            return found
        search_end = found


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


def find_rawtext_end_tag(html: str, name: str, pos: int, end: int) -> tuple[int | None, int]:
    needle = f"</{name}"
    needle_len = len(needle)
    search = pos
    while True:
        close = ascii_find(html, needle, search, end)
        if close == -1:
            return None, end
        after_name = close + needle_len
        if after_name < end and html[after_name] not in TAG_END_NAME_STOP:
            search = after_name
            continue
        tag_end = find_tag_end(html, after_name, end)
        if tag_end == -1:
            if after_name < end:
                # A space or "/" terminated the end-tag name but no ">" follows.
                # At EOF the end tag is still emitted (its attributes dropped), so
                # the raw text ends here. Only a bare "</name" at EOF, with no
                # terminator, stays raw text.
                return close, end
            search = after_name
            continue
        return close, tag_end + 1


def find_script_end_tag(html: str, pos: int, end: int) -> tuple[int | None, int]:
    search = pos
    escaped = False
    double_escaped = False

    while True:
        close, next_pos = find_rawtext_end_tag(html, "script", search, end)
        if close is None:
            return None, end
        if not escaped:
            comment_start = html.find("<!--", search, close)
            if comment_start == -1 or close < comment_start:
                return close, next_pos
            escaped = True
            # After "<!--" the tokenizer is in the script-data-escaped-dash-dash
            # state, so the two trailing dashes already count toward a closing
            # "-->". Resume the comment-end search at those dashes so "<!-->"
            # leaves the escaped state instead of being treated as still open.
            search = comment_start + 2
            continue

        script_start = find_script_start_marker(html, search, close, end) if not double_escaped else -1
        comment_end = html.find("-->", search, close)
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
            later_end = ascii_find(html, "</script", close + 8, next_pos)
            if later_end != -1:
                after_name = later_end + 8
                if after_name >= end or html[after_name] in TAG_END_NAME_STOP:
                    double_escaped = False
                    search = close + 8
                    continue
            # No earlier terminated "</script" can precede `close`: find_rawtext_end_tag
            # already returns the first terminated end tag, so the region between
            # `search` and `close` holds only non-terminated occurrences.
            double_escaped = False
            search = next_pos
            continue
        return close, next_pos


def find_script_start_marker(html: str, pos: int, end: int, text_end: int) -> int:
    # `end` bounds where a "<script" marker may start; `text_end` bounds the real
    # input so the script-data-double-escape-start terminator is checked against
    # the actual following character. When these differ, a marker whose name ends
    # exactly at `end` still has a following character (e.g. the "<" of a trailing
    # "</script>") that must be inspected rather than treated as end-of-input.
    search = pos
    needle = "<script"
    needle_len = len(needle)
    while True:
        start = ascii_find(html, needle, search, end)
        if start == -1:
            return -1
        after_name = start + needle_len
        if after_name >= text_end or html[after_name] in TAG_END_NAME_STOP:
            return start
        search = after_name
