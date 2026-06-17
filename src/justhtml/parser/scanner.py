from __future__ import annotations

SPACE = " \t\n\f\r"
TAG_NAME_STOP = "\t\n\f />"
ATTR_NAME_STOP = "\t\n\f />=\0\"'<"
ATTR_VALUE_STOP = SPACE + ">"
TAG_END_NAME_STOP = SPACE + "/>"


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
