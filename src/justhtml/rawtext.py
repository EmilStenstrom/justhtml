from __future__ import annotations

from .constants import HTML_SPACE_OR_TAG_END_CHARACTERS


def neutralize_rawtext_end_tag_sequences(text: str, tag_name: str) -> tuple[str, bool]:
    if not text:
        return text, False

    lower_text = text.lower()
    needle = f"</{tag_name}"
    needle_len = len(needle)
    out: list[str] = []
    start = 0
    changed = False

    while True:
        idx = lower_text.find(needle, start)
        if idx == -1:
            break

        boundary = idx + needle_len
        if boundary == len(text) or text[boundary] in HTML_SPACE_OR_TAG_END_CHARACTERS:
            out.append(text[start:idx])
            out.append("&lt;")
            start = idx + 1
            changed = True
            continue

        start = idx + 1

    if not changed:
        return text, False

    out.append(text[start:])
    return "".join(out), True
