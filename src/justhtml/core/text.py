from __future__ import annotations

import string

_ASCII_LOWER_TABLE = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)


def ascii_lower(text: str) -> str:
    """Fold A-Z for ASCII marker searches that reuse original offsets.

    Some parser and serializer paths search a folded copy of full text for
    ASCII HTML markers such as ``</script``, ``<!doctype``, and ``<![cdata[``.
    Match offsets are then reused against the original string, so the folded
    copy must preserve length, and it must not invent ASCII marker characters
    from non-ASCII text content. HTML markup matching is ASCII
    case-insensitive; it is not Unicode case-insensitive.

    ``str.lower()`` is still the fast path when it is provably equivalent for
    these searches. In Python's Unicode data, U+0130 LATIN CAPITAL LETTER I
    WITH DOT ABOVE is the only character whose lowercase mapping changes
    length, and U+212A KELVIN SIGN is the only non-ASCII character whose
    lowercase mapping becomes an ASCII letter (``k``). Inputs containing either
    shape take the strict A-Z translation path. That path deliberately keeps
    U+212A as U+212A; it is not a tag-name fold.
    """
    if text.isascii():
        return text.lower()
    if "\u0130" in text or "\u212a" in text:
        return text.translate(_ASCII_LOWER_TABLE)
    return text.lower()
