from __future__ import annotations

from typing import TYPE_CHECKING

from justhtml.core.constants import (
    HTML4_PUBLIC_PREFIXES,
    LIMITED_QUIRKY_PUBLIC_PREFIXES,
    QUIRKY_PUBLIC_MATCHES,
    QUIRKY_PUBLIC_PREFIXES,
    QUIRKY_SYSTEM_MATCHES,
)

if TYPE_CHECKING:
    from justhtml.core.types import Doctype


def contains_prefix(haystack: tuple[str, ...], needle: str) -> bool:
    return any(needle.startswith(prefix) for prefix in haystack)


def doctype_error_and_quirks(doctype: Doctype, iframe_srcdoc: bool = False) -> tuple[bool, str]:
    name = doctype.name.lower() if doctype.name else None
    public_id = doctype.public_id
    system_id = doctype.system_id

    acceptable: tuple[tuple[str | None, str | None, str | None], ...] = (
        ("html", None, None),
        ("html", None, "about:legacy-compat"),
        ("html", "-//W3C//DTD HTML 4.0//EN", None),
        ("html", "-//W3C//DTD HTML 4.0//EN", "http://www.w3.org/TR/REC-html40/strict.dtd"),
        ("html", "-//W3C//DTD HTML 4.01//EN", None),
        ("html", "-//W3C//DTD HTML 4.01//EN", "http://www.w3.org/TR/html4/strict.dtd"),
        ("html", "-//W3C//DTD XHTML 1.0 Strict//EN", "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd"),
        ("html", "-//W3C//DTD XHTML 1.1//EN", "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd"),
    )

    key = (name, public_id, system_id)
    parse_error = key not in acceptable

    public_lower = public_id.lower() if public_id else None
    system_lower = system_id.lower() if system_id else None

    quirks_mode: str
    if doctype.force_quirks:
        quirks_mode = "quirks"
    elif iframe_srcdoc:
        quirks_mode = "no-quirks"
    elif name != "html":
        quirks_mode = "quirks"
    elif public_lower in QUIRKY_PUBLIC_MATCHES:
        quirks_mode = "quirks"
    elif system_lower in QUIRKY_SYSTEM_MATCHES:
        quirks_mode = "quirks"
    elif public_lower and contains_prefix(QUIRKY_PUBLIC_PREFIXES, public_lower):
        quirks_mode = "quirks"
    elif public_lower and contains_prefix(LIMITED_QUIRKY_PUBLIC_PREFIXES, public_lower):
        quirks_mode = "limited-quirks"
    elif public_lower and contains_prefix(HTML4_PUBLIC_PREFIXES, public_lower):
        quirks_mode = "quirks" if system_lower is None else "limited-quirks"
    else:
        quirks_mode = "no-quirks"

    return parse_error, quirks_mode
