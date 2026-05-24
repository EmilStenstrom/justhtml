"""HTML encoding sniffing and decoding.

Implements the HTML encoding sniffing behavior needed for the html5lib-tests
encoding fixtures.

Inputs are bytes and an optional transport-supplied encoding label.
Outputs are a decoded Unicode string and the chosen encoding name.
"""

from __future__ import annotations

_ASCII_WHITESPACE: set[int] = {0x09, 0x0A, 0x0C, 0x0D, 0x20}


def _ascii_lower(b: int) -> int:
    # b is an int 0..255
    if 0x41 <= b <= 0x5A:
        return b | 0x20
    return b


def _is_ascii_alpha(b: int) -> bool:
    b = _ascii_lower(b)
    return 0x61 <= b <= 0x7A


def _skip_ascii_whitespace(data: bytes, i: int) -> int:
    n = len(data)
    while i < n and data[i] in _ASCII_WHITESPACE:
        i += 1
    return i


def _strip_ascii_whitespace(value: bytes | None) -> bytes | None:
    if value is None:
        return None
    start = 0
    end = len(value)
    while start < end and value[start] in _ASCII_WHITESPACE:
        start += 1
    while end > start and value[end - 1] in _ASCII_WHITESPACE:
        end -= 1
    return value[start:end]


def normalize_encoding_label(label: str | bytes | None) -> str | None:
    if not label:
        return None

    if isinstance(label, bytes):
        label = label.decode("ascii", "ignore")

    s = str(label).strip()
    if not s:
        return None

    s = s.lower()

    # Security: never allow utf-7.
    if s in {"utf-7", "utf7", "x-utf-7"}:
        return "windows-1252"

    if s in {"utf-8", "utf8"}:
        return "utf-8"

    # HTML treats latin-1 labels as windows-1252.
    if s in {
        "iso-8859-1",
        "iso8859-1",
        "latin1",
        "latin-1",
        "l1",
        "cp819",
        "ibm819",
    }:
        return "windows-1252"

    if s in {"windows-1252", "windows1252", "cp1252", "x-cp1252"}:
        return "windows-1252"

    if s in {"iso-8859-2", "iso8859-2", "latin2", "latin-2"}:
        return "iso-8859-2"

    if s in {"euc-jp", "eucjp"}:
        return "euc-jp"

    if s in {"utf-16", "utf16"}:
        return "utf-16"
    if s in {"utf-16le", "utf16le"}:
        return "utf-16le"
    if s in {"utf-16be", "utf16be"}:
        return "utf-16be"

    return None


def _normalize_meta_declared_encoding(label: bytes | None) -> str | None:
    enc = normalize_encoding_label(label)
    if enc is None:
        return None

    # Per HTML meta charset handling: ignore UTF-16/UTF-32 declarations and
    # treat them as UTF-8.
    if enc in {"utf-16", "utf-16le", "utf-16be", "utf-32", "utf-32le", "utf-32be"}:
        return "utf-8"

    return enc


def _sniff_bom(data: bytes) -> tuple[str | None, int]:
    if len(data) >= 3 and data[0:3] == b"\xef\xbb\xbf":
        return "utf-8", 3
    if len(data) >= 2 and data[0:2] == b"\xff\xfe":
        return "utf-16le", 2
    if len(data) >= 2 and data[0:2] == b"\xfe\xff":
        return "utf-16be", 2
    return None, 0


def _extract_charset_from_content(content_bytes: bytes) -> bytes | None:
    if not content_bytes:
        return None

    # Normalize whitespace to spaces for robust matching.
    b = bytearray()
    for ch in content_bytes:
        if ch in _ASCII_WHITESPACE:
            b.append(0x20)
        else:
            b.append(_ascii_lower(ch))
    s = bytes(b)

    idx = s.find(b"charset")
    if idx == -1:
        return None

    i = idx + len(b"charset")
    n = len(s)
    while i < n and s[i] in _ASCII_WHITESPACE:
        i += 1
    if i >= n or s[i] != 0x3D:  # '='
        return None
    i += 1
    while i < n and s[i] in _ASCII_WHITESPACE:
        i += 1
    if i >= n:
        return None

    quote: int | None = None
    if s[i] in (0x22, 0x27):  # '"' or "'"
        quote = s[i]
        i += 1

    start = i
    while i < n:
        ch = s[i]
        if quote is not None:
            if ch == quote:
                break
        else:
            if ch in _ASCII_WHITESPACE or ch == 0x3B:  # ';'
                break
        i += 1

    if quote is not None and (i >= n or s[i] != quote):
        return None

    return s[start:i]


def _prescan_for_meta_charset(data: bytes) -> str | None:
    # Scan up to 1024 bytes worth of non-comment input, but allow skipping
    # arbitrarily large comments (bounded by a hard cap).
    max_non_comment = 1024
    max_total_scan = 65536

    n = len(data)
    i = 0
    non_comment = 0

    while i < n and i < max_total_scan and non_comment < max_non_comment:
        if data[i] != 0x3C:  # '<'
            i += 1
            non_comment += 1
            continue

        # Comment
        if i + 3 < n and data[i + 1 : i + 4] == b"!--":
            end = data.find(b"-->", i + 4)
            if end == -1:
                return None
            i = end + 3
            continue

        # Tag open
        j = i + 1
        if j < n and data[j] == 0x2F:  # '/'
            # Skip end tag.
            k = i
            quote: int | None = None
            while k < n and k < max_total_scan and non_comment < max_non_comment:
                ch = data[k]
                if quote is None:
                    if ch in (0x22, 0x27):
                        quote = ch
                    elif ch == 0x3E:  # '>'
                        k += 1
                        non_comment += 1
                        break
                else:
                    if ch == quote:
                        quote = None
                k += 1
                non_comment += 1
            i = k
            continue

        if j >= n or not _is_ascii_alpha(data[j]):
            i += 1
            non_comment += 1
            continue

        name_start = j
        while j < n and _is_ascii_alpha(data[j]):
            j += 1

        tag_name = data[name_start:j]
        if tag_name.lower() != b"meta":
            # Skip the rest of this tag so we don't accidentally interpret '<'
            # inside an attribute value as a new tag.
            k = i
            quote = None
            while k < n and k < max_total_scan and non_comment < max_non_comment:
                ch = data[k]
                if quote is None:
                    if ch in (0x22, 0x27):
                        quote = ch
                    elif ch == 0x3E:  # '>'
                        k += 1
                        non_comment += 1
                        break
                else:
                    if ch == quote:
                        quote = None
                k += 1
                non_comment += 1
            i = k
            continue

        # Parse attributes until '>'
        charset: bytes | None = None
        http_equiv: bytes | None = None
        content: bytes | None = None

        k = j
        saw_gt = False
        start_i = i
        while k < n and k < max_total_scan:
            ch = data[k]
            if ch == 0x3E:  # '>'
                saw_gt = True
                k += 1
                break

            if ch == 0x3C:  # '<' - restart scanning from here
                break

            if ch in _ASCII_WHITESPACE or ch == 0x2F:  # '/'
                k += 1
                continue

            # Attribute name
            attr_start = k
            while k < n:
                ch = data[k]
                if ch in _ASCII_WHITESPACE or ch in {0x3D, 0x3E, 0x2F, 0x3C}:
                    break
                k += 1
            attr_name = data[attr_start:k].lower()
            k = _skip_ascii_whitespace(data, k)

            value: bytes | None = None
            if k < n and data[k] == 0x3D:  # '='
                k += 1
                k = _skip_ascii_whitespace(data, k)
                if k >= n:
                    break

                quote = None
                if data[k] in (0x22, 0x27):
                    quote = data[k]
                    k += 1
                    val_start = k
                    end_quote = data.find(bytes((quote,)), k)
                    if end_quote == -1:
                        # Unclosed quote: ignore this meta.
                        i += 1
                        non_comment += 1
                        charset = None
                        http_equiv = None
                        content = None
                        saw_gt = False
                        break
                    value = data[val_start:end_quote]
                    k = end_quote + 1
                else:
                    val_start = k
                    while k < n:
                        ch = data[k]
                        if ch in _ASCII_WHITESPACE or ch in {0x3E, 0x3C}:
                            break
                        k += 1
                    value = data[val_start:k]

            if attr_name == b"charset":
                charset = _strip_ascii_whitespace(value)
            elif attr_name == b"http-equiv":
                http_equiv = value
            elif attr_name == b"content":
                content = value

        if saw_gt:
            if charset:
                enc = _normalize_meta_declared_encoding(charset)
                if enc:
                    return enc

            if http_equiv and http_equiv.lower() == b"content-type" and content:
                extracted = _extract_charset_from_content(content)
                if extracted:
                    enc = _normalize_meta_declared_encoding(extracted)
                    if enc:
                        return enc

            # Continue scanning after this tag.
            i = k
            consumed = i - start_i
            non_comment += consumed
        else:
            # Continue scanning after this tag attempt
            i += 1
            non_comment += 1

    return None


def sniff_html_encoding(data: bytes, transport_encoding: str | None = None) -> tuple[str, int]:
    # Transport overrides everything.
    transport = normalize_encoding_label(transport_encoding)
    if transport:
        return transport, 0

    bom_enc, bom_len = _sniff_bom(data)
    if bom_enc:
        return bom_enc, bom_len

    meta_enc = _prescan_for_meta_charset(data)
    if meta_enc:
        return meta_enc, 0

    return "windows-1252", 0


def decode_html(data: bytes, transport_encoding: str | None = None) -> tuple[str, str]:
    """Decode an HTML byte stream using HTML encoding sniffing.

    Returns (text, encoding_name).
    """
    enc, bom_len = sniff_html_encoding(data, transport_encoding=transport_encoding)

    # Allowlist supported decoders.
    if enc not in {
        "utf-8",
        "windows-1252",
        "iso-8859-2",
        "euc-jp",
        "utf-16",
        "utf-16le",
        "utf-16be",
    }:  # pragma: no cover
        enc = "windows-1252"
        bom_len = 0

    payload = data[bom_len:] if bom_len else data

    if enc == "windows-1252":
        return payload.decode("cp1252"), "windows-1252"

    if enc == "iso-8859-2":
        return payload.decode("iso-8859-2", "replace"), "iso-8859-2"

    if enc == "euc-jp":
        return payload.decode("euc_jp", "replace"), "euc-jp"

    if enc == "utf-16le":
        return payload.decode("utf-16le", "replace"), "utf-16le"

    if enc == "utf-16be":
        return payload.decode("utf-16be", "replace"), "utf-16be"

    if enc == "utf-16":
        return payload.decode("utf-16", "replace"), "utf-16"

    # Default utf-8
    return payload.decode("utf-8", "replace"), "utf-8"
