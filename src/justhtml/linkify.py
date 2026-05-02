"""Text linkification scanner.

This module finds URL/email-like substrings in plain text.

It is intentionally HTML-agnostic: in JustHTML it is applied to DOM text nodes,
not to raw HTML strings.

The behavior is driven by vendored compliance fixtures from the upstream
`linkify-it` project (MIT licensed). See `tests/linkify-it/README.md`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class LinkMatch:
    start: int
    end: int
    text: str
    href: str
    kind: str  # "url" | "email"


DEFAULT_TLDS: Final[frozenset[str]] = frozenset(
    {
        # Keep this aligned with linkify-it's default list.
        # See: https://github.com/markdown-it/linkify-it/blob/master/index.mjs
        "biz",
        "com",
        "edu",
        "gov",
        "net",
        "org",
        "pro",
        "web",
        "xxx",
        "aero",
        "asia",
        "coop",
        "info",
        "museum",
        "name",
        "shop",
        "рф",
    }
)


# A pragmatic Unicode-aware domain label pattern.
#
# Use `\w` for Unicode letters/digits (and underscore), and reject underscores
# during validation. This is intentionally stricter than allowing all non-ASCII
# codepoints, and matches the fixture behavior around delimiter punctuation.
_LABEL_RE: Final[str] = (
    r"[0-9A-Za-z\w\u2600-\u27bf]"
    r"(?:[0-9A-Za-z\w\u2600-\u27bf-]{0,61}[0-9A-Za-z\w\u2600-\u27bf])?"
)

# A fast-ish candidate matcher. We do real validation after we find a candidate.
_CANDIDATE_PATTERN: Final[str] = "".join(
    [
        r"(?i)([^0-9A-Za-z_])",  # left boundary (avoid matching after underscore)
        r"(",  # candidate group
        r"(?:https?|ftp)://[^\s<>\uFF5C]+",  # absolute URL
        r"|mailto:[^\s<>\uFF5C]+",  # mailto
        r"|//[^\s<>\uFF5C]+",  # protocol-relative
        r"|(?:www\.)[^\s<>\uFF5C]+",  # www.
        rf"|[0-9A-Za-z][0-9A-Za-z.!#$%&'*+/=?^_`{{|}}~\-\"]*@(?:{_LABEL_RE}\.)+{_LABEL_RE}",  # email
        r"|(?:\d{1,3}\.){3}\d{1,3}(?:/[^\s<>\uFF5C]*)?",  # IPv4
        rf"|(?:{_LABEL_RE}\.)+{_LABEL_RE}(?:/[^\s<>\uFF5C]*)?",  # fuzzy domain/path
        r")",
    ]
)

_CANDIDATE_RE: Final[re.Pattern[str]] = re.compile(_CANDIDATE_PATTERN, re.UNICODE)

_TRAILING_PUNCT: Final[str] = ".,;:!?"

# RE pattern for 2-character TLDs, copied from linkify-it (MIT licensed).
_CC_TLD_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:a[cdefgilmnoqrstuwxz]|b[abdefghijmnorstvwyz]|c[acdfghiklmnoruvwxyz]|d[ejkmoz]|e[cegrstu]|f[ijkmor]|g[abdefghilmnpqrstuwy]|h[kmnrtu]|i[delmnoqrst]|j[emop]|k[eghimnprwyz]|l[abcikrstuvy]|m[acdeghklmnopqrstuvwxyz]|n[acefgilopruz]|om|p[aefghklmnrstwy]|qa|r[eosuw]|s[abcdeghijklmnortuvxyz]|t[cdfghjklmnortvwz]|u[agksyz]|v[aceginu]|w[fs]|y[et]|z[amw])$",
    re.IGNORECASE,
)


def _is_valid_tld(tld: str, *, extra_tlds: frozenset[str]) -> bool:
    t = (tld or "").lower()
    if not t:
        return False
    # Only valid 2-letter ccTLDs (avoid false positives like `.js`).
    if len(t) == 2 and _CC_TLD_RE.match(t) is not None:
        return True
    # Any punycode root.
    if t.startswith("xn--"):
        return True
    return t in DEFAULT_TLDS or t in extra_tlds


def _split_domain_for_tld(host: str) -> tuple[str, str] | None:
    # Return (domain_without_tld, tld).
    h = (host or "").strip().strip(".")
    if not h:
        return None
    if h.lower() == "localhost":
        return ("localhost", "")
    if "." not in h:
        return None
    base, tld = h.rsplit(".", 1)
    return (base, tld)


@dataclass(frozen=True, slots=True)
class LinkifyConfig:
    fuzzy_ip: bool = False
    extra_tlds: frozenset[str] = frozenset()

    @staticmethod
    def with_extra_tlds(extra_tlds: list[str] | tuple[str, ...] | set[str] | frozenset[str]) -> LinkifyConfig:
        return LinkifyConfig(extra_tlds=frozenset(str(t).lower() for t in extra_tlds))


def _is_valid_ipv4(host: str) -> bool:
    parts = host.split(".")
    if len(parts) != 4:
        return False
    for p in parts:
        if not p or len(p) > 3:
            return False
        if not p.isdigit():
            return False
        v = int(p)
        if v < 0 or v > 255:
            return False
    return True


def _punycode_host(host: str) -> str:
    # Safety default: normalize Unicode domains to punycode for href.
    try:
        return host.encode("idna").decode("ascii")
    except UnicodeError:
        return host


def _split_host_and_rest(raw: str) -> tuple[str, str]:
    # raw is after an optional scheme prefix (or for fuzzy domains, the whole).
    # Extract host[:port] and the rest (path/query/fragment).
    for i, ch in enumerate(raw):
        if ch in "/?#":
            return raw[:i], raw[i:]
    return raw, ""


def _strip_wrapping(raw: str) -> tuple[str, int, int]:
    # Trim common wrappers like <...> or quotes, but report how many chars were removed
    # from start/end so we can compute accurate offsets.
    start_trim = 0
    end_trim = 0

    if raw and raw[0] in "<\"'([{" and raw[-1] in ">\"')]}":
        # Angle brackets are common for autolinks.
        # Quotes/brackets: we strip them only if they wrap the candidate.
        raw = raw[1:-1]
        start_trim = 1
        end_trim = 1

    return raw, start_trim, end_trim


def _trim_trailing(candidate: str) -> str:
    # Remove trailing punctuation and unbalanced closing brackets.
    if not candidate:
        return candidate

    end = len(candidate)

    # First strip sentence punctuation.
    while end and candidate[end - 1] in _TRAILING_PUNCT:
        end -= 1

    if not end:
        return ""

    counts = {
        '"': 0,
        "'": 0,
        "(": 0,
        ")": 0,
        "[": 0,
        "]": 0,
        "{": 0,
        "}": 0,
        "<": 0,
        ">": 0,
    }
    for ch in candidate[:end]:
        if ch in counts:
            counts[ch] += 1

    # Then strip quoting terminators when unbalanced (treat quotes as wrappers).
    while end and candidate[end - 1] in "\"'":
        q = candidate[end - 1]
        if counts[q] % 2 == 1:
            counts[q] -= 1
            end -= 1
            continue
        break

    # Then strip unmatched closing brackets.
    # We treat ) ] } > as potentially closable.
    pairs = {")": "(", "]": "[", "}": "{", ">": "<"}
    while end and candidate[end - 1] in pairs:
        close = candidate[end - 1]
        open_ch = pairs[close]
        if counts[close] > counts[open_ch]:
            counts[close] -= 1
            end -= 1
            continue
        break

    return candidate[:end]


def _href_for(text: str) -> tuple[str, str]:
    lower = text.lower()

    if lower.startswith("mailto:"):
        return text, "email"

    if "@" in text and not lower.startswith(("http://", "https://", "ftp://", "//", "www.")):
        return f"mailto:{text}", "email"

    if lower.startswith(("http://", "https://", "ftp://", "//")):
        return text, "url"

    # www. and fuzzy domains default to http://
    return f"http://{text}", "url"


def _punycode_href(href: str) -> str:
    # Convert the host portion to punycode (IDNA), keeping the rest intact.
    lower = href.lower()
    prefix = ""
    rest = href

    if lower.startswith("mailto:"):
        return href

    if lower.startswith("http://"):
        prefix = href[:7]
        rest = href[7:]
    elif lower.startswith("https://"):
        prefix = href[:8]
        rest = href[8:]
    elif lower.startswith("ftp://"):
        prefix = href[:6]
        rest = href[6:]
    elif lower.startswith("//"):
        prefix = href[:2]
        rest = href[2:]
    else:
        # Shouldn't happen; fuzzy hrefs are normalized before calling.
        prefix = ""
        rest = href

    hostport, tail = _split_host_and_rest(rest)

    # Handle userinfo (user:pass@host)
    userinfo = ""
    hostport2 = hostport
    if "@" in hostport:
        userinfo, hostport2 = hostport.rsplit("@", 1)

    host = hostport2
    port = ""
    if hostport2.startswith("["):
        # IPv6-ish, don't punycode.
        return href
    if ":" in hostport2:
        host, port = hostport2.split(":", 1)

    host_pc = _punycode_host(host)
    rebuilt = host_pc
    if port:
        rebuilt = f"{rebuilt}:{port}"
    if userinfo:
        rebuilt = f"{userinfo}@{rebuilt}"

    return f"{prefix}{rebuilt}{tail}"


def find_links(text: str) -> list[LinkMatch]:
    return find_links_with_config(text, LinkifyConfig())


def find_links_with_config(text: str, config: LinkifyConfig) -> list[LinkMatch]:
    if not text:
        return []

    # Mirror linkify-it behavior: always scan with a leading boundary character.
    scan_text = "\n" + text

    out: list[LinkMatch] = []

    for m in _CANDIDATE_RE.finditer(scan_text):
        raw = m.group(2)

        # Compute absolute offsets (exclude the boundary prefix char).
        start = m.start(2) - 1
        end = m.end(2) - 1

        stripped, s_trim, e_trim = _strip_wrapping(raw)
        start += s_trim
        end -= e_trim

        cand = _trim_trailing(stripped)
        if not cand:
            continue

        # Markdown-style termination: `(...URL...)[...]` should stop at the `)`.
        lower = cand.lower()
        if lower.startswith(("http://", "https://", "ftp://")) and ")[" in cand:
            cand = cand.split(")[", 1)[0]
            cand = _trim_trailing(cand)
            if not cand:
                continue

        # Treat leading quotes as wrappers/delimiters, not part of the URL/email.
        if cand and cand[0] in "\"'" and 0 <= start < len(text) and text[start] == cand[0]:
            cand = cand[1:]
            start += 1
            if not cand:
                continue

        # Adjust end after trimming.
        end = start + len(cand)

        lower = cand.lower()

        # If this looks like a fuzzy domain that starts immediately after ://,
        # treat it as part of a broken/disabled schema (e.g. _http://example.com, hppt://example.com).
        if not lower.startswith(("http://", "https://", "ftp://", "mailto:", "//", "www.")) and "@" not in cand:
            if start >= 3 and text[start - 3 : start] == "://":
                continue
            if start > 0 and text[start - 1] in "/:@":
                continue

        # Validate fuzzy IP option.
        if (
            cand
            and cand[0].isdigit()
            and "." in cand
            and not lower.startswith(("http://", "https://", "ftp://", "//"))
        ):
            host, _ = _split_host_and_rest(cand)
            if host.replace(".", "").isdigit() and _is_valid_ipv4(host):
                if not config.fuzzy_ip:
                    continue

        # Validate // URLs: allow localhost or dotted domains, but not single-level.
        if lower.startswith("//"):
            # Protect against matching the // inside :// or ///.
            if start > 0 and text[start - 1] in ":/":
                continue
            after = cand[2:]
            hostport, _ = _split_host_and_rest(after)
            if not hostport:
                continue
            if hostport.startswith("["):
                continue
            host_only = hostport
            if "@" in host_only:
                host_only = host_only.rsplit("@", 1)[1]
            if ":" in host_only:
                host_only = host_only.split(":", 1)[0]
            if host_only.lower() != "localhost" and "." not in host_only:
                continue

            if "_" in host_only:
                continue

        # Validate fuzzy domains and emails with TLD allowlist.
        is_scheme = lower.startswith(("http://", "https://", "ftp://", "mailto:"))
        is_www = lower.startswith("www.")
        is_proto_rel = lower.startswith("//")

        if not is_scheme and not is_proto_rel and not is_www and "@" not in cand:
            host, _ = _split_host_and_rest(cand)
            if "_" in host:
                continue

            # IPv4 candidates don't use the TLD allowlist.
            if "." in host and host.replace(".", "").isdigit() and _is_valid_ipv4(host):
                pass
            else:
                parts = _split_domain_for_tld(host)
                if parts is None:
                    continue
                _base, tld = parts
                if not _is_valid_tld(tld, extra_tlds=config.extra_tlds):
                    continue

        if (
            "@" in cand
            and not lower.startswith(("http://", "https://", "ftp://", "//"))
            and not lower.startswith("mailto:")
        ):
            # Fuzzy email requires a valid TLD.
            local, domain = cand.rsplit("@", 1)
            _ = local
            host, _tail = _split_host_and_rest(domain)
            if "_" in host:
                continue
            parts = _split_domain_for_tld(host)
            if parts is None:
                continue
            _base, tld = parts
            if not _is_valid_tld(tld, extra_tlds=config.extra_tlds):
                continue

        # Validate basic URL host/port if scheme-based.
        if lower.startswith(("http://", "https://", "ftp://")):
            after = cand.split("://", 1)[1]
            hostport, _ = _split_host_and_rest(after)
            if not hostport:
                continue
            if "@" in hostport:
                hostport = hostport.rsplit("@", 1)[1]
            host = hostport
            if ":" in hostport and not hostport.startswith("["):
                host, port = hostport.split(":", 1)
                if port and (not port.isdigit() or int(port) > 65535):
                    continue
            if not host or host.startswith(("-", ".")) or host.endswith(("-", ".")) or ".." in host:
                continue
            if "_" in host:
                continue
            if "." in host and host.replace(".", "").isdigit() and not _is_valid_ipv4(host):
                continue

        href, kind = _href_for(cand)
        href = _punycode_href(href)

        out.append(LinkMatch(start=start, end=end, text=cand, href=href, kind=kind))

    # Avoid overlapping matches by keeping first-longest.
    if not out:
        return out
    out.sort(key=lambda x: (x.start, -(x.end - x.start)))
    filtered: list[LinkMatch] = []
    last_end = -1
    for lm in out:
        if lm.start < last_end:
            continue
        filtered.append(lm)
        last_end = lm.end
    return filtered
