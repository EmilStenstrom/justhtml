"""URL policy and URL-value sanitization helpers."""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import parse_qsl, quote, urlsplit

UrlFilter = Callable[[str, str, str], str | None]


class UnsafeHtmlError(ValueError):
    """Raised when unsafe HTML is encountered and unsafe_handling='raise'."""


UnsafeHandling = Literal["strip", "raise", "collect"]

DisallowedTagHandling = Literal["unwrap", "escape", "drop"]

UrlHandling = Literal["allow", "strip", "proxy"]


@dataclass(frozen=True, slots=True)
class UrlProxy:
    url: str
    param: str = "url"

    def __post_init__(self) -> None:
        proxy_url = str(self.url).strip()
        if not proxy_url:
            raise ValueError("UrlProxy.url must be a non-empty string")
        proxy_param = str(self.param)
        if not proxy_param:
            raise ValueError("UrlProxy.param must be a non-empty string")
        _validate_proxy_url(proxy_url, proxy_param)
        object.__setattr__(self, "url", proxy_url)
        object.__setattr__(self, "param", proxy_param)


@dataclass(frozen=True, slots=True)
class UrlRule:
    """Rule for a single URL-valued attribute (e.g. a[href], img[src]).

    This is intentionally rendering-oriented.

    - Returning/keeping a URL can still cause network requests when the output
        is rendered (notably for <img src>). Applications like email viewers often
        want to block remote loads by default.
    """

    # Allow same-document fragments (#foo). Typically safe.
    allow_fragment: bool = True

    # If set, protocol-relative URLs (//example.com) are resolved to this scheme
    # (e.g. "https") before checking allowed_schemes.
    # If None, protocol-relative URLs are disallowed.
    resolve_protocol_relative: str | None = "https"

    # Allow absolute URLs with these schemes (lowercase), e.g. {"https"}.
    # If empty, all absolute URLs with a scheme are disallowed.
    allowed_schemes: Collection[str] = field(default_factory=set)

    # If provided, absolute URLs are allowed only if the parsed host is in this
    # allowlist.
    allowed_hosts: Collection[str] | None = None

    # Optional per-rule handling override.
    # If None, UrlPolicy.default_handling is used after validation.
    handling: UrlHandling | None = None

    # Optional per-rule override of UrlPolicy.default_allow_relative.
    # If None, UrlPolicy.default_allow_relative is used.
    allow_relative: bool | None = None

    # Optional proxy override for absolute/protocol-relative URLs.
    # Used when the effective URL handling is "proxy".
    proxy: UrlProxy | None = None

    def __post_init__(self) -> None:
        # Accept lists/tuples from user code, normalize for internal use.
        if not isinstance(self.allowed_schemes, set):
            object.__setattr__(self, "allowed_schemes", set(self.allowed_schemes))
        if self.allowed_hosts is not None and not isinstance(self.allowed_hosts, set):
            object.__setattr__(self, "allowed_hosts", set(self.allowed_hosts))

        if self.proxy is not None and not isinstance(self.proxy, UrlProxy):
            raise TypeError("UrlRule.proxy must be a UrlProxy or None")

        if self.handling is not None:
            mode = str(self.handling)
            if mode not in {"allow", "strip", "proxy"}:
                raise ValueError("Invalid UrlRule.handling. Expected one of: 'allow', 'strip', 'proxy'")
            object.__setattr__(self, "handling", mode)

        if self.allow_relative is not None:
            object.__setattr__(self, "allow_relative", bool(self.allow_relative))


_URL_PROXY_RULE = UrlRule(
    resolve_protocol_relative=None,
    allowed_schemes=frozenset({"http", "https"}),
)


@dataclass(frozen=True, slots=True)
class UrlPolicy:
    # Default handling for URL-like attributes after they pass UrlRule checks.
    # - "allow": keep the URL as-is
    # - "strip": drop the attribute
    # - "proxy": rewrite the URL through a proxy (UrlPolicy.proxy or UrlRule.proxy)
    default_handling: UrlHandling = "allow"

    # Default allowance for relative URLs (including /path, ./path, ../path, ?query)
    # for URL-like attributes that have a matching UrlRule.
    default_allow_relative: bool = True

    # Rule configuration for URL-valued attributes.
    allow_rules: Mapping[tuple[str, str], UrlRule] = field(default_factory=dict)

    # Optional hook that can drop or rewrite URLs.
    # url_filter(tag, attr, value) should return:
    # - a replacement string to keep (possibly rewritten), or
    # - None to drop the attribute.
    url_filter: UrlFilter | None = None

    # Default proxy config used when a rule is handled with "proxy" and
    # the rule does not specify its own UrlRule.proxy override.
    proxy: UrlProxy | None = None

    def __post_init__(self) -> None:
        mode = str(self.default_handling)
        if mode not in {"allow", "strip", "proxy"}:
            raise ValueError("Invalid default_handling. Expected one of: 'allow', 'strip', 'proxy'")
        object.__setattr__(self, "default_handling", mode)

        object.__setattr__(self, "default_allow_relative", bool(self.default_allow_relative))

        if not isinstance(self.allow_rules, dict):
            object.__setattr__(self, "allow_rules", dict(self.allow_rules))

        if self.proxy is not None and not isinstance(self.proxy, UrlProxy):
            raise TypeError("UrlPolicy.proxy must be a UrlProxy or None")

        # Validate proxy configuration for any rules that are in proxy mode.
        for rule in self.allow_rules.values():
            if not isinstance(rule, UrlRule):
                raise TypeError("UrlPolicy.allow_rules values must be UrlRule")
            effective_handling = rule.handling if rule.handling is not None else mode
            if effective_handling == "proxy" and self.proxy is None and rule.proxy is None:
                raise ValueError("URL handling 'proxy' requires a UrlPolicy.proxy or a per-rule UrlRule.proxy")


def _proxy_url_value(*, proxy: UrlProxy, value: str) -> str:
    sep = "&" if "?" in proxy.url else "?"
    return f"{proxy.url}{sep}{quote(proxy.param, safe='')}={quote(value, safe='')}"


_URL_NORMALIZE_STRIP_TABLE = {i: None for i in range(0x21)}
_URL_NORMALIZE_STRIP_TABLE[0x7F] = None
# Used only for scheme checking: HTML URL parsing ignores ASCII whitespace
# around and within schemes, but ordinary spaces may still be preserved in
# allowed URLs for callers that later percent-encode them for a stricter context.
_URL_WHITESPACE_OR_CONTROL_REGEX: re.Pattern[str] = re.compile(r"[\x00-\x20\x7f]")
_URL_CONTROL_CHAR_REGEX: re.Pattern[str] = re.compile(r"[\x00-\x1f\x7f]")
_URL_AUTHORITY_WHITESPACE_REGEX: re.Pattern[str] = re.compile(r"[\x00-\x20\x7f]")

# Invisible Unicode commonly abused for obfuscation includes zero-width and
# bidi controls, variation selectors, and private-use characters.
_INVISIBLE_UNICODE_STRIP_REGEX: re.Pattern[str] = re.compile(
    r"[\u061C\u200B-\u200F\u202A-\u202E\u2060-\u2069\uFE00-\uFE0F\uFEFF\uE000-\uF8FF"
    r"\U000E0100-\U000E01EF\U000F0000-\U000FFFFD\U00100000-\U0010FFFD]"
)


def _normalize_url_for_checking(value: str) -> str:
    # Strip whitespace/control chars commonly used for scheme obfuscation.
    # Note: do not strip backslashes; they are not whitespace/control chars,
    # and removing them can turn invalid schemes into valid ones.
    #
    # Fast path: most URLs contain no control/space chars, so avoid allocating.
    if not _URL_WHITESPACE_OR_CONTROL_REGEX.search(value):
        return value
    return value.translate(_URL_NORMALIZE_STRIP_TABLE)


def _strip_invisible_unicode(value: str) -> str:
    if not _INVISIBLE_UNICODE_STRIP_REGEX.search(value):
        return value
    return _INVISIBLE_UNICODE_STRIP_REGEX.sub("", value)


def _prepare_standalone_url_value_for_checking(value: str) -> str:
    if "&" in value:
        from .entities import decode_entities_in_text  # noqa: PLC0415

        # Match HTML attribute parsing so helper APIs cannot accept a URL that
        # only turns into a disallowed scheme after embedding into markup.
        value = decode_entities_in_text(value, in_attribute=True)
    return _strip_invisible_unicode(value)


def _is_valid_scheme(scheme: str) -> bool:
    first = scheme[0]
    if not ("a" <= first <= "z" or "A" <= first <= "Z"):
        return False
    for ch in scheme[1:]:
        if "a" <= ch <= "z" or "A" <= ch <= "Z" or "0" <= ch <= "9" or ch in "+-.":
            continue
        return False
    return True


def _scheme_like_prefix(value: str) -> str | None:
    """Return text before a URL-like colon, if it precedes path/query/fragment separators."""
    idx = value.find(":")
    if idx <= 0:
        return None
    # Scheme must appear before any path/query/fragment separator.
    end = len(value)
    for sep in ("/", "?", "#"):
        j = value.find(sep)
        if j != -1 and j < end:
            end = j
    if idx >= end:
        return None
    return value[:idx]


def _get_scheme(value: str) -> str | None:
    """Return the URL scheme (lowercased) if present and valid, else None."""
    scheme = _scheme_like_prefix(value)
    if scheme is None:
        return None
    if not _is_valid_scheme(scheme):
        return None
    return scheme.lower()


def _has_invalid_scheme_like_prefix(value: str) -> bool:
    scheme = _scheme_like_prefix(value)
    return scheme is not None and not _is_valid_scheme(scheme)


def _url_host_matches_allowed_hosts(value: str, allowed_hosts: Collection[str]) -> bool:
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError:
        return False
    if _URL_AUTHORITY_WHITESPACE_REGEX.search(parsed.netloc):
        return False
    raw_host = _raw_authority_host(parsed.netloc)
    if not raw_host or not _uses_canonical_ascii_url_host(raw_host):
        return False
    host = (parsed.hostname or "").lower()
    return bool(host and host in allowed_hosts)


def _raw_authority_host(netloc: str) -> str:
    authority = netloc.rsplit("@", 1)[-1]
    if authority.startswith("["):
        end = authority.find("]")
        if end == -1:
            return ""
        return authority[1:end]
    return authority.rsplit(":", 1)[0] if ":" in authority else authority


def _uses_canonical_ascii_url_host(host: str) -> bool:
    if "%" in host:
        return False
    if any(ord(ch) > 0x7F for ch in host):
        return False
    if ":" not in host and _is_noncanonical_numeric_ipv4_host(host):
        return False
    return True


def _is_noncanonical_numeric_ipv4_host(host: str) -> bool:
    labels = host.split(".")
    if not 1 <= len(labels) <= 4:
        return False
    if not all(_is_legacy_ipv4_number(label) for label in labels):
        return False
    return not (
        len(labels) == 4
        and all(label.isdecimal() and str(int(label)) == label and 0 <= int(label) <= 255 for label in labels)
    )


def _is_legacy_ipv4_number(value: str) -> bool:
    if not value:
        return False
    if value.lower().startswith("0x"):
        return len(value) > 2 and all(ch in "0123456789abcdefABCDEF" for ch in value[2:])
    return value.isdecimal()


def _effective_proxy(*, url_policy: UrlPolicy, rule: UrlRule) -> UrlProxy | None:
    return rule.proxy if rule.proxy is not None else url_policy.proxy


def _effective_url_handling(*, url_policy: UrlPolicy, rule: UrlRule) -> UrlHandling:
    return rule.handling if rule.handling is not None else url_policy.default_handling


def _effective_allow_relative(*, url_policy: UrlPolicy, rule: UrlRule) -> bool:
    return rule.allow_relative if rule.allow_relative is not None else url_policy.default_allow_relative


def _validate_proxy_url(proxy_url: str, proxy_param: str) -> None:
    normalized_proxy_url = _normalize_url_for_checking(proxy_url)
    if _has_invalid_scheme_like_prefix(normalized_proxy_url):
        raise ValueError("UrlProxy.url contains an invalid URL scheme")
    scheme = _get_scheme(normalized_proxy_url)
    if scheme in {"http", "https"} and not normalized_proxy_url.startswith(f"{scheme}://"):
        raise ValueError("UrlProxy.url must be relative or use the http/https scheme")
    parsed_proxy_url = urlsplit(proxy_url)
    if parsed_proxy_url.fragment:
        raise ValueError("UrlProxy.url must not contain a fragment")
    if any(name == proxy_param for name, _value in parse_qsl(parsed_proxy_url.query, keep_blank_values=True)):
        raise ValueError("UrlProxy.url must not already contain UrlProxy.param")
    if (
        _sanitize_url_value_with_rule(
            rule=_URL_PROXY_RULE,
            value=proxy_url,
            tag="*",
            attr="urlproxy",
            handling="allow",
            allow_relative=True,
            proxy=None,
            url_filter=None,
            apply_filter=False,
        )
        is None
    ):
        raise ValueError("UrlProxy.url must be relative or use the http/https scheme")


def _sanitize_url_value_with_rule(
    *,
    rule: UrlRule,
    value: str,
    tag: str,
    attr: str,
    handling: UrlHandling,
    allow_relative: bool,
    proxy: UrlProxy | None,
    url_filter: UrlFilter | None,
    apply_filter: bool,
) -> str | None:
    v = value

    if apply_filter and url_filter is not None:
        rewritten = url_filter(tag, attr, v)
        if rewritten is None:
            return None
        v = _strip_invisible_unicode(rewritten)

    stripped = v.strip()
    if _URL_CONTROL_CHAR_REGEX.search(stripped):
        return None

    normalized = _normalize_url_for_checking(stripped)
    if not normalized:
        # If normalization removes everything, the value was empty/whitespace/
        # control-only. Drop it rather than keeping weird control characters.
        return None

    if "\\" in normalized:
        # Browsers normalize backslashes during navigation and resource loading.
        # Values like "\\evil.example/x" or "/\\evil.example/x" can become
        # remote network requests even though Python's URL parsing treats them
        # as relative or hostless. Reject them conservatively.
        return None

    if normalized.startswith("#"):
        if not rule.allow_fragment:
            return None
        if handling == "strip":
            return None
        if handling == "proxy":
            return None if proxy is None else _proxy_url_value(proxy=proxy, value=stripped)
        return stripped

    if handling == "proxy" and _has_invalid_scheme_like_prefix(normalized):
        # If proxying is enabled, do not treat scheme-obfuscation as a relative URL.
        # Some user agents normalize backslashes and other characters during navigation.
        return None

    if normalized.startswith("//"):
        if not rule.resolve_protocol_relative:
            return None

        # Resolve to absolute URL for checking.
        resolved_scheme = rule.resolve_protocol_relative.lower()
        resolved_url = f"{resolved_scheme}:{normalized}"
        raw_resolved_url = f"{resolved_scheme}:{stripped}"
        if resolved_scheme not in rule.allowed_schemes:
            return None

        if rule.allowed_hosts is not None:
            if not _url_host_matches_allowed_hosts(raw_resolved_url, rule.allowed_hosts):
                return None

        if handling == "strip":
            return None
        if handling == "proxy":
            return None if proxy is None else _proxy_url_value(proxy=proxy, value=resolved_url)
        return resolved_url

    scheme = _get_scheme(normalized)
    if scheme is not None:
        if scheme not in rule.allowed_schemes:
            return None
        if scheme in {"http", "https"} and not normalized.startswith(f"{scheme}://"):
            # URL-standard special schemes without "//" are base-dependent in
            # browsers (for example, "https:foo" can resolve like a relative
            # path on an HTTPS page). Do not let them bypass allow_relative or
            # host allowlists by treating them as ordinary absolute URLs.
            if rule.allowed_hosts is not None:
                return None
            if not allow_relative:
                return None
            if handling == "strip":
                return None
            if handling == "proxy":
                return None if proxy is None else _proxy_url_value(proxy=proxy, value=stripped)
            return stripped
        if rule.allowed_hosts is not None:
            if not _url_host_matches_allowed_hosts(stripped, rule.allowed_hosts):
                return None
        if handling == "strip":
            return None
        if handling == "proxy":
            return None if proxy is None else _proxy_url_value(proxy=proxy, value=stripped)
        return stripped

    if not allow_relative:
        return None

    if handling == "strip":
        return None
    if handling == "proxy":
        return None if proxy is None else _proxy_url_value(proxy=proxy, value=stripped)
    return stripped


def _sanitize_srcset_value(
    *,
    url_policy: UrlPolicy,
    rule: UrlRule,
    tag: str,
    attr: str,
    value: str,
) -> str | None:
    # Apply the URL filter once to the whole attribute value.
    v = value
    if url_policy.url_filter is not None:
        rewritten = url_policy.url_filter(tag, attr, v)
        if rewritten is None:
            return None
        v = _strip_invisible_unicode(rewritten)

    stripped = str(v).strip()
    if not stripped:
        return None

    handling = _effective_url_handling(url_policy=url_policy, rule=rule)
    allow_relative = _effective_allow_relative(url_policy=url_policy, rule=rule)
    proxy = _effective_proxy(url_policy=url_policy, rule=rule)

    out_candidates: list[str] = []
    for raw_candidate in stripped.split(","):
        c = raw_candidate.strip()
        if not c:
            continue

        parts = c.split(None, 1)
        url_token = parts[0]
        desc = parts[1].strip() if len(parts) == 2 else ""
        sanitized_url = _sanitize_url_value_with_rule(
            rule=rule,
            value=url_token,
            tag=tag,
            attr=attr,
            handling=handling,
            allow_relative=allow_relative,
            proxy=proxy,
            url_filter=None,
            apply_filter=False,
        )
        if sanitized_url is None:
            return None
        out_candidates.append(f"{sanitized_url} {desc}".strip())

    return None if not out_candidates else ", ".join(out_candidates)


def _sanitize_space_separated_url_list(
    *,
    url_policy: UrlPolicy,
    rule: UrlRule,
    tag: str,
    attr: str,
    value: str,
) -> str | None:
    v = value
    if url_policy.url_filter is not None:
        rewritten = url_policy.url_filter(tag, attr, v)
        if rewritten is None:
            return None
        v = _strip_invisible_unicode(rewritten)

    stripped = str(v).strip()
    if not stripped:
        return None

    tokens = stripped.split()

    out_tokens = _sanitize_url_tokens(tokens, url_policy=url_policy, rule=rule, tag=tag, attr=attr)

    return None if not out_tokens else " ".join(out_tokens)


def _sanitize_comma_or_space_separated_url_list(
    *,
    url_policy: UrlPolicy,
    rule: UrlRule,
    tag: str,
    attr: str,
    value: str,
) -> str | None:
    v = value
    if url_policy.url_filter is not None:
        rewritten = url_policy.url_filter(tag, attr, v)
        if rewritten is None:
            return None
        v = _strip_invisible_unicode(rewritten)

    stripped = str(v).strip()
    if not stripped:
        return None

    tokens = [token for token in re.split(r"[\s,]+", stripped) if token]

    out_tokens = _sanitize_url_tokens(tokens, url_policy=url_policy, rule=rule, tag=tag, attr=attr)

    return None if not out_tokens else " ".join(out_tokens)


def _sanitize_url_tokens(
    tokens: list[str],
    *,
    url_policy: UrlPolicy,
    rule: UrlRule,
    tag: str,
    attr: str,
) -> list[str] | None:
    handling = _effective_url_handling(url_policy=url_policy, rule=rule)
    allow_relative = _effective_allow_relative(url_policy=url_policy, rule=rule)
    proxy = _effective_proxy(url_policy=url_policy, rule=rule)

    out_tokens: list[str] = []
    for token in tokens:
        sanitized = _sanitize_url_value_with_rule(
            rule=rule,
            value=token,
            tag=tag,
            attr=attr,
            handling=handling,
            allow_relative=allow_relative,
            proxy=proxy,
            url_filter=None,
            apply_filter=False,
        )
        if sanitized is None:
            return None
        out_tokens.append(sanitized)

    return None if not out_tokens else out_tokens


_URL_LIKE_ATTRS: frozenset[str] = frozenset(
    {
        # Common URL-valued attributes.
        "href",
        "icon",
        "dynsrc",
        "lowsrc",
        "src",
        "srcset",
        "imagesrcset",
        "poster",
        "action",
        "formaction",
        "data",
        "cite",
        "background",
        "classid",
        "code",
        "codebase",
        "longdesc",
        "manifest",
        "object",
        "profile",
        "usemap",
        # Can trigger requests/pings.
        "archive",
        "ping",
        "attributionsrc",
    }
)

_URL_BEARING_PARAM_NAMES: frozenset[str] = frozenset(
    {
        "code",
        "codebase",
        "data",
        "filename",
        "href",
        "movie",
        "src",
        "url",
    }
)


def _url_rule_signature(rule: UrlRule) -> tuple[Any, ...]:
    allowed_schemes = tuple(sorted(str(s) for s in rule.allowed_schemes))
    allowed_hosts = None
    if rule.allowed_hosts is not None:
        allowed_hosts = tuple(sorted(str(h).lower() for h in rule.allowed_hosts))

    proxy_sig = None
    if rule.proxy is not None:
        proxy_sig = (rule.proxy.url, rule.proxy.param)

    return (
        rule.allow_fragment,
        rule.resolve_protocol_relative,
        allowed_schemes,
        allowed_hosts,
        rule.handling,
        rule.allow_relative,
        proxy_sig,
    )


def _url_policy_signature(url_policy: UrlPolicy) -> tuple[Any, ...]:
    allow_rules_sig = tuple(
        sorted(
            (
                (str(tag).lower(), str(attr).lower()),
                _url_rule_signature(rule),
            )
            for (tag, attr), rule in url_policy.allow_rules.items()
        )
    )

    proxy_sig = None
    if url_policy.proxy is not None:
        proxy_sig = (url_policy.proxy.url, url_policy.proxy.param)

    return (
        url_policy.default_handling,
        url_policy.default_allow_relative,
        allow_rules_sig,
        url_policy.url_filter,
        proxy_sig,
    )
