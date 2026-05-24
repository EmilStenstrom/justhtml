"""URL sanitization runtime helpers."""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Mapping
from typing import TYPE_CHECKING
from urllib.parse import parse_qsl, quote, urlsplit

from . import sanitize_url_policy as _url_policy

if TYPE_CHECKING:
    from .sanitize_url_spec import UrlSinkKind

UrlFilter = _url_policy.UrlFilter
UrlHandling = _url_policy.UrlHandling
UrlPolicy = _url_policy.UrlPolicy
UrlProxy = _url_policy.UrlProxy
UrlRule = _url_policy.UrlRule
UrlSinkHandler = Callable[..., str | None]

_URL_PROXY_RULE = UrlRule(
    resolve_protocol_relative=None,
    allowed_schemes=frozenset({"http", "https"}),
)


def _proxy_url_value(*, proxy: UrlProxy, value: str) -> str:
    sep = "&" if "?" in proxy.url else "?"
    return f"{proxy.url}{sep}{quote(proxy.param, safe='')}={quote(value, safe='')}"


_URL_NORMALIZE_STRIP_TABLE = {i: None for i in range(0x21)}
_URL_NORMALIZE_STRIP_TABLE[0x7F] = None
_URL_WHITESPACE_OR_CONTROL_REGEX: re.Pattern[str] = re.compile(r"[\x00-\x20\x7f]")
_URL_CONTROL_CHAR_REGEX: re.Pattern[str] = re.compile(r"[\x00-\x1f\x7f]")
_URL_AUTHORITY_WHITESPACE_REGEX: re.Pattern[str] = re.compile(r"[\x00-\x20\x7f]")

_INVISIBLE_UNICODE_STRIP_REGEX: re.Pattern[str] = re.compile(
    r"[\u061C\u200B-\u200F\u202A-\u202E\u2060-\u2069\uFE00-\uFE0F\uFEFF\uE000-\uF8FF"
    r"\U000E0100-\U000E01EF\U000F0000-\U000FFFFD\U00100000-\U0010FFFD]"
)


def _normalize_url_for_checking(value: str) -> str:
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
    idx = value.find(":")
    if idx <= 0:
        return None
    end = len(value)
    for sep in ("/", "?", "#"):
        j = value.find(sep)
        if j != -1 and j < end:
            end = j
    if idx >= end:
        return None
    return value[:idx]


def _get_scheme(value: str) -> str | None:
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
        return None

    if "\\" in normalized:
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
        return None

    if normalized.startswith("//"):
        if not rule.resolve_protocol_relative:
            return None

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


def _extract_meta_refresh_url(value: str) -> tuple[str, str] | None:
    prefix, sep, url_value = value.partition(";")
    url_prefix, url_sep, candidate = url_value.partition("=")
    if not sep or url_prefix.strip().lower() != "url" or not url_sep:
        return None

    candidate = candidate.strip()
    if not candidate:
        return None

    if candidate[0] in {"'", '"'}:
        quote_char = candidate[0]
        candidate = candidate[1:]
        end_quote = candidate.find(quote_char)
        if end_quote != -1:
            candidate = candidate[:end_quote]

    if not candidate:
        return None
    return prefix.strip(), candidate


def _sanitize_simple_url_sink_value(
    *,
    url_policy: UrlPolicy,
    rule: UrlRule,
    tag: str,
    attr: str,
    value: str,
) -> str | None:
    return _sanitize_url_value_with_rule(
        rule=rule,
        value=value,
        tag=tag,
        attr=attr,
        handling=_effective_url_handling(url_policy=url_policy, rule=rule),
        allow_relative=_effective_allow_relative(url_policy=url_policy, rule=rule),
        proxy=_effective_proxy(url_policy=url_policy, rule=rule),
        url_filter=url_policy.url_filter,
        apply_filter=True,
    )


def _sanitize_meta_refresh_sink_value(
    *,
    url_policy: UrlPolicy,
    rule: UrlRule,
    tag: str,
    attr: str,
    value: str,
) -> str | None:
    refresh_parts = _extract_meta_refresh_url(value)
    if refresh_parts is None:
        return None

    prefix, candidate = refresh_parts
    sanitized_url = _sanitize_simple_url_sink_value(
        url_policy=url_policy,
        rule=rule,
        tag=tag,
        attr=attr,
        value=candidate,
    )
    return None if sanitized_url is None else f"{prefix.strip()};url={sanitized_url}"


_URL_SINK_HANDLERS: Mapping[UrlSinkKind, UrlSinkHandler] = {
    "url": _sanitize_simple_url_sink_value,
    "srcset": _sanitize_srcset_value,
    "comma_or_space_list": _sanitize_comma_or_space_separated_url_list,
    "space_list": _sanitize_space_separated_url_list,
    "meta_refresh": _sanitize_meta_refresh_sink_value,
}


def _sanitize_url_sink_value(
    *,
    url_policy: UrlPolicy,
    rule: UrlRule,
    tag: str,
    attr: str,
    kind: UrlSinkKind,
    value: str,
) -> str | None:
    return _URL_SINK_HANDLERS[kind](
        url_policy=url_policy,
        rule=rule,
        tag=tag,
        attr=attr,
        value=value,
    )
