"""CSS sanitization helpers for JustHTML sanitizer policies."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .url import (
    UrlFilter,
    UrlHandling,
    UrlPolicy,
    UrlProxy,
    UrlRule,
    _effective_allow_relative,
    _effective_proxy,
    _effective_url_handling,
    _sanitize_url_value_with_rule,
)

if TYPE_CHECKING:
    from collections.abc import Collection


def _is_valid_css_property_name(name: str) -> bool:
    # Conservative: allow only ASCII letters/digits/hyphen.
    # This keeps parsing deterministic and avoids surprises with escapes.
    if not name:
        return False
    for ch in name:
        if "a" <= ch <= "z" or "0" <= ch <= "9" or ch == "-":
            continue
        return False
    return True


def _css_value_contains_disallowed_functions(value: str, *, allow_url: bool) -> bool:
    # Extremely conservative check: reject any declaration value that contains a
    # CSS function/construct that can load external resources.
    #
    # We intentionally do not try to parse full CSS (escapes, strings, etc.).
    # Instead, we scan while ignoring ASCII whitespace/control chars and CSS
    # comments, and we look for dangerous tokens in the normalized stream.
    #
    # If allow_url=True, url(...) is not considered disallowed (it is handled
    # separately by `_sanitize_css_url_functions`).
    if "\\" in value:
        return True

    normalized: list[str] = []
    i = 0
    n = len(value)
    while i < n:
        ch = value[i]

        # Treat CSS comments as ignorable, so obfuscation like u/**/rl( is caught.
        if ch == "/" and i + 1 < n and value[i + 1] == "*":
            i += 2
            while i + 1 < n:
                if value[i] == "*" and value[i + 1] == "/":
                    i += 2
                    break
                i += 1
            else:
                # Unterminated comments are invalid CSS; be conservative.
                return True
            continue

        o = ord(ch)
        if o <= 0x20 or o == 0x7F:
            i += 1
            continue

        normalized.append(chr(o + 0x20) if "A" <= ch <= "Z" else ch)
        i += 1

    compact = "".join(normalized)
    ambient_value = compact.removesuffix("!important")
    if ambient_value in {"inherit", "revert", "revert-layer", "unset"}:
        return True

    blocked_tokens = (
        "@import",
        "var(",
        ":inherit",
        ":revert",
        ":unset",
        "src(",
        "image(",
        "image-set(",
        "expression(",
        "progid:",
        "alphaimageloader",
        "behavior:",
        "-moz-binding",
    )
    if any(token in compact for token in blocked_tokens):
        return True

    return not allow_url and "url(" in compact


def _css_value_may_load_external_resource(value: str) -> bool:
    return _css_value_contains_disallowed_functions(value, allow_url=False)


def _css_value_has_disallowed_resource_functions(value: str) -> bool:
    """Return True if `value` contains disallowed CSS constructs (excluding url())."""

    return _css_value_contains_disallowed_functions(value, allow_url=True)


def _lookup_css_url_rule(*, url_policy: UrlPolicy, tag: str, prop: str) -> UrlRule | None:
    key = f"style:{prop}"
    return url_policy.allow_rules.get((tag, key)) or url_policy.allow_rules.get(("*", key))


def _sanitize_url_function_value(
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
    # Keep this parser intentionally conservative. We only support plain url(...)
    # without escapes and without nested parentheses inside the URL token.
    v = value

    if "\\" in v:
        return None

    # Reject comments entirely; they are commonly used for obfuscation.
    if "/*" in v:
        return None

    lower = v.lower()
    out_parts: list[str] = []
    i = 0
    replaced_any = False
    n = len(v)

    while True:
        j = lower.find("url(", i)
        if j == -1:
            out_parts.append(v[i:])
            break

        out_parts.append(v[i:j])
        k = j + 4  # after 'url('

        # Skip whitespace after 'url('
        while k < n and ord(v[k]) <= 0x20:
            k += 1
        if k >= n:
            return None

        quoted = v[k] in {'"', "'"}
        q = v[k] if quoted else ""
        if quoted:
            k += 1
            start = k
            end_quote = v.find(q, k)
            if end_quote == -1:
                return None
            url_raw = v[start:end_quote]
            k = end_quote + 1

            while k < n and ord(v[k]) <= 0x20:
                k += 1
            if k >= n or v[k] != ")":
                return None
            end_paren = k
        else:
            end_paren = v.find(")", k)
            if end_paren == -1:
                return None
            url_raw = v[k:end_paren].strip()
            if not url_raw:
                return None
            # Unquoted url(...) must not contain whitespace.
            if any(ord(ch) <= 0x20 or ord(ch) == 0x7F for ch in url_raw):
                return None

        # Require a clear token boundary after url(...). Without whitespace or a
        # delimiter, we can't safely reason about how the CSS parser will
        # interpret the value.
        next_idx = end_paren + 1
        if next_idx < n:
            nxt = v[next_idx]
            if not (ord(nxt) <= 0x20 or nxt in {",", "/"}):
                return None

        sanitized = _sanitize_url_value_with_rule(
            rule=rule,
            value=url_raw,
            tag=tag,
            attr=attr,
            handling=handling,
            allow_relative=allow_relative,
            proxy=proxy,
            url_filter=url_filter,
            apply_filter=apply_filter,
        )
        if sanitized is None:
            return None

        # Avoid generating CSS that needs escaping.
        for ch in sanitized:
            o = ord(ch)
            if o <= 0x20 or o == 0x7F or ch in {"'", '"', "(", ")", "\\"}:
                return None

        out_parts.append(f"url('{sanitized}')")
        replaced_any = True

        i = end_paren + 1

    return None if not replaced_any else "".join(out_parts)


def _sanitize_css_url_functions(*, url_policy: UrlPolicy, tag: str, prop: str, value: str) -> str | None:
    rule = _lookup_css_url_rule(url_policy=url_policy, tag=tag, prop=prop)
    if rule is None:
        return None

    return _sanitize_url_function_value(
        rule=rule,
        value=value,
        tag=tag,
        attr=f"style:{prop}",
        handling=_effective_url_handling(url_policy=url_policy, rule=rule),
        allow_relative=_effective_allow_relative(url_policy=url_policy, rule=rule),
        proxy=_effective_proxy(url_policy=url_policy, rule=rule),
        url_filter=url_policy.url_filter,
        apply_filter=True,
    )


def _sanitize_inline_style(
    *,
    allowed_css_properties: Collection[str],
    value: str,
    tag: str,
    url_policy: UrlPolicy | None = None,
) -> str | None:
    allowed = allowed_css_properties
    if not allowed:
        return None

    v = str(value)
    if not v:
        return None

    out_parts: list[str] = []
    lower_tag = str(tag).lower()
    for decl in v.split(";"):
        d = decl.strip()
        if not d:
            continue
        colon = d.find(":")
        if colon <= 0:
            continue

        prop = d[:colon].strip().lower()
        if not _is_valid_css_property_name(prop):
            continue
        if prop not in allowed:
            continue

        prop_value = d[colon + 1 :].strip()
        if not prop_value:
            continue

        if prop_value.lower() in {"inherit", "revert", "revert-layer", "unset"}:
            continue

        if _css_value_may_load_external_resource(prop_value):
            if url_policy is None:
                continue

            if _css_value_has_disallowed_resource_functions(prop_value):
                continue

            sanitized_with_urls = _sanitize_css_url_functions(
                url_policy=url_policy, tag=lower_tag, prop=prop, value=prop_value
            )
            if sanitized_with_urls is None:
                continue
            prop_value = sanitized_with_urls

        out_parts.append(f"{prop}: {prop_value}")

    if not out_parts:
        return None
    return "; ".join(out_parts)
