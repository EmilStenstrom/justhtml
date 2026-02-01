"""HTML sanitization policy API.

This module defines the public API for JustHTML sanitization.

The sanitizer operates on the parsed JustHTML DOM and is intentionally
policy-driven.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast
from urllib.parse import quote, urlsplit

from .tokens import ParseError

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
        proxy_url = str(self.url)
        if not proxy_url:
            raise ValueError("UrlProxy.url must be a non-empty string")
        object.__setattr__(self, "url", proxy_url)
        object.__setattr__(self, "param", str(self.param))


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
    # If None, the URL is kept ("allow") after it passes validation.
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


@dataclass(frozen=True, slots=True)
class UrlPolicy:
    # Default handling for URL-like attributes after they pass UrlRule checks.
    # - "allow": keep the URL as-is
    # - "strip": drop the attribute
    # - "proxy": rewrite the URL through a proxy (UrlPolicy.proxy or UrlRule.proxy)
    default_handling: UrlHandling = "strip"

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
            if rule.handling == "proxy" and self.proxy is None and rule.proxy is None:
                raise ValueError("UrlRule.handling='proxy' requires a UrlPolicy.proxy or a per-rule UrlRule.proxy")


def _proxy_url_value(*, proxy: UrlProxy, value: str) -> str:
    sep = "&" if "?" in proxy.url else "?"
    return f"{proxy.url}{sep}{proxy.param}={quote(value, safe='')}"


@dataclass(slots=True)
class UnsafeHandler:
    """Centralized handler for security findings.

    This is intentionally a small stateful object so multiple sanitization-
    related passes/transforms can share the same unsafe-handling behavior and
    (in collect mode) append into the same error list.
    """

    unsafe_handling: UnsafeHandling

    # Optional external sink (e.g. a JustHTML document's .errors list).
    # When set and unsafe_handling == "collect", security findings are written
    # into that list so multiple components can share a single sink.
    sink: list[ParseError] | None = None

    _errors: list[ParseError] | None = None

    def reset(self) -> None:
        if self.unsafe_handling != "collect":
            self._errors = None
            return

        if self.sink is None:
            self._errors = []
            return

        # Remove previously collected security findings from the shared sink to
        # avoid accumulating duplicates across multiple runs.
        errors = self.sink
        write_i = 0
        for e in errors:
            if e.category == "security":
                continue
            errors[write_i] = e
            write_i += 1
        del errors[write_i:]

    def collected(self) -> list[ParseError]:
        src = self.sink if self.sink is not None else self._errors
        if not src:
            return []

        if self.sink is not None:
            out = [e for e in src if e.category == "security"]
        else:
            out = list(src)
        out.sort(
            key=lambda e: (
                e.line if e.line is not None else 1_000_000_000,
                e.column if e.column is not None else 1_000_000_000,
            )
        )
        return out

    def handle(self, msg: str, *, node: Any | None = None) -> None:
        mode = self.unsafe_handling
        if mode == "strip":
            return
        if mode == "raise":
            raise UnsafeHtmlError(msg)
        if mode == "collect":
            dest = self.sink
            if dest is None:
                if self._errors is None:
                    self._errors = []
                dest = self._errors

            line: int | None = None
            column: int | None = None
            if node is not None:
                # Best-effort: use node origin metadata when enabled.
                # This stays allocation-light and avoids any input re-parsing.
                line = node.origin_line
                column = node.origin_col

            dest.append(
                ParseError(
                    "unsafe-html",
                    line=line,
                    column=column,
                    category="security",
                    message=msg,
                )
            )
            return
        raise AssertionError(f"Unhandled unsafe_handling: {mode!r}")


@dataclass(frozen=True, slots=True)
class SanitizationPolicy:
    """An allow-list driven policy for sanitizing a parsed DOM.

    This API is intentionally small. The implementation will interpret these
    fields strictly.

    - Tags not in `allowed_tags` are disallowed.
    - Attributes not in `allowed_attributes[tag]` (or `allowed_attributes["*"]`)
      are disallowed.
    - URL scheme checks apply to attributes listed in `url_attributes`.

    All tag and attribute names are expected to be ASCII-lowercase.
    """

    allowed_tags: Collection[str]
    allowed_attributes: Mapping[str, Collection[str]]

    # URL handling.
    url_policy: UrlPolicy = field(default_factory=UrlPolicy)

    drop_comments: bool = True
    drop_doctype: bool = True
    drop_foreign_namespaces: bool = True

    # Dangerous containers whose text payload should not be preserved.
    drop_content_tags: Collection[str] = field(default_factory=lambda: {"script", "style"})

    # Inline style allowlist.
    # Only applies when the `style` attribute is allowed for a tag.
    # If empty, inline styles are effectively disabled (style attributes are dropped).
    allowed_css_properties: Collection[str] = field(default_factory=set)

    # Link hardening.
    # If non-empty, ensure these tokens are present in <a rel="...">.
    # (The sanitizer will merge tokens; it will not remove existing ones.)
    force_link_rel: Collection[str] = field(default_factory=set)

    # Determines how unsafe input is handled.
    #
    # - "strip": Default. Remove/drop unsafe constructs and keep going.
    # - "raise": Raise UnsafeHtmlError on the first unsafe construct.
    #
    # This is intentionally a string mode (instead of a boolean) so we can add
    # more behaviors over time without changing the API shape.
    unsafe_handling: UnsafeHandling = "strip"

    # Determines how disallowed tags are handled.
    #
    # - "unwrap": Default. Drop the tag but keep/sanitize its children.
    # - "escape": Emit original tag tokens as text, keep/sanitize children.
    # - "drop": Drop the entire disallowed subtree.
    disallowed_tag_handling: DisallowedTagHandling = "unwrap"

    _unsafe_handler: UnsafeHandler = field(
        default_factory=lambda: UnsafeHandler("strip"),
        init=False,
        repr=False,
        compare=False,
    )

    # Internal caches to avoid per-node allocations in hot paths.
    _allowed_attrs_global: frozenset[str] = field(
        default_factory=frozenset,
        init=False,
        repr=False,
        compare=False,
    )
    _allowed_attrs_by_tag: dict[str, frozenset[str]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    # Cache for the compiled `Sanitize(policy=...)` transform pipeline.
    # This lets safe serialization reuse the same compiled transforms.
    _compiled_sanitize_transforms: list[Any] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        # Validate and normalize allowlists once so the sanitizer can do fast
        # membership checks.
        #
        # NOTE: Strings are iterables in Python. Passing e.g. "div" or
        # "attribute" by mistake would otherwise silently become a set of
        # characters ("d", "i", "v"), producing surprising behavior.
        if isinstance(self.allowed_tags, str):
            raise TypeError(
                "SanitizationPolicy.allowed_tags must be a collection of tag names (e.g. ['div']), not a string"
            )

        if isinstance(self.allowed_attributes, str) or not isinstance(self.allowed_attributes, Mapping):
            raise TypeError(
                "SanitizationPolicy.allowed_attributes must be a mapping like {'*': ['id'], 'a': ['href']}"
            )

        for tag, attrs in self.allowed_attributes.items():
            if isinstance(attrs, str):
                raise TypeError(
                    "SanitizationPolicy.allowed_attributes values must be collections of attribute names "
                    f"(e.g. {{'{tag}': ['class', 'id']}}), not a string"
                )

        normalized_tags = {str(t).strip().lower() for t in self.allowed_tags if str(t).strip()}
        object.__setattr__(self, "allowed_tags", normalized_tags)

        normalized_attrs: dict[str, set[str]] = {}
        for tag, attrs in self.allowed_attributes.items():
            tag_name = str(tag).strip().lower()
            if not tag_name:
                raise ValueError("SanitizationPolicy.allowed_attributes contains an empty tag key")

            attr_set = attrs if isinstance(attrs, set) else set(attrs)
            normalized_attr_set = {str(a).strip().lower() for a in attr_set if str(a).strip()}

            if tag_name in normalized_attrs:
                normalized_attrs[tag_name].update(normalized_attr_set)
            else:
                normalized_attrs[tag_name] = normalized_attr_set

        object.__setattr__(self, "allowed_attributes", normalized_attrs)

        if not isinstance(self.drop_content_tags, set):
            object.__setattr__(self, "drop_content_tags", set(self.drop_content_tags))

        if not isinstance(self.allowed_css_properties, set):
            object.__setattr__(self, "allowed_css_properties", set(self.allowed_css_properties))

        if not isinstance(self.force_link_rel, set):
            object.__setattr__(self, "force_link_rel", set(self.force_link_rel))

        unsafe_handling = str(self.unsafe_handling)
        if unsafe_handling not in {"strip", "raise", "collect"}:
            raise ValueError("Invalid unsafe_handling. Expected one of: 'strip', 'raise', 'collect'")
        object.__setattr__(self, "unsafe_handling", unsafe_handling)

        disallowed_tag_handling = str(self.disallowed_tag_handling)
        if disallowed_tag_handling not in {"unwrap", "escape", "drop"}:
            raise ValueError("Invalid disallowed_tag_handling. Expected one of: 'unwrap', 'escape', 'drop'")
        object.__setattr__(self, "disallowed_tag_handling", disallowed_tag_handling)

        # Centralize unsafe-handling logic so multiple passes can share it.
        handler = UnsafeHandler(cast("UnsafeHandling", unsafe_handling))
        handler.reset()
        object.__setattr__(self, "_unsafe_handler", handler)

        # Normalize rel tokens once so downstream sanitization can stay allocation-light.
        # (Downstream code expects lowercase tokens and ignores empty/whitespace.)
        if self.force_link_rel:
            normalized_force_link_rel = {t.strip().lower() for t in self.force_link_rel if str(t).strip()}
            object.__setattr__(self, "force_link_rel", normalized_force_link_rel)

        style_allowed = any("style" in attrs for attrs in self.allowed_attributes.values())
        if style_allowed and not self.allowed_css_properties:
            raise ValueError(
                "SanitizationPolicy allows the 'style' attribute but allowed_css_properties is empty. "
                "Either remove 'style' from allowed_attributes or set allowed_css_properties (for example CSS_PRESET_TEXT)."
            )

        allowed_attributes = self.allowed_attributes
        allowed_global = frozenset(allowed_attributes.get("*", ()))
        by_tag: dict[str, frozenset[str]] = {}
        for tag, attrs in allowed_attributes.items():
            if tag == "*":
                continue
            by_tag[tag] = frozenset(allowed_global.union(attrs))
        object.__setattr__(self, "_allowed_attrs_global", allowed_global)
        object.__setattr__(self, "_allowed_attrs_by_tag", by_tag)

    def reset_collected_security_errors(self) -> None:
        self._unsafe_handler.reset()

    def collected_security_errors(self) -> list[ParseError]:
        return self._unsafe_handler.collected()

    def handle_unsafe(self, msg: str, *, node: Any | None = None) -> None:
        self._unsafe_handler.handle(msg, node=node)


_URL_NORMALIZE_STRIP_TABLE = {i: None for i in range(0x21)}
_URL_NORMALIZE_STRIP_TABLE[0x7F] = None

_ATTR_DROP_PATTERNS: tuple[str, ...] = ("on*", "srcdoc", "*:*")
_ATTR_DROP_REGEX: re.Pattern[str] = re.compile(
    "^(?:" + "|".join(re.escape(p).replace(r"\*", ".*").replace(r"\?", ".") for p in _ATTR_DROP_PATTERNS) + ")$"
)


DEFAULT_POLICY: SanitizationPolicy = SanitizationPolicy(
    allowed_tags=[
        # Text / structure
        "p",
        "br",
        # Structure
        "div",
        "span",
        "blockquote",
        "pre",
        "code",
        # Headings
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        # Lists
        "ul",
        "ol",
        "li",
        # Tables
        "table",
        "thead",
        "tbody",
        "tfoot",
        "tr",
        "th",
        "td",
        # Text formatting
        "b",
        "strong",
        "i",
        "em",
        "u",
        "s",
        "sub",
        "sup",
        "small",
        "mark",
        # Quotes/code
        # Line breaks
        "hr",
        # Links and images
        "a",
        "img",
    ],
    allowed_attributes={
        "*": ["class", "id", "title", "lang", "dir"],
        "a": ["href", "title"],
        "img": ["src", "alt", "title", "width", "height", "loading", "decoding"],
        "th": ["colspan", "rowspan"],
        "td": ["colspan", "rowspan"],
    },
    url_policy=UrlPolicy(
        default_handling="strip",
        allow_rules={
            ("a", "href"): UrlRule(
                allowed_schemes=["http", "https", "mailto", "tel"],
                resolve_protocol_relative="https",
            ),
            ("img", "src"): UrlRule(
                allowed_schemes=[],
                resolve_protocol_relative=None,
            ),
        },
    ),
    allowed_css_properties=set(),
)


# A conservative preset for allowing a small amount of inline styling.
# This is intentionally focused on text-level styling and avoids layout/
# positioning properties that are commonly abused for UI redress.
CSS_PRESET_TEXT: frozenset[str] = frozenset(
    {
        "background-color",
        "color",
        "font-size",
        "font-style",
        "font-weight",
        "letter-spacing",
        "line-height",
        "text-align",
        "text-decoration",
        "text-transform",
        "white-space",
        "word-break",
        "word-spacing",
        "word-wrap",
    }
)


DEFAULT_DOCUMENT_POLICY: SanitizationPolicy = SanitizationPolicy(
    allowed_tags=sorted(set(DEFAULT_POLICY.allowed_tags) | {"html", "head", "body", "title"}),
    allowed_attributes=DEFAULT_POLICY.allowed_attributes,
    url_policy=DEFAULT_POLICY.url_policy,
    drop_comments=DEFAULT_POLICY.drop_comments,
    drop_doctype=DEFAULT_POLICY.drop_doctype,
    drop_foreign_namespaces=DEFAULT_POLICY.drop_foreign_namespaces,
    drop_content_tags=DEFAULT_POLICY.drop_content_tags,
    allowed_css_properties=DEFAULT_POLICY.allowed_css_properties,
    force_link_rel=DEFAULT_POLICY.force_link_rel,
)


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

    buf: list[str] = []
    max_len = len("alphaimageloader")

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

        lower_ch = chr(o + 0x20) if "A" <= ch <= "Z" else ch

        buf.append(lower_ch)
        if len(buf) > max_len:
            buf.pop(0)

        # Check for url( and image-set( anywhere in the normalized stream.
        if not allow_url and len(buf) >= 4 and buf[-4:] == ["u", "r", "l", "("]:
            return True
        if len(buf) >= 10 and buf[-10:] == [
            "i",
            "m",
            "a",
            "g",
            "e",
            "-",
            "s",
            "e",
            "t",
            "(",
        ]:
            return True

        # IE-only but still worth blocking defensively.
        if len(buf) >= 11 and buf[-11:] == [
            "e",
            "x",
            "p",
            "r",
            "e",
            "s",
            "s",
            "i",
            "o",
            "n",
            "(",
        ]:
            return True

        # Legacy IE CSS filters that can fetch remote resources.
        if len(buf) >= 7 and buf[-7:] == ["p", "r", "o", "g", "i", "d", ":"]:
            return True
        if len(buf) >= 16 and buf[-16:] == [
            "a",
            "l",
            "p",
            "h",
            "a",
            "i",
            "m",
            "a",
            "g",
            "e",
            "l",
            "o",
            "a",
            "d",
            "e",
            "r",
        ]:
            return True

        # Legacy bindings/behaviors that can pull remote content.
        if len(buf) >= 9 and buf[-9:] == ["b", "e", "h", "a", "v", "i", "o", "r", ":"]:
            return True
        if len(buf) >= 12 and buf[-12:] == [
            "-",
            "m",
            "o",
            "z",
            "-",
            "b",
            "i",
            "n",
            "d",
            "i",
            "n",
            "g",
        ]:
            return True

        i += 1

    return False


def _css_value_may_load_external_resource(value: str) -> bool:
    return _css_value_contains_disallowed_functions(value, allow_url=False)


def _css_value_has_disallowed_resource_functions(value: str) -> bool:
    """Return True if `value` contains disallowed CSS constructs (excluding url())."""

    return _css_value_contains_disallowed_functions(value, allow_url=True)


def _lookup_css_url_rule(*, url_policy: UrlPolicy, tag: str, prop: str) -> UrlRule | None:
    key = f"style:{prop}"
    return url_policy.allow_rules.get((tag, key)) or url_policy.allow_rules.get(("*", key))


def _sanitize_css_url_functions(*, url_policy: UrlPolicy, tag: str, prop: str, value: str) -> str | None:
    # Keep this parser intentionally conservative. We only support plain url(...)
    # without escapes and without nested parentheses inside the URL token.
    if "\\" in value:
        return None

    # Reject comments entirely; they are commonly used for obfuscation.
    if "/*" in value:
        return None

    rule = _lookup_css_url_rule(url_policy=url_policy, tag=tag, prop=prop)
    if rule is None:
        return None

    lower = value.lower()
    out_parts: list[str] = []
    i = 0
    replaced_any = False
    n = len(value)

    while True:
        j = lower.find("url(", i)
        if j == -1:
            out_parts.append(value[i:])
            break

        out_parts.append(value[i:j])
        k = j + 4  # after 'url('

        # Skip whitespace after 'url('
        while k < n and ord(value[k]) <= 0x20:
            k += 1
        if k >= n:
            return None

        quoted = value[k] in {'"', "'"}
        q = value[k] if quoted else ""
        if quoted:
            k += 1
            start = k
            end_quote = value.find(q, k)
            if end_quote == -1:
                return None
            url_raw = value[start:end_quote]
            k = end_quote + 1

            while k < n and ord(value[k]) <= 0x20:
                k += 1
            if k >= n or value[k] != ")":
                return None
            end_paren = k
        else:
            end_paren = value.find(")", k)
            if end_paren == -1:
                return None
            url_raw = value[k:end_paren].strip()
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
            nxt = value[next_idx]
            if not (ord(nxt) <= 0x20 or nxt in {",", "/"}):
                return None

        sanitized = _sanitize_url_value_with_rule(
            rule=rule,
            value=url_raw,
            tag=tag,
            attr=f"style:{prop}",
            handling=_effective_url_handling(url_policy=url_policy, rule=rule),
            allow_relative=_effective_allow_relative(url_policy=url_policy, rule=rule),
            proxy=_effective_proxy(url_policy=url_policy, rule=rule),
            url_filter=url_policy.url_filter,
            apply_filter=True,
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

        if _css_value_may_load_external_resource(prop_value):
            if url_policy is None:
                continue

            if _css_value_has_disallowed_resource_functions(prop_value):
                continue

            sanitized_with_urls = _sanitize_css_url_functions(
                url_policy=url_policy, tag=str(tag).lower(), prop=prop, value=prop_value
            )
            if sanitized_with_urls is None:
                continue
            prop_value = sanitized_with_urls

        out_parts.append(f"{prop}: {prop_value}")

    if not out_parts:
        return None
    return "; ".join(out_parts)


def _normalize_url_for_checking(value: str) -> str:
    # Strip whitespace/control chars commonly used for scheme obfuscation.
    # Note: do not strip backslashes; they are not whitespace/control chars,
    # and removing them can turn invalid schemes into valid ones.
    return value.translate(_URL_NORMALIZE_STRIP_TABLE)


def _is_valid_scheme(scheme: str) -> bool:
    first = scheme[0]
    if not ("a" <= first <= "z" or "A" <= first <= "Z"):
        return False
    for ch in scheme[1:]:
        if "a" <= ch <= "z" or "A" <= ch <= "Z" or "0" <= ch <= "9" or ch in "+-.":
            continue
        return False
    return True


def _has_scheme(value: str) -> bool:
    idx = value.find(":")
    if idx <= 0:
        return False
    # Scheme must appear before any path/query/fragment separator.
    end = len(value)
    for sep in ("/", "?", "#"):
        j = value.find(sep)
        if j != -1 and j < end:
            end = j
    if idx >= end:
        return False
    return _is_valid_scheme(value[:idx])


def _has_invalid_scheme_like_prefix(value: str) -> bool:
    idx = value.find(":")
    if idx <= 0:
        return False

    end = len(value)
    for sep in ("/", "?", "#"):
        j = value.find(sep)
        if j != -1 and j < end:
            end = j
    if idx >= end:
        return False

    return not _is_valid_scheme(value[:idx])


def _effective_proxy(*, url_policy: UrlPolicy, rule: UrlRule) -> UrlProxy | None:
    return rule.proxy if rule.proxy is not None else url_policy.proxy


def _effective_url_handling(*, url_policy: UrlPolicy, rule: UrlRule) -> UrlHandling:
    # URL-like attributes are allowlisted via UrlPolicy.allow_rules. When they are
    # allowlisted and the URL passes validation, the default action is to keep the URL.
    return rule.handling if rule.handling is not None else "allow"


def _effective_allow_relative(*, url_policy: UrlPolicy, rule: UrlRule) -> bool:
    return rule.allow_relative if rule.allow_relative is not None else url_policy.default_allow_relative


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
        v = rewritten

    stripped = str(v).strip()
    normalized = _normalize_url_for_checking(stripped)
    if not normalized:
        # If normalization removes everything, the value was empty/whitespace/
        # control-only. Drop it rather than keeping weird control characters.
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

        parsed = urlsplit(resolved_url)
        scheme = (parsed.scheme or "").lower()
        if scheme not in rule.allowed_schemes:
            return None

        if rule.allowed_hosts is not None:
            host = (parsed.hostname or "").lower()
            if not host or host not in rule.allowed_hosts:
                return None

        if handling == "strip":
            return None
        if handling == "proxy":
            return None if proxy is None else _proxy_url_value(proxy=proxy, value=resolved_url)
        return resolved_url

    if _has_scheme(normalized):
        parsed = urlsplit(normalized)
        scheme = (parsed.scheme or "").lower()
        if scheme not in rule.allowed_schemes:
            return None
        if rule.allowed_hosts is not None:
            host = (parsed.hostname or "").lower()
            if not host or host not in rule.allowed_hosts:
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
        v = rewritten

    stripped = str(v).strip()
    if not stripped:
        return None

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
            handling=_effective_url_handling(url_policy=url_policy, rule=rule),
            allow_relative=_effective_allow_relative(url_policy=url_policy, rule=rule),
            proxy=_effective_proxy(url_policy=url_policy, rule=rule),
            url_filter=None,
            apply_filter=False,
        )
        if sanitized_url is None:
            return None

        out_candidates.append(f"{sanitized_url} {desc}".strip())

    return None if not out_candidates else ", ".join(out_candidates)


_URL_LIKE_ATTRS: frozenset[str] = frozenset(
    {
        # Common URL-valued attributes.
        "href",
        "src",
        "srcset",
        "poster",
        "action",
        "formaction",
        "data",
        "cite",
        "background",
        # Can trigger requests/pings.
        "ping",
    }
)


def _sanitize(node: Any, *, policy: SanitizationPolicy | None = None) -> Any:
    """Return a sanitized clone of `node`.

    This returns a sanitized clone without mutating the original tree.
    For performance, it builds the sanitized clone in a single pass.
    """

    if policy is None:
        policy = DEFAULT_DOCUMENT_POLICY if node.name == "#document" else DEFAULT_POLICY

    # Escape-mode tag reconstruction may need access to the original source HTML.
    # Historically we allow a child element to inherit _source_html from an
    # ancestor container; keep that behavior even though we sanitize a clone.
    if policy.disallowed_tag_handling == "escape":
        root_source_html = getattr(node, "_source_html", None)
        if root_source_html:
            from .node import Template  # noqa: PLC0415

            stack: list[Any] = [node]
            while stack:
                current = stack.pop()
                current_source_html = getattr(current, "_source_html", None) or root_source_html

                children = getattr(current, "children", None) or []
                for child in children:
                    # Text does not have _source_html.
                    if getattr(child, "name", "") == "#text":
                        continue
                    if getattr(child, "_source_html", None) is None:
                        child._source_html = current_source_html
                    stack.append(child)

                if type(current) is Template and current.template_content is not None:
                    tc = current.template_content
                    if getattr(tc, "_source_html", None) is None:
                        tc._source_html = current_source_html
                    stack.append(tc)

    # We intentionally implement safe-output sanitization by applying the
    # `Sanitize(policy=...)` transform pipeline to a clone of the node.
    # This keeps a single canonical sanitization algorithm.
    from .transforms import Sanitize, apply_compiled_transforms, compile_transforms  # noqa: PLC0415

    compiled = policy._compiled_sanitize_transforms
    if compiled is None:
        compiled = compile_transforms((Sanitize(policy=policy),))
        object.__setattr__(policy, "_compiled_sanitize_transforms", compiled)

    # Container-root rule: transforms walk children of the provided root.
    # For non-container roots, wrap the cloned node in a document fragment so
    # the sanitizer can act on the root node itself.
    if node.name in {"#document", "#document-fragment"}:
        cloned = node.clone_node(deep=True)
        apply_compiled_transforms(cloned, compiled, errors=None)
        return cloned

    from .node import DocumentFragment  # noqa: PLC0415

    wrapper = DocumentFragment()
    wrapper.append_child(node.clone_node(deep=True))
    apply_compiled_transforms(wrapper, compiled, errors=None)

    children = wrapper.children or []
    if len(children) == 1:
        only = children[0]
        only.parent = None
        wrapper.children = []
        return only

    return wrapper


def sanitize_dom(
    node: Any,
    *,
    policy: SanitizationPolicy | None = None,
    errors: list[ParseError] | None = None,
) -> Any:
    """Sanitize a DOM tree in place.

    For document roots (`#document` or `#document-fragment`), this mutates the
    tree in place and returns the same root. For other nodes, the node is
    sanitized as if it were the only child of a document fragment; the returned
    node may need to be reattached by the caller.
    """

    if policy is None:
        policy = DEFAULT_DOCUMENT_POLICY if node.name == "#document" else DEFAULT_POLICY

    from .transforms import Sanitize, apply_compiled_transforms, compile_transforms  # noqa: PLC0415

    compiled = policy._compiled_sanitize_transforms
    if compiled is None:
        compiled = compile_transforms((Sanitize(policy=policy),))
        object.__setattr__(policy, "_compiled_sanitize_transforms", compiled)

    if node.name in {"#document", "#document-fragment"}:
        apply_compiled_transforms(node, compiled, errors=errors)
        return node

    from .node import DocumentFragment  # noqa: PLC0415

    wrapper = DocumentFragment()
    wrapper.append_child(node)
    apply_compiled_transforms(wrapper, compiled, errors=errors)

    children = wrapper.children or []
    if len(children) == 1:
        only = children[0]
        only.parent = None
        wrapper.children = []
        return only

    return wrapper
