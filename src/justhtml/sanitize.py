"""HTML sanitization policy API.

This module defines the public API for JustHTML sanitization.

The sanitizer operates on the parsed JustHTML DOM and is intentionally
policy-driven.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

from .rawtext import neutralize_rawtext_end_tag_sequences
from .sanitize_url import (
    _URL_BEARING_PARAM_NAMES,
    _URL_LIKE_ATTRS,
    DisallowedTagHandling,
    UnsafeHandling,
    UnsafeHtmlError,
    UrlFilter,
    UrlHandling,
    UrlPolicy,
    UrlProxy,
    UrlRule,
    _effective_allow_relative,
    _effective_proxy,
    _effective_url_handling,
    _is_legacy_ipv4_number,
    _is_noncanonical_numeric_ipv4_host,
    _prepare_standalone_url_value_for_checking,
    _raw_authority_host,
    _sanitize_comma_or_space_separated_url_list,
    _sanitize_space_separated_url_list,
    _sanitize_srcset_value,
    _sanitize_url_value_with_rule,
    _strip_invisible_unicode,
    _url_policy_signature,
)
from .selector import DEFAULT_SELECTOR_LIMITS, SelectorLimits
from .tokens import ParseError

if TYPE_CHECKING:
    from .node import NodeType

__all__ = [
    "CSS_PRESET_TEXT",
    "DEFAULT_DOCUMENT_POLICY",
    "DEFAULT_POLICY",
    "_URL_BEARING_PARAM_NAMES",
    "_URL_LIKE_ATTRS",
    "DisallowedTagHandling",
    "SanitizationPolicy",
    "UnsafeHandler",
    "UnsafeHandling",
    "UnsafeHtmlError",
    "UrlFilter",
    "UrlHandling",
    "UrlPolicy",
    "UrlProxy",
    "UrlRule",
    "_css_value_contains_disallowed_functions",
    "_css_value_has_disallowed_resource_functions",
    "_css_value_may_load_external_resource",
    "_effective_allow_relative",
    "_effective_proxy",
    "_effective_url_handling",
    "_is_legacy_ipv4_number",
    "_is_noncanonical_numeric_ipv4_host",
    "_is_valid_css_property_name",
    "_neutralize_rawtext_end_tag_sequences",
    "_prepare_standalone_url_value_for_checking",
    "_raw_authority_host",
    "_sanitize",
    "_sanitize_comma_or_space_separated_url_list",
    "_sanitize_css_url_functions",
    "_sanitize_inline_style",
    "_sanitize_rawtext_element_contents",
    "_sanitize_space_separated_url_list",
    "_sanitize_srcset_value",
    "_sanitize_url_function_value",
    "_sanitize_url_value_with_rule",
    "_seal_url_policy",
    "_strip_invisible_unicode",
    "sanitize_dom",
]


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


def _normalize_policy_name(value: Any) -> str:
    return str(value).strip().lower()


def _normalize_policy_name_set(values: Collection[Any]) -> set[str]:
    return {name for value in values if (name := _normalize_policy_name(value))}


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

    allowed_tags: frozenset[str]
    allowed_attributes: Mapping[str, Collection[str]]

    if TYPE_CHECKING:

        def __init__(
            self,
            allowed_tags: Collection[str],
            allowed_attributes: Mapping[str, Collection[str]],
            url_policy: UrlPolicy = ...,
            drop_comments: bool = True,
            drop_doctype: bool = True,
            drop_content_tags: Collection[str] = ...,
            allowed_css_properties: Collection[str] = ...,
            force_link_rel: Collection[str] = ...,
            unsafe_handling: UnsafeHandling = "strip",
            disallowed_tag_handling: DisallowedTagHandling = "unwrap",
            strip_invisible_unicode: bool = True,
            selector_limits: SelectorLimits = DEFAULT_SELECTOR_LIMITS,
        ) -> None: ...

    # URL handling.
    url_policy: UrlPolicy = field(default_factory=UrlPolicy)

    drop_comments: bool = True
    drop_doctype: bool = True

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

    # Strip invisible Unicode commonly abused for obfuscation in text and
    # attribute values, such as variation selectors, zero-width/bidi controls,
    # and private-use characters.
    strip_invisible_unicode: bool = True

    # Resource limits used by selector parsing and matching in sanitization
    # transform pipelines. Applications with known-large documents/selectors
    # can raise these while keeping conservative defaults for untrusted input.
    selector_limits: SelectorLimits = DEFAULT_SELECTOR_LIMITS

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
    _compiled_sanitize_transforms: tuple[Any, ...] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _compiled_sanitize_signature: Any = field(
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

        normalized_tags = frozenset(_normalize_policy_name_set(self.allowed_tags))
        object.__setattr__(self, "allowed_tags", normalized_tags)

        normalized_attrs: dict[str, set[str]] = {}
        for tag, attrs in self.allowed_attributes.items():
            tag_name = _normalize_policy_name(tag)
            if not tag_name:
                raise ValueError("SanitizationPolicy.allowed_attributes contains an empty tag key")

            attr_set = attrs if isinstance(attrs, set) else set(attrs)
            normalized_attr_set = _normalize_policy_name_set(attr_set)

            if tag_name in normalized_attrs:
                normalized_attrs[tag_name].update(normalized_attr_set)
            else:
                normalized_attrs[tag_name] = normalized_attr_set

        object.__setattr__(self, "allowed_attributes", normalized_attrs)

        normalized_drop_content_tags = _normalize_policy_name_set(self.drop_content_tags)
        object.__setattr__(self, "drop_content_tags", normalized_drop_content_tags)

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
        object.__setattr__(self, "strip_invisible_unicode", bool(self.strip_invisible_unicode))
        if not isinstance(self.selector_limits, SelectorLimits):
            raise TypeError("SanitizationPolicy.selector_limits must be a SelectorLimits instance")

        # Centralize unsafe-handling logic so multiple passes can share it.
        handler = UnsafeHandler(cast("UnsafeHandling", unsafe_handling))
        handler.reset()
        object.__setattr__(self, "_unsafe_handler", handler)

        # Normalize rel tokens once so downstream sanitization can stay allocation-light.
        # (Downstream code expects lowercase tokens and ignores empty/whitespace.)
        if self.force_link_rel:
            normalized_force_link_rel = _normalize_policy_name_set(self.force_link_rel)
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

    def collects_security_errors_into(self, sink: list[ParseError]) -> bool:
        """Return True if security findings are being collected into `sink`.

        This is intentionally a small helper to avoid other modules depending
        on the private UnsafeHandler implementation details.
        """
        return self._unsafe_handler.sink is sink

    def handle_unsafe(self, msg: str, *, node: Any | None = None) -> None:
        self._unsafe_handler.handle(msg, node=node)


def _compiled_sanitize_transforms_for_policy(policy: SanitizationPolicy) -> tuple[Any, ...]:
    from .transforms import Sanitize, compile_transforms  # noqa: PLC0415

    signature = _sanitization_policy_signature(policy)
    compiled = policy._compiled_sanitize_transforms
    if compiled is None or policy._compiled_sanitize_signature != signature:
        compiled = tuple(
            compile_transforms(
                (Sanitize(policy=policy),),
            )
        )
        object.__setattr__(policy, "_compiled_sanitize_transforms", compiled)
        object.__setattr__(policy, "_compiled_sanitize_signature", signature)
    return compiled


def _seal_url_policy(url_policy: UrlPolicy) -> None:
    sealed_rules: dict[tuple[str, str], UrlRule] = {}
    for (tag, attr), rule in url_policy.allow_rules.items():
        object.__setattr__(rule, "allowed_schemes", frozenset(str(s) for s in rule.allowed_schemes))
        if rule.allowed_hosts is not None:
            object.__setattr__(rule, "allowed_hosts", frozenset(str(h).lower() for h in rule.allowed_hosts))
        sealed_rules[(str(tag).lower(), str(attr).lower())] = rule

    object.__setattr__(url_policy, "allow_rules", MappingProxyType(sealed_rules))


def _seal_default_policy(policy: SanitizationPolicy) -> None:
    object.__setattr__(
        policy,
        "allowed_attributes",
        MappingProxyType({str(tag).lower(): frozenset(attrs) for tag, attrs in policy.allowed_attributes.items()}),
    )
    object.__setattr__(policy, "drop_content_tags", frozenset(policy.drop_content_tags))
    object.__setattr__(policy, "allowed_css_properties", frozenset(policy.allowed_css_properties))
    object.__setattr__(policy, "force_link_rel", frozenset(policy.force_link_rel))
    _seal_url_policy(policy.url_policy)


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
        "caption",
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
                handling="allow",
                resolve_protocol_relative="https",
            ),
            ("img", "src"): UrlRule(
                allowed_schemes=[],
                handling="allow",
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
    drop_doctype=False,
    drop_content_tags=DEFAULT_POLICY.drop_content_tags,
    allowed_css_properties=DEFAULT_POLICY.allowed_css_properties,
    force_link_rel=DEFAULT_POLICY.force_link_rel,
    strip_invisible_unicode=DEFAULT_POLICY.strip_invisible_unicode,
)

_seal_default_policy(DEFAULT_POLICY)
_seal_default_policy(DEFAULT_DOCUMENT_POLICY)


_RAWTEXT_SERIALIZATION_ELEMENTS: frozenset[str] = frozenset({"script", "style"})


def _neutralize_rawtext_end_tag_sequences(text: str, tag_name: str) -> tuple[str, bool]:
    return neutralize_rawtext_end_tag_sequences(text, tag_name)


def _record_rawtext_security_issue(
    *,
    policy: SanitizationPolicy,
    errors: list[ParseError] | None,
    code: str,
    message: str,
    node: Any,
) -> None:
    policy.handle_unsafe(message, node=node)
    if errors is None:
        return
    errors.append(
        ParseError(
            code,
            line=node.origin_line,
            column=node.origin_col,
            category="security",
            message=message,
        )
    )


def _sanitize_rawtext_element_contents(
    node: Any,
    *,
    policy: SanitizationPolicy,
    errors: list[ParseError] | None,
) -> None:
    from .node import Template  # noqa: PLC0415

    stack: list[Any] = [node]

    while stack:
        current = stack.pop()
        raw_name = current.name
        if type(raw_name) is str:
            name = raw_name if raw_name.islower() else raw_name.lower()
        else:  # pragma: no cover
            name = str(raw_name).lower()

        if name in _RAWTEXT_SERIALIZATION_ELEMENTS:
            children = current.children
            if not children:
                continue

            text_children: list[Any] = []
            text_parts: list[str] = []
            for child in children:
                if child.name == "#text":
                    text_children.append(child)
                    text_parts.append(child.data or "")
                    continue

                _record_rawtext_security_issue(
                    policy=policy,
                    errors=errors,
                    code="unsafe-rawtext-child",
                    message=f"Unsafe non-text child inside <{name}> was dropped",
                    node=child,
                )
                child.parent = None

            if not text_children:
                current.children = []
                continue

            combined_text = "".join(text_parts)
            primary_text = text_children[0]
            sanitized_text, changed = _neutralize_rawtext_end_tag_sequences(combined_text, str(name))

            if changed:
                _record_rawtext_security_issue(
                    policy=policy,
                    errors=errors,
                    code="unsafe-rawtext-end-tag",
                    message=f"Unsafe raw text inside <{name}> contains a closing tag sequence",
                    node=primary_text,
                )
                primary_text.data = sanitized_text
                for extra_text in text_children[1:]:
                    extra_text.parent = None
                current.children = [primary_text] if sanitized_text else []
            else:
                current.children = text_children

            if name == "style" and sanitized_text and _css_value_may_load_external_resource(sanitized_text):
                _record_rawtext_security_issue(
                    policy=policy,
                    errors=errors,
                    code="unsafe-style-resource",
                    message="Unsafe CSS inside <style> contains resource-loading constructs",
                    node=primary_text,
                )
                for child in current.children:
                    child.parent = None
                current.children = []
            continue

        children = current.children
        if children:
            stack.extend(reversed(children))

        if type(current) is Template and current.template_content is not None:
            stack.append(current.template_content)


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


def _sanitization_policy_signature(policy: SanitizationPolicy) -> tuple[Any, ...]:
    allowed_attributes_sig = tuple(
        sorted(
            (
                str(tag).lower(),
                tuple(sorted(str(attr).lower() for attr in attrs)),
            )
            for tag, attrs in policy.allowed_attributes.items()
        )
    )

    return (
        tuple(sorted(str(tag).lower() for tag in policy.allowed_tags)),
        allowed_attributes_sig,
        tuple(sorted(str(tag).lower() for tag in policy.drop_content_tags)),
        tuple(sorted(str(prop).lower() for prop in policy.allowed_css_properties)),
        tuple(sorted(str(token).lower() for token in policy.force_link_rel)),
        policy.disallowed_tag_handling,
        policy.strip_invisible_unicode,
        policy.selector_limits,
        _url_policy_signature(policy.url_policy),
    )


def _sanitize(node: Any, *, policy: SanitizationPolicy | None = None) -> Any:
    """Return a sanitized clone of `node`.

    This returns a sanitized clone without mutating the original tree.
    For performance, it builds the sanitized clone in a single pass.
    """

    if policy is None:
        policy = DEFAULT_DOCUMENT_POLICY if node.name == "#document" else DEFAULT_POLICY

    if policy.unsafe_handling == "collect":
        policy.reset_collected_security_errors()

    # Escape-mode tag reconstruction may need access to the original source HTML.
    # Historically we allow a child element to inherit _source_html from an
    # ancestor container; keep that behavior even though we sanitize a clone.
    if policy.disallowed_tag_handling == "escape":
        from .node import Node, Template  # noqa: PLC0415

        root_source_html = node._source_html if isinstance(node, Node) else None
        if root_source_html:
            stack: list[Node] = [node]
            while stack:
                current = stack.pop()
                current_source_html = current._source_html or root_source_html

                children = current.children or ()
                for child in children:
                    # Text does not have _source_html.
                    if not isinstance(child, Node):
                        continue
                    if child._source_html is None:
                        child._source_html = current_source_html
                    stack.append(child)

                if type(current) is Template and current.template_content is not None:
                    tc = current.template_content
                    if tc._source_html is None:
                        tc._source_html = current_source_html
                    stack.append(tc)

    # We intentionally implement safe-output sanitization by applying the
    # `Sanitize(policy=...)` transform pipeline to a clone of the node.
    # This keeps a single canonical sanitization algorithm.
    from .transforms import apply_compiled_transforms  # noqa: PLC0415

    compiled = _compiled_sanitize_transforms_for_policy(policy)

    # Container-root rule: transforms walk children of the provided root.
    # For non-container roots, wrap the cloned node in a document fragment so
    # the sanitizer can act on the root node itself.
    if node.name in {"#document", "#document-fragment"}:
        cloned = node.clone_node(deep=True)
        apply_compiled_transforms(cloned, compiled, errors=None)
        result: Any = cloned
    else:
        from .node import DocumentFragment  # noqa: PLC0415

        wrapper = DocumentFragment()
        wrapper.append_child(node.clone_node(deep=True))
        apply_compiled_transforms(wrapper, compiled, errors=None)

        children = cast("list[Any]", wrapper.children)
        if len(children) == 1:
            only = children[0]
            only.parent = None
            wrapper.children = []
            result = only
        else:
            result = wrapper

    return result


def _sanitize_dom_once(
    node: Any,
    *,
    policy: SanitizationPolicy,
    errors: list[ParseError] | None,
) -> Any:
    from .transforms import apply_compiled_transforms  # noqa: PLC0415

    compiled = _compiled_sanitize_transforms_for_policy(policy)

    if node.name in {"#document", "#document-fragment"}:
        apply_compiled_transforms(node, compiled, errors=errors)
        return node

    from .node import DocumentFragment  # noqa: PLC0415

    wrapper = DocumentFragment()
    wrapper.append_child(node)
    apply_compiled_transforms(wrapper, compiled, errors=errors)

    children = cast("list[Any]", wrapper.children)
    if len(children) == 1:
        only = children[0]
        only.parent = None
        wrapper.children = []
        return only

    return wrapper


def sanitize_dom(
    node: NodeType,
    *,
    policy: SanitizationPolicy | None = None,
    errors: list[ParseError] | None = None,
) -> NodeType:
    """Sanitize a DOM tree in place.

    For document roots (`#document` or `#document-fragment`), this mutates the
    tree in place and returns the same root. For other nodes, the node is
    sanitized as if it were the only child of a document fragment; the returned
    node may need to be reattached by the caller.
    """

    if policy is None:
        policy = DEFAULT_DOCUMENT_POLICY if node.name == "#document" else DEFAULT_POLICY

    if policy.unsafe_handling == "collect":
        policy.reset_collected_security_errors()

    result = _sanitize_dom_once(node, policy=policy, errors=errors)

    return cast("NodeType", result)
