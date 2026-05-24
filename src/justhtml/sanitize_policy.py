"""Sanitizer policy types, defaults, sealing, and compilation."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

from .sanitize_url import (
    DisallowedTagHandling,
    UnsafeHandling,
    UnsafeHtmlError,
    UrlPolicy,
    UrlRule,
    _url_policy_signature,
)
from .selector import DEFAULT_SELECTOR_LIMITS, SelectorLimits
from .tokens import ParseError


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
class CompiledSanitizationPolicy:
    """Cached sanitizer execution plan for a normalized policy."""

    policy: SanitizationPolicy
    signature: tuple[Any, ...]
    transforms: tuple[Any, ...]

    def apply_to(self, node: Any, *, errors: list[ParseError] | None = None) -> None:
        from .transforms import apply_compiled_transforms  # noqa: PLC0415

        apply_compiled_transforms(node, self.transforms, errors=errors)


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
    _compiled_sanitize_policy: CompiledSanitizationPolicy | None = field(
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

    def compile(self) -> CompiledSanitizationPolicy:
        return _compiled_sanitization_policy_for_policy(self)


def _compiled_sanitization_policy_for_policy(policy: SanitizationPolicy) -> CompiledSanitizationPolicy:
    from .transforms import Sanitize, compile_transforms  # noqa: PLC0415

    signature = _sanitization_policy_signature(policy)
    compiled_policy = policy._compiled_sanitize_policy
    if compiled_policy is None or compiled_policy.signature != signature:
        compiled_transforms = tuple(
            compile_transforms(
                (Sanitize(policy=policy),),
            )
        )
        compiled_policy = CompiledSanitizationPolicy(
            policy=policy,
            signature=signature,
            transforms=compiled_transforms,
        )
        object.__setattr__(policy, "_compiled_sanitize_transforms", compiled_transforms)
        object.__setattr__(policy, "_compiled_sanitize_signature", signature)
        object.__setattr__(policy, "_compiled_sanitize_policy", compiled_policy)
    return compiled_policy


def _compiled_sanitize_transforms_for_policy(policy: SanitizationPolicy) -> tuple[Any, ...]:
    return policy.compile().transforms


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
