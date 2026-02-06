"""Public transform specifications for JustHTML.

This module contains the user-facing transform dataclasses and enums.

The transform compiler/runtime (compile_transforms/apply_compiled_transforms)
remains in `justhtml.transforms` to preserve the public import paths:

  from justhtml.transforms import Drop, compile_transforms, apply_compiled_transforms
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, ClassVar

from .constants import WHITESPACE_PRESERVING_ELEMENTS

if TYPE_CHECKING:
    from collections.abc import Callable, Collection
    from typing import Any, Protocol

    from .node import Node
    from .sanitize import SanitizationPolicy, UrlPolicy

    class NodeCallback(Protocol):
        def __call__(self, node: Node) -> None: ...

    class EditAttrsCallback(Protocol):
        def __call__(self, node: Node) -> dict[str, str | None] | None: ...

    class ReportCallback(Protocol):
        def __call__(self, msg: str, *, node: Any | None = None) -> None: ...


class _StrEnum(str, Enum):
    """Backport of enum.StrEnum (Python 3.11+).

    We support Python 3.10+, so we use this small mixin instead.
    """


class DecideAction(_StrEnum):
    KEEP = "keep"
    DROP = "drop"
    UNWRAP = "unwrap"
    EMPTY = "empty"
    ESCAPE = "escape"


@dataclass(frozen=True, slots=True)
class SetAttrs:
    selector: str
    attrs: dict[str, str | None]
    enabled: bool
    callback: NodeCallback | None
    report: ReportCallback | None

    def __init__(
        self,
        selector: str,
        *,
        enabled: bool = True,
        callback: NodeCallback | None = None,
        report: ReportCallback | None = None,
        attributes: dict[str, str | None] | None = None,
        **attrs: str | None,
    ) -> None:
        object.__setattr__(self, "selector", str(selector))
        merged = dict(attributes) if attributes else {}
        merged.update(attrs)
        object.__setattr__(self, "attrs", merged)
        object.__setattr__(self, "enabled", bool(enabled))
        object.__setattr__(self, "callback", callback)
        object.__setattr__(self, "report", report)


@dataclass(frozen=True, slots=True)
class Drop:
    selector: str

    enabled: bool
    callback: NodeCallback | None
    report: ReportCallback | None

    def __init__(
        self,
        selector: str,
        *,
        enabled: bool = True,
        callback: NodeCallback | None = None,
        report: ReportCallback | None = None,
    ) -> None:
        object.__setattr__(self, "selector", str(selector))
        object.__setattr__(self, "enabled", bool(enabled))
        object.__setattr__(self, "callback", callback)
        object.__setattr__(self, "report", report)


@dataclass(frozen=True, slots=True)
class Unwrap:
    selector: str

    enabled: bool
    callback: NodeCallback | None
    report: ReportCallback | None

    def __init__(
        self,
        selector: str,
        *,
        enabled: bool = True,
        callback: NodeCallback | None = None,
        report: ReportCallback | None = None,
    ) -> None:
        object.__setattr__(self, "selector", str(selector))
        object.__setattr__(self, "enabled", bool(enabled))
        object.__setattr__(self, "callback", callback)
        object.__setattr__(self, "report", report)


@dataclass(frozen=True, slots=True)
class Escape:
    """Escape the matching element's tags, but keep its children.

    This replaces e.g. `<div>hi</div>` with the text nodes `<div>` and `</div>`
    (which will be HTML-escaped on serialization), while hoisting the element
    children in between.

    This is useful when you want to preserve "what the input looked like" for a
    specific tag without rendering it as markup.
    """

    selector: str

    enabled: bool
    callback: NodeCallback | None
    report: ReportCallback | None

    def __init__(
        self,
        selector: str,
        *,
        enabled: bool = True,
        callback: NodeCallback | None = None,
        report: ReportCallback | None = None,
    ) -> None:
        object.__setattr__(self, "selector", str(selector))
        object.__setattr__(self, "enabled", bool(enabled))
        object.__setattr__(self, "callback", callback)
        object.__setattr__(self, "report", report)


@dataclass(frozen=True, slots=True)
class Empty:
    selector: str

    enabled: bool
    callback: NodeCallback | None
    report: ReportCallback | None

    def __init__(
        self,
        selector: str,
        *,
        enabled: bool = True,
        callback: NodeCallback | None = None,
        report: ReportCallback | None = None,
    ) -> None:
        object.__setattr__(self, "selector", str(selector))
        object.__setattr__(self, "enabled", bool(enabled))
        object.__setattr__(self, "callback", callback)
        object.__setattr__(self, "report", report)


@dataclass(frozen=True, slots=True)
class Edit:
    selector: str
    func: NodeCallback
    enabled: bool
    callback: NodeCallback | None
    report: ReportCallback | None

    def __init__(
        self,
        selector: str,
        func: NodeCallback,
        *,
        enabled: bool = True,
        callback: NodeCallback | None = None,
        report: ReportCallback | None = None,
    ) -> None:
        object.__setattr__(self, "selector", str(selector))
        object.__setattr__(self, "func", func)
        object.__setattr__(self, "enabled", bool(enabled))
        object.__setattr__(self, "callback", callback)
        object.__setattr__(self, "report", report)


@dataclass(frozen=True, slots=True)
class EditDocument:
    """Edit the document root in-place.

    The callback is invoked exactly once with the provided root node.

    This is intended for operations that need access to the root container
    (e.g. #document / #document-fragment) which selector-based transforms do
    not visit.
    """

    func: NodeCallback
    enabled: bool
    callback: NodeCallback | None
    report: ReportCallback | None

    def __init__(
        self,
        func: NodeCallback,
        *,
        enabled: bool = True,
        callback: NodeCallback | None = None,
        report: ReportCallback | None = None,
    ) -> None:
        object.__setattr__(self, "func", func)
        object.__setattr__(self, "enabled", bool(enabled))
        object.__setattr__(self, "callback", callback)
        object.__setattr__(self, "report", report)


@dataclass(frozen=True, slots=True)
class Decide:
    """Perform structural actions based on a callback.

    This is a generic building block for policy-driven transforms.

    - For selectors other than "*", the selector is matched against element
        nodes using the normal selector engine.
    - For selector "*", the callback is invoked for every node type, including
        text/comment/doctype and document container nodes.

    The callback must return one of: Decide.KEEP, Decide.DROP, Decide.UNWRAP, Decide.EMPTY, Decide.ESCAPE.
    """

    selector: str
    func: Callable[[Node], DecideAction]
    enabled: bool
    callback: NodeCallback | None
    report: ReportCallback | None

    KEEP: ClassVar[DecideAction] = DecideAction.KEEP
    DROP: ClassVar[DecideAction] = DecideAction.DROP
    UNWRAP: ClassVar[DecideAction] = DecideAction.UNWRAP
    EMPTY: ClassVar[DecideAction] = DecideAction.EMPTY
    ESCAPE: ClassVar[DecideAction] = DecideAction.ESCAPE

    def __init__(
        self,
        selector: str,
        func: Callable[[Node], DecideAction],
        *,
        enabled: bool = True,
        callback: NodeCallback | None = None,
        report: ReportCallback | None = None,
    ) -> None:
        object.__setattr__(self, "selector", str(selector))
        object.__setattr__(self, "func", func)
        object.__setattr__(self, "enabled", bool(enabled))
        object.__setattr__(self, "callback", callback)
        object.__setattr__(self, "report", report)


@dataclass(frozen=True, slots=True)
class EditAttrs:
    """Edit element attributes using a callback.

    The callback is invoked for matching element/template nodes.

    - Return None to leave attributes unchanged.
    - Return a dict to replace the node's attributes with that dict.
    """

    selector: str
    func: EditAttrsCallback
    enabled: bool
    callback: NodeCallback | None
    report: ReportCallback | None

    def __init__(
        self,
        selector: str,
        func: EditAttrsCallback,
        *,
        enabled: bool = True,
        callback: NodeCallback | None = None,
        report: ReportCallback | None = None,
    ) -> None:
        object.__setattr__(self, "selector", str(selector))
        object.__setattr__(self, "func", func)
        object.__setattr__(self, "enabled", bool(enabled))
        object.__setattr__(self, "callback", callback)
        object.__setattr__(self, "report", report)


# Backwards-compatible alias.
RewriteAttrs = EditAttrs


@dataclass(frozen=True, slots=True)
class Linkify:
    """Linkify URLs/emails in text nodes.

    This transform scans DOM text nodes (not raw HTML strings) and wraps detected
    links in `<a href="...">...</a>`.
    """

    skip_tags: frozenset[str]
    fuzzy_ip: bool
    extra_tlds: frozenset[str]
    enabled: bool
    callback: NodeCallback | None
    report: ReportCallback | None

    def __init__(
        self,
        *,
        skip_tags: list[str] | tuple[str, ...] | set[str] | frozenset[str] = (
            "a",
            *WHITESPACE_PRESERVING_ELEMENTS,
        ),
        enabled: bool = True,
        fuzzy_ip: bool = False,
        extra_tlds: list[str] | tuple[str, ...] | set[str] | frozenset[str] = (),
        callback: NodeCallback | None = None,
        report: ReportCallback | None = None,
    ) -> None:
        object.__setattr__(self, "skip_tags", frozenset(str(t).lower() for t in skip_tags))
        object.__setattr__(self, "fuzzy_ip", bool(fuzzy_ip))
        object.__setattr__(self, "extra_tlds", frozenset(str(t).lower() for t in extra_tlds))
        object.__setattr__(self, "enabled", bool(enabled))
        object.__setattr__(self, "callback", callback)
        object.__setattr__(self, "report", report)


@dataclass(frozen=True, slots=True)
class CollapseWhitespace:
    """Collapse whitespace in text nodes.

    Collapses runs of HTML whitespace characters (space, tab, LF, CR, FF) into a
    single space.

    This is similar to `html5lib.filters.whitespace.Filter`.
    """

    skip_tags: frozenset[str]
    enabled: bool
    callback: NodeCallback | None
    report: ReportCallback | None

    def __init__(
        self,
        *,
        skip_tags: list[str] | tuple[str, ...] | set[str] | frozenset[str] = (
            *WHITESPACE_PRESERVING_ELEMENTS,
            "title",
        ),
        enabled: bool = True,
        callback: NodeCallback | None = None,
        report: ReportCallback | None = None,
    ) -> None:
        object.__setattr__(self, "skip_tags", frozenset(str(t).lower() for t in skip_tags))
        object.__setattr__(self, "enabled", bool(enabled))
        object.__setattr__(self, "callback", callback)
        object.__setattr__(self, "report", report)


@dataclass(frozen=True, slots=True)
class Sanitize:
    """Sanitize the in-memory tree.

    This transform replaces the current tree with a sanitized clone using the
    same sanitizer that powers JustHTML's safe-by-default construction
    (`sanitize=True`).

    Notes:
    - This runs once at parse/transform time.
        - If you apply transforms after `Sanitize`, they may reintroduce unsafe
            content. Keep `sanitize=True` if you need output safety.
    """

    policy: SanitizationPolicy | None
    enabled: bool
    callback: NodeCallback | None
    report: ReportCallback | None

    def __init__(
        self,
        policy: SanitizationPolicy | None = None,
        *,
        enabled: bool = True,
        callback: NodeCallback | None = None,
        report: ReportCallback | None = None,
    ) -> None:
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "enabled", bool(enabled))
        object.__setattr__(self, "callback", callback)
        object.__setattr__(self, "report", report)


@dataclass(frozen=True, slots=True)
class DropComments:
    """Drop comment nodes (#comment)."""

    enabled: bool
    callback: NodeCallback | None
    report: ReportCallback | None

    def __init__(
        self,
        *,
        enabled: bool = True,
        callback: NodeCallback | None = None,
        report: ReportCallback | None = None,
    ) -> None:
        object.__setattr__(self, "enabled", bool(enabled))
        object.__setattr__(self, "callback", callback)
        object.__setattr__(self, "report", report)


@dataclass(frozen=True, slots=True)
class DropDoctype:
    """Drop doctype nodes (!doctype)."""

    enabled: bool
    callback: NodeCallback | None
    report: ReportCallback | None

    def __init__(
        self,
        *,
        enabled: bool = True,
        callback: NodeCallback | None = None,
        report: ReportCallback | None = None,
    ) -> None:
        object.__setattr__(self, "enabled", bool(enabled))
        object.__setattr__(self, "callback", callback)
        object.__setattr__(self, "report", report)


@dataclass(frozen=True, slots=True)
class DropForeignNamespaces:
    """Drop elements in non-HTML namespaces."""

    enabled: bool
    callback: NodeCallback | None
    report: ReportCallback | None

    def __init__(
        self,
        *,
        enabled: bool = True,
        callback: NodeCallback | None = None,
        report: ReportCallback | None = None,
    ) -> None:
        object.__setattr__(self, "enabled", bool(enabled))
        object.__setattr__(self, "callback", callback)
        object.__setattr__(self, "report", report)


@dataclass(frozen=True, slots=True)
class DropAttrs:
    """Drop attributes whose names match simple patterns."""

    selector: str
    patterns: tuple[str, ...]
    enabled: bool
    callback: NodeCallback | None
    report: ReportCallback | None

    def __init__(
        self,
        selector: str,
        *,
        patterns: tuple[str, ...] = (),
        enabled: bool = True,
        callback: NodeCallback | None = None,
        report: ReportCallback | None = None,
    ) -> None:
        object.__setattr__(self, "selector", str(selector))
        object.__setattr__(
            self,
            "patterns",
            tuple(sorted({str(p).strip().lower() for p in patterns if str(p).strip()})),
        )
        object.__setattr__(self, "enabled", bool(enabled))
        object.__setattr__(self, "callback", callback)
        object.__setattr__(self, "report", report)


@dataclass(frozen=True, slots=True)
class AllowlistAttrs:
    """Retain only allowlisted attributes by tag and global allowlist."""

    selector: str
    allowed_attributes: dict[str, set[str]]
    enabled: bool
    callback: NodeCallback | None
    report: ReportCallback | None

    def __init__(
        self,
        selector: str,
        *,
        allowed_attributes: dict[str, Collection[str]],
        enabled: bool = True,
        callback: NodeCallback | None = None,
        report: ReportCallback | None = None,
    ) -> None:
        normalized: dict[str, set[str]] = {}
        for tag, attrs in allowed_attributes.items():
            normalized[str(tag)] = {str(a).lower() for a in attrs}
        object.__setattr__(self, "selector", str(selector))
        object.__setattr__(self, "allowed_attributes", normalized)
        object.__setattr__(self, "enabled", bool(enabled))
        object.__setattr__(self, "callback", callback)
        object.__setattr__(self, "report", report)


@dataclass(frozen=True, slots=True)
class DropUrlAttrs:
    """Validate and rewrite/drop URL-valued attributes based on UrlPolicy rules."""

    selector: str
    url_policy: UrlPolicy
    enabled: bool
    callback: NodeCallback | None
    report: ReportCallback | None

    def __init__(
        self,
        selector: str,
        *,
        url_policy: UrlPolicy,
        enabled: bool = True,
        callback: NodeCallback | None = None,
        report: ReportCallback | None = None,
    ) -> None:
        object.__setattr__(self, "selector", str(selector))
        object.__setattr__(self, "url_policy", url_policy)
        object.__setattr__(self, "enabled", bool(enabled))
        object.__setattr__(self, "callback", callback)
        object.__setattr__(self, "report", report)


@dataclass(frozen=True, slots=True)
class AllowStyleAttrs:
    """Sanitize inline style attributes when present."""

    selector: str
    allowed_css_properties: tuple[str, ...]
    enabled: bool
    callback: NodeCallback | None
    report: ReportCallback | None

    def __init__(
        self,
        selector: str,
        *,
        allowed_css_properties: Collection[str],
        enabled: bool = True,
        callback: NodeCallback | None = None,
        report: ReportCallback | None = None,
    ) -> None:
        object.__setattr__(self, "selector", str(selector))
        object.__setattr__(
            self,
            "allowed_css_properties",
            tuple(sorted({str(p).strip().lower() for p in allowed_css_properties if str(p).strip()})),
        )
        object.__setattr__(self, "enabled", bool(enabled))
        object.__setattr__(self, "callback", callback)
        object.__setattr__(self, "report", report)


@dataclass(frozen=True, slots=True)
class MergeAttrs:
    """Merge tokens into a whitespace-delimited attribute without removing existing ones."""

    tag: str
    attr: str
    tokens: tuple[str, ...]
    enabled: bool
    callback: NodeCallback | None
    report: ReportCallback | None

    def __init__(
        self,
        tag: str,
        *,
        attr: str,
        tokens: Collection[str],
        enabled: bool = True,
        callback: NodeCallback | None = None,
        report: ReportCallback | None = None,
    ) -> None:
        object.__setattr__(self, "tag", str(tag).lower())
        object.__setattr__(self, "attr", str(attr).lower())
        object.__setattr__(self, "tokens", tuple(sorted({str(t).strip().lower() for t in tokens if str(t).strip()})))
        object.__setattr__(self, "enabled", bool(enabled))
        object.__setattr__(self, "callback", callback)
        object.__setattr__(self, "report", report)


@dataclass(frozen=True, slots=True)
class PruneEmpty:
    """Recursively drop empty elements.

    This transform removes elements that are empty at that point in the
    transform pipeline.

    "Empty" means:
    - no element children, and
    - no non-whitespace text nodes (unless `strip_whitespace=False`).

    Comments/doctypes are ignored when determining emptiness.

    Notes:
    - Pruning uses a post-order traversal to be correct.
    """

    selector: str
    strip_whitespace: bool
    enabled: bool
    callback: NodeCallback | None
    report: ReportCallback | None

    def __init__(
        self,
        selector: str,
        *,
        strip_whitespace: bool = True,
        enabled: bool = True,
        callback: NodeCallback | None = None,
        report: ReportCallback | None = None,
    ) -> None:
        object.__setattr__(self, "selector", str(selector))
        object.__setattr__(self, "strip_whitespace", bool(strip_whitespace))
        object.__setattr__(self, "enabled", bool(enabled))
        object.__setattr__(self, "callback", callback)
        object.__setattr__(self, "report", report)


__all__ = [
    "AllowStyleAttrs",
    "AllowlistAttrs",
    "CollapseWhitespace",
    "Decide",
    "DecideAction",
    "Drop",
    "DropAttrs",
    "DropComments",
    "DropDoctype",
    "DropForeignNamespaces",
    "DropUrlAttrs",
    "Edit",
    "EditAttrs",
    "EditDocument",
    "Empty",
    "Escape",
    "Linkify",
    "MergeAttrs",
    "PruneEmpty",
    "RewriteAttrs",
    "Sanitize",
    "SetAttrs",
    "Unwrap",
]
