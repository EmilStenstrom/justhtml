"""Constructor-time DOM transforms.

These transforms are intended as a migration path for Bleach/html5lib-style
post-processing, but are implemented as DOM (tree) operations to match
JustHTML's architecture.

Safety model: transforms shape the in-memory tree; safe-by-default output is
still enforced by `to_html()`/`to_text()`/`to_markdown()` via sanitization.

Performance: selectors are compiled (parsed) once before application.
"""

from __future__ import annotations

import re
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, Literal, cast

from .constants import VOID_ELEMENTS, WHITESPACE_PRESERVING_ELEMENTS
from .linkify import LinkifyConfig, find_links_with_config
from .node import Element, Node, Template, Text
from .sanitize import (
    _URL_LIKE_ATTRS,
    DEFAULT_POLICY,
    SanitizationPolicy,
    UrlPolicy,
    _effective_allow_relative,
    _effective_proxy,
    _effective_url_handling,
    _sanitize_inline_style,
    _sanitize_srcset_value,
    _sanitize_url_value_with_rule,
)
from .selector import SelectorMatcher, parse_selector
from .serialize import serialize_end_tag, serialize_start_tag
from .tokens import ParseError

if TYPE_CHECKING:
    from collections.abc import Callable, Collection
    from typing import Any, Protocol

    from .selector import ParsedSelector

    class NodeCallback(Protocol):
        def __call__(self, node: Node) -> None: ...

    class EditAttrsCallback(Protocol):
        def __call__(self, node: Node) -> dict[str, str | None] | None: ...

    class ReportCallback(Protocol):
        def __call__(self, msg: str, *, node: Any | None = None) -> None: ...


# -----------------
# Public transforms
# -----------------


_ERROR_SINK: ContextVar[list[ParseError] | None] = ContextVar("justhtml_transform_error_sink", default=None)


def emit_error(
    code: str,
    *,
    node: Node | None = None,
    line: int | None = None,
    column: int | None = None,
    category: str = "transform",
    message: str | None = None,
) -> None:
    """Emit a ParseError from within a transform callback.

    Errors are appended to the active sink when transforms are applied (e.g.
    during JustHTML construction). If no sink is active, this is a no-op.
    """

    sink = _ERROR_SINK.get()
    if sink is None:
        return

    if node is not None:
        line = node.origin_line
        column = node.origin_col

    sink.append(
        ParseError(
            str(code),
            line=line,
            column=column,
            category=str(category),
            message=str(message) if message is not None else str(code),
        )
    )


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


def _collapse_html_space_characters(text: str) -> str:
    """Collapse runs of HTML whitespace characters to a single space.

    This mirrors html5lib's whitespace filter behavior: it does not trim.
    """

    # Fast path: no formatting whitespace and no double spaces.
    if "\t" not in text and "\n" not in text and "\r" not in text and "\f" not in text and "  " not in text:
        return text

    out: list[str] = []
    in_ws = False

    for ch in text:
        if ch == " " or ch == "\t" or ch == "\n" or ch == "\r" or ch == "\f":
            if in_ws:
                continue
            out.append(" ")
            in_ws = True
            continue

        out.append(ch)
        in_ws = False
    return "".join(out)


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


@dataclass(frozen=True, slots=True)
class Stage:
    """Group transforms into an explicit stage.

    Stages are intended to make transform passes explicit and readable.

    - Stages can be nested; nested stages are flattened.
    - If at least one Stage is present at the top level of a transform list,
        any top-level transforms around it are automatically grouped into
        implicit stages.
    """

    transforms: tuple[TransformSpec, ...]
    enabled: bool
    callback: NodeCallback | None
    report: ReportCallback | None

    def __init__(
        self,
        transforms: list[TransformSpec] | tuple[TransformSpec, ...],
        *,
        enabled: bool = True,
        callback: NodeCallback | None = None,
        report: ReportCallback | None = None,
    ) -> None:
        object.__setattr__(self, "transforms", tuple(transforms))
        object.__setattr__(self, "enabled", bool(enabled))
        object.__setattr__(self, "callback", callback)
        object.__setattr__(self, "report", report)


# -----------------
# Compilation
# -----------------


Transform = (
    SetAttrs
    | Drop
    | Unwrap
    | Escape
    | Empty
    | Edit
    | EditDocument
    | Decide
    | EditAttrs
    | Linkify
    | CollapseWhitespace
    | PruneEmpty
    | Sanitize
    | DropComments
    | DropDoctype
    | DropForeignNamespaces
    | DropAttrs
    | AllowlistAttrs
    | DropUrlAttrs
    | AllowStyleAttrs
    | MergeAttrs
)


_TRANSFORM_CLASSES: tuple[type[object], ...] = (
    SetAttrs,
    Drop,
    Unwrap,
    Escape,
    Empty,
    Edit,
    EditDocument,
    Decide,
    EditAttrs,
    Linkify,
    CollapseWhitespace,
    PruneEmpty,
    Sanitize,
    DropComments,
    DropDoctype,
    DropForeignNamespaces,
    DropAttrs,
    AllowlistAttrs,
    DropUrlAttrs,
    AllowStyleAttrs,
    MergeAttrs,
)

TransformSpec = Transform | Stage


@dataclass(frozen=True, slots=True)
class _CompiledCollapseWhitespaceTransform:
    kind: Literal["collapse_whitespace"]
    skip_tags: frozenset[str]
    callback: NodeCallback | None
    report: ReportCallback | None


@dataclass(frozen=True, slots=True)
class _CompiledSelectorTransform:
    kind: Literal["setattrs", "drop", "unwrap", "escape", "empty", "edit"]
    selector_str: str
    selector: ParsedSelector
    payload: dict[str, str | None] | NodeCallback | None
    callback: NodeCallback | None
    report: ReportCallback | None


@dataclass(frozen=True, slots=True)
class _CompiledLinkifyTransform:
    kind: Literal["linkify"]
    skip_tags: frozenset[str]
    config: LinkifyConfig
    callback: NodeCallback | None
    report: ReportCallback | None


@dataclass(frozen=True, slots=True)
class _CompiledEditDocumentTransform:
    kind: Literal["edit_document"]
    callback: NodeCallback


@dataclass(frozen=True, slots=True)
class _CompiledPruneEmptyTransform:
    kind: Literal["prune_empty"]
    selector_str: str
    selector: ParsedSelector
    strip_whitespace: bool
    callback: NodeCallback | None
    report: ReportCallback | None


@dataclass(frozen=True, slots=True)
class _CompiledStageBoundary:
    kind: Literal["stage_boundary"]


@dataclass(frozen=True, slots=True)
class _CompiledDecideTransform:
    kind: Literal["decide"]
    selector_str: str
    selector: ParsedSelector | None
    all_nodes: bool
    callback: Callable[[Node], DecideAction]


@dataclass(frozen=True, slots=True)
class _CompiledRewriteAttrsTransform:
    kind: Literal["rewrite_attrs"]
    selector_str: str
    selector: ParsedSelector | None
    all_nodes: bool
    func: EditAttrsCallback


@dataclass(frozen=True, slots=True)
class _CompiledDropCommentsTransform:
    kind: Literal["drop_comments"]
    callback: NodeCallback | None
    report: ReportCallback | None


@dataclass(frozen=True, slots=True)
class _CompiledDropDoctypeTransform:
    kind: Literal["drop_doctype"]
    callback: NodeCallback | None
    report: ReportCallback | None


@dataclass(frozen=True, slots=True)
class _CompiledMergeAttrTokensTransform:
    kind: Literal["merge_attr_tokens"]
    tag: str
    attr: str
    tokens: tuple[str, ...]
    callback: NodeCallback | None
    report: ReportCallback | None


@dataclass(frozen=True, slots=True)
class _CompiledStageHookTransform:
    kind: Literal["stage_hook"]
    index: int
    callback: NodeCallback | None
    report: ReportCallback | None


CompiledTransform = (
    _CompiledSelectorTransform
    | _CompiledDecideTransform
    | _CompiledRewriteAttrsTransform
    | _CompiledLinkifyTransform
    | _CompiledCollapseWhitespaceTransform
    | _CompiledPruneEmptyTransform
    | _CompiledEditDocumentTransform
    | _CompiledDropCommentsTransform
    | _CompiledDropDoctypeTransform
    | _CompiledMergeAttrTokensTransform
    | _CompiledStageHookTransform
    | _CompiledStageBoundary
)


def _iter_flattened_transforms(specs: list[TransformSpec] | tuple[TransformSpec, ...]) -> list[Transform]:
    out: list[Transform] = []

    def _walk(items: list[TransformSpec] | tuple[TransformSpec, ...]) -> None:
        for item in items:
            if isinstance(item, Stage):
                if item.enabled:
                    _walk(item.transforms)
                continue
            out.append(item)

    _walk(specs)
    return out


def _glob_match(pattern: str, text: str) -> bool:
    """Match a glob pattern against text.

    Supported wildcards:
    - '*' matches any sequence (including empty)
    - '?' matches any single character
    """

    if pattern == "*":
        return True
    if "*" not in pattern and "?" not in pattern:
        return pattern == text

    p_i = 0
    t_i = 0
    star_i = -1
    match_i = 0

    while t_i < len(text):
        if p_i < len(pattern) and (pattern[p_i] == "?" or pattern[p_i] == text[t_i]):
            p_i += 1
            t_i += 1
            continue

        if p_i < len(pattern) and pattern[p_i] == "*":
            star_i = p_i
            match_i = t_i
            p_i += 1
            continue

        if star_i != -1:
            p_i = star_i + 1
            match_i += 1
            t_i = match_i
            continue

        return False

    while p_i < len(pattern) and pattern[p_i] == "*":
        p_i += 1

    return p_i == len(pattern)


def _split_into_top_level_stages(specs: list[TransformSpec] | tuple[TransformSpec, ...]) -> list[Stage]:
    # Only enable auto-staging when a Stage is present at the top level.
    has_top_level_stage = any(isinstance(t, Stage) and t.enabled for t in specs)
    if not has_top_level_stage:
        return []

    stages: list[Stage] = []
    pending: list[TransformSpec] = []

    for item in specs:
        if isinstance(item, Stage):
            if not item.enabled:
                continue
            if pending:
                stages.append(Stage(pending))
                pending = []
            stages.append(item)
            continue

        pending.append(item)

    if pending:
        stages.append(Stage(pending))

    return stages


def compile_transforms(transforms: list[TransformSpec] | tuple[TransformSpec, ...]) -> list[CompiledTransform]:
    if not transforms:
        return []

    flattened = _iter_flattened_transforms(transforms)

    top_level_stages = _split_into_top_level_stages(transforms)
    if top_level_stages:
        # Stage is a pass boundary. Compile each stage separately and insert a
        # boundary marker so apply_compiled_transforms can flush batches.
        compiled_stage: list[CompiledTransform] = []
        for stage_i, stage in enumerate(top_level_stages):
            if stage_i:
                compiled_stage.append(_CompiledStageBoundary(kind="stage_boundary"))
            compiled_stage.append(
                _CompiledStageHookTransform(
                    kind="stage_hook",
                    index=stage_i,
                    callback=stage.callback,
                    report=stage.report,
                )
            )
            for inner in _iter_flattened_transforms(stage.transforms):
                compiled_stage.extend(compile_transforms((inner,)))
        return compiled_stage

    compiled: list[CompiledTransform] = []

    def _append_compiled(item: CompiledTransform) -> None:
        # Optimization: fuse adjacent EditAttrs transforms that target the
        # same selector. This preserves left-to-right semantics but reduces
        # per-node selector matching and callback overhead.
        if (
            compiled
            and isinstance(item, _CompiledRewriteAttrsTransform)
            and isinstance(compiled[-1], _CompiledRewriteAttrsTransform)
        ):
            prev = compiled[-1]
            if prev.selector_str == item.selector_str and prev.all_nodes == item.all_nodes:
                prev_cb = prev.func
                next_cb = item.func

                def _chained(
                    node: Node,
                    prev_cb: Callable[[Node], dict[str, str | None] | None] = prev_cb,
                    next_cb: Callable[[Node], dict[str, str | None] | None] = next_cb,
                ) -> dict[str, str | None] | None:
                    changed = False
                    out = prev_cb(node)
                    if out is not None:  # pragma: no cover
                        node.attrs = out
                        changed = True
                    out = next_cb(node)
                    if out is not None:
                        node.attrs = out
                        changed = True
                    return node.attrs if changed else None

                compiled[-1] = _CompiledRewriteAttrsTransform(
                    kind="rewrite_attrs",
                    selector_str=prev.selector_str,
                    selector=prev.selector,
                    all_nodes=prev.all_nodes,
                    func=_chained,
                )
                return

        compiled.append(item)

    for t in flattened:
        if not isinstance(t, _TRANSFORM_CLASSES):
            raise TypeError(f"Unsupported transform: {type(t).__name__}")
        if not t.enabled:
            continue
        if isinstance(t, SetAttrs):
            compiled.append(
                _CompiledSelectorTransform(
                    kind="setattrs",
                    selector_str=t.selector,
                    selector=parse_selector(t.selector),
                    payload=t.attrs,
                    callback=t.callback,
                    report=t.report,
                )
            )
            continue
        if isinstance(t, Drop):
            selector_str = t.selector

            # Fast-path: if selector is a simple comma-separated list of tag
            # names (e.g. "script, style"), avoid selector matching entirely.
            raw_parts = selector_str.split(",")
            tag_list: list[str] = []
            for part in raw_parts:
                p = part.strip().lower()
                if not p:
                    tag_list = []
                    break
                # Reject anything that isn't a plain tag name.
                if any(ch in p for ch in " .#[:>*+~\t\n\r\f"):
                    tag_list = []
                    break
                tag_list.append(p)

            if tag_list:
                tags = frozenset(tag_list)
                on_drop = t.callback
                on_report = t.report

                def _drop_if_tag(
                    node: Node,
                    tags: frozenset[str] = tags,
                    selector_str: str = selector_str,
                    on_drop: NodeCallback | None = on_drop,
                    on_report: ReportCallback | None = on_report,
                ) -> DecideAction:
                    name = node.name
                    if name.startswith("#") or name == "!doctype":
                        return Decide.KEEP
                    tag = str(name).lower()
                    if tag not in tags:
                        return Decide.KEEP
                    if on_drop is not None:
                        on_drop(node)
                    if on_report is not None:
                        on_report(f"Dropped tag '{tag}' (matched selector '{selector_str}')", node=node)
                    return Decide.DROP

                compiled.append(
                    _CompiledDecideTransform(
                        kind="decide",
                        selector_str="*",
                        selector=None,
                        all_nodes=True,
                        callback=_drop_if_tag,
                    )
                )
                continue

            compiled.append(
                _CompiledSelectorTransform(
                    kind="drop",
                    selector_str=selector_str,
                    selector=parse_selector(selector_str),
                    payload=None,
                    callback=t.callback,
                    report=t.report,
                )
            )
            continue
        if isinstance(t, Unwrap):
            compiled.append(
                _CompiledSelectorTransform(
                    kind="unwrap",
                    selector_str=t.selector,
                    selector=parse_selector(t.selector),
                    payload=None,
                    callback=t.callback,
                    report=t.report,
                )
            )
            continue

        if isinstance(t, Escape):
            compiled.append(
                _CompiledSelectorTransform(
                    kind="escape",
                    selector_str=t.selector,
                    selector=parse_selector(t.selector),
                    payload=None,
                    callback=t.callback,
                    report=t.report,
                )
            )
            continue
        if isinstance(t, Empty):
            compiled.append(
                _CompiledSelectorTransform(
                    kind="empty",
                    selector_str=t.selector,
                    selector=parse_selector(t.selector),
                    payload=None,
                    callback=t.callback,
                    report=t.report,
                )
            )
            continue
        if isinstance(t, Edit):
            selector_str = t.selector
            edit_func = t.func
            on_hook = t.callback
            on_report = t.report

            def _wrapped(
                node: Node,
                edit_func: NodeCallback = edit_func,
                selector_str: str = selector_str,
                on_hook: NodeCallback | None = on_hook,
                on_report: ReportCallback | None = on_report,
            ) -> None:
                if on_hook is not None:
                    on_hook(node)
                if on_report is not None:
                    tag = str(node.name).lower()
                    on_report(f"Edited <{tag}> (matched selector '{selector_str}')", node=node)
                edit_func(node)

            compiled.append(
                _CompiledSelectorTransform(
                    kind="edit",
                    selector_str=t.selector,
                    selector=parse_selector(t.selector),
                    payload=_wrapped,
                    callback=None,
                    report=None,
                )
            )
            continue

        if isinstance(t, EditDocument):
            edit_document_func = t.func
            on_hook = t.callback
            on_report = t.report

            def _wrapped_root(
                node: Node,
                edit_document_func: NodeCallback = edit_document_func,
                on_hook: NodeCallback | None = on_hook,
                on_report: ReportCallback | None = on_report,
            ) -> None:
                if on_hook is not None:
                    on_hook(node)
                if on_report is not None:
                    on_report("Edited document root", node=node)
                edit_document_func(node)

            compiled.append(_CompiledEditDocumentTransform(kind="edit_document", callback=_wrapped_root))
            continue

        if isinstance(t, Decide):
            selector_str = t.selector
            all_nodes = selector_str.strip() == "*"
            decide_func = t.func
            on_hook = t.callback
            on_report = t.report

            def _wrapped_decide(
                node: Node,
                decide_func: Callable[[Node], DecideAction] = decide_func,
                selector_str: str = selector_str,
                on_hook: NodeCallback | None = on_hook,
                on_report: ReportCallback | None = on_report,
            ) -> DecideAction:
                action = decide_func(node)
                if action is DecideAction.KEEP:
                    return action
                if on_hook is not None:
                    on_hook(node)
                if on_report is not None:
                    nm = node.name
                    label = str(nm).lower() if not nm.startswith("#") and nm != "!doctype" else str(nm)
                    on_report(f"Decide -> {action.value} '{label}' (matched selector '{selector_str}')", node=node)
                return action

            compiled.append(
                _CompiledDecideTransform(
                    kind="decide",
                    selector_str=selector_str,
                    selector=None if all_nodes else parse_selector(selector_str),
                    all_nodes=all_nodes,
                    callback=_wrapped_decide,
                )
            )
            continue

        if isinstance(t, EditAttrs):
            selector_str = t.selector
            all_nodes = selector_str.strip() == "*"
            edit_attrs_func = t.func
            on_hook = t.callback
            on_report = t.report

            def _wrapped_attrs(
                node: Node,
                edit_attrs_func: EditAttrsCallback = edit_attrs_func,
                selector_str: str = selector_str,
                on_hook: NodeCallback | None = on_hook,
                on_report: ReportCallback | None = on_report,
            ) -> dict[str, str | None] | None:
                out = edit_attrs_func(node)
                if out is None:
                    return None
                if on_hook is not None:
                    on_hook(node)
                if on_report is not None:
                    tag = str(node.name).lower()
                    on_report(f"Edited attributes on <{tag}> (matched selector '{selector_str}')", node=node)
                return out

            _append_compiled(
                _CompiledRewriteAttrsTransform(
                    kind="rewrite_attrs",
                    selector_str=selector_str,
                    selector=None if all_nodes else parse_selector(selector_str),
                    all_nodes=all_nodes,
                    func=_wrapped_attrs,
                )
            )
            continue

        if isinstance(t, Linkify):
            compiled.append(
                _CompiledLinkifyTransform(
                    kind="linkify",
                    skip_tags=t.skip_tags,
                    config=LinkifyConfig(fuzzy_ip=t.fuzzy_ip, extra_tlds=t.extra_tlds),
                    callback=t.callback,
                    report=t.report,
                )
            )
            continue

        if isinstance(t, CollapseWhitespace):
            compiled.append(
                _CompiledCollapseWhitespaceTransform(
                    kind="collapse_whitespace",
                    skip_tags=t.skip_tags,
                    callback=t.callback,
                    report=t.report,
                )
            )
            continue

        if isinstance(t, PruneEmpty):
            compiled.append(
                _CompiledPruneEmptyTransform(
                    kind="prune_empty",
                    selector_str=t.selector,
                    selector=parse_selector(t.selector),
                    strip_whitespace=t.strip_whitespace,
                    callback=t.callback,
                    report=t.report,
                )
            )
            continue

        if isinstance(t, DropComments):
            compiled.append(
                _CompiledDropCommentsTransform(
                    kind="drop_comments",
                    callback=t.callback,
                    report=t.report,
                )
            )
            continue

        if isinstance(t, DropDoctype):
            compiled.append(
                _CompiledDropDoctypeTransform(
                    kind="drop_doctype",
                    callback=t.callback,
                    report=t.report,
                )
            )
            continue

        if isinstance(t, DropForeignNamespaces):
            on_hook = t.callback
            on_report = t.report

            def _drop_foreign(
                node: Node,
                on_hook: NodeCallback | None = on_hook,
                on_report: ReportCallback | None = on_report,
            ) -> DecideAction:
                name = node.name
                if name.startswith("#") or name == "!doctype":
                    return Decide.KEEP
                ns = node.namespace
                if ns not in (None, "html"):
                    if on_hook is not None:
                        on_hook(node)
                    if on_report is not None:
                        tag = str(name).lower()
                        on_report(f"Unsafe tag '{tag}' (foreign namespace)", node=node)
                    return Decide.DROP
                return Decide.KEEP

            compiled.append(
                _CompiledDecideTransform(
                    kind="decide",
                    selector_str="*",
                    selector=None,
                    all_nodes=True,
                    callback=_drop_foreign,
                )
            )
            continue

        if isinstance(t, DropAttrs):
            patterns = t.patterns
            on_hook = t.callback
            on_report = t.report

            # Optimize pattern matching: Compile all patterns into one regex
            compiled_regex = _compile_patterns_to_regex(patterns)

            def _drop_attrs(
                node: Node,
                patterns: tuple[str, ...] = patterns,
                compiled_regex: re.Pattern[str] | None = compiled_regex,
                on_hook: NodeCallback | None = on_hook,
                on_report: ReportCallback | None = on_report,
            ) -> dict[str, str | None] | None:
                attrs = node.attrs
                if not attrs:
                    return None

                if not patterns:
                    return None

                out: dict[str, str | None] = {}
                changed = False
                for raw_key, value in attrs.items():
                    if not raw_key or not str(raw_key).strip():
                        continue
                    key = raw_key
                    if not key.islower():
                        key = key.lower()

                    if compiled_regex and compiled_regex.match(key):
                        if on_report is not None:
                            # Re-check to report which pattern matched (rare path)
                            found_pat = "?"
                            for pat in patterns:
                                if _glob_match(pat, key):  # pragma: no cover
                                    found_pat = pat
                                    break
                            on_report(
                                f"Unsafe attribute '{key}' (matched forbidden pattern '{found_pat}')",
                                node=node,
                            )
                        changed = True
                        continue

                    out[key] = value

                if not changed:
                    return None
                if on_hook is not None:
                    on_hook(node)  # pragma: no cover
                return out

            selector_str = t.selector
            all_nodes = selector_str.strip() == "*"
            _append_compiled(
                _CompiledRewriteAttrsTransform(
                    kind="rewrite_attrs",
                    selector_str=selector_str,
                    selector=None if all_nodes else parse_selector(selector_str),
                    all_nodes=all_nodes,
                    func=_drop_attrs,
                )
            )
            continue

        if isinstance(t, AllowlistAttrs):
            allowed_attributes = t.allowed_attributes
            on_hook = t.callback
            on_report = t.report
            allowed_global = allowed_attributes.get("*", set())
            allowed_by_tag: dict[str, set[str]] = {}
            for tag, attrs in allowed_attributes.items():
                if tag == "*":
                    continue
                allowed_by_tag[str(tag).lower()] = set(allowed_global).union(attrs)

            def _allowlist_attrs(
                node: Node,
                allowed_by_tag: dict[str, set[str]] = allowed_by_tag,
                allowed_global: set[str] = allowed_global,
                on_hook: NodeCallback | None = on_hook,
                on_report: ReportCallback | None = on_report,
            ) -> dict[str, str | None] | None:
                attrs = node.attrs
                if not attrs:
                    return None
                tag = str(node.name).lower()
                allowed = allowed_by_tag.get(tag, allowed_global)

                changed = False
                out: dict[str, str | None] = {}
                for raw_key, value in attrs.items():
                    raw_key_str = str(raw_key)
                    if not raw_key_str.strip():
                        # Drop invalid attribute names like '' or whitespace-only.
                        changed = True
                        continue
                    key = raw_key_str
                    if not key.islower():
                        key = key.lower()
                        changed = True  # pragma: no cover
                    if key in allowed:
                        out[key] = value
                    else:
                        changed = True
                        if on_report is not None:
                            on_report(f"Unsafe attribute '{key}' (not allowed)", node=node)
                if not changed:
                    return None
                if on_hook is not None:
                    on_hook(node)  # pragma: no cover
                return out

            selector_str = t.selector
            all_nodes = selector_str.strip() == "*"
            _append_compiled(
                _CompiledRewriteAttrsTransform(
                    kind="rewrite_attrs",
                    selector_str=selector_str,
                    selector=None if all_nodes else parse_selector(selector_str),
                    all_nodes=all_nodes,
                    func=_allowlist_attrs,
                )
            )
            continue

        if isinstance(t, DropUrlAttrs):
            url_policy = t.url_policy
            on_hook = t.callback
            on_report = t.report

            def _drop_url_attrs(
                node: Node,
                url_policy: UrlPolicy = url_policy,
                on_hook: NodeCallback | None = on_hook,
                on_report: ReportCallback | None = on_report,
            ) -> dict[str, str | None] | None:
                attrs = node.attrs
                if not attrs:
                    return None

                tag = str(node.name).lower()
                out = dict(attrs)
                changed = False
                for key in list(out.keys()):
                    if key not in _URL_LIKE_ATTRS:
                        continue

                    raw_value = out.get(key)
                    if raw_value is None:
                        if on_report is not None:  # pragma: no cover
                            on_report(f"Unsafe URL in attribute '{key}'", node=node)
                        out.pop(key, None)
                        changed = True
                        continue

                    rule = url_policy.allow_rules.get((tag, key))
                    if rule is None:
                        if on_report is not None:  # pragma: no cover
                            on_report(f"Unsafe URL in attribute '{key}' (no rule)", node=node)
                        out.pop(key, None)
                        changed = True
                        continue

                    if key == "srcset":
                        sanitized = _sanitize_srcset_value(
                            url_policy=url_policy,
                            rule=rule,
                            tag=tag,
                            attr=key,
                            value=str(raw_value),
                        )
                    else:
                        sanitized = _sanitize_url_value_with_rule(
                            rule=rule,
                            value=str(raw_value),
                            tag=tag,
                            attr=key,
                            handling=_effective_url_handling(url_policy=url_policy, rule=rule),
                            allow_relative=_effective_allow_relative(url_policy=url_policy, rule=rule),
                            proxy=_effective_proxy(url_policy=url_policy, rule=rule),
                            url_filter=url_policy.url_filter,
                            apply_filter=True,
                        )

                    if sanitized is None:
                        if on_report is not None:
                            on_report(f"Unsafe URL in attribute '{key}'", node=node)
                        out.pop(key, None)
                        changed = True
                        continue

                    out[key] = sanitized

                    if raw_value != sanitized:
                        changed = True

                if not changed:
                    return None
                if on_hook is not None:
                    on_hook(node)
                return out

            selector_str = t.selector
            all_nodes = selector_str.strip() == "*"
            _append_compiled(
                _CompiledRewriteAttrsTransform(
                    kind="rewrite_attrs",
                    selector_str=selector_str,
                    selector=None if all_nodes else parse_selector(selector_str),
                    all_nodes=all_nodes,
                    func=_drop_url_attrs,
                )
            )
            continue

        if isinstance(t, AllowStyleAttrs):
            allowed_css_properties = t.allowed_css_properties
            on_hook = t.callback
            on_report = t.report

            def _allow_style_attrs(
                node: Node,
                allowed_css_properties: tuple[str, ...] = allowed_css_properties,
                on_hook: NodeCallback | None = on_hook,
                on_report: ReportCallback | None = on_report,
            ) -> dict[str, str | None] | None:
                attrs = node.attrs
                if not attrs or "style" not in attrs:
                    return None

                raw_value = attrs.get("style")
                if raw_value is None:
                    if on_report is not None:
                        on_report("Unsafe inline style in attribute 'style'", node=node)
                    out = dict(attrs)
                    out.pop("style", None)
                    if on_hook is not None:
                        on_hook(node)
                    return out

                sanitized_style = _sanitize_inline_style(
                    allowed_css_properties=allowed_css_properties,
                    value=str(raw_value),
                    tag=str(node.name).lower(),
                    url_policy=None,
                )
                if sanitized_style is None:
                    if on_report is not None:
                        on_report("Unsafe inline style in attribute 'style'", node=node)
                    out = dict(attrs)
                    out.pop("style", None)
                    if on_hook is not None:
                        on_hook(node)
                    return out

                out = dict(attrs)
                out["style"] = sanitized_style
                if raw_value != sanitized_style and on_hook is not None:
                    on_hook(node)
                return out

            selector_str = t.selector
            all_nodes = selector_str.strip() == "*"
            _append_compiled(
                _CompiledRewriteAttrsTransform(
                    kind="rewrite_attrs",
                    selector_str=selector_str,
                    selector=None if all_nodes else parse_selector(selector_str),
                    all_nodes=all_nodes,
                    func=_allow_style_attrs,
                )
            )
            continue

        if isinstance(t, MergeAttrs):
            if not t.tokens:
                continue
            compiled.append(
                _CompiledMergeAttrTokensTransform(
                    kind="merge_attr_tokens",
                    tag=t.tag,
                    attr=t.attr,
                    tokens=t.tokens,
                    callback=t.callback,
                    report=t.report,
                )
            )
            continue

        if isinstance(t, Sanitize):
            policy = t.policy or DEFAULT_POLICY

            # Wrapper to trigger policy.handle_unsafe on security findings
            cb_sanitize = t.callback
            rep_sanitize = t.report

            def _report_unsafe(
                msg: str,
                *,
                node: Any | None = None,
                policy: SanitizationPolicy = policy,
                rep_sanitize: ReportCallback | None = rep_sanitize,
            ) -> None:
                policy.handle_unsafe(msg, node=node)
                if rep_sanitize:
                    rep_sanitize(msg, node=node)

            # Pre-calc parsing logic
            allowed_tags = frozenset(policy.allowed_tags)
            drop_content_tags = frozenset(policy.drop_content_tags)
            handling = policy.disallowed_tag_handling

            def _sanitize_node_decision(
                node: Node,
                allowed_tags: frozenset[str] = allowed_tags,
                drop_content_tags: frozenset[str] = drop_content_tags,
                handling: str = handling,
                policy: SanitizationPolicy = policy,
                cb: NodeCallback | None = cb_sanitize,
                rep: ReportCallback | None = rep_sanitize,
            ) -> DecideAction:
                name = node.name
                if name.startswith("#") or name == "!doctype":
                    return DecideAction.KEEP
                tag = str(name).lower()

                if tag in allowed_tags:
                    return DecideAction.KEEP

                if tag in drop_content_tags:
                    msg = f"Unsafe tag '{tag}' (dropped content)"
                    policy.handle_unsafe(msg, node=node)
                    if cb:
                        cb(node)
                    if rep:
                        rep(msg, node=node)
                    return DecideAction.DROP

                # Not allowed
                msg_unsafe = f"Unsafe tag '{tag}' (not allowed)"
                policy.handle_unsafe(msg_unsafe, node=node)
                if cb:
                    cb(node)
                if rep:
                    rep(msg_unsafe, node=node)

                if handling == "drop":
                    return DecideAction.DROP
                if handling == "unwrap":
                    return DecideAction.UNWRAP
                if handling == "escape":
                    return DecideAction.ESCAPE
                return DecideAction.DROP  # pragma: no cover

            # Pre-calc attributes logic
            effective_allowed_attrs = policy.allowed_attributes
            if policy.force_link_rel:
                # If force_link_rel is set, we must allow 'rel' on 'a' to preserve existing tokens.
                new_allowed = dict(policy.allowed_attributes)
                a_attrs = set(new_allowed.get("a", []))
                # Also check global allowed attrs
                global_attrs = set(new_allowed.get("*", []))

                if "rel" not in a_attrs and "rel" not in global_attrs:
                    a_attrs.add("rel")
                    new_allowed["a"] = a_attrs
                    effective_allowed_attrs = new_allowed

            sub: list[TransformSpec] = [
                DropComments(enabled=policy.drop_comments, callback=t.callback, report=t.report),
                DropDoctype(enabled=policy.drop_doctype, callback=t.callback, report=t.report),
                DropForeignNamespaces(
                    enabled=policy.drop_foreign_namespaces, callback=t.callback, report=_report_unsafe
                ),
                Decide(selector="*", func=_sanitize_node_decision),
                DropAttrs(
                    selector="*",
                    patterns=("on*", "srcdoc", "*:*"),
                    callback=t.callback,
                    report=_report_unsafe,
                ),
                AllowlistAttrs(
                    selector="*",
                    allowed_attributes=dict(effective_allowed_attrs),
                    callback=t.callback,
                    report=_report_unsafe,
                ),
                DropUrlAttrs(
                    selector="*",
                    url_policy=policy.url_policy,
                    callback=t.callback,
                    report=_report_unsafe,
                ),
                AllowStyleAttrs(
                    selector="*",
                    allowed_css_properties=policy.allowed_css_properties,
                    enabled=bool(policy.allowed_css_properties),
                    callback=t.callback,
                    report=_report_unsafe,
                ),
                MergeAttrs(
                    tag="a",
                    attr="rel",
                    tokens=policy.force_link_rel,
                    enabled=bool(policy.force_link_rel),
                    callback=t.callback,
                    report=t.report,
                ),
            ]

            # Recursive compile to enable optimization (RewriteAttrs fusion)
            for sub_t in compile_transforms(sub):
                _append_compiled(sub_t)

            continue

        raise TypeError(f"Unsupported transform: {type(t).__name__}")  # pragma: no cover

    return compiled


# -----------------
# Application
# -----------------


def _compile_patterns_to_regex(patterns: tuple[str, ...]) -> re.Pattern[str] | None:
    if not patterns:
        return None
    parts: list[str] = []
    for p in patterns:
        regex = re.escape(p)
        regex = regex.replace(r"\*", ".*")
        regex = regex.replace(r"\?", ".")
        parts.append(regex)
    full = "^(?:" + "|".join(parts) + ")$"
    return re.compile(full)


def apply_compiled_transforms(
    root: Node,
    compiled: list[CompiledTransform],
    *,
    errors: list[ParseError] | None = None,
) -> None:
    if not compiled:
        return

    token = _ERROR_SINK.set(errors)
    try:
        matcher = SelectorMatcher()

        def apply_walk_transforms(root_node: Node, walk_transforms: list[CompiledTransform]) -> None:
            if not walk_transforms:
                return

            def _raw_tag_text(node: Node, start_attr: str, end_attr: str) -> str | None:
                start = getattr(node, start_attr, None)
                end = getattr(node, end_attr, None)
                if start is None or end is None:
                    return None
                src = node._source_html
                if src is None:
                    cur: Node | None = node
                    while cur is not None and src is None:
                        cur = cur.parent
                        if cur is None:
                            break
                        src = cur._source_html
                    if src is not None:
                        node._source_html = src
                if src is None:
                    return None
                return src[start:end]

            def _reconstruct_start_tag(node: Node) -> str | None:
                if node.name.startswith("#") or node.name == "!doctype":
                    return None
                name = str(node.name)
                attrs = getattr(node, "attrs", None)
                tag = serialize_start_tag(name, attrs)
                if getattr(node, "_self_closing", False):
                    tag = f"{tag[:-1]}/>"
                return tag

            def _reconstruct_end_tag(node: Node) -> str | None:
                if getattr(node, "_self_closing", False):
                    return None

                # If explicit metadata says no end tag, respect it.
                if getattr(node, "_end_tag_present", None) is False:
                    return None

                # For nodes without metadata (or explicitly present), check void list.
                name = str(node.name)
                if name.startswith("#") or name == "!doctype":
                    return None

                if name.lower() in VOID_ELEMENTS:
                    return None

                return serialize_end_tag(name)

            linkify_skip_tags: frozenset[str] = frozenset().union(
                *(t.skip_tags for t in walk_transforms if isinstance(t, _CompiledLinkifyTransform))
            )
            whitespace_skip_tags: frozenset[str] = frozenset().union(
                *(t.skip_tags for t in walk_transforms if isinstance(t, _CompiledCollapseWhitespaceTransform))
            )

            # To preserve strict left-to-right semantics while still batching
            # compatible transforms into a single walk, we track the earliest
            # transform index that may run on a node.
            #
            # Example:
            #   transforms=[Drop("a"), Linkify()]
            # Linkify introduces <a> elements. Those <a> nodes must not be
            # processed by earlier transforms (like Drop("a")), because Drop has
            # already run conceptually.
            created_start_index: dict[int, int] = {}

            def _mark_start(n: object, start_index: int) -> None:
                key = id(n)
                created_start_index[key] = max(created_start_index.get(key, 0), start_index)

            def _escape_node(
                node: Node,
                *,
                parent: Node,
                idx: int,
                mark_new_start_index: int,
            ) -> None:
                """Escape a node by emitting its tags as text and hoisting its children."""
                raw_start = _raw_tag_text(node, "_start_tag_start", "_start_tag_end")
                if raw_start is None:
                    raw_start = _reconstruct_start_tag(node)
                raw_end = _raw_tag_text(node, "_end_tag_start", "_end_tag_end")
                if raw_end is None:
                    raw_end = _reconstruct_end_tag(node)

                if raw_start:
                    start_node = Text(raw_start)
                    _mark_start(start_node, mark_new_start_index)
                    parent.insert_before(start_node, node)

                moved: list[Node] = []
                if node.name != "#text" and node.children:
                    moved.extend(list(node.children))
                    node.children = []
                if type(node) is Template and node.template_content is not None:
                    tc = node.template_content
                    tc_children = tc.children or []
                    moved.extend(tc_children)
                    tc.children = []

                if moved:
                    for child in moved:
                        _mark_start(child, mark_new_start_index)
                        parent.insert_before(child, node)

                if raw_end:
                    end_node = Text(raw_end)
                    _mark_start(end_node, mark_new_start_index)
                    parent.insert_before(end_node, node)

                parent.remove_child(node)

            def apply_to_children(parent: Node, *, skip_linkify: bool, skip_whitespace: bool) -> None:
                children = parent.children
                if not children:
                    return

                i = 0
                while i < len(children):
                    node = children[i]
                    name = node.name

                    changed = False
                    start_at = created_start_index.get(id(node), 0)
                    for idx in range(start_at, len(walk_transforms)):
                        t = walk_transforms[idx]
                        # Dispatch based on 'kind' string to avoid expensive isinstance/class hierarchy checks
                        # in this hot loop (50k nodes * 10 transforms = 500k type checks otherwise).
                        k: str = t.kind

                        # DropComments
                        if k == "drop_comments":
                            if name == "#comment":
                                if TYPE_CHECKING:
                                    t = cast("_CompiledDropCommentsTransform", t)
                                if t.callback is not None:
                                    t.callback(node)
                                if t.report is not None:
                                    t.report("Dropped comment", node=node)
                                parent.remove_child(node)
                                changed = True
                                break
                            continue

                        # DropDoctype
                        if k == "drop_doctype":
                            if name == "!doctype":
                                if TYPE_CHECKING:
                                    t = cast("_CompiledDropDoctypeTransform", t)
                                if t.callback is not None:
                                    t.callback(node)  # pragma: no cover
                                if t.report is not None:
                                    t.report("Dropped doctype", node=node)  # pragma: no cover
                                parent.remove_child(node)
                                changed = True
                                break
                            continue

                        # MergeAttrs
                        if k == "merge_attr_tokens":
                            if not name.startswith("#") and name != "!doctype":
                                if TYPE_CHECKING:
                                    t = cast("_CompiledMergeAttrTokensTransform", t)
                                if str(name).lower() == t.tag:
                                    attrs = node.attrs
                                    existing_raw = attrs.get(t.attr)
                                    existing: list[str] = []
                                    if isinstance(existing_raw, str) and existing_raw:
                                        for tok in existing_raw.split():
                                            tt = tok.strip().lower()
                                            if tt and tt not in existing:
                                                existing.append(tt)

                                    changed_rel = False
                                    for tok in t.tokens:
                                        if tok not in existing:
                                            existing.append(tok)
                                            changed_rel = True
                                    normalized = " ".join(existing)
                                    if (
                                        changed_rel
                                        or (existing_raw is None and existing)
                                        or (isinstance(existing_raw, str) and existing_raw != normalized)
                                    ):
                                        attrs[t.attr] = normalized
                                        if t.callback is not None:
                                            t.callback(node)
                                        if t.report is not None:
                                            t.report(
                                                f"Merged tokens into attribute '{t.attr}' on <{t.tag}>",
                                                node=node,
                                            )
                            continue

                        # CollapseWhitespace
                        if k == "collapse_whitespace":
                            if name == "#text" and not skip_whitespace:
                                if TYPE_CHECKING:
                                    t = cast("_CompiledCollapseWhitespaceTransform", t)
                                text_data: str = str(getattr(node, "data", "") or "")
                                if text_data:
                                    collapsed = _collapse_html_space_characters(text_data)
                                    if collapsed != text_data:
                                        if t.callback is not None:
                                            t.callback(node)
                                        if t.report is not None:
                                            t.report("Collapsed whitespace in text node", node=node)
                                        node.data = collapsed
                            continue

                        # Linkify
                        if k == "linkify":
                            if name == "#text" and not skip_linkify:
                                if TYPE_CHECKING:
                                    t = cast("_CompiledLinkifyTransform", t)
                                linkify_text: str = str(getattr(node, "data", "") or "")
                                if linkify_text:
                                    matches = find_links_with_config(linkify_text, t.config)
                                    if matches:
                                        if t.callback is not None:
                                            t.callback(node)
                                        if t.report is not None:
                                            t.report(
                                                f"Linkified {len(matches)} link(s) in text node",
                                                node=node,
                                            )
                                        cursor = 0
                                        for m in matches:
                                            if m.start > cursor:
                                                txt = Text(linkify_text[cursor : m.start])
                                                _mark_start(txt, idx + 1)
                                                parent.insert_before(txt, node)

                                            ns = parent.namespace or "html"
                                            a = Element("a", {"href": m.href}, ns)
                                            a.append_child(Text(m.text))
                                            _mark_start(a, idx + 1)
                                            parent.insert_before(a, node)
                                            cursor = m.end

                                        if cursor < len(linkify_text):
                                            tail = Text(linkify_text[cursor:])
                                            _mark_start(tail, idx + 1)
                                            parent.insert_before(tail, node)

                                        parent.remove_child(node)
                                        changed = True
                                        break
                            continue

                        # Decide
                        if k == "decide":
                            if TYPE_CHECKING:
                                t = cast("_CompiledDecideTransform", t)
                            if t.all_nodes:
                                action = t.callback(node)
                            else:
                                if name.startswith("#") or name == "!doctype":
                                    continue
                                sel = t.selector
                                if TYPE_CHECKING:
                                    sel = cast("ParsedSelector", sel)
                                if not matcher.matches(node, sel):
                                    continue
                                action = t.callback(node)

                            if action is DecideAction.KEEP:
                                continue

                            if action is DecideAction.EMPTY:
                                if name != "#text" and node.children:
                                    for child in node.children:
                                        child.parent = None
                                    node.children = []
                                if type(node) is Template and node.template_content is not None:
                                    tc = node.template_content
                                    for child in tc.children or []:
                                        child.parent = None
                                    tc.children = []
                                continue

                            if action is DecideAction.UNWRAP:
                                moved_nodes: list[Node] = []
                                if name != "#text" and node.children:
                                    moved_nodes.extend(list(node.children))
                                    node.children = []
                                if type(node) is Template and node.template_content is not None:
                                    tc = node.template_content
                                    if tc.children:
                                        moved_nodes.extend(list(tc.children))
                                        tc.children = []
                                if moved_nodes:
                                    for child in moved_nodes:
                                        _mark_start(child, idx)
                                        parent.insert_before(child, node)
                                parent.remove_child(node)
                                changed = True
                                break

                            if action is DecideAction.ESCAPE:
                                # Mark created/hoisted nodes to start at the current transform to support recursive rules.
                                _escape_node(node, parent=parent, idx=idx, mark_new_start_index=idx)
                                changed = True
                                break

                            # action == DROP (and any invalid value)
                            parent.remove_child(node)
                            changed = True
                            break

                        # EditAttrs (rewrite_attrs)
                        if k == "rewrite_attrs":
                            if name.startswith("#") or name == "!doctype":
                                continue
                            if TYPE_CHECKING:
                                t = cast("_CompiledRewriteAttrsTransform", t)
                            if not t.all_nodes:
                                sel = t.selector
                                if TYPE_CHECKING:
                                    sel = cast("ParsedSelector", sel)
                                if not matcher.matches(node, sel):
                                    continue
                            new_attrs = t.func(node)
                            if new_attrs is not None:
                                node.attrs = new_attrs
                            continue

                        # Selector transforms
                        if TYPE_CHECKING:
                            t = cast("_CompiledSelectorTransform", t)
                        if name.startswith("#") or name == "!doctype":
                            continue

                        if not matcher.matches(node, t.selector):
                            continue

                        if t.kind == "setattrs":
                            patch = cast("dict[str, str | None]", t.payload)
                            attrs = node.attrs
                            changed_any = False
                            for k, v in patch.items():
                                key = str(k)
                                new_val = None if v is None else str(v)
                                if attrs.get(key) != new_val:
                                    attrs[key] = new_val
                                    changed_any = True
                            if changed_any:
                                if t.callback is not None:
                                    t.callback(node)
                                if t.report is not None:
                                    tag = str(node.name).lower()
                                    t.report(
                                        f"Set attributes on <{tag}> (matched selector '{t.selector_str}')", node=node
                                    )
                            continue

                        if t.kind == "edit":
                            cb = cast("NodeCallback", t.payload)
                            cb(node)
                            continue

                        if t.kind == "empty":
                            had_children = bool(node.children)
                            if node.children:
                                for child in node.children:
                                    child.parent = None
                                node.children = []
                            if type(node) is Template and node.template_content is not None:
                                tc = node.template_content
                                had_children = had_children or bool(tc.children)
                                for child in tc.children or []:
                                    child.parent = None
                                tc.children = []
                            if had_children:
                                if t.callback is not None:
                                    t.callback(node)
                                if t.report is not None:
                                    tag = str(node.name).lower()
                                    t.report(f"Emptied <{tag}> (matched selector '{t.selector_str}')", node=node)
                            continue

                        if t.kind == "drop":
                            if t.callback is not None:
                                t.callback(node)
                            if t.report is not None:
                                tag = str(node.name).lower()
                                t.report(f"Dropped <{tag}> (matched selector '{t.selector_str}')", node=node)
                            parent.remove_child(node)
                            changed = True
                            break

                        if t.kind == "escape":
                            if t.callback is not None:
                                t.callback(node)
                            if t.report is not None:
                                tag = str(node.name).lower()
                                t.report(f"Escaped <{tag}> (matched selector '{t.selector_str}')", node=node)

                            _escape_node(node, parent=parent, idx=idx, mark_new_start_index=idx)
                            changed = True
                            break

                        # t.kind == "unwrap".
                        if t.callback is not None:
                            t.callback(node)
                        if t.report is not None:
                            tag = str(node.name).lower()
                            t.report(f"Unwrapped <{tag}> (matched selector '{t.selector_str}')", node=node)

                        moved_nodes_unwrap: list[Node] = []
                        if node.children:
                            moved_nodes_unwrap.extend(list(node.children))
                            node.children = []

                        if type(node) is Template and node.template_content is not None:
                            tc = node.template_content
                            tc_children = tc.children or []
                            moved_nodes_unwrap.extend(tc_children)
                            tc.children = []

                        if moved_nodes_unwrap:
                            for child in moved_nodes_unwrap:
                                _mark_start(child, idx + 1)
                                parent.insert_before(child, node)
                        parent.remove_child(node)
                        changed = True
                        break

                    if changed:
                        continue

                    if name.startswith("#"):
                        # Document containers (e.g. nested #document-fragment) should
                        # still be traversed to reach their element descendants.
                        if node.children:
                            apply_to_children(node, skip_linkify=skip_linkify, skip_whitespace=skip_whitespace)
                    else:
                        tag = node.name.lower()
                        child_skip = skip_linkify or (tag in linkify_skip_tags)
                        child_skip_ws = skip_whitespace or (tag in whitespace_skip_tags)

                        if node.children:
                            apply_to_children(node, skip_linkify=child_skip, skip_whitespace=child_skip_ws)

                        if type(node) is Template and node.template_content is not None:
                            apply_to_children(
                                node.template_content, skip_linkify=child_skip, skip_whitespace=child_skip_ws
                            )

                    i += 1

            if type(root_node) is not Text:
                apply_to_children(root_node, skip_linkify=False, skip_whitespace=False)

                # Root template nodes need special handling since the main walk
                # only visits children of the provided root.
                if type(root_node) is Template and root_node.template_content is not None:
                    apply_to_children(root_node.template_content, skip_linkify=False, skip_whitespace=False)

        def apply_prune_transforms(root_node: Node, prune_transforms: list[_CompiledPruneEmptyTransform]) -> None:
            def _is_effectively_empty_element(n: Node, *, strip_whitespace: bool) -> bool:
                if n.namespace == "html" and n.name.lower() in VOID_ELEMENTS:
                    return False

                def _has_content(children: list[Node] | None) -> bool:
                    if not children:
                        return False
                    for ch in children:
                        nm = ch.name
                        if nm == "#text":
                            data = getattr(ch, "data", "") or ""
                            if strip_whitespace:
                                if str(data).strip():
                                    return True
                            else:
                                if str(data) != "":
                                    return True
                            continue
                        if nm.startswith("#"):
                            continue
                        return True
                    return False

                if _has_content(n.children):
                    return False

                if type(n) is Template and n.template_content is not None:
                    if _has_content(n.template_content.children):
                        return False

                return True

            stack: list[tuple[Node, bool]] = [(root_node, False)]
            while stack:
                node, visited = stack.pop()
                if not visited:
                    stack.append((node, True))

                    children = node.children or []
                    stack.extend((child, False) for child in reversed(children) if isinstance(child, Node))

                    if type(node) is Template and node.template_content is not None:
                        stack.append((node.template_content, False))
                    continue

                if node.parent is None:
                    continue
                if node.name.startswith("#"):
                    continue

                for pt in prune_transforms:
                    if matcher.matches(node, pt.selector):
                        if _is_effectively_empty_element(node, strip_whitespace=pt.strip_whitespace):
                            if pt.callback is not None:
                                pt.callback(node)
                            if pt.report is not None:
                                tag = str(node.name).lower()
                                pt.report(
                                    f"Pruned empty <{tag}> (matched selector '{pt.selector_str}')",
                                    node=node,
                                )
                            node.parent.remove_child(node)
                            break

        pending_walk: list[CompiledTransform] = []

        i = 0
        while i < len(compiled):
            t = compiled[i]
            if isinstance(
                t,
                (
                    _CompiledSelectorTransform,
                    _CompiledDecideTransform,
                    _CompiledRewriteAttrsTransform,
                    _CompiledLinkifyTransform,
                    _CompiledCollapseWhitespaceTransform,
                    _CompiledDropCommentsTransform,
                    _CompiledDropDoctypeTransform,
                    _CompiledMergeAttrTokensTransform,
                ),
            ):
                pending_walk.append(t)
                i += 1
                continue

            apply_walk_transforms(root, pending_walk)
            pending_walk = []

            if isinstance(t, _CompiledStageBoundary):
                i += 1
                continue

            if isinstance(t, _CompiledStageHookTransform):
                if t.callback is not None:
                    t.callback(root)
                if t.report is not None:
                    t.report(f"Stage {t.index + 1}", node=root)
                i += 1
                continue

            if isinstance(t, _CompiledEditDocumentTransform):
                t.callback(root)
                i += 1
                continue

            if isinstance(t, _CompiledPruneEmptyTransform):
                prune_batch: list[_CompiledPruneEmptyTransform] = [t]
                i += 1
                while i < len(compiled) and isinstance(compiled[i], _CompiledPruneEmptyTransform):
                    prune_batch.append(cast("_CompiledPruneEmptyTransform", compiled[i]))
                    i += 1
                apply_prune_transforms(root, prune_batch)
                continue

            raise TypeError(f"Unsupported compiled transform: {type(t).__name__}")

        apply_walk_transforms(root, pending_walk)
    finally:
        _ERROR_SINK.reset(token)
