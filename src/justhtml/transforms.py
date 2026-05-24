"""Constructor-time DOM transforms.

These transforms are intended as a migration path for Bleach/html5lib-style
post-processing, but are implemented as DOM (tree) operations to match
JustHTML's architecture.

Safety model: transforms shape the in-memory tree; safe-by-default output is
still enforced by `to_html()`/`to_text()`/`to_markdown()` via sanitization.

Performance: selectors are compiled (parsed) once before application.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Literal, cast

from .constants import HTML_FORMATTING_SPACE_CHARACTERS, HTML_SPACE_CHARACTERS
from .sanitize import (
    SanitizationPolicy,
)
from .sanitize import (
    UrlPolicy as _UrlPolicy,
)
from .selector import DEFAULT_SELECTOR_LIMITS, SelectorLimits
from .tokens import ParseError
from .transforms_linkify import CompiledLinkifyTransform
from .transforms_spec import (
    AllowlistAttrs,
    AllowStyleAttrs,
    CollapseWhitespace,
    Decide,
    DecideAction,
    Drop,
    DropAttrs,
    DropComments,
    DropDoctype,
    DropForeignNamespaces,
    DropUrlAttrs,
    Edit,
    EditAttrs,
    EditDocument,
    Empty,
    Escape,
    HardenRawtext,
    Linkify,
    MergeAttrs,
    PruneEmpty,
    Sanitize,
    SetAttrs,
    Unwrap,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Protocol

    from .node import Node
    from .selector import ParsedSelector

    class NodeCallback(Protocol):
        def __call__(self, node: Node) -> None: ...

    class EditAttrsCallback(Protocol):
        def __call__(self, node: Node) -> dict[str, str | None] | None: ...

    class ReportCallback(Protocol):
        def __call__(self, msg: str, *, node: Any | None = None) -> None: ...


UrlPolicy = _UrlPolicy


# -----------------
# Public API
# -----------------


_ERROR_SINK: ContextVar[list[ParseError] | None] = ContextVar("justhtml_transform_error_sink", default=None)
_FOREIGN_ROOT_TAGS: frozenset[str] = frozenset({"math", "svg"})


def _is_effectively_foreign_node(node: Node) -> bool:
    current: Node | None = node
    while current is not None:
        ns = current.namespace
        if ns not in (None, "html"):
            return True

        name = current.name
        if name.startswith("#") or name == "!doctype":
            current = current.parent
            continue

        lowered = name if name.islower() else name.lower()
        if lowered in _FOREIGN_ROOT_TAGS:
            return True

        current = current.parent

    return False


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


def _collapse_html_space_characters(text: str) -> str:
    """Collapse runs of HTML whitespace characters to a single space.

    This mirrors html5lib's whitespace filter behavior: it does not trim.
    """

    # Fast path: no formatting whitespace and no double spaces.
    if not any(ch in text for ch in HTML_FORMATTING_SPACE_CHARACTERS) and "  " not in text:
        return text

    out: list[str] = []
    in_ws = False

    for ch in text:
        if ch in HTML_SPACE_CHARACTERS:
            if in_ws:
                continue
            out.append(" ")
            in_ws = True
            continue

        out.append(ch)
        in_ws = False
    return "".join(out)


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
    | HardenRawtext
    | DropComments
    | DropDoctype
    | DropForeignNamespaces
    | DropAttrs
    | AllowlistAttrs
    | DropUrlAttrs
    | AllowStyleAttrs
    | MergeAttrs
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
class _CompiledEditAttrsTransform:
    kind: Literal["edit_attrs"]
    selector_str: str
    selector: ParsedSelector | None
    all_nodes: bool
    func: EditAttrsCallback


@dataclass(frozen=True, slots=True)
class _CompiledStripInvisibleUnicodeTransform:
    kind: Literal["strip_invisible_unicode"]
    callback: NodeCallback | None
    report: ReportCallback


class _CompiledEditAttrsChain:
    """Optimized chain of attribute transforms using a flat list instead of nested closures.

    This avoids the call-stack depth and allocation overhead of nested `_chained` closures.
    For N attribute transforms, nested closures have O(N) call depth; this has O(1).
    """

    __slots__ = ("all_nodes", "funcs", "kind", "selector", "selector_str")

    kind: Literal["edit_attrs_chain"]
    selector_str: str
    selector: ParsedSelector | None
    all_nodes: bool
    funcs: list[EditAttrsCallback]

    def __init__(
        self,
        selector_str: str,
        selector: ParsedSelector | None,
        all_nodes: bool,
        funcs: list[EditAttrsCallback],
    ) -> None:
        self.kind = "edit_attrs_chain"
        self.selector_str = selector_str
        self.selector = selector
        self.all_nodes = all_nodes
        self.funcs = funcs


class _CompiledDecideChain:
    """Optimized chain of decide transforms using a flat list instead of separate transforms.

    This avoids repeated selector matching and callback overhead for multiple decide transforms
    targeting the same selector. Each callback is called in order, short-circuiting on non-KEEP.
    """

    __slots__ = ("all_nodes", "callbacks", "kind", "selector", "selector_str")

    kind: Literal["decide_chain"]
    selector_str: str
    selector: ParsedSelector | None
    all_nodes: bool
    callbacks: list[Callable[[Node], DecideAction]]

    def __init__(
        self,
        selector_str: str,
        selector: ParsedSelector | None,
        all_nodes: bool,
        callbacks: list[Callable[[Node], DecideAction]],
    ) -> None:
        self.kind = "decide_chain"
        self.selector_str = selector_str
        self.selector = selector
        self.all_nodes = all_nodes
        self.callbacks = callbacks


class _CompiledDecideElementsChain:
    """Optimized decide chain that runs only on element/template nodes.

    Used internally by the sanitizer to avoid calling decide callbacks for
    text/comment/doctype/container nodes.
    """

    __slots__ = ("callbacks", "kind")

    kind: Literal["decide_elements_chain"]
    callbacks: list[Callable[[Node], DecideAction]]

    def __init__(self, callbacks: list[Callable[[Node], DecideAction]]) -> None:
        self.kind = "decide_elements_chain"
        self.callbacks = callbacks


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


@dataclass(frozen=True, slots=True)
class _CompiledHardenRawtextTransform:
    kind: Literal["harden_rawtext"]
    policy: SanitizationPolicy


CompiledTransform = (
    _CompiledSelectorTransform
    | _CompiledDecideTransform
    | _CompiledDecideChain
    | _CompiledDecideElementsChain
    | _CompiledEditAttrsTransform
    | _CompiledEditAttrsChain
    | _CompiledStripInvisibleUnicodeTransform
    | CompiledLinkifyTransform
    | _CompiledCollapseWhitespaceTransform
    | _CompiledPruneEmptyTransform
    | _CompiledEditDocumentTransform
    | _CompiledDropCommentsTransform
    | _CompiledDropDoctypeTransform
    | _CompiledMergeAttrTokensTransform
    | _CompiledStageHookTransform
    | _CompiledStageBoundary
    | _CompiledHardenRawtextTransform
)


def _selector_limits_from_compiled(
    compiled: list[CompiledTransform] | tuple[CompiledTransform, ...],
) -> SelectorLimits:
    for t in reversed(compiled):
        if isinstance(t, _CompiledHardenRawtextTransform):
            return t.policy.selector_limits

    return DEFAULT_SELECTOR_LIMITS


def _iter_flattened_transforms(specs: list[TransformSpec] | tuple[TransformSpec, ...]) -> list[Transform]:
    module = import_module(".transforms_compile", __package__)

    return cast("list[Transform]", module._iter_flattened_transforms(specs))


def _glob_match(pattern: str, text: str) -> bool:
    module = import_module(".transforms_compile", __package__)

    return cast("bool", module._glob_match(pattern, text))


def compile_transforms(
    transforms: list[TransformSpec] | tuple[TransformSpec, ...],
    *,
    _selector_limits: SelectorLimits | None = None,
) -> list[CompiledTransform]:
    module = import_module(".transforms_compile", __package__)

    return cast("list[CompiledTransform]", module.compile_transforms(transforms, _selector_limits=_selector_limits))


def apply_compiled_transforms(
    root: Node,
    compiled: list[CompiledTransform] | tuple[CompiledTransform, ...],
    *,
    errors: list[ParseError] | None = None,
) -> None:
    module = import_module(".transforms_runtime", __package__)

    module.apply_compiled_transforms(root, compiled, errors=errors)
