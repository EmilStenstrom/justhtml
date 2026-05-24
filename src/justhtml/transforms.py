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

from .constants import HTML_FORMATTING_SPACE_CHARACTERS, HTML_SPACE_CHARACTERS, VOID_ELEMENTS
from .node import Element, Node, Template, Text
from .sanitize import (
    SanitizationPolicy,
    _sanitize_rawtext_element_contents,
    _strip_invisible_unicode,
)
from .sanitize import (
    UrlPolicy as _UrlPolicy,
)
from .selector import DEFAULT_SELECTOR_LIMITS, SelectorLimits, SelectorMatcher
from .serialize import serialize_end_tag, serialize_start_tag
from .tokens import ParseError
from .transforms_linkify import CompiledLinkifyTransform, apply_linkify_transform
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
    if not compiled:
        return

    selector_limits = _selector_limits_from_compiled(compiled)
    token = _ERROR_SINK.set(errors)
    try:

        def apply_walk_transforms(
            root_node: Node,
            walk_transforms: list[CompiledTransform] | tuple[CompiledTransform, ...],
        ) -> None:
            if not walk_transforms:
                return

            def _raw_tag_text(node: Node, *, start_tag: bool) -> str | None:
                if not isinstance(node, Element):
                    return None
                if start_tag:
                    start = node._start_tag_start
                    end = node._start_tag_end
                else:
                    start = node._end_tag_start
                    end = node._end_tag_end
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
                tag = serialize_start_tag(name, node.attrs)
                if isinstance(node, Element) and node._self_closing:
                    tag = f"{tag[:-1]}/>"
                return tag

            def _reconstruct_end_tag(node: Node) -> str | None:
                if isinstance(node, Element):
                    if node._self_closing:
                        return None

                    # If explicit metadata says no end tag, respect it.
                    if node._end_tag_present is False:
                        return None

                # For nodes without metadata (or explicitly present), check void list.
                name = str(node.name)
                if name.startswith("#") or name == "!doctype":
                    return None

                if name.lower() in VOID_ELEMENTS:
                    return None

                return serialize_end_tag(name)

            linkify_skip_tags: frozenset[str] = frozenset().union(
                *(t.skip_tags for t in walk_transforms if isinstance(t, CompiledLinkifyTransform))
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
                # Most pipelines (including default sanitization) only ever
                # mark nodes with start_index=0, which is also the implicit
                # default. Avoid storing those entries so we can skip a hot
                # per-node dict lookup in the walker.
                if start_index <= 0:
                    return
                key = id(n)
                prev = created_start_index.get(key)
                if prev is None or start_index > prev:  # pragma: no branch
                    created_start_index[key] = start_index

            def _escape_node(
                node: Node,
                *,
                parent: Node,
                child_index: int,
                mark_new_start_index: int,
            ) -> None:
                """Escape a node by emitting its tags as text and hoisting its children."""
                raw_start = _raw_tag_text(node, start_tag=True)
                if raw_start is None:
                    raw_start = _reconstruct_start_tag(node)
                raw_end = _raw_tag_text(node, start_tag=False)
                if raw_end is None:
                    raw_end = _reconstruct_end_tag(node)

                replacement: list[Any] = []

                if raw_start:
                    start_node = Text(raw_start)
                    _mark_start(start_node, mark_new_start_index)
                    start_node.parent = parent
                    replacement.append(start_node)

                moved: list[Any] = []
                if node.name != "#text" and node.children:
                    moved = node.children
                    node.children = []
                if type(node) is Template and node.template_content is not None:
                    tc = node.template_content
                    if tc.children:
                        if moved:
                            moved.extend(tc.children)
                        else:
                            moved = tc.children
                        tc.children = []

                if moved:
                    for child in moved:
                        _mark_start(child, mark_new_start_index)
                        child.parent = parent
                    replacement.extend(moved)

                if raw_end:
                    end_node = Text(raw_end)
                    _mark_start(end_node, mark_new_start_index)
                    end_node.parent = parent
                    replacement.append(end_node)

                children = parent.children
                if children is None:  # pragma: no cover
                    raise ValueError(f"Node {parent.name} cannot have children")  # pragma: no cover
                if replacement:
                    children[child_index : child_index + 1] = replacement
                else:
                    children.pop(child_index)
                node.parent = None

            def _empty_node(node: Node, name: str) -> None:
                if name != "#text" and node.children:
                    for child in node.children:
                        child.parent = None
                    node.children = []
                if type(node) is Template and node.template_content is not None:
                    tc = node.template_content
                    for child in tc.children or ():
                        child.parent = None
                    tc.children = []

            def _detach_children_for_hoist(node: Node, name: str) -> list[Any]:
                moved: list[Any] = []
                if name != "#text" and node.children:
                    moved = node.children
                    node.children = []
                if type(node) is Template and node.template_content is not None:
                    tc = node.template_content
                    if tc.children:
                        if moved:
                            moved.extend(tc.children)
                        else:
                            moved = tc.children
                        tc.children = []
                return moved

            def _apply_decide_action(
                action: DecideAction,
                node: Node,
                *,
                name: str,
                parent: Node,
                children: list[Node],
                child_index: int,
                transform_index: int,
            ) -> bool:
                if action is DecideAction.EMPTY:
                    _empty_node(node, name)
                    return False

                if action is DecideAction.UNWRAP:
                    moved_nodes = _detach_children_for_hoist(node, name)
                    if moved_nodes:
                        for child in moved_nodes:
                            _mark_start(child, transform_index)
                            child.parent = parent
                        children[child_index : child_index + 1] = moved_nodes
                    else:
                        children.pop(child_index)
                    node.parent = None
                    return True

                if action is DecideAction.ESCAPE:
                    _escape_node(node, parent=parent, child_index=child_index, mark_new_start_index=transform_index)
                    return True

                children.pop(child_index)
                node.parent = None
                return True

            def apply_to_children(parent: Node, *, skip_linkify: bool, skip_whitespace: bool) -> None:
                # Iterative traversal avoids recursion overhead on large trees.
                # Semantics match the recursive implementation: depth-first, left-to-right.
                wt_len = len(walk_transforms)
                stack: list[tuple[Node, int, bool, bool]] = [(parent, 0, skip_linkify, skip_whitespace)]

                while stack:
                    parent, i, skip_linkify, skip_whitespace = stack[-1]
                    children = parent.children
                    if not children or i >= len(children):
                        stack.pop()
                        continue

                    node = children[i]
                    name = node.name
                    is_special = name[0] == "#"
                    is_doctype = name == "!doctype"
                    is_text = name == "#text"
                    is_comment = name == "#comment"

                    changed = False
                    matcher = SelectorMatcher(limits=selector_limits)
                    if created_start_index:
                        start_at = created_start_index.get(id(node), 0)
                    else:
                        start_at = 0
                    for idx in range(start_at, wt_len):
                        t: Any = walk_transforms[idx]
                        # Dispatch based on 'kind' string to avoid expensive isinstance/class hierarchy checks
                        # in this hot loop (50k nodes * 10 transforms = 500k type checks otherwise).
                        k: str = t.kind

                        # Decide (elements-only) chain - flat list iteration (optimized)
                        if k == "decide_elements_chain":
                            if is_special or is_doctype:
                                continue

                            action = DecideAction.KEEP
                            for chain_cb in t.callbacks:
                                action = chain_cb(node)
                                if action is not DecideAction.KEEP:
                                    break

                            if action is DecideAction.KEEP:
                                continue

                            changed = _apply_decide_action(
                                action,
                                node,
                                name=name,
                                parent=parent,
                                children=children,
                                child_index=i,
                                transform_index=idx,
                            )
                            if changed:
                                break
                            continue

                        # EditAttrs chain - flat list iteration (optimized)
                        if k == "edit_attrs_chain":
                            if is_special or is_doctype:
                                continue
                            if not t.all_nodes:
                                sel = t.selector
                                if not matcher.matches(node, sel):
                                    continue
                            # Inline the chain iteration to avoid method-call overhead
                            for chain_func in t.funcs:
                                chain_out = chain_func(node)
                                if chain_out is not None:
                                    node.attrs = chain_out
                            continue

                        # MergeAttrs
                        if k == "merge_attr_tokens":
                            if not is_special and not is_doctype:
                                if str(name).lower() == t.tag:
                                    attrs = node.attrs
                                    matched_keys: list[str] = []
                                    existing: list[str] = []
                                    for attr_key, attr_value in attrs.items():
                                        lower_key = attr_key if attr_key.islower() else attr_key.lower()
                                        if lower_key != t.attr:
                                            continue
                                        matched_keys.append(attr_key)
                                        if isinstance(attr_value, str) and attr_value:
                                            for tok in attr_value.split():
                                                tt = tok.strip().lower()
                                                if tt and tt not in existing:
                                                    existing.append(tt)

                                    changed_rel = False
                                    for tok in t.tokens:
                                        if tok not in existing:
                                            existing.append(tok)
                                            changed_rel = True
                                    normalized = " ".join(existing)
                                    existing_raw = attrs.get(t.attr)
                                    has_mixed_case_duplicates = bool(
                                        matched_keys and (len(matched_keys) != 1 or matched_keys[0] != t.attr)
                                    )
                                    if changed_rel or has_mixed_case_duplicates or existing_raw != normalized:
                                        for attr_key in matched_keys:
                                            if attr_key != t.attr:
                                                attrs.pop(attr_key, None)
                                        attrs[t.attr] = normalized
                                        if t.callback is not None:
                                            t.callback(node)
                                        if t.report is not None:
                                            t.report(
                                                f"Merged tokens into attribute '{t.attr}' on <{t.tag}>",
                                                node=node,
                                            )
                            continue

                        # DropComments
                        if k == "drop_comments":
                            if is_comment:
                                if t.callback is not None:
                                    t.callback(node)
                                if t.report is not None:
                                    t.report("Dropped comment", node=node)
                                children.pop(i)
                                node.parent = None
                                changed = True
                                break
                            continue

                        # DropDoctype
                        if k == "drop_doctype":
                            if is_doctype:
                                if t.callback is not None:
                                    t.callback(node)  # pragma: no cover
                                if t.report is not None:
                                    t.report("Dropped doctype", node=node)  # pragma: no cover
                                children.pop(i)
                                node.parent = None
                                changed = True
                                break
                            continue

                        # CollapseWhitespace
                        if k == "collapse_whitespace":
                            if is_text and not skip_whitespace:
                                text_data = str(node.data or "")
                                if text_data:
                                    collapsed = _collapse_html_space_characters(text_data)
                                    if collapsed != text_data:
                                        if t.callback is not None:
                                            t.callback(node)
                                        if t.report is not None:
                                            t.report("Collapsed whitespace in text node", node=node)
                                        node.data = collapsed
                            continue

                        # Strip invisible Unicode variation selectors
                        if k == "strip_invisible_unicode":
                            if is_text:
                                text_data = str(node.data or "")
                                if text_data:
                                    stripped_text = _strip_invisible_unicode(text_data)
                                    if stripped_text != text_data:
                                        if t.callback is not None:
                                            t.callback(node)
                                        t.report("Stripped invisible Unicode from text node", node=node)
                                        node.data = stripped_text
                                continue

                            if is_special or is_doctype:
                                continue

                            attrs = node.attrs
                            if not attrs:
                                continue

                            changed_keys: list[str] | None = None
                            for attr_name, raw_value in attrs.items():
                                if raw_value is None:
                                    continue
                                stripped_value = _strip_invisible_unicode(str(raw_value))
                                if stripped_value == raw_value:
                                    continue
                                attrs[attr_name] = stripped_value
                                if changed_keys is None:
                                    changed_keys = []
                                changed_keys.append(attr_name)

                            if changed_keys is not None:
                                if t.callback is not None:
                                    t.callback(node)
                                attrs_list = ", ".join(changed_keys)
                                t.report(
                                    f"Stripped invisible Unicode from attribute(s): {attrs_list}",
                                    node=node,
                                )
                            continue

                        # Linkify
                        if k == "linkify":
                            if is_text and not skip_linkify:
                                changed = apply_linkify_transform(
                                    parent=parent,
                                    node=node,
                                    children=children,
                                    child_index=i,
                                    transform_index=idx,
                                    transform=t,
                                    mark_start=_mark_start,
                                )
                                if changed:
                                    break
                            continue

                        # Decide
                        if k == "decide":
                            if t.all_nodes:
                                action = t.callback(node)
                            else:
                                if is_special or is_doctype:
                                    continue
                                sel = t.selector
                                if not matcher.matches(node, sel):
                                    continue
                                action = t.callback(node)

                            if action is DecideAction.KEEP:
                                continue

                            changed = _apply_decide_action(
                                action,
                                node,
                                name=name,
                                parent=parent,
                                children=children,
                                child_index=i,
                                transform_index=idx,
                            )
                            if changed:
                                break
                            continue

                        # Decide chain - flat list iteration (optimized)
                        if k == "decide_chain":
                            if t.all_nodes:
                                # Iterate through callbacks until one returns non-KEEP
                                action = DecideAction.KEEP
                                for chain_cb in t.callbacks:
                                    action = chain_cb(node)
                                    if action is not DecideAction.KEEP:
                                        break
                            else:
                                if is_special or is_doctype:
                                    continue
                                sel = t.selector
                                if not matcher.matches(node, sel):
                                    continue
                                action = DecideAction.KEEP
                                for chain_cb in t.callbacks:
                                    action = chain_cb(node)
                                    if action is not DecideAction.KEEP:
                                        break

                            if action is DecideAction.KEEP:
                                continue

                            changed = _apply_decide_action(
                                action,
                                node,
                                name=name,
                                parent=parent,
                                children=children,
                                child_index=i,
                                transform_index=idx,
                            )
                            if changed:
                                break
                            continue

                        # EditAttrs - single function
                        if k == "edit_attrs":
                            if is_special or is_doctype:
                                continue
                            if not t.all_nodes:
                                sel = t.selector
                                if not matcher.matches(node, sel):
                                    continue
                            new_attrs = t.func(node)
                            if new_attrs is not None:
                                node.attrs = new_attrs
                            continue

                        # Selector transforms
                        if is_special or is_doctype:
                            continue

                        if not matcher.matches(node, t.selector):
                            continue

                        if t.kind == "setattrs":
                            patch = t.payload
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
                            cb = t.payload
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
                                for child in tc.children or ():
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
                            children.pop(i)
                            node.parent = None
                            changed = True
                            break

                        if t.kind == "escape":
                            if t.callback is not None:
                                t.callback(node)
                            if t.report is not None:
                                tag = str(node.name).lower()
                                t.report(f"Escaped <{tag}> (matched selector '{t.selector_str}')", node=node)

                            _escape_node(node, parent=parent, child_index=i, mark_new_start_index=idx)
                            changed = True
                            break

                        # t.kind == "unwrap".
                        if t.callback is not None:
                            t.callback(node)
                        if t.report is not None:
                            tag = str(node.name).lower()
                            t.report(f"Unwrapped <{tag}> (matched selector '{t.selector_str}')", node=node)

                        moved_nodes_unwrap: list[Any] = []
                        if node.children:
                            moved_nodes_unwrap = node.children
                            node.children = []

                        if type(node) is Template and node.template_content is not None:
                            tc = node.template_content
                            if tc.children:
                                if moved_nodes_unwrap:
                                    moved_nodes_unwrap.extend(tc.children)
                                else:
                                    moved_nodes_unwrap = tc.children
                                tc.children = []

                        if moved_nodes_unwrap:
                            for child in moved_nodes_unwrap:
                                _mark_start(child, idx + 1)
                                child.parent = parent
                            children[i : i + 1] = moved_nodes_unwrap
                        else:
                            children.pop(i)
                        node.parent = None
                        changed = True
                        break

                    if changed:
                        continue

                    # No mutation: advance sibling index before descending.
                    stack[-1] = (parent, i + 1, skip_linkify, skip_whitespace)

                    if is_special:
                        # Document containers (e.g. nested #document-fragment) should
                        # still be traversed to reach their element descendants.
                        #
                        # Text nodes implement `children` as a property that
                        # allocates an empty list; avoid touching it.
                        if not is_text and not is_comment and node.children:
                            stack.append((node, 0, skip_linkify, skip_whitespace))
                        continue

                    if linkify_skip_tags or whitespace_skip_tags:
                        tag = node.name.lower()
                        child_skip = skip_linkify or (tag in linkify_skip_tags)
                        child_skip_ws = skip_whitespace or (tag in whitespace_skip_tags)
                    else:
                        # Common case (including default sanitization): no Linkify/CollapseWhitespace
                        # transforms are present, so skip-set checks are unnecessary.
                        child_skip = skip_linkify
                        child_skip_ws = skip_whitespace

                    # Traverse node children first, then template_content (matches recursive order).
                    if type(node) is Template and node.template_content is not None and node.template_content.children:
                        stack.append((node.template_content, 0, child_skip, child_skip_ws))
                    if node.children:
                        stack.append((node, 0, child_skip, child_skip_ws))

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
                            data = ch.data or ""
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

                    children = node.children or ()
                    stack.extend((child, False) for child in reversed(children) if isinstance(child, Node))

                    if type(node) is Template and node.template_content is not None:
                        stack.append((node.template_content, False))
                    continue

                if node.parent is None:
                    continue
                if node.name.startswith("#"):
                    continue

                matcher = SelectorMatcher(limits=selector_limits)
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
                    _CompiledDecideChain,
                    _CompiledDecideElementsChain,
                    _CompiledEditAttrsTransform,
                    _CompiledEditAttrsChain,
                    _CompiledStripInvisibleUnicodeTransform,
                    CompiledLinkifyTransform,
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

            if isinstance(t, _CompiledHardenRawtextTransform):
                _sanitize_rawtext_element_contents(root, policy=t.policy, errors=errors)
                i += 1
                continue

            raise TypeError(f"Unsupported compiled transform: {type(t).__name__}")

        apply_walk_transforms(root, pending_walk)
    finally:
        _ERROR_SINK.reset(token)
