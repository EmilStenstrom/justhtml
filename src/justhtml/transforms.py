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
from typing import TYPE_CHECKING, Literal, cast

from .constants import HTML_FORMATTING_SPACE_CHARACTERS, HTML_SPACE_CHARACTERS, VOID_ELEMENTS
from .node import Element, Node, Template, Text
from .sanitize import (
    DEFAULT_POLICY,
    SanitizationPolicy,
    UrlPolicy,
    _sanitize_inline_style,
    _sanitize_rawtext_element_contents,
    _strip_invisible_unicode,
)
from .sanitize_url import _sanitize_url_sink_value, _url_sink_kind_for_attr
from .selector import DEFAULT_SELECTOR_LIMITS, SelectorLimits, SelectorMatcher, parse_selector
from .serialize import serialize_end_tag, serialize_start_tag
from .tokens import ParseError
from .transforms_linkify import CompiledLinkifyTransform, apply_linkify_transform, compile_linkify_transform
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
class _CompiledSanitizeRawtextPolicy:
    kind: Literal["sanitize_rawtext_policy"]
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
    | _CompiledSanitizeRawtextPolicy
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


def _selector_limits_from_flattened(flattened: list[Transform]) -> SelectorLimits:
    for t in reversed(flattened):
        if not t.enabled:
            continue
        if isinstance(t, Sanitize):
            return (t.policy or DEFAULT_POLICY).selector_limits
    return DEFAULT_SELECTOR_LIMITS


def _selector_limits_from_compiled(
    compiled: list[CompiledTransform] | tuple[CompiledTransform, ...],
) -> SelectorLimits:
    for t in reversed(compiled):
        if isinstance(t, _CompiledSanitizeRawtextPolicy):
            return t.policy.selector_limits
    return DEFAULT_SELECTOR_LIMITS


def _compile_sanitize_transform(t: Sanitize) -> list[CompiledTransform]:
    policy = t.policy or DEFAULT_POLICY
    compiled: list[CompiledTransform] = []

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
        # This callback is used with an elements-only dispatcher; tag names
        # produced by the tokenizer are already ASCII-lowercased. Programmatic
        # DOM input may not be normalized, so match policy semantics
        # case-insensitively here as well.
        raw_tag = str(node.name)
        tag = raw_tag if raw_tag.islower() else raw_tag.lower()

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

    effective_allowed_attrs = policy.allowed_attributes
    if policy.force_link_rel:
        new_allowed = dict(policy.allowed_attributes)
        a_attrs = set(new_allowed.get("a", []))
        global_attrs = set(new_allowed.get("*", []))

        if "rel" not in a_attrs and "rel" not in global_attrs:
            a_attrs.add("rel")
            new_allowed["a"] = a_attrs
            effective_allowed_attrs = new_allowed

    cb_foreign = t.callback
    rep_foreign = _report_unsafe

    def _drop_foreign_namespace(
        node: Node,
        cb: NodeCallback | None = cb_foreign,
        rep: ReportCallback = rep_foreign,
    ) -> DecideAction:
        if _is_effectively_foreign_node(node):
            if cb is not None:
                cb(node)
            rep(f"Unsafe tag '{node.name}' (foreign namespace)", node=node)
            return DecideAction.DROP
        return DecideAction.KEEP

    compiled.append(_CompiledDecideElementsChain(callbacks=[_drop_foreign_namespace, _sanitize_node_decision]))

    if policy.strip_invisible_unicode:
        compiled.append(
            _CompiledStripInvisibleUnicodeTransform(
                kind="strip_invisible_unicode",
                callback=t.callback,
                report=_report_unsafe,
            )
        )

    sub_attrs: list[TransformSpec] = [
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
            url_policy=policy.url_policy,
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
    compiled.extend(compile_transforms(sub_attrs, _selector_limits=policy.selector_limits))

    if policy.drop_comments:
        compiled.append(
            _CompiledDropCommentsTransform(
                kind="drop_comments",
                callback=t.callback,
                report=_report_unsafe,
            )
        )
    if policy.drop_doctype:
        compiled.append(
            _CompiledDropDoctypeTransform(
                kind="drop_doctype",
                callback=t.callback,
                report=_report_unsafe,
            )
        )

    compiled.append(
        _CompiledSanitizeRawtextPolicy(
            kind="sanitize_rawtext_policy",
            policy=policy,
        )
    )
    return compiled


def _compile_drop_transform(t: Drop, parse: Callable[[str], ParsedSelector]) -> CompiledTransform:
    selector_str = t.selector

    raw_parts = selector_str.split(",")
    tag_list: list[str] = []
    for part in raw_parts:
        p = part.strip().lower()
        if not p:
            tag_list = []
            break
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

        return _CompiledDecideTransform(
            kind="decide",
            selector_str="*",
            selector=None,
            all_nodes=True,
            callback=_drop_if_tag,
        )

    return _CompiledSelectorTransform(
        kind="drop",
        selector_str=selector_str,
        selector=parse(selector_str),
        payload=None,
        callback=t.callback,
        report=t.report,
    )


def _compile_edit_transform(t: Edit, parse: Callable[[str], ParsedSelector]) -> _CompiledSelectorTransform:
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

    return _CompiledSelectorTransform(
        kind="edit",
        selector_str=selector_str,
        selector=parse(selector_str),
        payload=_wrapped,
        callback=None,
        report=None,
    )


def _compile_edit_document_transform(t: EditDocument) -> _CompiledEditDocumentTransform:
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

    return _CompiledEditDocumentTransform(kind="edit_document", callback=_wrapped_root)


def _compile_decide_transform(t: Decide, parse: Callable[[str], ParsedSelector]) -> _CompiledDecideTransform:
    selector_str = t.selector
    all_nodes = selector_str.strip() == "*"
    decide_func = t.func
    on_hook = t.callback
    on_report = t.report

    if on_hook is None and on_report is None:
        effective_callback = decide_func
    else:

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

        effective_callback = _wrapped_decide

    return _CompiledDecideTransform(
        kind="decide",
        selector_str=selector_str,
        selector=None if all_nodes else parse(selector_str),
        all_nodes=all_nodes,
        callback=effective_callback,
    )


def _compile_edit_attrs_transform(t: EditAttrs, parse: Callable[[str], ParsedSelector]) -> _CompiledEditAttrsTransform:
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

    return _CompiledEditAttrsTransform(
        kind="edit_attrs",
        selector_str=selector_str,
        selector=None if all_nodes else parse(selector_str),
        all_nodes=all_nodes,
        func=_wrapped_attrs,
    )


def _compile_drop_foreign_namespaces_transform(t: DropForeignNamespaces) -> _CompiledDecideTransform:
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
        if _is_effectively_foreign_node(node):
            if on_hook is not None:
                on_hook(node)
            if on_report is not None:
                tag = str(name).lower()
                on_report(f"Unsafe tag '{tag}' (foreign namespace)", node=node)
            return Decide.DROP
        return Decide.KEEP

    return _CompiledDecideTransform(
        kind="decide",
        selector_str="*",
        selector=None,
        all_nodes=True,
        callback=_drop_foreign,
    )


def _compile_drop_attrs_transform(t: DropAttrs, parse: Callable[[str], ParsedSelector]) -> _CompiledEditAttrsTransform:
    patterns = t.patterns
    on_hook = t.callback
    on_report = t.report
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

        if patterns == ("*:*", "on*", "srcdoc"):
            for key in attrs:
                lower_key = key if key.islower() else key.lower()
                if lower_key.startswith("on") or lower_key == "srcdoc" or ":" in lower_key:
                    break
            else:
                return None

            out = dict(attrs)
            for key in attrs:
                lower_key = key if key.islower() else key.lower()
                if not (lower_key.startswith("on") or lower_key == "srcdoc" or ":" in lower_key):
                    continue
                if on_report is not None:  # pragma: no cover
                    if lower_key == "srcdoc":
                        found_pat = "srcdoc"
                    elif ":" in lower_key:
                        found_pat = "*:*"
                    else:
                        found_pat = "on*"
                    on_report(
                        f"Unsafe attribute '{lower_key}' (matched forbidden pattern '{found_pat}')",
                        node=node,
                    )
                out.pop(key, None)
            if on_hook is not None:
                on_hook(node)  # pragma: no cover
            return out

        if compiled_regex is None:  # pragma: no cover
            return None
        for raw_key in attrs:
            if not raw_key or not str(raw_key).strip():
                continue
            key = raw_key
            if not key.islower():
                key = key.lower()
            if compiled_regex.match(key):
                break
        else:
            return None

        out2: dict[str, str | None] = {}
        for raw_key, value in attrs.items():
            if not raw_key or not str(raw_key).strip():
                continue
            key = raw_key
            if not key.islower():
                key = key.lower()

            if compiled_regex.match(key):
                if on_report is not None:
                    found_pat = "?"
                    for pat in patterns:
                        if _glob_match(pat, key):  # pragma: no cover
                            found_pat = pat
                            break
                    on_report(
                        f"Unsafe attribute '{key}' (matched forbidden pattern '{found_pat}')",
                        node=node,
                    )
                continue

            out2[key] = value

        if on_hook is not None:
            on_hook(node)  # pragma: no cover
        return out2

    selector_str = t.selector
    all_nodes = selector_str.strip() == "*"
    return _CompiledEditAttrsTransform(
        kind="edit_attrs",
        selector_str=selector_str,
        selector=None if all_nodes else parse(selector_str),
        all_nodes=all_nodes,
        func=_drop_attrs,
    )


def _compile_allowlist_attrs_transform(
    t: AllowlistAttrs, parse: Callable[[str], ParsedSelector]
) -> _CompiledEditAttrsTransform:
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
        tag = node.name
        if type(tag) is not str:  # pragma: no cover
            tag = str(tag)

        allowed = allowed_by_tag.get(tag)
        if allowed is None:
            if not tag.islower():  # pragma: no cover
                allowed = allowed_by_tag.get(tag.lower())
            if allowed is None:
                allowed = allowed_global

        for key in attrs:
            if type(key) is not str:
                break
            if not key or key not in allowed:
                break
        else:
            return None

        changed = False
        out: dict[str, str | None] = {}
        for raw_key in attrs:
            value = attrs[raw_key]
            raw_key_str = raw_key if type(raw_key) is str else str(raw_key)
            if not raw_key_str.strip():
                changed = True
                continue
            key = raw_key_str
            if key in allowed:
                out[key] = value
                continue

            if not key.islower():
                lowered = key.lower()
                if lowered in allowed:
                    out[lowered] = value
                    changed = True
                    continue
                key = lowered

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
    return _CompiledEditAttrsTransform(
        kind="edit_attrs",
        selector_str=selector_str,
        selector=None if all_nodes else parse(selector_str),
        all_nodes=all_nodes,
        func=_allowlist_attrs,
    )


def _compile_drop_url_attrs_transform(
    t: DropUrlAttrs, parse: Callable[[str], ParsedSelector]
) -> _CompiledEditAttrsTransform:
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

        tag = str(node.name)
        if not tag.islower():
            tag = tag.lower()
        to_drop: list[str] | None = None
        to_set: dict[str, str] | None = None

        for key in attrs:
            lower_key = key if key.islower() else key.lower()
            raw_value = attrs[key]

            sink_kind = _url_sink_kind_for_attr(tag=tag, attr=lower_key, attrs=attrs)
            if sink_kind is None:
                continue

            if raw_value is None:
                if on_report is not None:  # pragma: no cover
                    on_report(f"Unsafe URL in attribute '{lower_key}'", node=node)
                if to_drop is None:
                    to_drop = []
                to_drop.append(key)
                continue

            rule = url_policy.allow_rules.get((tag, lower_key))
            if rule is None:
                if on_report is not None:  # pragma: no cover
                    on_report(f"Unsafe URL in attribute '{lower_key}' (no rule)", node=node)
                if to_drop is None:
                    to_drop = []
                to_drop.append(key)
                continue

            sanitized = _sanitize_url_sink_value(
                url_policy=url_policy,
                rule=rule,
                tag=tag,
                attr=lower_key,
                kind=sink_kind,
                value=str(raw_value),
            )

            if sanitized is None:
                if on_report is not None:
                    on_report(f"Unsafe URL in attribute '{lower_key}'", node=node)
                if to_drop is None:
                    to_drop = []
                to_drop.append(key)
                continue

            if key != lower_key:
                if to_drop is None:
                    to_drop = []
                to_drop.append(key)
                if to_set is None:
                    to_set = {}
                to_set[lower_key] = sanitized
                continue

            if raw_value != sanitized:
                if to_set is None:
                    to_set = {}
                to_set[key] = sanitized

        if to_drop is None and to_set is None:
            return None

        out = dict(attrs)
        if to_drop is not None:
            for key in to_drop:
                out.pop(key, None)
        if to_set is not None:
            out.update(to_set)

        if on_hook is not None:
            on_hook(node)
        return out

    selector_str = t.selector
    all_nodes = selector_str.strip() == "*"
    return _CompiledEditAttrsTransform(
        kind="edit_attrs",
        selector_str=selector_str,
        selector=None if all_nodes else parse(selector_str),
        all_nodes=all_nodes,
        func=_drop_url_attrs,
    )


def _compile_allow_style_attrs_transform(
    t: AllowStyleAttrs, parse: Callable[[str], ParsedSelector]
) -> _CompiledEditAttrsTransform:
    allowed_css_properties = t.allowed_css_properties
    style_url_policy = t.url_policy
    on_hook = t.callback
    on_report = t.report

    def _allow_style_attrs(
        node: Node,
        allowed_css_properties: tuple[str, ...] = allowed_css_properties,
        url_policy: UrlPolicy | None = style_url_policy,
        on_hook: NodeCallback | None = on_hook,
        on_report: ReportCallback | None = on_report,
    ) -> dict[str, str | None] | None:
        attrs = node.attrs
        if not attrs:
            return None

        style_key: str | None = None
        for key in attrs:
            lower_key = key if key.islower() else key.lower()
            if lower_key == "style":
                style_key = key
                break

        if style_key is None:
            return None

        raw_value = attrs.get(style_key)
        if raw_value is None:
            if on_report is not None:
                on_report("Unsafe inline style in attribute 'style'", node=node)
            out = dict(attrs)
            out.pop(style_key, None)
            if on_hook is not None:
                on_hook(node)
            return out

        sanitized_style = _sanitize_inline_style(
            allowed_css_properties=allowed_css_properties,
            value=str(raw_value),
            tag=str(node.name).lower(),
            url_policy=url_policy,
        )
        if sanitized_style is None:
            if on_report is not None:
                on_report("Unsafe inline style in attribute 'style'", node=node)
            out = dict(attrs)
            out.pop(style_key, None)
            if on_hook is not None:
                on_hook(node)
            return out

        if style_key == "style" and raw_value == sanitized_style:
            return None

        out = dict(attrs)
        if style_key != "style":
            out.pop(style_key, None)
        out["style"] = sanitized_style
        if on_hook is not None:
            on_hook(node)
        return out

    selector_str = t.selector
    all_nodes = selector_str.strip() == "*"
    return _CompiledEditAttrsTransform(
        kind="edit_attrs",
        selector_str=selector_str,
        selector=None if all_nodes else parse(selector_str),
        all_nodes=all_nodes,
        func=_allow_style_attrs,
    )


def _compile_merge_attrs_transform(t: MergeAttrs) -> _CompiledMergeAttrTokensTransform | None:
    if not t.tokens:
        return None
    return _CompiledMergeAttrTokensTransform(
        kind="merge_attr_tokens",
        tag=t.tag,
        attr=t.attr,
        tokens=t.tokens,
        callback=t.callback,
        report=t.report,
    )


def _compile_selector_transform(
    *,
    kind: Literal["setattrs", "unwrap", "escape", "empty"],
    selector: str,
    parse: Callable[[str], ParsedSelector],
    payload: Any,
    callback: NodeCallback | None,
    report: ReportCallback | None,
) -> _CompiledSelectorTransform:
    return _CompiledSelectorTransform(
        kind=kind,
        selector_str=selector,
        selector=parse(selector),
        payload=payload,
        callback=callback,
        report=report,
    )


def _compile_collapse_whitespace_transform(t: CollapseWhitespace) -> _CompiledCollapseWhitespaceTransform:
    return _CompiledCollapseWhitespaceTransform(
        kind="collapse_whitespace",
        skip_tags=t.skip_tags,
        callback=t.callback,
        report=t.report,
    )


def _compile_prune_empty_transform(
    t: PruneEmpty, parse: Callable[[str], ParsedSelector]
) -> _CompiledPruneEmptyTransform:
    return _CompiledPruneEmptyTransform(
        kind="prune_empty",
        selector_str=t.selector,
        selector=parse(t.selector),
        strip_whitespace=t.strip_whitespace,
        callback=t.callback,
        report=t.report,
    )


def _compile_drop_comments_transform(t: DropComments) -> _CompiledDropCommentsTransform:
    return _CompiledDropCommentsTransform(
        kind="drop_comments",
        callback=t.callback,
        report=t.report,
    )


def _compile_drop_doctype_transform(t: DropDoctype) -> _CompiledDropDoctypeTransform:
    return _CompiledDropDoctypeTransform(
        kind="drop_doctype",
        callback=t.callback,
        report=t.report,
    )


def _append_compiled_transform(compiled: list[CompiledTransform], item: CompiledTransform) -> None:
    # Optimization: fuse adjacent EditAttrs transforms that target the same
    # selector into a flat chain. This avoids nested closure overhead.
    if compiled and isinstance(item, _CompiledEditAttrsTransform):
        prev = compiled[-1]
        if isinstance(prev, _CompiledEditAttrsChain):
            if prev.selector_str == item.selector_str and prev.all_nodes == item.all_nodes:
                prev.funcs.append(item.func)
                return
        if isinstance(prev, _CompiledEditAttrsTransform):
            if prev.selector_str == item.selector_str and prev.all_nodes == item.all_nodes:
                compiled[-1] = _CompiledEditAttrsChain(
                    selector_str=prev.selector_str,
                    selector=prev.selector,
                    all_nodes=prev.all_nodes,
                    funcs=[prev.func, item.func],
                )
                return

    # Optimization: fuse adjacent Decide transforms that target the same
    # selector into a flat chain. This avoids repeated dispatch overhead.
    if compiled and isinstance(item, _CompiledDecideTransform):
        prev = compiled[-1]
        if isinstance(prev, _CompiledDecideChain):
            if prev.selector_str == item.selector_str and prev.all_nodes == item.all_nodes:
                prev.callbacks.append(item.callback)
                return
        if isinstance(prev, _CompiledDecideTransform):
            if prev.selector_str == item.selector_str and prev.all_nodes == item.all_nodes:
                compiled[-1] = _CompiledDecideChain(
                    selector_str=prev.selector_str,
                    selector=prev.selector,
                    all_nodes=prev.all_nodes,
                    callbacks=[prev.callback, item.callback],
                )
                return

    compiled.append(item)


def _lower_setattrs_transform(
    t: Transform, parse: Callable[[str], ParsedSelector], compiled: list[CompiledTransform]
) -> None:
    item = cast("SetAttrs", t)
    compiled.append(
        _compile_selector_transform(
            kind="setattrs",
            selector=item.selector,
            parse=parse,
            payload=item.attrs,
            callback=item.callback,
            report=item.report,
        )
    )


def _lower_drop_transform(
    t: Transform, parse: Callable[[str], ParsedSelector], compiled: list[CompiledTransform]
) -> None:
    compiled.append(_compile_drop_transform(cast("Drop", t), parse))


def _lower_unwrap_transform(
    t: Transform, parse: Callable[[str], ParsedSelector], compiled: list[CompiledTransform]
) -> None:
    item = cast("Unwrap", t)
    compiled.append(
        _compile_selector_transform(
            kind="unwrap",
            selector=item.selector,
            parse=parse,
            payload=None,
            callback=item.callback,
            report=item.report,
        )
    )


def _lower_escape_transform(
    t: Transform, parse: Callable[[str], ParsedSelector], compiled: list[CompiledTransform]
) -> None:
    item = cast("Escape", t)
    compiled.append(
        _compile_selector_transform(
            kind="escape",
            selector=item.selector,
            parse=parse,
            payload=None,
            callback=item.callback,
            report=item.report,
        )
    )


def _lower_empty_transform(
    t: Transform, parse: Callable[[str], ParsedSelector], compiled: list[CompiledTransform]
) -> None:
    item = cast("Empty", t)
    compiled.append(
        _compile_selector_transform(
            kind="empty",
            selector=item.selector,
            parse=parse,
            payload=None,
            callback=item.callback,
            report=item.report,
        )
    )


def _lower_edit_transform(
    t: Transform, parse: Callable[[str], ParsedSelector], compiled: list[CompiledTransform]
) -> None:
    compiled.append(_compile_edit_transform(cast("Edit", t), parse))


def _lower_edit_document_transform(
    t: Transform, _parse: Callable[[str], ParsedSelector], compiled: list[CompiledTransform]
) -> None:
    compiled.append(_compile_edit_document_transform(cast("EditDocument", t)))


def _lower_decide_transform(
    t: Transform, parse: Callable[[str], ParsedSelector], compiled: list[CompiledTransform]
) -> None:
    _append_compiled_transform(compiled, _compile_decide_transform(cast("Decide", t), parse))


def _lower_edit_attrs_transform(
    t: Transform, parse: Callable[[str], ParsedSelector], compiled: list[CompiledTransform]
) -> None:
    _append_compiled_transform(compiled, _compile_edit_attrs_transform(cast("EditAttrs", t), parse))


def _lower_linkify_transform(
    t: Transform, _parse: Callable[[str], ParsedSelector], compiled: list[CompiledTransform]
) -> None:
    compiled.append(compile_linkify_transform(cast("Linkify", t)))


def _lower_collapse_whitespace_transform(
    t: Transform, _parse: Callable[[str], ParsedSelector], compiled: list[CompiledTransform]
) -> None:
    compiled.append(_compile_collapse_whitespace_transform(cast("CollapseWhitespace", t)))


def _lower_prune_empty_transform(
    t: Transform, parse: Callable[[str], ParsedSelector], compiled: list[CompiledTransform]
) -> None:
    compiled.append(_compile_prune_empty_transform(cast("PruneEmpty", t), parse))


def _lower_drop_comments_transform(
    t: Transform, _parse: Callable[[str], ParsedSelector], compiled: list[CompiledTransform]
) -> None:
    compiled.append(_compile_drop_comments_transform(cast("DropComments", t)))


def _lower_drop_doctype_transform(
    t: Transform, _parse: Callable[[str], ParsedSelector], compiled: list[CompiledTransform]
) -> None:
    compiled.append(_compile_drop_doctype_transform(cast("DropDoctype", t)))


def _lower_drop_foreign_namespaces_transform(
    t: Transform, _parse: Callable[[str], ParsedSelector], compiled: list[CompiledTransform]
) -> None:
    compiled.append(_compile_drop_foreign_namespaces_transform(cast("DropForeignNamespaces", t)))


def _lower_drop_attrs_transform(
    t: Transform, parse: Callable[[str], ParsedSelector], compiled: list[CompiledTransform]
) -> None:
    _append_compiled_transform(compiled, _compile_drop_attrs_transform(cast("DropAttrs", t), parse))


def _lower_allowlist_attrs_transform(
    t: Transform, parse: Callable[[str], ParsedSelector], compiled: list[CompiledTransform]
) -> None:
    _append_compiled_transform(compiled, _compile_allowlist_attrs_transform(cast("AllowlistAttrs", t), parse))


def _lower_drop_url_attrs_transform(
    t: Transform, parse: Callable[[str], ParsedSelector], compiled: list[CompiledTransform]
) -> None:
    _append_compiled_transform(compiled, _compile_drop_url_attrs_transform(cast("DropUrlAttrs", t), parse))


def _lower_allow_style_attrs_transform(
    t: Transform, parse: Callable[[str], ParsedSelector], compiled: list[CompiledTransform]
) -> None:
    _append_compiled_transform(compiled, _compile_allow_style_attrs_transform(cast("AllowStyleAttrs", t), parse))


def _lower_merge_attrs_transform(
    t: Transform, _parse: Callable[[str], ParsedSelector], compiled: list[CompiledTransform]
) -> None:
    compiled_merge = _compile_merge_attrs_transform(cast("MergeAttrs", t))
    if compiled_merge is not None:
        compiled.append(compiled_merge)


def _lower_sanitize_transform(
    t: Transform, _parse: Callable[[str], ParsedSelector], compiled: list[CompiledTransform]
) -> None:
    for sub_t in _compile_sanitize_transform(cast("Sanitize", t)):
        _append_compiled_transform(compiled, sub_t)


_TRANSFORM_LOWERERS = {
    SetAttrs: _lower_setattrs_transform,
    Drop: _lower_drop_transform,
    Unwrap: _lower_unwrap_transform,
    Escape: _lower_escape_transform,
    Empty: _lower_empty_transform,
    Edit: _lower_edit_transform,
    EditDocument: _lower_edit_document_transform,
    Decide: _lower_decide_transform,
    EditAttrs: _lower_edit_attrs_transform,
    Linkify: _lower_linkify_transform,
    CollapseWhitespace: _lower_collapse_whitespace_transform,
    PruneEmpty: _lower_prune_empty_transform,
    DropComments: _lower_drop_comments_transform,
    DropDoctype: _lower_drop_doctype_transform,
    DropForeignNamespaces: _lower_drop_foreign_namespaces_transform,
    DropAttrs: _lower_drop_attrs_transform,
    AllowlistAttrs: _lower_allowlist_attrs_transform,
    DropUrlAttrs: _lower_drop_url_attrs_transform,
    AllowStyleAttrs: _lower_allow_style_attrs_transform,
    MergeAttrs: _lower_merge_attrs_transform,
    Sanitize: _lower_sanitize_transform,
}


def compile_transforms(
    transforms: list[TransformSpec] | tuple[TransformSpec, ...],
    *,
    _selector_limits: SelectorLimits | None = None,
) -> list[CompiledTransform]:
    if not transforms:
        return []

    flattened = _iter_flattened_transforms(transforms)
    for t in flattened:
        if not isinstance(t, _TRANSFORM_CLASSES):
            raise TypeError(f"Unsupported transform: {type(t).__name__}")

    selector_limits = _selector_limits or _selector_limits_from_flattened(flattened)

    def _parse_selector(selector: str) -> ParsedSelector:
        return parse_selector(selector, limits=selector_limits)

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
                compiled_stage.extend(
                    compile_transforms(
                        (inner,),
                        _selector_limits=selector_limits,
                    )
                )
        return compiled_stage

    compiled: list[CompiledTransform] = []

    for t in flattened:
        if not t.enabled:
            continue
        lowerer = _TRANSFORM_LOWERERS.get(type(t))
        if lowerer is None:
            raise TypeError(f"Unsupported transform: {type(t).__name__}")  # pragma: no cover
        lowerer(t, _parse_selector, compiled)

    return compiled


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

            if isinstance(t, _CompiledSanitizeRawtextPolicy):
                _sanitize_rawtext_element_contents(root, policy=t.policy, errors=errors)
                i += 1
                continue

            raise TypeError(f"Unsupported compiled transform: {type(t).__name__}")

        apply_walk_transforms(root, pending_walk)
    finally:
        _ERROR_SINK.reset(token)
