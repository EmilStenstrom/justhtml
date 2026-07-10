"""Transform compilation helpers.

This module lowers high-level transform specs into the compact compiled
representation consumed by the runtime walker in ``transforms.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from justhtml.sanitizer import DEFAULT_POLICY, SanitizationPolicy, UrlPolicy, _sanitize_inline_style
from justhtml.sanitizer.rawtext import _RAWTEXT_SERIALIZATION_ELEMENTS
from justhtml.sanitizer.url import _URL_SINK_ATTRS, _sanitize_url_sink_value, _url_sink_kind_for_attr
from justhtml.selector import DEFAULT_SELECTOR_LIMITS, SelectorLimits, parse_selector

from . import (
    CompiledTransform,
    Stage,
    Transform,
    TransformSpec,
    _CompiledCollapseWhitespaceTransform,
    _CompiledDecideChain,
    _CompiledDecideElementsChain,
    _CompiledDecideTransform,
    _CompiledDropCommentsTransform,
    _CompiledDropDoctypeTransform,
    _CompiledDropForeignNamespacesTransform,
    _CompiledEditAttrsChain,
    _CompiledEditAttrsTransform,
    _CompiledEditDocumentTransform,
    _CompiledHardenRawtextTransform,
    _CompiledMergeAttrTokensTransform,
    _CompiledPruneEmptyTransform,
    _CompiledSelectorLimitsTransform,
    _CompiledSelectorTransform,
    _CompiledStageBoundary,
    _CompiledStageHookTransform,
    _CompiledStripInvisibleUnicodeTransform,
)
from .linkify import compile_linkify_transform
from .spec import (
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
    from typing import Protocol

    from justhtml.dom import Node
    from justhtml.selector import ParsedSelector

    class NodeCallback(Protocol):
        def __call__(self, node: Node) -> None: ...

    class EditAttrsCallback(Protocol):
        def __call__(self, node: Node) -> dict[str, str | None] | None: ...

    class ReportCallback(Protocol):
        def __call__(self, msg: str, *, node: Any | None = None) -> None: ...


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
    HardenRawtext,
    DropComments,
    DropDoctype,
    DropForeignNamespaces,
    DropAttrs,
    AllowlistAttrs,
    DropUrlAttrs,
    AllowStyleAttrs,
    MergeAttrs,
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

    compiled.append(
        _CompiledDropForeignNamespacesTransform(
            kind="drop_foreign_namespaces",
            callback=t.callback,
            report=_report_unsafe,
        )
    )
    compiled.append(_CompiledDecideElementsChain(callbacks=[_sanitize_node_decision]))

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

    if policy.allowed_tags & _RAWTEXT_SERIALIZATION_ELEMENTS:
        compiled.append(
            _CompiledHardenRawtextTransform(
                kind="harden_rawtext",
                policy=policy,
            )
        )
    else:
        compiled.append(
            _CompiledSelectorLimitsTransform(
                kind="selector_limits",
                selector_limits=policy.selector_limits,
            )
        )
    return compiled


def _compile_harden_rawtext_transform(t: HardenRawtext) -> CompiledTransform:
    return _CompiledHardenRawtextTransform(
        kind="harden_rawtext",
        policy=t.policy or DEFAULT_POLICY,
    )


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


def _compile_drop_foreign_namespaces_transform(t: DropForeignNamespaces) -> _CompiledDropForeignNamespacesTransform:
    return _CompiledDropForeignNamespacesTransform(
        kind="drop_foreign_namespaces",
        callback=t.callback,
        report=t.report,
    )


def _compile_drop_attrs_transform(t: DropAttrs, parse: Callable[[str], ParsedSelector]) -> _CompiledEditAttrsTransform:
    patterns = t.patterns
    on_hook = t.callback
    on_report = t.report

    def _drop_attrs(
        node: Node,
        patterns: tuple[str, ...] = patterns,
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

        for raw_key in attrs:
            if not raw_key or not str(raw_key).strip():
                continue
            key = raw_key
            if not key.islower():
                key = key.lower()
            if any(_glob_match(pattern, key) for pattern in patterns):
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

            if any(_glob_match(pattern, key) for pattern in patterns):
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
        url_sink_attrs: frozenset[str] = _URL_SINK_ATTRS,
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
            rule = url_policy.allow_rules.get((tag, lower_key)) or url_policy.allow_rules.get(("*", lower_key))
            if lower_key not in url_sink_attrs and rule is None:
                continue
            raw_value = attrs[key]

            sink_kind = (
                _url_sink_kind_for_attr(tag=tag, attr=lower_key, attrs=attrs) if lower_key in url_sink_attrs else "url"
            )
            if sink_kind is None:
                continue

            if raw_value is None:
                if on_report is not None:  # pragma: no cover
                    on_report(f"Unsafe URL in attribute '{lower_key}'", node=node)
                if to_drop is None:
                    to_drop = []
                to_drop.append(key)
                continue

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
        trim_blocks=t.trim_blocks,
        block_tags=t.block_tags,
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


def _lower_harden_rawtext_transform(
    t: Transform, _parse: Callable[[str], ParsedSelector], compiled: list[CompiledTransform]
) -> None:
    compiled.append(_compile_harden_rawtext_transform(cast("HardenRawtext", t)))


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
    HardenRawtext: _lower_harden_rawtext_transform,
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
