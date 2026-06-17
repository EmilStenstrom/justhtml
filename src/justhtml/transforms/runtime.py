"""Transform runtime walker.

This module applies compiled transform pipelines to the DOM tree. The compiled
IR and public facade remain in ``transforms.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from justhtml.core.constants import VOID_ELEMENTS
from justhtml.dom import Element, Node, Template, Text
from justhtml.sanitizer import _sanitize_rawtext_element_contents, _strip_invisible_unicode
from justhtml.selector import SelectorMatcher
from justhtml.serializer import serialize_end_tag, serialize_start_tag

from . import (
    _ERROR_SINK,
    _FOREIGN_ROOT_TAGS,
    CompiledTransform,
    _collapse_html_space_characters,
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
    _selector_limits_from_compiled,
)
from .linkify import CompiledLinkifyTransform, apply_linkify_transform
from .spec import DecideAction

if TYPE_CHECKING:
    from justhtml.core.types import ParseError


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

                    if node._end_tag_present is False:
                        return None

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

            created_start_index: dict[int, int] = {}

            def _mark_start(n: object, start_index: int) -> None:
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
                action: Any,
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

            def apply_to_children(
                parent: Node,
                *,
                skip_linkify: bool,
                skip_whitespace: bool,
                foreign_context: bool,
            ) -> None:
                wt_len = len(walk_transforms)
                stack: list[tuple[Node, int, bool, bool, bool]] = [
                    (parent, 0, skip_linkify, skip_whitespace, foreign_context)
                ]

                while stack:
                    parent, i, skip_linkify, skip_whitespace, foreign_context = stack[-1]
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
                    if foreign_context:
                        node_foreign_context = True
                    else:
                        ns = node.namespace
                        if ns not in (None, "html"):
                            node_foreign_context = True
                        elif is_special or is_doctype:
                            node_foreign_context = False
                        else:
                            lowered = name if name.islower() else name.lower()
                            node_foreign_context = lowered in _FOREIGN_ROOT_TAGS

                    changed = False
                    matcher: SelectorMatcher | None = None
                    if created_start_index:
                        start_at = created_start_index.get(id(node), 0)
                    else:
                        start_at = 0
                    for idx in range(start_at, wt_len):
                        t: Any = walk_transforms[idx]
                        k: str = t.kind

                        if k == "drop_foreign_namespaces":
                            if is_special or is_doctype:
                                continue
                            if not node_foreign_context:
                                continue
                            if t.callback is not None:
                                t.callback(node)
                            if t.report is not None:
                                tag = str(name).lower()
                                t.report(f"Unsafe tag '{tag}' (foreign namespace)", node=node)
                            children.pop(i)
                            node.parent = None
                            changed = True
                            break

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

                        if k == "edit_attrs_chain":
                            if is_special or is_doctype:
                                continue
                            if not t.all_nodes:
                                sel = t.selector
                                if matcher is None:
                                    matcher = SelectorMatcher(limits=selector_limits)
                                if not matcher.matches(node, sel):
                                    continue
                            for chain_func in t.funcs:
                                chain_out = chain_func(node)
                                if chain_out is not None:
                                    node.attrs = chain_out
                            continue

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

                        if k == "strip_invisible_unicode":
                            if is_text:
                                text_data = str(node.data or "")
                                if text_data and not text_data.isascii():
                                    stripped_text = _strip_invisible_unicode(text_data)
                                    if stripped_text != text_data:  # pragma: no branch
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
                                value = raw_value if type(raw_value) is str else str(raw_value)
                                if value.isascii():
                                    continue
                                stripped_value = _strip_invisible_unicode(value)
                                if stripped_value == value:
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

                        if k == "decide":
                            if t.all_nodes:
                                action = t.callback(node)
                            else:
                                if is_special or is_doctype:
                                    continue
                                sel = t.selector
                                if matcher is None:
                                    matcher = SelectorMatcher(limits=selector_limits)
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

                        if k == "decide_chain":
                            if t.all_nodes:
                                action = DecideAction.KEEP
                                for chain_cb in t.callbacks:
                                    action = chain_cb(node)
                                    if action is not DecideAction.KEEP:
                                        break
                            else:
                                if is_special or is_doctype:
                                    continue
                                sel = t.selector
                                if matcher is None:
                                    matcher = SelectorMatcher(limits=selector_limits)
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

                        if k == "edit_attrs":
                            if is_special or is_doctype:
                                continue
                            if not t.all_nodes:
                                sel = t.selector
                                if matcher is None:
                                    matcher = SelectorMatcher(limits=selector_limits)
                                if not matcher.matches(node, sel):
                                    continue
                            new_attrs = t.func(node)
                            if new_attrs is not None:
                                node.attrs = new_attrs
                            continue

                        if is_special or is_doctype:
                            continue

                        if matcher is None:
                            matcher = SelectorMatcher(limits=selector_limits)
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

                    stack[-1] = (parent, i + 1, skip_linkify, skip_whitespace, foreign_context)

                    if is_special:
                        if not is_text and not is_comment and node.children:
                            stack.append((node, 0, skip_linkify, skip_whitespace, node_foreign_context))
                        continue

                    if linkify_skip_tags or whitespace_skip_tags:
                        tag = node.name.lower()
                        child_skip = skip_linkify or (tag in linkify_skip_tags)
                        child_skip_ws = skip_whitespace or (tag in whitespace_skip_tags)
                    else:
                        child_skip = skip_linkify
                        child_skip_ws = skip_whitespace

                    if type(node) is Template and node.template_content is not None and node.template_content.children:
                        stack.append((node.template_content, 0, child_skip, child_skip_ws, node_foreign_context))
                    if node.children:
                        stack.append((node, 0, child_skip, child_skip_ws, node_foreign_context))

            if type(root_node) is not Text:
                apply_to_children(root_node, skip_linkify=False, skip_whitespace=False, foreign_context=False)

                if type(root_node) is Template and root_node.template_content is not None:
                    apply_to_children(
                        root_node.template_content,
                        skip_linkify=False,
                        skip_whitespace=False,
                        foreign_context=False,
                    )

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
                    _CompiledDropForeignNamespacesTransform,
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

            if isinstance(t, _CompiledSelectorLimitsTransform):
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
