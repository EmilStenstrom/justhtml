"""Pure-Python fused-engine prototypes.

This module holds narrowly-scoped pieces of the future fused parser while they
are being proven against the performance gate. The first piece is a specialized
default sanitizer that replaces the generic compiled-transform walker for the
safe-by-default constructor path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from justhtml.dom import Node, Template
from justhtml.sanitizer import DEFAULT_DOCUMENT_POLICY, DEFAULT_POLICY, SanitizationPolicy, _strip_invisible_unicode
from justhtml.sanitizer.url import _URL_SINK_ATTRS, _sanitize_url_sink_value, _url_sink_kind_for_attr

if TYPE_CHECKING:
    from collections.abc import Collection

_FOREIGN_ROOT_TAGS = {"math", "svg"}


def can_apply_default_sanitizer_fast_path(policy: SanitizationPolicy) -> bool:
    """Return True when ``apply_default_sanitizer_fast_path`` preserves behavior."""
    return policy is DEFAULT_POLICY or policy is DEFAULT_DOCUMENT_POLICY


def apply_default_sanitizer_fast_path(root: Node, policy: SanitizationPolicy) -> None:  # pragma: no cover
    """Apply the built-in sanitizer policies with one specialized DOM walk.

    This intentionally covers only the sealed default policies. Custom policies,
    callbacks, reports, and explicit transform pipelines stay on the generic
    transform runtime until the fused engine can model them exactly.
    """

    allowed_tags = policy.allowed_tags
    drop_content_tags = policy.drop_content_tags
    allowed_attributes = policy.allowed_attributes
    allowed_global = allowed_attributes.get("*", ())
    allowed_by_tag: dict[str, frozenset[str]] = {}
    for tag, attrs in allowed_attributes.items():
        if tag == "*":
            continue
        allowed_by_tag[str(tag).lower()] = frozenset(allowed_global).union(attrs)

    url_policy = policy.url_policy
    url_rules = url_policy.allow_rules
    drop_comments = policy.drop_comments
    drop_doctype = policy.drop_doctype
    strip_invisible = policy.strip_invisible_unicode

    def _allowed_attrs_for(tag: str) -> frozenset[str] | Collection[str]:
        allowed = allowed_by_tag.get(tag)
        return allowed if allowed is not None else allowed_global

    def _sanitize_attrs(node: Node, tag: str) -> None:
        attrs = node.attrs
        if not attrs:
            return

        allowed = _allowed_attrs_for(tag)
        out: dict[str, str | None] | None = None

        for raw_key, raw_value in attrs.items():
            key = raw_key if type(raw_key) is str else str(raw_key)
            if not key.strip():
                if out is None:
                    out = {}
                continue

            if key not in allowed:
                lowered = key if key.islower() else key.lower()
                if lowered not in allowed:
                    if out is None:
                        out = {}
                    continue
                key = lowered

            value = raw_value
            if strip_invisible and value is not None:
                value_str = value if type(value) is str else str(value)
                value = _strip_invisible_unicode(value_str)

            lower_key = key if key.islower() else key.lower()
            if lower_key in _URL_SINK_ATTRS:
                sink_kind = _url_sink_kind_for_attr(tag=tag, attr=lower_key, attrs=attrs)
                if sink_kind is not None:
                    if value is None:
                        if out is None:
                            out = {}
                        continue
                    rule = url_rules.get((tag, lower_key))
                    if rule is None:
                        if out is None:
                            out = {}
                        continue
                    sanitized = _sanitize_url_sink_value(
                        url_policy=url_policy,
                        rule=rule,
                        tag=tag,
                        attr=lower_key,
                        kind=sink_kind,
                        value=value,
                    )
                    if sanitized is None:
                        if out is None:
                            out = {}
                        continue
                    key = lower_key
                    value = sanitized

            if out is not None:
                out[key] = value
                continue

            if key != raw_key or value != raw_value:
                out = {}
                for prev_key, prev_value in attrs.items():
                    if prev_key == raw_key:
                        break
                    out[prev_key] = prev_value
                out[key] = value

        if out is not None:
            node.attrs = out

    def _children_for_unwrap(node: Node, parent: Node) -> list[Node]:
        moved: list[Node] = []
        if node.children:
            moved = node.children
            node.children = []
        if type(node) is Template and node.template_content is not None:
            content = node.template_content
            if content.children:
                if moved:
                    moved.extend(content.children)
                else:
                    moved = content.children
                content.children = []
        for child in moved:
            child.parent = parent
        return moved

    def _sanitize_children(parent: Node, *, foreign_context: bool) -> None:
        children = parent.children
        if not children:
            return

        i = 0
        while i < len(children):
            node = children[i]
            name = node.name

            if name == "#text":
                if strip_invisible and node.data and type(node.data) is str:
                    node.data = _strip_invisible_unicode(node.data)
                i += 1
                continue

            if name == "#comment":
                if drop_comments:
                    children.pop(i)
                    node.parent = None
                    continue
                i += 1
                continue

            if name == "!doctype":
                if drop_doctype:
                    children.pop(i)
                    node.parent = None
                    continue
                i += 1
                continue

            tag = name if name.islower() else name.lower()
            node_foreign = foreign_context or node.namespace not in (None, "html") or tag in _FOREIGN_ROOT_TAGS
            if node_foreign:
                children.pop(i)
                node.parent = None
                continue

            if tag not in allowed_tags:
                if tag in drop_content_tags:
                    children.pop(i)
                    node.parent = None
                    continue

                moved = _children_for_unwrap(node, parent)
                if moved:
                    children[i : i + 1] = moved
                else:
                    children.pop(i)
                node.parent = None
                continue

            _sanitize_attrs(node, tag)
            _sanitize_children(node, foreign_context=False)
            if type(node) is Template and node.template_content is not None:
                _sanitize_children(node.template_content, foreign_context=False)
            i += 1

    _sanitize_children(root, foreign_context=False)
    if type(root) is Template and root.template_content is not None:
        _sanitize_children(root.template_content, foreign_context=False)
