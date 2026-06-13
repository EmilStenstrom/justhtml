"""Pure-Python fused parse-engine prototypes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from justhtml.dom import Node, Template, Text
from justhtml.sanitizer import DEFAULT_DOCUMENT_POLICY, DEFAULT_POLICY, SanitizationPolicy, _strip_invisible_unicode
from justhtml.sanitizer.url import _URL_SINK_ATTRS, _sanitize_url_sink_value, _url_sink_kind_for_attr
from justhtml.tokenizer.tokens import CommentToken, DoctypeToken, Tag, TokenSinkResult
from justhtml.treebuilder import TreeBuilder

if TYPE_CHECKING:
    from collections.abc import Collection

_FOREIGN_ROOT_TAGS = {"math", "svg"}


def can_apply_default_sanitizer_fast_path(policy: SanitizationPolicy) -> bool:
    """Return True when ``apply_default_sanitizer_fast_path`` preserves behavior."""
    return policy is DEFAULT_POLICY or policy is DEFAULT_DOCUMENT_POLICY


class FusedDefaultTreeBuilder(TreeBuilder):  # pragma: no cover
    """Tree builder that sanitizes default-policy HTML before insertion.

    The tokenizer still owns lexical state, but the sink is a single
    sanitizer-aware tree-construction object: unsafe tags, attributes, comments,
    doctypes, and dropped-content text are filtered before the DOM sees them.
    """

    __slots__ = (
        "_fused_allowed_attributes",
        "_fused_allowed_by_tag",
        "_fused_allowed_global",
        "_fused_allowed_tags",
        "_fused_drop_comments",
        "_fused_drop_content_tags",
        "_fused_drop_doctype",
        "_fused_drop_stack",
        "_fused_nodes_to_unwrap",
        "_fused_policy",
        "_fused_strip_invisible",
        "_fused_url_policy",
        "_fused_url_rules",
    )

    def __init__(
        self,
        fragment_context: Any | None = None,
        iframe_srcdoc: bool = False,
        collect_errors: bool = False,
        scripting_enabled: bool = True,
        track_node_locations: bool = False,
        track_tag_spans: bool = False,
        *,
        policy: SanitizationPolicy,
    ) -> None:
        super().__init__(
            fragment_context=fragment_context,
            iframe_srcdoc=iframe_srcdoc,
            collect_errors=collect_errors,
            scripting_enabled=scripting_enabled,
            track_node_locations=track_node_locations,
            track_tag_spans=track_tag_spans,
        )
        self._fused_policy = policy
        self._fused_allowed_tags = policy.allowed_tags
        self._fused_drop_content_tags = policy.drop_content_tags
        self._fused_allowed_attributes = policy.allowed_attributes
        self._fused_allowed_global = policy.allowed_attributes.get("*", ())
        self._fused_allowed_by_tag = {
            str(tag).lower(): frozenset(self._fused_allowed_global).union(attrs)
            for tag, attrs in policy.allowed_attributes.items()
            if tag != "*"
        }
        self._fused_url_policy = policy.url_policy
        self._fused_url_rules = policy.url_policy.allow_rules
        self._fused_drop_comments = policy.drop_comments
        self._fused_drop_doctype = policy.drop_doctype
        self._fused_strip_invisible = policy.strip_invisible_unicode
        self._fused_drop_stack: list[str] = []
        self._fused_nodes_to_unwrap: list[Node] = []

    def _insert_element(self, tag: object, *, push: bool, namespace: str = "html") -> object:
        node = super()._insert_element(tag, push=push, namespace=namespace)
        if isinstance(node, Node):
            name = node.name if node.name.islower() else node.name.lower()
            if name not in self._fused_allowed_tags:
                self._fused_nodes_to_unwrap.append(node)
        return node

    def _fused_allowed_attrs_for(self, tag: str) -> frozenset[str] | Collection[str]:
        allowed = self._fused_allowed_by_tag.get(tag)
        return allowed if allowed is not None else self._fused_allowed_global

    def _fused_sanitize_attrs(self, tag: Tag, tag_name: str) -> bool:
        attrs = tag.attrs
        if not attrs:
            return True

        allowed = self._fused_allowed_attrs_for(tag_name)
        out: dict[str, str | None] = {}
        changed = False

        for raw_key, raw_value in attrs.items():
            key = raw_key if type(raw_key) is str else str(raw_key)
            if not key.strip():
                changed = True
                continue

            if key not in allowed:
                lowered = key if key.islower() else key.lower()
                if lowered not in allowed:
                    changed = True
                    continue
                key = lowered
                changed = True

            value = raw_value
            if self._fused_strip_invisible and value is not None:
                value_str = value if type(value) is str else str(value)
                value = _strip_invisible_unicode(value_str)
                changed = changed or value != raw_value

            lower_key = key if key.islower() else key.lower()
            if lower_key in _URL_SINK_ATTRS:
                sink_kind = _url_sink_kind_for_attr(tag=tag_name, attr=lower_key, attrs=attrs)
                if sink_kind is not None:
                    if value is None:
                        changed = True
                        continue
                    rule = self._fused_url_rules.get((tag_name, lower_key))
                    if rule is None:
                        changed = True
                        continue
                    sanitized = _sanitize_url_sink_value(
                        url_policy=self._fused_url_policy,
                        rule=rule,
                        tag=tag_name,
                        attr=lower_key,
                        kind=sink_kind,
                        value=value,
                    )
                    if sanitized is None:
                        changed = True
                        continue
                    key = lower_key
                    value = sanitized
                    changed = changed or key != raw_key or value != raw_value

            out[key] = value

        if changed:
            tag.attrs = out
        return True

    def _fused_inside_dropped_subtree(self, token: object) -> bool:
        if not self._fused_drop_stack:
            return False

        if type(token) is Tag:
            tag = token
            name = tag.name if tag.name.islower() else tag.name.lower()
            if tag.kind == Tag.START and name == self._fused_drop_stack[-1]:
                self._fused_drop_stack.append(name)
                return True
            if tag.kind == Tag.END and name == self._fused_drop_stack[-1]:
                self._fused_drop_stack.pop()
                if not self._fused_drop_stack:
                    self._fused_append_text_boundary()
                return True
        return True

    def _fused_append_text_boundary(self) -> None:
        if not self.open_elements:
            return
        target = self.open_elements[-1]
        parent = target.template_content if type(target) is Template else target
        if parent is None:
            return
        children = parent.children
        if children is None:
            return
        node = Text("")
        node.parent = parent
        children.append(node)

    def process_token(self, token: object) -> object:
        if self._fused_inside_dropped_subtree(token):
            return TokenSinkResult.Continue

        token_type = type(token)
        if token_type is CommentToken and self._fused_drop_comments:
            return TokenSinkResult.Continue

        if token_type is DoctypeToken and self._fused_drop_doctype:
            return TokenSinkResult.Continue

        if isinstance(token, Tag):
            tag = token
            name = tag.name if tag.name.islower() else tag.name.lower()

            if tag.kind == Tag.END:
                return super().process_token(tag)

            if name in _FOREIGN_ROOT_TAGS or name in self._fused_drop_content_tags:
                self._fused_drop_stack.append(name)
                return TokenSinkResult.Continue

            if name not in self._fused_allowed_tags:
                return super().process_token(tag)

            self._fused_sanitize_attrs(tag, name)
            return super().process_token(tag)

        return super().process_token(token)

    def process_characters(self, data: str) -> object:
        if self._fused_drop_stack:
            return TokenSinkResult.Continue
        if self._fused_strip_invisible and data and not data.isascii():
            data = _strip_invisible_unicode(data)
        return super().process_characters(data)

    def finish(self) -> Node:
        root = super().finish()
        self._fused_unwrap_recorded_nodes()
        self._fused_unwrap_unrecorded_disallowed_nodes(root)
        return root

    def _fused_unwrap_recorded_nodes(self) -> None:
        for node in reversed(self._fused_nodes_to_unwrap):
            if node.parent is not None:
                self._fused_unwrap_node(node)
        self._fused_nodes_to_unwrap.clear()

    def _fused_unwrap_unrecorded_disallowed_nodes(self, root: Node) -> None:
        stack: list[Node] = [root]
        while stack:
            node = stack.pop()
            children = node.children
            if children:
                stack.extend(reversed(children))
            if type(node) is Template and node.template_content is not None:
                stack.append(node.template_content)

            if node.parent is None:
                continue
            name = node.name
            if name.startswith("#") or name == "!doctype":
                continue
            tag = name if name.islower() else name.lower()
            if tag not in self._fused_allowed_tags:
                self._fused_unwrap_node(node)

    def _fused_unwrap_node(self, node: Node) -> None:
        parent = node.parent
        if parent is None or parent.children is None:
            return
        try:
            index = parent.children.index(node)
        except ValueError:
            return

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
        if moved:
            for child in moved:
                child.parent = parent
            parent.children[index : index + 1] = moved
        else:
            parent.children.pop(index)
        node.parent = None
