"""Experimental default-safe parser.

This is a proof-of-concept engine for the narrow ``JustHTML(str)`` default-safe
path. It intentionally does not reuse the tokenizer or treebuilder; the goal is
to measure whether a single Python loop can plausibly reach a 2x speedup before
investing in HTML5 parity.
"""

from __future__ import annotations

import re

from justhtml.core.constants import VOID_ELEMENTS
from justhtml.core.entities import decode_entities_in_text
from justhtml.dom import Document, DocumentFragment, Element, Node, Template, Text
from justhtml.sanitizer import DEFAULT_DOCUMENT_POLICY, DEFAULT_POLICY, _strip_invisible_unicode
from justhtml.sanitizer.url import _URL_SINK_ATTRS, _sanitize_url_sink_value, _url_sink_kind_for_attr

_TAG_NAME_RE = re.compile(r"[A-Za-z][^\t\n\f />]*")
_ATTR_NAME_RE = re.compile(r"[^\t\n\f />=\0\"'<]+")
_SPACE = " \t\n\f\r"
_DROP_CONTENT_TAGS = {"script", "style"}
_DROP_SUBTREE_TAGS = {"svg", "math"}


class DefaultSafeEngine:
    __slots__ = (
        "_allowed_attrs",
        "_allowed_by_tag",
        "_allowed_global",
        "_allowed_tags",
        "_body",
        "_doc",
        "_fragment",
        "_head",
        "_html",
        "_html_input",
        "_length",
        "_lower_input",
        "_policy",
        "_stack",
        "_url_policy",
        "_url_rules",
    )

    def __init__(self, html: str, *, fragment: bool) -> None:
        self._html_input = html
        self._length = len(html)
        self._lower_input = html.lower()
        self._fragment = bool(fragment)
        self._policy = DEFAULT_POLICY if fragment else DEFAULT_DOCUMENT_POLICY
        self._allowed_tags = self._policy.allowed_tags
        self._allowed_attrs = self._policy.allowed_attributes
        self._allowed_global = self._allowed_attrs.get("*", ())
        self._allowed_by_tag = {
            str(tag).lower(): frozenset(self._allowed_global).union(attrs)
            for tag, attrs in self._allowed_attrs.items()
            if tag != "*"
        }
        self._url_policy = self._policy.url_policy
        self._url_rules = self._url_policy.allow_rules
        self._doc: Document | DocumentFragment
        self._html: Element | None = None
        self._head: Element | None = None
        self._body: Element | DocumentFragment
        self._stack: list[Node | Text] = []

    def parse(self) -> Document | DocumentFragment:
        if self._fragment:
            root = DocumentFragment()
            self._doc = root
            self._body = root
            self._stack = [root]
        else:
            doc = Document()
            html_el = Element("html", {}, "html")
            head = Element("head", {}, "html")
            body = Element("body", {}, "html")
            self._append(doc, html_el)
            self._append(html_el, head)
            self._append(html_el, body)
            self._doc = doc
            self._html = html_el
            self._head = head
            self._body = body
            self._stack = [doc, html_el, body]

        self._parse_range(0, self._length)
        return self._doc

    def _append(self, parent: Node, node: Node | Text) -> None:
        children = parent.children
        if children is None:
            return
        if type(node) is Text and children and type(children[-1]) is Text:
            children[-1].data = (children[-1].data or "") + (node.data or "")
            return
        children.append(node)
        node.parent = parent

    def _current_parent(self) -> Node:
        current = self._stack[-1]
        if type(current) is Template and current.template_content is not None:
            return current.template_content
        return current  # type: ignore[return-value]

    def _parse_range(self, pos: int, end: int) -> int:
        html = self._html_input
        while pos < end:
            lt = html.find("<", pos, end)
            if lt == -1:
                self._append_text(html[pos:end])
                return end
            if lt > pos:
                self._append_text(html[pos:lt])
            pos = lt + 1
            if pos >= end:
                self._append_text("<")
                return end

            ch = html[pos]
            if ch == "!":
                if html.startswith("<!--", lt):
                    close = html.find("-->", pos + 1, end)
                    pos = end if close == -1 else close + 3
                    continue
                gt = html.find(">", pos + 1, end)
                pos = end if gt == -1 else gt + 1
                continue
            if ch == "/":
                pos = self._parse_end_tag(pos + 1, end)
                continue
            if not (("a" <= ch <= "z") or ("A" <= ch <= "Z")):
                self._append_text("<")
                continue
            pos = self._parse_start_tag(pos, end)
        return pos

    def _append_text(self, raw: str) -> None:
        if not raw:
            return
        text = raw
        if "&" in text:
            text = decode_entities_in_text(text)
        if not text.isascii():
            text = _strip_invisible_unicode(text)
        if text:
            self._append(self._current_parent(), Text(text))

    def _parse_end_tag(self, pos: int, end: int) -> int:
        html = self._html_input
        match = _TAG_NAME_RE.match(html, pos, end)
        if not match:
            return pos
        name = match.group(0)
        if not name.islower():
            name = name.lower()
        gt = html.find(">", match.end(), end)
        pos = end if gt == -1 else gt + 1

        if not self._fragment and name in {"html", "body"}:
            self._stack = [self._doc, self._html, self._body]  # type: ignore[list-item]
            return pos
        if not self._fragment and name == "head":
            self._stack = [self._doc, self._html, self._body]  # type: ignore[list-item]
            return pos

        stack = self._stack
        for idx in range(len(stack) - 1, 0, -1):
            node = stack[idx]
            if getattr(node, "name", None) == name:
                del stack[idx:]
                break
        return pos

    def _parse_start_tag(self, pos: int, end: int) -> int:
        html = self._html_input
        match = _TAG_NAME_RE.match(html, pos, end)
        if not match:
            self._append_text("<")
            return pos
        raw_name = match.group(0)
        name = raw_name if raw_name.islower() else raw_name.lower()
        pos = match.end()

        attrs, self_closing, pos = self._parse_attrs(pos, end)
        if not self._fragment:
            if name == "html":
                if self._html is not None:
                    self._html.attrs.update(self._sanitize_attrs("html", attrs))
                return pos
            if name == "head":
                if self._head is not None:
                    self._stack = [self._doc, self._html, self._head]  # type: ignore[list-item]
                return pos
            if name == "body":
                if isinstance(self._body, Element):
                    self._body.attrs.update(self._sanitize_attrs("body", attrs))
                self._stack = [self._doc, self._html, self._body]  # type: ignore[list-item]
                return pos

        if name in _DROP_CONTENT_TAGS:
            return self._skip_rawtext(name, pos, end)
        if name in _DROP_SUBTREE_TAGS:
            return self._skip_subtree(name, pos, end)

        parent: Node
        if name == "title" and not self._fragment and self._head is not None:
            parent = self._head
        else:
            parent = self._current_parent()

        if name == "tr" and getattr(parent, "name", None) == "table":
            tbody = Element("tbody", {}, "html")
            self._append(parent, tbody)
            self._stack.append(tbody)
            parent = tbody

        if name not in self._allowed_tags:
            return pos

        attrs = self._sanitize_attrs(name, attrs)
        node: Element
        if name == "template":
            node = Template(name, attrs, namespace="html")
        else:
            node = Element(name, attrs, "html")
        node._self_closing = self_closing
        self._append(parent, node)
        if name not in VOID_ELEMENTS and not self_closing:
            self._stack.append(node)
        return pos

    def _parse_attrs(self, pos: int, end: int) -> tuple[dict[str, str | None], bool, int]:
        html = self._html_input
        attrs: dict[str, str | None] = {}
        self_closing = False
        while pos < end:
            while pos < end and html[pos] in _SPACE:
                pos += 1
            if pos >= end:
                return attrs, self_closing, pos
            ch = html[pos]
            if ch == ">":
                return attrs, self_closing, pos + 1
            if ch == "/" and pos + 1 < end and html[pos + 1] == ">":
                return attrs, True, pos + 2

            match = _ATTR_NAME_RE.match(html, pos, end)
            if not match:
                pos += 1
                continue
            key = match.group(0)
            if not key.islower():
                key = key.lower()
            pos = match.end()
            while pos < end and html[pos] in _SPACE:
                pos += 1
            value: str | None = ""
            if pos < end and html[pos] == "=":
                pos += 1
                while pos < end and html[pos] in _SPACE:
                    pos += 1
                if pos < end and html[pos] in "\"'":
                    quote = html[pos]
                    pos += 1
                    value_start = pos
                    close = html.find(quote, pos, end)
                    if close == -1:
                        value = html[value_start:end]
                        pos = end
                    else:
                        value = html[value_start:close]
                        pos = close + 1
                else:
                    value_start = pos
                    while pos < end and html[pos] not in _SPACE + ">":
                        pos += 1
                    value = html[value_start:pos]
            if isinstance(value, str) and "&" in value:
                value = decode_entities_in_text(value, in_attribute=True)
            attrs[key] = value
        return attrs, self_closing, pos

    def _sanitize_attrs(self, tag: str, attrs: dict[str, str | None]) -> dict[str, str | None]:
        if not attrs:
            return {}
        allowed = self._allowed_by_tag.get(tag, self._allowed_global)
        out: dict[str, str | None] = {}
        for key, raw_value in attrs.items():
            if key not in allowed:
                continue
            value = raw_value
            if value is not None and not value.isascii():
                value = _strip_invisible_unicode(value)
            if key in _URL_SINK_ATTRS:
                kind = _url_sink_kind_for_attr(tag=tag, attr=key, attrs=attrs)
                if kind is not None:
                    if value is None:
                        continue
                    rule = self._url_rules.get((tag, key))
                    if rule is None:
                        continue
                    value = _sanitize_url_sink_value(
                        url_policy=self._url_policy,
                        rule=rule,
                        tag=tag,
                        attr=key,
                        kind=kind,
                        value=value,
                    )
                    if value is None:
                        continue
            out[key] = value
        return out

    def _skip_rawtext(self, name: str, pos: int, end: int) -> int:
        html = self._html_input
        close = self._lower_input.find(f"</{name}", pos, end)
        if close == -1:
            self._append(self._current_parent(), Text(""))
            return end
        gt = html.find(">", close + len(name) + 2, end)
        self._append(self._current_parent(), Text(""))
        return end if gt == -1 else gt + 1

    def _skip_subtree(self, name: str, pos: int, end: int) -> int:
        html = self._html_input
        depth = 1
        while pos < end and depth:
            lt = html.find("<", pos, end)
            if lt == -1:
                return end
            p = lt + 1
            is_end = p < end and html[p] == "/"
            if is_end:
                p += 1
            match = _TAG_NAME_RE.match(html, p, end)
            if not match:
                pos = p
                continue
            tag = match.group(0)
            if not tag.islower():
                tag = tag.lower()
            gt = html.find(">", match.end(), end)
            pos = end if gt == -1 else gt + 1
            if tag == name:
                depth += -1 if is_end else 1
        return pos
