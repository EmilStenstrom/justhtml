"""Experimental default-safe parser.

This is a proof-of-concept engine for the narrow ``JustHTML(str)`` default-safe
path. It intentionally does not reuse the tokenizer or treebuilder; the goal is
to measure whether a single Python loop can plausibly reach a 2x speedup before
investing in HTML5 parity.
"""

from __future__ import annotations

import re

from justhtml.core.constants import HEADING_ELEMENTS, IMPLIED_END_TAGS, TABLE_ALLOWED_CHILDREN, VOID_ELEMENTS
from justhtml.core.entities import decode_entities_in_text
from justhtml.dom import Document, DocumentFragment, Element, Node, Template, Text
from justhtml.sanitizer import DEFAULT_DOCUMENT_POLICY, DEFAULT_POLICY, _strip_invisible_unicode
from justhtml.sanitizer.url import _URL_SINK_ATTRS, _sanitize_url_sink_value, _url_sink_kind_for_attr
from justhtml.tokenizer.tokens import Doctype

_TAG_NAME_RE = re.compile(r"[A-Za-z][^\t\n\f />]*")
_ATTR_NAME_RE = re.compile(r"[^\t\n\f />=\0\"'<]+")
_DOCTYPE_RE = re.compile(
    r"""\s*([^\s>]+)(?:\s+(PUBLIC|SYSTEM)\s+(?:"([^"]*)"|'([^']*)')(?:\s+(?:"([^"]*)"|'([^']*)'))?)?""",
    re.IGNORECASE,
)
_SPACE = " \t\n\f\r"
_DROP_CONTENT_TAGS = {"script", "style"}
_DROP_SUBTREE_TAGS = {"svg", "math"}
_RCDATA_TAGS = {"title", "textarea"}
_RAWTEXT_AS_TEXT_TAGS = {"noscript"}
_HEAD_CONTENT_TAGS = {"base", "basefont", "bgsound", "link", "meta", "noscript", "script", "style", "template", "title"}
_P_CLOSING_START_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "center",
    "details",
    "dialog",
    "dir",
    "div",
    "dl",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "header",
    "hgroup",
    "hr",
    "main",
    "menu",
    "nav",
    "ol",
    "p",
    "pre",
    "search",
    "section",
    "summary",
    "table",
    "ul",
} | HEADING_ELEMENTS
_TABLE_SECTION_TAGS = {"tbody", "thead", "tfoot"}
_TABLE_CELL_TAGS = {"td", "th"}
_TABLE_FOSTER_TARGETS = {"table", "tbody", "tfoot", "thead", "tr"}


class DefaultSafeEngine:
    __slots__ = (
        "_allowed_attrs",
        "_allowed_by_tag",
        "_allowed_global",
        "_allowed_tags",
        "_body",
        "_body_explicit",
        "_doc",
        "_doctype_seen",
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
        self._doctype_seen = False
        self._body_explicit = False
        self._html: Element | None = None
        self._head: Element | None = None
        self._body: Element | DocumentFragment
        self._stack: list[Node] = []

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
                if self._lower_input.startswith("<!doctype", lt):
                    pos = self._parse_doctype(pos + 8, end)
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

    def _clean_text(self, raw: str) -> str:
        text = raw
        if "&" in text:
            text = decode_entities_in_text(text)
        if text and not text.isascii():
            text = _strip_invisible_unicode(text)
        return text

    def _append_text(self, raw: str) -> None:
        if not raw:
            return
        parent = self._current_parent()
        if (
            not self._fragment
            and parent is self._body
            and not self._body_explicit
            and not self._body_has_content()
            and raw.strip(_SPACE) == ""
        ):
            return
        text = self._clean_text(raw)
        if text:
            foster = self._foster_parent_for(parent) if text.strip(_SPACE) else None
            if foster is None:
                self._append(parent, Text(text))
            else:
                foster_parent, position = foster
                self._insert_at(foster_parent, position, Text(text))

    def _parse_doctype(self, pos: int, end: int) -> int:
        html = self._html_input
        gt = html.find(">", pos, end)
        doctype_end = end if gt == -1 else gt
        if not self._fragment and not self._policy.drop_doctype and not self._doctype_seen:
            raw = html[pos:doctype_end]
            match = _DOCTYPE_RE.match(raw)
            if match:
                name = match.group(1)
                kind = match.group(2)
                first_id = match.group(3) if match.group(3) is not None else match.group(4)
                second_id = match.group(5) if match.group(5) is not None else match.group(6)
                if kind is not None and kind.lower() == "public":
                    public_id = first_id
                    system_id = second_id
                elif kind is not None and kind.lower() == "system":
                    public_id = None
                    system_id = first_id
                else:
                    public_id = None
                    system_id = None
                self._prepend_doctype(Doctype(name.lower(), public_id, system_id))
            else:
                self._prepend_doctype(Doctype("html"))
        return end if gt == -1 else gt + 1

    def _prepend_doctype(self, doctype: Doctype) -> None:
        children = self._doc.children
        if children is None:
            return
        node = Node("!doctype", data=doctype)
        children.insert(0, node)
        node.parent = self._doc
        self._doctype_seen = True

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
        idx = self._find_open_index(name)
        if idx is None:
            if name == "p":
                self._insert_allowed_element("p", {}, False, self._current_parent())
                self._close_until("p")
            return pos
        if name in IMPLIED_END_TAGS:
            self._generate_implied_end_tags(name)
        del stack[idx:]
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
                self._body_explicit = True
                self._stack = [self._doc, self._html, self._body]  # type: ignore[list-item]
                return pos

        if not self._fragment and self._current_parent() is self._head and name not in _HEAD_CONTENT_TAGS:
            self._stack = [self._doc, self._html, self._body]  # type: ignore[list-item]

        if name in _DROP_CONTENT_TAGS:
            return self._skip_rawtext(name, pos, end)
        if name in _DROP_SUBTREE_TAGS:
            return self._skip_subtree(name, pos, end)
        if name in _RAWTEXT_AS_TEXT_TAGS:
            return self._parse_rawtext_as_text(name, pos, end)
        if name in _RCDATA_TAGS:
            return self._parse_rcdata_element(name, attrs, self_closing, pos, end)

        parent: Node
        if name in _HEAD_CONTENT_TAGS and not self._fragment and self._head is not None and not self._body_has_content():
            parent = self._head
        else:
            self._repair_stack_for_start(name)
            parent = self._current_parent()

        if name in _TABLE_SECTION_TAGS | {"tr", "td", "th"}:
            self._repair_table_for_start(name)
            parent = self._current_parent()

        if name not in self._allowed_tags:
            return pos

        self._insert_allowed_element(name, attrs, self_closing, parent)
        return pos

    def _insert_allowed_element(
        self,
        name: str,
        attrs: dict[str, str | None],
        self_closing: bool,
        parent: Node,
    ) -> Element:
        attrs = self._sanitize_attrs(name, attrs)
        node: Element
        if name == "template":
            node = Template(name, attrs, namespace="html")
        else:
            node = Element(name, attrs, "html")
        node._self_closing = self_closing
        foster = self._foster_parent_for(parent, for_tag=name)
        if foster is None:
            self._append(parent, node)
        else:
            foster_parent, position = foster
            self._insert_at(foster_parent, position, node)
        if name not in VOID_ELEMENTS and not self_closing:
            self._stack.append(node)
        return node

    def _insert_at(self, parent: Node, position: int, node: Node | Text) -> None:
        children = parent.children
        if children is None:
            return
        if type(node) is Text:
            if position > 0 and type(children[position - 1]) is Text:
                children[position - 1].data = (children[position - 1].data or "") + (node.data or "")
                return
            if position < len(children) and type(children[position]) is Text:
                children[position].data = (node.data or "") + (children[position].data or "")
                return
        children.insert(position, node)
        node.parent = parent

    def _body_has_content(self) -> bool:
        if self._fragment:
            return bool(self._body.children)
        body = self._body
        return bool(body.children)

    def _find_open_index(self, name: str) -> int | None:
        stack = self._stack
        for idx in range(len(stack) - 1, 0, -1):
            if getattr(stack[idx], "name", None) == name:
                return idx
        return None

    def _close_until(self, name: str) -> None:
        idx = self._find_open_index(name)
        if idx is not None:
            del self._stack[idx:]

    def _generate_implied_end_tags(self, exclude: str | None = None) -> None:
        stack = self._stack
        while len(stack) > 1:
            name = getattr(stack[-1], "name", None)
            if name in IMPLIED_END_TAGS and name != exclude:
                stack.pop()
                continue
            break

    def _repair_stack_for_start(self, name: str) -> None:
        if name in _P_CLOSING_START_TAGS and self._find_open_index("p") is not None:
            self._close_until("p")

        if name == "li":
            self._close_until("li")
            return

        if name in {"dd", "dt"}:
            self._close_until("dd")
            self._close_until("dt")
            return

        if name == "option":
            self._close_until("option")
            return

        if name == "optgroup":
            self._close_until("option")
            self._close_until("optgroup")
            return

        if name == "a":
            self._close_until("a")
            return

        if name in HEADING_ELEMENTS and getattr(self._stack[-1], "name", None) in HEADING_ELEMENTS:
            self._stack.pop()

    def _repair_table_for_start(self, name: str) -> None:
        if name in _TABLE_SECTION_TAGS:
            self._close_table_cell()
            self._close_until("tr")
            for section in _TABLE_SECTION_TAGS:
                self._close_until(section)
            if getattr(self._current_parent(), "name", None) != "table":
                table_idx = self._find_open_index("table")
                if table_idx is not None:
                    del self._stack[table_idx + 1 :]
            return

        if name == "tr":
            self._close_table_cell()
            self._close_until("tr")
            parent_name = getattr(self._current_parent(), "name", None)
            if parent_name == "table":
                self._insert_allowed_element("tbody", {}, False, self._current_parent())
            elif parent_name not in _TABLE_SECTION_TAGS:
                table_idx = self._find_open_index("table")
                if table_idx is not None:
                    del self._stack[table_idx + 1 :]
                    self._insert_allowed_element("tbody", {}, False, self._current_parent())
            return

        if name in _TABLE_CELL_TAGS:
            self._close_table_cell()
            parent_name = getattr(self._current_parent(), "name", None)
            if parent_name == "tr":
                return
            if parent_name == "table":
                self._insert_allowed_element("tbody", {}, False, self._current_parent())
                self._insert_allowed_element("tr", {}, False, self._current_parent())
                return
            if parent_name in _TABLE_SECTION_TAGS:
                self._insert_allowed_element("tr", {}, False, self._current_parent())
                return
            table_idx = self._find_open_index("table")
            if table_idx is not None:
                del self._stack[table_idx + 1 :]
                self._insert_allowed_element("tbody", {}, False, self._current_parent())
                self._insert_allowed_element("tr", {}, False, self._current_parent())

    def _close_table_cell(self) -> None:
        td_idx = self._find_open_index("td")
        th_idx = self._find_open_index("th")
        idxs = [idx for idx in (td_idx, th_idx) if idx is not None]
        if idxs:
            del self._stack[max(idxs) :]

    def _foster_parent_for(self, parent: Node, *, for_tag: str | None = None) -> tuple[Node, int] | None:
        if getattr(parent, "name", None) not in _TABLE_FOSTER_TARGETS:
            return None
        if for_tag is not None and for_tag in TABLE_ALLOWED_CHILDREN:
            return None
        table_idx = self._find_open_index("table")
        if table_idx is None:
            return None
        table = self._stack[table_idx]
        table_parent = table.parent
        children = table_parent.children if table_parent is not None else None
        if table_parent is None or children is None:
            return None
        try:
            return table_parent, children.index(table)
        except ValueError:
            return None

    def _consume_until_end_tag(self, name: str, pos: int, end: int) -> tuple[str, int]:
        html = self._html_input
        close = self._lower_input.find(f"</{name}", pos, end)
        if close == -1:
            return html[pos:end], end
        gt = html.find(">", close + len(name) + 2, end)
        return html[pos:close], (end if gt == -1 else gt + 1)

    def _parse_rawtext_as_text(self, name: str, pos: int, end: int) -> int:
        raw_text, pos = self._consume_until_end_tag(name, pos, end)
        if not raw_text:
            return pos
        text = raw_text if raw_text.isascii() else _strip_invisible_unicode(raw_text)
        if not text:
            return pos
        parent: Node
        if (
            not self._fragment
            and not self._body_explicit
            and not self._body_has_content()
            and self._head is not None
        ):
            parent = self._head
        else:
            parent = self._current_parent()
        self._append(parent, Text(text))
        return pos

    def _parse_rcdata_element(
        self,
        name: str,
        attrs: dict[str, str | None],
        self_closing: bool,
        pos: int,
        end: int,
    ) -> int:
        raw_text, pos = self._consume_until_end_tag(name, pos, end)
        text = self._clean_text(raw_text)
        if name not in self._allowed_tags:
            if text:
                self._append(self._current_parent(), Text(text))
            return pos

        parent: Node
        if name == "title" and not self._fragment and self._head is not None:
            parent = self._head
        else:
            self._repair_stack_for_start(name)
            parent = self._current_parent()
        node = self._insert_allowed_element(name, attrs, False if name in _RCDATA_TAGS else self_closing, parent)
        if text:
            self._append(node, Text(text))
        if self._stack and self._stack[-1] is node:
            self._stack.pop()
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
