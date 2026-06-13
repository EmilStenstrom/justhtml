"""Experimental default-safe parser.

This is a proof-of-concept engine for the narrow ``JustHTML(str)`` default-safe
path. It intentionally does not reuse the tokenizer or treebuilder; the goal is
to measure whether a single Python loop can plausibly reach a 2x speedup before
investing in HTML5 parity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from justhtml.core.constants import HEADING_ELEMENTS, IMPLIED_END_TAGS, TABLE_ALLOWED_CHILDREN, VOID_ELEMENTS
from justhtml.core.entities import decode_entities_in_text
from justhtml.dom import Document, DocumentFragment, Element, Node, Template, Text
from justhtml.sanitizer import DEFAULT_DOCUMENT_POLICY, DEFAULT_POLICY, _strip_invisible_unicode
from justhtml.sanitizer.url import _URL_SINK_ATTRS, _sanitize_url_sink_value, _url_sink_kind_for_attr
from justhtml.tokenizer.tokens import Doctype

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from justhtml.sanitizer import SanitizationPolicy
    from justhtml.sanitizer.url import UrlPolicy, UrlRule

_TAG_NAME_RE = re.compile(r"[A-Za-z][^\t\n\f />]*")
_ATTR_NAME_RE = re.compile(r"[^\t\n\f />=\0\"'<]+")
_DOCTYPE_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9:_-]*$")
_DOCTYPE_RE = re.compile(
    r"""\s*([^\s>]+)(?:\s+(PUBLIC|SYSTEM)\s+(?:"([^"]*)"|'([^']*)')(?:\s+(?:"([^"]*)"|'([^']*)'))?)?""",
    re.IGNORECASE,
)
_SPACE = " \t\n\f\r"
_DROP_CONTENT_TAGS = {"script", "style"}
_DROP_SUBTREE_TAGS = {"svg", "math"}
_RCDATA_TAGS = {"title", "textarea"}
_RAWTEXT_AS_TEXT_TAGS = {"noscript"}
_PLAINTEXT_TAGS = {"plaintext"}
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
_DEFINITION_SCOPE_BOUNDARIES = {"dl"}
_LIST_ITEM_SCOPE_BOUNDARIES = {"ol", "ul"}
_PRE_LINEFEED_IGNORING_TAGS = {"listing", "pre"}
_TABLE_CONTEXT_BOUNDARIES = frozenset({"table"})
_TABLE_SECTION_TAGS = {"tbody", "thead", "tfoot"}
_TABLE_CELL_TAGS = {"td", "th"}
_TABLE_FOSTER_TARGETS = {"table", "tbody", "tfoot", "thead", "tr"}
_FRAMESET_BODY_OK_TAGS = {"div", "p"}
_FRAMESET_BLOCKING_START_TAGS = {
    "applet",
    "area",
    "button",
    "dd",
    "dt",
    "embed",
    "iframe",
    "input",
    "keygen",
    "listing",
    "marquee",
    "object",
    "select",
    "textarea",
    "wbr",
    "xmp",
}


@dataclass(frozen=True, slots=True)
class EnginePlan:
    """Compiled execution plan for a sanitizer-aware parse run.

    The executor copies these fields into instance slots so the hot path stays
    close to the hardcoded PoC while giving policy/behavior compilation a clear
    boundary.
    """

    policy: SanitizationPolicy
    allowed_tags: frozenset[str]
    allowed_global_attrs: Collection[str]
    allowed_attrs_by_tag: Mapping[str, frozenset[str]]
    url_policy: UrlPolicy
    url_rules: Mapping[tuple[str, str], UrlRule]
    definition_scope_boundaries: frozenset[str]
    drop_doctype: bool
    drop_content_tags: frozenset[str]
    drop_subtree_tags: frozenset[str]
    frameset_body_ok_tags: frozenset[str]
    frameset_blocking_start_tags: frozenset[str]
    head_content_tags: frozenset[str]
    implied_end_tags: frozenset[str]
    list_item_scope_boundaries: frozenset[str]
    p_closing_start_tags: frozenset[str]
    pre_linefeed_ignoring_tags: frozenset[str]
    rawtext_as_text_tags: frozenset[str]
    rcdata_tags: frozenset[str]
    plaintext_tags: frozenset[str]
    strip_invisible_unicode: bool
    table_allowed_children: frozenset[str]
    table_cell_tags: frozenset[str]
    table_foster_targets: frozenset[str]
    table_section_tags: frozenset[str]
    void_elements: frozenset[str]


_ENGINE_PLAN_CACHE: dict[bool, EnginePlan] = {}


def compile_default_engine_plan(*, fragment: bool) -> EnginePlan:
    """Return the cached default-safe execution plan."""
    cache_key = bool(fragment)
    cached = _ENGINE_PLAN_CACHE.get(cache_key)
    if cached is not None:
        return cached

    policy = DEFAULT_POLICY if fragment else DEFAULT_DOCUMENT_POLICY
    allowed_attrs = policy.allowed_attributes
    allowed_global = allowed_attrs.get("*", ())
    allowed_by_tag = {
        str(tag).lower(): frozenset(allowed_global).union(attrs)
        for tag, attrs in allowed_attrs.items()
        if tag != "*"
    }
    plan = EnginePlan(
        policy=policy,
        allowed_tags=policy.allowed_tags,
        allowed_global_attrs=allowed_global,
        allowed_attrs_by_tag=allowed_by_tag,
        url_policy=policy.url_policy,
        url_rules=policy.url_policy.allow_rules,
        definition_scope_boundaries=frozenset(_DEFINITION_SCOPE_BOUNDARIES),
        drop_doctype=policy.drop_doctype,
        drop_content_tags=frozenset(policy.drop_content_tags),
        drop_subtree_tags=frozenset(_DROP_SUBTREE_TAGS),
        frameset_body_ok_tags=frozenset(_FRAMESET_BODY_OK_TAGS),
        frameset_blocking_start_tags=frozenset(_FRAMESET_BLOCKING_START_TAGS),
        head_content_tags=frozenset(_HEAD_CONTENT_TAGS),
        implied_end_tags=frozenset(IMPLIED_END_TAGS),
        list_item_scope_boundaries=frozenset(_LIST_ITEM_SCOPE_BOUNDARIES),
        p_closing_start_tags=frozenset(_P_CLOSING_START_TAGS),
        pre_linefeed_ignoring_tags=frozenset(_PRE_LINEFEED_IGNORING_TAGS),
        rawtext_as_text_tags=frozenset(_RAWTEXT_AS_TEXT_TAGS),
        rcdata_tags=frozenset(_RCDATA_TAGS),
        plaintext_tags=frozenset(_PLAINTEXT_TAGS),
        strip_invisible_unicode=policy.strip_invisible_unicode,
        table_allowed_children=frozenset(TABLE_ALLOWED_CHILDREN),
        table_cell_tags=frozenset(_TABLE_CELL_TAGS),
        table_foster_targets=frozenset(_TABLE_FOSTER_TARGETS),
        table_section_tags=frozenset(_TABLE_SECTION_TAGS),
        void_elements=VOID_ELEMENTS,
    )
    _ENGINE_PLAN_CACHE[cache_key] = plan
    return plan


class DefaultSafeEngine:
    __slots__ = (
        "_after_head",
        "_allowed_by_tag",
        "_allowed_global",
        "_allowed_tags",
        "_body",
        "_body_explicit",
        "_definition_scope_boundaries",
        "_doc",
        "_doctype_seen",
        "_drop_content_tags",
        "_drop_doctype",
        "_drop_subtree_tags",
        "_dropped_to_eof",
        "_explicit_head",
        "_explicit_html",
        "_fragment",
        "_frameset_blocked",
        "_frameset_blocking_start_tags",
        "_frameset_body_ok_tags",
        "_frameset_seen",
        "_head",
        "_head_content_tags",
        "_html",
        "_html_input",
        "_implied_end_tags",
        "_length",
        "_list_item_scope_boundaries",
        "_lower_input",
        "_p_closing_start_tags",
        "_plaintext_tags",
        "_plan",
        "_pre_linefeed_ignoring_tags",
        "_rawtext_as_text_tags",
        "_rcdata_tags",
        "_stack",
        "_strip_invisible_unicode",
        "_table_allowed_children",
        "_table_cell_tags",
        "_table_foster_targets",
        "_table_section_tags",
        "_url_policy",
        "_url_rules",
        "_void_elements",
    )

    def __init__(self, html: str, *, fragment: bool, plan: EnginePlan | None = None) -> None:
        self._html_input = html
        self._length = len(html)
        self._lower_input = html.lower()
        self._fragment = bool(fragment)
        self._plan = plan if plan is not None else compile_default_engine_plan(fragment=fragment)
        self._allowed_tags = self._plan.allowed_tags
        self._allowed_global = self._plan.allowed_global_attrs
        self._allowed_by_tag = self._plan.allowed_attrs_by_tag
        self._url_policy = self._plan.url_policy
        self._url_rules = self._plan.url_rules
        self._definition_scope_boundaries = self._plan.definition_scope_boundaries
        self._drop_doctype = self._plan.drop_doctype
        self._drop_content_tags = self._plan.drop_content_tags
        self._drop_subtree_tags = self._plan.drop_subtree_tags
        self._frameset_body_ok_tags = self._plan.frameset_body_ok_tags
        self._frameset_blocking_start_tags = self._plan.frameset_blocking_start_tags
        self._head_content_tags = self._plan.head_content_tags
        self._implied_end_tags = self._plan.implied_end_tags
        self._list_item_scope_boundaries = self._plan.list_item_scope_boundaries
        self._p_closing_start_tags = self._plan.p_closing_start_tags
        self._plaintext_tags = self._plan.plaintext_tags
        self._pre_linefeed_ignoring_tags = self._plan.pre_linefeed_ignoring_tags
        self._rawtext_as_text_tags = self._plan.rawtext_as_text_tags
        self._rcdata_tags = self._plan.rcdata_tags
        self._strip_invisible_unicode = self._plan.strip_invisible_unicode
        self._table_allowed_children = self._plan.table_allowed_children
        self._table_cell_tags = self._plan.table_cell_tags
        self._table_foster_targets = self._plan.table_foster_targets
        self._table_section_tags = self._plan.table_section_tags
        self._void_elements = self._plan.void_elements
        self._doc: Document | DocumentFragment
        self._after_head = False
        self._dropped_to_eof = False
        self._doctype_seen = False
        self._explicit_head = False
        self._explicit_html = False
        self._frameset_blocked = False
        self._frameset_seen = False
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
        self._finish_document_shell()
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
            if ch == "?":
                gt = html.find(">", pos + 1, end)
                pos = end if gt == -1 else gt + 1
                continue
            if not (("a" <= ch <= "z") or ("A" <= ch <= "Z")):
                self._append_text("<")
                continue
            pos = self._parse_start_tag(pos, end)
        return pos

    def _clean_text(self, raw: str, *, replace_null: bool = False) -> str:
        text = raw
        if "\r" in text:
            text = text.replace("\r\n", "\n").replace("\r", "\n")
        if "\0" in text:
            text = text.replace("\0", "\ufffd" if replace_null else "")
        if "&" in text:
            text = decode_entities_in_text(text)
        if self._strip_invisible_unicode and text and not text.isascii():
            text = _strip_invisible_unicode(text)
        return text

    def _append_text(self, raw: str) -> None:
        if not raw:
            return
        if not self._fragment and self._frameset_seen and not self._body_explicit:
            self._append_frameset_text(raw)
            return
        parent = self._current_parent()
        if not self._fragment and parent is self._html and self._after_head and raw.strip(_SPACE) != "":
            self._stack = [self._doc, self._html, self._body]  # type: ignore[list-item]
            self._after_head = False
            parent = self._body
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
            if not self._fragment and parent is self._html and self._after_head and text.strip(_SPACE) == "":
                body = self._body
                children = parent.children
                position = len(children) if children is not None else 0
                if children is not None:
                    try:
                        position = children.index(body)
                    except ValueError:
                        pass
                self._insert_at(parent, position, Text(text))
                return
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
        if (
            not self._fragment
            and not self._drop_doctype
            and not self._doctype_seen
            and not self._explicit_html
            and not self._frameset_seen
            and not self._body_has_content()
        ):
            raw = html[pos:doctype_end]
            match = _DOCTYPE_RE.match(raw)
            if match:
                raw_name = match.group(1).lower()
                name = raw_name if _DOCTYPE_NAME_RE.match(raw_name) else None
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
                self._prepend_doctype(Doctype(name, public_id, system_id, force_quirks=name is None))
            else:
                self._prepend_doctype(Doctype(None, force_quirks=True))
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
            if pos >= end:
                self._append_text("</")
                return end
            gt = html.find(">", pos, end)
            return end if gt == -1 else gt + 1
        name = match.group(0)
        if not name.islower():
            name = name.lower()
        gt = html.find(">", match.end(), end)
        pos = end if gt == -1 else gt + 1

        if not self._fragment and name in {"html", "body"}:
            if self._frameset_seen and not self._body_explicit:
                self._stack = [self._doc, self._html]  # type: ignore[list-item]
            else:
                self._stack = [self._doc, self._html, self._body]  # type: ignore[list-item]
            self._after_head = False
            return pos
        if not self._fragment and name == "head":
            self._stack = [self._doc, self._html]  # type: ignore[list-item]
            self._after_head = True
            return pos
        if self._frameset_seen and not self._body_explicit:
            return pos
        if name == "br":
            self._insert_allowed_element("br", {}, False, self._current_parent())
            return pos

        stack = self._stack
        idx = self._find_open_index(name)
        if idx is None:
            if name == "p":
                self._insert_allowed_element("p", {}, False, self._current_parent())
                self._close_until("p")
            return pos
        if name in self._implied_end_tags:
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
        if name == "image":
            name = "img"
        pos = match.end()

        attrs, self_closing, pos = self._parse_attrs(pos, end)
        if not self._fragment:
            if name == "html":
                self._explicit_html = True
                if self._html is not None:
                    self._html.attrs.update(self._sanitize_attrs("html", attrs))
                return pos
            if name == "head":
                self._explicit_head = True
                if self._head is not None:
                    self._stack = [self._doc, self._html, self._head]  # type: ignore[list-item]
                    self._after_head = False
                return pos
            if name == "body":
                if isinstance(self._body, Element):
                    self._body.attrs.update(self._sanitize_attrs("body", attrs))
                self._body_explicit = True
                self._stack = [self._doc, self._html, self._body]  # type: ignore[list-item]
                self._after_head = False
                return pos

        if not self._fragment and self._current_parent() is self._head and name not in self._head_content_tags:
            self._stack = [self._doc, self._html, self._body]  # type: ignore[list-item]
            self._after_head = False

        if not self._fragment and self._current_parent() is self._html and self._after_head and name != "body":
            self._stack = [self._doc, self._html, self._body]  # type: ignore[list-item]
            self._after_head = False

        if self._blocks_frameset(name, attrs):
            self._frameset_blocked = True

        if name == "frameset" and self._accept_frameset():
            return pos

        if self._frameset_seen and not self._body_explicit:
            if name == "noframes":
                return self._parse_raw_literal_text("noframes", pos, end)
            return pos

        if name in self._drop_content_tags:
            return self._skip_rawtext(name, pos, end)
        if name in self._drop_subtree_tags:
            return self._skip_subtree(name, pos, end)
        if name in self._rawtext_as_text_tags:
            return self._parse_rawtext_as_text(name, pos, end)
        if name in self._rcdata_tags:
            return self._parse_rcdata_element(name, attrs, self_closing, pos, end)
        if name in self._plaintext_tags:
            return self._parse_plaintext_as_text(pos, end)

        parent: Node
        if name in self._head_content_tags and not self._fragment and self._head is not None and not self._body_has_content():
            parent = self._head
        else:
            self._repair_stack_for_start(name)
            parent = self._current_parent()

        if name in self._table_section_tags or name in self._table_cell_tags or name == "tr":
            self._repair_table_for_start(name)
            parent = self._current_parent()
            parent_name = getattr(parent, "name", None)
            if name in self._table_section_tags:
                if parent_name != "table":
                    return pos
            elif name == "tr":
                if parent_name not in self._table_section_tags:
                    return pos
            elif parent_name != "tr":
                return pos

        skip_initial_lf = name in self._pre_linefeed_ignoring_tags
        if name not in self._allowed_tags:
            return self._skip_initial_linefeed(pos, end) if skip_initial_lf else pos

        self._insert_allowed_element(name, attrs, self_closing, parent)
        if skip_initial_lf:
            pos = self._skip_initial_linefeed(pos, end)
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
        if name not in self._void_elements and not self_closing:
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
            children = self._body.children
        else:
            children = self._body.children
        if not children:
            return False
        return any(type(child) is not Text or bool(child.data) for child in children)

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

    def _close_until_before_boundary(self, name: str, boundaries: frozenset[str]) -> None:
        stack = self._stack
        for idx in range(len(stack) - 1, 0, -1):
            node_name = getattr(stack[idx], "name", None)
            if node_name == name:
                del stack[idx:]
                return
            if node_name in boundaries:
                return

    def _generate_implied_end_tags(self, exclude: str | None = None) -> None:
        stack = self._stack
        while len(stack) > 1:
            name = getattr(stack[-1], "name", None)
            if name in self._implied_end_tags and name != exclude:
                stack.pop()
                continue
            break

    def _repair_stack_for_start(self, name: str) -> None:
        if name in self._p_closing_start_tags and self._find_open_index("p") is not None:
            self._close_until("p")

        if name == "li":
            self._close_until_before_boundary("li", self._list_item_scope_boundaries)
            return

        if name in {"dd", "dt"}:
            self._close_until_before_boundary("dd", self._definition_scope_boundaries)
            self._close_until_before_boundary("dt", self._definition_scope_boundaries)
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
        if name in self._table_section_tags:
            self._close_table_cell()
            self._close_until_before_boundary("tr", _TABLE_CONTEXT_BOUNDARIES)
            for section in self._table_section_tags:
                self._close_until_before_boundary(section, _TABLE_CONTEXT_BOUNDARIES)
            if getattr(self._current_parent(), "name", None) != "table":
                table_idx = self._find_open_index("table")
                if table_idx is not None:
                    del self._stack[table_idx + 1 :]
            return

        if name == "tr":
            self._close_table_cell()
            self._close_until_before_boundary("tr", _TABLE_CONTEXT_BOUNDARIES)
            parent_name = getattr(self._current_parent(), "name", None)
            if parent_name == "table":
                self._insert_allowed_element("tbody", {}, False, self._current_parent())
            elif parent_name not in self._table_section_tags:
                table_idx = self._find_open_index("table")
                if table_idx is not None:
                    del self._stack[table_idx + 1 :]
                    self._insert_allowed_element("tbody", {}, False, self._current_parent())
            return

        if name in self._table_cell_tags:
            self._close_table_cell()
            parent_name = getattr(self._current_parent(), "name", None)
            if parent_name == "tr":
                return
            if parent_name == "table":
                self._insert_allowed_element("tbody", {}, False, self._current_parent())
                self._insert_allowed_element("tr", {}, False, self._current_parent())
                return
            if parent_name in self._table_section_tags:
                self._insert_allowed_element("tr", {}, False, self._current_parent())
                return
            table_idx = self._find_open_index("table")
            if table_idx is not None:
                del self._stack[table_idx + 1 :]
                self._insert_allowed_element("tbody", {}, False, self._current_parent())
                self._insert_allowed_element("tr", {}, False, self._current_parent())

    def _close_table_cell(self) -> None:
        stack = self._stack
        for idx in range(len(stack) - 1, 0, -1):
            name = getattr(stack[idx], "name", None)
            if name == "table":
                return
            if name in self._table_cell_tags:
                del stack[idx:]
                return

    def _foster_parent_for(self, parent: Node, *, for_tag: str | None = None) -> tuple[Node, int] | None:
        if getattr(parent, "name", None) not in self._table_foster_targets:
            return None
        if for_tag is not None and for_tag in self._table_allowed_children:
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
        if "\r" in raw_text:
            raw_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
        text = (
            raw_text
            if raw_text.isascii() or not self._strip_invisible_unicode
            else _strip_invisible_unicode(raw_text)
        )
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
        text = self._clean_text(raw_text, replace_null=True)
        if name not in self._allowed_tags:
            if text:
                self._append(self._current_parent(), Text(text))
            return pos

        parent: Node
        current_parent = self._current_parent()
        if (
            name == "title"
            and not self._fragment
            and self._head is not None
            and (current_parent is self._head or (not self._body_explicit and not self._body_has_content()))
        ):
            parent = self._head
        else:
            self._repair_stack_for_start(name)
            parent = current_parent
        node = self._insert_allowed_element(name, attrs, False if name in self._rcdata_tags else self_closing, parent)
        if text:
            self._append(node, Text(text))
        if self._stack and self._stack[-1] is node:
            self._stack.pop()
        return pos

    def _parse_plaintext_as_text(self, pos: int, end: int) -> int:
        self._append_raw_literal_text(self._html_input[pos:end])
        return end

    def _parse_raw_literal_text(self, name: str, pos: int, end: int) -> int:
        text, pos = self._consume_until_end_tag(name, pos, end)
        self._append_raw_literal_text(text)
        return pos

    def _append_raw_literal_text(self, text: str) -> None:
        if "\r" in text:
            text = text.replace("\r\n", "\n").replace("\r", "\n")
        if "\0" in text:
            text = text.replace("\0", "\ufffd")
        if self._strip_invisible_unicode and text and not text.isascii():
            text = _strip_invisible_unicode(text)
        if text:
            parent = self._html if self._frameset_seen and not self._body_explicit and self._html is not None else self._current_parent()
            foster = self._foster_parent_for(parent) if text.strip(_SPACE) else None
            if foster is None:
                self._append(parent, Text(text))
            else:
                foster_parent, position = foster
                self._insert_at(foster_parent, position, Text(text))

    def _skip_initial_linefeed(self, pos: int, end: int) -> int:
        html = self._html_input
        if pos >= end:
            return pos
        if html[pos] == "\n":
            return pos + 1
        if html[pos] == "\r":
            return pos + 2 if pos + 1 < end and html[pos + 1] == "\n" else pos + 1
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
            if self._strip_invisible_unicode and value is not None and not value.isascii():
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
            self._dropped_to_eof = True
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
                self._dropped_to_eof = True
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
        if depth:
            self._dropped_to_eof = True
        return pos

    def _finish_document_shell(self) -> None:
        if self._fragment or (not self._dropped_to_eof and not self._frameset_seen):
            return
        html = self._html
        head = self._head
        body = self._body if isinstance(self._body, Element) else None
        if html is None or head is None or body is None:
            return
        if self._frameset_seen:
            if self._node_is_empty(body) and not self._body_explicit:
                self._remove_child(html, body)
            return
        if not self._node_is_empty(body) or self._body_explicit:
            return

        if self._explicit_head or not self._node_is_empty(head):
            self._remove_child(html, body)
            return

        if self._explicit_html:
            self._remove_child(html, head)
            self._remove_child(html, body)
            return

        self._remove_child(self._doc, html)

    def _node_is_empty(self, node: Node) -> bool:
        children = node.children
        if not children:
            return True
        return all(type(child) is Text and not child.data for child in children)

    def _remove_child(self, parent: Node, child: Node) -> None:
        children = parent.children
        if children is None:
            return
        try:
            children.remove(child)
        except ValueError:
            return
        child.parent = None

    def _accept_frameset(self) -> bool:
        if self._fragment or self._frameset_blocked or self._body_explicit or not isinstance(self._body, Element):
            return False
        if not self._body_allows_frameset(self._body):
            return False
        children = self._body.children
        if children is not None:
            children.clear()
        self._frameset_seen = True
        self._stack = [self._doc, self._html]  # type: ignore[list-item]
        self._after_head = False
        return True

    def _blocks_frameset(self, name: str, attrs: dict[str, str | None]) -> bool:
        if self._fragment or self._frameset_seen or name not in self._frameset_blocking_start_tags:
            return False
        if name == "input":
            input_type = attrs.get("type")
            return not (isinstance(input_type, str) and input_type.lower() == "hidden")
        return True

    def _body_allows_frameset(self, node: Node) -> bool:
        children = node.children
        if not children:
            return True
        for child in children:
            if type(child) is Text:
                if (child.data or "").strip(_SPACE):
                    return False
                continue
            name = getattr(child, "name", None)
            if name not in self._frameset_body_ok_tags or not self._body_allows_frameset(child):
                return False
        return True

    def _append_frameset_text(self, raw: str) -> None:
        if "\r" in raw:
            raw = raw.replace("\r\n", "\n").replace("\r", "\n")
        pos = 0
        end = len(raw)
        while pos < end and raw[pos] in _SPACE:
            pos += 1
        if pos and self._html is not None:
            self._append(self._html, Text(raw[:pos]))
