"""Specialized default-safe parser.

This engine targets the narrow ``JustHTML(str)`` default-safe path. It does not
reuse the tokenizer or treebuilder; scanning, tree construction, and default
sanitizer work are fused into one parser hot path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from justhtml.core.constants import (
    BUTTON_SCOPE_TERMINATORS,
    DEFINITION_SCOPE_TERMINATORS,
    FORMATTING_ELEMENTS,
    HEADING_ELEMENTS,
    IMPLIED_END_TAGS,
    LIST_ITEM_SCOPE_TERMINATORS,
    SPECIAL_ELEMENTS,
    TABLE_ALLOWED_CHILDREN,
    VOID_ELEMENTS,
)
from justhtml.core.entities import decode_entities_in_text
from justhtml.dom import Document, DocumentFragment, Element, Node, Template, Text
from justhtml.sanitizer import DEFAULT_DOCUMENT_POLICY, DEFAULT_POLICY, _strip_invisible_unicode
from justhtml.sanitizer.url import _URL_SINK_ATTRS, _sanitize_url_sink_value, _url_sink_kind_for_attr
from justhtml.tokenizer.tokens import Doctype
from justhtml.treebuilder.utils import doctype_error_and_quirks

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from justhtml.parser.context import FragmentContext
    from justhtml.sanitizer import SanitizationPolicy
    from justhtml.sanitizer.url import UrlPolicy, UrlRule
    from justhtml.sanitizer.url.spec import UrlSinkKind

_TAG_NAME_RE = re.compile(r"[A-Za-z][^\t\n\f />]*")
_DOCTYPE_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9:_-]*$")
_DOCTYPE_RE = re.compile(
    r"""\s*([^\s>]+)(?:\s*(PUBLIC|SYSTEM)\s*(?:(?:"([^"]*)"|'([^']*)')\s*(?:"([^"]*)"|'([^']*)')?)?)?""",
    re.IGNORECASE,
)
_SPACE = " \t\n\f\r"
_TAG_NAME_STOP = "\t\n\f />"
_ATTR_NAME_STOP = "\t\n\f />=\0\"'<"
_ATTR_VALUE_STOP = _SPACE + ">"
_TAG_END_NAME_STOP = _SPACE + "/>"
_DROP_CONTENT_TAGS = {"script", "style"}
_DROP_SUBTREE_TAGS = {"svg", "math"}
_RCDATA_TAGS = {"title", "textarea"}
_RAWTEXT_AS_TEXT_TAGS = {"iframe", "noembed", "noframes", "noscript", "xmp"}
_PLAINTEXT_TAGS = {"plaintext"}
_ACTIVE_FORMATTING_TAGS = FORMATTING_ELEMENTS
_PARSER_ONLY_NAMESPACE = "justhtml-parser-only"
_BUTTON_SCOPE_BOUNDARIES = frozenset({"button"})
_P_SCOPE_BOUNDARIES = frozenset(BUTTON_SCOPE_TERMINATORS)
_HEAD_CONTENT_TAGS = {
    "base",
    "basefont",
    "bgsound",
    "link",
    "meta",
    "noframes",
    "noscript",
    "script",
    "style",
    "template",
    "title",
}
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
    "listing",
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
    "dd",
    "dt",
    "li",
} | HEADING_ELEMENTS
_HEAD_NOSCRIPT_ALLOWED_START_TAGS = {"basefont", "bgsound", "link", "meta", "noframes", "style"}
_DEFINITION_SCOPE_BOUNDARIES = frozenset(DEFINITION_SCOPE_TERMINATORS)
_LIST_ITEM_SCOPE_BOUNDARIES = frozenset(LIST_ITEM_SCOPE_TERMINATORS)
_PRE_LINEFEED_IGNORING_TAGS = {"listing", "pre"}
_TABLE_CONTEXT_BOUNDARIES = frozenset({"table"})
_TABLE_SECTION_TAGS = {"tbody", "thead", "tfoot"}
_TABLE_CELL_TAGS = {"td", "th"}
_TABLE_FOSTER_TARGETS = {"table", "tbody", "tfoot", "thead", "tr"}
_TABLE_STRUCTURE_START_TAGS = {"caption", "col", "colgroup", "table", "tbody", "td", "tfoot", "th", "thead", "tr"}
_TEMPLATE_SCOPE_BOUNDARIES = frozenset({"template"})
_TEMPLATE_MODE_INITIAL = "template"
_TEMPLATE_MODE_BODY = "body"
_TEMPLATE_MODE_TABLE = "table"
_TEMPLATE_MODE_TABLE_BODY = "table_body"
_TEMPLATE_MODE_ROW = "row"
_TEMPLATE_MODE_CELL = "cell"
_TEMPLATE_MODE_COLGROUP = "colgroup"
_TEMPLATE_TABLE_CONTEXT_START_TAGS = {"caption", "col", "colgroup", "tbody", "td", "tfoot", "th", "thead", "tr"}
_TEMPLATE_TABLE_BODY_IGNORED_START_TAGS = {"caption", "col", "colgroup", "tbody", "tfoot", "thead", "table"}
_TEMPLATE_ROW_STRUCTURE_START_TAGS = {"caption", "col", "colgroup", "tbody", "tfoot", "thead", "tr", "table"}
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
_UNWRAP_CONSTRUCTION_SKIP_TAGS = {"col", "colgroup", "form", "menuitem"}


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
    active_formatting_tags: frozenset[str]
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
    special_elements: frozenset[str]
    strip_invisible_unicode: bool
    table_allowed_children: frozenset[str]
    table_cell_tags: frozenset[str]
    table_foster_targets: frozenset[str]
    table_section_tags: frozenset[str]
    tag_actions: Mapping[str, TagAction]
    void_elements: frozenset[str]


_ENGINE_PLAN_CACHE: dict[tuple[bool, bool], EnginePlan] = {}


@dataclass(frozen=True, slots=True)
class TagAction:
    """Compiled hot-path metadata for one start-tag name."""

    name: str
    allowed: bool
    allowed_attrs: frozenset[str]
    state_attrs: frozenset[str]
    url_attr_kinds: Mapping[str, UrlSinkKind]
    url_attr_rules: Mapping[str, UrlRule]
    scan_attrs: bool
    active_formatting: bool
    blocks_frameset: bool
    drop_content: bool
    drop_subtree: bool
    head_content: bool
    p_closing: bool
    plaintext: bool
    pre_linefeed: bool
    rawtext_as_text: bool
    rcdata: bool
    table_cell: bool
    table_section: bool
    void: bool


def _compile_tag_actions(
    *,
    allowed_tags: frozenset[str],
    allowed_global: Collection[str],
    allowed_by_tag: Mapping[str, frozenset[str]],
    url_rules: Mapping[tuple[str, str], UrlRule],
    rawtext_as_text_tags: set[str],
) -> dict[str, TagAction]:
    known_tags = set(allowed_tags)
    known_tags.update(_DROP_CONTENT_TAGS)
    known_tags.update(_DROP_SUBTREE_TAGS)
    known_tags.update(rawtext_as_text_tags)
    known_tags.update(_RCDATA_TAGS)
    known_tags.update(_PLAINTEXT_TAGS)
    known_tags.update(_ACTIVE_FORMATTING_TAGS)
    known_tags.update(_HEAD_CONTENT_TAGS)
    known_tags.update(_P_CLOSING_START_TAGS)
    known_tags.update(_PRE_LINEFEED_IGNORING_TAGS)
    known_tags.update(_TABLE_SECTION_TAGS)
    known_tags.update(_TABLE_CELL_TAGS)
    known_tags.update(_TABLE_STRUCTURE_START_TAGS)
    known_tags.update(_FRAMESET_BLOCKING_START_TAGS)
    known_tags.update(
        {"body", "button", "frameset", "head", "html", "image", "input", "optgroup", "option", "template", "tr"}
    )

    actions: dict[str, TagAction] = {}
    global_attrs = frozenset(allowed_global)
    for tag in known_tags:
        allowed = tag in allowed_tags
        allowed_attrs = allowed_by_tag.get(tag, global_attrs) if allowed else frozenset()
        if tag == "input":
            state_attrs = frozenset({"type"})
        elif tag == "option":
            state_attrs = frozenset({"selected"})
        else:
            state_attrs = frozenset()
        url_attr_kinds: dict[str, UrlSinkKind] = {}
        url_attr_rules: dict[str, UrlRule] = {}
        for attr in allowed_attrs.intersection(_URL_SINK_ATTRS):
            rule = url_rules.get((tag, attr))
            if rule is None:
                continue
            kind = _url_sink_kind_for_attr(tag=tag, attr=attr, attrs={attr: ""})
            if kind is None:
                continue
            url_attr_kinds[attr] = kind
            url_attr_rules[attr] = rule

        actions[tag] = TagAction(
            name=tag,
            allowed=allowed,
            allowed_attrs=allowed_attrs,
            state_attrs=state_attrs,
            url_attr_kinds=url_attr_kinds,
            url_attr_rules=url_attr_rules,
            scan_attrs=bool(allowed_attrs or state_attrs),
            active_formatting=tag in _ACTIVE_FORMATTING_TAGS,
            blocks_frameset=tag in _FRAMESET_BLOCKING_START_TAGS,
            drop_content=tag in _DROP_CONTENT_TAGS,
            drop_subtree=tag in _DROP_SUBTREE_TAGS,
            head_content=tag in _HEAD_CONTENT_TAGS,
            p_closing=tag in _P_CLOSING_START_TAGS,
            plaintext=tag in _PLAINTEXT_TAGS,
            pre_linefeed=tag in _PRE_LINEFEED_IGNORING_TAGS,
            rawtext_as_text=tag in rawtext_as_text_tags,
            rcdata=tag in _RCDATA_TAGS,
            table_cell=tag in _TABLE_CELL_TAGS,
            table_section=tag in _TABLE_SECTION_TAGS,
            void=tag in VOID_ELEMENTS,
        )
    return actions


def compile_default_engine_plan(*, fragment: bool, scripting_enabled: bool = True) -> EnginePlan:
    """Return the cached default-safe execution plan."""
    cache_key = (bool(fragment), bool(scripting_enabled))
    cached = _ENGINE_PLAN_CACHE.get(cache_key)
    if cached is not None:
        return cached

    policy = DEFAULT_POLICY if fragment else DEFAULT_DOCUMENT_POLICY
    allowed_attrs = policy.allowed_attributes
    allowed_global = allowed_attrs.get("*", ())
    allowed_by_tag = {
        str(tag).lower(): frozenset(allowed_global).union(attrs) for tag, attrs in allowed_attrs.items() if tag != "*"
    }
    rawtext_as_text_tags = _RAWTEXT_AS_TEXT_TAGS if scripting_enabled else _RAWTEXT_AS_TEXT_TAGS - {"noscript"}
    tag_actions = _compile_tag_actions(
        allowed_tags=policy.allowed_tags,
        allowed_global=allowed_global,
        allowed_by_tag=allowed_by_tag,
        url_rules=policy.url_policy.allow_rules,
        rawtext_as_text_tags=rawtext_as_text_tags,
    )
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
        active_formatting_tags=frozenset(_ACTIVE_FORMATTING_TAGS),
        frameset_body_ok_tags=frozenset(_FRAMESET_BODY_OK_TAGS),
        frameset_blocking_start_tags=frozenset(_FRAMESET_BLOCKING_START_TAGS),
        head_content_tags=frozenset(_HEAD_CONTENT_TAGS),
        implied_end_tags=frozenset(IMPLIED_END_TAGS),
        list_item_scope_boundaries=frozenset(_LIST_ITEM_SCOPE_BOUNDARIES),
        p_closing_start_tags=frozenset(_P_CLOSING_START_TAGS),
        pre_linefeed_ignoring_tags=frozenset(_PRE_LINEFEED_IGNORING_TAGS),
        rawtext_as_text_tags=frozenset(rawtext_as_text_tags),
        rcdata_tags=frozenset(_RCDATA_TAGS),
        plaintext_tags=frozenset(_PLAINTEXT_TAGS),
        special_elements=frozenset(SPECIAL_ELEMENTS),
        strip_invisible_unicode=policy.strip_invisible_unicode,
        table_allowed_children=frozenset(TABLE_ALLOWED_CHILDREN),
        table_cell_tags=frozenset(_TABLE_CELL_TAGS),
        table_foster_targets=frozenset(_TABLE_FOSTER_TARGETS),
        table_section_tags=frozenset(_TABLE_SECTION_TAGS),
        tag_actions=tag_actions,
        void_elements=VOID_ELEMENTS,
    )
    _ENGINE_PLAN_CACHE[cache_key] = plan
    return plan


@dataclass(slots=True)
class _FormattingEntry:
    name: str
    attrs: dict[str, str | None]
    node: Element
    signature: tuple[tuple[str, str], ...]


class DefaultSafeEngine:
    __slots__ = (
        "_active_formatting",
        "_active_formatting_dirty",
        "_active_formatting_tags",
        "_after_head",
        "_allowed_tags",
        "_body",
        "_body_explicit",
        "_body_mode_seen",
        "_definition_scope_boundaries",
        "_doc",
        "_doctype_seen",
        "_drop_content_tags",
        "_drop_doctype",
        "_drop_subtree_tags",
        "_dropped_to_eof",
        "_explicit_head",
        "_explicit_html",
        "_foster_next_table_whitespace",
        "_fragment",
        "_fragment_context_name",
        "_fragment_context_namespace",
        "_fragment_context_node",
        "_frameset_blocked",
        "_frameset_blocking_start_tags",
        "_frameset_body_ok_tags",
        "_frameset_seen",
        "_has_selectedcontent",
        "_head",
        "_head_content_tags",
        "_html",
        "_html_input",
        "_ignore_lf",
        "_implied_end_tags",
        "_in_colgroup",
        "_in_head_noscript",
        "_initial_mode_done",
        "_keep_empty_shell_on_eof",
        "_length",
        "_list_item_scope_boundaries",
        "_lower_input",
        "_nodes_to_unwrap",
        "_p_closing_start_tags",
        "_parser_only_template_depth",
        "_plaintext_tags",
        "_plan",
        "_pre_linefeed_ignoring_tags",
        "_quirks_mode",
        "_rawtext_as_text_tags",
        "_rcdata_tags",
        "_skip_escaped_comment_space",
        "_special_elements",
        "_stack",
        "_strip_invisible_unicode",
        "_table_allowed_children",
        "_table_cell_tags",
        "_table_foster_targets",
        "_table_section_tags",
        "_tag_actions",
        "_template_active_formatting_lengths",
        "_template_modes",
        "_url_policy",
        "_void_elements",
    )

    def __init__(
        self,
        html: str,
        *,
        fragment: bool,
        fragment_context: FragmentContext | None = None,
        scripting_enabled: bool = True,
        plan: EnginePlan | None = None,
    ) -> None:
        self._html_input = html
        self._length = len(html)
        self._lower_input = html.lower()
        self._fragment = bool(fragment)
        self._fragment_context_name = (
            fragment_context.tag_name.lower() if fragment_context is not None and fragment_context.tag_name else None
        )
        self._fragment_context_namespace = (
            fragment_context.namespace.lower() if fragment_context is not None and fragment_context.namespace else None
        )
        self._fragment_context_node: Element | None = None
        self._plan = (
            plan
            if plan is not None
            else compile_default_engine_plan(fragment=fragment, scripting_enabled=scripting_enabled)
        )
        self._active_formatting_tags = self._plan.active_formatting_tags
        self._allowed_tags = self._plan.allowed_tags
        self._url_policy = self._plan.url_policy
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
        self._parser_only_template_depth = 0
        self._plaintext_tags = self._plan.plaintext_tags
        self._pre_linefeed_ignoring_tags = self._plan.pre_linefeed_ignoring_tags
        self._rawtext_as_text_tags = self._plan.rawtext_as_text_tags
        self._rcdata_tags = self._plan.rcdata_tags
        self._special_elements = self._plan.special_elements
        self._strip_invisible_unicode = self._plan.strip_invisible_unicode
        self._table_allowed_children = self._plan.table_allowed_children
        self._table_cell_tags = self._plan.table_cell_tags
        self._table_foster_targets = self._plan.table_foster_targets
        self._table_section_tags = self._plan.table_section_tags
        self._tag_actions = self._plan.tag_actions
        self._void_elements = self._plan.void_elements
        self._doc: Document | DocumentFragment
        self._after_head = False
        self._dropped_to_eof = False
        self._doctype_seen = False
        self._explicit_head = False
        self._explicit_html = False
        self._foster_next_table_whitespace = False
        self._frameset_blocked = False
        self._frameset_seen = False
        self._has_selectedcontent = False
        self._ignore_lf = False
        self._in_colgroup = False
        self._in_head_noscript = False
        self._initial_mode_done = bool(fragment)
        self._keep_empty_shell_on_eof = False
        self._skip_escaped_comment_space = False
        self._quirks_mode = "no-quirks"
        self._body_explicit = False
        self._body_mode_seen = False
        self._html: Element | None = None
        self._head: Element | None = None
        self._body: Element | DocumentFragment
        self._active_formatting: list[_FormattingEntry] = []
        self._active_formatting_dirty = False
        self._nodes_to_unwrap: list[Element] = []
        self._template_active_formatting_lengths: list[int] = []
        self._template_modes: list[str] = []
        self._stack: list[Node] = []

    def parse(self) -> Document | DocumentFragment:
        if self._fragment:
            root = DocumentFragment()
            self._doc = root
            context_name = self._fragment_context_name
            html_context = self._fragment_context_namespace in {None, "html"}
            if html_context and context_name in self._rcdata_tags:
                self._body = root
                self._stack = [root]
                text = self._clean_text(self._html_input, replace_null=True)
                if context_name == "textarea" and text.startswith("\n"):
                    text = text[1:]
                if text:
                    self._append(root, Text(text))
                return root

            if html_context and context_name in self._plaintext_tags:
                self._body = root
                self._stack = [root]
                self._append_raw_literal_text(self._html_input)
                return root

            if html_context and (
                context_name in self._drop_content_tags or context_name in self._rawtext_as_text_tags
            ):
                self._body = root
                self._stack = [root]
                self._append_raw_literal_text(self._html_input)
                return root

            if context_name and context_name != "div":
                context = Element(context_name, {}, self._fragment_context_namespace or "html")
                self._append(root, context)
                self._fragment_context_node = context
                self._body = context
                self._stack = [root, context]
            else:
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
        self._project_selectedcontent()
        self._unwrap_recorded_nodes()
        self._finish_fragment_context()
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

    def _append_text_boundary(self, parent: Node) -> None:
        children = parent.children
        if children is None:
            return
        node = Text("")
        node.parent = parent
        children.append(node)

    def _finish_fragment_context(self) -> None:
        context = self._fragment_context_node
        if context is None:
            return
        root = self._doc
        children = root.children
        if children is None:
            return
        try:
            index = children.index(context)
        except ValueError:
            return
        replacement = list(context.children or ())
        for child in replacement:
            child.parent = root
        children[index : index + 1] = replacement
        context.children = []
        context.parent = None

    def _current_parent(self) -> Node:
        stack = self._stack
        current = stack[-1]
        if current.namespace == _PARSER_ONLY_NAMESPACE:
            idx = len(stack) - 2
            while idx > 0:
                current = stack[idx]
                if current.namespace != _PARSER_ONLY_NAMESPACE:
                    break
                idx -= 1
        if type(current) is Template and current.template_content is not None:
            return current.template_content
        return current  # type: ignore[return-value]

    def _push_parser_only_element(self, name: str) -> None:
        self._stack.append(Element(name, {}, _PARSER_ONLY_NAMESPACE))

    def _open_parser_only_template_index(self) -> int | None:
        stack = self._stack
        for idx in range(len(stack) - 1, 0, -1):
            node = stack[idx]
            if node.name == "template" and node.namespace == _PARSER_ONLY_NAMESPACE:
                return idx
        return None

    def _has_open_parser_only_template(self) -> bool:
        return self._parser_only_template_depth > 0

    def _current_template_mode(self) -> str | None:
        modes = self._template_modes
        return modes[-1] if modes else None

    def _start_parser_only_template(self) -> None:
        if (
            not self._fragment
            and self._head is not None
            and not self._body_explicit
            and not self._body_mode_seen
            and not self._body_has_content()
            and not self._has_open_parser_only_template()
        ):
            self._stack = [self._doc, self._html, self._head]  # type: ignore[list-item]
            self._after_head = False
            self._explicit_head = True
        else:
            self._repair_stack_for_start("template")
        self._push_parser_only_element("template")
        self._parser_only_template_depth += 1
        self._template_modes.append(_TEMPLATE_MODE_INITIAL)
        self._template_active_formatting_lengths.append(len(self._active_formatting))

    def _close_parser_only_template(self) -> bool:
        idx = self._open_parser_only_template_index()
        if idx is None:
            return False
        self._mark_active_formatting_dirty()
        del self._stack[idx:]
        self._parser_only_template_depth -= 1
        if self._template_modes:
            self._template_modes.pop()
        if self._template_active_formatting_lengths:
            active_len = self._template_active_formatting_lengths.pop()
            del self._active_formatting[active_len:]
            self._refresh_active_formatting_dirty()
        return True

    def _mark_initial_content(self) -> None:
        if not self._fragment and not self._initial_mode_done:
            self._quirks_mode = "quirks"
            self._initial_mode_done = True

    def _parse_range(self, pos: int, end: int) -> int:
        html = self._html_input
        while pos < end:
            if self._skip_escaped_comment_space:
                self._skip_escaped_comment_space = False
                if pos < end and html[pos] in _SPACE and html.startswith("-->", pos + 1):
                    pos += 1
                    if pos >= end:
                        return end
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
                    close = self._find_comment_end(pos + 1, end)
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

    def _find_comment_end(self, pos: int, end: int) -> int:
        html = self._html_input
        while True:
            close = html.find("--", pos, end)
            if close == -1:
                return -1
            suffix = close + 2
            if suffix < end and html[suffix] == ">":
                return close
            if suffix + 1 < end and html[suffix] == "!" and html[suffix + 1] == ">":
                return close + 1
            if suffix + 1 < end and html[suffix] == "-" and html[suffix + 1] == ">":
                return close + 1
            pos = suffix

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
        raw_is_space: bool | None = None
        foster_table_whitespace = False
        if self._foster_next_table_whitespace:
            foster_table_whitespace = True
            self._foster_next_table_whitespace = False
        if not self._fragment and not self._initial_mode_done:
            raw_is_space = raw.strip(_SPACE) == ""
            if not raw_is_space:
                self._mark_initial_content()
        if not self._fragment and self._frameset_seen and not self._body_explicit:
            self._append_frameset_text(raw)
            return
        if self._fragment and self._frameset_seen:
            return
        if self._fragment_context_name == "colgroup":
            return
        if self._parser_only_template_depth and self._template_modes[-1] == _TEMPLATE_MODE_COLGROUP:
            return
        parent = self._current_parent()
        parent_name = getattr(parent, "name", None)
        if self._active_formatting_dirty and parent_name not in self._table_cell_tags and parent_name != "caption":
            self._reconstruct_active_formatting()
            parent = self._current_parent()
        if (
            not self._fragment
            and self._head is not None
            and parent is self._head
            and not self._parser_only_template_depth
            and self._html is not None
        ):
            if raw_is_space is None:
                raw_is_space = raw.strip(_SPACE) == ""
            if not raw_is_space:
                self._stack = [self._doc, self._html, self._body]  # type: ignore[list-item]
                self._after_head = False
                self._body_mode_seen = True
                parent = self._body
        if not self._fragment and parent is self._html and self._after_head:
            if raw_is_space is None:
                raw_is_space = raw.strip(_SPACE) == ""
            if not raw_is_space:
                self._stack = [self._doc, self._html, self._body]  # type: ignore[list-item]
                self._after_head = False
                self._body_mode_seen = True
                parent = self._body
        if (
            not self._fragment
            and parent is self._body
            and not self._body_explicit
            and not self._body_mode_seen
            and not self._body_has_content()
        ):
            if raw_is_space is None:
                raw_is_space = raw.strip(_SPACE) == ""
            if raw_is_space:
                return
            raw = raw.lstrip(_SPACE)
        if self._in_colgroup and getattr(parent, "name", None) == "table":
            stripped = raw.lstrip(_SPACE)
            leading_len = len(raw) - len(stripped)
            if 0 < leading_len < len(raw):
                self._append(parent, Text(self._clean_text(raw[:leading_len])))
                raw = stripped
        text = self._clean_text(raw)
        if self._ignore_lf:
            self._ignore_lf = False
            text = text.removeprefix("\n")
        text_is_space: bool | None = None
        if self._in_head_noscript:
            text_is_space = text.strip(_SPACE) == "" if text else True
            if not text_is_space:
                self._leave_head_noscript_to_body()
                parent = self._body
        if text:
            if not self._fragment and parent is self._html and self._after_head:
                if text_is_space is None:
                    text_is_space = text.strip(_SPACE) == ""
                if text_is_space:
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
            foster = None
            if parent.name in self._table_foster_targets:
                is_table_space = text_is_space if text_is_space is not None else text.strip(_SPACE) == ""
                if not is_table_space or foster_table_whitespace:
                    foster = self._foster_parent_for(parent)
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
            and not self._initial_mode_done
            and not self._explicit_html
            and not self._body_explicit
            and not self._frameset_blocked
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
                doctype = Doctype(name, public_id, system_id, force_quirks=name is None)
                self._quirks_mode = doctype_error_and_quirks(doctype)[1]
                self._prepend_doctype(doctype)
            else:
                doctype = Doctype(None, force_quirks=True)
                self._quirks_mode = doctype_error_and_quirks(doctype)[1]
                self._prepend_doctype(doctype)
        return end if gt == -1 else gt + 1

    def _prepend_doctype(self, doctype: Doctype) -> None:
        children = self._doc.children
        if children is None:
            return
        node = Node("!doctype", data=doctype)
        children.insert(0, node)
        node.parent = self._doc
        self._doctype_seen = True
        self._initial_mode_done = True

    def _parse_end_tag(self, pos: int, end: int) -> int:
        html = self._html_input
        if pos >= end:
            self._append_text("</")
            return end
        ch = html[pos]
        if not (("a" <= ch <= "z") or ("A" <= ch <= "Z")):
            if pos >= end:
                self._append_text("</")
                return end
            gt = html.find(">", pos, end)
            return end if gt == -1 else gt + 1
        name_start = pos
        pos += 1
        while pos < end and html[pos] not in _TAG_NAME_STOP:
            pos += 1
        name = html[name_start:pos]
        if not name.islower():
            name = name.lower()
        if not self._initial_mode_done:
            self._mark_initial_content()
        action = self._tag_actions.get(name)
        gt = html.find(">", pos, end)
        pos = end if gt == -1 else gt + 1

        if not self._fragment and name in {"html", "body"}:
            self._in_colgroup = False
            parent_name = getattr(self._current_parent(), "name", None)
            if self._find_open_index("table") is not None and parent_name not in {"body", "html"}:
                return pos
            if self._frameset_seen and not self._body_explicit:
                self._stack = [self._doc, self._html]  # type: ignore[list-item]
            else:
                self._stack = [self._doc, self._html, self._body]  # type: ignore[list-item]
            self._after_head = False
            self._body_mode_seen = True
            return pos
        if not self._fragment and name == "head":
            self._in_colgroup = False
            self._stack = [self._doc, self._html]  # type: ignore[list-item]
            self._after_head = True
            return pos
        if name == "colgroup":
            self._in_colgroup = False
            return pos
        if name == "table":
            self._in_colgroup = False
        if name == "template":
            self._close_parser_only_template()
            return pos
        if self._parser_only_template_depth and self._handle_template_mode_end(name):
            return pos
        if self._frameset_seen and not self._body_explicit:
            return pos
        if (
            not self._fragment
            and self._head is not None
            and self._stack[-1] is self._head
            and name not in {"br", "body", "head", "html", "template"}
        ):
            if self._html is not None:
                self._stack = [self._doc, self._html, self._body]  # type: ignore[list-item]
                self._after_head = False
                self._body_mode_seen = True
            return pos
        if (
            not self._fragment
            and name == "br"
            and self._head is not None
            and self._stack[-1] is self._head
            and self._html is not None
        ):
            self._stack = [self._doc, self._html, self._body]  # type: ignore[list-item]
            self._after_head = False
            self._body_mode_seen = True
        if self._in_head_noscript:
            if name == "noscript":
                self._in_head_noscript = False
                return pos
            if name != "br":
                return pos
            self._leave_head_noscript_to_body()
        if name == "br":
            self._insert_allowed_element("br", {}, False, self._current_parent())
            return pos
        stack = self._stack
        if name in HEADING_ELEMENTS:
            idx = self._find_open_heading_index()
            if idx is None:
                return pos
            self._mark_active_formatting_dirty()
            del stack[idx:]
            return pos
        if action is not None and action.active_formatting:
            self._adoption_agency(name)
            return pos

        if (
            not self._parser_only_template_depth
            and len(stack) > 1
            and stack[-1].name == name
            and not (self._fragment_context_node is not None and stack[-1] is self._fragment_context_node)
        ):
            self._mark_active_formatting_dirty()
            stack.pop()
            return pos

        if name == "p":
            idx = self._find_open_index_before_boundary("p", _P_SCOPE_BOUNDARIES)
        elif name == "li":
            idx = self._find_open_index_before_boundary("li", self._list_item_scope_boundaries)
        elif name in {"dd", "dt"}:
            idx = self._find_open_index_before_boundary(name, self._definition_scope_boundaries)
        elif self._parser_only_template_depth:
            idx = self._find_open_index_in_current_scope(name)
        elif name not in self._special_elements and (action is None or not action.p_closing):
            idx = self._find_open_index_before_boundary(name, _BUTTON_SCOPE_BOUNDARIES | _TABLE_CONTEXT_BOUNDARIES)
        else:
            idx = self._find_open_index_before_boundary(name, _TABLE_CONTEXT_BOUNDARIES)
        if idx is None:
            if name == "p":
                if (
                    not self._fragment
                    and not self._body_explicit
                    and not self._body_mode_seen
                    and not self._body_has_content()
                ):
                    return pos
                self._insert_allowed_element("p", {}, False, self._current_parent())
                self._close_until_before_boundary("p", _P_SCOPE_BOUNDARIES)
            elif name in self._p_closing_start_tags:
                self._close_until_before_boundary("p", _P_SCOPE_BOUNDARIES)
            return pos
        if self._fragment_context_node is not None and stack[idx] is self._fragment_context_node:
            return pos
        if name in self._implied_end_tags:
            self._generate_implied_end_tags(name)
        self._mark_active_formatting_dirty()
        del stack[idx:]
        return pos

    def _parse_start_tag(self, pos: int, end: int) -> int:
        html = self._html_input
        name_start = pos
        if pos >= end:
            self._append_text("<")
            return pos
        pos += 1
        while pos < end and html[pos] not in _TAG_NAME_STOP:
            pos += 1
        raw_name = html[name_start:pos]
        if not raw_name:
            self._append_text("<")
            return pos
        name = raw_name if raw_name.islower() else raw_name.lower()
        if name == "image":
            name = "img"
        if not self._initial_mode_done:
            self._mark_initial_content()

        action = self._tag_actions.get(name)
        attrs, self_closing, pos, tag_closed = self._parse_attrs_for_action(action, pos, end)
        if not tag_closed:
            return pos
        in_parser_only_template = self._parser_only_template_depth > 0
        if not in_parser_only_template:
            if name == "colgroup":
                self._in_colgroup = True
            elif self._in_colgroup and name not in {"col", "template"}:
                self._in_colgroup = False
        if self._in_head_noscript:
            if name in {"head", "noscript"}:
                return pos
            if name not in _HEAD_NOSCRIPT_ALLOWED_START_TAGS and name != "html":
                self._leave_head_noscript_to_body()
        if not self._fragment and not in_parser_only_template:
            current_top = self._stack[-1]
            if name == "html":
                self._in_colgroup = False
                self._explicit_html = True
                if self._html is not None:
                    self._html.attrs.update(attrs)
                return pos
            if name == "head":
                self._in_colgroup = False
                self._explicit_head = True
                if self._head is not None:
                    self._stack = [self._doc, self._html, self._head]  # type: ignore[list-item]
                    self._after_head = False
                return pos
            if name == "body":
                self._in_colgroup = False
                if isinstance(self._body, Element):
                    self._body.attrs.update(attrs)
                self._body_explicit = True
                self._body_mode_seen = True
                self._stack = [self._doc, self._html, self._body]  # type: ignore[list-item]
                self._after_head = False
                return pos
            if not self._body_mode_seen and not (action.head_content if action is not None else False):
                self._body_mode_seen = True
        else:
            current_top = None

        if (
            not in_parser_only_template
            and not self._fragment
            and current_top is self._head
            and not (action.head_content if action is not None else False)
        ):
            self._stack = [self._doc, self._html, self._body]  # type: ignore[list-item]
            self._after_head = False
            self._body_mode_seen = True
            current_top = self._body

        if (
            not in_parser_only_template
            and not self._fragment
            and current_top is self._html
            and self._after_head
            and name not in {"body", "template"}
        ):
            if (
                action is not None
                and action.allowed
                and action.head_content
                and self._head is not None
                and not self._body_mode_seen
                and not self._body_has_content()
            ):
                self._stack = [self._doc, self._html, self._head]  # type: ignore[list-item]
                self._after_head = False
                current_top = self._head
            else:
                self._stack = [self._doc, self._html, self._body]  # type: ignore[list-item]
                self._after_head = False
                self._body_mode_seen = True
                current_top = self._body

        if not in_parser_only_template and self._blocks_frameset_action(action, attrs):
            self._frameset_blocked = True

        if name == "frameset" and self._accept_fragment_frameset():
            return pos

        if not in_parser_only_template and name == "frameset" and self._accept_frameset():
            return pos

        if self._frameset_seen and not self._body_explicit:
            if name == "noframes":
                return self._parse_raw_literal_text("noframes", pos, end)
            return pos

        if name == "frame":
            return pos

        if action is not None and action.drop_content:
            return self._skip_rawtext(name, pos, end)
        if action is not None and action.drop_subtree:
            return self._skip_subtree(name, pos, end)
        if action is not None and action.rawtext_as_text:
            return self._parse_rawtext_as_text(name, pos, end)
        if (
            name == "noscript"
            and not self._fragment
            and self._head is not None
            and not in_parser_only_template
            and self._stack[-1] is self._head
        ):
            self._in_head_noscript = True
            return pos
        if action is not None and action.rcdata:
            return self._parse_rcdata_element(name, attrs, self_closing, pos, end)
        if action is not None and action.plaintext:
            return self._parse_plaintext_as_text(pos, end)

        fragment_pos = self._handle_fragment_context_start(name, attrs, self_closing, pos)
        if fragment_pos is not None:
            return fragment_pos

        if name == "template" and name not in self._allowed_tags:
            self._start_parser_only_template()
            return pos

        if in_parser_only_template:
            template_pos = self._handle_template_mode_start(name, attrs, self_closing, pos)
            if template_pos is not None:
                return template_pos

        if name == "button" and name not in self._allowed_tags:
            if self._find_open_index_in_current_scope("button") is not None:
                self._close_until_before_boundary("button", frozenset({"template"}))
            self._insert_sanitized_element(name, attrs, self_closing, self._current_parent())
            return pos

        if action is not None and action.active_formatting:
            return self._parse_formatting_start(name, attrs, pos)

        parent: Node
        if (
            action is not None
            and action.allowed
            and action.head_content
            and not self._fragment
            and self._head is not None
            and not self._body_mode_seen
            and not self._body_has_content()
        ):
            parent = self._head
        else:
            self._repair_stack_for_start(name)
            parent = self._current_parent()

        if not in_parser_only_template and name == "table":
            table_idx = self._find_open_index("table")
            td_idx = self._find_open_index_before_boundary("td", _TABLE_CONTEXT_BOUNDARIES)
            th_idx = self._find_open_index_before_boundary("th", _TABLE_CONTEXT_BOUNDARIES)
            if table_idx is not None and td_idx is None and th_idx is None:
                self._mark_active_formatting_dirty()
                del self._stack[table_idx:]
                parent = self._current_parent()

        if not in_parser_only_template and (
            (action is not None and (action.table_section or action.table_cell))
            or name in {"caption", "col", "colgroup", "tr"}
        ):
            self._repair_table_for_start(name)
            parent = self._current_parent()
            parent_name = getattr(parent, "name", None)
            if name == "caption":
                if parent_name != "table":
                    return pos
            elif name in {"col", "colgroup"}:
                if parent_name != "table":
                    return pos
            elif action is not None and action.table_section:
                if parent_name != "table":
                    return pos
            elif name == "tr":
                if parent_name not in self._table_section_tags:
                    return pos
            elif parent_name != "tr":
                return pos

        if action is None or not action.allowed:
            if action is not None and action.pre_linefeed:
                self._ignore_lf = True
            if self._should_insert_unwrapped_element(name, action):
                if self._active_formatting_dirty:
                    self._reconstruct_active_formatting()
                    parent = self._current_parent()
                self._insert_sanitized_element(name, attrs, self_closing, parent)
            elif name == "menuitem" and self._active_formatting_dirty:
                self._reconstruct_active_formatting()
            return pos

        self._insert_allowed_element(name, attrs, self_closing, parent)
        if action.pre_linefeed:
            self._ignore_lf = True
        return pos

    def _insert_allowed_element(
        self,
        name: str,
        attrs: dict[str, str | None],
        self_closing: bool,
        parent: Node,
    ) -> Element:
        return self._insert_sanitized_element(name, attrs, self_closing, parent)

    def _insert_sanitized_element(
        self,
        name: str,
        attrs: dict[str, str | None],
        self_closing: bool,
        parent: Node,
    ) -> Element:
        node: Element
        if name == "template":
            node = Template(name, attrs, namespace="html")
        else:
            node = Element(name, attrs, "html")
        if name == "selectedcontent":
            self._has_selectedcontent = True
        is_void = name in self._void_elements
        node._self_closing = self_closing and is_void
        foster = (
            self._foster_parent_for(parent, for_tag=name)
            if parent.name in self._table_foster_targets and name not in self._table_allowed_children
            else None
        )
        if foster is None:
            self._append(parent, node)
        else:
            foster_parent, position = foster
            self._insert_at(foster_parent, position, node)
        if name not in self._allowed_tags:
            self._nodes_to_unwrap.append(node)
        if not is_void:
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
            if stack[idx].name == name:
                return idx
        return None

    def _find_open_index_before_boundary(self, name: str, boundaries: frozenset[str]) -> int | None:
        stack = self._stack
        for idx in range(len(stack) - 1, 0, -1):
            node_name = stack[idx].name
            if node_name == name:
                return idx
            if node_name in boundaries:
                return None
        return None

    def _find_open_index_in_current_scope(self, name: str) -> int | None:
        stack = self._stack
        for idx in range(len(stack) - 1, 0, -1):
            node = stack[idx]
            if node.name == name:
                return idx
            if node.name == "template" and node.namespace == _PARSER_ONLY_NAMESPACE:
                return None
        return None

    def _find_open_heading_index(self) -> int | None:
        stack = self._stack
        for idx in range(len(stack) - 1, 0, -1):
            node = stack[idx]
            if node.name in HEADING_ELEMENTS:
                return idx
            if node.name in _P_SCOPE_BOUNDARIES:
                return None
        return None

    def _set_current_template_mode(self, mode: str) -> None:
        if self._template_modes:
            self._template_modes[-1] = mode

    def _leave_head_noscript_to_body(self) -> None:
        self._in_head_noscript = False
        if self._fragment or self._html is None:
            return
        self._stack = [self._doc, self._html, self._body]  # type: ignore[list-item]
        self._after_head = False
        self._body_mode_seen = True

    def _close_to_fragment_context(self) -> bool:
        context = self._fragment_context_node
        if context is None:
            return False
        stack = self._stack
        try:
            idx = stack.index(context)
        except ValueError:
            return False
        if len(stack) > idx + 1:
            self._mark_active_formatting_dirty()
            del stack[idx + 1 :]
        return True

    def _allows_nested_fragment_table_start(self, name: str) -> bool:
        if name not in _TABLE_STRUCTURE_START_TAGS:
            return False
        parent_name = getattr(self._current_parent(), "name", None)
        if name == "table" and parent_name in self._table_cell_tags:
            return True
        context = self._fragment_context_node
        if context is None:
            return False
        try:
            context_idx = self._stack.index(context)
        except ValueError:
            return False
        return any(node.name == "table" for node in self._stack[context_idx + 1 :])

    def _handle_fragment_context_start(
        self,
        name: str,
        attrs: dict[str, str | None],
        self_closing: bool,
        pos: int,
    ) -> int | None:
        context_name = self._fragment_context_name
        if context_name is None:
            return None

        if context_name == "colgroup":
            return pos

        if context_name == "caption":
            return pos if name in _TABLE_STRUCTURE_START_TAGS and name != "table" else None

        if context_name == "table":
            if name in {"col", "colgroup", "table"}:
                return pos
            if name == "caption":
                if self._close_to_fragment_context():
                    self._insert_allowed_element(name, attrs, self_closing, self._current_parent())
                    return pos
            return None

        if context_name in self._table_section_tags:
            if self._allows_nested_fragment_table_start(name):
                return None
            if name == "tr":
                if self._close_to_fragment_context():
                    self._insert_allowed_element(name, attrs, self_closing, self._current_parent())
                    return pos
            if name in self._table_cell_tags:
                self._close_table_cell()
                if getattr(self._current_parent(), "name", None) != "tr" and self._close_to_fragment_context():
                    self._insert_allowed_element("tr", {}, False, self._current_parent())
                if getattr(self._current_parent(), "name", None) == "tr":
                    self._insert_allowed_element(name, attrs, self_closing, self._current_parent())
                return pos
            if name in _TABLE_STRUCTURE_START_TAGS:
                return pos
            return None

        if context_name == "tr":
            if self._allows_nested_fragment_table_start(name):
                return None
            if name in self._table_cell_tags:
                if self._close_to_fragment_context():
                    self._insert_allowed_element(name, attrs, self_closing, self._current_parent())
                    return pos
            if name in _TABLE_STRUCTURE_START_TAGS:
                return pos
        return None

    def _insert_template_mode_element(
        self,
        name: str,
        attrs: dict[str, str | None],
        self_closing: bool,
    ) -> Element | None:
        if name not in self._allowed_tags:
            if name in self._pre_linefeed_ignoring_tags:
                self._ignore_lf = True
            return None
        node = self._insert_allowed_element(name, attrs, self_closing, self._current_parent())
        if name in self._pre_linefeed_ignoring_tags:
            self._ignore_lf = True
        return node

    def _handle_template_mode_start(
        self,
        name: str,
        attrs: dict[str, str | None],
        self_closing: bool,
        pos: int,
    ) -> int | None:
        mode = self._current_template_mode()
        if mode is None:
            return None
        if name in {"html", "head", "body"}:
            return pos

        if mode == _TEMPLATE_MODE_COLGROUP:
            return pos

        if mode == _TEMPLATE_MODE_INITIAL:
            if name in {"tbody", "tfoot", "thead"}:
                self._set_current_template_mode(_TEMPLATE_MODE_TABLE_BODY)
                self._insert_template_mode_element(name, attrs, self_closing)
                return pos
            if name == "caption":
                self._set_current_template_mode(_TEMPLATE_MODE_TABLE)
                self._insert_template_mode_element(name, attrs, self_closing)
                return pos
            if name in {"col", "colgroup"}:
                self._set_current_template_mode(_TEMPLATE_MODE_COLGROUP)
                return pos
            if name == "tr":
                self._set_current_template_mode(_TEMPLATE_MODE_TABLE_BODY)
                return self._handle_template_table_body_start(name, attrs, self_closing, pos)
            if name in self._table_cell_tags:
                self._set_current_template_mode(_TEMPLATE_MODE_ROW)
                return self._handle_template_row_start(name, attrs, self_closing, pos)
            if name not in self._head_content_tags:
                self._set_current_template_mode(_TEMPLATE_MODE_BODY)
            return None

        if mode == _TEMPLATE_MODE_BODY:
            if name in _TEMPLATE_TABLE_CONTEXT_START_TAGS:
                return pos
            return None

        if mode == _TEMPLATE_MODE_TABLE:
            if name in {"tbody", "tfoot", "thead"}:
                self._set_current_template_mode(_TEMPLATE_MODE_TABLE_BODY)
                self._insert_template_mode_element(name, attrs, self_closing)
                return pos
            if name == "tr":
                self._set_current_template_mode(_TEMPLATE_MODE_TABLE_BODY)
                return self._handle_template_table_body_start(name, attrs, self_closing, pos)
            if name in self._table_cell_tags:
                self._set_current_template_mode(_TEMPLATE_MODE_TABLE_BODY)
                return self._handle_template_table_body_start(name, attrs, self_closing, pos)
            if name in {"col", "colgroup"}:
                self._set_current_template_mode(_TEMPLATE_MODE_COLGROUP)
                return pos
            return None

        if mode == _TEMPLATE_MODE_TABLE_BODY:
            return self._handle_template_table_body_start(name, attrs, self_closing, pos)
        if mode == _TEMPLATE_MODE_ROW:
            return self._handle_template_row_start(name, attrs, self_closing, pos)
        if mode == _TEMPLATE_MODE_CELL:
            if name in _TEMPLATE_TABLE_CONTEXT_START_TAGS:
                if self._close_template_cell():
                    return self._handle_template_mode_start(name, attrs, self_closing, pos)
                return pos
            return None
        return None

    def _handle_template_table_body_start(
        self,
        name: str,
        attrs: dict[str, str | None],
        self_closing: bool,
        pos: int,
    ) -> int | None:
        if name == "tr":
            self._set_current_template_mode(_TEMPLATE_MODE_ROW)
            self._insert_template_mode_element(name, attrs, self_closing)
            return pos
        if name in self._table_cell_tags:
            self._set_current_template_mode(_TEMPLATE_MODE_ROW)
            self._insert_template_mode_element("tr", {}, False)
            return self._handle_template_row_start(name, attrs, self_closing, pos)
        if name in _TEMPLATE_TABLE_BODY_IGNORED_START_TAGS:
            return pos
        return None

    def _handle_template_row_start(
        self,
        name: str,
        attrs: dict[str, str | None],
        self_closing: bool,
        pos: int,
    ) -> int | None:
        tr_index = self._find_open_index_before_boundary("tr", _TEMPLATE_SCOPE_BOUNDARIES)
        if name in self._table_cell_tags:
            self._set_current_template_mode(_TEMPLATE_MODE_CELL)
            self._insert_template_mode_element(name, attrs, self_closing)
            return pos
        if name in _TEMPLATE_ROW_STRUCTURE_START_TAGS:
            if tr_index is not None:
                self._close_until_before_boundary("tr", _TEMPLATE_SCOPE_BOUNDARIES)
                self._set_current_template_mode(_TEMPLATE_MODE_TABLE_BODY)
                return self._handle_template_mode_start(name, attrs, self_closing, pos)
            return pos
        if tr_index is not None:
            self._close_until_before_boundary("tr", _TEMPLATE_SCOPE_BOUNDARIES)
            self._set_current_template_mode(_TEMPLATE_MODE_TABLE_BODY)
        return None

    def _handle_template_mode_end(self, name: str) -> bool:
        mode = self._current_template_mode()
        if mode is None:
            return False
        if name in {"html", "head", "body"}:
            return True
        if mode == _TEMPLATE_MODE_CELL:
            if name in self._table_cell_tags:
                return self._close_template_cell()
            if name in {"table", "tbody", "tfoot", "thead", "tr"}:
                if self._find_open_index_before_boundary(name, _TEMPLATE_SCOPE_BOUNDARIES) is None:
                    return True
                self._close_template_cell()
                return False
        if mode == _TEMPLATE_MODE_ROW:
            if name == "tr":
                if self._close_template_row():
                    self._set_current_template_mode(_TEMPLATE_MODE_TABLE_BODY)
                return True
            if name in {"caption", "col", "colgroup", "td", "th"}:
                return True
        if mode == _TEMPLATE_MODE_TABLE_BODY:
            if name in self._table_section_tags:
                if self._close_until_before_boundary(name, _TEMPLATE_SCOPE_BOUNDARIES):
                    self._set_current_template_mode(_TEMPLATE_MODE_TABLE)
                return True
            if name in {"caption", "col", "colgroup", "td", "th", "tr"}:
                return True
        if mode == _TEMPLATE_MODE_COLGROUP:
            return name != "template"
        return False

    def _close_template_row(self) -> bool:
        return self._close_until_before_boundary("tr", _TEMPLATE_SCOPE_BOUNDARIES)

    def _close_template_cell(self) -> bool:
        stack = self._stack
        for idx in range(len(stack) - 1, 0, -1):
            node_name = stack[idx].name
            if node_name in self._table_cell_tags:
                self._mark_active_formatting_dirty()
                del stack[idx:]
                self._set_current_template_mode(_TEMPLATE_MODE_ROW)
                return True
            if node_name == "template":
                return False
        return False

    def _close_until(self, name: str) -> None:
        idx = self._find_open_index(name)
        if idx is not None:
            self._mark_active_formatting_dirty()
            del self._stack[idx:]

    def _close_until_before_boundary(self, name: str, boundaries: frozenset[str]) -> bool:
        stack = self._stack
        for idx in range(len(stack) - 1, 0, -1):
            node_name = getattr(stack[idx], "name", None)
            if node_name == name:
                if self._fragment_context_node is not None and stack[idx] is self._fragment_context_node:
                    return False
                self._mark_active_formatting_dirty()
                del stack[idx:]
                return True
            if node_name in boundaries:
                return False
        return False

    def _generate_implied_end_tags(self, exclude: str | None = None) -> None:
        stack = self._stack
        while len(stack) > 1:
            name = getattr(stack[-1], "name", None)
            if name in self._implied_end_tags and name != exclude:
                self._mark_active_formatting_dirty()
                stack.pop()
                continue
            break

    def _repair_stack_for_start(self, name: str) -> None:
        if (
            name in self._p_closing_start_tags
            and not (name == "table" and self._quirks_mode == "quirks")
            and self._find_open_index_before_boundary("p", _P_SCOPE_BOUNDARIES) is not None
        ):
            self._close_until_before_boundary("p", _P_SCOPE_BOUNDARIES)

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

        if name in HEADING_ELEMENTS and getattr(self._stack[-1], "name", None) in HEADING_ELEMENTS:
            self._mark_active_formatting_dirty()
            self._stack.pop()

    def _repair_table_for_start(self, name: str) -> None:
        table_idx = self._find_open_index("table")
        select_idx = self._find_open_index("select")
        if table_idx is not None and select_idx is not None and select_idx > table_idx:
            self._mark_active_formatting_dirty()
            del self._stack[select_idx:]

        if name in {"col", "colgroup"}:
            self._close_table_cell()
            self._close_until_before_boundary("tr", _TABLE_CONTEXT_BOUNDARIES)
            for section in self._table_section_tags:
                self._close_until_before_boundary(section, _TABLE_CONTEXT_BOUNDARIES)
            return

        if name == "caption":
            self._close_table_cell()
            self._close_until_before_boundary("tr", _TABLE_CONTEXT_BOUNDARIES)
            for section in self._table_section_tags:
                self._close_until_before_boundary(section, _TABLE_CONTEXT_BOUNDARIES)
            if getattr(self._current_parent(), "name", None) != "table":
                if table_idx is not None:
                    self._mark_active_formatting_dirty()
                    del self._stack[table_idx + 1 :]
            return

        if name in self._table_section_tags:
            self._close_table_cell()
            self._close_until_before_boundary("tr", _TABLE_CONTEXT_BOUNDARIES)
            for section in self._table_section_tags:
                self._close_until_before_boundary(section, _TABLE_CONTEXT_BOUNDARIES)
            if getattr(self._current_parent(), "name", None) != "table":
                if table_idx is not None:
                    self._mark_active_formatting_dirty()
                    del self._stack[table_idx + 1 :]
            return

        if name == "tr":
            self._close_table_cell()
            self._close_until_before_boundary("tr", _TABLE_CONTEXT_BOUNDARIES)
            self._close_stray_table_content_to_section()
            parent_name = getattr(self._current_parent(), "name", None)
            if parent_name == "table":
                self._insert_allowed_element("tbody", {}, False, self._current_parent())
            elif parent_name not in self._table_section_tags:
                if table_idx is not None:
                    self._mark_active_formatting_dirty()
                    del self._stack[table_idx + 1 :]
                    self._insert_allowed_element("tbody", {}, False, self._current_parent())
            return

        if name in self._table_cell_tags:
            self._close_table_cell()
            self._close_stray_table_content_to_section()
            tr_idx = self._find_open_index_before_boundary("tr", _TABLE_CONTEXT_BOUNDARIES)
            if tr_idx is not None and len(self._stack) > tr_idx + 1:
                self._mark_active_formatting_dirty()
                del self._stack[tr_idx + 1 :]
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
                self._mark_active_formatting_dirty()
                del self._stack[table_idx + 1 :]
                self._insert_allowed_element("tbody", {}, False, self._current_parent())
                self._insert_allowed_element("tr", {}, False, self._current_parent())

    def _close_stray_table_content_to_section(self) -> None:
        stack = self._stack
        for idx in range(len(stack) - 1, 0, -1):
            name = getattr(stack[idx], "name", None)
            if name in self._table_section_tags:
                if len(stack) > idx + 1:
                    self._mark_active_formatting_dirty()
                    del stack[idx + 1 :]
                return
            if name == "table" or name in self._table_cell_tags or name == "tr":
                return

    def _close_table_cell(self) -> None:
        stack = self._stack
        for idx in range(len(stack) - 1, 0, -1):
            name = getattr(stack[idx], "name", None)
            if name == "table":
                return
            if name in self._table_cell_tags:
                self._mark_active_formatting_dirty()
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
        if self._parser_only_template_depth:
            template_idx = self._open_parser_only_template_index()
            if template_idx is not None and table_idx < template_idx:
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
        close, next_pos = self._find_rawtext_end_tag(name, pos, end)
        if close is None:
            return html[pos:end], end
        return html[pos:close], next_pos

    def _find_rawtext_end_tag(self, name: str, pos: int, end: int) -> tuple[int | None, int]:
        html = self._html_input
        lower = self._lower_input
        needle = f"</{name}"
        needle_len = len(needle)
        search = pos
        while True:
            close = lower.find(needle, search, end)
            if close == -1:
                return None, end
            after_name = close + needle_len
            if after_name < end and html[after_name] not in _TAG_END_NAME_STOP:
                search = after_name
                continue
            tag_end = self._find_tag_end(after_name, end)
            if tag_end == -1:
                search = after_name
                continue
            next_pos = tag_end + 1
            return close, next_pos

    def _find_script_end_tag(self, pos: int, end: int) -> tuple[int | None, int]:
        lower = self._lower_input
        search = pos
        escaped = False
        double_escaped = False

        while True:
            close, next_pos = self._find_rawtext_end_tag("script", search, end)
            if close is None:
                return None, end
            if not escaped:
                comment_start = lower.find("<!--", search, close)
                if comment_start == -1 or close < comment_start:
                    return close, next_pos
                escaped = True
                search = comment_start + 4
                continue

            script_start = self._find_script_start_marker(search, close) if not double_escaped else -1
            comment_end = lower.find("-->", search, close)
            if (
                comment_end != -1
                and comment_end < close
                and (double_escaped or script_start == -1 or comment_end < script_start)
            ):
                escaped = False
                double_escaped = False
                search = comment_end + 3
                continue
            if script_start != -1 and script_start < close:
                double_escaped = True
                search = script_start + 7
                continue
            if double_escaped:
                double_escaped = False
                search = next_pos
                continue
            return close, next_pos

    def _find_script_start_marker(self, pos: int, end: int) -> int:
        html = self._html_input
        lower = self._lower_input
        search = pos
        needle = "<script"
        needle_len = len(needle)
        while True:
            start = lower.find(needle, search, end)
            if start == -1:
                return -1
            after_name = start + needle_len
            if after_name >= end or html[after_name] in _TAG_END_NAME_STOP:
                return start
            search = after_name

    def _find_tag_end(self, pos: int, end: int) -> int:
        html = self._html_input
        quote: str | None = None
        while pos < end:
            ch = html[pos]
            if quote is not None:
                if ch == quote:
                    quote = None
            elif ch == '"' or ch == "'":
                quote = ch
            elif ch == ">":
                return pos
            pos += 1
        return -1

    def _parse_rawtext_as_text(self, name: str, pos: int, end: int) -> int:
        raw_text, pos = self._consume_until_end_tag(name, pos, end)
        if not raw_text:
            return pos
        if "\r" in raw_text:
            raw_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
        text = (
            raw_text if raw_text.isascii() or not self._strip_invisible_unicode else _strip_invisible_unicode(raw_text)
        )
        if not text:
            return pos
        parent: Node
        if (
            not self._fragment
            and name in self._head_content_tags
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
        if name == "textarea" and text.startswith("\n"):
            text = text[1:]
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
            and (
                current_parent is self._head
                or (not self._body_explicit and not self._body_mode_seen and not self._body_has_content())
            )
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
        if self._find_open_index_before_boundary("p", _P_SCOPE_BOUNDARIES) is not None:
            self._close_until_before_boundary("p", _P_SCOPE_BOUNDARIES)
        if self._active_formatting_dirty:
            self._reconstruct_active_formatting()
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
            parent = (
                self._html
                if self._frameset_seen and not self._body_explicit and self._html is not None
                else self._current_parent()
            )
            foster = (
                self._foster_parent_for(parent)
                if parent.name in self._table_foster_targets and text.strip(_SPACE)
                else None
            )
            if foster is None:
                self._append(parent, Text(text))
            else:
                foster_parent, position = foster
                self._insert_at(foster_parent, position, Text(text))

    def _parse_formatting_start(self, name: str, attrs: dict[str, str | None], pos: int) -> int:
        if name == "a" and self._find_active_formatting_index("a") is not None:
            self._adoption_agency("a")
            self._remove_last_active_formatting_by_name("a")
            self._remove_last_open_element_by_name("a")
        elif name == "nobr" and self._find_open_index("nobr") is not None:
            self._adoption_agency("nobr")

        if self._active_formatting_dirty:
            self._reconstruct_active_formatting()
        signature = () if not attrs else self._attrs_signature(attrs)
        if len(self._active_formatting) >= 3:
            duplicate_index = self._find_active_formatting_duplicate(name, signature)
            if duplicate_index is not None:
                del self._active_formatting[duplicate_index]

        node = self._insert_sanitized_element(name, attrs, False, self._current_parent())
        self._append_active_formatting_entry(name, node.attrs, node, signature)
        return pos

    def _attrs_signature(self, attrs: dict[str, str | None]) -> tuple[tuple[str, str], ...]:
        if not attrs:
            return ()
        if len(attrs) == 1:
            name, value = next(iter(attrs.items()))
            return ((name, value or ""),)
        items = [(name, value or "") for name, value in attrs.items()]
        items.sort()
        return tuple(items)

    def _append_active_formatting_entry(
        self,
        name: str,
        attrs: dict[str, str | None],
        node: Element,
        signature: tuple[tuple[str, str], ...],
    ) -> None:
        entry_attrs = attrs if attrs else {}
        self._active_formatting.append(_FormattingEntry(name, entry_attrs, node, signature))
        self._active_formatting_dirty = False

    def _find_active_formatting_index(self, name: str) -> int | None:
        active = self._active_formatting
        for idx in range(len(active) - 1, -1, -1):
            if active[idx].name == name:
                return idx
        return None

    def _find_active_formatting_index_by_node(self, node: Node) -> int | None:
        active = self._active_formatting
        for idx in range(len(active) - 1, -1, -1):
            if active[idx].node is node:
                return idx
        return None

    def _find_active_formatting_duplicate(self, name: str, signature: tuple[tuple[str, str], ...]) -> int | None:
        matches = 0
        first_index: int | None = None
        for idx, entry in enumerate(self._active_formatting):
            if entry.name == name and entry.signature == signature:
                matches += 1
                if first_index is None:
                    first_index = idx
        return first_index if matches >= 3 else None

    def _remove_last_active_formatting_by_name(self, name: str) -> None:
        active = self._active_formatting
        for idx in range(len(active) - 1, -1, -1):
            if active[idx].name == name:
                del active[idx]
                return

    def _remove_last_open_element_by_name(self, name: str) -> None:
        stack = self._stack
        for idx in range(len(stack) - 1, 0, -1):
            if getattr(stack[idx], "name", None) == name:
                self._mark_active_formatting_dirty()
                del stack[idx]
                return

    def _reconstruct_active_formatting(self) -> None:
        active = self._active_formatting
        if not active:
            self._active_formatting_dirty = False
            return
        if not self._active_formatting_dirty and active[-1].node in self._stack:
            return

        idx = len(active) - 1
        while idx >= 0 and active[idx].node not in self._stack:
            idx -= 1
        idx += 1

        while idx < len(active):
            entry = active[idx]
            node = self._insert_sanitized_element(entry.name, entry.attrs.copy(), False, self._current_parent())
            entry.node = node
            idx += 1
        self._active_formatting_dirty = False

    def _adoption_agency(self, subject: str) -> None:
        stack = self._stack
        active = self._active_formatting
        for _ in range(8):
            formatting_index = self._find_active_formatting_index(subject)
            if formatting_index is None:
                if stack and getattr(stack[-1], "name", None) == subject:
                    self._close_until(subject)
                return

            entry = active[formatting_index]
            formatting_element = entry.node
            if stack and stack[-1] is formatting_element:
                stack.pop()
                del active[formatting_index]
                if not active:
                    self._active_formatting_dirty = False
                return
            try:
                formatting_stack_index = stack.index(formatting_element)
            except ValueError:
                del active[formatting_index]
                self._refresh_active_formatting_dirty()
                return

            if formatting_element is not stack[-1]:
                pass

            furthest_block: Node | None = None
            furthest_block_index: int | None = None
            for idx in range(formatting_stack_index + 1, len(stack)):
                candidate = stack[idx]
                if self._is_special_node(candidate):
                    furthest_block = candidate
                    furthest_block_index = idx
                    break

            if furthest_block is None:
                self._mark_active_formatting_dirty()
                del stack[formatting_stack_index:]
                del self._active_formatting[formatting_index]
                self._refresh_active_formatting_dirty()
                return
            if furthest_block_index is None:
                return

            bookmark = formatting_index + 1
            last_node: Node = furthest_block
            node_index = furthest_block_index

            inner_counter = 0
            while True:
                inner_counter += 1
                node_index -= 1
                node = stack[node_index]
                if node is formatting_element:
                    break

                node_formatting_index = self._find_active_formatting_index_by_node(node)
                if inner_counter > 3 and node_formatting_index is not None:
                    del self._active_formatting[node_formatting_index]
                    if node_formatting_index < bookmark:
                        bookmark -= 1
                    node_formatting_index = None

                if node_formatting_index is None:
                    self._mark_active_formatting_dirty()
                    del stack[node_index]
                    continue

                node_entry = self._active_formatting[node_formatting_index]
                new_node = self._clone_formatting_entry(node_entry)
                node_entry.node = new_node
                stack[node_index] = new_node
                node = new_node

                if last_node is furthest_block:
                    bookmark = node_formatting_index + 1

                self._detach_node(last_node)
                self._append_moved_node(node, last_node)
                last_node = node

            common_ancestor = stack[formatting_stack_index - 1]
            self._detach_node(last_node)
            foster = self._foster_parent_for(common_ancestor, for_tag=getattr(last_node, "name", None))
            if foster is None:
                self._append_moved_node(common_ancestor, last_node)
            else:
                foster_parent, position = foster
                self._insert_moved_node_at(foster_parent, position, last_node)

            new_formatting_element = self._clone_formatting_entry(entry)
            entry.node = new_formatting_element

            moved_children = list(furthest_block.children or ())
            if furthest_block.children is not None:
                furthest_block.children.clear()
            for child in moved_children:
                self._append_moved_node(new_formatting_element, child)

            self._append_moved_node(furthest_block, new_formatting_element)

            del self._active_formatting[formatting_index]
            bookmark -= 1
            if bookmark < 0:
                bookmark = 0
            if bookmark > len(self._active_formatting):
                bookmark = len(self._active_formatting)
            self._active_formatting.insert(bookmark, entry)

            try:
                self._mark_active_formatting_dirty()
                stack.remove(formatting_element)
            except ValueError:
                pass
            furthest_stack_index = stack.index(furthest_block)
            stack.insert(furthest_stack_index + 1, new_formatting_element)
            self._refresh_active_formatting_dirty()

    def _mark_active_formatting_dirty(self) -> None:
        if self._active_formatting:
            self._active_formatting_dirty = True

    def _refresh_active_formatting_dirty(self) -> None:
        active = self._active_formatting
        if not active:
            self._active_formatting_dirty = False
            return
        stack = self._stack
        self._active_formatting_dirty = any(entry.node not in stack for entry in active)

    def _clone_formatting_entry(self, entry: _FormattingEntry) -> Element:
        node = Element(entry.name, entry.attrs.copy(), "html")
        if entry.name not in self._allowed_tags:
            self._nodes_to_unwrap.append(node)
        return node

    def _is_special_node(self, node: Node) -> bool:
        return (
            getattr(node, "namespace", None) in {None, "html"}
            and getattr(node, "name", None) in self._special_elements
        )

    def _detach_node(self, node: Node) -> None:
        parent = node.parent
        children = parent.children if parent is not None else None
        if children is None:
            return
        try:
            children.remove(node)
        except ValueError:
            return
        node.parent = None

    def _append_moved_node(self, parent: Node, node: Node) -> None:
        children = parent.children
        if children is None:
            return
        children.append(node)
        node.parent = parent

    def _insert_moved_node_at(self, parent: Node, position: int, node: Node) -> None:
        children = parent.children
        if children is None:
            return
        children.insert(position, node)
        node.parent = parent

    def _should_insert_unwrapped_element(self, name: str, action: TagAction | None) -> bool:
        if name in _UNWRAP_CONSTRUCTION_SKIP_TAGS or name in self._void_elements:
            return False
        return action is None or not action.head_content

    def _unwrap_recorded_nodes(self) -> None:
        nodes = self._nodes_to_unwrap
        for node in reversed(nodes):
            if node.parent is not None:
                self._unwrap_node(node)
        nodes.clear()

    def _project_selectedcontent(self) -> None:
        if not self._has_selectedcontent:
            return
        root = self._doc
        pending = list(root.children or ())
        while pending:
            node = pending.pop()
            if type(node) is not Element:
                continue
            if node.name == "select":
                self._project_select_selectedcontent(node)
            children = node.children
            if children:
                pending.extend(reversed(children))

    def _project_select_selectedcontent(self, select: Element) -> None:
        children = select.children or []
        markers: list[Element] = []
        selected_option: Element | None = None
        first_option: Element | None = None
        pending = list(reversed(children))
        while pending:
            node = pending.pop()
            if type(node) is not Element:
                continue
            if node.name == "selectedcontent":
                markers.append(node)
                continue
            if node.name == "option":
                if first_option is None:
                    first_option = node
                if selected_option is None and "selected" in node.attrs:
                    selected_option = node
                continue
            nested = node.children
            if nested:
                pending.extend(reversed(nested))
        option = selected_option or first_option
        if option is None or not markers:
            return
        option_children = option.children or []
        for marker in markers:
            for child in option_children:
                clone = child.clone_node(deep=True) if isinstance(child, Element) else Text(child.data)
                self._append(marker, clone)

    def _unwrap_node(self, node: Element) -> None:
        parent = node.parent
        children = parent.children if parent is not None else None
        if parent is None or children is None:
            return
        try:
            index = children.index(node)
        except ValueError:
            return

        moved = list(node.children or ())
        if node.children is not None:
            node.children = []
        if type(node) is Template and node.template_content is not None:
            content = node.template_content
            if content.children:
                moved.extend(content.children)
                content.children = []
        if moved:
            for child in moved:
                child.parent = parent
            children[index : index + 1] = moved
        else:
            children.pop(index)
        node.parent = None

    def _skip_attrs(self, pos: int, end: int) -> tuple[dict[str, str | None], bool, int, bool]:
        html = self._html_input
        space = _SPACE
        attr_name_stop = _ATTR_NAME_STOP
        attr_value_stop = _ATTR_VALUE_STOP
        while pos < end:
            while pos < end and html[pos] in space:
                pos += 1
            if pos >= end:
                return {}, False, pos, False
            ch = html[pos]
            if ch == ">":
                return {}, False, pos + 1, True
            if ch == "/" and pos + 1 < end and html[pos + 1] == ">":
                return {}, True, pos + 2, True

            name_start = pos
            while pos < end and html[pos] not in attr_name_stop:
                pos += 1
            if pos == name_start:
                pos += 1
                continue
            while pos < end and html[pos] in space:
                pos += 1
            if pos < end and html[pos] == "=":
                pos += 1
                while pos < end and html[pos] in space:
                    pos += 1
                if pos < end and html[pos] in "\"'":
                    quote = html[pos]
                    close = html.find(quote, pos + 1, end)
                    if close == -1:
                        return {}, False, end, False
                    pos = close + 1
                else:
                    while pos < end and html[pos] not in attr_value_stop:
                        pos += 1
        return {}, False, pos, False

    def _parse_attrs_for_action(
        self,
        action: TagAction | None,
        pos: int,
        end: int,
    ) -> tuple[dict[str, str | None], bool, int, bool]:
        if action is None or not action.scan_attrs:
            return self._skip_attrs(pos, end)

        html = self._html_input
        space = _SPACE
        attr_name_stop = _ATTR_NAME_STOP
        attr_value_stop = _ATTR_VALUE_STOP
        attrs: dict[str, str | None] = {}
        allowed_attrs = action.allowed_attrs
        state_attrs = action.state_attrs
        url_attr_kinds = action.url_attr_kinds
        url_attr_rules = action.url_attr_rules
        tag = action.name

        while pos < end:
            while pos < end and html[pos] in space:
                pos += 1
            if pos >= end:
                return attrs, False, pos, False
            ch = html[pos]
            if ch == ">":
                return attrs, False, pos + 1, True
            if ch == "/" and pos + 1 < end and html[pos + 1] == ">":
                return attrs, True, pos + 2, True

            name_start = pos
            while pos < end and html[pos] not in attr_name_stop:
                pos += 1
            if pos == name_start:
                pos += 1
                continue
            raw_key = html[name_start:pos]
            key = raw_key if raw_key.islower() else raw_key.lower()
            keep_output = key in allowed_attrs
            keep_state = key in state_attrs

            while pos < end and html[pos] in space:
                pos += 1
            if not keep_output and not keep_state:
                if pos < end and html[pos] == "=":
                    pos += 1
                    while pos < end and html[pos] in space:
                        pos += 1
                    if pos < end and html[pos] in "\"'":
                        quote = html[pos]
                        close = html.find(quote, pos + 1, end)
                        if close == -1:
                            return attrs, False, end, False
                        pos = close + 1
                    else:
                        while pos < end and html[pos] not in attr_value_stop:
                            pos += 1
                continue

            value = ""
            if pos < end and html[pos] == "=":
                pos += 1
                while pos < end and html[pos] in space:
                    pos += 1
                if pos < end and html[pos] in "\"'":
                    quote = html[pos]
                    pos += 1
                    value_start = pos
                    close = html.find(quote, pos, end)
                    if close == -1:
                        return attrs, False, end, False
                    value = html[value_start:close]
                    pos = close + 1
                else:
                    value_start = pos
                    while pos < end and html[pos] not in attr_value_stop:
                        pos += 1
                    value = html[value_start:pos]

            if "&" in value:
                value = decode_entities_in_text(value, in_attribute=True)
            if keep_state and not keep_output:
                attrs[key] = value
                continue
            if self._strip_invisible_unicode and value and not value.isascii():
                value = _strip_invisible_unicode(value)
            if key in url_attr_kinds:
                rule = url_attr_rules.get(key)
                if rule is None:
                    continue
                sanitized = _sanitize_url_sink_value(
                    url_policy=self._url_policy,
                    rule=rule,
                    tag=tag,
                    attr=key,
                    kind=url_attr_kinds[key],
                    value=value,
                )
                if sanitized is None:
                    continue
                value = sanitized
            attrs[key] = value
        return attrs, False, pos, False

    def _skip_rawtext(self, name: str, pos: int, end: int) -> int:
        close, next_pos = (
            self._find_script_end_tag(pos, end) if name == "script" else self._find_rawtext_end_tag(name, pos, end)
        )
        if close is None:
            self._dropped_to_eof = True
            if name == "script" and self._script_eof_keeps_shell(pos, end):
                self._keep_empty_shell_on_eof = True
            self._append_text_boundary(self._current_parent())
            return end
        self._append_text_boundary(self._current_parent())
        if name in {"script", "style"}:
            self._skip_escaped_comment_space = True
            self._foster_next_table_whitespace = self._table_has_preceding_foster_text(self._current_parent())
        return next_pos

    def _skip_subtree(self, name: str, pos: int, end: int) -> int:
        html = self._html_input
        depth = 1
        while pos < end and depth:
            lt = html.find("<", pos, end)
            if lt == -1:
                self._dropped_to_eof = True
                return end
            p = lt + 1
            if self._lower_input.startswith("<![cdata[", lt):
                close = html.find("]]>", p + 8, end)
                if close == -1:
                    self._dropped_to_eof = True
                    return end
                pos = close + 3
                continue
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
            if not is_end and tag != name:
                if tag in self._plaintext_tags:
                    self._dropped_to_eof = True
                    return end
                if tag in self._rcdata_tags or tag in self._drop_content_tags or tag in self._rawtext_as_text_tags:
                    rawtext_close, rawtext_pos = (
                        self._find_script_end_tag(pos, end)
                        if tag == "script"
                        else self._find_rawtext_end_tag(tag, pos, end)
                    )
                    if rawtext_close is None:
                        self._dropped_to_eof = True
                        return end
                    pos = rawtext_pos
                    continue
            if tag == name:
                depth += -1 if is_end else 1
        if depth:
            self._dropped_to_eof = True
        return pos

    def _table_has_preceding_foster_text(self, parent: Node) -> bool:
        foster = self._foster_parent_for(parent)
        if foster is None:
            return False
        foster_parent, position = foster
        children = foster_parent.children
        if children is None or position <= 0:
            return False
        previous = children[position - 1]
        return type(previous) is Text and bool(previous.data)

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
        if self._keep_empty_shell_on_eof:
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

    def _script_eof_keeps_shell(self, pos: int, end: int) -> bool:
        raw = self._lower_input[pos:end].rstrip(_SPACE)
        if not raw.endswith("</script>"):
            return False
        comment = raw.find("<!--")
        if comment == -1:
            return False
        if raw.count("</script", comment + 4) < 2:
            return False
        return self._find_script_start_marker(pos + comment + 4, end) != -1

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
        self._mark_active_formatting_dirty()
        self._stack = [self._doc, self._html]  # type: ignore[list-item]
        self._after_head = False
        return True

    def _accept_fragment_frameset(self) -> bool:
        if not self._fragment or self._fragment_context_name != "html" or self._frameset_seen:
            return False
        if not self._body_allows_frameset(self._body):
            return False
        children = self._body.children
        if children is not None:
            children.clear()
        self._frameset_seen = True
        self._mark_active_formatting_dirty()
        return True

    def _blocks_frameset_action(self, action: TagAction | None, attrs: dict[str, str | None]) -> bool:
        if self._fragment or self._frameset_seen or action is None or not action.blocks_frameset:
            return False
        if action.name == "input":
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
        if self._html is None:
            return
        text = "".join(ch for ch in raw if ch in _SPACE)
        if text:
            self._append(self._html, Text(text))
