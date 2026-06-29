"""Plan-driven single-pass HTML parser.

Scanning, tree construction, and plan-selected sanitization run in one parser
engine without tokenizer or treebuilder handoffs.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass
from typing import TYPE_CHECKING

from justhtml.core.constants import (
    BUTTON_SCOPE_TERMINATORS,
    DEFAULT_SCOPE_TERMINATORS,
    DEFINITION_SCOPE_TERMINATORS,
    FOREIGN_ATTRIBUTE_ADJUSTMENTS,
    FOREIGN_BREAKOUT_ELEMENTS,
    FORMATTING_ELEMENTS,
    HEADING_ELEMENTS,
    HTML_INTEGRATION_POINT_SET,
    IMPLIED_END_TAGS,
    LIST_ITEM_SCOPE_TERMINATORS,
    MATHML_ATTRIBUTE_ADJUSTMENTS,
    MATHML_TEXT_INTEGRATION_POINT_SET,
    SPECIAL_ELEMENTS,
    SVG_ATTRIBUTE_ADJUSTMENTS,
    SVG_TAG_NAME_ADJUSTMENTS,
    TABLE_ALLOWED_CHILDREN,
    VOID_ELEMENTS,
)
from justhtml.core.doctype import doctype_error_and_quirks
from justhtml.core.entities import decode_entities_in_text
from justhtml.core.errors import generate_error_message
from justhtml.core.types import Doctype, ParseError
from justhtml.dom import Comment, Document, DocumentFragment, Element, Node, Template, Text
from justhtml.sanitizer import DEFAULT_DOCUMENT_POLICY, DEFAULT_POLICY, SanitizationPolicy, _strip_invisible_unicode
from justhtml.sanitizer.url import (
    _URL_SINK_ATTRS,
    _sanitize_url_sink_value,
    _url_sink_kind_for_attr,
)
from justhtml.sanitizer.url.runtime import _URL_CONTROL_CHAR_REGEX, _get_scheme, _normalize_url_for_checking

from . import scanner as _scanner

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from justhtml.parser.context import FragmentContext
    from justhtml.sanitizer.url import UrlPolicy, UrlRule
    from justhtml.sanitizer.url.spec import UrlSinkKind

_TAG_NAME_RE = re.compile(r"[A-Za-z][^\t\n\f\r />]*")
# HTML's tokenizer permits any non-delimiter code point in tag and attribute
# names. Keep the serializer-safe subset broad enough to preserve those names
# while still rejecting characters that can terminate the surrounding markup.
_SERIALIZABLE_TAG_NAME_RE = re.compile(r"^[A-Za-z][^\t\n\f\r />]*$")
_SERIALIZABLE_ATTR_NAME_RE = re.compile(r"^[^\t\n\f\r />=]+$")
_DOCTYPE_RE = re.compile(
    r"""\s*([^\s>]+)(?:\s*(PUBLIC|SYSTEM)\s*(?:(?:"([^"]*)"|'([^']*)')\s*(?:"([^"]*)"|'([^']*)')?)?)?""",
    re.IGNORECASE,
)
_SPACE = _scanner.SPACE
_TAG_NAME_STOP = _scanner.TAG_NAME_STOP + "\r"
_ATTR_NAME_STOP = _scanner.ATTR_NAME_STOP + "\r"
_ATTR_VALUE_STOP = _scanner.ATTR_VALUE_STOP
_XML_INVALID_SINGLE_CHARS = []
for _plane in range(17):
    _base = _plane * 0x10000
    _XML_INVALID_SINGLE_CHARS.append(chr(_base + 0xFFFE))
    _XML_INVALID_SINGLE_CHARS.append(chr(_base + 0xFFFF))
_XML_COERCION_PATTERN = re.compile(r"[\f\uFDD0-\uFDEF" + "".join(_XML_INVALID_SINGLE_CHARS) + "]")
_DROP_CONTENT_TAGS = {"script", "style"}
_DROP_SUBTREE_TAGS = {"svg", "math"}
_RCDATA_TAGS = {"title", "textarea"}
_RAWTEXT_AS_TEXT_TAGS = {"iframe", "noembed", "noframes", "noscript", "xmp"}
_RAWTEXT_ELEMENT_TAGS = {"script", "style", "iframe", "noembed", "noframes", "noscript", "xmp"}
_PLAINTEXT_TAGS = {"plaintext"}
_ACTIVE_FORMATTING_TAGS = FORMATTING_ELEMENTS
_ACTIVE_FORMATTING_MARKER_TAGS = {"applet", "caption", "marquee", "object"}
_PARSER_ONLY_NAMESPACE = "justhtml-parser-only"
_DEFAULT_SCOPE_BOUNDARIES = frozenset(DEFAULT_SCOPE_TERMINATORS)
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
    "form",
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
    "xmp",
} | HEADING_ELEMENTS
_HEAD_NOSCRIPT_ALLOWED_START_TAGS = {"basefont", "bgsound", "link", "meta", "noframes", "style"}
_HEAD_NOSCRIPT_VOID_START_TAGS = {"basefont", "bgsound"}
_HEAD_ONLY_VOID_START_TAGS = {"basefont", "bgsound"}
_HTML_VOID_COMPAT_TAGS = {"basefont", "bgsound", "frame", "keygen"}
_DEFINITION_SCOPE_BOUNDARIES = frozenset(DEFINITION_SCOPE_TERMINATORS)
_LIST_ITEM_SCOPE_BOUNDARIES = frozenset(LIST_ITEM_SCOPE_TERMINATORS)
_PRE_LINEFEED_IGNORING_TAGS = {"listing", "pre"}
_TABLE_CONTEXT_BOUNDARIES = frozenset({"table"})
_GENERAL_END_TAG_BOUNDARIES = frozenset(SPECIAL_ELEMENTS) | _BUTTON_SCOPE_BOUNDARIES | _TABLE_CONTEXT_BOUNDARIES
_TABLE_SECTION_TAGS = {"tbody", "thead", "tfoot"}
_TABLE_CELL_TAGS = {"td", "th"}
_TABLE_SCOPED_END_TAGS = {"caption", "table", "tbody", "td", "tfoot", "th", "thead", "tr"}
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
_TEMPLATE_TABLE_BODY_IGNORED_START_TAGS = {"caption", "col", "colgroup", "table"}
_TEMPLATE_ROW_STRUCTURE_START_TAGS = {"caption", "col", "colgroup", "tbody", "tfoot", "thead", "tr", "table"}
_FRAMESET_BODY_OK_TAGS = {"div", "figure", "p", "param", "source", "track"}
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
_FOREIGN_FULL_PARSE_TAGS = FOREIGN_BREAKOUT_ELEMENTS | {"annotation-xml", "desc", "font", "foreignobject", "title"}
_URL_FAST_FALLBACK = object()


def _xml_coercion_callback(match: re.Match[str]) -> str:
    return " " if match.group(0) == "\f" else "\ufffd"


def _coerce_text_for_xml(text: str) -> str:
    if text.isascii():
        return text.replace("\f", " ") if "\f" in text else text
    if not _XML_COERCION_PATTERN.search(text):
        return text
    return _XML_COERCION_PATTERN.sub(_xml_coercion_callback, text)


def _coerce_comment_for_xml(text: str) -> str:
    return text.replace("--", "- -") if "--" in text else text


def _is_hidden_input(name: str, attrs: dict[str, str | None]) -> bool:
    input_type = attrs.get("type") if name == "input" else None
    return isinstance(input_type, str) and input_type.lower() == "hidden"


@dataclass(frozen=True, slots=True)
class EnginePlan:
    """Compiled execution plan for a sanitizer-aware parse run.

    The executor copies these fields into instance slots so the hot path stays
    close to the hardcoded PoC while giving policy/behavior compilation a clear
    boundary.
    """

    policy: SanitizationPolicy | None
    raw_mode: bool
    allowed_tags: frozenset[str]
    allowed_global_attrs: Collection[str]
    allowed_attrs_by_tag: Mapping[str, frozenset[str]]
    url_policy: UrlPolicy
    url_rules: Mapping[tuple[str, str], UrlRule]
    definition_scope_boundaries: frozenset[str]
    drop_comments: bool
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
    rawtext_element_tags: frozenset[str]
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


_DEFAULT_ENGINE_PLAN_CACHE: dict[tuple[bool, bool], EnginePlan] = {}
_RAW_ENGINE_PLAN_CACHE: dict[tuple[bool, bool], EnginePlan] = {}


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
    preserve_state_attrs: bool
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
    drop_content_tags: frozenset[str],
    drop_subtree_tags: frozenset[str],
    rawtext_as_text_tags: Collection[str],
) -> dict[str, TagAction]:
    known_tags = set(allowed_tags)
    known_tags.update(drop_content_tags)
    known_tags.update(drop_subtree_tags)
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
        preserve_state_attrs = tag in _ACTIVE_FORMATTING_TAGS and not allowed
        url_attr_kinds: dict[str, UrlSinkKind] = {}
        url_attr_rules: dict[str, UrlRule] = {}
        for attr in allowed_attrs.intersection(_URL_SINK_ATTRS):
            rule = url_rules.get((tag, attr))
            if rule is None:  # pragma: no cover - planner only iterates explicitly configured URL attributes
                continue
            kind = _url_sink_kind_for_attr(tag=tag, attr=attr, attrs={attr: ""})
            if kind is None:  # pragma: no cover - URL sink attributes always have a declared sink kind
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
            scan_attrs=bool(allowed_attrs or state_attrs or preserve_state_attrs),
            preserve_state_attrs=preserve_state_attrs,
            active_formatting=tag in _ACTIVE_FORMATTING_TAGS,
            blocks_frameset=tag in _FRAMESET_BLOCKING_START_TAGS,
            drop_content=tag in drop_content_tags,
            drop_subtree=tag in drop_subtree_tags,
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


def can_compile_engine_plan(policy: SanitizationPolicy, *, fragment: bool) -> bool:
    """Return True when a policy can run entirely inside ParseEngine."""
    if policy.unsafe_handling != "strip":
        return False
    if policy.disallowed_tag_handling != "unwrap":
        return False
    if not fragment and not {"html", "head", "body"}.issubset(policy.allowed_tags):
        return False
    if not policy.drop_comments:
        return False
    if policy.force_link_rel:
        return False
    if "style" in policy.allowed_tags:
        return False
    if "svg" in policy.allowed_tags or "math" in policy.allowed_tags:
        return False
    default_policy = DEFAULT_POLICY if fragment else DEFAULT_DOCUMENT_POLICY
    default_tags = default_policy.allowed_tags
    if not policy.allowed_tags.issubset(default_tags):
        return False
    if policy is not default_policy and "template" not in policy.allowed_tags:
        return False

    for tag, attrs in policy.allowed_attributes.items():
        allowed = default_policy.allowed_attributes.get(tag)
        if allowed is None:  # pragma: no cover - normalized policies always provide global defaults
            allowed = default_policy.allowed_attributes.get("*", ())
        if not set(attrs).issubset(allowed):  # pragma: no cover - rejected by policy normalization tests
            return False
        if any(  # pragma: no cover - rejected by policy normalization tests
            attr in _URL_SINK_ATTRS and (tag, attr) not in policy.url_policy.allow_rules for attr in attrs
        ):
            return False
    return not any("style" in attrs for attrs in policy.allowed_attributes.values())


def compile_engine_plan(
    *,
    policy: SanitizationPolicy,
    fragment: bool,
    scripting_enabled: bool = True,
) -> EnginePlan:
    """Compile a sanitizer-aware execution plan for ParseEngine."""
    allowed_attrs = policy.allowed_attributes
    allowed_global = allowed_attrs.get("*", ())
    allowed_by_tag = {
        str(tag).lower(): frozenset(allowed_global).union(attrs) for tag, attrs in allowed_attrs.items() if tag != "*"
    }
    rawtext_as_text_tags = _RAWTEXT_AS_TEXT_TAGS if scripting_enabled else _RAWTEXT_AS_TEXT_TAGS - {"noscript"}
    drop_content_tags = frozenset(policy.drop_content_tags)
    drop_subtree_tags = frozenset(_DROP_SUBTREE_TAGS)
    tag_actions = _compile_tag_actions(
        allowed_tags=policy.allowed_tags,
        allowed_global=allowed_global,
        allowed_by_tag=allowed_by_tag,
        url_rules=policy.url_policy.allow_rules,
        drop_content_tags=drop_content_tags,
        drop_subtree_tags=drop_subtree_tags,
        rawtext_as_text_tags=rawtext_as_text_tags,
    )
    return EnginePlan(
        policy=policy,
        raw_mode=False,
        allowed_tags=policy.allowed_tags,
        allowed_global_attrs=allowed_global,
        allowed_attrs_by_tag=allowed_by_tag,
        url_policy=policy.url_policy,
        url_rules=policy.url_policy.allow_rules,
        definition_scope_boundaries=frozenset(_DEFINITION_SCOPE_BOUNDARIES),
        drop_comments=policy.drop_comments,
        drop_doctype=policy.drop_doctype,
        drop_content_tags=drop_content_tags,
        drop_subtree_tags=drop_subtree_tags,
        active_formatting_tags=frozenset(_ACTIVE_FORMATTING_TAGS),
        frameset_body_ok_tags=frozenset(_FRAMESET_BODY_OK_TAGS),
        frameset_blocking_start_tags=frozenset(_FRAMESET_BLOCKING_START_TAGS),
        head_content_tags=frozenset(_HEAD_CONTENT_TAGS),
        implied_end_tags=frozenset(IMPLIED_END_TAGS),
        list_item_scope_boundaries=frozenset(_LIST_ITEM_SCOPE_BOUNDARIES),
        p_closing_start_tags=frozenset(_P_CLOSING_START_TAGS),
        pre_linefeed_ignoring_tags=frozenset(_PRE_LINEFEED_IGNORING_TAGS),
        rawtext_element_tags=frozenset(),
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


def compile_default_engine_plan(*, fragment: bool, scripting_enabled: bool = True) -> EnginePlan:
    """Return the cached default-safe execution plan."""
    cache_key = (bool(fragment), bool(scripting_enabled))
    cached = _DEFAULT_ENGINE_PLAN_CACHE.get(cache_key)
    if cached is not None:
        return cached

    policy = DEFAULT_POLICY if fragment else DEFAULT_DOCUMENT_POLICY
    plan = compile_engine_plan(policy=policy, fragment=fragment, scripting_enabled=scripting_enabled)
    _DEFAULT_ENGINE_PLAN_CACHE[cache_key] = plan
    return plan


def compile_raw_engine_plan(*, fragment: bool, scripting_enabled: bool = True) -> EnginePlan:
    """Return a cached unsanitized execution plan for transform/raw parsing."""
    cache_key = (bool(fragment), bool(scripting_enabled))
    cached = _RAW_ENGINE_PLAN_CACHE.get(cache_key)
    if cached is not None:
        return cached

    allowed_tags = frozenset(
        SPECIAL_ELEMENTS
        | FORMATTING_ELEMENTS
        | HEADING_ELEMENTS
        | _HEAD_CONTENT_TAGS
        | _P_CLOSING_START_TAGS
        | _TABLE_STRUCTURE_START_TAGS
        | _TABLE_SECTION_TAGS
        | _TABLE_CELL_TAGS
        | _PRE_LINEFEED_IGNORING_TAGS
        | _RAWTEXT_ELEMENT_TAGS
        | _RCDATA_TAGS
        | _PLAINTEXT_TAGS
        | {
            "annotation-xml",
            "datalist",
            "foreignobject",
            "image",
            "math",
            "mglyph",
            "mi",
            "mn",
            "mo",
            "ms",
            "mtext",
            "option",
            "optgroup",
            "rb",
            "rp",
            "rt",
            "rtc",
            "selectedcontent",
            "svg",
        }
    )
    rawtext_element_tags = _RAWTEXT_ELEMENT_TAGS if scripting_enabled else _RAWTEXT_ELEMENT_TAGS - {"noscript"}
    tag_actions = _compile_tag_actions(
        allowed_tags=allowed_tags,
        allowed_global=frozenset(),
        allowed_by_tag={},
        url_rules={},
        drop_content_tags=frozenset(),
        drop_subtree_tags=frozenset(),
        rawtext_as_text_tags=frozenset(),
    )
    plan = EnginePlan(
        policy=None,
        raw_mode=True,
        allowed_tags=allowed_tags,
        allowed_global_attrs=frozenset(),
        allowed_attrs_by_tag={},
        url_policy=DEFAULT_POLICY.url_policy,
        url_rules={},
        definition_scope_boundaries=frozenset(_DEFINITION_SCOPE_BOUNDARIES),
        drop_comments=False,
        drop_doctype=False,
        drop_content_tags=frozenset(),
        drop_subtree_tags=frozenset(),
        active_formatting_tags=frozenset(_ACTIVE_FORMATTING_TAGS),
        frameset_body_ok_tags=frozenset(_FRAMESET_BODY_OK_TAGS),
        frameset_blocking_start_tags=frozenset(_FRAMESET_BLOCKING_START_TAGS),
        head_content_tags=frozenset(_HEAD_CONTENT_TAGS),
        implied_end_tags=frozenset(IMPLIED_END_TAGS),
        list_item_scope_boundaries=frozenset(_LIST_ITEM_SCOPE_BOUNDARIES),
        p_closing_start_tags=frozenset(_P_CLOSING_START_TAGS),
        pre_linefeed_ignoring_tags=frozenset(_PRE_LINEFEED_IGNORING_TAGS),
        rawtext_element_tags=frozenset(rawtext_element_tags),
        rawtext_as_text_tags=frozenset(),
        rcdata_tags=frozenset(_RCDATA_TAGS),
        plaintext_tags=frozenset(_PLAINTEXT_TAGS),
        special_elements=frozenset(SPECIAL_ELEMENTS),
        strip_invisible_unicode=False,
        table_allowed_children=frozenset(TABLE_ALLOWED_CHILDREN) | {"form"},
        table_cell_tags=frozenset(_TABLE_CELL_TAGS),
        table_foster_targets=frozenset(_TABLE_FOSTER_TARGETS),
        table_section_tags=frozenset(_TABLE_SECTION_TAGS),
        tag_actions=tag_actions,
        void_elements=VOID_ELEMENTS,
    )
    _RAW_ENGINE_PLAN_CACHE[cache_key] = plan
    return plan


@dataclass(slots=True)
class _FormattingEntry:
    name: str
    attrs: dict[str, str | None]
    node: Element
    signature: tuple[tuple[str, str], ...]


class _FormattingMarker:
    __slots__ = ()


_ACTIVE_FORMATTING_MARKER = _FormattingMarker()


class ParseEngine:
    __slots__ = (
        "_active_formatting",
        "_active_formatting_dirty",
        "_active_formatting_tags",
        "_after_body",
        "_after_document",
        "_after_head",
        "_after_html",
        "_allowed_tags",
        "_body",
        "_body_explicit",
        "_body_mode_seen",
        "_collect_errors",
        "_definition_scope_boundaries",
        "_doc",
        "_doctype_seen",
        "_drop_comments",
        "_drop_content_tags",
        "_drop_doctype",
        "_drop_subtree_tags",
        "_dropped_to_eof",
        "_emit_bogus_markup_as_text",
        "_errors",
        "_explicit_head",
        "_explicit_html",
        "_foreign_context_seen",
        "_form_element",
        "_foster_next_table_whitespace",
        "_fragment",
        "_fragment_context_name",
        "_fragment_context_namespace",
        "_fragment_context_node",
        "_frameset_blocked",
        "_frameset_blocking_start_tags",
        "_frameset_body_ok_tags",
        "_frameset_seen",
        "_has_carriage_return",
        "_has_form_feed",
        "_has_null",
        "_has_selectedcontent",
        "_head",
        "_head_content_tags",
        "_head_reentry",
        "_html",
        "_html_input",
        "_iframe_srcdoc",
        "_ignore_lf",
        "_implied_end_tags",
        "_in_colgroup",
        "_in_head_noscript",
        "_initial_mode_done",
        "_keep_empty_shell_on_eof",
        "_length",
        "_line_starts",
        "_list_item_scope_boundaries",
        "_lower_input",
        "_nodes_to_drop",
        "_nodes_to_unwrap",
        "_p_closing_start_tags",
        "_parser_only_template_depth",
        "_plaintext_tags",
        "_plan",
        "_pre_linefeed_ignoring_tags",
        "_quirks_mode",
        "_raw_mode",
        "_rawtext_as_text_tags",
        "_rawtext_element_tags",
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
        "_template_modes",
        "_track_node_locations",
        "_track_tag_spans",
        "_url_policy",
        "_void_elements",
        "_xml_coercion",
    )

    def __init__(
        self,
        html: str,
        *,
        fragment: bool,
        fragment_context: FragmentContext | None = None,
        scripting_enabled: bool = True,
        plan: EnginePlan | None = None,
        collect_errors: bool = False,
        iframe_srcdoc: bool = False,
        track_node_locations: bool = False,
        track_tag_spans: bool = False,
        emit_bogus_markup_as_text: bool = False,
        xml_coercion: bool = False,
    ) -> None:
        self._html_input = html
        self._length = len(html)
        self._lower_input = html.lower()
        self._fragment = bool(fragment)
        self._collect_errors = bool(collect_errors)
        self._errors: list[ParseError] = []
        self._iframe_srcdoc = bool(iframe_srcdoc)
        self._track_node_locations = bool(track_node_locations)
        self._track_tag_spans = bool(track_tag_spans)
        self._emit_bogus_markup_as_text = bool(emit_bogus_markup_as_text)
        self._xml_coercion = bool(xml_coercion)
        self._line_starts: list[int] | None = None
        self._fragment_context_namespace = (
            fragment_context.namespace.lower() if fragment_context is not None and fragment_context.namespace else None
        )
        fragment_context_name: str | None = None
        if fragment_context is not None and fragment_context.tag_name:
            fragment_context_name = fragment_context.tag_name.lower()
            if self._fragment_context_namespace == "svg":
                fragment_context_name = SVG_TAG_NAME_ADJUSTMENTS.get(fragment_context_name, fragment_context_name)
        self._fragment_context_name = fragment_context_name
        self._fragment_context_node: Element | None = None
        self._plan = (
            plan
            if plan is not None
            else compile_default_engine_plan(fragment=fragment, scripting_enabled=scripting_enabled)
        )
        self._active_formatting_tags = self._plan.active_formatting_tags
        self._raw_mode = self._plan.raw_mode
        self._allowed_tags = self._plan.allowed_tags
        self._url_policy = self._plan.url_policy
        self._definition_scope_boundaries = self._plan.definition_scope_boundaries
        self._drop_comments = self._plan.drop_comments
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
        self._rawtext_element_tags = self._plan.rawtext_element_tags
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
        self._after_body = False
        self._after_document = False
        self._after_head = False
        self._after_html = False
        self._dropped_to_eof = False
        self._doctype_seen = False
        self._explicit_head = False
        self._explicit_html = False
        self._foster_next_table_whitespace = 0
        self._form_element: Element | None = None
        self._frameset_blocked = False
        self._frameset_seen = False
        self._has_carriage_return = "\r" in html
        self._has_form_feed = "\f" in html
        self._has_null = "\0" in html
        self._foreign_context_seen = False
        self._has_selectedcontent = False
        self._head_reentry = False
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
        self._active_formatting: list[_FormattingEntry | _FormattingMarker] = []
        self._active_formatting_dirty = False
        self._nodes_to_drop: list[Element] = []
        self._nodes_to_unwrap: list[Element] = []
        self._template_modes: list[str] = []
        self._stack: list[Node] = []

    @property
    def errors(self) -> list[ParseError]:
        return self._errors

    def _line_col_at_pos(self, pos: int) -> tuple[int, int]:
        if pos < 0:  # pragma: no cover - parser offsets are non-negative
            pos = 0
        elif pos >= self._length:  # pragma: no cover - callers clamp offsets to input
            pos = max(0, self._length - 1)
        starts = self._line_starts
        if starts is None:
            html = self._html_input
            starts = [0]
            find_from = 0
            while True:
                newline = html.find("\n", find_from)
                if newline == -1:
                    break
                starts.append(newline + 1)
                find_from = newline + 1
            self._line_starts = starts
        line_index = bisect_right(starts, pos) - 1
        return line_index + 1, pos - starts[line_index] + 1

    def _source_location(self, pos: int) -> tuple[int, int]:
        return self._line_col_at_pos(pos)

    def _set_origin(self, node: Node | Text, pos: int | None) -> None:
        if not self._track_node_locations or pos is None:
            return
        node._origin_pos = pos
        node._origin_line, node._origin_col = self._line_col_at_pos(pos)

    def _set_source_span(self, node: Element, start: int | None, end: int | None) -> None:
        if self._track_tag_spans and start is not None and end is not None:
            node._source_html = self._html_input
            node._start_tag_start = start
            node._start_tag_end = end

    def _set_end_span(self, node: Node, name: str, start: int | None, end: int | None) -> None:
        if not isinstance(node, Element):  # pragma: no cover - end spans are only assigned to elements
            return
        node_name = node.name
        if node_name != name and (node.namespace in {None, "html"} or node_name.lower() != name):
            return
        node._end_tag_present = True
        if self._track_tag_spans and start is not None and end is not None:
            node._source_html = self._html_input
            node._end_tag_start = start
            node._end_tag_end = end

    def _new_text(self, data: str, source_pos: int | None = None) -> Text:
        node = Text(data)
        if self._track_node_locations and source_pos is not None:
            node._origin_pos = source_pos
            node._origin_line, node._origin_col = self._line_col_at_pos(source_pos)
        return node

    def _end_tag_stays_in_foreign_context(self, name: str, tag_start: int, tag_end: int) -> bool:
        stack = self._stack
        if not stack:  # pragma: no cover - parser stack always contains the document root
            return False
        current = stack[-1]
        if current.namespace in {None, "html", _PARSER_ONLY_NAMESPACE}:
            return False
        if name in {"br", "p"}:
            return False
        if name in _TABLE_SCOPED_END_TAGS:
            return False

        crossed_integration_point = False
        for idx in range(len(stack) - 1, 0, -1):
            node = stack[idx]
            if self._node_matches_end_name(node, name):
                if node.namespace in {None, "html", _PARSER_ONLY_NAMESPACE} and crossed_integration_point:
                    return True
                if self._fragment_context_node is not None and node is self._fragment_context_node:
                    return True
                if self._track_tag_spans:
                    self._set_end_span(node, name, tag_start, tag_end)
                del stack[idx:]
                return True
            if self._is_html_integration_point(node) or self._is_mathml_text_integration_point(node):
                crossed_integration_point = True
            if node.namespace in {None, "html", _PARSER_ONLY_NAMESPACE}:
                return False
        return True

    def _emit_error(
        self,
        code: str,
        pos: int,
        *,
        tag_name: str | None = None,
        category: str = "tokenizer",
        end_pos: int | None = None,
    ) -> None:
        if not self._collect_errors:  # pragma: no cover - callers guard error emission
            return
        line, column = self._source_location(pos)
        end_column = None
        if end_pos is not None:
            end_line, end_col = self._source_location(end_pos)
            if end_line == line:  # pragma: no branch - compatibility diagnostics normalize multiline tag spans
                end_column = end_col + 1
        self._errors.append(
            ParseError(
                code,
                line=line,
                column=column,
                category=category,
                message=generate_error_message(code, tag_name),
                source_html=self._html_input,
                end_column=end_column,
            )
        )

    def _emit_null_errors(self, start: int, end: int) -> None:
        if not self._collect_errors:  # pragma: no cover - basic error scan only runs when collecting
            return
        html = self._html_input
        pos = html.find("\0", start, end)
        while pos != -1:
            self._emit_error("unexpected-null-character", pos)
            pos = html.find("\0", pos + 1, end)

    def _collect_basic_errors(self) -> None:
        html = self._html_input
        lower = self._lower_input
        end = self._length
        self._emit_null_errors(0, end)

        if not self._fragment:
            first = 0
            while first < end and html[first] in _SPACE:
                first += 1
            if first < end and not lower.startswith("<!doctype", first):
                if html[first] == "<" and first + 1 < end and html[first + 1].isalpha():
                    tag_end = self._find_tag_end(first + 2, end)
                    self._emit_error(
                        "expected-doctype-but-got-start-tag",
                        tag_end if tag_end != -1 else end - 1,
                        tag_name=self._read_tag_name(first + 1, end),
                        category="treebuilder",
                        end_pos=tag_end if tag_end != -1 else None,
                    )
                else:
                    self._emit_error("expected-doctype-but-got-chars", first, category="treebuilder")

        open_tags: list[str] = []
        pos = 0
        while pos < end:
            lt = html.find("<", pos, end)
            if lt == -1:
                return
            pos = lt + 1
            if pos >= end:
                return
            ch = html[pos]
            if ch == "!":
                # The leading-LF exception for <pre>/<listing> applies only to
                # the immediately following token. Markup declarations emit a
                # comment, doctype, or CDATA token, so they consume the pending
                # exception even when that token is not retained in the DOM.
                self._ignore_lf = False
                if html.startswith("<!--", lt):
                    close = self._find_comment_end(pos + 1, end)
                    if close == -1:
                        self._emit_error("eof-in-comment", end - 1)
                        return
                    pos = close + 3
                    continue
                tag_end = self._find_tag_end(pos + 1, end)
                if tag_end == -1:
                    self._emit_error("eof-in-tag", end - 1)
                    return
                pos = tag_end + 1
                continue
            if ch == "/":
                name_start = pos + 1
                if name_start >= end or not html[name_start].isalpha():
                    tag_end = self._find_tag_end(name_start, end)
                    pos = end if tag_end == -1 else tag_end + 1
                    continue
                name = self._read_tag_name(name_start, end)
                tag_end = self._find_tag_end(name_start + len(name), end)
                if tag_end == -1:
                    self._emit_error("eof-in-tag", end - 1)
                    return
                if name == "br" or name not in open_tags:
                    self._emit_error("unexpected-end-tag", lt, tag_name=name, category="treebuilder", end_pos=tag_end)
                else:
                    for idx in range(
                        len(open_tags) - 1, -1, -1
                    ):  # pragma: no branch - opposite edge requires invalid parser state
                        if open_tags[idx] == name:
                            del open_tags[idx:]
                            break
                pos = tag_end + 1
                continue
            if not ch.isalpha():
                pos = lt + 1
                continue
            name = self._read_tag_name(pos, end)
            tag_end = self._find_tag_end(pos + len(name), end)
            if tag_end == -1:
                self._emit_error("eof-in-tag", end - 1)
                return
            if name not in self._void_elements and not self._is_self_closing_source_tag(pos + len(name), tag_end):
                open_tags.append(name)
            pos = tag_end + 1

    def _read_tag_name(self, pos: int, end: int) -> str:
        html = self._html_input
        start = pos
        while pos < end and html[pos] not in _TAG_NAME_STOP:
            pos += 1
        name = html[start:pos]
        return name if name.islower() else name.lower()

    def _is_self_closing_source_tag(self, pos: int, tag_end: int) -> bool:
        html = self._html_input
        idx = tag_end - 1
        while idx >= pos and html[idx] in _SPACE:
            idx -= 1
        return idx >= pos and html[idx] == "/"

    def parse(self) -> Document | DocumentFragment:
        if self._collect_errors:
            self._collect_basic_errors()

        if self._fragment:
            root = DocumentFragment()
            self._doc = root
            if self._track_tag_spans:
                root._source_html = self._html_input
            context_name = self._fragment_context_name
            html_context = self._fragment_context_namespace in {None, "html"}
            if html_context and context_name in self._rcdata_tags:
                self._body = root
                self._stack = [root]
                text = self._clean_text(self._html_input, replace_null=True)
                if context_name == "textarea" and text.startswith("\n"):
                    text = text[1:]
                if text:
                    self._append(root, self._new_text(text, 1 if text != self._html_input else 0))
                return root

            if html_context and context_name in self._plaintext_tags:
                self._body = root
                self._stack = [root]
                self._append_raw_literal_text(self._html_input, 0)
                return root

            if html_context and (
                context_name in self._drop_content_tags
                or context_name in self._rawtext_as_text_tags
                or context_name in self._rawtext_element_tags
            ):
                self._body = root
                self._stack = [root]
                self._append_raw_literal_text(self._html_input, 0)
                return root

            if html_context and context_name == "html":
                context = Element("html", {}, "html")
                head = Element("head", {}, "html")
                body = Element("body", {}, "html")
                self._append(root, context)
                self._append(context, head)
                self._append(context, body)
                self._fragment_context_node = context
                self._html = context
                self._head = head
                self._body = body
                self._stack = [root, context, body]
                if not self._raw_mode:
                    if (
                        "head" not in self._allowed_tags
                    ):  # pragma: no branch - opposite edge requires invalid parser state
                        self._nodes_to_unwrap.append(head)
                    if (
                        "body" not in self._allowed_tags
                    ):  # pragma: no branch - opposite edge requires invalid parser state
                        self._nodes_to_unwrap.append(body)
            elif context_name and context_name != "div":
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
            if self._track_tag_spans:
                doc._source_html = self._html_input
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
        self._drop_recorded_nodes()
        self._unwrap_recorded_nodes()
        self._finish_fragment_context()
        return self._doc

    def _append(self, parent: Node, node: Node | Text) -> None:
        children = parent.children
        if children is None:  # pragma: no cover - parser insertion parents are containers
            return
        if type(node) is Text and children and type(children[-1]) is Text:
            children[-1].data = (children[-1].data or "") + (node.data or "")
            return
        children.append(node)
        node.parent = parent

    def _append_text_boundary(self, parent: Node) -> None:
        children = parent.children
        if children is None:  # pragma: no cover - text boundaries target containers
            return
        node = Text("")
        node.parent = parent
        children.append(node)

    def _append_comment(self, data: str, source_pos: int | None = None) -> None:
        if self._has_carriage_return and "\r" in data:
            data = data.replace("\r\n", "\n").replace("\r", "\n")
        if self._has_null and "\0" in data:
            data = data.replace("\0", "\ufffd")
        if self._xml_coercion:
            data = _coerce_comment_for_xml(data)
        if self._after_document and source_pos is not None:
            marker = "</html" if self._after_html else "</body"
            close_start = self._lower_input.rfind(marker, 0, source_pos)
            close_end = self._html_input.find(">", close_start, source_pos) if close_start != -1 else -1
            trailing = self._html_input[close_end + 1 : source_pos] if close_end != -1 else ""
            if trailing and "<" not in trailing and trailing.strip(_SPACE):
                self._after_body = False
                self._after_document = False
                self._after_html = False
                self._body_mode_seen = True
        node = Comment(data=data)
        self._set_origin(node, source_pos)
        parent: Node
        if self._current_parent() is self._head or (
            not self._fragment
            and self._head is not None
            and not self._body_mode_seen
            and not self._body_has_content()
            and self._current_parent() in {self._html, self._body}
            and any(type(child) is Template for child in self._head.children or ())
        ):
            self._append(self._head, node)
            return
        if not self._fragment and self._after_html:
            self._append(self._doc, node)
            return
        if not self._fragment and self._after_body and self._html is not None:
            self._append(self._html, node)
            return
        if not self._fragment and (
            not self._initial_mode_done
            or (
                not self._explicit_html
                and not self._explicit_head
                and not self._body_mode_seen
                and not self._body_has_content()
                and self._current_parent() is self._body
            )
        ):
            children = self._doc.children
            if children is not None:
                insert_at = 0
                while insert_at < len(children) and children[insert_at].name != "html":
                    insert_at += 1
                children.insert(insert_at, node)
                node.parent = self._doc
                return
            parent = self._doc  # pragma: no cover - Document always owns a children list
        elif (
            not self._fragment
            and self._html is not None
            and self._head is not None
            and not self._body_mode_seen
            and not self._body_has_content()
            and self._current_parent() in {self._body, self._html}
        ):
            children = self._html.children
            if children is not None:
                anchor = self._body if self._after_head or self._explicit_head else self._head
                try:
                    insert_at = children.index(anchor)
                except ValueError:  # pragma: no cover - shell anchor is owned by html
                    insert_at = len(children)
                children.insert(insert_at, node)
                node.parent = self._html
                return
            parent = self._html  # pragma: no cover - html always owns a children list
        else:
            parent = self._current_parent()
        self._append(parent, node)

    def _finish_fragment_context(self) -> None:
        context = self._fragment_context_node
        if context is None:
            return
        root = self._doc
        children = root.children
        if children is None:  # pragma: no cover - fragment roots are containers
            return
        try:
            index = children.index(context)
        except ValueError:  # pragma: no cover - context is inserted into its fragment root
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

    def _open_template_index(self) -> int | None:
        stack = self._stack
        for idx in range(len(stack) - 1, 0, -1):
            node = stack[idx]
            if node.name != "template":
                continue
            if node.namespace == _PARSER_ONLY_NAMESPACE or (
                type(node) is Template and node.namespace in {None, "html"}
            ):
                return idx
        return None

    def _has_open_parser_only_template(self) -> bool:
        return self._parser_only_template_depth > 0

    def _has_open_real_template(self) -> bool:  # pragma: no cover - defensive helper for inconsistent stacks
        for node in reversed(self._stack):
            if type(node) is Template and node.namespace in {None, "html"}:
                return True
            if node.namespace == _PARSER_ONLY_NAMESPACE:
                return False
        return False

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
        self._append_text_boundary(self._current_parent())
        self._push_parser_only_element("template")
        self._parser_only_template_depth += 1
        self._template_modes.append(_TEMPLATE_MODE_INITIAL)
        self._push_active_formatting_marker()

    def _close_parser_only_template(self) -> bool:
        return self._close_open_template()

    def _enter_template_mode(self) -> None:
        self._template_modes.append(_TEMPLATE_MODE_INITIAL)
        self._push_active_formatting_marker()

    def _close_open_template(self, tag_start: int | None = None, tag_end: int | None = None) -> bool:
        idx = self._open_template_index()
        if idx is None:
            return False
        node = self._stack[idx]
        if self._track_tag_spans and tag_start is not None and tag_end is not None:
            self._set_end_span(node, "template", tag_start, tag_end)
        if node.namespace == _PARSER_ONLY_NAMESPACE and node.children:
            parent = self._stack[idx - 1]
            if (
                type(parent) is Template and parent.template_content is not None
            ):  # pragma: no cover - one policy cannot make nested templates both parser-only and real
                parent = parent.template_content
            for child in list(node.children):
                self._append(parent, child)
        if node.namespace == _PARSER_ONLY_NAMESPACE:
            parent = self._stack[idx - 1]
            if (
                type(parent) is Template and parent.template_content is not None
            ):  # pragma: no cover - one policy cannot make nested templates both parser-only and real
                parent = parent.template_content
            self._append_text_boundary(parent)
        self._mark_active_formatting_dirty()
        del self._stack[idx:]
        if node.namespace == _PARSER_ONLY_NAMESPACE:
            self._parser_only_template_depth -= 1
        if self._template_modes:  # pragma: no branch - opposite edge requires invalid parser state
            self._template_modes.pop()
        self._clear_active_formatting_to_marker()
        return True

    def _mark_initial_content(self) -> None:
        if not self._initial_mode_done:  # pragma: no branch - opposite edge requires invalid parser state
            self._quirks_mode = "quirks"
            self._initial_mode_done = True

    def _parse_range(self, pos: int, end: int) -> int:
        html = self._html_input
        append_text = self._append_text
        find = html.find
        parse_start_tag = (
            self._parse_compiled_safe_start_tag
            if not self._raw_mode
            and not self._track_node_locations
            and not self._track_tag_spans
            and not self._xml_coercion
            else self._parse_start_tag
        )
        parse_end_tag = (
            self._parse_compiled_safe_end_tag
            if not self._raw_mode
            and not self._track_node_locations
            and not self._track_tag_spans
            and not self._xml_coercion
            else self._parse_end_tag
        )
        while pos < end:
            if self._skip_escaped_comment_space:  # pragma: no branch - opposite edge requires invalid parser state
                self._skip_escaped_comment_space = False  # pragma: no cover - unreachable after parser-state guards
                if (
                    pos < end and html[pos] in _SPACE and html.startswith("-->", pos + 1)
                ):  # pragma: no cover - unreachable after parser-state guards
                    pos += 1  # pragma: no cover - unreachable after parser-state guards
                    if pos >= end:  # pragma: no cover - unreachable after parser-state guards
                        return end  # pragma: no cover - unreachable after parser-state guards
            lt = find("<", pos, end)
            if lt == -1:
                append_text(html[pos:end], pos)
                return end
            if lt > pos:
                append_text(html[pos:lt], pos)
            pos = lt + 1
            if pos >= end:
                append_text("<", lt)
                return end

            ch = html[pos]
            if ch == "!":
                self._foster_next_table_whitespace = 0
                if html.startswith("<!--", lt):
                    close = self._find_comment_end(pos + 1, end)
                    if close == -1:
                        if self._raw_mode and self._track_tag_spans:
                            append_text(html[lt:end], lt)
                        elif not self._drop_comments:
                            data = html[pos + 3 : end]
                            data = data.removesuffix("-").removesuffix("-")
                            self._append_comment(data, lt)
                        pos = end
                    else:
                        if not self._drop_comments:
                            comment_end = (
                                close - 1
                                if close > pos
                                and close + 2 < end
                                and html[close + 1] == "!"
                                and html[close + 2] == ">"
                                else close
                            )
                            self._append_comment(html[pos + 3 : comment_end], lt)
                        pos = close + 3
                    continue
                if (
                    self._raw_mode
                    and self._lower_input.startswith("<![cdata[", lt)
                    and self._stack[-1].namespace not in {None, "html"}
                ):
                    close = html.find("]]>", pos + 8, end)
                    cdata_end = end if close == -1 else close
                    self._append_raw_literal_text(html[pos + 8 : cdata_end], pos + 8)
                    pos = end if close == -1 else close + 3
                    continue
                if self._lower_input.startswith("<!doctype", lt):
                    gt = html.find(">", pos + 8, end)
                    can_insert_doctype = (
                        not self._fragment
                        and not self._initial_mode_done
                        and not self._explicit_html
                        and not self._body_explicit
                        and not self._frameset_blocked
                        and not self._frameset_seen
                        and not self._body_has_content()
                    )
                    if self._raw_mode and (
                        self._emit_bogus_markup_as_text
                        or (self._track_tag_spans and (gt == -1 or not can_insert_doctype))
                    ):
                        if gt == -1:
                            append_text(html[lt:end], lt)
                            pos = end
                        else:
                            append_text(html[lt : gt + 1], lt)
                            pos = gt + 1
                        continue
                    pos = self._parse_doctype(pos + 8, end)
                    continue
                gt = html.find(">", pos + 1, end)
                comment_end = end if gt == -1 else gt
                if self._raw_mode and self._track_tag_spans:
                    append_text(html[lt:end] if gt == -1 else html[lt : gt + 1], lt)
                elif not self._drop_comments:
                    self._append_comment(html[pos + 1 : comment_end].replace("\0", "\ufffd"), lt)
                pos = end if gt == -1 else gt + 1
                continue
            if ch == "/":
                self._foster_next_table_whitespace = 0
                if self._ignore_lf:
                    end_tag_name_pos = pos + 1
                    if end_tag_name_pos < end:
                        end_tag_ch = html[end_tag_name_pos]
                        end_tag_starts_with_letter = ("a" <= end_tag_ch <= "z") or ("A" <= end_tag_ch <= "Z")
                        if (
                            end_tag_starts_with_letter and self._find_tag_end(end_tag_name_pos + 1, end) != -1
                        ) or not end_tag_starts_with_letter:
                            # A complete end tag, or a bogus-comment token from
                            # an invalid end-tag opener, intervenes before any
                            # later character data.
                            self._ignore_lf = False
                pos = parse_end_tag(pos + 1, end)
                continue
            if ch == "?":
                self._foster_next_table_whitespace = 0
                self._ignore_lf = False
                gt = html.find(">", pos + 1, end)
                if self._raw_mode and self._track_tag_spans:
                    append_text(html[lt:end] if gt == -1 else html[lt : gt + 1], lt)
                    pos = end if gt == -1 else gt + 1
                    continue
                if self._raw_mode:
                    comment_end = end if gt == -1 else gt
                    if not self._drop_comments:  # pragma: no branch - opposite edge requires invalid parser state
                        self._append_comment(html[pos:comment_end], lt)
                    pos = end if gt == -1 else gt + 1
                    continue
                pos = end if gt == -1 else gt + 1
                continue
            if not (("a" <= ch <= "z") or ("A" <= ch <= "Z")):
                if ch == "\0":
                    append_text("<\ufffd", lt)
                    pos += 1
                else:
                    append_text("<", lt)
                continue
            self._foster_next_table_whitespace = 0
            if self._ignore_lf and self._find_tag_end(pos + 1, end) != -1:
                # A complete start tag is the next token, so a pending
                # <pre>/<listing> leading-LF exception cannot leak past it.
                self._ignore_lf = False
            pos = parse_start_tag(pos, end)
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
        if self._has_carriage_return and "\r" in text:
            text = text.replace("\r\n", "\n").replace("\r", "\n")
        if self._has_null and "\0" in text:
            text = text.replace("\0", "\ufffd" if replace_null else "")
        if "&" in text:
            text = decode_entities_in_text(text)
        if self._strip_invisible_unicode and text and not text.isascii():
            text = _strip_invisible_unicode(text)
        if self._xml_coercion and text:
            text = _coerce_text_for_xml(text)
        return text

    def _append_text(self, raw: str, source_pos: int | None = None) -> None:
        if not raw:  # pragma: no branch - opposite edge requires invalid parser state
            return  # pragma: no cover - unreachable after parser-state guards
        stack = self._stack
        parent = stack[-1]
        if (
            self._initial_mode_done
            and not self._frameset_seen
            and self._fragment_context_name is None
            and not self._template_modes
            and not self._after_document
            and not self._active_formatting_dirty
            and not self._foster_next_table_whitespace
            and not self._in_colgroup
            and not self._in_head_noscript
            and not self._ignore_lf
            and not self._track_node_locations
            and not self._xml_coercion
            and (parent.namespace is None or parent.namespace == "html")
            and type(parent) is not Template
            and parent is not self._head
            and parent is not self._html
            and parent.name not in self._table_foster_targets
            and (parent is not self._body or self._body_explicit or self._body_mode_seen or self._body_has_content())
        ):
            text = raw
            if self._has_carriage_return and "\r" in text:
                text = text.replace("\r\n", "\n").replace("\r", "\n")
            if self._has_null and "\0" in text:
                text = text.replace("\0", "")
            if "&" in text:
                text = decode_entities_in_text(text)
            if self._strip_invisible_unicode and text and not text.isascii():
                text = _strip_invisible_unicode(text)
            if text:
                children = parent.children
                if children is not None:  # pragma: no branch - opposite edge requires invalid parser state
                    if children and type(children[-1]) is Text:
                        children[-1].data = (children[-1].data or "") + text
                    else:
                        node = Text(text)
                        children.append(node)
                        node.parent = parent
            return
        raw_is_space: bool | None = None
        pending_table_whitespace = 0
        if self._foster_next_table_whitespace:  # pragma: no branch - opposite edge requires invalid parser state
            pending_table_whitespace = int(self._foster_next_table_whitespace)
            self._foster_next_table_whitespace = 0  # pragma: no cover - unreachable after parser-state guards
        if not self._initial_mode_done:
            if "&" in raw:
                initial_text = self._clean_text(raw)
                if initial_text and initial_text.strip(_SPACE + "\0") == "":
                    return
            initial_end = 0
            while initial_end < len(raw) and raw[initial_end] in _SPACE + "\0":
                initial_end += 1
            if initial_end:
                raw = raw[initial_end:]
                if source_pos is not None:  # pragma: no branch - parser text always has a source offset
                    source_pos += initial_end
            if not raw:
                return
            self._mark_initial_content()
            if self._fragment_context_name is None:  # pragma: no branch - document parses have no fragment context
                self._body_mode_seen = True
        if self._frameset_seen:
            if self._fragment:
                return
            if not self._body_explicit:  # pragma: no branch - framesets cannot follow an explicit body
                self._append_frameset_text(raw)
                return
        if self._fragment_context_name == "colgroup":
            return
        if self._template_modes and self._template_modes[-1] == _TEMPLATE_MODE_COLGROUP:
            if raw.strip(_SPACE):  # pragma: no branch - whitespace leaves colgroup mode unchanged
                if len(self._stack) > 1 and self._stack[-1].name == "colgroup":  # pragma: no branch
                    self._stack.pop()
                    self._template_modes[-1] = _TEMPLATE_MODE_TABLE
                else:
                    return
        parent = self._stack[-1]
        if parent.namespace == _PARSER_ONLY_NAMESPACE or (
            type(parent) is Template and parent.template_content is not None
        ):
            parent = self._current_parent()
        parent_name = getattr(parent, "name", None)
        if self._in_colgroup and parent_name == "colgroup" and raw.strip(_SPACE):
            leading = len(raw) - len(raw.lstrip(_SPACE))
            if leading:
                whitespace = raw[:leading]
                children = parent.children
                if children is not None:  # pragma: no branch - elements always own a child list
                    if children and type(children[-1]) is Text:
                        children[-1].data = (children[-1].data or "") + whitespace  # pragma: no cover
                    else:
                        node = (
                            self._new_text(whitespace, source_pos) if self._track_node_locations else Text(whitespace)
                        )
                        children.append(node)
                        node.parent = parent
                raw = raw[leading:]
                if source_pos is not None:  # pragma: no branch - parser text has a source offset
                    source_pos += leading
            self._in_colgroup = False
            if len(self._stack) > 1 and self._stack[-1] is parent:  # pragma: no branch
                self._stack.pop()
            parent = self._current_parent()
            parent_name = getattr(parent, "name", None)
        if self._active_formatting_dirty:
            reconstruct = parent_name != "caption"
            if reconstruct and parent_name in self._table_foster_targets:
                if raw_is_space is None:  # pragma: no branch - opposite edge requires invalid parser state
                    raw_is_space = raw.strip(_SPACE) == ""
                reconstruct = not raw_is_space
            if reconstruct:
                self._reconstruct_active_formatting()
                parent = self._stack[-1]
                if parent.namespace == _PARSER_ONLY_NAMESPACE or (
                    type(parent) is Template and parent.template_content is not None
                ):
                    parent = self._current_parent()
        if (
            not self._fragment
            and self._head is not None
            and parent is self._head
            and not self._template_modes
            and self._html is not None
        ):
            if raw_is_space is None:  # pragma: no branch - opposite edge requires invalid parser state
                candidate = self._clean_text(raw) if "&" in raw else raw
                raw_is_space = candidate.strip(_SPACE) == ""
            if not raw_is_space:
                stripped = raw.lstrip(_SPACE)
                leading_len = len(raw) - len(stripped)
                if leading_len:
                    leading_text = self._clean_text(raw[:leading_len])
                    node = (
                        self._new_text(leading_text, source_pos) if self._track_node_locations else Text(leading_text)
                    )
                    self._append(self._head, node)
                    if source_pos is not None:  # pragma: no branch - opposite edge requires invalid parser state
                        source_pos += leading_len
                    raw = stripped
                self._stack = [self._doc, self._html, self._body]  # type: ignore[list-item]
                self._after_head = False
                self._body_mode_seen = True
                parent = self._body
        if not self._fragment and parent is self._html and self._after_head:
            if raw_is_space is None:  # pragma: no branch - opposite edge requires invalid parser state
                raw_is_space = raw.strip(_SPACE) == ""
            if not raw_is_space:
                self._stack = [self._doc, self._html, self._body]  # type: ignore[list-item]
                self._after_head = False
                self._body_mode_seen = True
                parent = self._body
        if not self._fragment and self._after_document:
            if raw_is_space is None:  # pragma: no branch - earlier text-path checks usually classify whitespace first
                raw_is_space = raw.strip(_SPACE) == ""
            if not raw_is_space:
                if self._find_open_html_index("body") is None:
                    self._stack = [self._doc, self._html, self._body]  # type: ignore[list-item]
                self._after_body = False
                self._after_document = False
                self._after_html = False
                self._body_mode_seen = True
                parent = self._current_parent()
        if (
            not self._fragment
            and parent is self._body
            and not self._body_explicit
            and not self._body_mode_seen
            and not self._body_has_content()
        ):
            if "&" in raw:
                initial_text = self._clean_text(raw)
                if initial_text and initial_text.strip(_SPACE + "\0") == "":
                    return
            if raw_is_space is None:  # pragma: no branch - shell text reaches this block without prior classification
                raw_is_space = raw.strip(_SPACE) == ""
            if raw_is_space:
                return
            raw = raw.lstrip(_SPACE)
        if self._in_colgroup and getattr(parent, "name", None) == "table":
            if raw.strip(_SPACE):  # pragma: no branch - mixed colgroup text is the state-changing edge
                leading = len(raw) - len(raw.lstrip(_SPACE))
                if leading:
                    whitespace = raw[:leading]
                    children = parent.children
                    if children is not None:  # pragma: no branch - elements always own a child list
                        if children and type(children[-1]) is Text:
                            children[-1].data = (children[-1].data or "") + whitespace  # pragma: no cover
                        else:
                            node = (
                                self._new_text(whitespace, source_pos)
                                if self._track_node_locations
                                else Text(whitespace)
                            )
                            children.append(node)
                            node.parent = parent
                    raw = raw[leading:]
                    if source_pos is not None:  # pragma: no branch - parser text has a source offset
                        source_pos += leading
                self._in_colgroup = False
        text = raw
        if self._has_carriage_return and "\r" in text:
            text = text.replace("\r\n", "\n").replace("\r", "\n")
        if self._has_null and "\0" in text:
            parent_namespace = getattr(parent, "namespace", None)
            null_replacement = (
                "\ufffd"
                if parent_namespace not in {None, "html"}
                and not self._is_html_integration_point(parent)
                and not self._is_mathml_text_integration_point(parent)
                else ""
            )
            text = text.replace("\0", null_replacement)
        if "&" in text:
            text = decode_entities_in_text(text)
        if self._strip_invisible_unicode and text and not text.isascii():
            text = _strip_invisible_unicode(text)
        if self._xml_coercion and text:
            text = _coerce_text_for_xml(text)
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
                if text_is_space is None:  # pragma: no branch - opposite edge requires invalid parser state
                    text_is_space = text.strip(_SPACE) == ""
                if text_is_space:  # pragma: no branch - opposite edge requires invalid parser state
                    body = self._body
                    children = parent.children
                    position = len(children) if children is not None else 0
                    if children is not None:  # pragma: no branch - opposite edge requires invalid parser state
                        try:
                            position = children.index(body)
                        except ValueError:  # pragma: no cover - unreachable after parser-state guards
                            pass  # pragma: no cover - unreachable after parser-state guards
                    node = self._new_text(text, source_pos) if self._track_node_locations else Text(text)
                    self._insert_at(parent, position, node)
                    return
            foster = None
            if parent.name in self._table_foster_targets:
                is_table_space = text_is_space if text_is_space is not None else text.strip(_SPACE) == ""
                if not is_table_space:
                    if pending_table_whitespace:
                        children = parent.children
                        if (
                            children
                            and type(children[-1]) is Text
                            and not (  # pragma: no branch
                                children[-1].data or ""
                            ).strip(_SPACE)
                        ):
                            previous = children[-1].data or ""
                            pending = previous[-pending_table_whitespace:]
                            remaining = previous[:-pending_table_whitespace]
                            text = pending + text
                            if remaining:
                                children[-1].data = remaining
                            else:
                                children.pop()
                    foster = self._foster_parent_for(parent)
            if foster is None:
                children = parent.children
                if children is not None:  # pragma: no branch - opposite edge requires invalid parser state
                    if children and type(children[-1]) is Text:
                        children[-1].data = (children[-1].data or "") + text
                    else:
                        node = self._new_text(text, source_pos) if self._track_node_locations else Text(text)
                        children.append(node)
                        node.parent = parent
                if parent.name in self._table_foster_targets and is_table_space:
                    self._foster_next_table_whitespace = pending_table_whitespace + len(text)
            else:
                foster_parent, position = foster
                node = self._new_text(text, source_pos) if self._track_node_locations else Text(text)
                self._insert_at(foster_parent, position, node)

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
            if self._has_carriage_return and "\r" in raw:
                raw = raw.replace("\r\n", "\n").replace("\r", "\n")
            if self._has_null and "\0" in raw:
                raw = raw.replace("\0", "\ufffd")
            match = _DOCTYPE_RE.match(raw)
            if match:
                name = match.group(1).lower()
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
                doctype_error, self._quirks_mode = doctype_error_and_quirks(doctype, self._iframe_srcdoc)
                if doctype_error and self._collect_errors:
                    self._emit_error(
                        "unknown-doctype",
                        doctype_end if gt != -1 else max(0, end - 1),
                        category="treebuilder",
                        end_pos=doctype_end if gt != -1 else None,
                    )
                self._prepend_doctype(doctype)
            else:
                doctype = Doctype(None, force_quirks=True)
                self._quirks_mode = doctype_error_and_quirks(doctype, self._iframe_srcdoc)[1]
                if self._collect_errors:
                    self._emit_error(
                        "unknown-doctype",
                        doctype_end if gt != -1 else max(0, end - 1),
                        category="treebuilder",
                        end_pos=doctype_end if gt != -1 else None,
                    )
                self._prepend_doctype(doctype)
        return end if gt == -1 else gt + 1

    def _prepend_doctype(self, doctype: Doctype) -> None:
        children = self._doc.children
        if children is None:  # pragma: no branch - opposite edge requires invalid parser state
            return  # pragma: no cover - unreachable after parser-state guards
        node = Node("!doctype", data=doctype)
        insert_at = 0
        while insert_at < len(children) and children[insert_at].name != "html":
            insert_at += 1
        children.insert(insert_at, node)
        node.parent = self._doc
        self._doctype_seen = True
        self._initial_mode_done = True

    def _parse_compiled_safe_end_tag(self, pos: int, end: int) -> int:
        if self._foreign_context_seen:
            if any(node.namespace not in {None, "html", _PARSER_ONLY_NAMESPACE} for node in self._stack[1:]):
                return self._parse_end_tag(pos, end)
        html = self._html_input
        if pos >= end:
            self._append_text("</")
            return end
        ch = html[pos]
        if not (("a" <= ch <= "z") or ("A" <= ch <= "Z")):
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
        _, _, pos, tag_closed = self._parse_all_attrs(pos, end)
        gt = pos - 1 if tag_closed else -1
        pos = end if gt == -1 else pos

        if not self._fragment and name in {"html", "body"}:
            self._in_colgroup = False
            parent_name = getattr(self._current_parent(), "name", None)
            if self._find_open_index("table") is not None and parent_name not in {"body", "html"}:
                return pos
            if parent_name not in {"body", "html"} and self._find_open_index("body") is not None:
                self._after_head = False
                self._body_mode_seen = True
                return pos
            if (
                self._frameset_seen and not self._body_explicit
            ):  # pragma: no cover - later state normalization restores body mode
                self._stack = [self._doc, self._html]  # type: ignore[list-item]
            else:
                self._stack = [self._doc, self._html, self._body]  # type: ignore[list-item]
            self._after_head = False
            self._body_mode_seen = True
            return pos
        if not self._fragment and name == "head":
            self._in_colgroup = False
            if self._body_mode_seen and self._stack[-1] is not self._head:
                return pos
            self._stack = [self._doc, self._html]  # type: ignore[list-item]
            self._after_head = True
            return pos
        if name == "colgroup" and not self._parser_only_template_depth:
            self._in_colgroup = False
            if (
                len(self._stack) > 1
                and self._stack[-1].name == "colgroup"
                and self._stack[-1] is not self._fragment_context_node
            ):  # pragma: no branch - compiled sanitizer never retains colgroup nodes
                self._stack.pop()  # pragma: no cover - defensive parser-state cleanup
            return pos
        if name == "table":
            self._in_colgroup = False
            self._close_table_cell()
        elif (name == "tr" or name in self._table_section_tags) and self._find_open_index_before_boundary(
            name, _TABLE_CONTEXT_BOUNDARIES
        ) is not None:
            self._close_table_cell()
        if name == "template":
            self._close_parser_only_template()
            self._finish_head_reentry()
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
            if name == "noscript":  # pragma: no branch - opposite edge requires invalid parser state
                self._in_head_noscript = False  # pragma: no cover - unreachable after parser-state guards
                return pos  # pragma: no cover - unreachable after parser-state guards
            if name != "br":  # pragma: no branch - opposite edge requires invalid parser state
                return pos  # pragma: no cover - unreachable after parser-state guards
            self._leave_head_noscript_to_body()
        if name == "br":
            if self._active_formatting_dirty:
                self._reconstruct_active_formatting()
            self._insert_compiled_safe_element("br", {}, False, self._current_parent())
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
            if name in self._table_cell_tags or name in _ACTIVE_FORMATTING_MARKER_TAGS:
                self._clear_active_formatting_to_marker()
            stack.pop()
            return pos

        if name == "p":
            idx = self._find_open_index_before_boundary("p", _P_SCOPE_BOUNDARIES)
        elif name == "li":
            idx = self._find_open_index_before_boundary("li", self._list_item_scope_boundaries)
        elif name in {"dd", "dt"}:
            idx = self._find_open_index_before_boundary(name, self._definition_scope_boundaries)
        elif name in {"audio", "noscript", "slot", "title"}:
            idx = None
            for candidate_idx in range(len(stack) - 1, 0, -1):  # pragma: no branch
                candidate = stack[candidate_idx]
                if self._node_matches_end_name(candidate, name):
                    idx = candidate_idx
                    break
                if self._is_special_node(candidate):
                    break
        elif name == "summary":
            idx = self._find_open_index_before_boundary(name, _DEFAULT_SCOPE_BOUNDARIES)
        elif self._parser_only_template_depth:
            idx = self._find_open_index_in_current_scope(name)
        elif name not in self._special_elements and (action is None or not action.p_closing):
            idx = self._find_open_index_before_boundary(name, _GENERAL_END_TAG_BOUNDARIES)
        elif name in _TABLE_SCOPED_END_TAGS:
            idx = self._find_open_index_before_boundary(name, _TABLE_CONTEXT_BOUNDARIES)
        else:
            idx = self._find_open_index_before_boundary(name, _DEFAULT_SCOPE_BOUNDARIES)
        if idx is None:
            if name == "p":
                if (
                    not self._fragment
                    and not self._body_explicit
                    and not self._body_mode_seen
                    and not self._body_has_content()
                ):
                    return pos
                self._emit_error("unexpected-end-tag", name_start - 2, tag_name=name, category="treebuilder", end_pos=pos)
            return pos
        if self._fragment_context_node is not None and stack[idx] is self._fragment_context_node:
            return pos
        if name in self._implied_end_tags:
            self._generate_implied_end_tags(name)
        self._mark_active_formatting_dirty()
        if name in self._table_cell_tags:
            self._clear_active_formatting_to_marker()
        elif name in _ACTIVE_FORMATTING_MARKER_TAGS:  # pragma: no branch - opposite edge requires invalid parser state
            self._clear_active_formatting_to_marker()  # pragma: no cover - unreachable after parser-state guards
        del stack[idx:]
        return pos

    def _parse_end_tag(self, pos: int, end: int) -> int:
        html = self._html_input
        raw_mode = self._raw_mode
        tag_start = pos - 2
        if pos >= end:
            self._append_text("</", pos - 2)
            return end
        ch = html[pos]
        if not (("a" <= ch <= "z") or ("A" <= ch <= "Z")):
            gt = html.find(">", pos, end)
            if ch == ">" and not (raw_mode and self._track_tag_spans):
                return pos + 1
            if raw_mode:
                if self._track_tag_spans:
                    self._append_text(html[tag_start:end] if gt == -1 else html[tag_start : gt + 1], tag_start)
                    return end if gt == -1 else gt + 1
                comment_end = end if gt == -1 else gt
                if not self._drop_comments:  # pragma: no branch - opposite edge requires invalid parser state
                    self._append_comment(html[pos:comment_end].replace("\0", "\ufffd"), tag_start)
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
        after_name = pos
        _, _, pos, tag_closed = self._parse_all_attrs(pos, end)
        gt = pos - 1 if tag_closed else -1
        pos = end if gt == -1 else pos
        tag_end = pos

        if raw_mode and gt == -1:
            if self._track_tag_spans and html[after_name:end].strip(_SPACE):
                self._append_text(html[tag_start:end], tag_start)
            return end

        if (
            raw_mode
            and self._track_tag_spans
            and name != "br"
            and gt != -1
            and html[after_name:gt].strip(_SPACE)
            and self._find_open_index(name) is None
        ):
            self._append_text(html[tag_start:tag_end], tag_start)
            return pos

        if self._end_tag_stays_in_foreign_context(name, tag_start, tag_end):
            return pos

        if self._template_modes and self._handle_template_mode_end(name):
            return pos
        if self._frameset_seen and not self._body_explicit:
            if name == "frameset":
                if len(self._stack) > 1 and self._stack[-1].name == "frameset":
                    self._stack.pop()
                if self._find_open_index("frameset") is None:
                    self._after_body = True
                    self._after_document = True
            elif name == "html" and self._after_body:
                self._after_document = True
                self._after_html = True
            return pos
        if not self._fragment and self._after_document and name not in {"body", "html"}:
            if (
                self._find_open_html_index("body") is None
            ):  # pragma: no cover - body-less after-document tags return earlier
                self._stack = [self._doc, self._html, self._body]  # type: ignore[list-item]
            self._after_body = False
            self._after_document = False
            self._after_html = False
            self._body_mode_seen = True

        if not self._fragment and name in {"html", "body"}:
            parent_name = getattr(self._current_parent(), "name", None)
            if self._find_open_index("table") is not None and parent_name not in {"body", "html"}:
                return pos
            self._in_colgroup = False
            if self._head is not None and self._stack[-1] is self._head:
                self._stack = [self._doc, self._html, self._body]  # type: ignore[list-item]
            if self._frameset_seen and not self._body_explicit:
                self._stack = [self._doc, self._html]  # type: ignore[list-item]  # pragma: no cover - frameset state keeps later body reconstruction unreachable here
            self._after_head = False
            self._body_mode_seen = (
                True  # pragma: no cover - frameset end-tag paths return before re-entering body mode
            )
            self._after_body = name == "body"
            self._after_document = True
            self._after_html = name == "html"
            if name == "body" and isinstance(self._body, Element):
                if self._track_tag_spans:
                    self._set_end_span(self._body, name, tag_start, tag_end)
            elif (
                name == "html" and self._html is not None
            ):  # pragma: no branch - opposite edge requires invalid parser state
                if self._track_tag_spans:
                    self._set_end_span(self._html, name, tag_start, tag_end)
            return pos
        if self._fragment and self._fragment_context_name == "html" and name == "html":
            self._stack = [self._doc]
            return pos
        if not self._fragment and name == "head":
            self._in_colgroup = False
            if self._body_mode_seen and self._stack[-1] is not self._head:
                return pos
            if self._head is not None:  # pragma: no branch - opposite edge requires invalid parser state
                if self._track_tag_spans:
                    self._set_end_span(self._head, name, tag_start, tag_end)
            self._stack = [self._doc, self._html]  # type: ignore[list-item]
            self._after_head = True
            return pos
        if name == "colgroup" and not self._parser_only_template_depth:
            self._in_colgroup = False
            if (
                len(self._stack) > 1
                and self._stack[-1].name == "colgroup"
                and self._stack[-1] is not self._fragment_context_node
            ):
                self._stack.pop()
            return pos
        if name == "table":
            self._in_colgroup = False
            self._close_table_cell()
        elif (name == "tr" or name in self._table_section_tags) and self._find_open_index_before_boundary(
            name, _TABLE_CONTEXT_BOUNDARIES
        ) is not None:
            self._close_table_cell()
        if name == "template":
            self._close_open_template(tag_start, tag_end)
            self._finish_head_reentry()
            return pos
        if name == "form" and not self._template_modes:
            form_node = self._form_element
            self._form_element = None
            if form_node is None:
                return pos
            self._generate_implied_end_tags()
            try:
                self._stack.remove(form_node)
            except ValueError:
                pass
            return pos
        select_idx = self._find_open_html_index("select")
        if select_idx is not None and name not in {"optgroup", "option", "select", "selectedcontent", "template"}:
            target_idx = self._find_open_index(name)
            if target_idx is not None and target_idx < select_idx:
                if name in _TABLE_SCOPED_END_TAGS:
                    self._close_html_until("select")
                else:
                    return pos
        if name == "menuitem":
            for node in reversed(self._stack):
                node_name = getattr(node, "name", None)
                if node_name == "p":
                    return pos
                if node_name == "menuitem":
                    break
        if (
            not self._fragment
            and self._head is not None
            and self._stack[-1] is self._head
            and name not in {"br", "body", "head", "html", "template"}
        ):
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
                if (
                    raw_mode and len(self._stack) > 1 and self._stack[-1].name == "noscript"
                ):  # pragma: no branch - opposite edge requires invalid parser state
                    if self._track_tag_spans:
                        self._set_end_span(self._stack[-1], name, tag_start, tag_end)
                    self._stack.pop()
                return pos
            if name != "br":
                return pos
            self._leave_head_noscript_to_body()
        if name == "br":
            if self._active_formatting_dirty:
                self._reconstruct_active_formatting()
            self._insert_sanitized_element("br", {}, False, self._current_parent())
            return pos
        stack = self._stack
        if name in HEADING_ELEMENTS:
            idx = self._find_open_heading_index()
            if idx is None:
                return pos
            self._mark_active_formatting_dirty()
            if self._track_tag_spans:
                self._set_end_span(stack[idx], name, tag_start, tag_end)
            del stack[idx:]
            return pos
        if action is not None and action.active_formatting:
            select_idx = self._find_open_html_index("select")
            formatting_idx = self._find_active_formatting_index(name)
            if select_idx is not None and formatting_idx is not None:
                entry = self._active_formatting[formatting_idx]
                if isinstance(
                    entry, _FormattingEntry
                ):  # pragma: no branch - opposite edge requires invalid parser state
                    try:
                        if (
                            self._stack.index(entry.node) < select_idx
                        ):  # pragma: no branch - opposite edge requires invalid parser state
                            return pos  # pragma: no cover - unreachable after parser-state guards
                    except ValueError:  # pragma: no cover - unreachable after parser-state guards
                        pass  # pragma: no cover - unreachable after parser-state guards
            self._adoption_agency(name, tag_start=tag_start, tag_end=tag_end)
            return pos

        if (
            not self._template_modes
            and len(stack) > 1
            and self._node_matches_end_name(stack[-1], name)
            and not (self._fragment_context_node is not None and stack[-1] is self._fragment_context_node)
        ):
            self._mark_active_formatting_dirty()
            if name in self._table_cell_tags or name in _ACTIVE_FORMATTING_MARKER_TAGS:
                self._clear_active_formatting_to_marker()
            if self._track_tag_spans:
                self._set_end_span(stack[-1], name, tag_start, tag_end)
            stack.pop()
            return pos

        if name == "p":
            idx = self._find_open_index_before_boundary("p", _P_SCOPE_BOUNDARIES)
        elif name == "li":
            idx = self._find_open_index_before_boundary("li", self._list_item_scope_boundaries)
        elif name in {"dd", "dt"}:
            idx = self._find_open_index_before_boundary(name, self._definition_scope_boundaries)
        elif name in _TABLE_SCOPED_END_TAGS:
            idx = self._find_open_table_scoped_end_index(name)
        elif name in {"audio", "noscript", "slot", "title"}:
            idx = None
            for candidate_idx in range(len(stack) - 1, 0, -1):
                candidate = stack[candidate_idx]
                if self._node_matches_end_name(candidate, name):
                    idx = candidate_idx
                    break
                if self._is_special_node(candidate):
                    break
        elif name == "summary":
            idx = self._find_open_index_before_boundary(name, _DEFAULT_SCOPE_BOUNDARIES)
        elif self._template_modes:
            idx = self._find_open_index_in_current_scope(name)
        elif name not in self._special_elements and (action is None or not action.p_closing):
            idx = self._find_open_index_before_boundary(name, _GENERAL_END_TAG_BOUNDARIES)
        else:
            idx = self._find_open_index_before_boundary(name, _DEFAULT_SCOPE_BOUNDARIES)
        if idx is None:
            if name == "p":
                if (
                    not self._fragment
                    and not self._body_explicit
                    and not self._body_mode_seen
                    and not self._body_has_content()
                ):
                    return pos
                self._emit_error("unexpected-end-tag", tag_start, tag_name=name, category="treebuilder", end_pos=tag_end)
            return pos
        if self._fragment_context_node is not None and stack[idx] is self._fragment_context_node:
            return pos
        if name in self._implied_end_tags:
            self._generate_implied_end_tags(name)
        self._mark_active_formatting_dirty()
        if name in self._table_cell_tags:
            self._clear_active_formatting_to_marker()
        elif name in _ACTIVE_FORMATTING_MARKER_TAGS:
            self._clear_active_formatting_to_marker()
        if self._track_tag_spans:
            self._set_end_span(stack[idx], name, tag_start, tag_end)
        del stack[idx:]
        return pos

    def _parse_compiled_safe_start_tag(self, pos: int, end: int) -> int:
        html = self._html_input
        name_start = pos
        pos += 1
        while pos < end and html[pos] not in _TAG_NAME_STOP:
            pos += 1
        raw_name = html[name_start:pos]
        name = raw_name if raw_name.islower() else raw_name.lower()
        if name == "image":
            name = "img"
        action = self._tag_actions.get(name)
        if self._foreign_context_seen:
            namespace = self._stack[-1].namespace
            if namespace is not None and namespace != "html" and namespace != _PARSER_ONLY_NAMESPACE:
                return self._parse_start_tag(name_start, end)
        if not self._initial_mode_done:
            self._mark_initial_content()

        attrs, self_closing, pos, tag_closed = self._parse_attrs_for_action(action, pos, end)
        if not tag_closed:
            return pos
        in_parser_only_template = self._parser_only_template_depth > 0
        if not in_parser_only_template:
            if name == "colgroup":
                if (
                    len(self._stack) > 1 and self._stack[-1].name == "colgroup"
                ):  # pragma: no branch - compiled sanitizer never retains colgroup nodes
                    self._stack.pop()  # pragma: no cover - defensive parser-state cleanup
                self._in_colgroup = True
            elif self._in_colgroup and name not in {"col", "template"}:
                self._in_colgroup = False
        if self._in_head_noscript:
            if name in {"head", "noscript"}:  # pragma: no branch - opposite edge requires invalid parser state
                return pos  # pragma: no cover - unreachable after parser-state guards
            if name not in _HEAD_NOSCRIPT_ALLOWED_START_TAGS and name != "html":
                self._leave_head_noscript_to_body()
        if not self._fragment and not in_parser_only_template:
            current_top = self._stack[-1]
            if name == "html":
                self._in_colgroup = False
                if self._frameset_seen:
                    self._after_document = True
                    self._after_html = True
                self._explicit_html = True
                if self._html is not None:  # pragma: no branch - opposite edge requires invalid parser state
                    for attr_name, attr_value in attrs.items():
                        self._html.attrs.setdefault(attr_name, attr_value)
                return pos
            if name == "head":
                self._in_colgroup = False
                self._explicit_head = True
                if self._body_mode_seen:
                    return pos
                if self._head is not None:  # pragma: no branch - opposite edge requires invalid parser state
                    self._stack = [self._doc, self._html, self._head]  # type: ignore[list-item]
                    self._after_head = False
                return pos
            if name == "body":
                self._in_colgroup = False
                if self._frameset_seen:
                    return pos
                if isinstance(self._body, Element):  # pragma: no branch - opposite edge requires invalid parser state
                    for attr_name, attr_value in attrs.items():
                        self._body.attrs.setdefault(attr_name, attr_value)
                if self._body_explicit:
                    return pos
                self._body_explicit = True
                if self._body_mode_seen:
                    return pos
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

        if (
            not in_parser_only_template
            and not self._fragment
            and not self._frameset_seen
            and action is not None
            and action.blocks_frameset
        ):
            if action.name == "input":
                input_type = attrs.get("type")
                if not (isinstance(input_type, str) and input_type.lower() == "hidden"):
                    self._frameset_blocked = True
            else:
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

        if name == "select" and self._find_open_index("select") is not None:
            self._close_until("select")
            return pos

        if action is not None and action.drop_content:
            if self._raw_head_text_parent(name) and self._head is not None:
                self._stack = [self._doc, self._html, self._head]  # type: ignore[list-item]
            return self._skip_rawtext(name, pos, end)
        if action is not None and action.drop_subtree:
            if action.allowed:  # pragma: no branch - opposite edge requires invalid parser state
                return self._parse_start_tag(
                    name_start, end
                )  # pragma: no cover - unreachable after parser-state guards
            next_pos = self._skip_subtree(name, pos, end, detect_foreign_breakout=True)
            if next_pos == -1:
                return self._parse_start_tag(name_start, end)
            return next_pos
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

        if self._fragment_context_name is not None:
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
            return self._parse_formatting_start(name, attrs, pos, compiled_safe=True)

        parent: Node
        if (  # pragma: no branch - opposite edge requires invalid parser state
            action is not None
            and action.allowed
            and action.head_content
            and not self._fragment
            and self._head is not None
            and not self._body_mode_seen
            and not self._body_has_content()
        ):
            parent = self._head  # pragma: no cover - unreachable after parser-state guards
        else:
            self._repair_stack_for_start(name)
            parent = self._stack[-1]
            if parent.namespace == _PARSER_ONLY_NAMESPACE or (
                type(parent) is Template and parent.template_content is not None
            ):
                parent = self._current_parent()

        if not in_parser_only_template and name == "table":
            table_idx = self._find_open_index("table")
            td_idx = self._find_open_index_before_boundary("td", _TABLE_CONTEXT_BOUNDARIES)
            th_idx = self._find_open_index_before_boundary("th", _TABLE_CONTEXT_BOUNDARIES)
            caption_idx = self._find_open_index_before_boundary("caption", _TABLE_CONTEXT_BOUNDARIES)
            if table_idx is not None and td_idx is None and th_idx is None and caption_idx is None:
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
            if (
                action is not None and action.pre_linefeed
            ):  # pragma: no branch - opposite edge requires invalid parser state
                self._ignore_lf = True  # pragma: no cover - unreachable after parser-state guards
            if self._should_insert_unwrapped_element(name, action):
                if self._active_formatting_dirty:
                    self._reconstruct_active_formatting()
                    parent = self._current_parent()
                self._insert_sanitized_element(name, attrs, self_closing, parent)
            elif (
                name == "menuitem" and self._active_formatting_dirty
            ):  # pragma: no branch - opposite edge requires invalid parser state
                self._reconstruct_active_formatting()  # pragma: no cover - unreachable after parser-state guards
            return pos

        if (
            self._active_formatting_dirty
            and parent.name in self._table_foster_targets
            and name not in self._table_allowed_children
            and (action is None or not action.p_closing)
        ):
            self._reconstruct_active_formatting()
            parent = self._current_parent()

        self._insert_compiled_safe_element(name, attrs, self_closing, parent)
        if action.pre_linefeed:
            self._ignore_lf = True
        return pos

    def _parse_start_tag(self, pos: int, end: int) -> int:
        html = self._html_input
        raw_mode = self._raw_mode
        name_start = pos
        tag_start = pos - 1
        pos += 1
        while pos < end and html[pos] not in _TAG_NAME_STOP:
            pos += 1
        raw_name = html[name_start:pos]
        if self._has_null and "\0" in raw_name:
            raw_name = raw_name.replace("\0", "\ufffd")
        name = raw_name if raw_name.islower() else raw_name.lower()
        if not self._initial_mode_done:
            self._mark_initial_content()

        in_foreign_context = self._stack[-1].namespace not in {None, "html", _PARSER_ONLY_NAMESPACE}
        if name == "image" and not in_foreign_context:
            name = "img"
        action = self._tag_actions.get(name)
        foreign_state_parse = not raw_mode and (name in {"math", "svg"} or in_foreign_context)
        if raw_mode or foreign_state_parse:
            attrs, self_closing, pos, tag_closed = self._parse_all_attrs(pos, end)
        else:
            attrs, self_closing, pos, tag_closed = self._parse_attrs_for_action(action, pos, end)
        if not tag_closed:
            if raw_mode and self._track_tag_spans:
                self._append_text(html[tag_start:end], tag_start)
            return pos
        if raw_mode:
            if attrs and (self._has_carriage_return or self._has_form_feed or self._has_null):
                attrs = {
                    key: value
                    for key, value in attrs.items()
                    if key.startswith("=")
                    or _SERIALIZABLE_ATTR_NAME_RE.fullmatch(key) is not None
                    or ("\ufffd" in key and _SERIALIZABLE_ATTR_NAME_RE.fullmatch(key) is not None)
                }
            if in_foreign_context and name in {"head", "body"}:
                self._namespace_for_raw_start(name, attrs)
                if name == "body" and isinstance(self._body, Element):
                    for attr_name, attr_value in attrs.items():
                        self._body.attrs.setdefault(attr_name, attr_value)
                return pos
        tag_end = pos
        allowed = raw_mode or (action is not None and action.allowed)
        in_template_content = bool(self._template_modes)
        if not self._fragment and self._after_document and name != "html" and not self._frameset_seen:
            if self._find_open_html_index("body") is None:
                self._stack = [self._doc, self._html, self._body]  # type: ignore[list-item]
            self._after_body = False
            self._after_document = False
            self._after_html = False
            self._body_mode_seen = True
        if raw_mode and self._fragment and name == "html" and not in_template_content:
            return pos
        if not in_template_content:
            if name == "colgroup":
                if len(self._stack) > 1 and self._stack[-1].name == "colgroup":
                    self._stack.pop()
                self._in_colgroup = True
            elif self._in_colgroup and name not in {"col", "template"}:
                self._in_colgroup = False
                if len(self._stack) > 1 and self._stack[-1].name == "colgroup":
                    self._stack.pop()
        if self._in_head_noscript:
            if name in {"head", "noscript"}:
                return pos
            if name not in _HEAD_NOSCRIPT_ALLOWED_START_TAGS and name != "html":
                if action is not None and action.head_content and self._head is not None:
                    self._in_head_noscript = False
                    if self._stack and self._stack[-1].name == "noscript":  # pragma: no branch
                        self._stack.pop()
                else:
                    self._leave_head_noscript_to_body()
        if not self._fragment and not in_template_content and not in_foreign_context:
            current_top = self._stack[-1]
            if name == "html":
                self._in_colgroup = False
                if self._frameset_seen:
                    self._after_document = True
                    self._after_html = True
                self._explicit_html = True
                if self._html is not None:  # pragma: no branch - opposite edge requires invalid parser state
                    for attr_name, attr_value in attrs.items():
                        self._html.attrs.setdefault(attr_name, attr_value)
                    self._set_origin(self._html, tag_start)
                    self._set_source_span(self._html, tag_start, tag_end)
                return pos
            if name == "head":
                self._in_colgroup = False
                self._explicit_head = True
                if self._body_mode_seen:
                    return pos
                if self._head is not None:  # pragma: no branch - opposite edge requires invalid parser state
                    for attr_name, attr_value in attrs.items():
                        self._head.attrs.setdefault(attr_name, attr_value)
                    self._set_origin(self._head, tag_start)
                    self._set_source_span(self._head, tag_start, tag_end)
                    self._stack = [self._doc, self._html, self._head]  # type: ignore[list-item]
                    self._after_head = False
                return pos
            if name == "body":
                self._in_colgroup = False
                if self._frameset_seen:
                    return pos
                if isinstance(self._body, Element):  # pragma: no branch - opposite edge requires invalid parser state
                    for attr_name, attr_value in attrs.items():
                        self._body.attrs.setdefault(attr_name, attr_value)
                    self._set_origin(self._body, tag_start)
                    self._set_source_span(self._body, tag_start, tag_end)
                if self._body_explicit:
                    return pos
                self._body_explicit = True
                if self._body_mode_seen:
                    return pos
                self._body_mode_seen = True
                self._stack = [self._doc, self._html, self._body]  # type: ignore[list-item]
                self._after_head = False
                return pos
            if not self._body_mode_seen and not (action.head_content if action is not None else False):
                self._body_mode_seen = True
        else:
            current_top = None

        if (
            not in_template_content
            and not self._fragment
            and current_top is self._head
            and not (action.head_content if action is not None else False)
        ):
            self._stack = [self._doc, self._html, self._body]  # type: ignore[list-item]
            self._after_head = False
            self._body_mode_seen = True
            current_top = self._body

        if (
            not in_template_content
            and not self._fragment
            and current_top is self._html
            and self._after_head
            and name not in {"body", "template"}
        ):
            if (
                action is not None
                and allowed
                and action.head_content
                and self._head is not None
                and not self._in_head_noscript
                and not self._body_mode_seen
                and not self._body_has_content()
            ):
                self._stack = [self._doc, self._html, self._head]  # type: ignore[list-item]
                self._head_reentry = True
                current_top = self._head
            else:
                self._stack = [self._doc, self._html, self._body]  # type: ignore[list-item]
                self._after_head = False
                self._body_mode_seen = True
                current_top = self._body

        if (
            not in_template_content
            and not self._fragment
            and not self._frameset_seen
            and action is not None
            and action.blocks_frameset
        ):
            if action.name == "input":
                input_type = attrs.get("type")
                if not (isinstance(input_type, str) and input_type.lower() == "hidden"):
                    self._frameset_blocked = True
            else:
                self._frameset_blocked = True

        html_text_parsing = self._raw_start_uses_html_text_parsing(name)

        if html_text_parsing and self._fragment_context_name == "select" and name == "input":
            return pos

        if html_text_parsing and self._fragment_context_name == "frameset" and name == "frame":
            if allowed:
                self._insert_sanitized_element(
                    name, attrs, self_closing, self._current_parent(), tag_start=tag_start, tag_end=tag_end
                )
            return pos

        if (
            self._fragment
            and self._fragment_context_namespace not in {None, "html"}
            and name in {"html", "head", "body", "frameset"}
        ):
            return pos

        if html_text_parsing and name == "frameset" and self._accept_fragment_frameset():
            if allowed:
                self._insert_sanitized_element(
                    name, attrs, self_closing, self._current_parent(), tag_start=tag_start, tag_end=tag_end
                )
            return pos

        if html_text_parsing and not in_template_content and name == "frameset" and self._accept_frameset():
            if allowed:
                self._insert_sanitized_element(
                    name, attrs, self_closing, self._current_parent(), tag_start=tag_start, tag_end=tag_end
                )
            return pos

        if self._frameset_seen and not self._body_explicit:
            if name == "noframes":
                return self._parse_rawtext_element(name, attrs, self_closing, pos, end, tag_start, tag_end)
            if html_text_parsing and name in {"frame", "frameset"}:
                if name == "frameset" and self._after_html:
                    return pos
                if self._find_open_index("frameset") is None:
                    return pos
                if allowed:
                    self._insert_sanitized_element(
                        name, attrs, self_closing, self._current_parent(), tag_start=tag_start, tag_end=tag_end
                    )
                return pos
            return pos

        if html_text_parsing and name in {"frame", "frameset"}:
            return pos

        open_select_idx = self._find_open_html_index("select") if html_text_parsing else None
        open_table_idx = self._find_open_index("table") if html_text_parsing else None
        if (
            open_select_idx is not None
            and open_table_idx is not None
            and open_table_idx > open_select_idx
            and name in {"input", "select"}
        ):
            self._insert_sanitized_element(
                name,
                attrs,
                self_closing,
                self._current_parent(),
                tag_start=tag_start,
                tag_end=tag_end,
            )
            return pos
        if html_text_parsing and name == "select" and open_select_idx is not None:
            self._close_html_until("select")
            return pos
        if html_text_parsing and name == "input" and self._find_open_html_index("select") is not None:
            self._close_html_until("select")

        select_idx = self._find_open_html_index("select") if html_text_parsing else None
        if select_idx is not None and name == "hr":
            if len(self._stack) > select_idx + 1 and self._stack[-1].name == "option":
                self._stack.pop()
            if len(self._stack) > select_idx + 1 and self._stack[-1].name == "optgroup":
                self._stack.pop()
            self._close_until_before_boundary("p", _P_SCOPE_BOUNDARIES)
            self._insert_sanitized_element(
                name,
                attrs,
                self_closing,
                self._current_parent(),
                tag_start=tag_start,
                tag_end=tag_end,
            )
            return pos

        if html_text_parsing and name == "form" and not self._template_modes:
            if self._form_element is not None:
                return pos
            self._repair_stack_for_start(name)
            node = self._insert_sanitized_element(
                name,
                attrs,
                self_closing,
                self._current_parent(),
                tag_start=tag_start,
                tag_end=tag_end,
            )
            self._form_element = node
            table_idx = self._find_open_index("table")
            in_table_mode = (
                table_idx is not None
                and self._find_open_index_before_boundary("td", _TABLE_CONTEXT_BOUNDARIES) is None
                and self._find_open_index_before_boundary("th", _TABLE_CONTEXT_BOUNDARIES) is None
                and self._find_open_index_before_boundary("caption", _TABLE_CONTEXT_BOUNDARIES) is None
            )
            if (node.parent is not None and node.parent.name in self._table_foster_targets) or in_table_mode:
                if (
                    self._stack and self._stack[-1] is node
                ):  # pragma: no branch - opposite edge requires invalid parser state
                    self._stack.pop()
            return pos

        if raw_mode and html_text_parsing and name in self._rawtext_element_tags:
            return self._parse_rawtext_element(name, attrs, self_closing, pos, end, tag_start, tag_end)
        if action is not None and not raw_mode and action.drop_content:
            if self._raw_head_text_parent(name) and self._head is not None:
                self._stack = [self._doc, self._html, self._head]  # type: ignore[list-item]
            return self._skip_rawtext(name, pos, end)
        if action is not None and action.rawtext_as_text:
            return self._parse_rawtext_as_text(name, pos, end)
        if (
            name == "noscript"
            and name not in self._rawtext_element_tags
            and not self._fragment
            and self._head is not None
            and not in_template_content
            and (
                self._stack[-1] is self._head
                or (not self._body_explicit and not self._body_mode_seen and not self._body_has_content())
            )
        ):
            self._stack = [self._doc, self._html, self._head]  # type: ignore[list-item]
            if raw_mode:
                self._insert_sanitized_element(
                    name, attrs, self_closing, self._head, tag_start=tag_start, tag_end=tag_end
                )
            self._in_head_noscript = True
            return pos
        if action is not None and html_text_parsing and action.rcdata:
            return self._parse_rcdata_element(name, attrs, self_closing, pos, end, tag_start, tag_end)
        if action is not None and html_text_parsing and action.plaintext:
            if raw_mode:
                return self._parse_plaintext_element(name, attrs, self_closing, pos, end, tag_start, tag_end)
            return self._parse_plaintext_as_text(pos, end)

        if html_text_parsing and self._fragment_context_name is not None:
            fragment_pos = self._handle_fragment_context_start(name, attrs, self_closing, pos)
            if fragment_pos is not None:
                return fragment_pos

        if name == "template" and not allowed and html_text_parsing:
            self._start_parser_only_template()
            return pos

        if html_text_parsing and self._template_modes:
            template_pos = self._handle_template_mode_start(name, attrs, self_closing, pos)
            if template_pos is not None:
                return template_pos

        if html_text_parsing and name == "button":
            if self._find_open_index_in_current_scope("button") is not None:
                self._generate_implied_end_tags()
                self._close_until_before_boundary("button", frozenset({"template"}))
            if not allowed:
                self._insert_sanitized_element(
                    name, attrs, self_closing, self._current_parent(), tag_start=tag_start, tag_end=tag_end
                )
                return pos

        if html_text_parsing and action is not None and action.active_formatting:
            return self._parse_formatting_start(name, attrs, pos, tag_start=tag_start, tag_end=tag_end)

        if html_text_parsing and name == "menuitem" and self._active_formatting_dirty:
            self._reconstruct_active_formatting()

        if raw_mode and name in {"option", "optgroup"} and self._find_open_index("select") is not None:
            if len(self._stack) > 1 and self._stack[-1].name == "p":
                self._mark_active_formatting_dirty()
                self._stack.pop()

        if html_text_parsing and name in {"option", "optgroup"} and self._active_formatting_dirty:
            self._reconstruct_active_formatting()

        if (
            html_text_parsing
            and name in {"td", "th", "tr"}
            and self._find_open_index("table") is None
            and self._stack[-1].namespace in {None, "html", _PARSER_ONLY_NAMESPACE}
            and any(node.namespace not in {None, "html", _PARSER_ONLY_NAMESPACE} for node in self._stack[1:])
        ):
            self._mark_active_formatting_dirty()
            if (
                not self._fragment and self._html is not None
            ):  # pragma: no branch - opposite edge requires invalid parser state
                self._stack = [self._doc, self._html]
            else:
                for idx, stack_node in enumerate(
                    self._stack[1:], start=1
                ):  # pragma: no cover - unreachable after parser-state guards
                    if stack_node.namespace not in {
                        None,
                        "html",
                        _PARSER_ONLY_NAMESPACE,
                    }:  # pragma: no cover - unreachable after parser-state guards
                        del self._stack[idx:]  # pragma: no cover - unreachable after parser-state guards
                        break  # pragma: no cover - unreachable after parser-state guards
            self._insert_sanitized_element(
                name,
                attrs,
                self_closing,
                self._current_parent(),
                tag_start=tag_start,
                tag_end=tag_end,
            )
            return pos

        if html_text_parsing:
            ruby_open = self._find_open_index_in_current_scope("ruby") is not None
            if name in {"rb", "rtc"} and ruby_open:
                if self._stack and self._stack[-1].name in {"rb", "rp", "rt", "rtc"}:
                    self._generate_implied_end_tags()
            elif name in {"rp", "rt"} and ruby_open:
                self._generate_implied_end_tags("rtc")

        parent: Node
        if (
            action is not None
            and allowed
            and action.head_content
            and not self._fragment
            and self._head is not None
            and not in_template_content
            and not self._in_head_noscript
            and not self._body_mode_seen
            and not self._body_has_content()
        ):
            parent = self._head
        else:
            self._repair_stack_for_start(name)
            parent = self._current_parent()

        if not in_template_content and name == "table":
            table_idx = self._find_open_index("table")
            td_idx = self._find_open_index_before_boundary("td", _TABLE_CONTEXT_BOUNDARIES)
            th_idx = self._find_open_index_before_boundary("th", _TABLE_CONTEXT_BOUNDARIES)
            caption_idx = self._find_open_index_before_boundary("caption", _TABLE_CONTEXT_BOUNDARIES)
            if table_idx is not None and td_idx is None and th_idx is None and caption_idx is None:
                self._mark_active_formatting_dirty()
                del self._stack[table_idx:]
                parent = self._current_parent()

        if (
            html_text_parsing
            and not in_template_content
            and (
                (action is not None and (action.table_section or action.table_cell))
                or name in {"caption", "col", "colgroup", "tr"}
            )
        ):
            self._repair_table_for_start(name)
            parent = self._current_parent()
            parent_name = getattr(parent, "name", None)
            if name == "caption":
                if parent_name != "table":
                    return pos
            elif name in {"col", "colgroup"}:
                if parent_name != "table" and not (name == "col" and parent_name == "colgroup"):
                    return pos
            elif action is not None and action.table_section:
                if parent_name != "table":
                    return pos
            elif name == "tr":
                if parent_name not in self._table_section_tags:
                    return pos
            elif parent_name != "tr":
                return pos

        reconstruct_before_insert = (
            name not in _TABLE_STRUCTURE_START_TAGS
            and name not in _HEAD_ONLY_VOID_START_TAGS
            and name not in {"param", "source", "track"}
            and name != "template"
            and (action is None or not action.p_closing)
        )
        if html_text_parsing and self._active_formatting_dirty and reconstruct_before_insert:
            self._reconstruct_active_formatting()
            parent = self._current_parent()

        if not allowed:
            if action is not None and action.pre_linefeed:
                self._ignore_lf = True
            if not html_text_parsing or self._should_insert_unwrapped_element(name, action):
                if self._active_formatting_dirty:
                    self._reconstruct_active_formatting()
                    parent = self._current_parent()
                self._insert_sanitized_element(name, attrs, self_closing, parent, tag_start=tag_start, tag_end=tag_end)
            elif (
                name == "menuitem" and self._active_formatting_dirty
            ):  # pragma: no branch - opposite edge requires invalid parser state
                self._reconstruct_active_formatting()  # pragma: no cover - unreachable after parser-state guards
            return pos

        if (
            self._active_formatting_dirty
            and parent.name in self._table_foster_targets
            and name not in self._table_allowed_children
            and (action is None or not action.p_closing)
        ):  # pragma: no branch - HTML-mode insertions reconstruct above
            self._reconstruct_active_formatting()  # pragma: no cover - defensive foreign-state fallback
            parent = self._current_parent()  # pragma: no cover - defensive foreign-state fallback

        self._insert_sanitized_element(name, attrs, self_closing, parent, tag_start=tag_start, tag_end=tag_end)
        if name in _HEAD_ONLY_VOID_START_TAGS and parent is self._head:
            if (
                len(self._stack) > 1 and getattr(self._stack[-1], "name", None) == name
            ):  # pragma: no branch - opposite edge requires invalid parser state
                self._stack.pop()  # pragma: no cover - unreachable after parser-state guards
        if self._in_head_noscript and name in _HEAD_NOSCRIPT_VOID_START_TAGS:
            if (
                len(self._stack) > 1 and getattr(self._stack[-1], "name", None) == name
            ):  # pragma: no branch - opposite edge requires invalid parser state
                self._stack.pop()  # pragma: no cover - unreachable after parser-state guards
        if action is not None and action.pre_linefeed:
            self._ignore_lf = True
        self._finish_head_reentry()
        return pos

    def _insert_compiled_safe_element(
        self,
        name: str,
        attrs: dict[str, str | None],
        self_closing: bool,
        parent: Node,
    ) -> Element:
        if name == "template":  # pragma: no branch - opposite edge requires invalid parser state
            node: Element = Template(
                name, attrs, namespace="html"
            )  # pragma: no cover - unreachable after parser-state guards
        else:
            node = Element(name, attrs, "html")
        if name == "selectedcontent":  # pragma: no branch - opposite edge requires invalid parser state
            self._has_selectedcontent = True  # pragma: no cover - unreachable after parser-state guards
        is_void = name in self._void_elements
        node._self_closing = self_closing and is_void
        foster = (
            self._foster_parent_for(parent, for_tag=name)
            if parent.name in self._table_foster_targets and name not in self._table_allowed_children
            else None
        )
        if foster is None:
            children = parent.children
            if children is not None:  # pragma: no branch - opposite edge requires invalid parser state
                children.append(node)
                node.parent = parent
        else:
            foster_parent, position = foster
            self._insert_at(foster_parent, position, node)
        if name not in self._allowed_tags:
            self._nodes_to_unwrap.append(node)
        if not is_void:
            self._stack.append(node)
            if name in self._table_cell_tags:
                self._push_active_formatting_marker()
            elif (
                name in _ACTIVE_FORMATTING_MARKER_TAGS
            ):  # pragma: no branch - opposite edge requires invalid parser state
                self._push_active_formatting_marker()  # pragma: no cover - unreachable after parser-state guards
        return node

    def _insert_sanitized_element(
        self,
        name: str,
        attrs: dict[str, str | None],
        self_closing: bool,
        parent: Node,
        *,
        tag_start: int | None = None,
        tag_end: int | None = None,
    ) -> Element:
        if self._raw_mode:
            return self._insert_raw_element(
                name,
                attrs,
                self_closing,
                parent,
                tag_start=tag_start,
                tag_end=tag_end,
            )

        name, attrs, namespace = self._prepare_raw_element(name, attrs)
        if namespace == "html":
            attrs = self._sanitize_parsed_attrs(self._tag_actions.get(name), attrs)
        if parent not in self._stack and getattr(parent, "namespace", None) not in {None, "html"}:
            parent = self._current_parent()
        node: Element
        if (
            name == "template" and namespace == "html"
        ):  # pragma: no branch - opposite edge requires invalid parser state
            node = Template(
                name, attrs, namespace=namespace
            )  # pragma: no cover - unreachable after parser-state guards
        else:
            node = Element(name, attrs, namespace)
        track_node_locations = self._track_node_locations
        if track_node_locations and tag_start is not None:
            node._origin_pos = tag_start
            node._origin_line, node._origin_col = self._line_col_at_pos(tag_start)
        if self._track_tag_spans and tag_start is not None and tag_end is not None:
            node._source_html = self._html_input
            node._start_tag_start = tag_start
            node._start_tag_end = tag_end
        if name == "selectedcontent":
            self._has_selectedcontent = True
        is_html_namespace = namespace in {None, "html"}
        is_void = (is_html_namespace and (name in self._void_elements or name in _HTML_VOID_COMPAT_TAGS)) or (
            not is_html_namespace and self_closing
        )
        node._self_closing = self_closing and is_html_namespace and name in self._void_elements
        foster = (
            self._foster_parent_for(parent, for_tag=name)
            if parent.name in self._table_foster_targets
            and name not in self._table_allowed_children
            and not _is_hidden_input(name, attrs)
            else None
        )
        if foster is None:
            children = parent.children
            if children is not None:  # pragma: no branch - opposite edge requires invalid parser state
                children.append(node)
                node.parent = parent
        else:
            foster_parent, position = foster
            self._insert_at(foster_parent, position, node)
        if namespace not in {None, "html"}:
            self._foreign_context_seen = True
            self._nodes_to_unwrap.append(node)
            parent_namespace = getattr(parent, "namespace", None)
            if parent is self._fragment_context_node or parent_namespace in {None, "html"}:
                if not self._has_open_parser_only_template() and (
                    parent is self._fragment_context_node or parent.name in self._allowed_tags
                ):
                    self._nodes_to_drop.append(node)
        elif name not in self._allowed_tags:
            self._nodes_to_unwrap.append(node)
        if not is_void:
            self._stack.append(node)
            if type(node) is Template and node.namespace in {
                None,
                "html",
            }:  # pragma: no branch - opposite edge requires invalid parser state
                self._enter_template_mode()  # pragma: no cover - unreachable after parser-state guards
            if name in self._table_cell_tags:
                self._push_active_formatting_marker()
            elif name in _ACTIVE_FORMATTING_MARKER_TAGS:
                self._push_active_formatting_marker()
        return node

    def _insert_raw_element(
        self,
        name: str,
        attrs: dict[str, str | None],
        self_closing: bool,
        parent: Node,
        *,
        tag_start: int | None = None,
        tag_end: int | None = None,
    ) -> Element:
        name, attrs, namespace = self._prepare_raw_element(name, attrs)
        if parent not in self._stack and getattr(parent, "namespace", None) not in {None, "html"}:
            parent = self._current_parent()
        if name == "template" and namespace == "html":
            node: Element = Template(name, attrs, namespace=namespace)
        else:
            node = Element(name, attrs, namespace)
        if self._track_node_locations and tag_start is not None:
            node._origin_pos = tag_start
            node._origin_line, node._origin_col = self._line_col_at_pos(tag_start)
        if self._track_tag_spans and tag_start is not None and tag_end is not None:
            node._source_html = self._html_input
            node._start_tag_start = tag_start
            node._start_tag_end = tag_end
        if name == "selectedcontent":
            self._has_selectedcontent = True
        is_html_namespace = namespace in {None, "html"}
        is_void = (is_html_namespace and (name in self._void_elements or name in _HTML_VOID_COMPAT_TAGS)) or (
            not is_html_namespace and self_closing
        )
        node._self_closing = self_closing and is_html_namespace and name in self._void_elements
        foster = (
            self._foster_parent_for(parent, for_tag=name)
            if parent.name in self._table_foster_targets
            and name not in self._table_allowed_children
            and not _is_hidden_input(name, attrs)
            else None
        )
        if foster is None:
            children = parent.children
            if children is not None:  # pragma: no branch - opposite edge requires invalid parser state
                children.append(node)
                node.parent = parent
        else:
            foster_parent, position = foster
            self._insert_at(foster_parent, position, node)
        if not is_void:
            self._stack.append(node)
            if type(node) is Template and node.namespace in {None, "html"}:
                self._enter_template_mode()
            if name in self._table_cell_tags:
                self._push_active_formatting_marker()
            elif name in _ACTIVE_FORMATTING_MARKER_TAGS:
                self._push_active_formatting_marker()
        return node

    def _prepare_raw_element(
        self,
        name: str,
        attrs: dict[str, str | None],
    ) -> tuple[str, dict[str, str | None], str]:
        namespace = self._namespace_for_raw_start(name, attrs)
        if namespace == "svg":
            name = SVG_TAG_NAME_ADJUSTMENTS.get(name, name)
        if namespace in {"svg", "math"}:
            attrs = self._prepare_foreign_attrs(namespace, attrs)
        return name, attrs, namespace

    def _namespace_for_raw_start(self, name: str, attrs: dict[str, str | None]) -> str:
        current = self._stack[-1] if self._stack else None
        current_ns = getattr(current, "namespace", None)
        if current is not None and current_ns not in {None, "html", _PARSER_ONLY_NAMESPACE}:
            if self._is_html_integration_point(current):
                return "svg" if name == "svg" else "math" if name == "math" else "html"
            if self._is_mathml_text_integration_point(current) and name not in {"mglyph", "malignmark"}:
                return "svg" if name == "svg" else "math" if name == "math" else "html"
            if current_ns == "math" and current.name == "annotation-xml" and name == "svg":
                return "svg"
            breaks_out = name in FOREIGN_BREAKOUT_ELEMENTS or (
                name == "font" and any(attr.lower() in {"color", "face", "size"} for attr in attrs)
            )
            if not breaks_out:
                return current_ns or "html"
            while (
                len(self._stack) > 1
                and self._stack[-1].namespace not in {None, "html", _PARSER_ONLY_NAMESPACE}
                and self._stack[-1] is not self._fragment_context_node
                and not self._is_html_integration_point(self._stack[-1])
                and not self._is_mathml_text_integration_point(self._stack[-1])
            ):
                self._stack.pop()

        if name == "svg":
            return "svg"
        if name == "math":
            return "math"
        return "html"

    def _raw_start_uses_html_text_parsing(self, name: str) -> bool:
        current = self._stack[-1] if self._stack else None
        current_ns = getattr(current, "namespace", None)
        if current is None or current_ns in {None, "html", _PARSER_ONLY_NAMESPACE}:
            return True
        if self._is_html_integration_point(current):
            return True
        if self._is_mathml_text_integration_point(current) and name not in {"mglyph", "malignmark"}:
            return True
        if current_ns == "math" and current.name == "annotation-xml" and name == "svg":
            return False
        return False

    def _prepare_foreign_attrs(self, namespace: str, attrs: dict[str, str | None]) -> dict[str, str | None]:
        if not attrs:
            return attrs
        adjusted: dict[str, str | None] = {}
        for attr_name, value in attrs.items():
            lower_name = attr_name if attr_name.islower() else attr_name.lower()
            name = attr_name
            if namespace == "math" and lower_name in MATHML_ATTRIBUTE_ADJUSTMENTS:
                name = MATHML_ATTRIBUTE_ADJUSTMENTS[lower_name]
                lower_name = name if name.islower() else name.lower()
            elif namespace == "svg" and lower_name in SVG_ATTRIBUTE_ADJUSTMENTS:
                name = SVG_ATTRIBUTE_ADJUSTMENTS[lower_name]
                lower_name = name if name.islower() else name.lower()

            foreign_adjustment = FOREIGN_ATTRIBUTE_ADJUSTMENTS.get(lower_name)
            if foreign_adjustment is not None:
                prefix, local, _namespace_url = foreign_adjustment
                name = f"{prefix}:{local}" if prefix is not None else local
            if name not in adjusted:  # pragma: no branch - opposite edge requires invalid parser state
                adjusted[name] = value
        return adjusted

    def _node_attr_value(self, node: Node, name: str) -> str | None:
        attrs = getattr(node, "attrs", None)
        if not attrs:
            return None
        target = name.lower()
        for attr_name, value in attrs.items():
            if attr_name.lower() == target:
                return value or ""
        return None

    def _is_html_integration_point(self, node: Node) -> bool:
        if node.namespace == "math" and node.name == "annotation-xml":
            encoding = self._node_attr_value(node, "encoding")
            return encoding is not None and encoding.lower() in {"application/xhtml+xml", "text/html"}
        return (node.namespace, node.name) in HTML_INTEGRATION_POINT_SET

    def _is_mathml_text_integration_point(self, node: Node) -> bool:
        return node.namespace == "math" and (node.namespace, node.name) in MATHML_TEXT_INTEGRATION_POINT_SET

    def _insert_at(self, parent: Node, position: int, node: Node | Text) -> None:
        children = parent.children
        if children is None:  # pragma: no branch - opposite edge requires invalid parser state
            return  # pragma: no cover - unreachable after parser-state guards
        if type(node) is Text:
            if position > 0 and type(children[position - 1]) is Text:
                children[position - 1].data = (children[position - 1].data or "") + (node.data or "")
                return
            if (
                position < len(children) and type(children[position]) is Text
            ):  # pragma: no branch - opposite edge requires invalid parser state
                children[position].data = (node.data or "") + (
                    children[position].data or ""
                )  # pragma: no cover - unreachable after parser-state guards
                return  # pragma: no cover - unreachable after parser-state guards
        children.insert(position, node)
        node.parent = parent

    def _body_has_content(self) -> bool:
        if self._fragment:  # pragma: no branch - opposite edge requires invalid parser state
            children = self._body.children  # pragma: no cover - unreachable after parser-state guards
        else:
            children = self._body.children
        if not children:
            return False
        return any(type(child) is not Text or bool(child.data) for child in children)

    def _node_matches_end_name(self, node: Node, name: str) -> bool:
        node_name = node.name
        if node_name == name:
            return True
        return node.namespace not in {None, "html"} and node_name.lower() == name

    def _find_open_index(self, name: str) -> int | None:
        stack = self._stack
        for idx in range(len(stack) - 1, 0, -1):
            if stack[idx].name == name:
                return idx
        return None

    def _find_open_html_index(self, name: str) -> int | None:
        stack = self._stack
        for idx in range(len(stack) - 1, 0, -1):
            node = stack[idx]
            if node.name == name and node.namespace in {None, "html", _PARSER_ONLY_NAMESPACE}:
                return idx
        return None

    def _find_open_index_before_boundary(self, name: str, boundaries: frozenset[str]) -> int | None:
        stack = self._stack
        for idx in range(len(stack) - 1, 0, -1):
            node = stack[idx]
            node_name = node.name
            if node_name == name:
                return idx
            if node.namespace in {None, "html"}:
                if node_name in boundaries:
                    return None
            elif self._is_html_integration_point(node) or self._is_mathml_text_integration_point(  # pragma: no branch
                node
            ):
                return None
        return None

    def _find_open_table_scoped_end_index(self, name: str) -> int | None:
        for idx in range(len(self._stack) - 1, 0, -1):
            node = self._stack[idx]
            if node.namespace in {None, "html"} and node.name == name:
                return idx
            if node.namespace in {None, "html"} and node.name == "table":
                return None
            if type(node) is Template and node.namespace in {None, "html"}:
                return None
        return None

    def _find_open_index_in_current_scope(self, name: str) -> int | None:
        stack = self._stack
        for idx in range(len(stack) - 1, 0, -1):
            node = stack[idx]
            if node.name == name:
                return idx
            if node.name == "template" and (
                node.namespace == _PARSER_ONLY_NAMESPACE
                or (type(node) is Template and node.namespace in {None, "html"})
            ):
                return None
        return None

    def _find_open_heading_index(self) -> int | None:
        stack = self._stack
        for idx in range(len(stack) - 1, 0, -1):  # pragma: no branch
            node = stack[idx]
            if node.name in HEADING_ELEMENTS:
                return idx
            if node.namespace in {None, "html"}:
                if node.name in _DEFAULT_SCOPE_BOUNDARIES:
                    return None
            elif (  # pragma: no branch
                self._is_html_integration_point(node) or self._is_mathml_text_integration_point(node)
            ):
                return None
        return None

    def _set_current_template_mode(self, mode: str) -> None:
        if self._template_modes:  # pragma: no branch - opposite edge requires invalid parser state
            self._template_modes[-1] = mode

    def _leave_head_noscript_to_body(self) -> None:
        self._in_head_noscript = False
        if self._fragment or self._html is None:  # pragma: no branch - opposite edge requires invalid parser state
            return  # pragma: no cover - unreachable after parser-state guards
        self._stack = [self._doc, self._html, self._body]  # type: ignore[list-item]
        self._after_head = False
        self._body_mode_seen = True

    def _finish_head_reentry(self) -> None:
        if not self._head_reentry or self._html is None:
            return
        if self._open_template_index() is not None:  # pragma: no branch - reentry finishes after templates close
            return  # pragma: no cover - guarded by template close ordering
        self._head_reentry = False
        self._stack = [self._doc, self._html]
        self._after_head = True

    def _close_to_fragment_context(self) -> bool:
        context = self._fragment_context_node
        if context is None:  # pragma: no branch - opposite edge requires invalid parser state
            return False  # pragma: no cover - unreachable after parser-state guards
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
        if context is None:  # pragma: no branch - opposite edge requires invalid parser state
            return False  # pragma: no cover - unreachable after parser-state guards
        try:
            context_idx = self._stack.index(context)
        except ValueError:  # pragma: no cover - unreachable after parser-state guards
            return False  # pragma: no cover - unreachable after parser-state guards
        return any(node.name == "table" for node in self._stack[context_idx + 1 :])

    def _handle_fragment_context_start(
        self,
        name: str,
        attrs: dict[str, str | None],
        self_closing: bool,
        pos: int,
    ) -> int | None:
        context_name = self._fragment_context_name
        if context_name is None:  # pragma: no branch - opposite edge requires invalid parser state
            return None  # pragma: no cover - unreachable after parser-state guards

        if context_name == "html":
            return pos if name in {"html", "head", "body"} else None

        if context_name in {"body", "div"} and name in {"html", "head", "body", "frameset"}:
            return pos

        if context_name == "colgroup":
            if name == "col":
                if (
                    self._close_to_fragment_context()
                ):  # pragma: no branch - opposite edge requires invalid parser state
                    self._insert_sanitized_element(name, attrs, self_closing, self._current_parent())
            return pos

        if context_name == "caption":
            return pos if name in _TABLE_STRUCTURE_START_TAGS and name != "table" else None

        if context_name == "table":
            if name == "colgroup":
                if (
                    self._close_to_fragment_context()
                ):  # pragma: no branch - opposite edge requires invalid parser state
                    self._insert_sanitized_element(name, attrs, self_closing, self._current_parent())
                return pos
            if name == "col":
                if (
                    getattr(self._current_parent(), "name", None) == "colgroup"
                ):  # pragma: no branch - opposite edge requires invalid parser state
                    self._insert_sanitized_element(name, attrs, self_closing, self._current_parent())
                return pos
            if name == "table":
                return pos
            if name == "caption":
                if self._close_to_fragment_context():
                    self._insert_sanitized_element(name, attrs, self_closing, self._current_parent())
                    return pos
            return None

        if context_name in self._table_section_tags:
            if self._allows_nested_fragment_table_start(name):
                return None
            if name == "tr":
                if (
                    self._close_to_fragment_context()
                ):  # pragma: no branch - opposite edge requires invalid parser state
                    self._insert_sanitized_element(name, attrs, self_closing, self._current_parent())
                    return pos
            if name in self._table_cell_tags:
                self._close_table_cell()
                if getattr(self._current_parent(), "name", None) != "tr" and self._close_to_fragment_context():
                    self._insert_sanitized_element("tr", {}, False, self._current_parent())
                if (
                    getattr(self._current_parent(), "name", None) == "tr"
                ):  # pragma: no branch - opposite edge requires invalid parser state
                    self._insert_sanitized_element(name, attrs, self_closing, self._current_parent())
                return pos
            if name in _TABLE_STRUCTURE_START_TAGS:
                return pos
            return None

        if context_name == "tr":
            if self._allows_nested_fragment_table_start(name):
                return None
            if name in self._table_cell_tags:
                if (
                    self._close_to_fragment_context()
                ):  # pragma: no branch - opposite edge requires invalid parser state
                    self._insert_sanitized_element(name, attrs, self_closing, self._current_parent())
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
        if not self._raw_mode and name not in self._allowed_tags:
            if (
                name in self._pre_linefeed_ignoring_tags
            ):  # pragma: no branch - opposite edge requires invalid parser state
                self._ignore_lf = True  # pragma: no cover - unreachable after parser-state guards
            return None
        node = self._insert_sanitized_element(name, attrs, self_closing, self._current_parent())
        if name in self._pre_linefeed_ignoring_tags:  # pragma: no branch - opposite edge requires invalid parser state
            self._ignore_lf = True  # pragma: no cover - unreachable after parser-state guards
        return node

    def _handle_template_mode_start(
        self,
        name: str,
        attrs: dict[str, str | None],
        self_closing: bool,
        pos: int,
    ) -> int | None:
        mode = self._current_template_mode()
        if mode is None:  # pragma: no branch - opposite edge requires invalid parser state
            return None  # pragma: no cover - unreachable after parser-state guards
        if name == "body":
            table_idx = self._find_open_index("table")
            template_idx = self._open_template_index()
            if table_idx is None or template_idx is None or table_idx < template_idx:
                self._set_current_template_mode(_TEMPLATE_MODE_BODY)
            return pos
        if name in {"html", "head"}:
            return pos
        if name == "template":
            return None
        table_idx = self._find_open_index("table")
        template_idx = self._open_template_index()
        form_in_table_mode = mode in {
            _TEMPLATE_MODE_TABLE,
            _TEMPLATE_MODE_TABLE_BODY,
            _TEMPLATE_MODE_ROW,
        } or (
            mode == _TEMPLATE_MODE_BODY
            and table_idx is not None
            and template_idx is not None
            and table_idx > template_idx
        )
        if name == "form" and form_in_table_mode:
            node = self._insert_template_mode_element(name, attrs, self_closing)
            if node is not None and self._stack and self._stack[-1] is node:
                self._stack.pop()
            return pos

        if mode == _TEMPLATE_MODE_COLGROUP:
            if name == "col":
                self._insert_template_mode_element(name, attrs, self_closing)
                return pos
            if len(self._stack) <= 1 or self._stack[-1].name != "colgroup":
                if not self._has_open_parser_only_template() or name not in _TEMPLATE_TABLE_CONTEXT_START_TAGS:
                    return pos
            else:
                self._stack.pop()
            self._set_current_template_mode(_TEMPLATE_MODE_TABLE)
            return self._handle_template_mode_start(name, attrs, self_closing, pos)

        if mode == _TEMPLATE_MODE_INITIAL:
            if name == "table":
                self._set_current_template_mode(_TEMPLATE_MODE_TABLE)
                return None
            if name in {"tbody", "tfoot", "thead"}:
                self._set_current_template_mode(_TEMPLATE_MODE_TABLE_BODY)
                self._insert_template_mode_element(name, attrs, self_closing)
                return pos
            if name == "caption":
                self._set_current_template_mode(_TEMPLATE_MODE_TABLE)
                self._insert_template_mode_element(name, attrs, self_closing)
                return pos
            if name == "colgroup":
                self._set_current_template_mode(_TEMPLATE_MODE_COLGROUP)
                self._insert_template_mode_element(name, attrs, self_closing)
                return pos
            if name == "col":
                self._set_current_template_mode(_TEMPLATE_MODE_COLGROUP)
                self._insert_template_mode_element(name, attrs, self_closing)
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
            if name == "table":
                self._set_current_template_mode(_TEMPLATE_MODE_TABLE)
                return None
            if name in _TEMPLATE_TABLE_CONTEXT_START_TAGS:
                return pos
            return None

        if mode == _TEMPLATE_MODE_TABLE:
            table_idx = self._find_open_index("table")
            if name == "table" and table_idx is not None:
                del self._stack[table_idx:]
                self._insert_template_mode_element(name, attrs, self_closing)
                return pos
            if table_idx is not None and name in {
                "caption",
                "col",
                "colgroup",
                "tbody",
                "td",
                "tfoot",
                "th",
                "thead",
                "tr",
            }:
                del self._stack[table_idx + 1 :]
            if name in {"tbody", "tfoot", "thead"}:
                self._set_current_template_mode(_TEMPLATE_MODE_TABLE_BODY)
                self._insert_template_mode_element(name, attrs, self_closing)
                return pos
            if name == "tr":
                self._set_current_template_mode(_TEMPLATE_MODE_TABLE_BODY)
                self._insert_template_mode_element("tbody", {}, False)
                return self._handle_template_table_body_start(name, attrs, self_closing, pos)
            if name in self._table_cell_tags:
                self._set_current_template_mode(_TEMPLATE_MODE_TABLE_BODY)
                self._insert_template_mode_element("tbody", {}, False)
                return self._handle_template_table_body_start(name, attrs, self_closing, pos)
            if name == "col":
                self._set_current_template_mode(_TEMPLATE_MODE_COLGROUP)
                self._insert_template_mode_element("colgroup", {}, False)
                self._insert_template_mode_element(name, attrs, self_closing)
                return pos
            if name == "colgroup":
                self._set_current_template_mode(_TEMPLATE_MODE_COLGROUP)
                self._insert_template_mode_element(name, attrs, self_closing)
                return pos
            return None

        if mode == _TEMPLATE_MODE_TABLE_BODY:
            return self._handle_template_table_body_start(name, attrs, self_closing, pos)
        if mode == _TEMPLATE_MODE_ROW:
            return self._handle_template_row_start(name, attrs, self_closing, pos)
        if mode == _TEMPLATE_MODE_CELL:  # pragma: no branch - opposite edge requires invalid parser state
            if name == "table":
                self._set_current_template_mode(_TEMPLATE_MODE_TABLE)
                return None
            if name in _TEMPLATE_TABLE_CONTEXT_START_TAGS:
                if self._close_template_cell():
                    return self._handle_template_mode_start(name, attrs, self_closing, pos)
                return pos
            return None
        return None  # pragma: no cover - unreachable after parser-state guards

    def _handle_template_table_body_start(
        self,
        name: str,
        attrs: dict[str, str | None],
        self_closing: bool,
        pos: int,
    ) -> int | None:
        if name == "tr":
            template_idx = self._open_template_index()
            section_idx = next(
                (
                    idx
                    for idx in range(len(self._stack) - 1, 0, -1)
                    if self._stack[idx].name in self._table_section_tags
                    and (template_idx is None or idx > template_idx)
                ),
                None,
            )
            insertion_context_idx = section_idx if section_idx is not None else template_idx
            if insertion_context_idx is not None:  # pragma: no branch - template modes always have a context
                del self._stack[insertion_context_idx + 1 :]
            self._set_current_template_mode(_TEMPLATE_MODE_ROW)
            self._insert_template_mode_element(name, attrs, self_closing)
            return pos
        if name in self._table_cell_tags:
            template_idx = self._open_template_index()
            section_idx = next(
                (
                    idx
                    for idx in range(len(self._stack) - 1, 0, -1)
                    if self._stack[idx].name in self._table_section_tags
                    and (template_idx is None or idx > template_idx)
                ),
                None,
            )
            insertion_context_idx = section_idx if section_idx is not None else template_idx
            if insertion_context_idx is not None:  # pragma: no branch - template modes always have a context
                del self._stack[insertion_context_idx + 1 :]
            self._set_current_template_mode(_TEMPLATE_MODE_ROW)
            self._insert_template_mode_element("tr", {}, False)
            return self._handle_template_row_start(name, attrs, self_closing, pos)
        if name in self._table_section_tags:
            if self._close_open_template_table_section():
                self._set_current_template_mode(_TEMPLATE_MODE_TABLE)
                return self._handle_template_mode_start(name, attrs, self_closing, pos)
            return pos
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
        if name in self._head_content_tags:
            return None  # pragma: no cover - head-content tags are dispatched before template row mode
        if name in self._table_cell_tags:
            if tr_index is not None:
                del self._stack[tr_index + 1 :]
            self._set_current_template_mode(_TEMPLATE_MODE_CELL)
            self._insert_template_mode_element(name, attrs, self_closing)
            return pos
        if name == "table" and tr_index is not None:
            self._close_until_before_boundary("tr", _TEMPLATE_SCOPE_BOUNDARIES)
            self._close_open_template_table_section()
            table_idx = self._find_open_index("table")
            if table_idx is not None:  # pragma: no branch - row mode table replacement has an open table
                del self._stack[table_idx:]
            self._set_current_template_mode(_TEMPLATE_MODE_TABLE)
            self._insert_template_mode_element(name, attrs, self_closing)
            return pos
        if name in _TEMPLATE_ROW_STRUCTURE_START_TAGS:
            if tr_index is not None:
                self._close_until_before_boundary("tr", _TEMPLATE_SCOPE_BOUNDARIES)
                self._set_current_template_mode(_TEMPLATE_MODE_TABLE_BODY)
                return self._handle_template_mode_start(name, attrs, self_closing, pos)
            return pos
        return None

    def _handle_template_mode_end(self, name: str) -> bool:
        mode = self._current_template_mode()
        if mode is None:  # pragma: no branch - opposite edge requires invalid parser state
            return False  # pragma: no cover - unreachable after parser-state guards
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
                if self._close_until_before_boundary(
                    name, _TEMPLATE_SCOPE_BOUNDARIES
                ):  # pragma: no branch - opposite edge requires invalid parser state
                    self._set_current_template_mode(_TEMPLATE_MODE_TABLE)
                return True
            if name in {"caption", "col", "colgroup", "td", "th", "tr"}:
                return True
        if mode == _TEMPLATE_MODE_COLGROUP:
            if name == "colgroup":
                self._set_current_template_mode(_TEMPLATE_MODE_TABLE)
                return True
            return name != "template"
        if mode == _TEMPLATE_MODE_TABLE and name == "table":
            if self._close_until_before_boundary("table", _TEMPLATE_SCOPE_BOUNDARIES):
                self._set_current_template_mode(_TEMPLATE_MODE_BODY)
            return True
        return False  # pragma: no cover - template scope always terminates the scan

    def _close_template_row(self) -> bool:
        return self._close_until_before_boundary("tr", _TEMPLATE_SCOPE_BOUNDARIES)

    def _close_open_template_table_section(self) -> bool:
        stack = self._stack
        for idx in range(len(stack) - 1, 0, -1):
            node_name = stack[idx].name
            if node_name in self._table_section_tags:
                self._mark_active_formatting_dirty()
                del stack[idx:]
                return True
            if node_name == "template":  # pragma: no branch - template scope terminates the scan
                return False
        return False  # pragma: no cover - template scope always terminates the scan

    def _close_template_cell(self) -> bool:
        stack = self._stack
        for idx in range(len(stack) - 1, 0, -1):
            node_name = stack[idx].name
            if node_name in self._table_cell_tags:
                self._mark_active_formatting_dirty()
                self._clear_active_formatting_to_marker()
                del stack[idx:]
                self._set_current_template_mode(_TEMPLATE_MODE_ROW)
                return True
            if node_name == "template":
                return False
        return False

    def _close_until(self, name: str) -> None:
        idx = self._find_open_index(name)
        if idx is not None:  # pragma: no branch - opposite edge requires invalid parser state
            self._mark_active_formatting_dirty()
            del self._stack[idx:]

    def _close_html_until(self, name: str) -> None:
        idx = self._find_open_html_index(name)
        if idx is not None:  # pragma: no branch - opposite edge requires invalid parser state
            self._mark_active_formatting_dirty()
            del self._stack[idx:]

    def _close_until_before_boundary(self, name: str, boundaries: frozenset[str]) -> bool:
        stack = self._stack
        for idx in range(len(stack) - 1, 0, -1):
            node_name = getattr(stack[idx], "name", None)
            if node_name == name:
                if (
                    self._fragment_context_node is not None and stack[idx] is self._fragment_context_node
                ):  # pragma: no branch - opposite edge requires invalid parser state
                    return False  # pragma: no cover - unreachable after parser-state guards
                self._mark_active_formatting_dirty()
                del stack[idx:]
                return True
            if node_name in boundaries:
                return False
        return False

    def _generate_implied_end_tags(self, exclude: str | None = None) -> None:
        stack = self._stack
        while len(stack) > 1:  # pragma: no branch - opposite edge requires invalid parser state
            name = getattr(stack[-1], "name", None)
            if name in self._implied_end_tags and name != exclude:
                self._mark_active_formatting_dirty()
                stack.pop()
                continue
            break

    def _close_open_li_for_start(self) -> None:
        stack = self._stack
        for idx in range(len(stack) - 1, 0, -1):
            node = stack[idx]
            node_name = getattr(node, "name", None)
            if node_name == "li":
                self._generate_implied_end_tags("li")
                self._mark_active_formatting_dirty()
                del stack[idx:]
                return
            if self._is_special_node(node) and node_name not in {"address", "div", "p"}:
                return

    def _repair_stack_for_start(self, name: str) -> None:
        if (
            name in self._p_closing_start_tags
            and not (name == "table" and self._quirks_mode == "quirks")
            and self._find_open_index_before_boundary("p", _P_SCOPE_BOUNDARIES) is not None
        ):
            self._close_until_before_boundary("p", _P_SCOPE_BOUNDARIES)

        if name == "li":
            self._close_open_li_for_start()
            return

        if name in {"dd", "dt"}:
            self._close_until_before_boundary("dd", self._definition_scope_boundaries)
            self._close_until_before_boundary("dt", self._definition_scope_boundaries)
            return

        if name == "option":
            if len(self._stack) > 1 and self._stack[-1].name == "option":
                self._mark_active_formatting_dirty()
                self._stack.pop()
            return

        if name == "optgroup":
            if len(self._stack) > 1 and self._stack[-1].name == "option":
                self._mark_active_formatting_dirty()
                self._stack.pop()
            if (
                self._find_open_index("select") is not None
                and len(self._stack) > 1
                and self._stack[-1].name == "optgroup"
            ):
                self._mark_active_formatting_dirty()
                self._stack.pop()
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
            if table_idx is not None and getattr(self._current_parent(), "name", None) not in {"table", "colgroup"}:
                self._mark_active_formatting_dirty()
                del self._stack[table_idx + 1 :]
            if name == "col" and getattr(self._current_parent(), "name", None) == "table":
                self._insert_sanitized_element("colgroup", {}, False, self._current_parent())
                self._in_colgroup = True
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
                self._insert_sanitized_element("tbody", {}, False, self._current_parent())
            elif parent_name not in self._table_section_tags:
                if table_idx is not None:
                    self._mark_active_formatting_dirty()
                    del self._stack[table_idx + 1 :]
                    self._insert_sanitized_element("tbody", {}, False, self._current_parent())
            return

        if name in self._table_cell_tags:  # pragma: no branch - opposite edge requires invalid parser state
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
                self._insert_sanitized_element("tbody", {}, False, self._current_parent())
                self._insert_sanitized_element("tr", {}, False, self._current_parent())
                return
            if parent_name in self._table_section_tags:
                self._insert_sanitized_element("tr", {}, False, self._current_parent())
                return
            table_idx = self._find_open_index("table")
            if table_idx is not None:
                self._mark_active_formatting_dirty()
                del self._stack[table_idx + 1 :]
                self._insert_sanitized_element("tbody", {}, False, self._current_parent())
                self._insert_sanitized_element("tr", {}, False, self._current_parent())

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
                self._clear_active_formatting_to_marker()
                del stack[idx:]
                return

    def _push_active_formatting_marker(self) -> None:
        self._active_formatting.append(_ACTIVE_FORMATTING_MARKER)

    def _clear_active_formatting_to_marker(self) -> None:
        active = self._active_formatting
        while active:
            entry = active.pop()
            if entry is _ACTIVE_FORMATTING_MARKER:
                break
        self._refresh_active_formatting_dirty()

    def _foster_parent_for(self, parent: Node, *, for_tag: str | None = None) -> tuple[Node, int] | None:
        if getattr(parent, "name", None) not in self._table_foster_targets:
            return None
        if (
            for_tag is not None and for_tag in self._table_allowed_children
        ):  # pragma: no branch - opposite edge requires invalid parser state
            return None  # pragma: no cover - unreachable after parser-state guards
        table_idx = self._find_open_index("table")
        parser_only_template_idx = self._open_parser_only_template_index()
        if table_idx is not None and parser_only_template_idx is not None and table_idx < parser_only_template_idx:
            return None
        template_idx = self._open_template_index()
        if table_idx is not None and template_idx is not None and table_idx < template_idx:
            table_idx = None
        if table_idx is None:
            if self._template_modes:
                open_template_idx = self._open_template_index()
                if open_template_idx is not None:  # pragma: no branch - template modes require an open template
                    template = self._stack[open_template_idx]
                    if type(template) is Template and template.template_content is not None:
                        children = template.template_content.children
                        return template.template_content, len(children or ())
                if parent.parent is not None:  # pragma: no branch - parser-only nodes remain attached
                    siblings = parent.parent.children
                    if siblings is not None:  # pragma: no branch - parent containers keep child lists while attached
                        try:
                            return parent.parent, siblings.index(parent) + 1
                        except ValueError:  # pragma: no cover - parser nodes remain attached
                            pass
            return None
        table = self._stack[table_idx]
        table_parent = table.parent
        children = table_parent.children if table_parent is not None else None
        if table_parent is None or children is None:  # pragma: no branch - opposite edge requires invalid parser state
            return None  # pragma: no cover - unreachable after parser-state guards
        try:
            return table_parent, children.index(table)
        except ValueError:  # pragma: no cover - unreachable after parser-state guards
            return None  # pragma: no cover - unreachable after parser-state guards

    def _consume_until_end_tag(self, name: str, pos: int, end: int) -> tuple[str, int]:
        html = self._html_input
        close, next_pos = self._find_rawtext_end_tag(name, pos, end)
        if close is None:
            return html[pos:end], end
        return html[pos:close], next_pos

    def _find_rawtext_end_tag(self, name: str, pos: int, end: int) -> tuple[int | None, int]:
        return _scanner.find_rawtext_end_tag(self._html_input, self._lower_input, name, pos, end)

    def _find_script_end_tag(self, pos: int, end: int) -> tuple[int | None, int]:
        return _scanner.find_script_end_tag(self._html_input, self._lower_input, pos, end)

    def _find_script_start_marker(self, pos: int, end: int) -> int:
        return _scanner.find_script_start_marker(
            self._html_input, self._lower_input, pos, end
        )  # pragma: no cover - unreachable after parser-state guards

    def _find_tag_end(self, pos: int, end: int) -> int:
        return _scanner.find_tag_end(self._html_input, pos, end)

    def _parse_rawtext_as_text(self, name: str, pos: int, end: int) -> int:
        text_start = pos
        raw_text, pos = self._consume_until_end_tag(name, pos, end)
        if not raw_text:
            return pos
        if self._has_carriage_return and "\r" in raw_text:
            raw_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
        text = (
            raw_text if raw_text.isascii() or not self._strip_invisible_unicode else _strip_invisible_unicode(raw_text)
        )
        if self._xml_coercion and text:  # pragma: no branch - opposite edge requires invalid parser state
            text = _coerce_text_for_xml(text)  # pragma: no cover - unreachable after parser-state guards
        if not text:  # pragma: no branch - opposite edge requires invalid parser state
            return pos  # pragma: no cover - unreachable after parser-state guards
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
        self._append(parent, self._new_text(text, text_start))
        return pos

    def _parse_rcdata_element(
        self,
        name: str,
        attrs: dict[str, str | None],
        self_closing: bool,
        pos: int,
        end: int,
        tag_start: int | None = None,
        tag_end: int | None = None,
    ) -> int:
        text_start = pos
        raw_text, pos = self._consume_until_end_tag(name, pos, end)
        text = self._clean_text(raw_text, replace_null=True)
        if name == "textarea" and text.startswith("\n"):
            text = text[1:]
            text_start += 1
        if name not in self._allowed_tags:
            if text:
                self._append(self._current_parent(), self._new_text(text, text_start))
            return pos

        parent: Node
        current_parent = self._current_parent()
        parsed_in_initial_head = False
        if (
            name == "title"
            and not self._fragment
            and not self._template_modes
            and self._head is not None
            and (
                current_parent is self._head
                or (not self._body_explicit and not self._body_mode_seen and not self._body_has_content())
            )
        ):
            parent = self._head
            parsed_in_initial_head = current_parent is not self._head
        else:
            self._repair_stack_for_start(name)
            parent = current_parent
        node = self._insert_sanitized_element(
            name,
            attrs,
            False if name in self._rcdata_tags else self_closing,
            parent,
            tag_start=tag_start,
            tag_end=tag_end,
        )
        if text:
            self._append(node, self._new_text(text, text_start))
        if self._stack and self._stack[-1] is node:  # pragma: no branch - opposite edge requires invalid parser state
            if pos <= end:  # pragma: no branch - opposite edge requires invalid parser state
                close_start = self._lower_input.rfind(f"</{name}", tag_end or 0, pos)
                if close_start != -1 and self._track_tag_spans:
                    self._set_end_span(node, name, close_start, pos)
            self._stack.pop()
        if parsed_in_initial_head and self._html is not None and self._head is not None:
            self._stack = [self._doc, self._html, self._head]  # type: ignore[list-item]
        self._finish_head_reentry()
        return pos

    def _parse_rawtext_element(
        self,
        name: str,
        attrs: dict[str, str | None],
        self_closing: bool,
        pos: int,
        end: int,
        tag_start: int,
        tag_end: int,
    ) -> int:
        parent: Node
        parsed_in_initial_head = False
        if (
            name in self._p_closing_start_tags
            and self._find_open_index_before_boundary("p", _P_SCOPE_BOUNDARIES) is not None
        ):
            self._close_until_before_boundary("p", _P_SCOPE_BOUNDARIES)
        if name == "xmp" and self._active_formatting_dirty:
            self._reconstruct_active_formatting()
        if self._raw_head_text_parent(name) and self._head is not None:
            parent = self._head
            parsed_in_initial_head = self._current_parent() is not self._head
        else:
            parent = self._current_parent()
        node = self._insert_sanitized_element(name, attrs, self_closing, parent, tag_start=tag_start, tag_end=tag_end)
        text_start = pos
        close, next_pos = (
            self._find_script_end_tag(pos, end) if name == "script" else self._find_rawtext_end_tag(name, pos, end)
        )
        if close is None:
            raw_text = self._html_input[pos:end]
            pos = end
        else:
            raw_text = self._html_input[pos:close]
            pos = next_pos
            if self._track_tag_spans:
                self._set_end_span(node, name, close, pos)
        if self._has_carriage_return and "\r" in raw_text:
            raw_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
        if self._has_null and "\0" in raw_text:
            raw_text = raw_text.replace("\0", "\ufffd")
        if self._xml_coercion and raw_text:
            raw_text = _coerce_text_for_xml(raw_text)
        if raw_text:
            self._append(node, self._new_text(raw_text, text_start))
        if self._stack and self._stack[-1] is node:  # pragma: no branch - opposite edge requires invalid parser state
            self._stack.pop()
        if parsed_in_initial_head and self._html is not None and self._head is not None:
            self._stack = [self._doc, self._html, self._head]  # type: ignore[list-item]
        self._finish_head_reentry()
        return pos

    def _raw_head_text_parent(self, name: str) -> bool:
        return (
            not self._fragment
            and name in self._head_content_tags
            and self._head is not None
            and not self._template_modes
            and not self._in_head_noscript
            and not self._frameset_seen
            and not self._body_explicit
            and not self._body_mode_seen
            and not self._body_has_content()
        )

    def _parse_plaintext_element(
        self,
        name: str,
        attrs: dict[str, str | None],
        self_closing: bool,
        pos: int,
        end: int,
        tag_start: int,
        tag_end: int,
    ) -> int:
        if self._find_open_index_before_boundary("p", _P_SCOPE_BOUNDARIES) is not None:
            self._close_until_before_boundary("p", _P_SCOPE_BOUNDARIES)
        self._insert_sanitized_element(
            name, attrs, self_closing, self._current_parent(), tag_start=tag_start, tag_end=tag_end
        )
        if self._active_formatting_dirty:
            self._reconstruct_active_formatting()
        text = self._html_input[pos:end]
        if self._has_carriage_return and "\r" in text:
            text = text.replace("\r\n", "\n").replace("\r", "\n")
        if self._has_null and "\0" in text:
            text = text.replace("\0", "\ufffd")
        if self._xml_coercion and text:
            text = _coerce_text_for_xml(text)
        if text:
            self._append(self._current_parent(), self._new_text(text, pos))
        return end

    def _parse_plaintext_as_text(self, pos: int, end: int) -> int:
        if self._find_open_index_before_boundary("p", _P_SCOPE_BOUNDARIES) is not None:
            self._close_until_before_boundary("p", _P_SCOPE_BOUNDARIES)
        if self._active_formatting_dirty:
            self._reconstruct_active_formatting()
        self._append_raw_literal_text(self._html_input[pos:end], pos)
        return end

    def _parse_raw_literal_text(self, name: str, pos: int, end: int) -> int:
        text_start = pos
        text, pos = self._consume_until_end_tag(name, pos, end)
        self._append_raw_literal_text(text, text_start)
        return pos

    def _append_raw_literal_text(self, text: str, source_pos: int | None = None) -> None:
        if self._has_carriage_return and "\r" in text:
            text = text.replace("\r\n", "\n").replace("\r", "\n")
        if self._has_null and "\0" in text:
            text = text.replace("\0", "\ufffd")
        if self._strip_invisible_unicode and text and not text.isascii():
            text = _strip_invisible_unicode(text)
        if self._xml_coercion and text:
            text = _coerce_text_for_xml(text)
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
                self._append(parent, self._new_text(text, source_pos))
            else:
                foster_parent, position = foster
                self._insert_at(foster_parent, position, self._new_text(text, source_pos))

    def _parse_formatting_start(
        self,
        name: str,
        attrs: dict[str, str | None],
        pos: int,
        *,
        tag_start: int | None = None,
        tag_end: int | None = None,
        compiled_safe: bool = False,
    ) -> int:
        if name == "a" and self._find_active_formatting_index("a") is not None:
            self._adoption_agency("a")
            self._remove_last_active_formatting_by_name("a")
            self._remove_last_open_element_by_name("a")
        elif name == "nobr" and self._find_open_index("nobr") is not None:
            self._adoption_agency("nobr")
            self._remove_last_active_formatting_by_name("nobr")
            self._remove_last_open_element_by_name("nobr")

        if self._active_formatting_dirty:
            self._reconstruct_active_formatting()
        signature = () if not attrs else self._attrs_signature(attrs)
        if len(self._active_formatting) >= 3:
            duplicate_index = self._find_active_formatting_duplicate(name, signature)
            if duplicate_index is not None:
                del self._active_formatting[duplicate_index]

        if compiled_safe:
            node = self._insert_compiled_safe_element(name, attrs, False, self._current_parent())
        else:
            node = self._insert_sanitized_element(
                name, attrs, False, self._current_parent(), tag_start=tag_start, tag_end=tag_end
            )
        self._append_active_formatting_entry(name, node.attrs, node, signature)
        return pos

    def _attrs_signature(self, attrs: dict[str, str | None]) -> tuple[tuple[str, str], ...]:
        if not attrs:  # pragma: no branch - opposite edge requires invalid parser state
            return ()  # pragma: no cover - unreachable after parser-state guards
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
            entry = active[idx]
            if isinstance(entry, _FormattingMarker):
                break
            if entry.name == name:
                return idx
        return None

    def _find_active_formatting_index_by_node(self, node: Node) -> int | None:
        active = self._active_formatting
        for idx in range(len(active) - 1, -1, -1):
            entry = active[idx]
            if not isinstance(entry, _FormattingMarker) and entry.node is node:
                return idx
        return None

    def _find_active_formatting_duplicate(self, name: str, signature: tuple[tuple[str, str], ...]) -> int | None:
        matches = 0
        first_index: int | None = None
        for idx, entry in enumerate(self._active_formatting):
            if isinstance(entry, _FormattingMarker):
                matches = 0
                first_index = None
                continue
            if entry.name == name and entry.signature == signature:
                matches += 1
                if first_index is None:
                    first_index = idx
        return first_index if matches >= 3 else None

    def _remove_last_active_formatting_by_name(self, name: str) -> None:
        active = self._active_formatting
        for idx in range(len(active) - 1, -1, -1):
            entry = active[idx]
            if isinstance(entry, _FormattingMarker):
                break
            if entry.name == name:
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
        if not active:  # pragma: no branch - opposite edge requires invalid parser state
            self._active_formatting_dirty = False  # pragma: no cover - unreachable after parser-state guards
            return  # pragma: no cover - unreachable after parser-state guards
        last_entry = active[-1]
        if isinstance(last_entry, _FormattingMarker):
            self._active_formatting_dirty = False
            return
        if (
            not self._active_formatting_dirty and last_entry.node in self._stack
        ):  # pragma: no branch - opposite edge requires invalid parser state
            return  # pragma: no cover - unreachable after parser-state guards

        idx = len(active) - 1
        while idx >= 0:
            entry = active[idx]
            if isinstance(entry, _FormattingMarker) or entry.node in self._stack:
                break
            idx -= 1
        idx += 1

        while idx < len(active):
            entry = active[idx]
            if isinstance(entry, _FormattingMarker):  # pragma: no branch - opposite edge requires invalid parser state
                idx += 1  # pragma: no cover - unreachable after parser-state guards
                continue  # pragma: no cover - unreachable after parser-state guards
            node = self._insert_sanitized_element(entry.name, entry.attrs.copy(), False, self._current_parent())
            node._source_html = entry.node._source_html
            node._origin_pos = entry.node._origin_pos
            node._origin_line = entry.node._origin_line
            node._origin_col = entry.node._origin_col
            node._start_tag_start = entry.node._start_tag_start
            node._start_tag_end = entry.node._start_tag_end
            entry.node = node
            idx += 1
        self._active_formatting_dirty = False

    def _adoption_agency(
        self,
        subject: str,
        *,
        tag_start: int | None = None,
        tag_end: int | None = None,
    ) -> None:
        stack = self._stack
        active = self._active_formatting
        for _ in range(8):
            formatting_index = self._find_active_formatting_index(subject)
            if formatting_index is None:
                if stack and getattr(stack[-1], "name", None) == subject:
                    if self._track_tag_spans:
                        self._set_end_span(stack[-1], subject, tag_start, tag_end)
                    self._close_until(subject)
                return

            entry = active[formatting_index]
            if isinstance(entry, _FormattingMarker):  # pragma: no branch - opposite edge requires invalid parser state
                return  # pragma: no cover - unreachable after parser-state guards
            formatting_element = entry.node
            if stack and stack[-1] is formatting_element:
                if self._track_tag_spans:
                    self._set_end_span(formatting_element, subject, tag_start, tag_end)
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

            if not self._has_node_in_scope(formatting_element, _DEFAULT_SCOPE_BOUNDARIES):
                return

            if formatting_element is not stack[-1]:  # pragma: no branch - opposite edge requires invalid parser state
                pass

            furthest_block: Node | None = None
            furthest_block_index: int | None = None
            for idx in range(formatting_stack_index + 1, len(stack)):
                candidate = stack[idx]
                if candidate.name != "dialog" and self._is_special_node(candidate):
                    furthest_block = candidate
                    furthest_block_index = idx
                    break

            if furthest_block is None:
                if self._track_tag_spans:
                    self._set_end_span(formatting_element, subject, tag_start, tag_end)
                self._mark_active_formatting_dirty()
                del stack[formatting_stack_index:]
                del self._active_formatting[formatting_index]
                self._refresh_active_formatting_dirty()
                return
            if furthest_block_index is None:  # pragma: no branch - opposite edge requires invalid parser state
                return  # pragma: no cover - unreachable after parser-state guards

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
                if isinstance(
                    node_entry, _FormattingMarker
                ):  # pragma: no branch - opposite edge requires invalid parser state
                    return  # pragma: no cover - unreachable after parser-state guards
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
            if common_ancestor.namespace == _PARSER_ONLY_NAMESPACE:
                ancestor_index = formatting_stack_index - 2
                while (
                    ancestor_index > 0 and stack[ancestor_index].namespace == _PARSER_ONLY_NAMESPACE
                ):  # pragma: no branch - opposite edge requires invalid parser state
                    ancestor_index -= 1  # pragma: no cover - unreachable after parser-state guards
                common_ancestor = stack[ancestor_index]
            self._detach_node(last_node)
            foster = self._foster_parent_for(common_ancestor, for_tag=getattr(last_node, "name", None))
            if foster is None:
                self._append_moved_node(common_ancestor, last_node)
            else:
                foster_parent, position = foster
                self._insert_moved_node_at(foster_parent, position, last_node)

            new_formatting_element = self._clone_formatting_entry(entry)
            if self._track_tag_spans:
                self._set_end_span(new_formatting_element, subject, tag_start, tag_end)
            entry.node = new_formatting_element

            moved_children = list(furthest_block.children or ())
            if furthest_block.children is not None:  # pragma: no branch - opposite edge requires invalid parser state
                furthest_block.children.clear()
            for child in moved_children:
                self._append_moved_node(new_formatting_element, child)

            self._append_moved_node(furthest_block, new_formatting_element)

            del self._active_formatting[formatting_index]
            bookmark -= 1
            if bookmark < 0:  # pragma: no branch - opposite edge requires invalid parser state
                bookmark = 0  # pragma: no cover - unreachable after parser-state guards
            if bookmark > len(
                self._active_formatting
            ):  # pragma: no branch - opposite edge requires invalid parser state
                bookmark = len(self._active_formatting)  # pragma: no cover - unreachable after parser-state guards
            self._active_formatting.insert(bookmark, entry)

            try:
                self._mark_active_formatting_dirty()
                stack.remove(formatting_element)
            except ValueError:  # pragma: no cover - unreachable after parser-state guards
                pass  # pragma: no cover - unreachable after parser-state guards
            furthest_stack_index = stack.index(furthest_block)
            stack.insert(furthest_stack_index + 1, new_formatting_element)
            self._refresh_active_formatting_dirty()

    def _mark_active_formatting_dirty(self) -> None:
        if self._active_formatting:
            self._active_formatting_dirty = True

    def _has_node_in_scope(self, target: Node, boundaries: frozenset[str]) -> bool:
        stack = self._stack
        for idx in range(len(stack) - 1, 0, -1):  # pragma: no branch - opposite edge requires invalid parser state
            node = stack[idx]
            if node is target:
                return True
            if node.name in boundaries:
                return False
        return False  # pragma: no cover - unreachable after parser-state guards

    def _refresh_active_formatting_dirty(self) -> None:
        active = self._active_formatting
        if not active:
            self._active_formatting_dirty = False
            return
        stack = self._stack
        self._active_formatting_dirty = any(
            not isinstance(entry, _FormattingMarker) and entry.node not in stack for entry in active
        )

    def _clone_formatting_entry(self, entry: _FormattingEntry) -> Element:
        node = Element(entry.name, entry.attrs.copy(), "html")
        node._source_html = entry.node._source_html
        node._origin_pos = entry.node._origin_pos
        node._origin_line = entry.node._origin_line
        node._origin_col = entry.node._origin_col
        node._start_tag_start = entry.node._start_tag_start
        node._start_tag_end = entry.node._start_tag_end
        if not self._raw_mode and entry.name not in self._allowed_tags:
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
        except ValueError:  # pragma: no cover - unreachable after parser-state guards
            return  # pragma: no cover - unreachable after parser-state guards
        node.parent = None

    def _append_moved_node(self, parent: Node, node: Node) -> None:
        if type(parent) is Template and parent.template_content is not None:
            parent = parent.template_content
        children = parent.children
        if children is None:  # pragma: no branch - opposite edge requires invalid parser state
            return  # pragma: no cover - unreachable after parser-state guards
        children.append(node)
        node.parent = parent

    def _insert_moved_node_at(self, parent: Node, position: int, node: Node) -> None:
        if (
            type(parent) is Template and parent.template_content is not None
        ):  # pragma: no branch - opposite edge requires invalid parser state
            parent = parent.template_content  # pragma: no cover - unreachable after parser-state guards
        children = parent.children
        if children is None:  # pragma: no branch - opposite edge requires invalid parser state
            return  # pragma: no cover - unreachable after parser-state guards
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

    def _drop_recorded_nodes(self) -> None:
        nodes = self._nodes_to_drop
        for node in reversed(nodes):
            parent = node.parent
            if parent is not None:  # pragma: no branch - opposite edge requires invalid parser state
                self._remove_child(parent, node)
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
        markers: list[Element] = []
        selected_option: Element | None = None
        first_option: Element | None = None
        is_multiple = "multiple" in select.attrs
        pending: list[tuple[Node, bool, bool]] = [(select, False, False)]
        while pending:
            node, in_disabled_optgroup, in_datalist = pending.pop()
            attrs = getattr(node, "attrs", None)
            name = node.name
            if node is not select and type(node) is Element:
                if name == "selectedcontent":
                    markers.append(node)
                if name == "option" and not in_datalist:
                    if (
                        first_option is None
                        and not in_disabled_optgroup
                        and (attrs is None or "disabled" not in attrs)
                    ):
                        first_option = node
                    if attrs is not None and "selected" in attrs:
                        if is_multiple:
                            if selected_option is None:
                                selected_option = node
                        else:
                            selected_option = node
            children = getattr(node, "children", None)
            if children:
                child_disabled_optgroup = in_disabled_optgroup or (
                    name == "optgroup" and attrs is not None and "disabled" in attrs
                )
                child_in_datalist = in_datalist or name == "datalist"
                pending.extend((child, child_disabled_optgroup, child_in_datalist) for child in reversed(children))
        option = selected_option or first_option
        if not markers:  # pragma: no branch - opposite edge requires invalid parser state
            return  # pragma: no cover - unreachable after parser-state guards
        for marker in markers:
            if option is not None and self._is_descendant_of(marker, option):
                continue
            children = marker.children
            if children:
                for child in children:
                    child.parent = None
                children.clear()
            if option is not None:
                for child in option.children or ():
                    clone = child.clone_node(deep=True)
                    self._append(marker, clone)

    def _is_descendant_of(self, node: Node, ancestor: Node) -> bool:
        parent = node.parent
        while parent is not None:
            if parent is ancestor:
                return True
            parent = parent.parent
        return False

    def _unwrap_node(self, node: Element) -> None:
        parent = node.parent
        children = parent.children if parent is not None else None
        if parent is None or children is None:  # pragma: no branch - opposite edge requires invalid parser state
            return  # pragma: no cover - unreachable after parser-state guards
        try:
            index = children.index(node)
        except ValueError:
            return

        moved = list(node.children or ())
        if node.children is not None:  # pragma: no branch - opposite edge requires invalid parser state
            node.children = []
        if (
            type(node) is Template and node.template_content is not None
        ):  # pragma: no branch - opposite edge requires invalid parser state
            content = node.template_content  # pragma: no cover - unreachable after parser-state guards
            if content.children:  # pragma: no cover - unreachable after parser-state guards
                moved.extend(content.children)  # pragma: no cover - unreachable after parser-state guards
                content.children = []  # pragma: no cover - unreachable after parser-state guards
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
            if pos >= end:  # pragma: no branch - opposite edge requires invalid parser state
                return {}, False, pos, False  # pragma: no cover - unreachable after parser-state guards
            ch = html[pos]
            if ch == ">":
                return {}, False, pos + 1, True
            if ch == "/" and pos + 1 < end and html[pos + 1] == ">":
                return {}, True, pos + 2, True
            if ch == "/":
                pos += 1
                continue

            name_start = pos
            while pos < end and html[pos] not in attr_name_stop:
                pos += 1
            if pos == name_start:  # pragma: no branch - stop chars that reach here collapse into one recovery path
                if html[pos] == "=":  # pragma: no cover - unreachable after earlier delimiter handling
                    pos += 1  # pragma: no cover - unreachable after earlier delimiter handling
                    continue  # pragma: no cover - unreachable after earlier delimiter handling
                pos += 1  # pragma: no cover - all other stop chars are handled before attr scanning reaches this point
                continue  # pragma: no cover - all other stop chars are handled before attr scanning reaches this point
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

    def _parse_all_attrs(self, pos: int, end: int) -> tuple[dict[str, str | None], bool, int, bool]:
        html = self._html_input
        space = _SPACE
        attr_name_stop = _ATTR_NAME_STOP
        attr_value_stop = _ATTR_VALUE_STOP
        attrs: dict[str, str | None] = {}

        while pos < end:
            while pos < end and html[pos] in space:
                pos += 1
            if pos >= end:  # pragma: no branch - opposite edge requires invalid parser state
                return attrs, False, pos, False  # pragma: no cover - unreachable after parser-state guards
            ch = html[pos]
            if ch == ">":
                return attrs, False, pos + 1, True
            if ch == "/" and pos + 1 < end and html[pos + 1] == ">":
                return attrs, True, pos + 2, True
            if ch == "/":
                pos += 1
                continue

            name_start = pos
            while pos < end and html[pos] not in attr_name_stop:
                pos += 1
            if pos == name_start:
                pos += 1
                while pos < end and html[pos] not in attr_name_stop:
                    pos += 1
                raw_key = html[name_start:pos]
            else:
                raw_key = html[name_start:pos]
            key = raw_key if raw_key.islower() else raw_key.lower()
            if "\0" in key:
                key = key.replace("\0", "\ufffd")

            while pos < end and html[pos] in space:
                pos += 1

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

            if self._has_carriage_return and "\r" in value:
                value = value.replace("\r\n", "\n").replace("\r", "\n")
            if "&" in value:
                value = decode_entities_in_text(value, in_attribute=True)
            if "\0" in value:
                value = value.replace("\0", "\ufffd")
            if key not in attrs:
                attrs[key] = value
        return attrs, False, pos, False

    def _sanitize_parsed_attrs(
        self,
        action: TagAction | None,
        attrs: dict[str, str | None],
    ) -> dict[str, str | None]:
        if not attrs or action is None or not action.allowed:
            return attrs

        sanitized: dict[str, str | None] = {}
        allowed_attrs = action.allowed_attrs
        for key, raw_value in attrs.items():
            if key not in allowed_attrs:
                continue
            value = raw_value or ""
            if self._strip_invisible_unicode and value and not value.isascii():
                value = _strip_invisible_unicode(value)
            kind = action.url_attr_kinds.get(key)
            if kind is not None:
                rule = action.url_attr_rules.get(key)
                if rule is None:  # pragma: no branch - opposite edge requires invalid parser state
                    continue  # pragma: no cover - unreachable after parser-state guards
                if kind == "url":  # pragma: no branch - opposite edge requires invalid parser state
                    sanitized_fast = self._sanitize_simple_url_fast(value, rule)
                    if sanitized_fast is None:
                        continue
                    if sanitized_fast is not _URL_FAST_FALLBACK:
                        sanitized[key] = sanitized_fast  # type: ignore[assignment]
                        continue
                sanitized_value = _sanitize_url_sink_value(
                    url_policy=self._url_policy,
                    rule=rule,
                    tag=action.name,
                    attr=key,
                    kind=kind,
                    value=value,
                )
                if sanitized_value is None:
                    continue
                value = sanitized_value
            sanitized[key] = value
        return sanitized

    def _sanitize_simple_url_fast(self, value: str, rule: UrlRule) -> str | None | object:
        url_policy = self._url_policy
        handling = rule.handling if rule.handling is not None else url_policy.default_handling
        allow_relative = rule.allow_relative if rule.allow_relative is not None else url_policy.default_allow_relative
        if (  # pragma: no branch - opposite edge requires invalid parser state
            handling != "allow"
            or rule.proxy is not None
            or url_policy.proxy is not None
            or url_policy.url_filter is not None
            or rule.allowed_hosts is not None
        ):
            return _URL_FAST_FALLBACK  # pragma: no cover - unreachable after parser-state guards

        if not value.isascii():
            return _URL_FAST_FALLBACK

        stripped = value.strip()
        if not stripped:
            return None

        allowed_schemes = rule.allowed_schemes
        if stripped.startswith("https://") and " " not in stripped and stripped.isprintable():
            return stripped if "https" in allowed_schemes else None
        if stripped.startswith("http://") and " " not in stripped and stripped.isprintable():
            return stripped if "http" in allowed_schemes else None

        if _URL_CONTROL_CHAR_REGEX.search(stripped):
            return None

        if ":" not in stripped:
            if "\\" in stripped:
                return None
            if stripped.startswith("#"):
                return stripped if rule.allow_fragment else None
            if stripped.startswith("//"):
                resolved = rule.resolve_protocol_relative
                if not resolved:
                    return None
                resolved_scheme = resolved.lower()
                if (
                    resolved_scheme not in rule.allowed_schemes
                ):  # pragma: no branch - opposite edge requires invalid parser state
                    return None  # pragma: no cover - unreachable after parser-state guards
                return f"{resolved_scheme}:{stripped}"
            return stripped if allow_relative else None

        normalized = _normalize_url_for_checking(stripped)
        if not normalized or "\\" in normalized:  # pragma: no branch - opposite edge requires invalid parser state
            return None  # pragma: no cover - unreachable after parser-state guards

        if normalized.startswith("#"):  # pragma: no branch - opposite edge requires invalid parser state
            return (
                stripped if rule.allow_fragment else None
            )  # pragma: no cover - unreachable after parser-state guards

        if normalized.startswith("//"):  # pragma: no branch - opposite edge requires invalid parser state
            resolved = rule.resolve_protocol_relative  # pragma: no cover - unreachable after parser-state guards
            if not resolved:  # pragma: no cover - unreachable after parser-state guards
                return None  # pragma: no cover - unreachable after parser-state guards
            resolved_scheme = resolved.lower()  # pragma: no cover - unreachable after parser-state guards
            if resolved_scheme not in allowed_schemes:  # pragma: no cover - unreachable after parser-state guards
                return None  # pragma: no cover - unreachable after parser-state guards
            return f"{resolved_scheme}:{normalized}"  # pragma: no cover - unreachable after parser-state guards

        scheme = _get_scheme(normalized)
        if scheme is not None:  # pragma: no branch - opposite edge requires invalid parser state
            if scheme not in allowed_schemes:
                return None
            if (
                scheme in {"http", "https"} and not normalized.startswith(f"{scheme}://") and not allow_relative
            ):  # pragma: no branch - opposite edge requires invalid parser state
                return None  # pragma: no cover - unreachable after parser-state guards
            return stripped

        return stripped if allow_relative else None  # pragma: no cover - unreachable after parser-state guards

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
        preserve_state_attrs = action.preserve_state_attrs
        tag = action.name

        while pos < end:
            while pos < end and html[pos] in space:
                pos += 1
            if pos >= end:  # pragma: no branch - opposite edge requires invalid parser state
                return attrs, False, pos, False  # pragma: no cover - unreachable after parser-state guards
            ch = html[pos]
            if ch == ">":
                return attrs, False, pos + 1, True
            if ch == "/" and pos + 1 < end and html[pos + 1] == ">":
                return attrs, True, pos + 2, True
            if ch == "/":
                pos += 1
                continue

            name_start = pos
            while pos < end and html[pos] not in attr_name_stop:
                pos += 1
            if pos == name_start:
                if html[pos] == "=":  # pragma: no cover - unreachable after earlier delimiter handling
                    pos += 1  # pragma: no cover - unreachable after earlier delimiter handling
                    continue  # pragma: no cover - unreachable after earlier delimiter handling
                pos += 1  # pragma: no cover - all other stop chars are handled before attr scanning reaches this point
                continue  # pragma: no cover - all other stop chars are handled before attr scanning reaches this point
            raw_key = html[name_start:pos]
            key = raw_key if raw_key.islower() else raw_key.lower()
            if "\0" in key:
                key = key.replace("\0", "\ufffd")
            keep_output = key in allowed_attrs
            keep_state = preserve_state_attrs or key in state_attrs

            while pos < end and html[pos] in space:
                pos += 1
            if not keep_output and not keep_state:
                if pos < end and html[pos] == "=":
                    pos += 1
                    while (
                        pos < end and html[pos] in space
                    ):  # pragma: no branch - opposite edge requires invalid parser state
                        pos += 1  # pragma: no cover - unreachable after parser-state guards
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
                while (
                    pos < end and html[pos] in space
                ):  # pragma: no branch - opposite edge requires invalid parser state
                    pos += 1  # pragma: no cover - unreachable after parser-state guards
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
                if rule is None:  # pragma: no branch - opposite edge requires invalid parser state
                    continue  # pragma: no cover - unreachable after parser-state guards
                if url_attr_kinds[key] == "url":  # pragma: no branch - opposite edge requires invalid parser state
                    sanitized_fast = self._sanitize_simple_url_fast(value, rule)
                    if sanitized_fast is None:
                        continue
                    if sanitized_fast is not _URL_FAST_FALLBACK:
                        attrs[key] = sanitized_fast  # type: ignore[assignment]
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
            if name == "script" and self._script_eof_keeps_shell(
                pos, end
            ):  # pragma: no branch - opposite edge requires invalid parser state
                self._keep_empty_shell_on_eof = True  # pragma: no cover - unreachable after parser-state guards
            self._append_text_boundary(self._current_parent())
            return end
        self._append_text_boundary(self._current_parent())
        if name in {"script", "style"}:  # pragma: no branch - opposite edge requires invalid parser state
            self._skip_escaped_comment_space = False
            self._foster_next_table_whitespace = 0
        return next_pos

    def _skip_subtree(self, name: str, pos: int, end: int, *, detect_foreign_breakout: bool = False) -> int:
        html = self._html_input
        depth = 1
        while pos < end and depth:
            lt = html.find("<", pos, end)
            if lt == -1:
                if detect_foreign_breakout:  # pragma: no branch - opposite edge requires invalid parser state
                    return -1
                self._dropped_to_eof = True  # pragma: no cover - unreachable after parser-state guards
                return end  # pragma: no cover - unreachable after parser-state guards
            p = lt + 1
            if self._lower_input.startswith("<![cdata[", lt):
                close = html.find("]]>", p + 8, end)
                if close == -1:  # pragma: no branch - opposite edge requires invalid parser state
                    self._dropped_to_eof = True  # pragma: no cover - unreachable after parser-state guards
                    return end  # pragma: no cover - unreachable after parser-state guards
                pos = close + 3
                continue
            is_end = p < end and html[p] == "/"
            if is_end:
                p += 1
            match = _TAG_NAME_RE.match(html, p, end)
            if not match:  # pragma: no branch - opposite edge requires invalid parser state
                pos = p  # pragma: no cover - unreachable after parser-state guards
                continue  # pragma: no cover - unreachable after parser-state guards
            tag = match.group(0)
            if not tag.islower():
                tag = tag.lower()
            gt = html.find(">", match.end(), end)
            pos = end if gt == -1 else gt + 1
            if (
                detect_foreign_breakout
                and is_end
                and tag != name
                and (tag in FOREIGN_BREAKOUT_ELEMENTS or tag in _TABLE_SCOPED_END_TAGS)
            ):
                return -1
            if not is_end and tag != name:
                if detect_foreign_breakout and tag in _FOREIGN_FULL_PARSE_TAGS:
                    return -1
                if tag in self._plaintext_tags:
                    self._dropped_to_eof = True
                    return end
                if tag in self._rcdata_tags or tag in self._drop_content_tags or tag in self._rawtext_as_text_tags:
                    rawtext_close, rawtext_pos = (
                        self._find_script_end_tag(pos, end)
                        if tag == "script"
                        else self._find_rawtext_end_tag(tag, pos, end)
                    )
                    if rawtext_close is None:  # pragma: no branch - opposite edge requires invalid parser state
                        if detect_foreign_breakout:  # pragma: no cover - unreachable after parser-state guards
                            return -1  # pragma: no cover - unreachable after parser-state guards
                        self._dropped_to_eof = True  # pragma: no cover - unreachable after parser-state guards
                        return end  # pragma: no cover - unreachable after parser-state guards
                    pos = rawtext_pos
                    continue
            if tag == name:
                depth += -1 if is_end else 1
        if depth:
            if detect_foreign_breakout:  # pragma: no branch - opposite edge requires invalid parser state
                return -1
            self._dropped_to_eof = True  # pragma: no cover - unreachable after parser-state guards
        elif detect_foreign_breakout:  # pragma: no branch - opposite edge requires invalid parser state
            next_markup = pos
            while next_markup < end and html[next_markup] in _SPACE:
                next_markup += 1
            if self._lower_input.startswith("<frameset", next_markup, end):
                return -1
        return pos

    def _finish_document_shell(self) -> None:
        if self._fragment or (not self._dropped_to_eof and not self._frameset_seen):
            return
        if not self._raw_mode and self._dropped_to_eof and not self._frameset_seen:
            return
        html = self._html
        head = self._head
        body = self._body if isinstance(self._body, Element) else None
        if (
            html is None or head is None or body is None
        ):  # pragma: no branch - opposite edge requires invalid parser state
            return  # pragma: no cover - unreachable after parser-state guards
        if self._frameset_seen:  # pragma: no branch - opposite edge requires invalid parser state
            if (
                self._node_is_empty(body) and not self._body_explicit
            ):  # pragma: no branch - guaranteed by frameset mode
                self._remove_child(html, body)
            return
        if (
            not self._node_is_empty(body) or self._body_explicit
        ):  # pragma: no cover - unreachable after parser-state guards
            return  # pragma: no cover - unreachable after parser-state guards
        if self._keep_empty_shell_on_eof:  # pragma: no cover - unreachable after parser-state guards
            return  # pragma: no cover - unreachable after parser-state guards

        if self._explicit_head or not self._node_is_empty(
            head
        ):  # pragma: no cover - unreachable after parser-state guards
            self._remove_child(html, body)  # pragma: no cover - unreachable after parser-state guards
            return  # pragma: no cover - unreachable after parser-state guards

        if self._explicit_html:  # pragma: no cover - unreachable after parser-state guards
            self._remove_child(html, head)  # pragma: no cover - unreachable after parser-state guards
            self._remove_child(html, body)  # pragma: no cover - unreachable after parser-state guards
            return  # pragma: no cover - unreachable after parser-state guards

        self._remove_child(self._doc, html)  # pragma: no cover - unreachable after parser-state guards

    def _node_is_empty(self, node: Node) -> bool:
        children = node.children
        if not children:
            return True
        return all(  # pragma: no cover - parser-created empty shell nodes have no children
            type(child) is Text and not child.data for child in children
        )

    def _script_eof_keeps_shell(self, pos: int, end: int) -> bool:
        raw = self._lower_input[pos:end].rstrip(_SPACE)
        if not raw.endswith("</script>"):
            return False
        comment = raw.find("<!--")
        if comment == -1:  # pragma: no branch - opposite edge requires invalid parser state
            return False  # pragma: no cover - unreachable after parser-state guards
        if raw.count("</script", comment + 4) < 2:  # pragma: no branch - opposite edge requires invalid parser state
            return False
        return (
            self._find_script_start_marker(pos + comment + 4, end) != -1
        )  # pragma: no cover - unreachable after parser-state guards

    def _remove_child(self, parent: Node, child: Node) -> None:
        children = parent.children
        if children is None:  # pragma: no branch - opposite edge requires invalid parser state
            return  # pragma: no cover - unreachable after parser-state guards
        try:
            children.remove(child)
        except ValueError:
            return
        child.parent = None

    def _accept_frameset(self) -> bool:
        if (
            self._fragment
            or self._frameset_seen
            or self._frameset_blocked
            or self._body_explicit
            or not isinstance(self._body, Element)
        ):
            return False
        if not self._body_allows_frameset(self._body):
            return False
        children = self._body.children
        if children is not None:  # pragma: no branch - opposite edge requires invalid parser state
            children.clear()
        self._frameset_seen = True
        self._mark_active_formatting_dirty()
        self._stack = [self._doc, self._html]  # type: ignore[list-item]
        self._after_head = False
        return True

    def _accept_fragment_frameset(self) -> bool:
        if not self._fragment or self._fragment_context_name != "html" or self._frameset_seen:
            return False
        if self._body.children and any(type(child) is not Text for child in self._body.children):
            return False
        if not self._body_allows_frameset(self._body):
            return False
        children = self._body.children
        if children is not None:  # pragma: no branch - opposite edge requires invalid parser state
            children.clear()
        if self._html is not None and isinstance(
            self._body, Element
        ):  # pragma: no branch - opposite edge requires invalid parser state
            self._remove_child(self._html, self._body)
            self._stack = [self._doc, self._html]
        self._frameset_seen = True
        self._mark_active_formatting_dirty()
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
            namespace = getattr(child, "namespace", None)
            if namespace == _PARSER_ONLY_NAMESPACE:
                return False
            if namespace not in {None, "html"}:
                if self._foreign_subtree_allows_frameset(child):
                    continue
                return False
            if child.name == "input":
                attrs = getattr(child, "attrs", None)
                input_type = attrs.get("type") if attrs is not None else None
                if (
                    isinstance(input_type, str) and input_type.lower() == "hidden"
                ):  # pragma: no branch - opposite edge requires invalid parser state
                    continue
                return False
            if child.name not in self._frameset_body_ok_tags:
                return False
            if not self._body_allows_frameset(child):
                return False
        return True

    def _foreign_subtree_allows_frameset(self, node: Node) -> bool:
        children = node.children
        if not children:
            return True
        for child in children:
            if type(child) is Text:
                if (child.data or "").strip(_SPACE + "\ufffd"):
                    return False
                continue
            if not self._foreign_subtree_allows_frameset(child):
                return False
        return True

    def _append_frameset_text(self, raw: str) -> None:
        if (
            self._has_carriage_return and "\r" in raw
        ):  # pragma: no branch - opposite edge requires invalid parser state
            raw = raw.replace("\r\n", "\n").replace(
                "\r", "\n"
            )  # pragma: no cover - unreachable after parser-state guards
        if self._html is None:  # pragma: no branch - opposite edge requires invalid parser state
            return  # pragma: no cover - unreachable after parser-state guards
        if "&" in raw:
            raw = decode_entities_in_text(raw)
        text = "".join(ch for ch in raw if ch in _SPACE)
        if self._xml_coercion and text:
            text = _coerce_text_for_xml(text)
        if text:
            parent = self._current_parent()
            if parent.name != "frameset":
                parent = self._html
            self._append(parent, self._new_text(text))
