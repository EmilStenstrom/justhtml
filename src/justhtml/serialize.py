"""HTML serialization utilities for JustHTML DOM nodes."""

# ruff: noqa: PERF401

from __future__ import annotations

import re
from enum import Enum
from typing import Any
from urllib.parse import quote as url_quote

from .constants import FOREIGN_ATTRIBUTE_ADJUSTMENTS, SPECIAL_ELEMENTS, VOID_ELEMENTS, WHITESPACE_PRESERVING_ELEMENTS

# Matches characters that prevent an attribute value from being unquoted.
# Note: This matches the logic of the previous loop-based implementation.
# It checks for space characters, quotes, equals sign, and greater-than.
_UNQUOTED_ATTR_VALUE_INVALID = re.compile(r'[ \t\n\f\r"\'=>]')
_LITERAL_TEXT_SERIALIZATION_ELEMENTS = frozenset({"script", "style"})
_SERIALIZABLE_TAG_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9:_-]*$")
_SERIALIZABLE_ATTR_NAME_RE = re.compile(r"^[A-Za-z_:][A-Za-z0-9:._-]*$")


class HTMLContext(str, Enum):
    HTML = "html"
    # Serialize node to HTML markup, then JS-string escape the resulting HTML.
    # Use this when embedding HTML markup into a JavaScript string literal.
    JS_STRING = "js_string"
    # Serialize node to HTML markup, then escape it for a quoted HTML attribute value.
    # This is useful for attributes that are later parsed as HTML (e.g. iframe[srcdoc]).
    HTML_ATTR_VALUE = "html_attr_value"
    # Serialize node to text, then percent-encode it.
    # Use this for URL attributes like href or src.
    URL = "url"


def _escape_text(text: str | None) -> str:
    if not text:
        return ""
    if "&" not in text and "<" not in text and ">" not in text:
        return text
    # Minimal, but matches html5lib serializer expectations in core cases.
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _validate_serializable_tag_name(name: str) -> str:
    if not _SERIALIZABLE_TAG_NAME_RE.match(name):
        raise ValueError(f"Unsafe element name for serialization: {name!r}")
    return name


def _validate_serializable_attr_name(name: str) -> str:
    if not _SERIALIZABLE_ATTR_NAME_RE.match(name):
        raise ValueError(f"Unsafe attribute name for serialization: {name!r}")
    return name


def _serialize_comment_data(data: str | None) -> str:
    if not data:
        return ""

    out = data
    while "--" in out:
        out = out.replace("--", "- -")

    if out.endswith("-"):
        out += " "

    return out


def _neutralize_rawtext_end_tag_sequences(text: str, tag_name: str) -> str:
    if not text:
        return text

    lower_text = text.lower()
    needle = f"</{tag_name}"
    needle_len = len(needle)
    out: list[str] = []
    start = 0
    changed = False

    while True:
        idx = lower_text.find(needle, start)
        if idx == -1:
            break

        boundary = idx + needle_len
        if boundary == len(text) or text[boundary] in " \t\n\r\f/>":
            out.append(text[start:idx])
            out.append("&lt;")
            start = idx + 1
            changed = True
            continue

        start = idx + 1

    if not changed:
        return text

    out.append(text[start:])
    return "".join(out)


def _serialize_text_for_parent(text: str | None, parent_name: str | None) -> str:
    if not text:
        return ""
    if parent_name is not None:
        normalized_parent_name = parent_name if parent_name.islower() else parent_name.lower()
        if normalized_parent_name in _LITERAL_TEXT_SERIALIZATION_ELEMENTS:
            return _neutralize_rawtext_end_tag_sequences(text, normalized_parent_name)
    return _escape_text(text)


def _escape_js_string(value: str, *, quote: str = '"') -> str:
    if quote not in {'"', "'"}:
        raise ValueError("quote must be ' or \"")

    if not value:
        return ""

    out: list[str] = []
    for ch in value:
        if ch == "\\":
            out.append("\\\\")
        elif ch == quote:
            out.append("\\" + quote)
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\b":
            out.append("\\b")
        elif ch == "\f":
            out.append("\\f")
        elif ch == "<":
            out.append("\\u003c")
        elif ch == ">":
            out.append("\\u003e")
        elif ch == "\u2028":
            out.append("\\u2028")
        elif ch == "\u2029":
            out.append("\\u2029")
        else:
            out.append(ch)
    return "".join(out)


def _escape_url_value(value: str) -> str:
    if not value:
        return ""
    # Preserve common URL separators while percent-encoding other characters.
    return url_quote(value, safe="/:@?&=#+-._~")


def _choose_attr_quote(value: str | None, forced_quote_char: str | None = None) -> str:
    if forced_quote_char in {'"', "'"}:
        return forced_quote_char
    if value is None:
        return '"'
    # value is assumed to be a string
    if '"' in value and "'" not in value:
        return "'"
    return '"'


def _escape_attr_value(value: str | None, quote_char: str) -> str:
    if value is None:
        return ""
    if "&" not in value and "<" not in value and ">" not in value and quote_char not in value:
        return value
    escaped = value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    if quote_char == '"':
        return escaped.replace('"', "&quot;")
    return escaped.replace("'", "&#39;")


def _can_unquote_attr_value(value: str | None) -> bool:
    if value is None:
        return False
    # Optimization: use regex instead of loop
    return not _UNQUOTED_ATTR_VALUE_INVALID.search(value)


def _serialize_doctype(node: Any) -> str:
    doctype = node.data
    name = "html"
    public_id = None
    system_id = None

    if isinstance(doctype, str):
        name = doctype
    elif doctype is not None:
        name = doctype.name or ""
        public_id = doctype.public_id
        system_id = doctype.system_id

    if name:
        name = _validate_serializable_tag_name(name)

    parts: list[str] = ["<!DOCTYPE"]
    if name:
        parts.extend((" ", name))

    if public_id is not None:
        parts.extend((' PUBLIC "', _escape_attr_value(public_id, '"'), '"'))
        if system_id is not None:
            parts.extend((' "', _escape_attr_value(system_id, '"'), '"'))
    elif system_id is not None:
        parts.extend((' SYSTEM "', _escape_attr_value(system_id, '"'), '"'))

    parts.append(">")
    return "".join(parts)


def serialize_start_tag(
    name: str,
    attrs: dict[str, str | None] | None,
    *,
    quote_attr_values: bool = True,
    minimize_boolean_attributes: bool = True,
    quote_char: str | None = None,
    use_trailing_solidus: bool = False,
    is_void: bool = False,
) -> str:
    name = _validate_serializable_tag_name(name)
    parts: list[str] = ["<", name]
    if attrs:
        parts_extend = parts.extend
        for raw_key, value in attrs.items():
            key = _validate_serializable_attr_name(raw_key)
            if minimize_boolean_attributes:
                if value is None or value == "" or value == key:
                    parts_extend((" ", key))
                    continue
                if len(value) == len(key) and value.lower() == key:
                    parts_extend((" ", key))
                    continue

            if value is None or value == "":
                parts_extend((" ", key, '=""'))
                continue

            if not quote_attr_values and _can_unquote_attr_value(value):
                escaped = value.replace("&", "&amp;").replace("<", "&lt;")
                parts_extend((" ", key, "=", escaped))
                continue

            quote = _choose_attr_quote(value, quote_char)
            escaped = _escape_attr_value(value, quote)
            parts_extend((" ", key, "=", quote, escaped, quote))

    if use_trailing_solidus and is_void:
        parts.append(" />")
    else:
        parts.append(">")
    return "".join(parts)


def serialize_end_tag(name: str) -> str:
    name = _validate_serializable_tag_name(name)
    return f"</{name}>"


def _node_to_html_compact(node: Any) -> str:
    """Serialize a node subtree to compact HTML (no newlines/indentation).

    This is a hot path for `to_html(..., pretty=False)`, so it is implemented
    iteratively to avoid recursion overhead and per-node list allocations.
    """

    parts: list[str] = []
    append = parts.append
    stack: list[Any] = [node]
    _end = object()
    stack_append = stack.append
    stack_pop = stack.pop

    void_elements = VOID_ELEMENTS
    serialize_text = _serialize_text_for_parent
    serialize_start_tag_ = serialize_start_tag

    while stack:
        item = stack_pop()
        if type(item) is tuple and item and item[0] is _end:
            append("</" + item[1] + ">")
            continue

        name: str = item.name

        if name == "#text":
            data = item.data
            if data:
                parent = item.parent
                parent_name = parent.name if parent is not None else None
                append(serialize_text(data, parent_name))
            continue

        if name == "#comment":
            append(f"<!--{_serialize_comment_data(item.data)}-->")
            continue

        if name == "!doctype":
            append(_serialize_doctype(item))
            continue

        if name == "#document" or name == "#document-fragment":
            doc_children = item.children
            if doc_children:
                for child in reversed(doc_children):
                    if child is not None:  # pragma: no branch
                        stack_append(child)
            continue

        # Element node.
        append(serialize_start_tag_(name, item.attrs))

        if name in void_elements:
            continue

        # Template special handling: HTML templates store contents in `template_content`.
        children: list[Any] = item.children
        if name == "template":
            tc = item.template_content
            ns = item.namespace
            if tc is not None and (ns is None or ns == "html"):
                children = tc.children

        # Push an end-tag marker, then children in reverse so they serialize
        # left-to-right.
        stack_append((_end, name))
        if children:
            for child in reversed(children):
                if child is not None:  # pragma: no branch
                    stack_append(child)

    return "".join(parts)


def to_html(
    node: Any,
    indent: int = 0,
    indent_size: int = 2,
    *,
    pretty: bool = True,
    context: HTMLContext | None = None,
    quote: str = '"',
) -> str:
    """Convert node to HTML string."""
    if pretty:
        html = _node_to_html(node, indent, indent_size, in_pre=False)
    else:
        html = _node_to_html_compact(node)

    if context is None:
        context = HTMLContext.HTML

    if not isinstance(context, HTMLContext):
        raise TypeError(f"Unknown serialization context: {context}")

    if context == HTMLContext.HTML:
        return html

    if context == HTMLContext.JS_STRING:
        return _escape_js_string(html, quote=quote)

    if context == HTMLContext.HTML_ATTR_VALUE:
        if quote not in {'"', "'"}:
            raise ValueError("quote must be ' or \"")
        return _escape_attr_value(html, quote)

    if context == HTMLContext.URL:
        # For URL context, we assume the node content is the URL.
        # We strip surrounding whitespace and percent-encode.

        # First get text content (URLs shouldn't have markup)
        text = node.to_text() if hasattr(node, "to_text") else html
        return url_quote(text.strip(), safe="/:@?&=#+-._~")

    raise TypeError(f"Unknown serialization context: {context}")  # pragma: no cover


def _collapse_html_whitespace(text: str) -> str:
    """Collapse HTML whitespace runs to a single space and trim edges.

    This matches how HTML rendering treats most whitespace in text nodes, and is
    used only for pretty-printing in non-preformatted contexts.
    """
    if not text:
        return ""

    # Optimization: split() handles whitespace collapsing efficiently.
    # Note: split() treats \v as whitespace, which is not HTML whitespace.
    # But \v is extremely rare in HTML.
    if "\v" in text:
        parts: list[str] = []
        in_whitespace = False
        for ch in text:
            if ch in {" ", "\t", "\n", "\f", "\r"}:
                if not in_whitespace:
                    parts.append(" ")
                    in_whitespace = True
                continue

            parts.append(ch)
            in_whitespace = False

        collapsed = "".join(parts)
        return collapsed.strip(" ")

    return " ".join(text.split())


def _normalize_formatting_whitespace(text: str) -> str:
    """Normalize formatting whitespace within a text node.

    Converts newlines/tabs/CR/FF to regular spaces and collapses runs that
    include such formatting whitespace to a single space.

    Pure space runs are preserved as-is (so existing double-spaces remain).
    """
    if not text:
        return ""

    if "\n" not in text and "\r" not in text and "\t" not in text and "\f" not in text:
        return text

    starts_with_formatting = text[0] in {"\n", "\r", "\t", "\f"}
    ends_with_formatting = text[-1] in {"\n", "\r", "\t", "\f"}

    out: list[str] = []
    in_ws = False
    saw_formatting_ws = False

    for ch in text:
        if ch == " ":
            if in_ws:
                # Only collapse if this whitespace run included formatting whitespace.
                if saw_formatting_ws:
                    continue
                out.append(" ")
                continue
            in_ws = True
            saw_formatting_ws = False
            out.append(" ")
            continue

        if ch in {"\n", "\r", "\t", "\f"}:
            if in_ws:
                saw_formatting_ws = True
                continue
            in_ws = True
            saw_formatting_ws = True
            out.append(" ")
            continue

        in_ws = False
        saw_formatting_ws = False
        out.append(ch)

    normalized = "".join(out)
    if starts_with_formatting and normalized.startswith(" "):
        normalized = normalized[1:]
    if ends_with_formatting and normalized.endswith(" "):
        normalized = normalized[:-1]
    return normalized


def _is_whitespace_text_node(node: Any) -> bool:
    return node.name == "#text" and (node.data or "").strip() == ""


def _is_blocky_element(node: Any) -> bool:
    # Treat elements as block-ish if they are block-level *or* contain any block-level
    # descendants. This keeps pretty-printing readable for constructs like <a><div>...</div></a>.
    try:
        name = node.name
    except AttributeError:
        return False
    if name in {"#text", "#comment", "!doctype"}:
        return False
    if name in SPECIAL_ELEMENTS:
        return True

    try:
        children = node.children
    except AttributeError:
        return False
    if not children:
        return False

    stack: list[Any] = list(children)
    while stack:
        child = stack.pop()
        if child is None:
            continue
        child_name = child.name
        if child_name in SPECIAL_ELEMENTS:
            return True
        if child_name in {"#text", "#comment", "!doctype"}:
            continue
        grand_children = child.children
        if grand_children:
            stack.extend(grand_children)

    return False


_LAYOUT_BLOCK_ELEMENTS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "body",
    "caption",
    "center",
    "dd",
    "details",
    "dialog",
    "dir",
    "div",
    "dl",
    "dt",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hgroup",
    "hr",
    "html",
    "iframe",
    "li",
    "listing",
    "main",
    "marquee",
    "menu",
    "nav",
    "noframes",
    "noscript",
    "ol",
    "p",
    "plaintext",
    "pre",
    "search",
    "section",
    "summary",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "tr",
    "ul",
}


_FORMAT_SEP = object()


def _is_layout_blocky_element(node: Any) -> bool:
    # Similar to _is_blocky_element(), but limited to actual layout blocks.
    # This avoids turning inline-ish "special" elements like <script> into
    # multiline pretty-print breaks in contexts like <p>.
    try:
        name = node.name
    except AttributeError:
        return False
    if name in {"#text", "#comment", "!doctype"}:
        return False
    if name in _LAYOUT_BLOCK_ELEMENTS:
        return True

    try:
        children = node.children
    except AttributeError:
        return False
    if not children:
        return False

    stack: list[Any] = list(children)
    while stack:
        child = stack.pop()
        if child is None:
            continue
        child_name = child.name
        if child_name in _LAYOUT_BLOCK_ELEMENTS:
            return True
        if child_name in {"#text", "#comment", "!doctype"}:
            continue
        grand_children = child.children
        if grand_children:
            stack.extend(grand_children)

    return False


def _pretty_renders_nonempty(node: Any, *, in_pre: bool) -> bool:
    stack: list[tuple[Any, bool]] = [(node, in_pre)]

    while stack:
        current, current_in_pre = stack.pop()
        name = current.name
        if name == "#text":
            data = current.data or ""
            if data if current_in_pre else data.strip():
                return True
            continue

        if name in {"#comment", "!doctype"}:
            return True

        if name in {"#document", "#document-fragment"}:
            for child in reversed(current.children or []):
                stack.append((child, current_in_pre))
            continue

        return True

    return False


def _is_formatting_whitespace_text(data: str) -> bool:
    # Formatting whitespace is something users typically don't intend to preserve
    # exactly (e.g. newlines/indentation, or large runs of spaces).
    if not data:
        return False
    if "\n" in data or "\r" in data or "\t" in data or "\f" in data:
        return True
    return len(data) > 2


def _should_pretty_indent_children(children: list[Any]) -> bool:
    for child in children:
        if child is None:
            continue
        name = child.name
        if name == "#comment":
            return False
        if name == "#text" and (child.data or "").strip():
            return False

    element_children: list[Any] = [
        child for child in children if child is not None and child.name not in {"#text", "#comment"}
    ]
    if not element_children:
        return True
    if len(element_children) == 1:
        only_child = element_children[0]
        if _is_blocky_element(only_child):
            return True
        return False

    # Safe indentation rule: only insert inter-element whitespace when we won't
    # be placing it between two adjacent inline/phrasing elements.
    prev_is_blocky = _is_blocky_element(element_children[0])
    for child in element_children[1:]:
        current_is_blocky = _is_blocky_element(child)
        if not prev_is_blocky and not current_is_blocky:
            return False
        prev_is_blocky = current_is_blocky
    return True


def _node_to_html(node: Any, indent: int = 0, indent_size: int = 2, *, in_pre: bool) -> str:
    """Helper to convert a node to HTML using an explicit stack."""
    tasks: list[Any] = [("visit", node, indent, in_pre)]
    results: list[str] = []

    while tasks:
        task = tasks.pop()
        kind = task[0]

        if kind == "visit":
            current, current_indent, current_in_pre = task[1], task[2], task[3]
            name: str = current.name

            if name == "#document":
                child_specs = [(child, current_indent, False) for child in current.children if child is not None]
                tasks.append(("collect_join", "\n", False, child_specs, 0, []))
                continue

            prefix = " " * (current_indent * indent_size) if not current_in_pre else ""
            content_pre = current_in_pre or name in WHITESPACE_PRESERVING_ELEMENTS
            newline = "\n" if not content_pre else ""

            if name == "#text":
                text: str | None = current.data
                parent = current.parent
                parent_name = parent.name if parent is not None else None
                if not current_in_pre:
                    text = text.strip() if text else ""
                    results.append(f"{prefix}{_serialize_text_for_parent(text, parent_name)}" if text else "")
                else:
                    results.append(_serialize_text_for_parent(text, parent_name))
                continue

            if name == "#comment":
                results.append(f"{prefix}<!--{_serialize_comment_data(current.data)}-->")
                continue

            if name == "!doctype":
                results.append(f"{prefix}{_serialize_doctype(current)}")
                continue

            if name == "#document-fragment":
                child_specs = [
                    (child, current_indent, current_in_pre) for child in current.children if child is not None
                ]
                tasks.append(("collect_join", newline, True, child_specs, 0, []))
                continue

            open_tag = serialize_start_tag(name, current.attrs)
            close_tag = serialize_end_tag(name)

            if name in VOID_ELEMENTS:
                results.append(f"{prefix}{open_tag}")
                continue

            children: list[Any] = (
                current.template_content.children
                if name == "template" and current.namespace in {None, "html"} and current.template_content is not None
                else current.children
            )

            if not children:
                results.append(f"{prefix}{open_tag}{close_tag}")
                continue

            if not content_pre:
                all_text = True
                for child in children:
                    if child is None:
                        continue
                    if child.name != "#text":
                        all_text = False
                        break
                if all_text:
                    text_content = current.to_text(separator="", strip=False)
                    text_content = _collapse_html_whitespace(text_content)
                    results.append(f"{prefix}{open_tag}{_serialize_text_for_parent(text_content, name)}{close_tag}")
                    continue

            if content_pre:
                child_specs = [(child, current_indent + 1, True) for child in children if child is not None]
                tasks.append(("collect_wrap_join", prefix, open_tag, close_tag, "", child_specs, 0, []))
                continue

            if name in SPECIAL_ELEMENTS:
                can_indent = True
                for child in children:
                    if child is None:
                        continue
                    if child.name == "#comment" or (child.name == "#text" and (child.data or "").strip()):
                        can_indent = False
                        break

                visible_children = [
                    child
                    for child in children
                    if child is not None
                    and not _is_whitespace_text_node(child)
                    and _pretty_renders_nonempty(child, in_pre=content_pre)
                ]
                if can_indent and visible_children:
                    child_specs = [(child, current_indent + 1, False) for child in visible_children]
                    line_templates = [("", [("child", idx)], False) for idx in range(len(child_specs))]
                    tasks.append(
                        ("collect_wrap_lines", prefix, open_tag, close_tag, line_templates, True, child_specs, 0, [])
                    )
                    continue

                has_comment = any(child is not None and child.name == "#comment" for child in children)
                if not has_comment:
                    non_none_children = [child for child in children if child is not None]
                    has_separator = False
                    for child in non_none_children[1:-1]:
                        if child.name != "#text":
                            continue
                        data = child.data or ""
                        if data.strip() == "" and _is_formatting_whitespace_text(data):
                            has_separator = True
                            break

                    if has_separator:
                        items: list[Any] = []
                        last_was_sep = False
                        for child in non_none_children:
                            if child.name == "#text":
                                data = child.data or ""
                                if data.strip() == "" and _is_formatting_whitespace_text(data):
                                    if not last_was_sep:
                                        items.append(_FORMAT_SEP)
                                        last_was_sep = True
                                    continue
                            items.append(child)
                            last_was_sep = False

                        while items and items[0] is _FORMAT_SEP:
                            items.pop(0)
                        while items and items[-1] is _FORMAT_SEP:
                            items.pop()

                        runs: list[list[Any]] = []
                        current_run: list[Any] = []
                        for item in items:
                            if item is _FORMAT_SEP:
                                runs.append(current_run)
                                current_run = []
                                continue
                            current_run.append(item)
                        runs.append(current_run)
                        runs = [run for run in runs if run]

                        run_child_specs: list[tuple[Any, int, bool]] = []
                        run_line_templates: list[tuple[str, list[tuple[str, Any]], bool]] = []
                        can_apply = True
                        for run in runs:
                            blocky_elements = [
                                child
                                for child in run
                                if child.name not in {"#text", "#comment"} and _is_blocky_element(child)
                            ]
                            if blocky_elements and len(run) != 1:
                                can_apply = False
                                break

                            if len(run) == 1 and run[0].name != "#text":
                                idx = len(run_child_specs)
                                run_child_specs.append((run[0], current_indent + 1, False))
                                run_line_templates.append(("", [("child", idx)], False))
                                continue

                            parts_template: list[tuple[str, Any]] = []
                            for child in run:
                                if child.name == "#text":
                                    data = child.data or ""
                                    if not data.strip():
                                        parts_template.append(("lit", data))
                                    else:
                                        parts_template.append(
                                            ("lit", _escape_text(_normalize_formatting_whitespace(data)))
                                        )
                                    continue

                                idx = len(run_child_specs)
                                run_child_specs.append((child, 0, False))
                                parts_template.append(("child", idx))

                            run_line_templates.append(
                                (" " * ((current_indent + 1) * indent_size), parts_template, False)
                            )

                        if can_apply and run_line_templates:
                            tasks.append(
                                (
                                    "collect_wrap_lines",
                                    prefix,
                                    open_tag,
                                    close_tag,
                                    run_line_templates,
                                    True,
                                    run_child_specs,
                                    0,
                                    [],
                                )
                            )
                            continue

            if not _should_pretty_indent_children(children):
                if name in SPECIAL_ELEMENTS:
                    has_comment = any(child is not None and child.name == "#comment" for child in children)
                    if not has_comment:
                        has_blocky_child = any(
                            child is not None
                            and child.name not in {"#text", "#comment"}
                            and _is_layout_blocky_element(child)
                            for child in children
                        )
                        has_non_whitespace_text = any(
                            child is not None and child.name == "#text" and (child.data or "").strip()
                            for child in children
                        )

                        if has_blocky_child and has_non_whitespace_text:
                            child_specs = []
                            mixed_line_templates: list[tuple[str, list[tuple[str, Any]], bool]] = []
                            inline_parts: list[tuple[str, Any]] = []

                            first_non_none_index = None
                            last_non_none_index = None
                            for i, child in enumerate(children):
                                if child is None:
                                    continue
                                if first_non_none_index is None:
                                    first_non_none_index = i
                                last_non_none_index = i

                            inline_line_prefix = " " * ((current_indent + 1) * indent_size)

                            def flush_inline_parts(
                                target_lines: list[tuple[str, list[tuple[str, Any]], bool]] = mixed_line_templates,
                                line_prefix: str = inline_line_prefix,
                            ) -> None:
                                nonlocal inline_parts
                                if inline_parts:
                                    target_lines.append((line_prefix, inline_parts, True))
                                    inline_parts = []

                            for i, child in enumerate(children):
                                if child is None:
                                    continue

                                if child.name == "#text":
                                    data = child.data or ""
                                    if not data.strip():
                                        if i == first_non_none_index or i == last_non_none_index:
                                            continue
                                        if "\n" in data or "\r" in data or "\t" in data or len(data) > 2:
                                            flush_inline_parts()
                                        else:
                                            inline_parts.append(("lit", data))
                                        continue

                                    inline_parts.append(("lit", _escape_text(_normalize_formatting_whitespace(data))))
                                    continue

                                if _is_layout_blocky_element(child):
                                    flush_inline_parts()
                                    idx = len(child_specs)
                                    child_specs.append((child, current_indent + 1, False))
                                    mixed_line_templates.append(("", [("child", idx)], False))
                                    continue

                                idx = len(child_specs)
                                child_specs.append((child, 0, False))
                                inline_parts.append(("child", idx))

                            flush_inline_parts()

                            if mixed_line_templates:  # pragma: no branch
                                tasks.append(
                                    (
                                        "collect_wrap_lines",
                                        prefix,
                                        open_tag,
                                        close_tag,
                                        mixed_line_templates,
                                        True,
                                        child_specs,
                                        0,
                                        [],
                                    )
                                )
                                continue

                    has_comment = False
                    has_element = False
                    has_whitespace_between_elements = False
                    first_element_index = None
                    last_element_index = None
                    previous_was_element = False
                    saw_whitespace_since_last_element = False
                    for i, child in enumerate(children):
                        if child is None:
                            continue
                        if child.name == "#comment":
                            has_comment = True
                            break
                        if child.name == "#text":
                            if previous_was_element and not (child.data or "").strip():
                                saw_whitespace_since_last_element = True
                            continue

                        has_element = True
                        if first_element_index is None:
                            first_element_index = i
                        last_element_index = i
                        if previous_was_element and saw_whitespace_since_last_element:
                            has_whitespace_between_elements = True
                        previous_was_element = True
                        saw_whitespace_since_last_element = False

                    can_indent_non_whitespace_text = True
                    if has_element and first_element_index is not None and last_element_index is not None:
                        for i, child in enumerate(children):
                            if child is None or child.name != "#text" or not (child.data or "").strip():
                                continue
                            if i < first_element_index or first_element_index < i < last_element_index:
                                can_indent_non_whitespace_text = False
                                break

                    if (
                        has_element
                        and has_whitespace_between_elements
                        and not has_comment
                        and can_indent_non_whitespace_text
                    ):
                        child_specs = []
                        element_line_templates: list[tuple[str, list[tuple[str, Any]], bool]] = []
                        for child in children:
                            if child is None:
                                continue
                            if child.name == "#text":
                                text = _collapse_html_whitespace(child.data or "")
                                if text:
                                    element_line_templates.append(
                                        (
                                            " " * ((current_indent + 1) * indent_size),
                                            [("lit", _escape_text(text))],
                                            False,
                                        )
                                    )
                                continue
                            if not _pretty_renders_nonempty(child, in_pre=content_pre):
                                continue
                            idx = len(child_specs)
                            child_specs.append((child, current_indent + 1, False))
                            element_line_templates.append(("", [("child", idx)], False))

                        if element_line_templates:
                            tasks.append(
                                (
                                    "collect_wrap_lines",
                                    prefix,
                                    open_tag,
                                    close_tag,
                                    element_line_templates,
                                    True,
                                    child_specs,
                                    0,
                                    [],
                                )
                            )
                            continue

                child_specs = []
                compact_parts_template: list[tuple[str, Any]] = []
                first_non_none_index = None
                last_non_none_index = None
                for i, child in enumerate(children):
                    if child is None:
                        continue
                    if first_non_none_index is None:
                        first_non_none_index = i
                    last_non_none_index = i

                for i, child in enumerate(children):
                    if child is None:
                        continue
                    if child.name == "#text":
                        data = child.data or ""
                        if not data.strip():
                            if i == first_non_none_index or i == last_non_none_index:
                                continue
                            if "\n" in data or "\r" in data or "\t" in data or len(data) > 2:
                                compact_parts_template.append(("lit", " "))
                                continue
                        data = _normalize_formatting_whitespace(data)
                        compact_parts_template.append(("lit", _escape_text(data)))
                        continue

                    idx = len(child_specs)
                    child_specs.append((child, 0, False))
                    compact_parts_template.append(("child", idx))

                tasks.append(
                    ("collect_wrap_parts", prefix, open_tag, close_tag, compact_parts_template, child_specs, 0, [])
                )
                continue

            child_specs = [
                (child, current_indent + 1, content_pre)
                for child in children
                if child is not None and (content_pre or not _is_whitespace_text_node(child))
            ]
            line_templates = [("", [("child", idx)], False) for idx in range(len(child_specs))]
            tasks.append(
                ("collect_wrap_lines", prefix, open_tag, close_tag, line_templates, False, child_specs, 0, [])
            )
            continue

        if kind == "collect_join":
            sep, filter_empty, child_specs, index, child_results = task[1], task[2], task[3], task[4], task[5]
            if index:
                child_results.append(results.pop())
            if index < len(child_specs):
                child, child_indent, child_in_pre = child_specs[index]
                tasks.append(("collect_join", sep, filter_empty, child_specs, index + 1, child_results))
                tasks.append(("visit", child, child_indent, child_in_pre))
                continue
            if filter_empty:
                child_results = [value for value in child_results if value]
            results.append(sep.join(child_results))
            continue

        if kind == "collect_wrap_join":
            prefix, open_tag, close_tag, sep, child_specs, index, child_results = (
                task[1],
                task[2],
                task[3],
                task[4],
                task[5],
                task[6],
                task[7],
            )
            if index:
                child_results.append(results.pop())
            if index < len(child_specs):
                child, child_indent, child_in_pre = child_specs[index]
                tasks.append(
                    ("collect_wrap_join", prefix, open_tag, close_tag, sep, child_specs, index + 1, child_results)
                )
                tasks.append(("visit", child, child_indent, child_in_pre))
                continue
            results.append(f"{prefix}{open_tag}{sep.join(child_results)}{close_tag}")
            continue

        if kind == "collect_wrap_parts":
            prefix, open_tag, close_tag, parts_template, child_specs, index, child_results = (
                task[1],
                task[2],
                task[3],
                task[4],
                task[5],
                task[6],
                task[7],
            )
            if index:
                child_results.append(results.pop())
            if index < len(child_specs):
                child, child_indent, child_in_pre = child_specs[index]
                tasks.append(
                    (
                        "collect_wrap_parts",
                        prefix,
                        open_tag,
                        close_tag,
                        parts_template,
                        child_specs,
                        index + 1,
                        child_results,
                    )
                )
                tasks.append(("visit", child, child_indent, child_in_pre))
                continue
            parts: list[str] = []
            for part_kind, value in parts_template:
                if part_kind == "lit":
                    parts.append(value)
                else:
                    parts.append(child_results[value])
            results.append(f"{prefix}{open_tag}{''.join(parts)}{close_tag}")
            continue

        if kind != "collect_wrap_lines":  # pragma: no cover
            raise RuntimeError(f"Unknown serialization task kind: {kind}")
        prefix, open_tag, close_tag, line_templates, skip_empty, child_specs, index, child_results = (
            task[1],
            task[2],
            task[3],
            task[4],
            task[5],
            task[6],
            task[7],
            task[8],
        )
        if index:
            child_results.append(results.pop())
        if index < len(child_specs):
            child, child_indent, child_in_pre = child_specs[index]
            tasks.append(
                (
                    "collect_wrap_lines",
                    prefix,
                    open_tag,
                    close_tag,
                    line_templates,
                    skip_empty,
                    child_specs,
                    index + 1,
                    child_results,
                )
            )
            tasks.append(("visit", child, child_indent, child_in_pre))
            continue
        lines: list[str] = []
        for line_prefix, parts_template, strip_spaces in line_templates:
            line_parts: list[str] = []
            for part_kind, value in parts_template:
                if part_kind == "lit":
                    line_parts.append(value)
                else:
                    line_parts.append(child_results[value])
            line = "".join(line_parts)
            if strip_spaces:
                line = line.strip(" ")
            if line or not skip_empty:
                lines.append(f"{line_prefix}{line}")
        results.append("\n".join([f"{prefix}{open_tag}", *lines, f"{prefix}{close_tag}"]))

    return results[-1] if results else ""


def to_test_format(node: Any, indent: int = 0) -> str:
    """Convert node to html5lib test format string.

    This format is used by html5lib-tests for validating parser output.
    Uses '| ' prefixes and specific indentation rules.
    """
    if node.name in {"#document", "#document-fragment"}:
        parts = [_node_to_test_format(child, 0) for child in node.children]
        return "\n".join(parts)
    return _node_to_test_format(node, indent)


def _node_to_test_format(node: Any, indent: int) -> str:
    """Helper to convert a node to test format."""
    if node.name == "#comment":
        comment: str = node.data or ""
        return f"| {' ' * indent}<!-- {comment} -->"

    if node.name == "!doctype":
        return _doctype_to_test_format(node)

    if node.name == "#text":
        text: str = node.data or ""
        return f'| {" " * indent}"{text}"'

    # Regular element
    line = f"| {' ' * indent}<{_qualified_name(node)}>"
    attribute_lines = _attrs_to_test_format(node, indent)

    # Template special handling (only HTML namespace templates have template_content)
    if node.name == "template" and node.namespace in {None, "html"} and node.template_content is not None:
        sections: list[str] = [line]
        if attribute_lines:
            sections.extend(attribute_lines)
        content_line = f"| {' ' * (indent + 2)}content"
        sections.append(content_line)
        sections.extend(_node_to_test_format(child, indent + 4) for child in node.template_content.children)
        return "\n".join(sections)

    # Regular element with children
    child_lines = [_node_to_test_format(child, indent + 2) for child in node.children] if node.children else []

    sections = [line]
    if attribute_lines:
        sections.extend(attribute_lines)
    sections.extend(child_lines)
    return "\n".join(sections)


def _qualified_name(node: Any) -> str:
    """Get the qualified name of a node (with namespace prefix if needed)."""
    if node.namespace and node.namespace not in {"html", None}:
        return f"{node.namespace} {node.name}"
    return str(node.name)


def _attrs_to_test_format(node: Any, indent: int) -> list[str]:
    """Format element attributes for test output."""
    if not node.attrs:
        return []

    formatted: list[str] = []
    padding = " " * (indent + 2)

    # Prepare display names for sorting
    display_attrs: list[tuple[str, str]] = []
    namespace: str | None = node.namespace
    for attr_name, attr_value in node.attrs.items():
        value = attr_value or ""
        display_name = attr_name
        if namespace and namespace not in {None, "html"}:
            lower_name = attr_name.lower()
            if lower_name in FOREIGN_ATTRIBUTE_ADJUSTMENTS:
                display_name = attr_name.replace(":", " ")
        display_attrs.append((display_name, value))

    # Sort by display name for canonical test output
    display_attrs.sort(key=lambda x: x[0])

    for display_name, value in display_attrs:
        formatted.append(f'| {padding}{display_name}="{value}"')
    return formatted


def _doctype_to_test_format(node: Any) -> str:
    """Format DOCTYPE node for test output."""
    doctype = node.data

    name: str = doctype.name or ""
    public_id: str | None = doctype.public_id
    system_id: str | None = doctype.system_id

    parts: list[str] = ["| <!DOCTYPE"]
    if name:
        parts.append(f" {name}")
    else:
        parts.append(" ")

    if public_id is not None or system_id is not None:
        pub = public_id if public_id is not None else ""
        sys = system_id if system_id is not None else ""
        parts.append(f' "{pub}"')
        parts.append(f' "{sys}"')

    parts.append(">")
    return "".join(parts)
