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


def _escape_html_text(value: str) -> str:
    if not value:
        return ""
    if "&" not in value and "<" not in value and ">" not in value:
        return value
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


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
    parts: list[str] = ["<", name]
    if attrs:
        parts_append = parts.append
        for key, value in attrs.items():
            if minimize_boolean_attributes:
                if value is None or value == "" or value == key:
                    parts_append(" " + key)
                    continue
                if len(value) == len(key) and value.lower() == key:
                    parts_append(" " + key)
                    continue

            if value is None or value == "":
                parts_append(f' {key}=""')
                continue

            if not quote_attr_values and _can_unquote_attr_value(value):
                escaped = value.replace("&", "&amp;").replace("<", "&lt;")
                parts_append(f" {key}={escaped}")
                continue

            quote = _choose_attr_quote(value, quote_char)
            escaped = _escape_attr_value(value, quote)
            parts_append(f" {key}={quote}{escaped}{quote}")

    if use_trailing_solidus and is_void:
        parts.append(" />")
    else:
        parts.append(">")
    return "".join(parts)


def serialize_end_tag(name: str) -> str:
    return f"</{name}>"


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
    if node.name == "#document":
        # Document root - just render children
        parts: list[str] = []
        for child in node.children or []:
            parts.append(_node_to_html(child, indent, indent_size, pretty, in_pre=False))
        html = "\n".join(parts) if pretty else "".join(parts)
    else:
        html = _node_to_html(node, indent, indent_size, pretty, in_pre=False)

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
        children = node.children or []
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
        children = node.children or []
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


def _node_to_html(node: Any, indent: int = 0, indent_size: int = 2, pretty: bool = True, *, in_pre: bool) -> str:
    """Helper to convert a node to HTML."""
    name: str = node.name

    if not pretty:
        # Compact mode: no indentation/newlines. Keep this hot path tight.

        if name == "#text":
            return _escape_text(node.data) if node.data else ""

        if name == "#comment":
            return f"<!--{node.data or ''}-->"

        if name == "!doctype":
            return "<!DOCTYPE html>"

        if name == "#document-fragment":
            compact_parts: list[str] = []
            for child in node.children or []:
                if child is None:  # pragma: no cover
                    continue
                compact_parts.append(_node_to_html(child, 0, indent_size, pretty=False, in_pre=False))
            return "".join(compact_parts)

        compact_attrs: dict[str, str | None] = node.attrs or {}
        open_tag = serialize_start_tag(name, compact_attrs)

        if name in VOID_ELEMENTS:
            return open_tag

        if name == "template" and node.namespace in {None, "html"} and node.template_content is not None:
            compact_children: list[Any] = node.template_content.children or []
        else:
            compact_children = node.children or []

        if not compact_children:
            return f"{open_tag}{serialize_end_tag(name)}"

        compact_parts = [open_tag]
        for child in compact_children:
            if child is None:  # pragma: no cover
                continue
            compact_parts.append(_node_to_html(child, 0, indent_size, pretty=False, in_pre=False))
        compact_parts.append(serialize_end_tag(name))
        return "".join(compact_parts)

    prefix = " " * (indent * indent_size) if not in_pre else ""
    content_pre = in_pre or name in WHITESPACE_PRESERVING_ELEMENTS
    newline = "\n" if not content_pre else ""

    # Text node
    if name == "#text":
        text: str | None = node.data
        if not in_pre:
            text = text.strip() if text else ""
            if text:
                return f"{prefix}{_escape_text(text)}"
            return ""
        return _escape_text(text) if text else ""

    # Comment node
    if name == "#comment":
        return f"{prefix}<!--{node.data or ''}-->"

    # Doctype
    if name == "!doctype":
        return f"{prefix}<!DOCTYPE html>"

    # Document fragment
    if name == "#document-fragment":
        parts: list[str] = []
        for child in node.children or []:
            child_html = _node_to_html(child, indent, indent_size, pretty, in_pre=in_pre)
            if child_html:
                parts.append(child_html)
        return newline.join(parts)

    # Element node
    attrs: dict[str, str | None] = node.attrs or {}

    # Build opening tag
    open_tag = serialize_start_tag(name, attrs)

    # Void elements
    if name in VOID_ELEMENTS:
        return f"{prefix}{open_tag}"

    # Elements with children
    # Template special handling: HTML templates store contents in `template_content`.
    if name == "template" and node.namespace in {None, "html"} and node.template_content is not None:
        children: list[Any] = node.template_content.children or []
    else:
        children = node.children or []
    if not children:
        return f"{prefix}{open_tag}{serialize_end_tag(name)}"

    if not content_pre:
        # Check if all children are text-only (inline rendering)
        all_text = True
        for child in children:
            if child is None:
                continue
            if child.name != "#text":
                all_text = False
                break

        if all_text:
            # Serializer controls sanitization at the to_html() entry point; avoid
            # implicit re-sanitization during rendering.
            text_content = node.to_text(separator="", strip=False)
            text_content = _collapse_html_whitespace(text_content)
            return f"{prefix}{open_tag}{_escape_text(text_content)}{serialize_end_tag(name)}"

    if content_pre:
        inner = "".join(
            _node_to_html(child, indent + 1, indent_size, pretty, in_pre=True)
            for child in children
            if child is not None
        )
        return f"{prefix}{open_tag}{inner}{serialize_end_tag(name)}"

    if not content_pre and name in SPECIAL_ELEMENTS:
        # For block-ish containers that only have element children (and/or
        # whitespace-only text nodes), prefer a multiline layout for readability
        # even when children are inline elements.
        can_indent = True
        for child in children:
            if child is None:
                continue
            if child.name == "#comment":
                can_indent = False
                break
            if child.name == "#text" and (child.data or "").strip():
                can_indent = False
                break

        if can_indent:
            inner_lines: list[str] = []
            for child in children:
                if child is None:
                    continue
                if _is_whitespace_text_node(child):
                    continue
                child_html = _node_to_html(child, indent + 1, indent_size, pretty, in_pre=content_pre)
                if child_html:
                    inner_lines.append(child_html)

            if inner_lines:
                parts = [f"{prefix}{open_tag}"]
                parts.extend(inner_lines)
                parts.append(f"{prefix}{serialize_end_tag(name)}")
                return "\n".join(parts)

        # Smart pretty-printing: if the author already inserted formatting whitespace
        # between siblings, we can split into "inline runs" and put each run on its
        # own line without introducing new inter-token whitespace.
        has_comment = any(child is not None and child.name == "#comment" for child in children)
        if not has_comment:
            non_none_children: list[Any] = [child for child in children if child is not None]

            # Only enable this mode if there is at least one formatting whitespace text node
            # between non-whitespace siblings.
            has_separator = False
            for child in non_none_children[1:-1]:
                if child.name != "#text":
                    continue
                data = child.data or ""
                if data.strip() != "":
                    continue
                if _is_formatting_whitespace_text(data):
                    has_separator = True
                    break

            if has_separator:
                # Build runs by splitting on formatting whitespace text nodes.
                # Keep small spacing nodes (" " or "  ") inside runs.
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

                # Only apply if we can render each run either as a single blocky element
                # (possibly multiline) or as a single-line inline run.
                smart_lines: list[str] = []
                can_apply = True
                for run in runs:
                    blocky_elements = [c for c in run if c.name not in {"#text", "#comment"} and _is_blocky_element(c)]
                    if blocky_elements and len(run) != 1:
                        can_apply = False
                        break

                    if len(run) == 1 and run[0].name != "#text":
                        child_html = _node_to_html(run[0], indent + 1, indent_size, pretty=True, in_pre=content_pre)
                        smart_lines.append(child_html)
                        continue

                    # Inline run: render on one line.
                    run_parts: list[str] = []
                    for c in run:
                        if c.name == "#text":
                            data = c.data or ""
                            if not data.strip():
                                # Formatting whitespace never appears inside runs (it is used as a separator).
                                # Preserve intentional tiny spacing.
                                run_parts.append(data)
                                continue

                            run_parts.append(_escape_text(_normalize_formatting_whitespace(data)))
                            continue

                        # Render inline elements without their own leading indentation.
                        child_html = _node_to_html(c, 0, indent_size, pretty=True, in_pre=content_pre)
                        run_parts.append(child_html)

                    smart_lines.append(f"{' ' * ((indent + 1) * indent_size)}{''.join(run_parts)}")

                if can_apply and smart_lines:
                    return f"{prefix}{open_tag}\n" + "\n".join(smart_lines) + f"\n{prefix}{serialize_end_tag(name)}"

    if not content_pre and not _should_pretty_indent_children(children):
        # For block-ish elements that contain only element children and whitespace-only
        # text nodes, we can still format each child on its own line (only when there
        # is already whitespace separating element siblings).
        if name in SPECIAL_ELEMENTS:
            # Mixed content in block-ish containers: if we encounter a blocky child
            # (e.g. <ul>) adjacent to inline text, printing everything on one line
            # both hurts readability and can lose indentation inside the block subtree.
            # In that case, put inline runs and blocky children on their own lines.
            has_comment = any(child is not None and child.name == "#comment" for child in children)
            if not has_comment:
                has_blocky_child = any(
                    child is not None and child.name not in {"#text", "#comment"} and _is_layout_blocky_element(child)
                    for child in children
                )
                has_non_whitespace_text = any(
                    child is not None and child.name == "#text" and (child.data or "").strip() for child in children
                )

                if has_blocky_child and has_non_whitespace_text:
                    mixed_multiline_lines: list[str] = []
                    inline_parts: list[str] = []

                    mixed_first_non_none_index: int | None = None
                    mixed_last_non_none_index: int | None = None
                    for i, child in enumerate(children):
                        if child is None:
                            continue
                        if mixed_first_non_none_index is None:
                            mixed_first_non_none_index = i
                        mixed_last_non_none_index = i

                    def flush_inline() -> None:
                        if not inline_parts:
                            return
                        line = "".join(inline_parts).strip(" ")
                        inline_parts.clear()
                        if line:
                            mixed_multiline_lines.append(f"{' ' * ((indent + 1) * indent_size)}{line}")

                    for i, child in enumerate(children):
                        if child is None:
                            continue

                        if child.name == "#text":
                            data = child.data or ""
                            if not data.strip():
                                # Drop leading/trailing formatting whitespace.
                                if i == mixed_first_non_none_index or i == mixed_last_non_none_index:
                                    continue
                                # Preserve intentional small spacing, but treat formatting whitespace
                                # as a separator between inline runs (new line).
                                if "\n" in data or "\r" in data or "\t" in data or len(data) > 2:
                                    flush_inline()
                                else:
                                    inline_parts.append(data)
                                continue

                            data = _normalize_formatting_whitespace(data)
                            inline_parts.append(_escape_text(data))
                            continue

                        if _is_layout_blocky_element(child):
                            flush_inline()
                            mixed_multiline_lines.append(
                                _node_to_html(child, indent + 1, indent_size, pretty=True, in_pre=content_pre)
                            )
                            continue

                        # Inline element: keep it in the current line without leading indentation.
                        inline_parts.append(_node_to_html(child, 0, indent_size, pretty=True, in_pre=content_pre))

                    flush_inline()
                    inner = "\n".join(line for line in mixed_multiline_lines if line)
                    return f"{prefix}{open_tag}\n{inner}\n{prefix}{serialize_end_tag(name)}"

            has_comment = False
            has_element = False
            has_whitespace_between_elements = False

            first_element_index: int | None = None
            last_element_index: int | None = None

            previous_was_element = False
            saw_whitespace_since_last_element = False
            for i, child in enumerate(children):
                if child is None:
                    continue
                if child.name == "#comment":
                    has_comment = True
                    break
                if child.name == "#text":
                    # Track whether there is already whitespace between element siblings.
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
                    if child is None or child.name != "#text":
                        continue
                    if not (child.data or "").strip():
                        continue
                    # Only allow non-whitespace text *after* the last element.
                    # Leading text or text between elements could gain new spaces
                    # due to indentation/newlines.
                    if i < first_element_index or first_element_index < i < last_element_index:
                        can_indent_non_whitespace_text = False
                        break

            if has_element and has_whitespace_between_elements and not has_comment and can_indent_non_whitespace_text:
                element_multiline_lines: list[str] = []
                for child in children:
                    if child is None:
                        continue
                    if child.name == "#text":
                        text = _collapse_html_whitespace(child.data or "")
                        if text:
                            element_multiline_lines.append(f"{' ' * ((indent + 1) * indent_size)}{_escape_text(text)}")
                        continue
                    child_html = _node_to_html(child, indent + 1, indent_size, pretty=True, in_pre=content_pre)
                    if child_html:
                        element_multiline_lines.append(child_html)
                if element_multiline_lines:
                    inner = "\n".join(element_multiline_lines)
                    return f"{prefix}{open_tag}\n{inner}\n{prefix}{serialize_end_tag(name)}"

        inner_parts: list[str] = []

        compact_first_non_none_index: int | None = None
        compact_last_non_none_index: int | None = None
        for i, child in enumerate(children):
            if child is None:
                continue
            if compact_first_non_none_index is None:
                compact_first_non_none_index = i
            compact_last_non_none_index = i

        for i, child in enumerate(children):
            if child is None:
                continue

            if child.name == "#text":
                data = child.data or ""
                if not data.strip():
                    # Drop leading/trailing formatting whitespace in compact mode.
                    if i == compact_first_non_none_index or i == compact_last_non_none_index:
                        continue
                    # Preserve intentional small spacing, but collapse large formatting gaps.
                    if "\n" in data or "\r" in data or "\t" in data or len(data) > 2:
                        inner_parts.append(" ")
                        continue

                data = _normalize_formatting_whitespace(data)
                child_html = _escape_text(data) if data else ""
            else:
                # Even when we can't safely insert whitespace *between* siblings, we can
                # still pretty-print each element subtree to improve readability.
                child_html = _node_to_html(child, 0, indent_size, pretty=True, in_pre=content_pre)
            if child_html:
                inner_parts.append(child_html)

        return f"{prefix}{open_tag}{''.join(inner_parts)}{serialize_end_tag(name)}"

    # Render with child indentation
    parts = [f"{prefix}{open_tag}"]
    for child in children:
        if not content_pre and _is_whitespace_text_node(child):
            continue
        child_html = _node_to_html(child, indent + 1, indent_size, pretty, in_pre=content_pre)
        parts.append(child_html)
    parts.append(f"{prefix}{serialize_end_tag(name)}")
    return newline.join(parts)


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
