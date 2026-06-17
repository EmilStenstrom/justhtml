from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal, TypeAlias

from justhtml.core.constants import (
    FOREIGN_BREAKOUT_ELEMENTS,
    HTML_INTEGRATION_POINT_SET,
    MATHML_TEXT_INTEGRATION_POINT_SET,
    SVG_TAG_NAME_ADJUSTMENTS,
)
from justhtml.core.entities import decode_entities_in_text

from . import scanner as _scanner
from .encoding import decode_html

if TYPE_CHECKING:
    from collections.abc import Generator

StartEvent: TypeAlias = tuple[Literal["start"], tuple[str, dict[str, str | None]]]
EndEvent: TypeAlias = tuple[Literal["end"], str]
TextEvent: TypeAlias = tuple[Literal["text"], str]
CommentEvent: TypeAlias = tuple[Literal["comment"], str]
DoctypeEvent: TypeAlias = tuple[Literal["doctype"], tuple[str | None, str | None, str | None]]
StreamEvent: TypeAlias = StartEvent | EndEvent | TextEvent | CommentEvent | DoctypeEvent

_DOCTYPE_RE = re.compile(
    r"""\s*([^\s>]+)(?:\s*(PUBLIC|SYSTEM)\s*(?:(?:"([^"]*)"|'([^']*)')\s*(?:"([^"]*)"|'([^']*)')?)?)?""",
    re.IGNORECASE,
)
_RAWTEXT_TAGS = frozenset({"script", "style", "iframe", "noembed", "noframes", "noscript", "xmp"})
_RCDATA_TAGS = frozenset({"textarea", "title"})
_PLAINTEXT_TAGS = frozenset({"plaintext"})
_SPACE = _scanner.SPACE
_TAG_NAME_STOP = _scanner.TAG_NAME_STOP
_ATTR_NAME_STOP = _scanner.ATTR_NAME_STOP
_ATTR_VALUE_STOP = _scanner.ATTR_VALUE_STOP
_TAG_END_NAME_STOP = _scanner.TAG_END_NAME_STOP


class _StreamNode:
    __slots__ = ("attrs", "name", "namespace")

    attrs: dict[str, str | None]
    name: str
    namespace: str

    def __init__(self, name: str, namespace: str, attrs: dict[str, str | None] | None = None) -> None:
        self.attrs = attrs or {}
        self.name = name
        self.namespace = namespace


class _StreamScanner:
    __slots__ = ("_html", "_lower", "_open_elements")

    _html: str
    _lower: str
    _open_elements: list[_StreamNode]

    def __init__(self, html: str) -> None:
        self._html = html
        self._lower = html.lower()
        self._open_elements = []

    def scan(self) -> Generator[StreamEvent, None, None]:
        html = self._html
        end = len(html)
        pos = 0
        text_buffer: list[str] = []

        while pos < end:
            lt = html.find("<", pos, end)
            if lt == -1:
                self._append_text(text_buffer, html[pos:end])
                break
            if lt > pos:
                self._append_text(text_buffer, html[pos:lt])

            pos = lt + 1
            if pos >= end:
                self._append_text(text_buffer, "<")
                break

            ch = html[pos]
            if ch == "!":
                handled, event, new_pos, text = self._parse_markup_declaration(pos + 1, end)
                if handled:
                    if text is not None:
                        self._append_text(text_buffer, text, raw=True)
                    elif event is not None:
                        yield from self._flush_text(text_buffer)
                        yield event
                    pos = new_pos
                    continue
                self._append_text(text_buffer, "<")
                continue

            if ch == "/":
                parsed = self._parse_end_tag(pos + 1, end)
                if parsed is None:
                    self._append_text(text_buffer, "</")
                    pos += 1
                    continue
                name, new_pos = parsed
                yield from self._flush_text(text_buffer)
                yield ("end", name)
                self._pop_for_end_tag(name)
                pos = new_pos
                continue

            if not _is_ascii_alpha(ch):
                self._append_text(text_buffer, "<")
                continue

            parsed_start = self._parse_start_tag(pos, end)
            if parsed_start is None:
                self._append_text(text_buffer, "<")
                continue

            name, attrs, self_closing, new_pos = parsed_start
            yield from self._flush_text(text_buffer)
            yield ("start", (name, attrs.copy()))

            namespace = self._namespace_for_start_tag(name, attrs)
            if not (self_closing and namespace not in {None, "html"}):
                adjusted_name = self._adjusted_name_for_namespace(name, namespace)
                self._open_elements.append(_StreamNode(adjusted_name, namespace, attrs.copy()))

            if namespace in {None, "html"}:
                if name in _PLAINTEXT_TAGS:
                    self._append_text(text_buffer, html[new_pos:end], raw=True)
                    pos = end
                    continue
                if name in _RAWTEXT_TAGS or name in _RCDATA_TAGS:
                    close, after_close = self._find_rawtext_end_tag(name, new_pos, end)
                    text_end = end if close is None else close
                    self._append_text(text_buffer, html[new_pos:text_end], raw=name not in _RCDATA_TAGS)
                    if close is None:
                        pos = end
                        continue
                    yield from self._flush_text(text_buffer)
                    yield ("end", name)
                    self._pop_for_end_tag(name)
                    pos = after_close
                    continue

            pos = new_pos

        yield from self._flush_text(text_buffer)

    def _parse_markup_declaration(
        self, pos: int, end: int
    ) -> tuple[bool, CommentEvent | DoctypeEvent | None, int, str | None]:
        html = self._html
        lower = self._lower

        if html.startswith("--", pos):
            comment_start = pos + 2
            comment_end = html.find("-->", comment_start, end)
            if comment_end == -1:
                return True, ("comment", html[comment_start:end].replace("\0", "\ufffd")), end, None
            return True, ("comment", html[comment_start:comment_end].replace("\0", "\ufffd")), comment_end + 3, None

        if lower.startswith("doctype", pos):
            tag_end = self._find_tag_end(pos + 7, end)
            if tag_end == -1:
                tag_end = end
                next_pos = end
            else:
                next_pos = tag_end + 1
            data = html[pos + 7 : tag_end]
            match = _DOCTYPE_RE.match(data)
            if not match:
                return True, ("doctype", (None, None, None)), next_pos, None
            name = match.group(1)
            if name and not name.islower():
                name = name.lower()
            public_id = match.group(3) if match.group(3) is not None else match.group(4)
            system_id = match.group(5) if match.group(5) is not None else match.group(6)
            return True, ("doctype", (name, public_id, system_id)), next_pos, None

        if lower.startswith("[cdata[", pos) and self._in_foreign_context():
            cdata_start = pos + 7
            cdata_end = html.find("]]>", cdata_start, end)
            if cdata_end == -1:
                return True, None, end, html[cdata_start:end]
            return True, None, cdata_end + 3, html[cdata_start:cdata_end]

        comment_end = html.find(">", pos, end)
        if comment_end == -1:
            return True, ("comment", html[pos:end].replace("\0", "\ufffd")), end, None
        return True, ("comment", html[pos:comment_end].replace("\0", "\ufffd")), comment_end + 1, None

    def _parse_start_tag(self, pos: int, end: int) -> tuple[str, dict[str, str | None], bool, int] | None:
        html = self._html
        name_start = pos
        while pos < end and html[pos] not in _TAG_NAME_STOP:
            pos += 1
        if pos == name_start:
            return None
        name = html[name_start:pos]
        if not name.islower():
            name = name.lower()
        attrs, self_closing, pos, tag_closed = self._parse_attrs(pos, end)
        if not tag_closed:
            return None
        return name, attrs, self_closing, pos

    def _parse_end_tag(self, pos: int, end: int) -> tuple[str, int] | None:
        html = self._html
        name_start = pos
        if pos >= end or not _is_ascii_alpha(html[pos]):
            return None
        while pos < end and html[pos] not in _TAG_END_NAME_STOP:
            pos += 1
        if pos == name_start:
            return None
        name = html[name_start:pos]
        if not name.islower():
            name = name.lower()
        tag_end = self._find_tag_end(pos, end)
        return name, end if tag_end == -1 else tag_end + 1

    def _parse_attrs(self, pos: int, end: int) -> tuple[dict[str, str | None], bool, int, bool]:
        html = self._html
        attrs: dict[str, str | None] = {}

        while pos < end:
            while pos < end and html[pos] in _SPACE:
                pos += 1
            if pos >= end:
                return attrs, False, pos, False
            ch = html[pos]
            if ch == ">":
                return attrs, False, pos + 1, True
            if ch == "/" and pos + 1 < end and html[pos + 1] == ">":
                return attrs, True, pos + 2, True

            name_start = pos
            while pos < end and html[pos] not in _ATTR_NAME_STOP:
                pos += 1
            if pos == name_start:
                pos += 1
                continue
            raw_key = html[name_start:pos]
            key = raw_key if raw_key.islower() else raw_key.lower()

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
                        return attrs, False, end, False
                    value = html[value_start:close]
                    pos = close + 1
                else:
                    value_start = pos
                    while pos < end and html[pos] not in _ATTR_VALUE_STOP:
                        pos += 1
                    value = html[value_start:pos]

                if value and "&" in value:
                    value = decode_entities_in_text(value, in_attribute=True)
                if value and "\0" in value:
                    value = value.replace("\0", "\ufffd")

            if key not in attrs:
                attrs[key] = value

        return attrs, False, pos, False

    def _find_rawtext_end_tag(self, name: str, pos: int, end: int) -> tuple[int | None, int]:
        if name == "script":
            return _scanner.find_script_end_tag(self._html, self._lower, pos, end)
        return _scanner.find_rawtext_end_tag(self._html, self._lower, name, pos, end)

    def _find_script_end_tag(self, pos: int, end: int) -> tuple[int | None, int]:
        return _scanner.find_script_end_tag(self._html, self._lower, pos, end)

    def _find_tag_end(self, pos: int, end: int) -> int:
        return _scanner.find_tag_end(self._html, pos, end)

    def _append_text(self, text_buffer: list[str], data: str, *, raw: bool = False) -> None:
        if not data:
            return
        if "\r" in data:
            data = data.replace("\r\n", "\n").replace("\r", "\n")
        if "\0" in data:
            data = data.replace("\0", "\ufffd")
        if not raw and "&" in data:
            data = decode_entities_in_text(data, in_attribute=False)
        if data:
            text_buffer.append(data)

    def _flush_text(self, text_buffer: list[str]) -> Generator[TextEvent, None, None]:
        if not text_buffer:
            return
        text = "".join(text_buffer)
        text_buffer.clear()
        if text:
            yield ("text", text)

    def _font_breaks_out_of_foreign_content(self, attrs: dict[str, str | None]) -> bool:
        for name in attrs:
            if name.lower() in {"color", "face", "size"}:
                return True
        return False

    def _node_attribute_value(self, node: _StreamNode, name: str) -> str | None:
        target = name.lower()
        for attr_name, attr_value in node.attrs.items():
            if attr_name.lower() == target:
                return attr_value or ""
        return None

    def _is_html_integration_point(self, node: _StreamNode) -> bool:
        if node.namespace == "math" and node.name == "annotation-xml":
            encoding = self._node_attribute_value(node, "encoding")
            return encoding is not None and encoding.lower() in {"application/xhtml+xml", "text/html"}
        return (node.namespace, node.name) in HTML_INTEGRATION_POINT_SET

    def _is_mathml_text_integration_point(self, node: _StreamNode) -> bool:
        return (node.namespace, node.name) in MATHML_TEXT_INTEGRATION_POINT_SET

    def _adjusted_name_for_namespace(self, name: str, namespace: str) -> str:
        if namespace == "svg":
            return SVG_TAG_NAME_ADJUSTMENTS.get(name, name)
        return name

    def _namespace_from_html_context(self, name: str) -> str:
        if name == "svg":
            return "svg"
        if name == "math":
            return "math"
        return "html"

    def _namespace_for_start_tag(self, name: str, attrs: dict[str, str | None]) -> str:
        parent = self._open_elements[-1] if self._open_elements else None
        parent_namespace = parent.namespace if parent is not None else "html"

        if parent is not None:
            if self._is_html_integration_point(parent):
                return self._namespace_from_html_context(name)
            if self._is_mathml_text_integration_point(parent) and name not in {"mglyph", "malignmark"}:
                return self._namespace_from_html_context(name)
            if parent_namespace == "math" and parent.name == "annotation-xml" and name == "svg":
                return "svg"

        if parent_namespace not in {None, "html"}:
            breaks_out = name in FOREIGN_BREAKOUT_ELEMENTS or (
                name == "font" and self._font_breaks_out_of_foreign_content(attrs)
            )
            if breaks_out:
                while self._open_elements and self._open_elements[-1].namespace not in {None, "html"}:
                    self._open_elements.pop()
            else:
                return parent_namespace

        return self._namespace_from_html_context(name)

    def _pop_foreign_context(self) -> None:
        while self._open_elements and self._open_elements[-1].namespace not in {None, "html"}:
            self._open_elements.pop()

    def _pop_for_end_tag(self, name: str) -> None:
        if not self._open_elements:
            return

        name_lower = name.lower()
        current = self._open_elements[-1]
        if current.namespace not in {None, "html"} and name_lower in {"br", "p"}:
            self._pop_foreign_context()
            return

        for index in range(len(self._open_elements) - 1, -1, -1):
            node = self._open_elements[index]
            if node.name.lower() == name_lower:
                del self._open_elements[index:]
                return
            if node.namespace in {None, "html"}:
                break

        self._open_elements.pop()

    def _in_foreign_context(self) -> bool:
        if not self._open_elements:
            return False
        node = self._open_elements[-1]
        return node.namespace not in {None, "html"}


def _is_ascii_alpha(ch: str) -> bool:
    return ("a" <= ch <= "z") or ("A" <= ch <= "Z")


def stream(
    html: str | bytes | bytearray | memoryview,
    *,
    encoding: str | None = None,
) -> Generator[StreamEvent, None, None]:
    """
    Stream HTML events from the given HTML string.
    Yields tuples of (event_type, data).
    """
    html_str: str
    if isinstance(html, (bytes, bytearray, memoryview)):
        html_str, _ = decode_html(bytes(html), transport_encoding=encoding)
    else:
        html_str = html
    yield from _StreamScanner(html_str).scan()
