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
    __slots__ = ("_html", "_name_counts", "_open_elements")

    _html: str
    _open_elements: list[_StreamNode]
    _name_counts: dict[str, int]

    def __init__(self, html: str) -> None:
        self._html = html
        self._open_elements = []
        self._name_counts = {}

    def _push_open_element(self, node: _StreamNode) -> None:
        self._open_elements.append(node)
        key = node.name.lower()
        self._name_counts[key] = self._name_counts.get(key, 0) + 1

    def _pop_open_element(self) -> _StreamNode:
        node = self._open_elements.pop()
        key = node.name.lower()
        self._name_counts[key] -= 1
        return node

    def _truncate_open_elements(self, index: int) -> None:
        removed = self._open_elements[index:]
        del self._open_elements[index:]
        for node in removed:
            key = node.name.lower()
            self._name_counts[key] -= 1

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
                        yield from self._flush_text(text_buffer)
                        self._append_text(text_buffer, text, raw=True)
                        yield from self._flush_text(text_buffer)
                    elif event is not None:  # pragma: no branch - declarations always produce event or text
                        yield from self._flush_text(text_buffer)
                        yield event
                    pos = new_pos
                    continue
                # All markup declarations, including bogus declarations, are
                # consumed by _parse_markup_declaration.
                raise AssertionError("unhandled markup declaration")  # pragma: no cover

            if ch == "/":
                next_pos = pos + 1
                if next_pos < end and not _is_ascii_alpha(html[next_pos]):
                    if html[next_pos] == ">":
                        pos = next_pos + 1
                        continue
                    comment_end = html.find(">", next_pos, end)
                    data_end = end if comment_end == -1 else comment_end
                    yield from self._flush_text(text_buffer)
                    yield ("comment", html[next_pos:data_end].replace("\0", "\ufffd"))
                    pos = end if comment_end == -1 else comment_end + 1
                    continue
                parsed = self._parse_end_tag(pos + 1, end)
                if parsed is None:  # pragma: no cover - guarded by ASCII-alpha check
                    # The caller only reaches this point for an ASCII letter.
                    self._append_text(text_buffer, "</")
                    pos += 1
                    continue
                name, new_pos = parsed
                if new_pos == end and html[-1] != ">":
                    pos = end
                    continue
                yield from self._flush_text(text_buffer)
                yield ("end", name)
                self._pop_for_end_tag(name)
                pos = new_pos
                continue

            if ch == "?":
                comment_end = html.find(">", pos + 1, end)
                data_end = end if comment_end == -1 else comment_end
                yield from self._flush_text(text_buffer)
                yield ("comment", html[pos:data_end].replace("\0", "\ufffd"))
                pos = end if comment_end == -1 else comment_end + 1
                continue

            if not _is_ascii_alpha(ch):
                yield from self._flush_text(text_buffer)
                self._append_text(text_buffer, "<")
                continue

            parsed_start = self._parse_start_tag(pos, end)
            if parsed_start is None:
                pos = end
                break

            name, attrs, self_closing, new_pos = parsed_start
            yield from self._flush_text(text_buffer)
            yield ("start", (name, attrs.copy()))

            namespace = self._namespace_for_start_tag(name, attrs)
            if not (self_closing and namespace not in {None, "html"}):
                adjusted_name = self._adjusted_name_for_namespace(name, namespace)
                self._push_open_element(_StreamNode(adjusted_name, namespace, attrs.copy()))

            if namespace in {None, "html"}:
                if name in _PLAINTEXT_TAGS:
                    self._append_text(text_buffer, html[new_pos:end], raw=True, replace_null=True)
                    pos = end
                    continue
                if name in _RAWTEXT_TAGS or name in _RCDATA_TAGS:
                    close, after_close = self._find_rawtext_end_tag(name, new_pos, end)
                    text_end = end if close is None else close
                    self._append_text(
                        text_buffer,
                        html[new_pos:text_end],
                        raw=name not in _RCDATA_TAGS,
                        replace_null=True,
                    )
                    if close is None:
                        pos = end
                        continue
                    yield from self._flush_text(text_buffer)
                    if after_close == end and html[-1] != ">":  # pragma: no cover
                        # Scanner helpers report an unclosed end tag as no match.
                        pos = end
                        continue
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

        if html.startswith("--", pos):
            comment_start = pos + 2
            if comment_start < end and html[comment_start] == ">":
                return True, ("comment", ""), comment_start + 1, None
            if html.startswith("->", comment_start):
                return True, ("comment", ""), comment_start + 2, None
            normal_end = html.find("-->", comment_start, end)
            bang_end = html.find("--!>", comment_start, end)
            candidates = [candidate for candidate in (normal_end, bang_end) if candidate != -1]
            if candidates:
                comment_end = min(candidates)
                closing_len = 4 if comment_end == bang_end else 3
                data = html[comment_start:comment_end].replace("\0", "\ufffd")
                return True, ("comment", data), comment_end + closing_len, None
            data = html[comment_start:end]
            data = data.removesuffix("--")
            return True, ("comment", data.replace("\0", "\ufffd")), end, None

        if _scanner.ascii_startswith(html, "doctype", pos, end):
            # A ">" terminates a DOCTYPE even inside a quoted identifier.
            # Quote-aware tag scanning can otherwise swallow following text
            # when the declaration is malformed.
            tag_end = html.find(">", pos + 7, end)
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
            if "\0" in name:
                name = name.replace("\0", "\ufffd")
            if public_id is not None and "\0" in public_id:
                public_id = public_id.replace("\0", "\ufffd")
            if system_id is not None and "\0" in system_id:
                system_id = system_id.replace("\0", "\ufffd")
            return True, ("doctype", (name, public_id, system_id)), next_pos, None

        if _scanner.ascii_startswith(html, "[cdata[", pos, end) and self._in_foreign_context():
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
            return None  # pragma: no cover - the ASCII-alpha guard advances pos
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
            if "\0" in key:
                key = key.replace("\0", "\ufffd")

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
            return _scanner.find_script_end_tag(self._html, pos, end)
        return _scanner.find_rawtext_end_tag(self._html, name, pos, end)

    def _find_tag_end(self, pos: int, end: int) -> int:
        return _scanner.find_tag_end(self._html, pos, end)

    def _append_text(
        self,
        text_buffer: list[str],
        data: str,
        *,
        raw: bool = False,
        replace_null: bool = False,
    ) -> None:
        if not data:
            return
        if "\r" in data:
            data = data.replace("\r\n", "\n").replace("\r", "\n")
        if replace_null and "\0" in data:
            data = data.replace("\0", "\ufffd")
        if not raw and "&" in data:
            data = decode_entities_in_text(data, in_attribute=False)
        if data:  # pragma: no branch - decoding cannot empty non-empty input
            text_buffer.append(data)

    def _flush_text(self, text_buffer: list[str]) -> Generator[TextEvent, None, None]:
        if not text_buffer:
            return
        text = "".join(text_buffer)
        text_buffer.clear()
        if text:  # pragma: no branch - joining a non-empty buffer is non-empty
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
                    self._pop_open_element()
            else:
                return parent_namespace

        return self._namespace_from_html_context(name)

    def _pop_foreign_context(self) -> None:
        while self._open_elements and self._open_elements[-1].namespace not in {None, "html"}:
            self._pop_open_element()

    def _pop_for_end_tag(self, name: str) -> None:
        if not self._open_elements:
            return

        name_lower = name.lower()
        current = self._open_elements[-1]
        if current.namespace not in {None, "html"} and name_lower in {"br", "p"}:
            self._pop_foreign_context()
            return

        # Skip the scan entirely when the name isn't open anywhere on the
        # stack: an unmatched end tag deep inside foreign content (svg/math)
        # would otherwise scan the whole stack on every single end tag,
        # making a run of unmatched end tags quadratic overall.
        if self._name_counts.get(name_lower, 0) > 0:
            for index in range(len(self._open_elements) - 1, -1, -1):  # pragma: no branch
                node = self._open_elements[index]
                if node.name.lower() == name_lower:
                    self._truncate_open_elements(index)
                    return
                if node.namespace in {None, "html"}:
                    break
            # Unreachable: count_of > 0 guarantees a match exists at some index
            # in this exact range, so the loop above always returns or breaks
            # before exhausting it.

        self._pop_open_element()

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
