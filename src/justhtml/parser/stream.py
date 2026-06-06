from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

if TYPE_CHECKING:
    from collections.abc import Generator

from justhtml.core.constants import (
    FOREIGN_BREAKOUT_ELEMENTS,
    HTML_INTEGRATION_POINT_SET,
    MATHML_TEXT_INTEGRATION_POINT_SET,
    SVG_TAG_NAME_ADJUSTMENTS,
)
from justhtml.tokenizer import Tokenizer
from justhtml.tokenizer.tokens import CommentToken, DoctypeToken, Tag

from .encoding import decode_html

StartEvent: TypeAlias = tuple[Literal["start"], tuple[str, dict[str, str | None]]]
EndEvent: TypeAlias = tuple[Literal["end"], str]
TextEvent: TypeAlias = tuple[Literal["text"], str]
CommentEvent: TypeAlias = tuple[Literal["comment"], str]
DoctypeEvent: TypeAlias = tuple[Literal["doctype"], tuple[str | None, str | None, str | None]]
StreamEvent: TypeAlias = StartEvent | EndEvent | TextEvent | CommentEvent | DoctypeEvent


class _DummyNode:
    __slots__ = ("attrs", "name", "namespace")

    attrs: dict[str, str | None]
    name: str
    namespace: str

    def __init__(self, name: str, namespace: str, attrs: dict[str, str | None] | None = None) -> None:
        self.attrs = attrs or {}
        self.name = name
        self.namespace = namespace


class StreamSink:
    """A sink that buffers tokens for the stream API."""

    tokens: list[StreamEvent]
    open_elements: list[_DummyNode]

    def __init__(self) -> None:
        self.tokens = []
        self.open_elements = []  # Required by tokenizer for rawtext checks

    def _font_breaks_out_of_foreign_content(self, attrs: dict[str, str | None]) -> bool:
        for name in attrs:
            if name.lower() in {"color", "face", "size"}:
                return True
        return False

    def _node_attribute_value(self, node: _DummyNode, name: str) -> str | None:
        target = name.lower()
        for attr_name, attr_value in node.attrs.items():
            if attr_name.lower() == target:
                return attr_value or ""
        return None

    def _is_html_integration_point(self, node: _DummyNode) -> bool:
        if node.namespace == "math" and node.name == "annotation-xml":
            encoding = self._node_attribute_value(node, "encoding")
            return encoding is not None and encoding.lower() in {"application/xhtml+xml", "text/html"}
        return (node.namespace, node.name) in HTML_INTEGRATION_POINT_SET

    def _is_mathml_text_integration_point(self, node: _DummyNode) -> bool:
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

    def _namespace_for_start_tag(self, token: Tag) -> str:
        name = token.name
        parent = self.open_elements[-1] if self.open_elements else None
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
                name == "font" and self._font_breaks_out_of_foreign_content(token.attrs)
            )
            if breaks_out:
                while self.open_elements and self.open_elements[-1].namespace not in {None, "html"}:
                    self.open_elements.pop()
                parent_namespace = self.open_elements[-1].namespace if self.open_elements else "html"
            else:
                return parent_namespace

        return self._namespace_from_html_context(name)

    def _pop_foreign_context(self) -> None:
        while self.open_elements and self.open_elements[-1].namespace not in {None, "html"}:
            self.open_elements.pop()

    def _pop_for_end_tag(self, name: str) -> None:
        if not self.open_elements:
            return

        name_lower = name.lower()
        current = self.open_elements[-1]
        if current.namespace not in {None, "html"} and name_lower in {"br", "p"}:
            self._pop_foreign_context()
            return

        for index in range(len(self.open_elements) - 1, -1, -1):
            node = self.open_elements[index]
            if node.name.lower() == name_lower:
                del self.open_elements[index:]
                return
            if node.namespace in {None, "html"}:
                break

        self.open_elements.pop()

    def process_token(self, token: Tag | CommentToken | DoctypeToken | Any) -> int:
        # Tokenizer reuses token objects, so we must copy data
        if isinstance(token, Tag):
            # Copy tag data
            if token.kind == Tag.START:
                self.tokens.append(("start", (token.name, token.attrs.copy())))
            else:
                self.tokens.append(("end", token.name))
            # Maintain open_elements stack for tokenizer rawtext/CDATA checks.
            if token.kind == Tag.START:
                namespace = self._namespace_for_start_tag(token)
                if not (token.self_closing and namespace not in {None, "html"}):
                    name = self._adjusted_name_for_namespace(token.name, namespace)
                    self.open_elements.append(_DummyNode(name, namespace, token.attrs.copy()))
            else:  # Tag.END
                self._pop_for_end_tag(token.name)

        elif isinstance(token, CommentToken):
            self.tokens.append(("comment", token.data))

        elif isinstance(token, DoctypeToken):
            dt = token.doctype
            self.tokens.append(("doctype", (dt.name, dt.public_id, dt.system_id)))

        return 0  # TokenSinkResult.Continue

    def process_characters(self, data: str) -> None:
        """Handle character data from tokenizer."""
        self.tokens.append(("text", data))


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
    sink = StreamSink()
    tokenizer = Tokenizer(sink)
    tokenizer.initialize(html_str)

    while True:
        # Run one step of the tokenizer
        is_eof = tokenizer.step()

        # Yield any tokens produced by this step
        if sink.tokens:
            # Coalesce text tokens
            text_buffer: list[str] = []
            for event, data in sink.tokens:
                if event == "text":
                    text_buffer.append(cast("str", data))
                else:
                    if text_buffer:
                        yield ("text", "".join(text_buffer))
                        text_buffer = []
                    yield cast("StartEvent | EndEvent | CommentEvent | DoctypeEvent", (event, data))

            if text_buffer:
                yield ("text", "".join(text_buffer))

            sink.tokens.clear()

        if is_eof:
            break
