from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

if TYPE_CHECKING:
    from collections.abc import Generator

from .encoding import decode_html
from .tokenizer import Tokenizer
from .tokens import CommentToken, DoctypeToken, Tag

StartEvent: TypeAlias = tuple[Literal["start"], tuple[str, dict[str, str | None]]]
EndEvent: TypeAlias = tuple[Literal["end"], str]
TextEvent: TypeAlias = tuple[Literal["text"], str]
CommentEvent: TypeAlias = tuple[Literal["comment"], str]
DoctypeEvent: TypeAlias = tuple[Literal["doctype"], tuple[str | None, str | None, str | None]]
StreamEvent: TypeAlias = StartEvent | EndEvent | TextEvent | CommentEvent | DoctypeEvent


class _DummyNode:
    namespace: str = "html"


class StreamSink:
    """A sink that buffers tokens for the stream API."""

    tokens: list[StreamEvent]
    open_elements: list[_DummyNode]

    def __init__(self) -> None:
        self.tokens = []
        self.open_elements = []  # Required by tokenizer for rawtext checks

    def process_token(self, token: Tag | CommentToken | DoctypeToken | Any) -> int:
        # Tokenizer reuses token objects, so we must copy data
        if isinstance(token, Tag):
            # Copy tag data
            if token.kind == Tag.START:
                self.tokens.append(("start", (token.name, token.attrs.copy())))
            else:
                self.tokens.append(("end", token.name))
            # Maintain open_elements stack for tokenizer's rawtext checks
            if token.kind == Tag.START:
                # We need a dummy object with namespace for tokenizer checks
                # Tokenizer checks: stack[-1].namespace
                # We can just use a simple object
                self.open_elements.append(_DummyNode())
            else:  # Tag.END
                if self.open_elements:
                    self.open_elements.pop()
                # If open_elements is empty, we ignore the end tag for rawtext tracking purposes
                # (it's an unmatched end tag at the root level)

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
