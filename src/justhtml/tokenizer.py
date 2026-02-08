from __future__ import annotations

import re
from bisect import bisect_right
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from .entities import decode_entities_in_text
from .errors import generate_error_message
from .tokens import AnyToken, CharacterTokens, CommentToken, Doctype, DoctypeToken, EOFToken, ParseError, Tag

_ATTR_VALUE_UNQUOTED_TERMINATORS = "\t\n\f >&\"'<=`\0"
_ASCII_LOWER_TABLE = str.maketrans({chr(code): chr(code + 32) for code in range(65, 91)})
_RCDATA_ELEMENTS = {"title", "textarea"}
_RAWTEXT_SWITCH_TAGS = {
    "script",
    "style",
    "xmp",
    "iframe",
    "noembed",
    "noframes",
    "textarea",
    "title",
}

_ATTR_VALUE_DOUBLE_PATTERN = re.compile(r'["&\0]')
_ATTR_VALUE_SINGLE_PATTERN = re.compile(r"['&\0]")
_ATTR_VALUE_UNQUOTED_PATTERN = re.compile(f"[{re.escape(_ATTR_VALUE_UNQUOTED_TERMINATORS)}]")
_ATTR_VALUE_UNQUOTED_END_PATTERN = re.compile(r"[ \t\n\f>]")
_ATTR_VALUE_UNQUOTED_FAST_BAD_PATTERN = re.compile(r"""[\x00"'<=`]""")

_TAG_NAME_RUN_PATTERN = re.compile(r"[^\t\n\f />\0]+")
_ATTR_NAME_RUN_PATTERN = re.compile(r"[^\t\n\f />=\0\"'<]+")
_COMMENT_RUN_PATTERN = re.compile(r"[^-\0]+")

# XML Coercion Regex
_xml_invalid_single_chars = []
for _plane in range(17):
    _base = _plane * 0x10000
    _xml_invalid_single_chars.append(chr(_base + 0xFFFE))
    _xml_invalid_single_chars.append(chr(_base + 0xFFFF))

_XML_COERCION_PATTERN = re.compile(r"[\f\uFDD0-\uFDEF" + "".join(_xml_invalid_single_chars) + "]")


def _is_noncharacter_codepoint(codepoint: int) -> bool:
    if 0xFDD0 <= codepoint <= 0xFDEF:
        return True
    last = codepoint & 0xFFFF
    return last == 0xFFFE or last == 0xFFFF


def _xml_coercion_callback(match: re.Match[str]) -> str:
    if match.group(0) == "\f":
        return " "
    return "\ufffd"


def _coerce_text_for_xml(text: str) -> str:
    """Apply XML coercion to text content."""
    # Fast path for ASCII
    if text.isascii():
        if "\f" in text:
            return text.replace("\f", " ")
        return text

    if not _XML_COERCION_PATTERN.search(text):
        return text
    return _XML_COERCION_PATTERN.sub(_xml_coercion_callback, text)


def _coerce_comment_for_xml(text: str) -> str:
    """Apply XML coercion to comment content - handle double hyphens."""
    # Replace -- with - - (with space)
    if "--" in text:
        return text.replace("--", "- -")
    return text


class TokenizerOpts:
    __slots__ = (
        "discard_bom",
        "emit_bogus_markup_as_text",
        "exact_errors",
        "initial_rawtext_tag",
        "initial_state",
        "scripting_enabled",
        "xml_coercion",
    )

    discard_bom: bool
    exact_errors: bool
    initial_rawtext_tag: str | None
    initial_state: int | None
    scripting_enabled: bool
    xml_coercion: bool

    def __init__(
        self,
        exact_errors: bool = False,
        discard_bom: bool = True,
        emit_bogus_markup_as_text: bool = False,
        initial_state: int | None = None,
        initial_rawtext_tag: str | None = None,
        scripting_enabled: bool = True,
        xml_coercion: bool = False,
    ) -> None:
        self.exact_errors = bool(exact_errors)
        self.discard_bom = bool(discard_bom)
        self.emit_bogus_markup_as_text = bool(emit_bogus_markup_as_text)
        self.initial_state = initial_state
        self.initial_rawtext_tag = initial_rawtext_tag
        self.scripting_enabled = bool(scripting_enabled)
        self.xml_coercion = bool(xml_coercion)

    def copy(self) -> TokenizerOpts:
        """Return a shallow copy of these options.

        JustHTML may adjust some tokenizer options at runtime (based on e.g.
        fragment context and escape-mode sanitization). Copying avoids mutating
        a caller-provided TokenizerOpts instance.
        """
        return TokenizerOpts(
            exact_errors=self.exact_errors,
            discard_bom=self.discard_bom,
            emit_bogus_markup_as_text=self.emit_bogus_markup_as_text,
            initial_state=self.initial_state,
            initial_rawtext_tag=self.initial_rawtext_tag,
            scripting_enabled=self.scripting_enabled,
            xml_coercion=self.xml_coercion,
        )


class Tokenizer:
    DATA = 0
    TAG_OPEN = 1
    END_TAG_OPEN = 2
    TAG_NAME = 3
    BEFORE_ATTRIBUTE_NAME = 4
    ATTRIBUTE_NAME = 5
    AFTER_ATTRIBUTE_NAME = 6
    BEFORE_ATTRIBUTE_VALUE = 7
    ATTRIBUTE_VALUE_DOUBLE = 8
    ATTRIBUTE_VALUE_SINGLE = 9
    ATTRIBUTE_VALUE_UNQUOTED = 10
    AFTER_ATTRIBUTE_VALUE_QUOTED = 11
    SELF_CLOSING_START_TAG = 12
    MARKUP_DECLARATION_OPEN = 13
    COMMENT_START = 14
    COMMENT_START_DASH = 15
    COMMENT = 16
    COMMENT_END_DASH = 17
    COMMENT_END = 18
    COMMENT_END_BANG = 19
    BOGUS_COMMENT = 20
    DOCTYPE = 21
    BEFORE_DOCTYPE_NAME = 22
    DOCTYPE_NAME = 23
    AFTER_DOCTYPE_NAME = 24
    BOGUS_DOCTYPE = 25
    AFTER_DOCTYPE_PUBLIC_KEYWORD = 26
    AFTER_DOCTYPE_SYSTEM_KEYWORD = 27
    BEFORE_DOCTYPE_PUBLIC_IDENTIFIER = 28
    DOCTYPE_PUBLIC_IDENTIFIER_DOUBLE_QUOTED = 29
    DOCTYPE_PUBLIC_IDENTIFIER_SINGLE_QUOTED = 30
    AFTER_DOCTYPE_PUBLIC_IDENTIFIER = 31
    BETWEEN_DOCTYPE_PUBLIC_AND_SYSTEM_IDENTIFIERS = 32
    BEFORE_DOCTYPE_SYSTEM_IDENTIFIER = 33
    DOCTYPE_SYSTEM_IDENTIFIER_DOUBLE_QUOTED = 34
    DOCTYPE_SYSTEM_IDENTIFIER_SINGLE_QUOTED = 35
    AFTER_DOCTYPE_SYSTEM_IDENTIFIER = 36
    CDATA_SECTION = 37
    CDATA_SECTION_BRACKET = 38
    CDATA_SECTION_END = 39
    RCDATA = 40
    RCDATA_LESS_THAN_SIGN = 41
    RCDATA_END_TAG_OPEN = 42
    RCDATA_END_TAG_NAME = 43
    RAWTEXT = 44
    RAWTEXT_LESS_THAN_SIGN = 45
    RAWTEXT_END_TAG_OPEN = 46
    RAWTEXT_END_TAG_NAME = 47
    PLAINTEXT = 48
    SCRIPT_DATA_ESCAPED = 49
    SCRIPT_DATA_ESCAPED_DASH = 50
    SCRIPT_DATA_ESCAPED_DASH_DASH = 51
    SCRIPT_DATA_ESCAPED_LESS_THAN_SIGN = 52
    SCRIPT_DATA_ESCAPED_END_TAG_OPEN = 53
    SCRIPT_DATA_ESCAPED_END_TAG_NAME = 54
    SCRIPT_DATA_DOUBLE_ESCAPE_START = 55
    SCRIPT_DATA_DOUBLE_ESCAPED = 56
    SCRIPT_DATA_DOUBLE_ESCAPED_DASH = 57
    SCRIPT_DATA_DOUBLE_ESCAPED_DASH_DASH = 58
    SCRIPT_DATA_DOUBLE_ESCAPED_LESS_THAN_SIGN = 59
    SCRIPT_DATA_DOUBLE_ESCAPE_END = 60

    __slots__ = (
        "_comment_token",
        "_newline_positions",
        "_state_handlers",
        "_tag_token",
        "buffer",
        "collect_errors",
        "current_attr_name",
        "current_attr_value",
        "current_attr_value_has_amp",
        "current_char",
        "current_comment",
        "current_doctype_force_quirks",
        "current_doctype_name",
        "current_doctype_public",
        "current_doctype_system",
        "current_tag_attrs",
        "current_tag_kind",
        "current_tag_name",
        "current_tag_self_closing",
        "current_token_start_pos",
        "errors",
        "last_start_tag_name",
        "last_token_column",
        "last_token_line",
        "last_token_start_pos",
        "length",
        "opts",
        "original_tag_name",
        "pos",
        "rawtext_tag_name",
        "reconsume",
        "sink",
        "state",
        "temp_buffer",
        "text_buffer",
        "text_start_pos",
        "track_node_locations",
        "track_tag_positions",
    )

    _comment_token: CommentToken
    _newline_positions: list[int] | None
    _state_handlers: list[Callable[[Tokenizer], bool]]
    _tag_token: Tag
    buffer: str
    collect_errors: bool
    track_tag_positions: bool
    track_node_locations: bool
    current_attr_name: list[str]
    current_attr_value: list[str]
    current_attr_value_has_amp: bool
    current_char: str | None
    current_comment: list[str]
    current_doctype_force_quirks: bool
    current_doctype_name: list[str]
    current_doctype_public: list[str] | None
    current_doctype_system: list[str] | None
    current_tag_attrs: dict[str, str | None]
    current_tag_kind: int
    current_tag_name: list[str]
    current_tag_self_closing: bool
    current_token_start_pos: int
    errors: list[ParseError]
    last_start_tag_name: str | None
    last_token_column: int
    last_token_line: int
    last_token_start_pos: int | None
    length: int
    opts: TokenizerOpts
    original_tag_name: list[str]
    pos: int
    rawtext_tag_name: str | None
    reconsume: bool
    sink: Any
    state: int
    temp_buffer: list[str]
    text_buffer: list[str]
    text_start_pos: int

    # _STATE_HANDLERS is defined at the end of the file

    def __init__(
        self,
        sink: Any,
        opts: TokenizerOpts | None = None,
        *,
        collect_errors: bool = False,
        track_node_locations: bool = False,
        track_tag_positions: bool = False,
    ) -> None:
        self.sink = sink
        self.opts = opts or TokenizerOpts()
        self.collect_errors = collect_errors
        self.track_node_locations = bool(track_node_locations)
        self.track_tag_positions = bool(track_tag_positions)
        self.errors = []

        self.state = self.DATA
        self.buffer = ""
        self.length = 0
        self.pos = 0
        self.reconsume = False
        self.current_char = ""
        self.last_token_line = 1
        self.last_token_column = 0
        self.current_token_start_pos = 0
        self.last_token_start_pos = None

        # Reusable buffers to avoid per-token allocations.
        self.text_buffer = []
        self.text_start_pos = 0
        self.current_tag_name = []
        self.current_tag_attrs = {}
        self.current_attr_name = []
        self.current_attr_value = []
        self.current_attr_value_has_amp = False
        self.current_tag_self_closing = False
        self.current_tag_kind = Tag.START
        self.current_comment = []
        self.current_doctype_name = []
        self.current_doctype_public = None  # None = not set, [] = empty string
        self.current_doctype_system = None  # None = not set, [] = empty string
        self.current_doctype_force_quirks = False
        self.last_start_tag_name = None
        self.rawtext_tag_name = None
        self.original_tag_name = []
        self.temp_buffer = []
        self._tag_token = Tag(Tag.START, "", {}, False)
        self._comment_token = CommentToken("")

    def initialize(self, html: str | None) -> None:
        if html and html[0] == "\ufeff" and self.opts.discard_bom:
            html = html[1:]

        # Normalize newlines per §13.2.2.5
        if html:
            if "\r" in html:
                html = html.replace("\r\n", "\n").replace("\r", "\n")

        self.buffer = html or ""
        self.length = len(self.buffer)
        self.pos = 0
        self.reconsume = False
        self.current_char = ""
        self.last_token_line = 1
        self.last_token_column = 0
        self.current_token_start_pos = 0
        self.last_token_start_pos = None
        self.errors = []
        self.text_buffer.clear()
        self.text_start_pos = 0
        self.current_tag_name.clear()
        self.current_tag_attrs = {}
        self.current_attr_name.clear()
        self.current_attr_value.clear()
        self.current_attr_value_has_amp = False
        self.current_comment.clear()
        self.current_doctype_name.clear()
        self.current_doctype_public = None
        self.current_doctype_system = None
        self.current_doctype_force_quirks = False
        self.current_tag_self_closing = False
        self.current_tag_kind = Tag.START
        self.rawtext_tag_name = self.opts.initial_rawtext_tag
        self.temp_buffer.clear()
        self.last_start_tag_name = None
        self._tag_token.kind = Tag.START
        self._tag_token.name = ""
        self._tag_token.attrs = {}
        self._tag_token.self_closing = False
        self._tag_token.start_pos = None
        self._tag_token.end_pos = None

        initial_state = self.opts.initial_state
        if isinstance(initial_state, int):
            self.state = initial_state
        else:
            self.state = self.DATA

        # Pre-compute newline positions for O(log n) line lookups.
        # Only do this when errors are collected or when node locations are requested.
        if self.collect_errors or self.track_node_locations:
            self._newline_positions = []
            pos = -1
            buffer = self.buffer
            while True:
                pos = buffer.find("\n", pos + 1)
                if pos == -1:
                    break
                self._newline_positions.append(pos)
        else:
            self._newline_positions = None

    def _get_line_at_pos(self, pos: int) -> int:
        """Get line number (1-indexed) for a position using binary search."""
        # Line number = count of newlines before pos + 1
        newline_positions = self._newline_positions
        if newline_positions is None:  # pragma: no cover
            return 1
        return bisect_right(newline_positions, pos - 1) + 1

    def location_at_pos(self, pos: int) -> tuple[int, int]:
        """Return (line, column) for a 0-indexed offset in the current buffer.

        Column is 1-indexed. Newline positions are computed lazily when needed.
        """
        newline_positions = self._newline_positions
        if newline_positions is None:
            newline_positions = []
            scan = -1
            buffer = self.buffer
            while True:
                scan = buffer.find("\n", scan + 1)
                if scan == -1:
                    break
                newline_positions.append(scan)
            self._newline_positions = newline_positions

        line_index = bisect_right(newline_positions, pos - 1)
        line = line_index + 1

        # Compute column using newline index rather than rfind() to avoid O(n) scans.
        if line_index == 0:
            last_newline = -1
        else:
            last_newline = newline_positions[line_index - 1]
        column = pos - last_newline
        return line, column

    def step(self) -> bool:
        """Run one step of the tokenizer state machine. Returns True if EOF reached."""
        handler = self._STATE_HANDLERS[self.state]  # type: ignore[attr-defined]
        return handler(self)  # type: ignore[no-any-return]

    def run(self, html: str | None) -> None:
        self.initialize(html)
        handlers = self._STATE_HANDLERS  # type: ignore[attr-defined]
        while True:
            if handlers[self.state](self):  # type: ignore[no-any-return]
                break

    # ---------------------
    # Helper methods
    # ---------------------

    def _peek_char(self, offset: int) -> str | None:
        """Peek ahead at character at current position + offset without consuming"""
        peek_pos = self.pos + offset
        if peek_pos < self.length:
            return self.buffer[peek_pos]
        return None

    # ---------------------
    # State handlers
    # ---------------------

    def _state_data(self) -> bool:
        buffer = self.buffer
        length = self.length
        pos = self.pos
        while True:
            if self.reconsume:
                # Note: reconsume is never True at EOF in DATA state
                self.reconsume = False
                self.pos -= 1
                pos = self.pos

            if pos >= length:
                self.pos = length
                self.current_char = None
                self._flush_text()
                self._emit_token(EOFToken())
                return True

            # Optimized loop using find
            next_lt = buffer.find("<", pos)

            if next_lt == -1:
                next_lt = length

            end = next_lt

            if end > pos:
                chunk = buffer[pos:end]
                if self.collect_errors and not chunk.isascii():
                    base_pos = pos
                    for offset, ch in enumerate(chunk):
                        if _is_noncharacter_codepoint(ord(ch)):
                            self._emit_error_at_pos("noncharacter-in-input-stream", base_pos + offset)
                self._append_text(chunk)

                pos = end
                self.pos = pos
                if pos >= length:
                    continue

            # After find("<"), we're always at '<' unless reconsume is True
            # But reconsume only happens after TAG_OPEN which reconsumed '<'
            c = buffer[pos]
            pos += 1
            self.pos = pos
            self.current_char = c
            # c is always '<' here due to find() optimization above
            self.current_token_start_pos = pos - 1
            # Optimization: Peek ahead for common tag starts
            if pos < length:
                nc = buffer[pos]
                if ("a" <= nc <= "z") or ("A" <= nc <= "Z"):
                    self._flush_text()
                    # Inline _start_tag(Tag.START)
                    self.current_tag_kind = Tag.START
                    self.current_tag_name.clear()
                    self.current_attr_name.clear()
                    self.current_attr_value.clear()
                    self.current_attr_value_has_amp = False
                    self.current_tag_self_closing = False

                    # Fast-path: consume the full tag name run here to avoid
                    # the TAG_NAME handler overhead in the common case.
                    name_match = _TAG_NAME_RUN_PATTERN.match(buffer, pos)
                    if name_match:
                        name = name_match.group(0)
                        if not name.islower():
                            name = name.translate(_ASCII_LOWER_TABLE)
                        self.current_tag_name.append(name)
                        pos = name_match.end()
                        self.pos = pos

                        if pos < length:
                            next_char = buffer[pos]
                            if next_char in (" ", "\t", "\n", "\f"):
                                self.pos = pos + 1
                                self.state = self.BEFORE_ATTRIBUTE_NAME
                                return self._state_before_attribute_name()
                            if next_char == ">":
                                self.pos = pos + 1
                                if not self._emit_current_tag():
                                    self.state = self.DATA
                                    pos = self.pos
                                    continue
                                return False
                            if next_char == "/":
                                self.pos = pos + 1
                                self.state = self.SELF_CLOSING_START_TAG
                                return self._state_self_closing_start_tag()

                        # Fall back to TAG_NAME for rare cases (e.g. EOF or NUL).
                        self.state = self.TAG_NAME
                        return self._state_tag_name()

                    # Defensive fallback: should not happen, but keep the old path.
                    if not name_match:  # pragma: no cover
                        if "A" <= nc <= "Z":
                            nc = chr(ord(nc) + 32)
                        self.current_tag_name.append(nc)
                        self.pos += 1
                        self.state = self.TAG_NAME
                        return self._state_tag_name()

                if nc == "!" and not self.opts.emit_bogus_markup_as_text:
                    # Optimization: Peek ahead for comments
                    if pos + 2 < length and buffer[pos + 1] == "-" and buffer[pos + 2] == "-":
                        self._flush_text()
                        self.pos += 3  # Consume !--
                        self.current_comment.clear()
                        self.state = self.COMMENT_START
                        return self._state_comment_start()

                if nc == "/":
                    # Check next char for end tag
                    if pos + 1 < length:
                        nnc = buffer[pos + 1]
                        if ("a" <= nnc <= "z") or ("A" <= nnc <= "Z"):
                            self._flush_text()
                            # Inline _start_tag(Tag.END)
                            self.current_tag_kind = Tag.END
                            self.current_tag_name.clear()
                            self.current_attr_name.clear()
                            self.current_attr_value.clear()
                            self.current_attr_value_has_amp = False
                            self.current_tag_self_closing = False

                            # Fast-path: consume full end tag name run here.
                            name_match = _TAG_NAME_RUN_PATTERN.match(buffer, pos + 1)
                            if name_match:
                                name = name_match.group(0)
                                if not name.islower():
                                    name = name.translate(_ASCII_LOWER_TABLE)
                                self.current_tag_name.append(name)
                                pos = name_match.end()
                                self.pos = pos

                                if pos < length:
                                    next_char = buffer[pos]
                                    if next_char in (" ", "\t", "\n", "\f"):
                                        if self.opts.emit_bogus_markup_as_text:
                                            return self._emit_raw_end_tag_as_text(pos)
                                        self.pos = pos + 1
                                        self.state = self.BEFORE_ATTRIBUTE_NAME
                                        return self._state_before_attribute_name()
                                    if next_char == ">":
                                        self.pos = pos + 1
                                        self._emit_current_tag()
                                        self.state = self.DATA
                                        pos = self.pos
                                        continue
                                    if next_char == "/":
                                        if self.opts.emit_bogus_markup_as_text:
                                            return self._emit_raw_end_tag_as_text(pos)
                                        self.pos = pos + 1
                                        self.state = self.SELF_CLOSING_START_TAG
                                        return self._state_self_closing_start_tag()

                                self.state = self.TAG_NAME
                                return self._state_tag_name()

                            # Defensive fallback: old path.
                            if not name_match:  # pragma: no cover
                                if "A" <= nnc <= "Z":
                                    nnc = chr(ord(nnc) + 32)
                                self.current_tag_name.append(nnc)
                                self.pos += 2  # Consume / and nnc
                                self.state = self.TAG_NAME
                                return self._state_tag_name()

            self._flush_text()
            self.state = self.TAG_OPEN
            return self._state_tag_open()

    def _state_tag_open(self) -> bool:
        c = self._get_char()
        if c is None:
            self._emit_error("eof-before-tag-name")
            self._append_text("<")
            self._flush_text()
            self._emit_token(EOFToken())
            return True
        if c == "!":
            if self.opts.emit_bogus_markup_as_text:
                self._append_text("<!")
                self.state = self.DATA
                return False
            self.state = self.MARKUP_DECLARATION_OPEN
            return False
        if c == "/":
            self.state = self.END_TAG_OPEN
            return False
        if c == "?":
            if self.opts.emit_bogus_markup_as_text:
                self._append_text("<?")
                self.state = self.DATA
                return False
            self._emit_error("unexpected-question-mark-instead-of-tag-name")
            self.current_comment.clear()
            self._reconsume_current()
            self.state = self.BOGUS_COMMENT
            return False

        self._emit_error("invalid-first-character-of-tag-name")
        self._append_text("<")
        self._reconsume_current()
        self.state = self.DATA
        return False

    def _state_end_tag_open(self) -> bool:
        c = self._get_char()
        if c is None:
            self._emit_error("eof-before-tag-name")
            if self.opts.emit_bogus_markup_as_text:
                self._append_text("</")
                self._flush_text()
                self._emit_token(EOFToken())
                return True
            self._append_text("<")
            self._append_text("/")
            self._flush_text()
            self._emit_token(EOFToken())
            return True
        if c == ">":
            self._emit_error("empty-end-tag")
            if self.opts.emit_bogus_markup_as_text:
                self._append_text("</>")
                self.state = self.DATA
                return False
            self.state = self.DATA
            return False

        if self.opts.emit_bogus_markup_as_text:
            self._append_text("</")
            self._append_text(c)
            self.state = self.DATA
            return False

        self._emit_error("invalid-first-character-of-tag-name")
        self.current_comment.clear()
        self._reconsume_current()
        self.state = self.BOGUS_COMMENT
        return False

    def _state_tag_name(self) -> bool:
        replacement = "\ufffd"
        append_tag_char = self.current_tag_name.append
        buffer = self.buffer
        length = self.length
        pos = self.pos

        while True:
            # Inline _consume_tag_name_run
            # Note: reconsume is never True when entering TAG_NAME
            if pos < length:
                # Optimization: Check for common terminators before regex
                match = None
                if buffer[pos] not in "\t\n\f />\0":
                    match = _TAG_NAME_RUN_PATTERN.match(buffer, pos)

                if match:
                    chunk = match.group(0)
                    if not chunk.islower():
                        chunk = chunk.translate(_ASCII_LOWER_TABLE)
                    append_tag_char(chunk)
                    pos = match.end()

                    if pos < length:
                        next_char = buffer[pos]
                        if next_char in (" ", "\t", "\n", "\f"):
                            if self.current_tag_kind == Tag.END and self.opts.emit_bogus_markup_as_text:
                                return self._emit_raw_end_tag_as_text(pos)
                            pos += 1
                            self.pos = pos
                            self.state = self.BEFORE_ATTRIBUTE_NAME
                            return self._state_before_attribute_name()
                        if next_char == ">":
                            pos += 1
                            self.pos = pos
                            if not self._emit_current_tag():
                                self.state = self.DATA
                            return False
                        if next_char == "/":
                            if self.current_tag_kind == Tag.END and self.opts.emit_bogus_markup_as_text:
                                return self._emit_raw_end_tag_as_text(pos)
                            pos += 1
                            self.pos = pos
                            self.state = self.SELF_CLOSING_START_TAG
                            return self._state_self_closing_start_tag()

            # Inline _get_char
            # Note: reconsume is never True in this state.
            if pos >= length:
                c: str | None = None
            else:
                c = buffer[pos]
                pos += 1
            self.current_char = c
            if c is None:
                self.pos = pos
                self._emit_error("eof-in-tag")
                self._emit_incomplete_tag_as_text()
                self._emit_token(EOFToken())
                return True
            if c in ("\t", "\n", "\f", " "):
                if self.current_tag_kind == Tag.END and self.opts.emit_bogus_markup_as_text:
                    self.pos = pos
                    return self._emit_raw_end_tag_as_text(pos)
                self.pos = pos
                self.state = self.BEFORE_ATTRIBUTE_NAME
                return self._state_before_attribute_name()
            if c == "/":
                if self.current_tag_kind == Tag.END and self.opts.emit_bogus_markup_as_text:
                    self.pos = pos
                    return self._emit_raw_end_tag_as_text(pos)
                self.pos = pos
                self.state = self.SELF_CLOSING_START_TAG
                return self._state_self_closing_start_tag()
            if c == ">":
                # In slow path, tag name is only first char (from DATA),
                # so no rawtext elements possible - always set DATA state
                self.pos = pos
                self._emit_current_tag()
                self.state = self.DATA
                return False
            # c == "\0" - the only remaining possibility after fast-path
            self.pos = pos
            self._emit_error("unexpected-null-character")
            append_tag_char(replacement)

    def _state_before_attribute_name(self) -> bool:
        buffer = self.buffer
        length = self.length

        pos = self.pos
        while True:
            if self.reconsume:
                # Reconsume happens e.g. after attribute value (quoted) when
                # there's missing whitespace between attributes.
                self.reconsume = False
                c = self.current_char

                if c in (" ", "\n", "\t", "\f"):
                    continue

                if c is None:
                    self.pos = length
                    self._emit_error("eof-in-tag")
                    self._emit_incomplete_tag_as_text()
                    self._flush_text()
                    self._emit_token(EOFToken())
                    return True

                if c == "/":
                    self.state = self.SELF_CLOSING_START_TAG
                    return False
                if c == ">":
                    self._finish_attribute()
                    if not self._emit_current_tag():
                        self.state = self.DATA
                    return False
                if c == "=":
                    self._emit_error("unexpected-equals-sign-before-attribute-name")
                    self.current_attr_name.clear()
                    self.current_attr_value.clear()
                    self.current_attr_value_has_amp = False
                    self.current_attr_name.append("=")
                    self.state = self.ATTRIBUTE_NAME
                    return False  # Let main loop dispatch to avoid recursion

                self.current_attr_name.clear()
                self.current_attr_value.clear()
                self.current_attr_value_has_amp = False
                if c == "\0":
                    self._emit_error("unexpected-null-character")
                    c = "\ufffd"
                elif "A" <= c <= "Z":
                    c = chr(ord(c) + 32)
                self.current_attr_name.append(c)
                self.state = self.ATTRIBUTE_NAME
                return False  # Let main loop dispatch to avoid recursion

            # Optimization: Skip whitespace (common).
            if pos < length and buffer[pos] in " \t\n\f":
                # ASCII whitespace runs between attributes are typically short
                # (often a single space), so a tight Python loop outperforms
                # the regex matcher here.
                while pos < length and buffer[pos] in " \t\n\f":
                    pos += 1
                continue

            if pos >= length:
                self.pos = length
                self._emit_error("eof-in-tag")
                self._emit_incomplete_tag_as_text()
                self._flush_text()
                self._emit_token(EOFToken())
                return True

            c = buffer[pos]
            self.current_char = c

            if c == "/":
                self.pos = pos + 1
                self.state = self.SELF_CLOSING_START_TAG
                return False
            if c == ">":
                self.pos = pos + 1
                self._finish_attribute()
                if not self._emit_current_tag():
                    self.state = self.DATA
                return False
            if c == "=":
                self.pos = pos + 1
                self._emit_error("unexpected-equals-sign-before-attribute-name")
                self.current_attr_name.clear()
                self.current_attr_value.clear()
                self.current_attr_value_has_amp = False
                self.current_attr_name.append("=")
                self.state = self.ATTRIBUTE_NAME
                return False  # Let main loop dispatch to avoid recursion

            # Do not consume the first attribute character here. Let
            # ATTRIBUTE_NAME consume the full run so we avoid building a 2-part
            # name buffer in the common case.
            self.current_attr_name.clear()
            self.current_attr_value.clear()
            self.current_attr_value_has_amp = False

            # Fast-path: consume the full attribute name run here to avoid the
            # ATTRIBUTE_NAME handler overhead in the common case.
            name_match = _ATTR_NAME_RUN_PATTERN.match(buffer, pos)
            if name_match:
                name = name_match.group(0)
                if not name.islower():
                    name = name.translate(_ASCII_LOWER_TABLE)
                attrs = self.current_tag_attrs
                is_duplicate = name in attrs
                pos = name_match.end()
                self.pos = pos

                if pos < length:
                    next_char = buffer[pos]
                    if next_char == "=":
                        # Fast-path: common quoted attribute values like
                        # `class="..."`. Avoid bouncing through
                        # BEFORE_ATTRIBUTE_VALUE -> ATTRIBUTE_VALUE_* ->
                        # AFTER_ATTRIBUTE_VALUE_QUOTED when we don't need
                        # error-precise behavior.
                        if not self.collect_errors:
                            value_pos = pos + 1
                            if value_pos < length and buffer[value_pos] in " \t\n\f":
                                while value_pos < length and buffer[value_pos] in " \t\n\f":
                                    value_pos += 1

                            if value_pos < length:
                                quote = buffer[value_pos]
                                if quote in ('"', "'"):
                                    end_quote = buffer.find(quote, value_pos + 1)
                                    if end_quote != -1:
                                        raw_value = buffer[value_pos + 1 : end_quote]
                                        if "\0" in raw_value:
                                            raw_value = raw_value.replace("\0", "\ufffd")
                                        if not is_duplicate:
                                            value = raw_value
                                            if "&" in value:
                                                value = decode_entities_in_text(
                                                    value, in_attribute=True, report_error=None
                                                )
                                            attrs[name] = value
                                        self.pos = end_quote + 1

                                        # Inline _state_after_attribute_value_quoted()
                                        if self.pos >= length:
                                            self.current_char = None
                                            self._emit_error("eof-in-tag")
                                            self._emit_incomplete_tag_as_text()
                                            self._flush_text()
                                            self._emit_token(EOFToken())
                                            return True

                                        c2 = buffer[self.pos]
                                        self.pos += 1
                                        self.current_char = c2

                                        if c2 in ("\t", "\n", "\f", " "):
                                            self.state = self.BEFORE_ATTRIBUTE_NAME
                                            pos = self.pos
                                            continue
                                        if c2 == "/":
                                            self.state = self.SELF_CLOSING_START_TAG
                                            return self._state_self_closing_start_tag()
                                        if c2 == ">":
                                            if not self._emit_current_tag():
                                                self.state = self.DATA
                                            return False

                                        self._emit_error("missing-whitespace-between-attributes")
                                        self._reconsume_current()
                                        self.state = self.BEFORE_ATTRIBUTE_NAME
                                        return False
                                else:
                                    # Fast-path: unquoted attribute values like `id=foo`.
                                    # Only handle values that don't contain characters that
                                    # require error handling in the spec ("'<=>` and NUL).
                                    end_match = _ATTR_VALUE_UNQUOTED_END_PATTERN.search(buffer, value_pos)
                                    end_pos = end_match.start() if end_match else length
                                    raw_value = buffer[value_pos:end_pos]
                                    if not _ATTR_VALUE_UNQUOTED_FAST_BAD_PATTERN.search(raw_value):
                                        if not is_duplicate:
                                            value = raw_value
                                            if "&" in value:
                                                value = decode_entities_in_text(
                                                    value, in_attribute=True, report_error=None
                                                )
                                            attrs[name] = value

                                        self.pos = end_pos
                                        pos = end_pos
                                        if end_pos >= length:
                                            continue

                                        end_char = buffer[end_pos]
                                        if end_char in ("\t", "\n", "\f", " "):
                                            self.pos = end_pos + 1
                                            self.state = self.BEFORE_ATTRIBUTE_NAME
                                            pos = self.pos
                                            continue

                                        # end_char == ">"
                                        self.pos = end_pos + 1
                                        if not self._emit_current_tag():
                                            self.state = self.DATA
                                        return False

                        # Fallback: let the dedicated state handlers deal with
                        # unquoted values and edge cases.
                        self.current_attr_name.append(name)
                        self.pos = pos + 1
                        self.state = self.BEFORE_ATTRIBUTE_VALUE
                        return self._state_before_attribute_value()
                    if next_char in (" ", "\t", "\n", "\f"):
                        self.pos = pos + 1
                        if not is_duplicate:
                            attrs[name] = ""
                        self.state = self.AFTER_ATTRIBUTE_NAME
                        return False
                    if next_char == ">":
                        self.pos = pos + 1
                        if not is_duplicate:
                            attrs[name] = ""
                        if not self._emit_current_tag():
                            self.state = self.DATA
                        return False
                    if next_char == "/":
                        self.pos = pos + 1
                        if not is_duplicate:
                            attrs[name] = ""
                        self.state = self.SELF_CLOSING_START_TAG
                        return self._state_self_closing_start_tag()

                # Rare cases (EOF, NUL, etc.): fall back to ATTRIBUTE_NAME to
                # finish parsing the attribute name and proceed.
                self.current_attr_name.append(name)
                self.state = self.ATTRIBUTE_NAME
                return self._state_attribute_name()

            # Fallback: let ATTRIBUTE_NAME handle invalid starts.
            self.pos = pos
            self.state = self.ATTRIBUTE_NAME
            return self._state_attribute_name()

    def _state_attribute_name(self) -> bool:
        replacement = "\ufffd"
        append_attr_char = self.current_attr_name.append
        buffer = self.buffer
        length = self.length
        pos = self.pos

        while True:
            # Inline _consume_attribute_name_run
            # Note: reconsume is never True in this state.
            if pos < length:
                # Optimization: Check for common terminators before regex
                match = None
                if buffer[pos] not in "\t\n\f />=\0\"'<":
                    match = _ATTR_NAME_RUN_PATTERN.match(buffer, pos)

                if match:
                    chunk = match.group(0)
                    if not chunk.islower():
                        chunk = chunk.translate(_ASCII_LOWER_TABLE)
                    append_attr_char(chunk)
                    pos = match.end()

                    if pos < length:
                        next_char = buffer[pos]
                        if next_char == "=":
                            pos += 1
                            self.pos = pos
                            self.state = self.BEFORE_ATTRIBUTE_VALUE
                            return self._state_before_attribute_value()
                        if next_char in (" ", "\t", "\n", "\f"):
                            pos += 1
                            self.pos = pos
                            self._finish_attribute()
                            self.state = self.AFTER_ATTRIBUTE_NAME
                            return False  # Let main loop dispatch to avoid recursion
                        if next_char == ">":
                            pos += 1
                            self.pos = pos
                            self._finish_attribute()
                            if not self._emit_current_tag():
                                self.state = self.DATA
                            return False
                        if next_char == "/":
                            pos += 1
                            self.pos = pos
                            self._finish_attribute()
                            self.state = self.SELF_CLOSING_START_TAG
                            return self._state_self_closing_start_tag()

            # Inline _get_char (reconsume is never True in this state)
            if pos >= length:
                c: str | None = None
            else:
                c = buffer[pos]
                pos += 1
            self.current_char = c
            self.pos = pos
            if c is None:
                self._emit_error("eof-in-tag")
                self._emit_incomplete_tag_as_text()
                self._flush_text()
                self._emit_token(EOFToken())
                return True
            if c in ("\t", "\n", "\f", " "):
                self._finish_attribute()
                self.state = self.AFTER_ATTRIBUTE_NAME
                return False  # Let main loop dispatch to avoid recursion
            if c == "/":
                self._finish_attribute()
                self.state = self.SELF_CLOSING_START_TAG
                return self._state_self_closing_start_tag()
            if c == "=":
                self.state = self.BEFORE_ATTRIBUTE_VALUE
                return self._state_before_attribute_value()
            if c == ">":
                self._finish_attribute()
                if not self._emit_current_tag():
                    self.state = self.DATA
                return False
            if c == "\0":
                self._emit_error("unexpected-null-character")
                append_attr_char(replacement)
                continue
            self._emit_error("unexpected-character-in-attribute-name")
            append_attr_char(c)

    def _state_after_attribute_name(self) -> bool:
        buffer = self.buffer
        length = self.length

        while True:
            # Optimization: Skip whitespace
            if not self.reconsume:
                if self.pos < length:
                    if buffer[self.pos] in " \t\n\f":
                        p = self.pos
                        while p < length and buffer[p] in " \t\n\f":
                            p += 1
                        self.pos = p

            # Inline _get_char
            if self.pos >= length:
                c = None
            else:
                c = buffer[self.pos]
                self.pos += 1

            self.current_char = c

            if c in (" ", "\n", "\t", "\f"):
                continue

            if c is None:
                self._emit_error("eof-in-tag")
                self._emit_incomplete_tag_as_text()
                self._flush_text()
                self._emit_token(EOFToken())
                return True
            if c == "/":
                self._finish_attribute()
                self.state = self.SELF_CLOSING_START_TAG
                return False
            if c == "=":
                self.state = self.BEFORE_ATTRIBUTE_VALUE
                return False
            if c == ">":
                self._finish_attribute()
                if not self._emit_current_tag():
                    self.state = self.DATA
                return False
            self._finish_attribute()
            self.current_attr_name.clear()
            self.current_attr_value.clear()
            self.current_attr_value_has_amp = False
            if c == "\0":
                self._emit_error("unexpected-null-character")
                c = "\ufffd"
            elif "A" <= c <= "Z":
                c = chr(ord(c) + 32)
            self.current_attr_name.append(c)
            self.state = self.ATTRIBUTE_NAME
            return False  # Let main loop dispatch to avoid recursion

    def _state_before_attribute_value(self) -> bool:
        while True:
            # Inline _get_char (reconsume is never True in this state)
            pos = self.pos
            if pos >= self.length:
                c: str | None = None
            else:
                c = self.buffer[pos]
                self.pos = pos + 1
            self.current_char = c
            if c is None:
                self._emit_error("eof-in-tag")
                self._emit_incomplete_tag_as_text()
                self._flush_text()
                self._emit_token(EOFToken())
                return True
            if c in ("\t", "\n", "\f", " "):
                continue
            if c == '"':
                self.state = self.ATTRIBUTE_VALUE_DOUBLE
                return self._state_attribute_value_double()
            if c == "'":
                self.state = self.ATTRIBUTE_VALUE_SINGLE
                return self._state_attribute_value_single()
            if c == ">":
                self._emit_error("missing-attribute-value")
                self._finish_attribute()
                if not self._emit_current_tag():
                    self.state = self.DATA
                return False
            self._reconsume_current()
            self.state = self.ATTRIBUTE_VALUE_UNQUOTED
            return self._state_attribute_value_unquoted()

    def _state_attribute_value_double(self) -> bool:
        replacement = "\ufffd"
        stop_pattern = _ATTR_VALUE_DOUBLE_PATTERN
        buffer = self.buffer
        length = self.length

        while True:
            # Inline _consume_attribute_value_run
            pos = self.pos
            if pos < length:
                # Optimization: Optimistically look for quote
                next_quote = buffer.find('"', pos)
                if next_quote == -1:
                    next_quote = length

                # Check if we skipped other terminators
                chunk = buffer[pos:next_quote]
                if "&" in chunk or "\0" in chunk:
                    # Fallback to regex if complex chars present
                    match = stop_pattern.search(buffer, pos)
                    end = length if match is None else match.start()
                else:
                    end = next_quote

                if end > pos:
                    # chunk is already valid if we took the fast path
                    if end != next_quote:
                        chunk = buffer[pos:end]

                    self.current_attr_value.append(chunk)
                    self.pos = end

            # Inlined _get_char logic
            if self.pos >= length:
                self.current_char = None
                self._emit_error("eof-in-tag")
                self._emit_incomplete_tag_as_text()
                self._emit_token(EOFToken())
                return True

            c = buffer[self.pos]
            self.pos += 1

            self.current_char = c

            if c == '"':
                self.state = self.AFTER_ATTRIBUTE_VALUE_QUOTED
                return self._state_after_attribute_value_quoted()
            if c == "&":
                self.current_attr_value.append("&")
                self.current_attr_value_has_amp = True
            else:
                # c == "\0" - the only remaining possibility after fast-path
                self._emit_error("unexpected-null-character")
                self.current_attr_value.append(replacement)

    def _state_attribute_value_single(self) -> bool:
        replacement = "\ufffd"
        stop_pattern = _ATTR_VALUE_SINGLE_PATTERN
        buffer = self.buffer
        length = self.length

        while True:
            # Inline _consume_attribute_value_run
            pos = self.pos
            if pos < length:
                # Optimization: Optimistically look for quote
                next_quote = buffer.find("'", pos)
                if next_quote == -1:
                    next_quote = length

                # Check if we skipped other terminators
                chunk = buffer[pos:next_quote]
                if "&" in chunk or "\0" in chunk:
                    # Fallback to regex if complex chars present
                    match = stop_pattern.search(buffer, pos)
                    end = length if match is None else match.start()
                else:
                    end = next_quote

                if end > pos:
                    # chunk is already valid if we took the fast path
                    if end != next_quote:
                        chunk = buffer[pos:end]

                    self.current_attr_value.append(chunk)
                    self.pos = end

            # Inlined _get_char logic
            if self.pos >= length:
                self.current_char = None
                self._emit_error("eof-in-tag")
                self._emit_incomplete_tag_as_text()
                self._emit_token(EOFToken())
                return True

            c = buffer[self.pos]
            self.pos += 1

            self.current_char = c

            if c == "'":
                self.state = self.AFTER_ATTRIBUTE_VALUE_QUOTED
                return self._state_after_attribute_value_quoted()
            if c == "&":
                self.current_attr_value.append("&")
                self.current_attr_value_has_amp = True
            else:
                # c == "\0" - the only remaining possibility after fast-path
                self._emit_error("unexpected-null-character")
                self.current_attr_value.append(replacement)

    def _state_attribute_value_unquoted(self) -> bool:
        replacement = "\ufffd"
        stop_pattern = _ATTR_VALUE_UNQUOTED_PATTERN
        buffer = self.buffer
        length = self.length

        while True:
            # Inline _consume_attribute_value_run
            if not self.reconsume:
                pos = self.pos
                if pos < length:
                    match = stop_pattern.search(buffer, pos)
                    # Note: match is always found - pattern matches terminators or EOF
                    end = match.start() if match else length

                    if end > pos:
                        self.current_attr_value.append(buffer[pos:end])
                        self.pos = end

            # Inline _get_char
            if self.reconsume:
                self.reconsume = False
                c = self.current_char
            elif self.pos >= length:
                c = None
            else:
                c = buffer[self.pos]
                self.pos += 1
            self.current_char = c

            if c is None:
                # Per HTML5 spec: EOF in attribute value is a parse error
                # The incomplete tag is discarded (not emitted)
                self._emit_error("eof-in-tag")
                self._emit_incomplete_tag_as_text()
                self._emit_token(EOFToken())
                return True
            if c in ("\t", "\n", "\f", " "):
                self._finish_attribute()
                self.state = self.BEFORE_ATTRIBUTE_NAME
                return False
            if c == ">":
                self._finish_attribute()
                if not self._emit_current_tag():
                    self.state = self.DATA
                return False
            if c == "&":
                self.current_attr_value.append("&")
                self.current_attr_value_has_amp = True
                continue
            if c in ('"', "'", "<", "=", "`"):
                self._emit_error("unexpected-character-in-unquoted-attribute-value")
            if c == "\0":
                self._emit_error("unexpected-null-character")
                self.current_attr_value.append(replacement)
                continue
            self.current_attr_value.append(c)

    def _state_after_attribute_value_quoted(self) -> bool:
        """After attribute value (quoted) state per HTML5 spec §13.2.5.42"""
        # Inline _get_char
        if self.pos >= self.length:
            c: str | None = None
        else:
            c = self.buffer[self.pos]
            self.pos += 1
        self.current_char = c

        if c is None:
            self._emit_error("eof-in-tag")
            self._emit_incomplete_tag_as_text()
            self._flush_text()
            self._emit_token(EOFToken())
            return True
        if c in ("\t", "\n", "\f", " "):
            self._finish_attribute()
            self.state = self.BEFORE_ATTRIBUTE_NAME
            return False
        if c == "/":
            self._finish_attribute()
            self.state = self.SELF_CLOSING_START_TAG
            return False
        if c == ">":
            self._finish_attribute()
            if not self._emit_current_tag():
                self.state = self.DATA
            return False
        # Anything else: parse error, reconsume in before attribute name state
        self._emit_error("missing-whitespace-between-attributes")
        self._finish_attribute()
        self._reconsume_current()
        self.state = self.BEFORE_ATTRIBUTE_NAME
        return False

    def _state_self_closing_start_tag(self) -> bool:
        c = self._get_char()
        if c is None:
            self._emit_error("eof-in-tag")
            self._emit_incomplete_tag_as_text()
            self._flush_text()
            self._emit_token(EOFToken())
            return True
        if c == ">":
            self.current_tag_self_closing = True
            self._emit_current_tag()
            self.state = self.DATA
            return False
        self._emit_error("unexpected-character-after-solidus-in-tag")
        self._reconsume_current()
        self.state = self.BEFORE_ATTRIBUTE_NAME
        return False

    def _state_markup_declaration_open(self) -> bool:
        # Note: Comment handling (<!--) is optimized in DATA state fast-path
        # This code only handles DOCTYPE and CDATA, or malformed markup
        if self._consume_case_insensitive("DOCTYPE"):
            self.current_doctype_name.clear()
            self.current_doctype_public = None
            self.current_doctype_system = None
            self.current_doctype_force_quirks = False
            self.state = self.DOCTYPE
            return False
        if self._consume_if("[CDATA["):
            # CDATA sections are only valid in foreign content (SVG/MathML)
            # Check if the adjusted current node is in a foreign namespace
            stack = self.sink.open_elements
            if stack:
                current = stack[-1]
                if current and current.namespace not in {None, "html"}:
                    # Proper CDATA section in foreign content
                    self.state = self.CDATA_SECTION
                    return False
            # Treat as bogus comment in HTML context, preserving "[CDATA[" prefix
            self._emit_error("cdata-in-html-content")
            self.current_comment.clear()
            # Add the consumed "[CDATA[" text to the comment
            for ch in "[CDATA[":
                self.current_comment.append(ch)
            self.state = self.BOGUS_COMMENT
            return False
        self._emit_error("incorrectly-opened-comment")
        self.current_comment.clear()
        # Don't reconsume - bogus comment starts from current position
        self.state = self.BOGUS_COMMENT
        return False

    def _state_comment_start(self) -> bool:
        replacement = "\ufffd"
        c = self._get_char()
        if c is None:
            self._emit_error("eof-in-comment")
            self._emit_comment()
            self._emit_token(EOFToken())
            return True
        if c == "-":
            self.state = self.COMMENT_START_DASH
            return False
        if c == ">":
            self._emit_error("abrupt-closing-of-empty-comment")
            self._emit_comment()
            self.state = self.DATA
            return False
        if c == "\0":
            self._emit_error("unexpected-null-character")
            self.current_comment.append(replacement)
        else:
            self.current_comment.append(c)
        self.state = self.COMMENT
        return False

    def _state_comment_start_dash(self) -> bool:
        replacement = "\ufffd"
        c = self._get_char()
        if c is None:
            self._emit_error("eof-in-comment")
            self._emit_comment()
            self._emit_token(EOFToken())
            return True
        if c == "-":
            self.state = self.COMMENT_END
            return False
        if c == ">":
            self._emit_error("abrupt-closing-of-empty-comment")
            self._emit_comment()
            self.state = self.DATA
            return False
        if c == "\0":
            self._emit_error("unexpected-null-character")
            self.current_comment.extend(("-", replacement))
        else:
            self.current_comment.extend(("-", c))
        self.state = self.COMMENT
        return False

    def _state_comment(self) -> bool:
        replacement = "\ufffd"
        while True:
            if self._consume_comment_run():
                continue
            # Inline _get_char
            if self.pos >= self.length:
                c: str | None = None
            else:
                c = self.buffer[self.pos]
                self.pos += 1
            self.current_char = c

            if c is None:
                self._emit_error("eof-in-comment")
                self._emit_comment()
                self._emit_token(EOFToken())
                return True
            if c == "-":
                self.state = self.COMMENT_END_DASH
                return False
            # c == "\0" - the only remaining possibility after _consume_comment_run
            self._emit_error("unexpected-null-character")
            self.current_comment.append(replacement)

    def _state_comment_end_dash(self) -> bool:
        replacement = "\ufffd"
        c = self._get_char()
        if c is None:
            self._emit_error("eof-in-comment")
            self._emit_comment()
            self._emit_token(EOFToken())
            return True
        if c == "-":
            self.state = self.COMMENT_END
            return False
        if c == "\0":
            self._emit_error("unexpected-null-character")
            self.current_comment.extend(("-", replacement))
            self.state = self.COMMENT
            return False
        # Per spec: append "-" and current char, switch to COMMENT state
        self.current_comment.extend(("-", c))
        self.state = self.COMMENT
        return False

    def _state_comment_end(self) -> bool:
        replacement = "\ufffd"
        c = self._get_char()
        if c is None:
            self._emit_error("eof-in-comment")
            self._emit_comment()
            self._emit_token(EOFToken())
            return True
        if c == ">":
            self._emit_comment()
            self.state = self.DATA
            return False
        if c == "!":
            self.state = self.COMMENT_END_BANG
            return False
        if c == "-":
            self.current_comment.append("-")
            return False
        if c == "\0":
            self._emit_error("unexpected-null-character")
            self.current_comment.extend(("--", replacement))
            self.state = self.COMMENT
            return False
        self._emit_error("incorrectly-closed-comment")
        self.current_comment.extend(("--", c))
        self.state = self.COMMENT
        return False

    def _state_comment_end_bang(self) -> bool:
        replacement = "\ufffd"
        c = self._get_char()
        if c is None:
            self._emit_error("eof-in-comment")
            self._emit_comment()
            self._emit_token(EOFToken())
            return True
        if c == "-":
            self.current_comment.append("-")
            self.current_comment.append("-")
            self.current_comment.append("!")
            self.state = self.COMMENT_END_DASH
            return False
        if c == ">":
            self._emit_error("incorrectly-closed-comment")
            self._emit_comment()
            self.state = self.DATA
            return False
        if c == "\0":
            self._emit_error("unexpected-null-character")
            self.current_comment.append("-")
            self.current_comment.append("-")
            self.current_comment.append("!")
            self.current_comment.append(replacement)
            self.state = self.COMMENT
            return False
        self.current_comment.append("-")
        self.current_comment.append("-")
        self.current_comment.append("!")
        self.current_comment.append(c)
        self.state = self.COMMENT
        return False

    def _state_bogus_comment(self) -> bool:
        replacement = "\ufffd"
        while True:
            c = self._get_char()
            if c is None:
                self._emit_comment()
                self._emit_token(EOFToken())
                return True
            if c == ">":
                self._emit_comment()
                self.state = self.DATA
                return False
            if c == "\0":
                self.current_comment.append(replacement)
            else:
                self.current_comment.append(c)

    def _state_doctype(self) -> bool:
        c = self._get_char()
        if c is None:
            self._emit_error("eof-in-doctype")
            self.current_doctype_force_quirks = True
            self._emit_doctype()
            self._emit_token(EOFToken())
            return True
        if c in ("\t", "\n", "\f", " "):
            self.state = self.BEFORE_DOCTYPE_NAME
            return False
        if c == ">":
            self._emit_error("expected-doctype-name-but-got-right-bracket")
            self.current_doctype_force_quirks = True
            self._emit_doctype()
            self.state = self.DATA
            return False
        self._emit_error("missing-whitespace-before-doctype-name")
        self._reconsume_current()
        self.state = self.BEFORE_DOCTYPE_NAME
        return False

    def _state_before_doctype_name(self) -> bool:
        while True:
            c = self._get_char()
            if c is None:
                self._emit_error("eof-in-doctype")
                self.current_doctype_force_quirks = True
                self._emit_doctype()
                self._emit_token(EOFToken())
                return True
            if c in ("\t", "\n", "\f", " "):
                return False
            if c == ">":
                self._emit_error("expected-doctype-name-but-got-right-bracket")
                self.current_doctype_force_quirks = True
                self._emit_doctype()
                self.state = self.DATA
                return False
            if "A" <= c <= "Z":
                self.current_doctype_name.append(chr(ord(c) + 32))
            elif c == "\0":
                self._emit_error("unexpected-null-character")
                self.current_doctype_name.append("\ufffd")
            else:
                self.current_doctype_name.append(c)
            self.state = self.DOCTYPE_NAME
            return False

    def _state_doctype_name(self) -> bool:
        while True:
            c = self._get_char()
            if c is None:
                self._emit_error("eof-in-doctype")
                self.current_doctype_force_quirks = True
                self._emit_doctype()
                self._emit_token(EOFToken())
                return True
            if c in ("\t", "\n", "\f", " "):
                self.state = self.AFTER_DOCTYPE_NAME
                return False
            if c == ">":
                self._emit_doctype()
                self.state = self.DATA
                return False
            if "A" <= c <= "Z":
                self.current_doctype_name.append(chr(ord(c) + 32))
                continue
            if c == "\0":
                self._emit_error("unexpected-null-character")
                self.current_doctype_name.append("\ufffd")
                continue
            self.current_doctype_name.append(c)

    def _state_after_doctype_name(self) -> bool:
        if self._consume_case_insensitive("PUBLIC"):
            self.state = self.AFTER_DOCTYPE_PUBLIC_KEYWORD
            return False
        if self._consume_case_insensitive("SYSTEM"):
            self.state = self.AFTER_DOCTYPE_SYSTEM_KEYWORD
            return False
        while True:
            c = self._get_char()
            if c is None:
                self._emit_error("eof-in-doctype")
                self.current_doctype_force_quirks = True
                self._emit_doctype()
                self._emit_token(EOFToken())
                return True
            if c in ("\t", "\n", "\f", " "):
                continue
            if c == ">":
                self._emit_doctype()
                self.state = self.DATA
                return False
            self._emit_error("missing-whitespace-after-doctype-name")
            self.current_doctype_force_quirks = True
            self._reconsume_current()
            self.state = self.BOGUS_DOCTYPE
            return False

    def _state_after_doctype_public_keyword(self) -> bool:
        while True:
            c = self._get_char()
            if c is None:
                self._emit_error("missing-quote-before-doctype-public-identifier")
                self.current_doctype_force_quirks = True
                self._emit_doctype()
                self._emit_token(EOFToken())
                return True
            if c in ("\t", "\n", "\f", " "):
                self.state = self.BEFORE_DOCTYPE_PUBLIC_IDENTIFIER
                return False
            if c == '"':
                self._emit_error("missing-whitespace-before-doctype-public-identifier")
                self.current_doctype_public = []
                self.state = self.DOCTYPE_PUBLIC_IDENTIFIER_DOUBLE_QUOTED
                return False
            if c == "'":
                self._emit_error("missing-whitespace-before-doctype-public-identifier")
                self.current_doctype_public = []
                self.state = self.DOCTYPE_PUBLIC_IDENTIFIER_SINGLE_QUOTED
                return False
            if c == ">":
                self._emit_error("missing-doctype-public-identifier")
                self.current_doctype_force_quirks = True
                self._emit_doctype()
                self.state = self.DATA
                return False
            self._emit_error("unexpected-character-after-doctype-public-keyword")
            self.current_doctype_force_quirks = True
            self._reconsume_current()
            self.state = self.BOGUS_DOCTYPE
            return False

    def _state_after_doctype_system_keyword(self) -> bool:
        while True:
            c = self._get_char()
            if c is None:
                self._emit_error("missing-quote-before-doctype-system-identifier")
                self.current_doctype_force_quirks = True
                self._emit_doctype()
                self._emit_token(EOFToken())
                return True
            if c in ("\t", "\n", "\f", " "):
                self.state = self.BEFORE_DOCTYPE_SYSTEM_IDENTIFIER
                return False
            if c == '"':
                self._emit_error("missing-whitespace-after-doctype-public-identifier")
                self.current_doctype_system = []
                self.state = self.DOCTYPE_SYSTEM_IDENTIFIER_DOUBLE_QUOTED
                return False
            if c == "'":
                self._emit_error("missing-whitespace-after-doctype-public-identifier")
                self.current_doctype_system = []
                self.state = self.DOCTYPE_SYSTEM_IDENTIFIER_SINGLE_QUOTED
                return False
            if c == ">":
                self._emit_error("missing-doctype-system-identifier")
                self.current_doctype_force_quirks = True
                self._emit_doctype()
                self.state = self.DATA
                return False
            self._emit_error("unexpected-character-after-doctype-system-keyword")
            self.current_doctype_force_quirks = True
            self._reconsume_current()
            self.state = self.BOGUS_DOCTYPE
            return False

    def _state_before_doctype_public_identifier(self) -> bool:
        while True:
            c = self._get_char()
            if c is None:
                self._emit_error("missing-doctype-public-identifier")
                self.current_doctype_force_quirks = True
                self._emit_doctype()
                self._emit_token(EOFToken())
                return True
            if c in ("\t", "\n", "\f", " "):
                continue
            if c == '"':
                self.current_doctype_public = []
                self.state = self.DOCTYPE_PUBLIC_IDENTIFIER_DOUBLE_QUOTED
                return False
            if c == "'":
                self.current_doctype_public = []
                self.state = self.DOCTYPE_PUBLIC_IDENTIFIER_SINGLE_QUOTED
                return False
            if c == ">":
                self._emit_error("missing-doctype-public-identifier")
                self.current_doctype_force_quirks = True
                self._emit_doctype()
                self.state = self.DATA
                return False
            self._emit_error("missing-quote-before-doctype-public-identifier")
            self.current_doctype_force_quirks = True
            self._reconsume_current()
            self.state = self.BOGUS_DOCTYPE
            return False

    def _state_doctype_public_identifier_double_quoted(self) -> bool:
        if self.current_doctype_public is None:  # pragma: no cover
            self.current_doctype_public = []
        while True:
            c = self._get_char()
            if c is None:
                self._emit_error("eof-in-doctype-public-identifier")
                self.current_doctype_force_quirks = True
                self._emit_doctype()
                self._emit_token(EOFToken())
                return True
            if c == '"':
                self.state = self.AFTER_DOCTYPE_PUBLIC_IDENTIFIER
                return False
            if c == "\0":
                self._emit_error("unexpected-null-character")
                self.current_doctype_public.append("\ufffd")
                continue
            if c == ">":
                self._emit_error("abrupt-doctype-public-identifier")
                self.current_doctype_force_quirks = True
                self._emit_doctype()
                self.state = self.DATA
                return False
            self.current_doctype_public.append(c)

    def _state_doctype_public_identifier_single_quoted(self) -> bool:
        if self.current_doctype_public is None:  # pragma: no cover
            self.current_doctype_public = []
        while True:
            c = self._get_char()
            if c is None:
                self._emit_error("eof-in-doctype-public-identifier")
                self.current_doctype_force_quirks = True
                self._emit_doctype()
                self._emit_token(EOFToken())
                return True
            if c == "'":
                self.state = self.AFTER_DOCTYPE_PUBLIC_IDENTIFIER
                return False
            if c == "\0":
                self._emit_error("unexpected-null-character")
                self.current_doctype_public.append("\ufffd")
                continue
            if c == ">":
                self._emit_error("abrupt-doctype-public-identifier")
                self.current_doctype_force_quirks = True
                self._emit_doctype()
                self.state = self.DATA
                return False
            self.current_doctype_public.append(c)

    def _state_after_doctype_public_identifier(self) -> bool:
        while True:
            c = self._get_char()
            if c is None:
                self._emit_error("missing-whitespace-between-doctype-public-and-system-identifiers")
                self.current_doctype_force_quirks = True
                self._emit_doctype()
                self._emit_token(EOFToken())
                return True
            if c in ("\t", "\n", "\f", " "):
                self.state = self.BETWEEN_DOCTYPE_PUBLIC_AND_SYSTEM_IDENTIFIERS
                return False
            if c == ">":
                self._emit_doctype()
                self.state = self.DATA
                return False
            if c == '"':
                self._emit_error("missing-whitespace-between-doctype-public-and-system-identifiers")
                self.current_doctype_system = []
                self.state = self.DOCTYPE_SYSTEM_IDENTIFIER_DOUBLE_QUOTED
                return False
            if c == "'":
                self._emit_error("missing-whitespace-between-doctype-public-and-system-identifiers")
                self.current_doctype_system = []
                self.state = self.DOCTYPE_SYSTEM_IDENTIFIER_SINGLE_QUOTED
                return False
            self._emit_error("unexpected-character-after-doctype-public-identifier")
            self.current_doctype_force_quirks = True
            self._reconsume_current()
            self.state = self.BOGUS_DOCTYPE
            return False

    def _state_between_doctype_public_and_system_identifiers(self) -> bool:
        while True:
            c = self._get_char()
            if c is None:
                self._emit_error("missing-quote-before-doctype-system-identifier")
                self.current_doctype_force_quirks = True
                self._emit_doctype()
                self._emit_token(EOFToken())
                return True
            if c in ("\t", "\n", "\f", " "):
                continue
            if c == ">":
                self._emit_doctype()
                self.state = self.DATA
                return False
            if c == '"':
                self.current_doctype_system = []
                self.state = self.DOCTYPE_SYSTEM_IDENTIFIER_DOUBLE_QUOTED
                return False
            if c == "'":
                self.current_doctype_system = []
                self.state = self.DOCTYPE_SYSTEM_IDENTIFIER_SINGLE_QUOTED
                return False
            self._emit_error("missing-quote-before-doctype-system-identifier")
            self.current_doctype_force_quirks = True
            self._reconsume_current()
            self.state = self.BOGUS_DOCTYPE
            return False

    def _state_before_doctype_system_identifier(self) -> bool:
        while True:
            c = self._get_char()
            if c is None:
                self._emit_error("missing-doctype-system-identifier")
                self.current_doctype_force_quirks = True
                self._emit_doctype()
                self._emit_token(EOFToken())
                return True
            if c in ("\t", "\n", "\f", " "):
                continue
            if c == '"':
                self.current_doctype_system = []
                self.state = self.DOCTYPE_SYSTEM_IDENTIFIER_DOUBLE_QUOTED
                return False
            if c == "'":
                self.current_doctype_system = []
                self.state = self.DOCTYPE_SYSTEM_IDENTIFIER_SINGLE_QUOTED
                return False
            if c == ">":
                self._emit_error("missing-doctype-system-identifier")
                self.current_doctype_force_quirks = True
                self._emit_doctype()
                self.state = self.DATA
                return False
            self._emit_error("missing-quote-before-doctype-system-identifier")
            self.current_doctype_force_quirks = True
            self._reconsume_current()
            self.state = self.BOGUS_DOCTYPE
            return False

    def _state_doctype_system_identifier_double_quoted(self) -> bool:
        if self.current_doctype_system is None:  # pragma: no cover
            self.current_doctype_system = []
        while True:
            c = self._get_char()
            if c is None:
                self._emit_error("eof-in-doctype-system-identifier")
                self.current_doctype_force_quirks = True
                self._emit_doctype()
                self._emit_token(EOFToken())
                return True
            if c == '"':
                self.state = self.AFTER_DOCTYPE_SYSTEM_IDENTIFIER
                return False
            if c == "\0":
                self._emit_error("unexpected-null-character")
                self.current_doctype_system.append("\ufffd")
                continue
            if c == ">":
                self._emit_error("abrupt-doctype-system-identifier")
                self.current_doctype_force_quirks = True
                self._emit_doctype()
                self.state = self.DATA
                return False
            self.current_doctype_system.append(c)

    def _state_doctype_system_identifier_single_quoted(self) -> bool:
        if self.current_doctype_system is None:  # pragma: no cover
            self.current_doctype_system = []
        while True:
            c = self._get_char()
            if c is None:
                self._emit_error("eof-in-doctype-system-identifier")
                self.current_doctype_force_quirks = True
                self._emit_doctype()
                self._emit_token(EOFToken())
                return True
            if c == "'":
                self.state = self.AFTER_DOCTYPE_SYSTEM_IDENTIFIER
                return False
            if c == "\0":
                self._emit_error("unexpected-null-character")
                self.current_doctype_system.append("\ufffd")
                continue
            if c == ">":
                self._emit_error("abrupt-doctype-system-identifier")
                self.current_doctype_force_quirks = True
                self._emit_doctype()
                self.state = self.DATA
                return False
            self.current_doctype_system.append(c)

    def _state_after_doctype_system_identifier(self) -> bool:
        while True:
            c = self._get_char()
            if c is None:
                self._emit_error("eof-in-doctype")
                self.current_doctype_force_quirks = True
                self._emit_doctype()
                self._emit_token(EOFToken())
                return True
            if c in ("\t", "\n", "\f", " "):
                continue
            if c == ">":
                self._emit_doctype()
                self.state = self.DATA
                return False
            self._emit_error("unexpected-character-after-doctype-system-identifier")
            self._reconsume_current()
            self.state = self.BOGUS_DOCTYPE
            return False

    def _state_bogus_doctype(self) -> bool:
        while True:
            c = self._get_char()
            if c is None:
                self._emit_doctype()
                self._emit_token(EOFToken())
                return True
            if c == ">":
                self._emit_doctype()
                self.state = self.DATA
                return False

    # ---------------------
    # Low-level helpers
    # ---------------------

    def _get_char(self) -> str | None:
        if self.reconsume:
            self.reconsume = False
            return self.current_char

        pos = self.pos
        if pos >= self.length:
            self.current_char = None
            return None

        c = self.buffer[pos]
        self.pos = pos + 1
        self.current_char = c
        if c == "<":
            self.current_token_start_pos = pos
        if self.collect_errors and not c.isascii() and _is_noncharacter_codepoint(ord(c)):
            self._emit_error_at_pos("noncharacter-in-input-stream", pos)
        return c

    def _reconsume_current(self) -> None:
        self.reconsume = True

    def _append_text(self, text: str) -> None:
        """Append text to buffer, recording start position if this is the first chunk."""
        if (self.collect_errors or self.track_node_locations) and not self.text_buffer:
            # Record where text started (current position before this chunk)
            self.text_start_pos = self.pos
        self.text_buffer.append(text)

    def _flush_text(self) -> None:
        if not self.text_buffer:
            return

        # Optimization: Avoid join for single chunk
        # text_buffer is never populated with empty strings
        if len(self.text_buffer) == 1:
            data = self.text_buffer[0]
        else:
            data = "".join(self.text_buffer)

        raw_len = len(data) if self.collect_errors else 0

        self.text_buffer.clear()
        # U+0000 NULL is a parse error in text.
        # Emit one error per NULL at the *actual* character position.
        if self.collect_errors and "\0" in data:
            base_pos = self.text_start_pos
            search_from = 0
            while True:
                idx = data.find("\0", search_from)
                if idx == -1:
                    break
                error_pos = base_pos + idx

                # Compute column at error_pos (1-indexed).
                last_newline = self.buffer.rfind("\n", 0, error_pos + 1)
                if last_newline == -1:
                    column = error_pos + 1
                else:
                    column = error_pos - last_newline
                line = self._get_line_at_pos(error_pos)

                message = generate_error_message("unexpected-null-character")
                self.errors.append(
                    ParseError(
                        "unexpected-null-character",
                        line=line,
                        column=column,
                        category="tokenizer",
                        message=message,
                        source_html=self.buffer,
                    )
                )

                search_from = idx + 1

        # Per HTML5 spec:
        # - RCDATA state (title, textarea): decode character references
        # - RAWTEXT state (style, script, etc): do NOT decode
        # - PLAINTEXT state: do NOT decode
        # - CDATA sections: do NOT decode
        if self.state >= self.PLAINTEXT or self.CDATA_SECTION <= self.state <= self.CDATA_SECTION_END:
            pass
        elif self.state >= self.RAWTEXT:
            pass
        else:
            if "&" in data:
                report_error = self._emit_error if self.collect_errors else None
                data = decode_entities_in_text(data, report_error=report_error)
        # Apply XML coercion if enabled
        if self.opts.xml_coercion:
            data = _coerce_text_for_xml(data)

        # Record position at END of raw text (1-indexed column = raw_len)
        if self.collect_errors:
            self._record_text_end_position(raw_len)
        if self.track_node_locations:
            self.last_token_start_pos = self.text_start_pos
        self.sink.process_characters(data)
        # Note: process_characters never returns Plaintext or RawData
        # State switches happen via _emit_current_tag instead

    def _finish_attribute(self) -> None:
        attr_name_buffer = self.current_attr_name
        if not attr_name_buffer:
            return
        if len(attr_name_buffer) == 1:
            name = attr_name_buffer[0]
        else:
            name = "".join(attr_name_buffer)
        attrs = self.current_tag_attrs
        is_duplicate = name in attrs
        attr_name_buffer.clear()
        attr_value_buffer = self.current_attr_value
        if is_duplicate:
            self._emit_error("duplicate-attribute")
            attr_value_buffer.clear()
            self.current_attr_value_has_amp = False
            return
        if not attr_value_buffer:
            value = ""
        elif len(attr_value_buffer) == 1:
            value = attr_value_buffer[0]
        else:
            value = "".join(attr_value_buffer)
        if self.current_attr_value_has_amp:
            report_error = self._emit_error if self.collect_errors else None
            value = decode_entities_in_text(value, in_attribute=True, report_error=report_error)
        attrs[name] = value
        attr_value_buffer.clear()
        self.current_attr_value_has_amp = False

    def _emit_current_tag(self) -> bool:
        name_parts = self.current_tag_name
        part_count = len(name_parts)
        # Note: part_count is always >= 1 because fast-path appends before entering TAG_NAME
        if part_count == 1:
            name = name_parts[0]
        else:
            name = "".join(name_parts)
        attrs = self.current_tag_attrs
        self.current_tag_attrs = {}

        tag = self._tag_token
        tag.kind = self.current_tag_kind
        tag.name = name
        tag.attrs = attrs
        tag.self_closing = self.current_tag_self_closing
        if self.track_tag_positions:
            start_pos = self.current_token_start_pos
            tag.start_pos = start_pos
            tag.end_pos = self.pos
            if self.track_node_locations:
                self.last_token_start_pos = start_pos

        switched_to_rawtext = False
        if self.current_tag_kind == Tag.START:
            self.last_start_tag_name = name
            needs_rawtext_check = (
                name in _RAWTEXT_SWITCH_TAGS
                or name == "plaintext"
                or (name == "noscript" and self.opts.scripting_enabled)
            )
            if needs_rawtext_check:
                stack = self.sink.open_elements
                current_node = stack[-1] if stack else None
                namespace = current_node.namespace if current_node else None
                if namespace is None or namespace == "html":
                    if name in _RCDATA_ELEMENTS:
                        self.state = self.RCDATA
                        self.rawtext_tag_name = name
                        switched_to_rawtext = True
                    elif name in _RAWTEXT_SWITCH_TAGS or name == "noscript":
                        self.state = self.RAWTEXT
                        self.rawtext_tag_name = name
                        switched_to_rawtext = True
                    else:
                        # Must be "plaintext" - the only other way needs_rawtext_check can be True
                        self.state = self.PLAINTEXT
                        switched_to_rawtext = True
        # Remember current state before emitting

        # Emit token to sink
        if self.collect_errors:
            self._record_token_position()
        result = self.sink.process_token(tag)
        if result == 1:  # TokenSinkResult.Plaintext
            self.state = self.PLAINTEXT
            switched_to_rawtext = True

        self.current_tag_self_closing = False
        self.current_tag_kind = Tag.START
        return switched_to_rawtext

    def _emit_incomplete_tag_as_text(self) -> None:
        if not self.opts.emit_bogus_markup_as_text:
            return
        start = self.current_token_start_pos
        if start is None:  # pragma: no cover
            return
        raw = self.buffer[start : self.pos]
        if raw:  # pragma: no branch
            self._emit_token(CharacterTokens(raw))

    def _emit_raw_end_tag_as_text(self, pos: int) -> bool:
        end = self.buffer.find(">", pos)
        if end == -1:
            self.pos = self.length
            self._emit_incomplete_tag_as_text()
            self._emit_token(EOFToken())
            return True
        self.pos = end + 1
        raw = self.buffer[self.current_token_start_pos : self.pos]
        if raw:  # pragma: no branch
            self._emit_token(CharacterTokens(raw))
        self.state = self.DATA
        return False

    def _emit_comment(self) -> None:
        data = "".join(self.current_comment)
        self.current_comment.clear()
        # Apply XML coercion if enabled
        if self.opts.xml_coercion:
            data = _coerce_comment_for_xml(data)
        self._comment_token.data = data
        if self.track_node_locations:
            self._comment_token.start_pos = self.current_token_start_pos
            self.last_token_start_pos = self._comment_token.start_pos
        self._emit_token(self._comment_token)

    def _emit_doctype(self) -> None:
        name = "".join(self.current_doctype_name) if self.current_doctype_name else None
        # If public_id/system_id is a list (even empty), join it; if None, keep None
        public_id = "".join(self.current_doctype_public) if self.current_doctype_public is not None else None
        system_id = "".join(self.current_doctype_system) if self.current_doctype_system is not None else None
        doctype = Doctype(
            name=name,
            public_id=public_id,
            system_id=system_id,
            force_quirks=self.current_doctype_force_quirks,
        )
        self.current_doctype_name.clear()
        self.current_doctype_public = None
        self.current_doctype_system = None
        self.current_doctype_force_quirks = False
        self._emit_token(DoctypeToken(doctype))

    def _emit_token(self, token: AnyToken) -> None:
        if self.collect_errors:
            self._record_token_position()
        self.sink.process_token(token)
        # Note: process_token never returns Plaintext or RawData for state switches
        # State switches happen via _emit_current_tag checking sink response

    def _record_token_position(self) -> None:
        """Record current position as 0-indexed column for the last emitted token.

        Per the spec, the position should be at the end of the token (after the last char).
        """
        # pos points after the last consumed character, which is exactly what we want
        pos = self.pos
        last_newline = self.buffer.rfind("\n", 0, pos)
        if last_newline == -1:
            column = pos  # 0-indexed from start
        else:
            column = pos - last_newline - 1  # 0-indexed from after newline
        self.last_token_line = self._get_line_at_pos(pos)
        self.last_token_column = column

    def _record_text_end_position(self, raw_len: int) -> None:
        """Record position at end of text token (after last character).

        Uses text_start_pos + raw_len to compute where text ends, matching html5lib's
        behavior of reporting the column of the last character (1-indexed).
        """
        # Position of last character of text (0-indexed)
        end_pos = self.text_start_pos + raw_len
        last_newline = self.buffer.rfind("\n", 0, end_pos)
        if last_newline == -1:
            column = end_pos  # 1-indexed column = end_pos (position after last char)
        else:
            column = end_pos - last_newline - 1
        self.last_token_line = self._get_line_at_pos(end_pos)
        self.last_token_column = column

    def _emit_error(self, code: str) -> None:
        if not self.collect_errors:
            return
        # Compute column on-demand: scan backwards to find last newline
        pos = max(0, self.pos - 1)  # Current position being processed
        last_newline = self.buffer.rfind("\n", 0, pos + 1)
        if last_newline == -1:
            column = pos + 1  # 1-indexed from start of input
        else:
            column = pos - last_newline  # 1-indexed from after newline

        message = generate_error_message(code)
        line = self._get_line_at_pos(self.pos)
        self.errors.append(
            ParseError(code, line=line, column=column, category="tokenizer", message=message, source_html=self.buffer)
        )

    def _emit_error_at_pos(self, code: str, pos: int) -> None:
        last_newline = self.buffer.rfind("\n", 0, pos + 1)
        if last_newline == -1:
            column = pos + 1
        else:
            column = pos - last_newline

        message = generate_error_message(code)
        line = self._get_line_at_pos(pos)
        self.errors.append(
            ParseError(code, line=line, column=column, category="tokenizer", message=message, source_html=self.buffer)
        )

    def _consume_if(self, literal: str) -> bool:
        end = self.pos + len(literal)
        if end > self.length:
            return False
        segment = self.buffer[self.pos : end]
        if segment != literal:
            return False
        self.pos = end
        return True

    def _consume_case_insensitive(self, literal: str) -> bool:
        end = self.pos + len(literal)
        if end > self.length:
            return False
        segment = self.buffer[self.pos : end]
        if segment.lower() != literal.lower():
            return False
        self.pos = end
        return True

    def _consume_comment_run(self) -> bool:
        # Note: Comments are never reconsumed
        pos = self.pos
        length = self.length
        if pos >= length:
            return False

        match = _COMMENT_RUN_PATTERN.match(self.buffer, pos)
        if match:
            chunk = match.group(0)
            self.current_comment.append(chunk)
            self.pos = match.end()
            return True
        return False

    def _state_cdata_section(self) -> bool:
        # CDATA section state - consume characters until we see ']'
        while True:
            c = self._get_char()
            if c is None:
                self._emit_error("eof-in-cdata")
                self._flush_text()
                self._emit_token(EOFToken())
                return True
            if c == "]":
                self.state = self.CDATA_SECTION_BRACKET
                return False
            self._append_text(c)

    def _state_cdata_section_bracket(self) -> bool:
        # Seen one ']', check for second ']'
        c = self._get_char()
        if c == "]":
            self.state = self.CDATA_SECTION_END
            return False
        # False alarm, emit the ']' we saw and continue
        self._append_text("]")
        if c is None:
            self._emit_error("eof-in-cdata")
            self._flush_text()
            self._emit_token(EOFToken())
            return True
        self._reconsume_current()
        self.state = self.CDATA_SECTION
        return False

    def _state_cdata_section_end(self) -> bool:
        # Seen ']]', check for '>'
        c = self._get_char()
        if c == ">":
            # End of CDATA section
            self._flush_text()
            self.state = self.DATA
            return False
        # Not the end - we saw ']]' but not '>'. Emit one ']' and check if the next char is another ']'
        self._append_text("]")
        if c is None:
            # EOF after ']]' - emit the second ']' too
            self._append_text("]")
            self._emit_error("eof-in-cdata")
            self._flush_text()
            self._emit_token(EOFToken())
            return True
        if c == "]":
            # Still might be ']]>' sequence, stay in CDATA_SECTION_END
            return False
        # Not a bracket, so emit the second ']', reconsume current char and go back to CDATA_SECTION
        self._append_text("]")
        self._reconsume_current()
        self.state = self.CDATA_SECTION
        return False

    def _state_rcdata(self) -> bool:
        buffer = self.buffer
        length = self.length
        pos = self.pos
        while True:
            if self.reconsume:
                self.reconsume = False
                if self.current_char is None:
                    self._flush_text()
                    self._emit_token(EOFToken())
                    return True
                self.pos -= 1
                pos = self.pos

            # Optimized loop using find
            lt_index = buffer.find("<", pos)
            amp_index = buffer.find("&", pos)
            null_index = buffer.find("\0", pos)

            # Find the nearest special character
            next_special = length
            if lt_index != -1:
                next_special = lt_index
            if amp_index != -1 and amp_index < next_special:
                next_special = amp_index
            if null_index != -1 and null_index < next_special:
                next_special = null_index

            # Consume everything up to the special character
            if next_special > pos:
                chunk = buffer[pos:next_special]
                self._append_text(chunk)
                pos = next_special
                self.pos = pos

            # Handle EOF
            if pos >= length:
                self._flush_text()
                self._emit_token(EOFToken())
                return True

            # Handle special characters - we're at one of them after find()
            if null_index == pos:
                self._emit_error("unexpected-null-character")
                self._append_text("\ufffd")
                pos += 1
                self.pos = pos
            elif amp_index == pos:
                # Ampersand in RCDATA - will be decoded by _flush_text
                self._append_text("&")
                pos += 1
                self.pos = pos
            else:
                # lt_index == pos - the only remaining possibility
                # Less-than sign - might be start of end tag
                pos += 1
                self.pos = pos
                self.state = self.RCDATA_LESS_THAN_SIGN
                return False

    def _state_rcdata_less_than_sign(self) -> bool:
        c = self._get_char()
        if c == "/":
            self.current_tag_name.clear()
            self.state = self.RCDATA_END_TAG_OPEN
            return False
        self._append_text("<")
        self._reconsume_current()
        self.state = self.RCDATA
        return False

    def _state_rcdata_end_tag_open(self) -> bool:
        c = self._get_char()
        if c is not None and ("A" <= c <= "Z" or "a" <= c <= "z"):
            self.current_tag_name.append(c.lower())
            self.original_tag_name.append(c)
            self.state = self.RCDATA_END_TAG_NAME
            return False
        self.text_buffer.extend(("<", "/"))
        self._reconsume_current()
        self.state = self.RCDATA
        return False

    def _state_rcdata_end_tag_name(self) -> bool:
        # Check if this matches the opening tag name
        while True:
            c = self._get_char()
            if c is not None and ("A" <= c <= "Z" or "a" <= c <= "z"):
                self.current_tag_name.append(c.lower())
                self.original_tag_name.append(c)
                continue
            # End of tag name - check if it matches
            tag_name = "".join(self.current_tag_name)
            if tag_name == self.rawtext_tag_name:
                if c == ">":
                    attrs: dict[str, str | None] = {}
                    tag = Tag(Tag.END, tag_name, attrs, False)
                    self._flush_text()
                    self._emit_token(tag)
                    self.state = self.DATA
                    self.rawtext_tag_name = None
                    self.original_tag_name.clear()
                    return False
                if c in (" ", "\t", "\n", "\r", "\f"):
                    # Whitespace after tag name - switch to BEFORE_ATTRIBUTE_NAME
                    self.current_tag_kind = Tag.END
                    self.current_tag_attrs = {}
                    self.state = self.BEFORE_ATTRIBUTE_NAME
                    return False
                if c == "/":
                    self._flush_text()
                    self.current_tag_kind = Tag.END
                    self.current_tag_attrs = {}
                    self.state = self.SELF_CLOSING_START_TAG
                    return False
            # If we hit EOF or tag doesn't match, emit as text
            if c is None:
                # EOF - emit incomplete tag as text (preserve original case) then EOF
                self.text_buffer.extend(("<", "/"))
                for ch in self.original_tag_name:
                    self._append_text(ch)
                self.current_tag_name.clear()
                self.original_tag_name.clear()
                self._flush_text()
                self._emit_token(EOFToken())
                return True
            # Not a matching end tag - emit as text (preserve original case)
            self.text_buffer.extend(("<", "/"))
            for ch in self.original_tag_name:
                self._append_text(ch)
            self.current_tag_name.clear()
            self.original_tag_name.clear()
            self._reconsume_current()
            self.state = self.RCDATA
            return False

    def _state_rawtext(self) -> bool:
        buffer = self.buffer
        length = self.length
        pos = self.pos
        while True:
            if self.reconsume:
                self.reconsume = False
                if self.current_char is None:
                    self._flush_text()
                    self._emit_token(EOFToken())
                    return True
                self.pos -= 1
                pos = self.pos

            # Optimized loop using find
            lt_index = buffer.find("<", pos)
            null_index = buffer.find("\0", pos)
            next_special = lt_index if lt_index != -1 else length
            if null_index != -1 and null_index < next_special:
                if null_index > pos:
                    chunk = buffer[pos:null_index]
                    self._append_text(chunk)
                self._emit_error("unexpected-null-character")
                self._append_text("\ufffd")
                pos = null_index + 1
                self.pos = pos
                continue
            if lt_index == -1:
                if pos < length:
                    chunk = buffer[pos:length]
                    self._append_text(chunk)
                self.pos = length
                self._flush_text()
                self._emit_token(EOFToken())
                return True
            if lt_index > pos:
                chunk = buffer[pos:lt_index]
                self._append_text(chunk)
            pos = lt_index + 1
            self.pos = pos
            # Handle script escaped transition before treating '<' as markup boundary
            if self.rawtext_tag_name == "script":
                next1 = self._peek_char(0)
                next2 = self._peek_char(1)
                next3 = self._peek_char(2)
                if next1 == "!" and next2 == "-" and next3 == "-":
                    self.text_buffer.extend(["<", "!", "-", "-"])
                    self._get_char()
                    self._get_char()
                    self._get_char()
                    self.state = self.SCRIPT_DATA_ESCAPED
                    return False
            self.state = self.RAWTEXT_LESS_THAN_SIGN
            return False

    def _state_rawtext_less_than_sign(self) -> bool:
        c = self._get_char()
        if c == "/":
            self.current_tag_name.clear()
            self.state = self.RAWTEXT_END_TAG_OPEN
            return False
        self._append_text("<")
        self._reconsume_current()
        self.state = self.RAWTEXT
        return False

    def _state_rawtext_end_tag_open(self) -> bool:
        c = self._get_char()
        if c is not None and ("A" <= c <= "Z" or "a" <= c <= "z"):
            self.current_tag_name.append(c.lower())
            self.original_tag_name.append(c)
            self.state = self.RAWTEXT_END_TAG_NAME
            return False
        self.text_buffer.extend(("<", "/"))
        self._reconsume_current()
        self.state = self.RAWTEXT
        return False

    def _state_rawtext_end_tag_name(self) -> bool:
        # Check if this matches the opening tag name
        while True:
            c = self._get_char()
            if c is not None and ("A" <= c <= "Z" or "a" <= c <= "z"):
                self.current_tag_name.append(c.lower())
                self.original_tag_name.append(c)
                continue
            # End of tag name - check if it matches
            tag_name = "".join(self.current_tag_name)
            if tag_name == self.rawtext_tag_name:
                if c == ">":
                    attrs: dict[str, str | None] = {}
                    tag = Tag(Tag.END, tag_name, attrs, False)
                    self._flush_text()
                    self._emit_token(tag)
                    self.state = self.DATA
                    self.rawtext_tag_name = None
                    self.original_tag_name.clear()
                    return False
                if c in (" ", "\t", "\n", "\r", "\f"):
                    # Whitespace after tag name - switch to BEFORE_ATTRIBUTE_NAME
                    self.current_tag_kind = Tag.END
                    self.current_tag_attrs = {}
                    self.state = self.BEFORE_ATTRIBUTE_NAME
                    return False
                if c == "/":
                    self._flush_text()
                    self.current_tag_kind = Tag.END
                    self.current_tag_attrs = {}
                    self.state = self.SELF_CLOSING_START_TAG
                    return False
            # If we hit EOF or tag doesn't match, emit as text
            if c is None:
                # EOF - emit incomplete tag as text (preserve original case) then EOF
                self.text_buffer.extend(("<", "/"))
                for ch in self.original_tag_name:
                    self._append_text(ch)
                self.current_tag_name.clear()
                self.original_tag_name.clear()
                self._flush_text()
                self._emit_token(EOFToken())
                return True
            # Not a matching end tag - emit as text (preserve original case)
            self.text_buffer.extend(("<", "/"))
            for ch in self.original_tag_name:
                self._append_text(ch)
            self.current_tag_name.clear()
            self.original_tag_name.clear()
            self._reconsume_current()
            self.state = self.RAWTEXT
            return False

    def _state_plaintext(self) -> bool:
        # PLAINTEXT state - consume everything as text, no end tag
        if self.pos < self.length:
            remaining = self.buffer[self.pos :]
            # Replace null bytes with replacement character
            if "\0" in remaining:
                remaining = remaining.replace("\0", "\ufffd")
                self._emit_error("unexpected-null-character")
            self._append_text(remaining)
            self.pos = self.length
        self._flush_text()
        self._emit_token(EOFToken())
        return True

    def _state_script_data_escaped(self) -> bool:
        c = self._get_char()
        if c is None:
            self._flush_text()
            self._emit_token(EOFToken())
            return True
        if c == "-":
            self._append_text("-")
            self.state = self.SCRIPT_DATA_ESCAPED_DASH
            return False
        if c == "<":
            self.state = self.SCRIPT_DATA_ESCAPED_LESS_THAN_SIGN
            return False
        if c == "\0":
            self._emit_error("unexpected-null-character")
            self._append_text("\ufffd")
            return False
        self._append_text(c)
        return False

    def _state_script_data_escaped_dash(self) -> bool:
        c = self._get_char()
        if c is None:
            self._flush_text()
            self._emit_token(EOFToken())
            return True
        if c == "-":
            self._append_text("-")
            self.state = self.SCRIPT_DATA_ESCAPED_DASH_DASH
            return False
        if c == "<":
            self.state = self.SCRIPT_DATA_ESCAPED_LESS_THAN_SIGN
            return False
        if c == "\0":
            self._emit_error("unexpected-null-character")
            self._append_text("\ufffd")
            self.state = self.SCRIPT_DATA_ESCAPED
            return False
        self._append_text(c)
        self.state = self.SCRIPT_DATA_ESCAPED
        return False

    def _state_script_data_escaped_dash_dash(self) -> bool:
        c = self._get_char()
        if c is None:
            self._flush_text()
            self._emit_token(EOFToken())
            return True
        if c == "-":
            self._append_text("-")
            return False
        if c == "<":
            self._append_text("<")
            self.state = self.SCRIPT_DATA_ESCAPED_LESS_THAN_SIGN
            return False
        if c == ">":
            self._append_text(">")
            self.state = self.RAWTEXT
            return False
        if c == "\0":
            self._emit_error("unexpected-null-character")
            self._append_text("\ufffd")
            self.state = self.SCRIPT_DATA_ESCAPED
            return False
        self._append_text(c)
        self.state = self.SCRIPT_DATA_ESCAPED
        return False

    def _state_script_data_escaped_less_than_sign(self) -> bool:
        c = self._get_char()
        if c == "/":
            self.temp_buffer.clear()
            self.state = self.SCRIPT_DATA_ESCAPED_END_TAG_OPEN
            return False
        if c is not None and ("A" <= c <= "Z" or "a" <= c <= "z"):
            self.temp_buffer.clear()
            self._append_text("<")
            self._reconsume_current()
            self.state = self.SCRIPT_DATA_DOUBLE_ESCAPE_START
            return False
        self._append_text("<")
        self._reconsume_current()
        self.state = self.SCRIPT_DATA_ESCAPED

        return False

    def _state_script_data_escaped_end_tag_open(self) -> bool:
        c = self._get_char()
        if c is not None and ("A" <= c <= "Z" or "a" <= c <= "z"):
            self.current_tag_name.clear()
            self.original_tag_name.clear()
            self._reconsume_current()
            self.state = self.SCRIPT_DATA_ESCAPED_END_TAG_NAME
            return False
        self.text_buffer.extend(("<", "/"))
        self._reconsume_current()
        self.state = self.SCRIPT_DATA_ESCAPED
        return False

    def _state_script_data_escaped_end_tag_name(self) -> bool:
        c = self._get_char()
        if c is not None and ("A" <= c <= "Z" or "a" <= c <= "z"):
            self.current_tag_name.append(c.lower())
            self.original_tag_name.append(c)
            self.temp_buffer.append(c)
            return False
        # Check if this is an appropriate end tag
        tag_name = "".join(self.current_tag_name)
        is_appropriate = tag_name == self.rawtext_tag_name

        if is_appropriate:
            if c in (" ", "\t", "\n", "\r", "\f"):
                self.current_tag_kind = Tag.END
                self.current_tag_attrs = {}
                self.state = self.BEFORE_ATTRIBUTE_NAME
                return False
            if c == "/":
                self._flush_text()
                self.current_tag_kind = Tag.END
                self.current_tag_attrs = {}
                self.state = self.SELF_CLOSING_START_TAG
                return False
            if c == ">":
                self._flush_text()
                attrs: dict[str, str | None] = {}
                tag = Tag(Tag.END, tag_name, attrs, False)
                self._emit_token(tag)
                self.state = self.DATA
                self.rawtext_tag_name = None
                self.current_tag_name.clear()
                self.original_tag_name.clear()
                return False
        # Not an appropriate end tag
        self.text_buffer.extend(("<", "/"))
        for ch in self.temp_buffer:
            self._append_text(ch)
        self._reconsume_current()
        self.state = self.SCRIPT_DATA_ESCAPED
        return False

    def _state_script_data_double_escape_start(self) -> bool:
        c = self._get_char()
        if c in (" ", "\t", "\n", "\r", "\f", "/", ">"):
            # Check if temp_buffer contains "script"
            temp = "".join(self.temp_buffer).lower()
            if temp == "script":
                self.state = self.SCRIPT_DATA_DOUBLE_ESCAPED
            else:
                self.state = self.SCRIPT_DATA_ESCAPED
            self._append_text(c)
            return False
        if c is not None and ("A" <= c <= "Z" or "a" <= c <= "z"):
            self.temp_buffer.append(c)
            self._append_text(c)
            return False
        self._reconsume_current()
        self.state = self.SCRIPT_DATA_ESCAPED
        return False

    def _state_script_data_double_escaped(self) -> bool:
        c = self._get_char()
        if c is None:
            self._flush_text()
            self._emit_token(EOFToken())
            return True
        if c == "-":
            self._append_text("-")
            self.state = self.SCRIPT_DATA_DOUBLE_ESCAPED_DASH
            return False
        if c == "<":
            self._append_text("<")
            self.state = self.SCRIPT_DATA_DOUBLE_ESCAPED_LESS_THAN_SIGN
            return False
        if c == "\0":
            self._emit_error("unexpected-null-character")
            self._append_text("\ufffd")
            return False
        self._append_text(c)
        return False

    def _state_script_data_double_escaped_dash(self) -> bool:
        c = self._get_char()
        if c is None:
            self._flush_text()
            self._emit_token(EOFToken())
            return True
        if c == "-":
            self._append_text("-")
            self.state = self.SCRIPT_DATA_DOUBLE_ESCAPED_DASH_DASH
            return False
        if c == "<":
            self._append_text("<")
            self.state = self.SCRIPT_DATA_DOUBLE_ESCAPED_LESS_THAN_SIGN
            return False
        if c == "\0":
            self._emit_error("unexpected-null-character")
            self._append_text("\ufffd")
            self.state = self.SCRIPT_DATA_DOUBLE_ESCAPED
            return False
        self._append_text(c)
        self.state = self.SCRIPT_DATA_DOUBLE_ESCAPED
        return False

    def _state_script_data_double_escaped_dash_dash(self) -> bool:
        c = self._get_char()
        if c is None:
            self._flush_text()
            self._emit_token(EOFToken())
            return True
        if c == "-":
            self._append_text("-")
            return False
        if c == "<":
            self._append_text("<")
            self.state = self.SCRIPT_DATA_DOUBLE_ESCAPED_LESS_THAN_SIGN

            return False
        if c == ">":
            self._append_text(">")
            self.state = self.RAWTEXT

            return False
        if c == "\0":
            self._emit_error("unexpected-null-character")
            self._append_text("\ufffd")
            self.state = self.SCRIPT_DATA_DOUBLE_ESCAPED
            return False
        self._append_text(c)
        self.state = self.SCRIPT_DATA_DOUBLE_ESCAPED
        return False

    def _state_script_data_double_escaped_less_than_sign(self) -> bool:
        c = self._get_char()
        if c == "/":
            self.temp_buffer.clear()
            self._append_text("/")
            self.state = self.SCRIPT_DATA_DOUBLE_ESCAPE_END
            return False
        if c is not None and ("A" <= c <= "Z" or "a" <= c <= "z"):
            self.temp_buffer.clear()
            self._reconsume_current()
            self.state = self.SCRIPT_DATA_DOUBLE_ESCAPE_START
            return False
        self._reconsume_current()
        self.state = self.SCRIPT_DATA_DOUBLE_ESCAPED
        return False

    def _state_script_data_double_escape_end(self) -> bool:
        c = self._get_char()
        if c in (" ", "\t", "\n", "\r", "\f", "/", ">"):
            # Check if temp_buffer contains "script"
            temp = "".join(self.temp_buffer).lower()

            if temp == "script":
                self.state = self.SCRIPT_DATA_ESCAPED
            else:
                self.state = self.SCRIPT_DATA_DOUBLE_ESCAPED
            self._append_text(c)
            return False
        if c is not None and ("A" <= c <= "Z" or "a" <= c <= "z"):
            self.temp_buffer.append(c)
            self._append_text(c)
            return False
        self._reconsume_current()
        self.state = self.SCRIPT_DATA_DOUBLE_ESCAPED
        return False


Tokenizer._STATE_HANDLERS = [  # type: ignore[attr-defined]
    Tokenizer._state_data,
    Tokenizer._state_tag_open,
    Tokenizer._state_end_tag_open,
    Tokenizer._state_tag_name,
    Tokenizer._state_before_attribute_name,
    Tokenizer._state_attribute_name,
    Tokenizer._state_after_attribute_name,
    Tokenizer._state_before_attribute_value,
    Tokenizer._state_attribute_value_double,
    Tokenizer._state_attribute_value_single,
    Tokenizer._state_attribute_value_unquoted,
    Tokenizer._state_after_attribute_value_quoted,
    Tokenizer._state_self_closing_start_tag,
    Tokenizer._state_markup_declaration_open,
    Tokenizer._state_comment_start,
    Tokenizer._state_comment_start_dash,
    Tokenizer._state_comment,
    Tokenizer._state_comment_end_dash,
    Tokenizer._state_comment_end,
    Tokenizer._state_comment_end_bang,
    Tokenizer._state_bogus_comment,
    Tokenizer._state_doctype,
    Tokenizer._state_before_doctype_name,
    Tokenizer._state_doctype_name,
    Tokenizer._state_after_doctype_name,
    Tokenizer._state_bogus_doctype,
    Tokenizer._state_after_doctype_public_keyword,
    Tokenizer._state_after_doctype_system_keyword,
    Tokenizer._state_before_doctype_public_identifier,
    Tokenizer._state_doctype_public_identifier_double_quoted,
    Tokenizer._state_doctype_public_identifier_single_quoted,
    Tokenizer._state_after_doctype_public_identifier,
    Tokenizer._state_between_doctype_public_and_system_identifiers,
    Tokenizer._state_before_doctype_system_identifier,
    Tokenizer._state_doctype_system_identifier_double_quoted,
    Tokenizer._state_doctype_system_identifier_single_quoted,
    Tokenizer._state_after_doctype_system_identifier,
    Tokenizer._state_cdata_section,
    Tokenizer._state_cdata_section_bracket,
    Tokenizer._state_cdata_section_end,
    Tokenizer._state_rcdata,
    Tokenizer._state_rcdata_less_than_sign,
    Tokenizer._state_rcdata_end_tag_open,
    Tokenizer._state_rcdata_end_tag_name,
    Tokenizer._state_rawtext,
    Tokenizer._state_rawtext_less_than_sign,
    Tokenizer._state_rawtext_end_tag_open,
    Tokenizer._state_rawtext_end_tag_name,
    Tokenizer._state_plaintext,
    Tokenizer._state_script_data_escaped,
    Tokenizer._state_script_data_escaped_dash,
    Tokenizer._state_script_data_escaped_dash_dash,
    Tokenizer._state_script_data_escaped_less_than_sign,
    Tokenizer._state_script_data_escaped_end_tag_open,
    Tokenizer._state_script_data_escaped_end_tag_name,
    Tokenizer._state_script_data_double_escape_start,
    Tokenizer._state_script_data_double_escaped,
    Tokenizer._state_script_data_double_escaped_dash,
    Tokenizer._state_script_data_double_escaped_dash_dash,
    Tokenizer._state_script_data_double_escaped_less_than_sign,
    Tokenizer._state_script_data_double_escape_end,
]
