# ruff: noqa: S101, PLW2901

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .constants import (
    BUTTON_SCOPE_TERMINATORS,
    DEFAULT_SCOPE_TERMINATORS,
    DEFINITION_SCOPE_TERMINATORS,
    FOREIGN_ATTRIBUTE_ADJUSTMENTS,
    FOREIGN_BREAKOUT_ELEMENTS,
    FORMAT_MARKER,
    FORMATTING_ELEMENTS,
    HTML_INTEGRATION_POINT_SET,
    IMPLIED_END_TAGS,
    LIST_ITEM_SCOPE_TERMINATORS,
    MATHML_ATTRIBUTE_ADJUSTMENTS,
    MATHML_TEXT_INTEGRATION_POINT_SET,
    SPECIAL_ELEMENTS,
    SVG_ATTRIBUTE_ADJUSTMENTS,
    SVG_TAG_NAME_ADJUSTMENTS,
    TABLE_ALLOWED_CHILDREN,
    TABLE_FOSTER_TARGETS,
    TABLE_SCOPE_TERMINATORS,
)
from .errors import generate_error_message
from .node import Comment, Document, DocumentFragment, Element, Node, Template, Text
from .tokens import AnyToken, CharacterTokens, CommentToken, DoctypeToken, EOFToken, ParseError, Tag, TokenSinkResult
from .treebuilder_modes import TreeBuilderModesMixin
from .treebuilder_utils import (
    InsertionMode,
    is_all_whitespace,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class TreeBuilder(TreeBuilderModesMixin):
    __slots__ = (
        "_body_end_handlers",
        "_body_start_handlers",
        "_body_token_handlers",
        "_mode_handlers",
        "_pending_end_tag_end",
        "_pending_end_tag_name",
        "_pending_end_tag_start",
        "active_formatting",
        "collect_errors",
        "document",
        "errors",
        "form_element",
        "fragment_context",
        "fragment_context_element",
        "frameset_ok",
        "head_element",
        "iframe_srcdoc",
        "ignore_lf",
        "insert_from_table",
        "mode",
        "open_elements",
        "original_mode",
        "pending_table_text",
        "pending_table_text_should_error",
        "quirks_mode",
        "scripting_enabled",
        "table_text_original_mode",
        "template_modes",
        "tokenizer",
        "tokenizer_state_override",
        "track_tag_spans",
    )

    _body_end_handlers: dict[str, Callable[[TreeBuilder, Any], Any]]
    _body_start_handlers: dict[str, Callable[[TreeBuilder, Any], Any]]
    _body_token_handlers: dict[str, Callable[[TreeBuilder, Any], Any]]
    _mode_handlers: dict[InsertionMode, Callable[[TreeBuilder, Any], Any]]
    _pending_end_tag_name: str | None
    _pending_end_tag_start: int | None
    _pending_end_tag_end: int | None
    track_tag_spans: bool
    active_formatting: list[Any]
    collect_errors: bool
    document: Node
    errors: list[ParseError]
    form_element: Any | None
    fragment_context: Any | None
    fragment_context_element: Any | None
    frameset_ok: bool
    head_element: Any | None
    iframe_srcdoc: bool
    ignore_lf: bool
    insert_from_table: bool
    mode: InsertionMode
    open_elements: list[Any]
    original_mode: InsertionMode | None  # type: ignore[assignment]
    pending_table_text: list[str]
    pending_table_text_should_error: bool
    quirks_mode: str
    scripting_enabled: bool
    table_text_original_mode: InsertionMode | None  # type: ignore[assignment]
    template_modes: list[InsertionMode]
    tokenizer: Any | None
    tokenizer_state_override: Any | None  # type: ignore[assignment]

    def __init__(
        self,
        fragment_context: Any | None = None,
        iframe_srcdoc: bool = False,
        collect_errors: bool = False,
        scripting_enabled: bool = True,
        track_tag_spans: bool = False,
    ) -> None:
        self.fragment_context = fragment_context
        self.iframe_srcdoc = iframe_srcdoc
        self.collect_errors = collect_errors
        self.scripting_enabled = bool(scripting_enabled)
        self.track_tag_spans = bool(track_tag_spans)
        self.errors = []
        self.tokenizer = None  # Set by parser after tokenizer is created
        self.fragment_context_element = None
        if fragment_context is not None:
            self.document = DocumentFragment()
        else:
            self.document = Document()
        self.mode = InsertionMode.INITIAL
        self.original_mode = None
        self.table_text_original_mode = None
        self.open_elements = []
        self._pending_end_tag_name = None
        self._pending_end_tag_start = None
        self._pending_end_tag_end = None
        self.head_element = None
        self.form_element = None
        self.frameset_ok = True
        self.quirks_mode = "no-quirks"
        self.ignore_lf = False
        self.active_formatting = []
        self.pending_table_text_should_error = False
        self.insert_from_table = False
        self.pending_table_text = []
        self.template_modes = []
        self.tokenizer_state_override = None
        if fragment_context is not None:
            # Fragment parsing per HTML5 spec
            root = self._create_element("html", None, {})
            self.document.append_child(root)
            self.open_elements.append(root)
            # Set mode based on context element name
            namespace = fragment_context.namespace
            context_name = fragment_context.tag_name or ""
            name = context_name.lower()

            # Create a fake context element to establish foreign content context
            # Per spec: "Create an element for the token in the given namespace"
            if namespace and namespace not in {None, "html"}:
                adjusted_name = context_name
                if namespace == "svg":
                    adjusted_name = self._adjust_svg_tag_name(context_name)
                context_element = self._create_element(adjusted_name, namespace, {})
                root.append_child(context_element)
                self.open_elements.append(context_element)
                self.fragment_context_element = context_element

            # For html context, don't pre-create head/body - start in BEFORE_HEAD mode
            # This allows frameset and other elements to be inserted properly
            if name == "html":
                self.mode = InsertionMode.BEFORE_HEAD
            # Table modes only apply to HTML namespace fragments (namespace is None or "html")
            elif namespace in {None, "html"} and name in {"tbody", "thead", "tfoot"}:
                self.mode = InsertionMode.IN_TABLE_BODY
            elif namespace in {None, "html"} and name == "tr":
                self.mode = InsertionMode.IN_ROW
            elif namespace in {None, "html"} and name in {"td", "th"}:
                self.mode = InsertionMode.IN_CELL
            elif namespace in {None, "html"} and name == "caption":
                self.mode = InsertionMode.IN_CAPTION
            elif namespace in {None, "html"} and name == "colgroup":
                self.mode = InsertionMode.IN_COLUMN_GROUP
            elif namespace in {None, "html"} and name == "table":
                self.mode = InsertionMode.IN_TABLE
            else:
                self.mode = InsertionMode.IN_BODY
            # For fragments, frameset_ok starts as False per HTML5 spec
            # This prevents frameset from being inserted in fragment contexts
            self.frameset_ok = False

    def _set_quirks_mode(self, mode: str) -> None:
        self.quirks_mode = mode

    def _parse_error(self, code: str, tag_name: str | None = None, token: AnyToken | None = None) -> None:
        if not self.collect_errors:
            return
        # Use the position of the last emitted token (set by tokenizer before emit)
        line = None
        column = None
        end_column = None
        if self.tokenizer:  # pragma: no branch
            line = self.tokenizer.last_token_line
            column = self.tokenizer.last_token_column

            # Calculate start and end columns based on token type for precise highlighting
            # Note: column from tokenizer points AFTER the last character (0-indexed)
            if token is not None and isinstance(token, Tag):
                # Tag: <name> or </name> plus attributes
                tag_len = len(token.name) + 2  # < + name + >
                if token.kind == Tag.END:
                    tag_len += 1  # </name>
                # Add attribute lengths
                for attr_name, attr_value in token.attrs.items():
                    tag_len += 1 + len(attr_name)  # space + name
                    if attr_value:
                        tag_len += 1 + 2 + len(attr_value)  # = + "value"
                if token.self_closing:
                    tag_len += 1  # /
                # column points after >, so start is column - tag_len + 1 (for 1-indexed)
                start_column = column - tag_len + 1
                column = start_column
                end_column = column + tag_len

        message = generate_error_message(code, tag_name)
        source_html = self.tokenizer.buffer if self.tokenizer else None
        self.errors.append(
            ParseError(
                code,
                line=line,
                column=column,
                category="treebuilder",
                message=message,
                source_html=source_html,
                end_column=end_column,
            )
        )

    def _has_element_in_scope(
        self, target: str, terminators: set[str] | None = None, check_integration_points: bool = True
    ) -> bool:
        if terminators is None:
            terminators = DEFAULT_SCOPE_TERMINATORS
        open_elements = self.open_elements
        for i in range(len(open_elements) - 1, -1, -1):
            node = open_elements[i]
            name = node.name
            if name == target:
                return True
            ns = node.namespace
            if ns == "html" or ns is None:
                if name in terminators:
                    return False
            elif check_integration_points and (
                self._is_html_integration_point(node) or self._is_mathml_text_integration_point(node)
            ):
                return False
        return False

    def _has_element_in_button_scope(self, target: str) -> bool:
        return self._has_element_in_scope(target, BUTTON_SCOPE_TERMINATORS)

    def _pop_until_inclusive(self, name: str) -> None:
        # Callers ensure element exists on stack
        while self.open_elements:  # pragma: no branch
            node = self._pop_current()
            if node.name == name:
                break

    def _pop_until_any_inclusive(self, names: set[str]) -> None:
        # Pop elements until we find one in names (callers ensure element exists)
        while self.open_elements:
            node = self._pop_current()
            if node.name in names:
                return

    def _close_p_element(self) -> bool:
        if self._has_element_in_button_scope("p"):
            self._generate_implied_end_tags("p")
            if self.open_elements[-1].name != "p":
                self._parse_error("unexpected-end-tag", tag_name="p")
            self._pop_until_inclusive("p")
            return True
        return False

    def process_token(self, token: Any) -> Any:
        # Optimization: Use type() identity check instead of isinstance
        token_type = type(token)
        if token_type is DoctypeToken:
            # Check for foreign content first - DOCTYPE in SVG/MathML is a parse error
            if self.open_elements:
                current = self.open_elements[-1]
                if current.namespace not in {None, "html"}:
                    self._parse_error("unexpected-doctype")
                    return TokenSinkResult.Continue
            return self._handle_doctype(token)

        current_token = token
        force_html_mode = False
        if token_type is Tag and token.kind == Tag.END:
            self._pending_end_tag_name = token.name
            if self.track_tag_spans:
                self._pending_end_tag_start = token.start_pos
                self._pending_end_tag_end = token.end_pos
            else:
                self._pending_end_tag_start = None
                self._pending_end_tag_end = None

        # Cache mode handlers list for speed
        mode_handlers = self._MODE_HANDLERS

        try:
            while True:
                # Update token type for current token (it might have changed if reprocessed)
                token_type = type(current_token)

                # Optimization: Check for HTML namespace first (common case)
                current_node = self.open_elements[-1] if self.open_elements else None
                is_html_namespace = current_node is None or current_node.namespace in {None, "html"}

                if force_html_mode or is_html_namespace:
                    force_html_mode = False
                    if self.mode == InsertionMode.IN_BODY:
                        # Inline _mode_in_body for performance
                        if token_type is Tag:
                            # Inline _handle_tag_in_body
                            if current_token.kind == 0:  # Tag.START
                                name = current_token.name
                                if name == "div" or name == "ul" or name == "ol":
                                    # Close <p> if it's in button scope.
                                    self._close_p_element()
                                    self._insert_element(current_token, push=True)
                                    result = None
                                elif name == "p":
                                    result = self._handle_body_start_paragraph(current_token)  # type: ignore[func-returns-value]
                                elif name == "span":
                                    if self.active_formatting:
                                        self._reconstruct_active_formatting_elements()
                                    self._insert_element(current_token, push=True)
                                    self.frameset_ok = False
                                    result = None
                                elif name == "a":
                                    result = self._handle_body_start_a(current_token)  # type: ignore[func-returns-value]
                                elif name == "br" or name == "img":
                                    if self.active_formatting:
                                        self._reconstruct_active_formatting_elements()
                                    self._insert_element(current_token, push=False)
                                    self.frameset_ok = False
                                    result = None
                                elif name == "hr":
                                    self._close_p_element()
                                    self._insert_element(current_token, push=False)
                                    self.frameset_ok = False
                                    result = None
                                else:
                                    handler = self._BODY_START_HANDLERS.get(name)
                                    if handler:
                                        result = handler(self, current_token)
                                    else:
                                        # Inline _handle_body_start_default
                                        # Elements here have no special handler - never in FRAMESET_NEUTRAL/FORMATTING_ELEMENTS
                                        if self.active_formatting:
                                            self._reconstruct_active_formatting_elements()
                                        self._insert_element(current_token, push=True)
                                        if current_token.self_closing:
                                            self._parse_error(
                                                "non-void-html-element-start-tag-with-trailing-solidus",
                                                tag_name=current_token.name,
                                            )
                                        self.frameset_ok = False
                                        result = None
                            else:
                                name = current_token.name
                                if name == "br":
                                    self._parse_error("unexpected-end-tag", tag_name=name)
                                    br_tag = Tag(0, "br", {}, False)
                                    result = self._handle_body_start_br(br_tag)  # type: ignore[func-returns-value]
                                elif name in FORMATTING_ELEMENTS:
                                    self._adoption_agency(name)
                                    result = None
                                else:
                                    handler = self._BODY_END_HANDLERS.get(name)
                                    if handler:
                                        result = handler(self, current_token)
                                    else:
                                        self._any_other_end_tag(name)
                                        result = None
                        elif token_type is CharacterTokens:
                            # Inline _handle_characters_in_body
                            # Only non-whitespace data reaches here (whitespace handled in process_characters)
                            self.frameset_ok = False
                            self._reconstruct_active_formatting_elements()
                            self._append_text(current_token.data)
                            result = None
                        elif token_type is CommentToken:
                            result = self._handle_comment_in_body(current_token)  # type: ignore[func-returns-value]
                        else:  # EOFToken
                            result = self._handle_eof_in_body(current_token)
                    else:
                        result = mode_handlers[self.mode](self, current_token)
                elif self._should_use_foreign_content(current_token):
                    result = self._process_foreign_content(current_token)
                else:
                    # Foreign content stack logic
                    current = current_node
                    # Only pop foreign elements if we're NOT at an HTML/MathML integration point
                    # and NOT about to insert a new foreign element (svg/math)
                    if not isinstance(current_token, EOFToken):
                        # Don't pop at integration points - they stay on stack to receive content
                        if self._is_html_integration_point(current) or self._is_mathml_text_integration_point(current):
                            pass
                        # Don't pop when inserting new svg/math elements
                        if isinstance(current_token, Tag) and current_token.kind == Tag.START:
                            # Optimization: Tokenizer already lowercases tag names
                            name_lower = current_token.name
                            if name_lower in {"svg", "math"}:
                                pass

                    # Special handling: text at integration points inserts directly, bypassing mode dispatch
                    if isinstance(current_token, CharacterTokens):
                        if self._is_mathml_text_integration_point(current):
                            # Tokenizer guarantees non-empty data
                            data = current_token.data
                            if "\x00" in data:
                                data = data.replace("\x00", "")
                            if data:
                                if not is_all_whitespace(data):
                                    self._reconstruct_active_formatting_elements()
                                    self.frameset_ok = False
                                self._append_text(data)
                            result = None
                        else:
                            result = mode_handlers[self.mode](self, current_token)
                    else:
                        # At integration points inside foreign content, check if table tags make sense.
                        if (
                            (
                                self._is_mathml_text_integration_point(current)
                                or self._is_html_integration_point(current)
                            )
                            and isinstance(current_token, Tag)
                            and current_token.kind == Tag.START
                            and self.mode not in {InsertionMode.IN_BODY}
                        ):
                            # Check if we're in a table mode but without an actual table in scope
                            # If so, table tags should be ignored (use IN_BODY mode)
                            is_table_mode = self.mode in {
                                InsertionMode.IN_TABLE,
                                InsertionMode.IN_TABLE_BODY,
                                InsertionMode.IN_ROW,
                                InsertionMode.IN_CELL,
                                InsertionMode.IN_CAPTION,
                                InsertionMode.IN_COLUMN_GROUP,
                            }
                            has_table_in_scope = self._has_in_table_scope("table")
                            if is_table_mode and not has_table_in_scope:
                                # Temporarily use IN_BODY mode for this tag
                                saved_mode = self.mode
                                self.mode = InsertionMode.IN_BODY
                                result = mode_handlers[self.mode](self, current_token)
                                # Restore mode if no mode change was requested
                                if self.mode == InsertionMode.IN_BODY:  # pragma: no branch
                                    self.mode = saved_mode
                            else:
                                result = mode_handlers[self.mode](self, current_token)
                        else:
                            result = mode_handlers[self.mode](self, current_token)

                if result is None:
                    result_to_return = self.tokenizer_state_override or TokenSinkResult.Continue
                    self.tokenizer_state_override = None
                    return result_to_return
                # Result is (instruction, mode, token) or (instruction, mode, token, force_html)
                _instruction, mode, token_override = result[0], result[1], result[2]
                if len(result) == 4:
                    force_html_mode = result[3]
                # All mode handlers that return a tuple use "reprocess" instruction
                self.mode = mode
                current_token = token_override
                # Continue loop to reprocess
        finally:
            self._pending_end_tag_name = None
            self._pending_end_tag_start = None
            self._pending_end_tag_end = None

    def finish(self) -> Node:
        if self.fragment_context is not None:
            # For fragments, remove the html wrapper and promote its children
            # Note: html element is always created in fragment setup, so children[0] is always "html"
            assert self.document.children is not None
            root = self.document.children[0]
            context_elem = self.fragment_context_element
            if context_elem is not None and context_elem.parent is root:
                for child in list(context_elem.children):
                    context_elem.remove_child(child)
                    root.append_child(child)
                root.remove_child(context_elem)
            for child in list(root.children):
                root.remove_child(child)
                self.document.append_child(child)
            self.document.remove_child(root)

        # Populate selectedcontent elements per HTML5 spec
        self._populate_selectedcontent(self.document)

        if self.tokenizer is not None and self.track_tag_spans:  # pragma: no branch
            self.document._source_html = self.tokenizer.buffer

        return self.document

    # Insertion mode dispatch ------------------------------------------------

    def _append_comment_to_document(self, text: str) -> None:
        node = Comment(data=text)
        if self.tokenizer is not None and self.tokenizer.track_node_locations:
            node._origin_pos = self.tokenizer.last_token_start_pos
            if node._origin_pos is not None:
                node._origin_line, node._origin_col = self.tokenizer.location_at_pos(node._origin_pos)
        self.document.append_child(node)

    def _append_comment(self, text: str, parent: Any | None = None) -> None:
        if parent is None:
            parent = self._current_node_or_html()
        # If parent is a template, insert into its content fragment
        if type(parent) is Template and parent.template_content:
            parent = parent.template_content
        node = Comment(data=text)
        if self.tokenizer is not None and self.tokenizer.track_node_locations:
            node._origin_pos = self.tokenizer.last_token_start_pos
            if node._origin_pos is not None:
                node._origin_line, node._origin_col = self.tokenizer.location_at_pos(node._origin_pos)
        parent.append_child(node)

    def _append_text(self, text: str) -> None:
        if self.ignore_lf:
            self.ignore_lf = False
            if text.startswith("\n"):
                text = text[1:]
                if not text:
                    return

        if "\f" in text:
            text = text.replace("\f", " ")

        # Guard against empty stack
        if not self.open_elements:  # pragma: no cover
            return

        # Fast path optimization for common case
        target = self.open_elements[-1]

        if target.name not in TABLE_FOSTER_TARGETS and type(target) is not Template:
            children = target.children
            if children:
                last_child = children[-1]
                if type(last_child) is Text:
                    last_child.data = (last_child.data or "") + text
                    return

            node = Text(text)
            if self.tokenizer is not None and self.tokenizer.track_node_locations:
                node._origin_pos = self.tokenizer.last_token_start_pos
                if node._origin_pos is not None:
                    node._origin_line, node._origin_col = self.tokenizer.location_at_pos(node._origin_pos)
            children.append(node)
            node.parent = target
            return

        target = self._current_node_or_html()
        foster_parenting = self._should_foster_parenting(target, is_text=True)

        # Reconstruct active formatting BEFORE getting insertion location when foster parenting
        if foster_parenting:
            self._reconstruct_active_formatting_elements()

        # Always use appropriate insertion location to handle templates
        parent, position = self._appropriate_insertion_location(foster_parenting=foster_parenting)

        # Coalesce with adjacent text node if possible
        if position > 0 and parent.children[position - 1].name == "#text":
            parent.children[position - 1].data = (parent.children[position - 1].data or "") + text
            return

        node = Text(text)
        if self.tokenizer is not None and self.tokenizer.track_node_locations:
            node._origin_pos = self.tokenizer.last_token_start_pos
            if node._origin_pos is not None:
                node._origin_line, node._origin_col = self.tokenizer.location_at_pos(node._origin_pos)
        children = parent.children
        if position < len(children):
            children.insert(position, node)
        else:
            children.append(node)
        node.parent = parent

    def _current_node_or_html(self) -> Any:
        if self.open_elements:
            return self.open_elements[-1]
        # Stack empty - find html element in document children
        # (may not be first if there are comments/doctype before it)
        children = self.document.children
        if children is not None:
            for child in children:
                if child.name == "html":
                    return child
            # Edge case: no html found, return first child or None
            return children[0] if children else None  # pragma: no cover
        return None  # pragma: no cover

    def _create_root(self, attrs: dict[str, str | None]) -> Any:
        node = Node("html", attrs=attrs, namespace="html")
        self.document.append_child(node)
        self.open_elements.append(node)
        return node

    def _insert_element(self, tag: Any, *, push: bool, namespace: str = "html") -> Any:
        node: Element | Template
        if tag.name == "template" and namespace == "html":
            node = Template(tag.name, attrs=tag.attrs, namespace=namespace)
        else:
            node = Element(tag.name, attrs=tag.attrs, namespace=namespace)
        if self.track_tag_spans:
            node._start_tag_start = tag.start_pos
            node._start_tag_end = tag.end_pos
        node._self_closing = bool(getattr(tag, "self_closing", False))

        if self.tokenizer is not None and self.tokenizer.track_node_locations:
            node._origin_pos = tag.start_pos
            if node._origin_pos is not None:
                node._origin_line, node._origin_col = self.tokenizer.location_at_pos(node._origin_pos)

        # Fast path for common case: not inserting from table
        if not self.insert_from_table:
            target = self._current_node_or_html()

            # Handle template content insertion
            if type(target) is Template:
                parent = target.template_content
            else:
                parent = target

            if parent is not None:  # pragma: no branch
                parent.append_child(node)

            if push:
                self.open_elements.append(node)
            return node

        target = self._current_node_or_html()
        foster_parenting = self._should_foster_parenting(target, for_tag=tag.name)
        parent, position = self._appropriate_insertion_location(foster_parenting=foster_parenting)
        self._insert_node_at(parent, position, node)
        if push:
            self.open_elements.append(node)
        return node

    def _insert_phantom(self, name: str) -> Any:
        attrs: dict[str, str | None] = {}
        tag = Tag(Tag.START, name, attrs, False)
        return self._insert_element(tag, push=True)

    def _insert_body_if_missing(self) -> None:
        html_node = self._find_last_on_stack("html")
        node = Node("body", namespace="html")
        if html_node is not None:  # pragma: no branch
            html_node.append_child(node)
            node.parent = html_node
        self.open_elements.append(node)

    def _create_element(self, name: str, namespace: str | None, attrs: dict[str, str | None]) -> Any:
        ns = namespace or "html"
        return Element(name, attrs, ns)

    def _maybe_mark_end_tag(self, node: Any) -> None:
        if self._pending_end_tag_name is None:
            return
        if getattr(node, "name", None) != self._pending_end_tag_name:
            return
        node._end_tag_present = True
        if self.track_tag_spans:
            node._end_tag_start = self._pending_end_tag_start
            node._end_tag_end = self._pending_end_tag_end
        self._pending_end_tag_name = None
        self._pending_end_tag_start = None
        self._pending_end_tag_end = None

    def _pop_current(self) -> Any:
        node = self.open_elements.pop()
        self._maybe_mark_end_tag(node)
        return node

    def _in_scope(self, name: str) -> bool:
        return self._has_element_in_scope(name, DEFAULT_SCOPE_TERMINATORS)

    def _close_element_by_name(self, name: str) -> None:
        # Simple element closing - pops from the named element onwards
        # Used for explicit closing (e.g., when button start tag closes existing button)
        # Caller guarantees name is on the stack via _has_in_scope check
        index = len(self.open_elements) - 1
        while index >= 0:  # pragma: no branch
            if self.open_elements[index].name == name:
                self._maybe_mark_end_tag(self.open_elements[index])
                del self.open_elements[index:]
                return
            index -= 1

    def _any_other_end_tag(self, name: str) -> None:
        # Spec: "Any other end tag" in IN_BODY mode
        # Loop through stack backwards (always terminates: html is special)
        index = len(self.open_elements) - 1
        while index >= 0:  # pragma: no branch
            node = self.open_elements[index]

            # If node's name matches the end tag name
            if node.name == name:
                # Generate implied end tags (except for this name)
                # If current node is not this node, parse error
                if index != len(self.open_elements) - 1:
                    self._parse_error("end-tag-too-early")
                self._maybe_mark_end_tag(node)
                # Pop all elements from this node onwards
                del self.open_elements[index:]
                return

            # If node is a special element, parse error and ignore the tag
            if self._is_special_element(node):
                self._parse_error("unexpected-end-tag", tag_name=name)
                return  # Ignore the end tag

            # Continue to next node (previous in stack)
            index -= 1

    def _add_missing_attributes(self, node: Any, attrs: dict[str, str]) -> None:
        if not attrs:
            return
        existing = node.attrs
        for name, value in attrs.items():
            if name not in existing:
                existing[name] = value

    def _remove_from_open_elements(self, node: Any) -> bool:
        for index, current in enumerate(self.open_elements):
            if current is node:
                self._maybe_mark_end_tag(current)
                del self.open_elements[index]
                return True
        return False

    def _is_special_element(self, node: Any) -> bool:
        if node.namespace not in {None, "html"}:
            return False
        return node.name in SPECIAL_ELEMENTS

    def _find_active_formatting_index(self, name: str) -> int | None:
        for index in range(len(self.active_formatting) - 1, -1, -1):
            entry = self.active_formatting[index]
            if entry is FORMAT_MARKER:
                break
            if entry["name"] == name:
                return index
        return None

    def _find_active_formatting_index_by_node(self, node: Any) -> int | None:
        for index in range(len(self.active_formatting) - 1, -1, -1):
            entry = self.active_formatting[index]
            if entry is not FORMAT_MARKER and entry["node"] is node:
                return index
        return None

    def _clone_attributes(self, attrs: dict[str, str | None]) -> dict[str, str | None]:
        return attrs.copy() if attrs else {}

    def _attrs_signature(self, attrs: dict[str, str | None]) -> tuple[tuple[str, str], ...]:
        if not attrs:
            return ()
        items: list[tuple[str, str]] = []
        for name, value in attrs.items():
            items.append((name, value or ""))
        items.sort()
        return tuple(items)

    def _find_active_formatting_duplicate(self, name: str, attrs: dict[str, str | None]) -> int | None:
        signature = self._attrs_signature(attrs)
        matches: list[int] = []
        for index, entry in enumerate(self.active_formatting):
            if entry is FORMAT_MARKER:
                matches.clear()
                continue
            existing_signature = entry["signature"]
            if entry["name"] == name and existing_signature == signature:
                matches.append(index)
        if len(matches) >= 3:
            return matches[0]
        return None

    def _has_active_formatting_entry(self, name: str) -> bool:
        for index in range(len(self.active_formatting) - 1, -1, -1):
            entry = self.active_formatting[index]
            if entry is FORMAT_MARKER:
                break
            if entry["name"] == name:
                return True
        return False

    def _remove_last_active_formatting_by_name(self, name: str) -> None:
        for index in range(len(self.active_formatting) - 1, -1, -1):
            entry = self.active_formatting[index]
            if entry is FORMAT_MARKER:
                break
            if entry["name"] == name:
                del self.active_formatting[index]
                return

    def _remove_last_open_element_by_name(self, name: str) -> None:
        for index in range(len(self.open_elements) - 1, -1, -1):
            if self.open_elements[index].name == name:
                self._maybe_mark_end_tag(self.open_elements[index])
                del self.open_elements[index]
                return

    def _append_active_formatting_entry(self, name: str, attrs: dict[str, str | None], node: Any) -> None:
        entry_attrs = self._clone_attributes(attrs)
        signature = self._attrs_signature(entry_attrs)
        self.active_formatting.append(
            {
                "name": name,
                "attrs": entry_attrs,
                "node": node,
                "signature": signature,
            },
        )

    def _clear_active_formatting_up_to_marker(self) -> None:
        while self.active_formatting:
            entry = self.active_formatting.pop()
            if entry is FORMAT_MARKER:
                break

    def _push_formatting_marker(self) -> None:
        self.active_formatting.append(FORMAT_MARKER)

    def _remove_formatting_entry(self, index: int) -> None:
        assert 0 <= index < len(self.active_formatting), f"Invalid index: {index}"
        del self.active_formatting[index]

    def _reconstruct_active_formatting_elements(self) -> None:
        if not self.active_formatting:
            return
        last_entry = self.active_formatting[-1]
        if last_entry is FORMAT_MARKER or last_entry["node"] in self.open_elements:
            return

        index = len(self.active_formatting) - 1
        while True:
            index -= 1
            if index < 0:
                break
            entry = self.active_formatting[index]
            if entry is FORMAT_MARKER or entry["node"] in self.open_elements:
                index += 1
                break
        if index < 0:
            index = 0
        while index < len(self.active_formatting):
            entry = self.active_formatting[index]
            tag = Tag(Tag.START, entry["name"], self._clone_attributes(entry["attrs"]), False)
            new_node = self._insert_element(tag, push=True)
            if self.tokenizer is not None and self.tokenizer.track_node_locations:
                new_node._origin_pos = entry["node"].origin_offset
                new_node._origin_line = entry["node"].origin_line
                new_node._origin_col = entry["node"].origin_col
            entry["node"] = new_node
            index += 1

    def _insert_node_at(self, parent: Any, index: int, node: Any) -> None:
        children = parent.children
        if index is not None and index < len(children):
            children.insert(index, node)
        else:
            children.append(node)
        node.parent = parent

    def _find_last_on_stack(self, name: str) -> Any | None:
        for node in reversed(self.open_elements):
            if node.name == name:
                return node
        return None

    def _clear_stack_until(self, names: set[str]) -> None:
        # All callers include "html" in names, so this always terminates via break
        while self.open_elements:
            node = self.open_elements[-1]
            if node.name in names and node.namespace in {None, "html"}:
                break
            self._pop_current()

    def _generate_implied_end_tags(self, exclude: str | None = None) -> None:
        # Always terminates: html is not in IMPLIED_END_TAGS
        while self.open_elements:  # pragma: no branch
            node = self.open_elements[-1]
            if node.name in IMPLIED_END_TAGS and node.name != exclude:
                self._pop_current()
                continue
            break

    def _has_in_table_scope(self, name: str) -> bool:
        return self._has_element_in_scope(name, TABLE_SCOPE_TERMINATORS, check_integration_points=False)

    def _close_table_cell(self) -> bool:
        if self._has_in_table_scope("td"):
            self._end_table_cell("td")
            return True
        if self._has_in_table_scope("th"):
            self._end_table_cell("th")
            return True
        return False

    def _end_table_cell(self, name: str) -> None:
        self._generate_implied_end_tags(name)
        while self.open_elements:
            node = self._pop_current()
            if node.name == name and node.namespace in {None, "html"}:
                break
        self._clear_active_formatting_up_to_marker()
        self.mode = InsertionMode.IN_ROW

    def _flush_pending_table_text(self) -> None:
        data = "".join(self.pending_table_text)
        self.pending_table_text.clear()
        if not data:  # pragma: no cover
            return
        if is_all_whitespace(data):
            self._append_text(data)
            return

        if self.pending_table_text_should_error:
            # html5lib reports one foster-parenting error per non-whitespace character.
            for ch in data:
                if ch not in " \t\n\r\f":
                    self._parse_error("foster-parenting-character")
        self.pending_table_text_should_error = False

        previous = self.insert_from_table
        self.insert_from_table = True
        try:
            self._reconstruct_active_formatting_elements()
            self._append_text(data)
        finally:
            self.insert_from_table = previous

    def _close_table_element(self) -> bool:
        if not self._has_in_table_scope("table"):
            self._parse_error("unexpected-end-tag", tag_name="table")
            return False
        self._generate_implied_end_tags()
        # Table verified in scope above
        while self.open_elements:  # pragma: no branch
            node = self._pop_current()
            if node.name == "table":
                break
        self._reset_insertion_mode()
        return True

    def _reset_insertion_mode(self) -> None:
        # Walk stack backwards - html element always terminates
        idx = len(self.open_elements) - 1
        while idx >= 0:
            node = self.open_elements[idx]
            name = node.name
            if name == "select":
                self.mode = InsertionMode.IN_SELECT
                return
            if name == "td" or name == "th":
                self.mode = InsertionMode.IN_CELL
                return
            if name == "tr":
                self.mode = InsertionMode.IN_ROW
                return
            if name in {"tbody", "tfoot", "thead"}:
                self.mode = InsertionMode.IN_TABLE_BODY
                return
            if name == "caption":
                self.mode = InsertionMode.IN_CAPTION
                return
            if name == "table":
                self.mode = InsertionMode.IN_TABLE
                return
            if name == "template":
                # Return the last template mode from the stack
                if self.template_modes:
                    self.mode = self.template_modes[-1]
                    return
            if name == "head":
                # If we're resetting and head is on stack, stay in IN_HEAD
                self.mode = InsertionMode.IN_HEAD
                return
            if name == "html":
                self.mode = InsertionMode.IN_BODY
                return
            idx -= 1
        # Empty stack fallback
        self.mode = InsertionMode.IN_BODY

    def _should_foster_parenting(self, target: Any, *, for_tag: str | None = None, is_text: bool = False) -> bool:
        if not self.insert_from_table:
            return False
        if target.name not in TABLE_FOSTER_TARGETS:
            return False
        if is_text:
            return True
        if for_tag in TABLE_ALLOWED_CHILDREN:
            return False
        return True

    def _lower_ascii(self, value: str) -> str:
        return value.lower() if value else ""

    def _adjust_svg_tag_name(self, name: str) -> str:
        lowered = self._lower_ascii(name)
        return SVG_TAG_NAME_ADJUSTMENTS.get(lowered, name)

    def _prepare_foreign_attributes(self, namespace: str, attrs: dict[str, str | None]) -> dict[str, str | None]:
        if not attrs:
            return {}
        adjusted: dict[str, str | None] = {}
        for name, value in attrs.items():
            lower_name = self._lower_ascii(name)
            if namespace == "math" and lower_name in MATHML_ATTRIBUTE_ADJUSTMENTS:
                name = MATHML_ATTRIBUTE_ADJUSTMENTS[lower_name]
                lower_name = self._lower_ascii(name)
            elif namespace == "svg" and lower_name in SVG_ATTRIBUTE_ADJUSTMENTS:
                name = SVG_ATTRIBUTE_ADJUSTMENTS[lower_name]
                lower_name = self._lower_ascii(name)

            foreign_adjustment = FOREIGN_ATTRIBUTE_ADJUSTMENTS.get(lower_name)
            if foreign_adjustment is not None:
                prefix, local, _ = foreign_adjustment
                name = f"{prefix}:{local}"

            # Tokenizer deduplicates attributes, so name collision impossible here
            adjusted[name] = value
        return adjusted

    def _node_attribute_value(self, node: Any, name: str) -> str | None:
        target = self._lower_ascii(name)
        for attr_name, attr_value in node.attrs.items():
            if self._lower_ascii(attr_name) == target:
                return attr_value or ""
        return None

    def _is_html_integration_point(self, node: Any) -> bool:
        # annotation-xml is an HTML integration point only with specific encoding values
        if node.namespace == "math" and node.name == "annotation-xml":
            encoding = self._node_attribute_value(node, "encoding")
            if encoding:
                enc_lower = encoding.lower()
                if enc_lower in {"text/html", "application/xhtml+xml"}:
                    return True
            return False  # annotation-xml without proper encoding is NOT an integration point
        # SVG foreignObject, desc, and title are always HTML integration points
        return (node.namespace, node.name) in HTML_INTEGRATION_POINT_SET

    def _is_mathml_text_integration_point(self, node: Any) -> bool:
        if node.namespace != "math":
            return False
        return (node.namespace, node.name) in MATHML_TEXT_INTEGRATION_POINT_SET

    def _adjusted_current_node(self) -> Any:
        return self.open_elements[-1]

    def _should_use_foreign_content(self, token: AnyToken) -> bool:
        current = self._adjusted_current_node()
        # HTML namespace elements don't use foreign content rules
        # (unreachable in practice as foreign content mode only entered for foreign elements)
        if current.namespace in {None, "html"}:
            return False  # pragma: no cover

        if isinstance(token, EOFToken):
            return False

        if self._is_mathml_text_integration_point(current):
            if isinstance(token, CharacterTokens):
                return False
            if isinstance(token, Tag) and token.kind == Tag.START:
                name_lower = self._lower_ascii(token.name)
                if name_lower not in {"mglyph", "malignmark"}:
                    return False

        if current.namespace == "math" and current.name == "annotation-xml":
            if isinstance(token, Tag) and token.kind == Tag.START:
                if self._lower_ascii(token.name) == "svg":
                    return False

        if self._is_html_integration_point(current):
            if isinstance(token, CharacterTokens):
                return False
            if isinstance(token, Tag) and token.kind == Tag.START:
                return False

        return True

    def _foreign_breakout_font(self, tag: Any) -> bool:
        for name in tag.attrs.keys():
            if self._lower_ascii(name) in {"color", "face", "size"}:
                return True
        return False

    def _pop_until_html_or_integration_point(self) -> None:
        # Always terminates: html element has html namespace
        while self.open_elements:  # pragma: no branch
            node = self.open_elements[-1]
            if node.namespace in {None, "html"}:
                return
            if self._is_html_integration_point(node):
                return
            if self.fragment_context_element is not None and node is self.fragment_context_element:
                return
            self._pop_current()

    def _process_foreign_content(self, token: AnyToken) -> Any | None:
        current = self._adjusted_current_node()

        if isinstance(token, CharacterTokens):
            raw = token.data or ""
            cleaned = []
            has_non_null_non_ws = False
            for ch in raw:
                if ch == "\x00":
                    self._parse_error("invalid-codepoint-in-foreign-content")
                    cleaned.append("\ufffd")
                    continue
                cleaned.append(ch)
                if ch not in "\t\n\f\r ":
                    has_non_null_non_ws = True
            data = "".join(cleaned)
            if has_non_null_non_ws:
                self.frameset_ok = False
            self._append_text(data)
            return None

        if isinstance(token, CommentToken):
            self._append_comment(token.data)
            return None

        # Foreign content only receives CharacterTokens, CommentToken, or Tag (not EOF)
        assert isinstance(token, Tag), f"Unexpected token type in foreign content: {type(token)}"
        name_lower = self._lower_ascii(token.name)
        if token.kind == Tag.START:
            if name_lower in FOREIGN_BREAKOUT_ELEMENTS or (
                name_lower == "font" and self._foreign_breakout_font(token)
            ):
                self._parse_error("unexpected-html-element-in-foreign-content")
                self._pop_until_html_or_integration_point()
                self._reset_insertion_mode()
                return ("reprocess", self.mode, token, True)

            namespace = current.namespace
            adjusted_name = token.name
            if namespace == "svg":
                adjusted_name = self._adjust_svg_tag_name(token.name)
            attrs = self._prepare_foreign_attributes(namespace, token.attrs)
            new_tag = Tag(Tag.START, adjusted_name, attrs, token.self_closing)
            # For foreign elements, honor the self-closing flag
            self._insert_element(new_tag, push=not token.self_closing, namespace=namespace)
            return None

        # Only START and END tag kinds exist, and START returns above
        assert token.kind == Tag.END, f"Unexpected tag kind: {token.kind}"
        name_lower = self._lower_ascii(token.name)

        # Special case: </br> and </p> end tags trigger breakout from foreign content
        if name_lower in {"br", "p"}:
            self._parse_error("unexpected-html-element-in-foreign-content")
            self._pop_until_html_or_integration_point()
            self._reset_insertion_mode()
            return ("reprocess", self.mode, token, True)

        # Process foreign end tag per spec: walk stack backwards looking for match
        idx = len(self.open_elements) - 1
        first = True
        while idx >= 0:
            node = self.open_elements[idx]
            is_html = node.namespace in {None, "html"}
            name_eq = self._lower_ascii(node.name) == name_lower

            # Check if this node matches the end tag (case-insensitive)
            if name_eq:
                if self.fragment_context_element is not None and node is self.fragment_context_element:
                    self._parse_error("unexpected-end-tag-in-fragment-context")
                    return None
                # If matched element is HTML namespace, break out to HTML mode
                if is_html:
                    return ("reprocess", self.mode, token, True)
                # Otherwise it's a foreign element - pop everything from this point up
                self._maybe_mark_end_tag(node)
                del self.open_elements[idx:]
                return None

            # Per HTML5 spec: if first node doesn't match, it's a parse error
            if first:
                self._parse_error("unexpected-end-tag", tag_name=token.name)
                first = False

            # If we hit an HTML element that doesn't match, process in secondary mode
            if is_html:
                return ("reprocess", self.mode, token, True)

            idx -= 1
        # Stack exhausted without finding match - ignore tag (defensive, html always terminates)
        return None  # pragma: no cover

    def _appropriate_insertion_location(
        self, override_target: Any | None = None, *, foster_parenting: bool = False
    ) -> tuple[Any, int]:
        if override_target is not None:
            target = override_target
        else:
            target = self._current_node_or_html()

        if foster_parenting and target.name in {"table", "tbody", "tfoot", "thead", "tr"}:
            last_template = self._find_last_on_stack("template")
            last_table = self._find_last_on_stack("table")
            if last_template is not None and (
                last_table is None or self.open_elements.index(last_template) > self.open_elements.index(last_table)
            ):
                return last_template.template_content, len(last_template.template_content.children)
            # No table on stack - fall back to inserting in target
            if last_table is None:
                return target, len(target.children)
            parent = last_table.parent
            # Table has no parent (e.g., detached) - fall back to target
            if parent is None:  # pragma: no cover
                children = target.children
                return target, len(children) if children is not None else 0
            assert parent.children is not None
            position = parent.children.index(last_table)
            return parent, position

        # If target is a template element, insert into its content document fragment
        if type(target) is Template and target.template_content:
            children = target.template_content.children
            return target.template_content, len(children) if children is not None else 0

        target_children = target.children
        return target, len(target_children) if target_children is not None else 0

    def _populate_selectedcontent(self, root: Any) -> None:
        """Populate selectedcontent elements with content from selected option.

        Per HTML5 spec: selectedcontent mirrors the content of the selected option,
        or the first option if none is selected.
        """
        # Find all select elements
        selects: list[Any] = []
        self._find_elements(root, "select", selects)

        for select in selects:
            # Find selectedcontent element in this select
            selectedcontent = self._find_element(select, "selectedcontent")
            if not selectedcontent:
                continue

            # Find all option elements
            options: list[Any] = []
            self._find_elements(select, "option", options)

            # Find selected option or use first one
            selected_option = None
            for opt in options:
                if opt.attrs:
                    for attr_name in opt.attrs.keys():
                        if attr_name == "selected":
                            selected_option = opt
                            break
                if selected_option:
                    break

            if not selected_option:
                selected_option = options[0]

            # Clone content from selected option to selectedcontent
            self._clone_children(selected_option, selectedcontent)

    def _find_elements(self, node: Any, name: str, result: list[Any]) -> None:
        """Recursively find all elements with given name."""
        if node.name == name:
            result.append(node)

        if node.has_child_nodes():
            for child in node.children:
                self._find_elements(child, name, result)

    def _find_element(self, node: Any, name: str) -> Any | None:
        """Find first element with given name."""
        if node.name == name:
            return node

        if node.has_child_nodes():
            for child in node.children:
                result = self._find_element(child, name)
                if result:
                    return result
        return None

    def _clone_children(self, source: Any, target: Any) -> None:
        """Deep clone all children from source to target."""
        for child in source.children:
            target.append_child(child.clone_node(deep=True))

    def _has_in_scope(self, name: str) -> bool:
        return self._has_element_in_scope(name, DEFAULT_SCOPE_TERMINATORS)

    def _has_in_list_item_scope(self, name: str) -> bool:
        return self._has_element_in_scope(name, LIST_ITEM_SCOPE_TERMINATORS)

    def _has_in_definition_scope(self, name: str) -> bool:
        return self._has_element_in_scope(name, DEFINITION_SCOPE_TERMINATORS)

    def _has_any_in_scope(self, names: set[str]) -> bool:
        # Always terminates: html is in DEFAULT_SCOPE_TERMINATORS
        terminators = DEFAULT_SCOPE_TERMINATORS
        idx = len(self.open_elements) - 1
        while idx >= 0:
            node = self.open_elements[idx]
            if node.name in names:
                return True
            if node.namespace in {None, "html"} and node.name in terminators:
                return False
            idx -= 1
        return False  # pragma: no cover - html always terminates

    def process_characters(self, data: str) -> Any:
        """Optimized path for character tokens."""
        # Check for foreign content first
        current_node = self.open_elements[-1] if self.open_elements else None
        is_html_namespace = current_node is None or current_node.namespace in {None, "html"}

        if not is_html_namespace:
            return self.process_token(CharacterTokens(data))

        if self.mode == InsertionMode.IN_BODY:
            if not data:
                return TokenSinkResult.Continue
            if "\x00" in data:
                data = data.replace("\x00", "")
                if not data:
                    return TokenSinkResult.Continue

            if is_all_whitespace(data):
                if self.active_formatting:
                    self._reconstruct_active_formatting_elements()
                self._append_text(data)
                return TokenSinkResult.Continue

            if self.active_formatting:
                self._reconstruct_active_formatting_elements()
            self.frameset_ok = False
            self._append_text(data)
            return TokenSinkResult.Continue

        return self.process_token(CharacterTokens(data))
