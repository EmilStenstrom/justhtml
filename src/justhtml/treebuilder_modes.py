# ruff: noqa: S101, RUF012
# mypy: disable-error-code="attr-defined, has-type, var-annotated, assignment"

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from .constants import (
    FORMAT_MARKER,
    FORMATTING_ELEMENTS,
    HEADING_ELEMENTS,
)
from .node import Comment, Node, Template
from .tokens import AnyToken, CharacterTokens, CommentToken, DoctypeToken, EOFToken, Tag, TokenSinkResult
from .treebuilder_utils import (
    InsertionMode,
    doctype_error_and_quirks,
    is_all_whitespace,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    ModeResultTuple = tuple[str, InsertionMode, AnyToken] | tuple[str, InsertionMode, AnyToken, bool]
    "Result is (instruction, mode, token) or (instruction, mode, token, force_html)"

_CLEAR_STACK_UNTIL_TABLE_TEMPLATE_HTML = frozenset(("table", "template", "html"))
_CLEAR_STACK_UNTIL_TBODY_TFOOT_THEAD_TEMPLATE_HTML = frozenset(("tbody", "tfoot", "thead", "template", "html"))
_CLEAR_STACK_UNTIL_TR_TEMPLATE_HTML = frozenset(("tr", "template", "html"))


class TreeBuilderModesMixin:
    def _handle_doctype(self, token: DoctypeToken) -> Literal[0]:
        if self.mode != InsertionMode.INITIAL:
            self._parse_error("unexpected-doctype")
            return TokenSinkResult.Continue

        doctype = token.doctype
        parse_error, quirks_mode = doctype_error_and_quirks(doctype, self.iframe_srcdoc)

        node = Node("!doctype", data=doctype)
        self.document.append_child(node)

        if parse_error:
            self._parse_error("unknown-doctype")

        self._set_quirks_mode(quirks_mode)
        self.mode = InsertionMode.BEFORE_HTML
        return TokenSinkResult.Continue

    def _mode_initial(self, token: Any) -> ModeResultTuple | None:
        if isinstance(token, CharacterTokens):
            if is_all_whitespace(token.data):
                return None
            self._parse_error("expected-doctype-but-got-chars")
            self._set_quirks_mode("quirks")
            return ("reprocess", InsertionMode.BEFORE_HTML, token)
        if isinstance(token, CommentToken):
            self._append_comment_to_document(token.data)
            return None
        if isinstance(token, EOFToken):
            self._parse_error("expected-doctype-but-got-eof")
            self._set_quirks_mode("quirks")
            self.mode = InsertionMode.BEFORE_HTML
            return ("reprocess", InsertionMode.BEFORE_HTML, token)
        # Only Tags remain - no DOCTYPE seen, so quirks mode
        if token.kind == Tag.START:
            self._parse_error("expected-doctype-but-got-start-tag", tag_name=token.name)
        else:
            self._parse_error("expected-doctype-but-got-end-tag", tag_name=token.name)
        self._set_quirks_mode("quirks")
        return ("reprocess", InsertionMode.BEFORE_HTML, token)

    def _mode_before_html(self, token: AnyToken) -> ModeResultTuple | None:
        if isinstance(token, CharacterTokens) and is_all_whitespace(token.data):
            return None
        if isinstance(token, CommentToken):
            self._append_comment_to_document(token.data)
            return None
        if isinstance(token, Tag):
            if token.kind == Tag.START and token.name == "html":
                self._create_root(token.attrs)
                self.mode = InsertionMode.BEFORE_HEAD
                return None
            if token.kind == Tag.END and token.name in {"head", "body", "html", "br"}:
                self._create_root({})
                self.mode = InsertionMode.BEFORE_HEAD
                return ("reprocess", InsertionMode.BEFORE_HEAD, token)
            if token.kind == Tag.END:
                # Ignore other end tags
                self._parse_error("unexpected-end-tag-before-html", tag_name=token.name)
                return None
        if isinstance(token, EOFToken):
            self._create_root({})
            self.mode = InsertionMode.BEFORE_HEAD
            return ("reprocess", InsertionMode.BEFORE_HEAD, token)

        if isinstance(token, CharacterTokens):
            stripped = token.data.lstrip("\t\n\f\r ")
            if len(stripped) != len(token.data):
                token = CharacterTokens(stripped)

        self._create_root({})
        self.mode = InsertionMode.BEFORE_HEAD
        return ("reprocess", InsertionMode.BEFORE_HEAD, token)

    def _mode_before_head(self, token: AnyToken) -> ModeResultTuple | None:
        if isinstance(token, CharacterTokens):
            data = token.data or ""
            if "\x00" in data:
                self._parse_error("invalid-codepoint-before-head")
                data = data.replace("\x00", "")
                if not data:
                    return None
            if is_all_whitespace(data):
                return None
            token = CharacterTokens(data)
        if isinstance(token, CommentToken):
            self._append_comment(token.data)
            return None
        if isinstance(token, Tag):
            if token.kind == Tag.START and token.name == "html":
                # Duplicate html tag - add attributes to existing html element
                # Note: open_elements[0] is always html at this point (created in BEFORE_HTML mode)
                html = self.open_elements[0]
                self._add_missing_attributes(html, token.attrs)
                return None
            if token.kind == Tag.START and token.name == "head":
                head = self._insert_element(token, push=True)
                self.head_element = head
                self.mode = InsertionMode.IN_HEAD
                return None
            if token.kind == Tag.END and token.name in {"head", "body", "html", "br"}:
                self.head_element = self._insert_phantom("head")
                self.mode = InsertionMode.IN_HEAD
                return ("reprocess", InsertionMode.IN_HEAD, token)
            if token.kind == Tag.END:
                # Ignore other end tags
                self._parse_error("unexpected-end-tag-before-head", tag_name=token.name)
                return None
        if isinstance(token, EOFToken):
            self.head_element = self._insert_phantom("head")
            self.mode = InsertionMode.IN_HEAD
            return ("reprocess", InsertionMode.IN_HEAD, token)

        self.head_element = self._insert_phantom("head")
        self.mode = InsertionMode.IN_HEAD
        return ("reprocess", InsertionMode.IN_HEAD, token)

    def _mode_in_head(self, token: AnyToken) -> ModeResultTuple | None:
        if isinstance(token, CharacterTokens):
            if is_all_whitespace(token.data):
                self._append_text(token.data)
                return None
            data = token.data or ""
            i = 0
            while i < len(data) and data[i] in "\t\n\f\r ":
                i += 1
            leading_ws = data[:i]
            remaining = data[i:]
            if leading_ws:
                current = self.open_elements[-1] if self.open_elements else None
                if current is not None and current.has_child_nodes():
                    self._append_text(leading_ws)
            self._pop_current()
            self.mode = InsertionMode.AFTER_HEAD
            return ("reprocess", InsertionMode.AFTER_HEAD, CharacterTokens(remaining))
        if isinstance(token, CommentToken):
            self._append_comment(token.data)
            return None
        if isinstance(token, Tag):
            if token.kind == Tag.START and token.name == "html":
                # Pop head and transition to AFTER_HEAD, then reprocess
                self._pop_current()
                self.mode = InsertionMode.AFTER_HEAD
                return ("reprocess", InsertionMode.AFTER_HEAD, token)
            if token.kind == Tag.START and token.name in {"base", "basefont", "bgsound", "link", "meta"}:
                self._insert_element(token, push=False)
                return None
            if token.kind == Tag.START and token.name == "template":
                self._insert_element(token, push=True)
                self._push_formatting_marker()
                self.frameset_ok = False
                self.mode = InsertionMode.IN_TEMPLATE
                self.template_modes.append(InsertionMode.IN_TEMPLATE)
                return None
            if token.kind == Tag.END and token.name == "template":
                # Check if template is on the stack (don't use scope check as table blocks it)
                has_template = any(node.name == "template" for node in self.open_elements)
                if not has_template:
                    return None
                self._generate_implied_end_tags()
                self._pop_until_inclusive("template")
                self._clear_active_formatting_up_to_marker()
                # template_modes always non-empty here since we passed has_template check
                self.template_modes.pop()
                self._reset_insertion_mode()
                return None
            if token.kind == Tag.START and token.name in {"title", "style", "script", "noframes"}:
                self._insert_element(token, push=True)
                self.original_mode = self.mode
                self.mode = InsertionMode.TEXT
                return None
            if token.kind == Tag.START and token.name == "noscript":
                if self.scripting_enabled:
                    self._insert_element(token, push=True)
                    self.original_mode = self.mode
                    self.mode = InsertionMode.TEXT
                else:
                    # Scripting is disabled: parse noscript content as HTML
                    self._insert_element(token, push=True)
                    self.mode = InsertionMode.IN_HEAD_NOSCRIPT
                return None
            if token.kind == Tag.END and token.name == "head":
                self._pop_current()
                self.mode = InsertionMode.AFTER_HEAD
                return None
            if token.kind == Tag.END and token.name in {"body", "html", "br"}:
                self._pop_current()
                self.mode = InsertionMode.AFTER_HEAD
                return ("reprocess", InsertionMode.AFTER_HEAD, token)
        if isinstance(token, EOFToken):
            self._pop_current()
            self.mode = InsertionMode.AFTER_HEAD
            return ("reprocess", InsertionMode.AFTER_HEAD, token)

        self._pop_current()
        self.mode = InsertionMode.AFTER_HEAD
        return ("reprocess", InsertionMode.AFTER_HEAD, token)

    def _mode_in_head_noscript(self, token: AnyToken) -> ModeResultTuple | None:
        """Handle tokens in 'in head noscript' insertion mode (scripting disabled)."""
        if isinstance(token, CharacterTokens):
            data = token.data or ""
            # Whitespace: process using in head rules
            if is_all_whitespace(data):
                return self._mode_in_head(token)
            # Non-whitespace: parse error, pop noscript, reprocess in head
            self._parse_error("unexpected-start-tag", tag_name="text")
            self._pop_current()  # Pop noscript
            self.mode = InsertionMode.IN_HEAD
            return ("reprocess", InsertionMode.IN_HEAD, token)
        if isinstance(token, CommentToken):
            return self._mode_in_head(token)
        if isinstance(token, Tag):
            if token.kind == Tag.START:
                if token.name == "html":
                    return self._mode_in_body(token)
                if token.name in {"basefont", "bgsound", "link", "meta", "noframes", "style"}:
                    return self._mode_in_head(token)
                if token.name in {"head", "noscript"}:
                    self._parse_error("unexpected-start-tag", tag_name=token.name)
                    return None  # Ignore
                # Any other start tag: parse error, pop noscript, reprocess in head
                self._parse_error("unexpected-start-tag", tag_name=token.name)
                self._pop_current()  # Pop noscript
                self.mode = InsertionMode.IN_HEAD
                return ("reprocess", InsertionMode.IN_HEAD, token)
            # token.kind == Tag.END
            if token.name == "noscript":
                self._pop_current()  # Pop noscript
                self.mode = InsertionMode.IN_HEAD
                return None
            if token.name == "br":
                self._parse_error("unexpected-end-tag", tag_name=token.name)
                self._pop_current()  # Pop noscript
                self.mode = InsertionMode.IN_HEAD
                return ("reprocess", InsertionMode.IN_HEAD, token)
            # Any other end tag: parse error, ignore
            self._parse_error("unexpected-end-tag", tag_name=token.name)
            return None
        if isinstance(token, EOFToken):
            self._parse_error("expected-closing-tag-but-got-eof", tag_name="noscript")
            self._pop_current()  # Pop noscript
            self.mode = InsertionMode.IN_HEAD
            return ("reprocess", InsertionMode.IN_HEAD, token)
        # All token types are handled above - CharacterTokens, CommentToken, Tag, EOFToken
        return None  # pragma: no cover

    def _mode_after_head(self, token: AnyToken) -> ModeResultTuple | None:
        if isinstance(token, CharacterTokens):
            data = token.data or ""
            if "\x00" in data:
                data = data.replace("\x00", "")
            if not data or is_all_whitespace(data):
                if data:
                    self._append_text(data)
                return None
            self._insert_body_if_missing()
            return ("reprocess", InsertionMode.IN_BODY, CharacterTokens(data))
        if isinstance(token, CommentToken):
            self._append_comment(token.data)
            return None
        if isinstance(token, Tag):
            if token.kind == Tag.START and token.name == "html":
                self._insert_body_if_missing()
                return ("reprocess", InsertionMode.IN_BODY, token)
            if token.kind == Tag.START and token.name == "body":
                self._insert_element(token, push=True)
                self.mode = InsertionMode.IN_BODY
                self.frameset_ok = False
                return None
            if token.kind == Tag.START and token.name == "frameset":
                self._insert_element(token, push=True)
                self.mode = InsertionMode.IN_FRAMESET
                return None
            # Special handling: input type="hidden" doesn't create body or affect frameset_ok
            if token.kind == Tag.START and token.name == "input":
                input_type = None
                for name, value in token.attrs.items():
                    if name == "type":
                        input_type = (value or "").lower()
                        break
                if input_type == "hidden":
                    # Parse error but ignore - don't create body, don't insert element
                    self._parse_error("unexpected-hidden-input-after-head")
                    return None
                # Non-hidden input creates body
                self._insert_body_if_missing()
                return ("reprocess", InsertionMode.IN_BODY, token)
            if token.kind == Tag.START and token.name in {
                "base",
                "basefont",
                "bgsound",
                "link",
                "meta",
                "title",
                "style",
                "script",
                "noscript",
            }:
                self.open_elements.append(self.head_element)
                result = self._mode_in_head(token)
                # Remove the head element from wherever it is in the stack
                # (it might not be at the end if we inserted other elements like <title>)
                self.open_elements.remove(self.head_element)
                return result
            if token.kind == Tag.START and token.name == "template":
                # Template in after-head needs special handling:
                # Process in IN_HEAD mode, which will switch to IN_TEMPLATE
                # Don't remove head from stack - let normal processing continue
                self.open_elements.append(self.head_element)
                self.mode = InsertionMode.IN_HEAD
                return ("reprocess", InsertionMode.IN_HEAD, token)
            if token.kind == Tag.END and token.name == "template":
                has_template = any(node.name == "template" for node in self.open_elements)
                if not has_template:
                    self._parse_error("unexpected-end-tag", tag_name=token.name)
                    return None
                return self._mode_in_head(token)
            if token.kind == Tag.END and token.name == "body":
                self._insert_body_if_missing()
                return ("reprocess", InsertionMode.IN_BODY, token)
            if token.kind == Tag.END and token.name in {"html", "br"}:
                self._insert_body_if_missing()
                return ("reprocess", InsertionMode.IN_BODY, token)
            if token.kind == Tag.END:
                # Ignore other end tags
                self._parse_error("unexpected-end-tag-after-head", tag_name=token.name)
                return None
        if isinstance(token, EOFToken):
            self._insert_body_if_missing()
            self.mode = InsertionMode.IN_BODY
            return ("reprocess", InsertionMode.IN_BODY, token)

        self._insert_body_if_missing()
        return ("reprocess", InsertionMode.IN_BODY, token)

    def _mode_text(self, token: AnyToken) -> ModeResultTuple | None:
        if isinstance(token, CharacterTokens):
            self._append_text(token.data)
            return None
        if isinstance(token, EOFToken):
            # Get the tag name of the unclosed element
            tag_name = self.open_elements[-1].name if self.open_elements else None
            self._parse_error("expected-named-closing-tag-but-got-eof", tag_name=tag_name)
            self._pop_current()
            self.mode = self.original_mode or InsertionMode.IN_BODY
            return ("reprocess", self.mode, token)
        # End tag
        self._pop_current()
        self.mode = self.original_mode or InsertionMode.IN_BODY
        return None

    def _mode_in_body(self, token: Any) -> ModeResultTuple | None:
        handler = self._BODY_TOKEN_HANDLERS.get(type(token))
        return handler(self, token) if handler else None

    def _handle_characters_in_body(self, token: CharacterTokens) -> None:
        data = token.data or ""
        if "\x00" in data:
            self._parse_error("invalid-codepoint")
            data = data.replace("\x00", "")
        if is_all_whitespace(data):
            self._reconstruct_active_formatting_elements()
            self._append_text(data)
            return
        self._reconstruct_active_formatting_elements()
        self.frameset_ok = False
        self._append_text(data)
        return

    def _handle_comment_in_body(self, token: CommentToken) -> None:
        self._append_comment(token.data)
        return

    def _handle_tag_in_body(self, token: Tag) -> ModeResultTuple | None:
        if token.kind == Tag.START:
            handler = self._BODY_START_HANDLERS.get(token.name)
            if handler:
                return handler(self, token)
            return self._handle_body_start_default(token)
        name = token.name

        # Special case: </br> end tag is treated as <br> start tag
        if name == "br":
            self._parse_error("unexpected-end-tag", tag_name=name, token=token)
            br_tag = Tag(Tag.START, "br", {}, False)
            return self._mode_in_body(br_tag)

        if name in FORMATTING_ELEMENTS:
            self._adoption_agency(name)
            return None
        handler = self._BODY_END_HANDLERS.get(name)
        if handler:
            return handler(self, token)
        # Any other end tag
        self._any_other_end_tag(token.name)
        return None

    def _handle_eof_in_body(self, token: EOFToken) -> ModeResultTuple | None:
        # If we're in a template, handle EOF in template mode first
        if self.template_modes:
            return self._mode_in_template(token)
        # Check for unclosed elements (excluding html, body, head which are implicit)
        for node in self.open_elements:
            if node.name not in {
                "dd",
                "dt",
                "li",
                "optgroup",
                "option",
                "p",
                "rb",
                "rp",
                "rt",
                "rtc",
                "tbody",
                "td",
                "tfoot",
                "th",
                "thead",
                "tr",
                "body",
                "html",
            }:
                self._parse_error("expected-closing-tag-but-got-eof", tag_name=node.name)
                break
        self.mode = InsertionMode.AFTER_BODY
        return ("reprocess", InsertionMode.AFTER_BODY, token)

    # ---------------------
    # Body mode start tag handlers
    # ---------------------

    def _handle_body_start_html(self, token: Tag) -> None:
        if self.template_modes:
            self._parse_error("unexpected-start-tag", tag_name=token.name)
            return
        # Per spec: parse error; merge attributes onto existing <html>.
        self._parse_error("unexpected-start-tag", tag_name=token.name)
        # In IN_BODY mode, html element is always at open_elements[0]
        if self.open_elements:  # pragma: no branch
            html = self.open_elements[0]
            self._add_missing_attributes(html, token.attrs)
        return

    def _handle_body_start_body(self, token: Tag) -> None:
        if self.template_modes:
            self._parse_error("unexpected-start-tag", tag_name=token.name)
            return
        if len(self.open_elements) > 1:
            self._parse_error("unexpected-start-tag", tag_name=token.name)
            body = self.open_elements[1] if len(self.open_elements) > 1 else None
            if body and body.name == "body":
                self._add_missing_attributes(body, token.attrs)
            self.frameset_ok = False
            return
        self.frameset_ok = False
        return

    def _handle_body_start_head(self, token: Tag) -> None:
        self._parse_error("unexpected-start-tag", tag_name=token.name)
        return

    def _handle_body_start_in_head(self, token: Tag) -> ModeResultTuple | None:
        return self._mode_in_head(token)

    def _handle_body_start_block_with_p(self, token: Tag) -> None:
        self._close_p_element()
        self._insert_element(token, push=True)
        return

    def _handle_body_start_heading(self, token: Tag) -> None:
        self._close_p_element()
        if self.open_elements and self.open_elements[-1].name in HEADING_ELEMENTS:
            self._parse_error("unexpected-start-tag", tag_name=token.name)
            self._pop_current()
        self._insert_element(token, push=True)
        self.frameset_ok = False
        return

    def _handle_body_start_pre_listing(self, token: Tag) -> None:
        self._close_p_element()
        self._insert_element(token, push=True)
        self.ignore_lf = True
        self.frameset_ok = False
        return

    def _handle_body_start_form(self, token: Tag) -> None:
        if self.form_element is not None:
            self._parse_error("unexpected-start-tag", tag_name=token.name)
            return
        self._close_p_element()
        node = self._insert_element(token, push=True)
        self.form_element = node
        self.frameset_ok = False
        return

    def _handle_body_start_button(self, token: Tag) -> None:
        if self._has_in_scope("button"):
            self._parse_error("unexpected-start-tag-implies-end-tag", tag_name=token.name)
            self._close_element_by_name("button")
        self._insert_element(token, push=True)
        self.frameset_ok = False
        return

    def _handle_body_start_paragraph(self, token: Tag) -> None:
        self._close_p_element()
        self._insert_element(token, push=True)
        return

    def _handle_body_start_math(self, token: Tag) -> None:
        self._reconstruct_active_formatting_elements()
        attrs = self._prepare_foreign_attributes("math", token.attrs)
        new_tag = Tag(Tag.START, token.name, attrs, token.self_closing)
        self._insert_element(new_tag, push=not token.self_closing, namespace="math")
        return

    def _handle_body_start_svg(self, token: Tag) -> None:
        self._reconstruct_active_formatting_elements()
        adjusted_name = self._adjust_svg_tag_name(token.name)
        attrs = self._prepare_foreign_attributes("svg", token.attrs)
        new_tag = Tag(Tag.START, adjusted_name, attrs, token.self_closing)
        self._insert_element(new_tag, push=not token.self_closing, namespace="svg")
        return

    def _handle_body_start_li(self, token: Tag) -> None:
        self.frameset_ok = False
        self._close_p_element()
        if self._has_in_list_item_scope("li"):
            self._pop_until_inclusive("li")
        self._insert_element(token, push=True)
        return

    def _handle_body_start_dd_dt(self, token: Tag) -> None:
        self.frameset_ok = False
        self._close_p_element()
        name = token.name
        if name == "dd":
            if self._has_in_definition_scope("dd"):
                self._pop_until_inclusive("dd")
            if self._has_in_definition_scope("dt"):
                self._pop_until_inclusive("dt")
        else:
            if self._has_in_definition_scope("dt"):
                self._pop_until_inclusive("dt")
            if self._has_in_definition_scope("dd"):
                self._pop_until_inclusive("dd")
        self._insert_element(token, push=True)
        return

    def _adoption_agency(self, subject: Any) -> None:
        # 1. If the current node is the subject, and it is not in the active formatting elements list...
        if self.open_elements and self.open_elements[-1].name == subject:
            if not self._has_active_formatting_entry(subject):
                self._pop_until_inclusive(subject)
                return

        # 2. Outer loop
        for _ in range(8):
            # 3. Find formatting element
            formatting_element_index = self._find_active_formatting_index(subject)
            if formatting_element_index is None:
                # html5lib reports a parse error when an end tag for a formatting
                # element triggers the adoption agency algorithm but no matching
                # active formatting entry exists.
                self._parse_error("adoption-agency-1.3")
                return

            formatting_element_entry = self.active_formatting[formatting_element_index]
            formatting_element = formatting_element_entry["node"]

            # 4. If formatting element is not in open elements
            if formatting_element not in self.open_elements:
                self._parse_error("adoption-agency-1.3")
                self._remove_formatting_entry(formatting_element_index)
                return

            # 5. If formatting element is in open elements but not in scope
            if not self._has_element_in_scope(formatting_element.name):
                self._parse_error("adoption-agency-1.3")
                return

            # 6. If formatting element is not the current node
            if formatting_element is not self.open_elements[-1]:
                self._parse_error("adoption-agency-1.3")

            # 7. Find furthest block
            furthest_block = None
            furthest_block_index = None
            formatting_element_in_open_index = self.open_elements.index(formatting_element)

            for i in range(formatting_element_in_open_index + 1, len(self.open_elements)):
                candidate = self.open_elements[i]
                if self._is_special_element(candidate):
                    furthest_block = candidate
                    furthest_block_index = i
                    break

            if furthest_block is None:
                # formatting_element is known to be on the stack
                while True:
                    popped = self._pop_current()
                    if popped is formatting_element:
                        break
                self._remove_formatting_entry(formatting_element_index)
                return
            assert furthest_block_index is not None

            # 8. Bookmark
            bookmark = formatting_element_index + 1

            # 9. Node and Last Node
            last_node = furthest_block
            node_index = furthest_block_index

            # 10. Inner loop
            inner_loop_counter = 0
            while True:
                inner_loop_counter += 1

                # 10.1 Node = element above node
                node_index -= 1
                node = self.open_elements[node_index]

                # 10.2 If node is formatting element, break
                if node is formatting_element:
                    break

                # 10.3 Find active formatting entry for node
                node_formatting_index = self._find_active_formatting_index_by_node(node)

                if inner_loop_counter > 3 and node_formatting_index is not None:
                    self._remove_formatting_entry(node_formatting_index)
                    if node_formatting_index < bookmark:
                        bookmark -= 1
                    node_formatting_index = None

                if node_formatting_index is None:
                    del self.open_elements[node_index]
                    continue

                # 10.4 Replace entry with new element
                entry = self.active_formatting[node_formatting_index]
                new_element = self._create_element(entry["name"], entry["node"].namespace, entry["attrs"])
                if self.tokenizer is not None and self.tokenizer.track_node_locations:
                    new_element._origin_pos = entry["node"].origin_offset
                    new_element._origin_line = entry["node"].origin_line
                    new_element._origin_col = entry["node"].origin_col
                entry["node"] = new_element
                self.open_elements[node_index] = new_element
                node = new_element

                # 10.5 If last node is furthest block, update bookmark
                if last_node is furthest_block:
                    bookmark = node_formatting_index + 1

                # 10.6 Reparent last_node
                if last_node.parent:
                    last_node.parent.remove_child(last_node)
                node.append_child(last_node)

                # 10.7
                last_node = node

            # 11. Insert last_node into common ancestor
            common_ancestor = self.open_elements[formatting_element_in_open_index - 1]
            if last_node.parent:
                last_node.parent.remove_child(last_node)

            if self._should_foster_parenting(common_ancestor, for_tag=last_node.name):
                parent, position = self._appropriate_insertion_location(common_ancestor, foster_parenting=True)
                self._insert_node_at(parent, position, last_node)
            else:
                if type(common_ancestor) is Template and common_ancestor.template_content:
                    common_ancestor.template_content.append_child(last_node)
                else:
                    common_ancestor.append_child(last_node)

            # 12. Create new formatting element
            entry = self.active_formatting[formatting_element_index]
            new_formatting_element = self._create_element(entry["name"], entry["node"].namespace, entry["attrs"])
            if self.tokenizer is not None and self.tokenizer.track_node_locations:
                new_formatting_element._origin_pos = entry["node"].origin_offset
                new_formatting_element._origin_line = entry["node"].origin_line
                new_formatting_element._origin_col = entry["node"].origin_col
            entry["node"] = new_formatting_element

            # 13. Move children of furthest block
            moved_children = furthest_block.children
            furthest_block.children = []
            for child in moved_children:
                new_formatting_element.append_child(child)

            furthest_block.append_child(new_formatting_element)

            # 14. Remove formatting element from active formatting and insert new at bookmark
            # Per spec, bookmark is always > formatting_element_index (starts at fmt_idx+1,
            # can only be set to higher values or decremented when entries above fmt_idx are removed)
            self._remove_formatting_entry(formatting_element_index)
            bookmark -= 1
            self.active_formatting.insert(bookmark, entry)

            # 15. Remove formatting element from open elements and insert new one
            self.open_elements.remove(formatting_element)
            furthest_block_index = self.open_elements.index(furthest_block)
            self.open_elements.insert(furthest_block_index + 1, new_formatting_element)

    def _handle_body_start_a(self, token: Tag) -> None:
        if self._has_active_formatting_entry("a"):
            self._parse_error("unexpected-start-tag-implies-end-tag", tag_name=token.name)
            self._adoption_agency("a")
            self._remove_last_active_formatting_by_name("a")
            self._remove_last_open_element_by_name("a")
        self._reconstruct_active_formatting_elements()
        node = self._insert_element(token, push=True)
        self._append_active_formatting_entry("a", token.attrs, node)
        return

    def _handle_body_start_formatting(self, token: Tag) -> None:
        name = token.name
        if name == "nobr" and self._in_scope("nobr"):
            self._adoption_agency("nobr")
            self._remove_last_active_formatting_by_name("nobr")
            self._remove_last_open_element_by_name("nobr")
        self._reconstruct_active_formatting_elements()
        duplicate_index = self._find_active_formatting_duplicate(name, token.attrs)
        if duplicate_index is not None:
            self._remove_formatting_entry(duplicate_index)
        node = self._insert_element(token, push=True)
        self._append_active_formatting_entry(name, token.attrs, node)
        return

    def _handle_body_start_applet_like(self, token: Tag) -> None:
        self._reconstruct_active_formatting_elements()
        self._insert_element(token, push=True)
        self._push_formatting_marker()
        self.frameset_ok = False
        return

    def _handle_body_start_br(self, token: Tag) -> None:
        self._close_p_element()
        self._reconstruct_active_formatting_elements()
        self._insert_element(token, push=False)
        self.frameset_ok = False
        return

    def _handle_body_start_frameset(self, token: Tag) -> None:
        if not self.frameset_ok:
            self._parse_error("unexpected-start-tag-ignored", tag_name=token.name)
            return
        # Find body element on the stack (may not exist if already in frameset)
        body_index = None
        for i, elem in enumerate(self.open_elements):
            if elem.name == "body":
                body_index = i
                break
        if body_index is None:
            # No body on stack (e.g., nested frameset after mode reset), ignore
            self._parse_error("unexpected-start-tag-ignored", tag_name=token.name)
            return
        body_elem = self.open_elements[body_index]
        body_elem.parent.remove_child(body_elem)
        self.open_elements = self.open_elements[:body_index]
        self._insert_element(token, push=True)
        self.mode = InsertionMode.IN_FRAMESET
        return

    # ---------------------
    # Body mode end tag handlers
    # ---------------------

    def _handle_body_end_body(self, token: Tag) -> None:
        if self._in_scope("body"):
            self.mode = InsertionMode.AFTER_BODY
        return

    def _handle_body_end_html(self, token: Tag) -> ModeResultTuple | None:
        if self._in_scope("body"):
            return ("reprocess", InsertionMode.AFTER_BODY, token)
        return None

    def _handle_body_end_p(self, token: Tag) -> None:
        if not self._close_p_element():
            self._parse_error("unexpected-end-tag", tag_name=token.name)
            phantom = Tag(Tag.START, "p", {}, False)
            self._insert_element(phantom, push=True)
            self._close_p_element()
        return

    def _handle_body_end_li(self, token: Tag) -> None:
        if not self._has_in_list_item_scope("li"):
            self._parse_error("unexpected-end-tag", tag_name=token.name)
            return
        self._pop_until_inclusive("li")
        return

    def _handle_body_end_dd_dt(self, token: Tag) -> None:
        name = token.name
        if not self._has_in_definition_scope(name):
            self._parse_error("unexpected-end-tag", tag_name=name)
            return
        # Callers guarantee one of these is present on the stack.
        while self.open_elements:  # pragma: no branch
            node = self._pop_current()
            if node.name in ("dd", "dt"):  # pragma: no branch
                break

    def _handle_body_end_form(self, token: Tag) -> None:
        if self.form_element is None:
            self._parse_error("unexpected-end-tag", tag_name=token.name)
            return
        removed = self._remove_from_open_elements(self.form_element)
        self.form_element = None
        if not removed:
            self._parse_error("unexpected-end-tag", tag_name=token.name)
        return

    def _handle_body_end_applet_like(self, token: Tag) -> None:
        name = token.name
        if not self._in_scope(name):
            self._parse_error("unexpected-end-tag", tag_name=name)
            return
        # Element verified in scope above
        while self.open_elements:  # pragma: no branch
            popped = self._pop_current()
            if popped.name == name:
                break
        self._clear_active_formatting_up_to_marker()
        return

    def _handle_body_end_heading(self, token: Tag) -> None:
        name = token.name
        if not self._has_any_in_scope(HEADING_ELEMENTS):
            self._parse_error("unexpected-end-tag", tag_name=name)
            return
        self._generate_implied_end_tags()
        if self.open_elements and self.open_elements[-1].name != name:
            self._parse_error("end-tag-too-early", tag_name=name)
        # Heading verified in scope by caller
        while self.open_elements:  # pragma: no branch
            popped = self._pop_current()
            if popped.name in HEADING_ELEMENTS:
                break
        return

    def _handle_body_end_block(self, token: Tag) -> None:
        name = token.name
        if not self._in_scope(name):
            self._parse_error("unexpected-end-tag", tag_name=name)
            return
        self._generate_implied_end_tags()
        if self.open_elements and self.open_elements[-1].name != name:
            self._parse_error("end-tag-too-early", tag_name=name)
        self._pop_until_inclusive(name)
        return

    def _handle_body_end_template(self, token: Tag) -> None:
        has_template = any(node.name == "template" for node in self.open_elements)
        if not has_template:
            self._parse_error("unexpected-end-tag", tag_name=token.name)
            return
        self._generate_implied_end_tags()
        self._pop_until_inclusive("template")
        self._clear_active_formatting_up_to_marker()
        # Pop template mode if available
        if self.template_modes:  # pragma: no branch
            self.template_modes.pop()
        self._reset_insertion_mode()
        return

    def _handle_body_start_structure_ignored(self, token: Tag) -> None:
        self._parse_error("unexpected-start-tag-ignored", tag_name=token.name)
        return

    def _handle_body_start_col_or_frame(self, token: Tag) -> None:
        if self.fragment_context is None:
            self._parse_error("unexpected-start-tag-ignored", tag_name=token.name)
            return
        self._insert_element(token, push=False)
        return

    def _handle_body_start_image(self, token: Tag) -> None:
        self._parse_error("image-start-tag", tag_name=token.name)
        img_token = Tag(Tag.START, "img", token.attrs, token.self_closing)
        self._reconstruct_active_formatting_elements()
        self._insert_element(img_token, push=False)
        self.frameset_ok = False
        return

    def _handle_body_start_void_with_formatting(self, token: Tag) -> None:
        self._reconstruct_active_formatting_elements()
        self._insert_element(token, push=False)
        self.frameset_ok = False
        return

    def _handle_body_start_simple_void(self, token: Tag) -> None:
        self._insert_element(token, push=False)
        return

    def _handle_body_start_input(self, token: Tag) -> None:
        input_type = None
        for name, value in token.attrs.items():
            if name == "type":
                input_type = (value or "").lower()
                break
        self._insert_element(token, push=False)
        if input_type != "hidden":
            self.frameset_ok = False
        return

    def _handle_body_start_table(self, token: Tag) -> None:
        if self.quirks_mode != "quirks":
            self._close_p_element()
        self._insert_element(token, push=True)
        self.frameset_ok = False
        self.mode = InsertionMode.IN_TABLE
        return

    def _handle_body_start_plaintext_xmp(self, token: Tag) -> None:
        self._close_p_element()
        self._insert_element(token, push=True)
        self.frameset_ok = False
        if token.name == "plaintext":
            self.tokenizer_state_override = TokenSinkResult.Plaintext
        else:
            # xmp, iframe, noembed, noframes
            self.original_mode = self.mode
            self.mode = InsertionMode.TEXT
        return

    def _handle_body_start_noscript(self, token: Tag) -> None:
        self._handle_body_start_default(token)
        return

    def _handle_body_start_textarea(self, token: Tag) -> None:
        self._insert_element(token, push=True)
        self.ignore_lf = True
        self.frameset_ok = False
        return

    def _handle_body_start_select(self, token: Tag) -> None:
        self._reconstruct_active_formatting_elements()
        self._insert_element(token, push=True)
        self.frameset_ok = False
        self._reset_insertion_mode()
        return

    def _handle_body_start_option(self, token: Tag) -> None:
        if self.open_elements and self.open_elements[-1].name == "option":
            self._pop_current()
        self._reconstruct_active_formatting_elements()
        self._insert_element(token, push=True)
        return

    def _handle_body_start_optgroup(self, token: Tag) -> None:
        if self.open_elements and self.open_elements[-1].name == "option":
            self._pop_current()
        self._reconstruct_active_formatting_elements()
        self._insert_element(token, push=True)
        return

    def _handle_body_start_rp_rt(self, token: Tag) -> None:
        self._generate_implied_end_tags(exclude="rtc")
        self._insert_element(token, push=True)
        return

    def _handle_body_start_rb_rtc(self, token: Tag) -> None:
        if self.open_elements and self.open_elements[-1].name in {"rb", "rp", "rt", "rtc"}:
            self._generate_implied_end_tags()
        self._insert_element(token, push=True)
        return

    def _handle_body_start_table_parse_error(self, token: Tag) -> None:
        self._parse_error("unexpected-start-tag", tag_name=token.name)
        return

    def _handle_body_start_default(self, token: Tag) -> ModeResultTuple | None:
        self._reconstruct_active_formatting_elements()
        self._insert_element(token, push=True)
        if token.self_closing:
            self._parse_error("non-void-html-element-start-tag-with-trailing-solidus", tag_name=token.name)
        # Elements reaching here have no handler - never in FRAMESET_NEUTRAL/FORMATTING_ELEMENTS
        self.frameset_ok = False
        return None

    def _mode_in_table(self, token: AnyToken) -> ModeResultTuple | None:
        if isinstance(token, CharacterTokens):
            data = token.data or ""
            if "\x00" in data:
                data = data.replace("\x00", "")
                if not data:
                    return None
                token = CharacterTokens(data)

            if is_all_whitespace(data):
                self._append_text(data)
                return None

            # html5lib-tests expect that some table foster-parenting text triggered by a
            # misnested formatting element (<a>) only produces an implied-end-tag error
            # when the table closes, not an additional character-in-table error.
            suppress_table_char_error = False
            if self.active_formatting:
                for idx in range(len(self.active_formatting) - 1, -1, -1):
                    entry = self.active_formatting[idx]
                    if entry is FORMAT_MARKER:
                        break
                    if entry["name"] == "a":
                        if entry["node"] not in self.open_elements:
                            suppress_table_char_error = True
                        break

            if not suppress_table_char_error:
                self.pending_table_text_should_error = True
            else:
                self.pending_table_text_should_error = False
            self.pending_table_text = []
            self.table_text_original_mode = self.mode
            self.mode = InsertionMode.IN_TABLE_TEXT
            return ("reprocess", InsertionMode.IN_TABLE_TEXT, token)
        if isinstance(token, CommentToken):
            self._append_comment(token.data)
            return None
        if isinstance(token, Tag):
            name = token.name
            if token.kind == Tag.START:
                if name == "caption":
                    self._clear_stack_until(_CLEAR_STACK_UNTIL_TABLE_TEMPLATE_HTML)
                    self._push_formatting_marker()
                    self._insert_element(token, push=True)
                    self.mode = InsertionMode.IN_CAPTION
                    return None
                if name == "colgroup":
                    self._clear_stack_until(_CLEAR_STACK_UNTIL_TABLE_TEMPLATE_HTML)
                    self._insert_element(token, push=True)
                    self.mode = InsertionMode.IN_COLUMN_GROUP
                    return None
                if name == "col":
                    self._clear_stack_until(_CLEAR_STACK_UNTIL_TABLE_TEMPLATE_HTML)
                    implied = Tag(Tag.START, "colgroup", {}, False)
                    self._insert_element(implied, push=True)
                    self.mode = InsertionMode.IN_COLUMN_GROUP
                    return ("reprocess", InsertionMode.IN_COLUMN_GROUP, token)
                if name in {"tbody", "tfoot", "thead"}:
                    self._clear_stack_until(_CLEAR_STACK_UNTIL_TABLE_TEMPLATE_HTML)
                    self._insert_element(token, push=True)
                    self.mode = InsertionMode.IN_TABLE_BODY
                    return None
                if name in {"td", "th", "tr"}:
                    self._clear_stack_until(_CLEAR_STACK_UNTIL_TABLE_TEMPLATE_HTML)
                    implied = Tag(Tag.START, "tbody", {}, False)
                    self._insert_element(implied, push=True)
                    self.mode = InsertionMode.IN_TABLE_BODY
                    return ("reprocess", InsertionMode.IN_TABLE_BODY, token)
                if name == "table":
                    self._parse_error("unexpected-start-tag-implies-end-tag", tag_name=name)
                    closed = self._close_table_element()
                    if closed:
                        return ("reprocess", self.mode, token)
                    return None
                if name in {"style", "script"}:
                    # Per HTML5 spec: style and script are inserted directly into the table
                    # (not processed as in-head which would move them)
                    self._insert_element(token, push=True)
                    self.original_mode = self.mode
                    self.mode = InsertionMode.TEXT
                    return None
                if name == "template":
                    # Template is handled by delegating to IN_HEAD
                    return self._mode_in_head(token)
                if name == "input":
                    input_type = None
                    for attr_name, attr_value in token.attrs.items():
                        if attr_name == "type":
                            input_type = (attr_value or "").lower()
                            break
                    if input_type == "hidden":
                        self._parse_error("unexpected-hidden-input-in-table")
                        self._insert_element(token, push=True)
                        self._pop_current()  # push=True always adds to stack
                        return None
                if name == "form":
                    self._parse_error("unexpected-form-in-table")
                    if self.form_element is None:
                        node = self._insert_element(token, push=True)
                        self.form_element = node
                        self._pop_current()  # push=True always adds to stack
                    return None
                self._parse_error("foster-parenting-start-tag", tag_name=name)
                previous = self.insert_from_table
                self.insert_from_table = True
                try:
                    return self._mode_in_body(token)
                finally:
                    self.insert_from_table = previous
            else:
                if name == "table":
                    self._close_table_element()
                    return None
                if name in {"body", "caption", "col", "colgroup", "html", "tbody", "td", "tfoot", "th", "thead", "tr"}:
                    self._parse_error("unexpected-end-tag", tag_name=name)
                    return None
                self._parse_error("unexpected-end-tag-implies-table-voodoo", tag_name=name)
                previous = self.insert_from_table
                self.insert_from_table = True
                try:
                    return self._mode_in_body(token)
                finally:
                    self.insert_from_table = previous
        # Per spec, only CharacterTokens, CommentToken, Tag, and EOFToken exist
        assert isinstance(token, EOFToken), f"Unexpected token type: {type(token)}"
        # If we're in a template, handle EOF in template mode first
        if self.template_modes:
            return self._mode_in_template(token)
        if self._has_in_table_scope("table"):
            self._parse_error("eof-in-table")
        return None

    def _mode_in_table_text(self, token: AnyToken) -> ModeResultTuple | None:
        if isinstance(token, CharacterTokens):
            # IN_TABLE mode guarantees non-empty data
            data = token.data
            self.pending_table_text.append(data)
            return None

        if (
            self.pending_table_text
            and isinstance(token, Tag)
            and token.kind == Tag.END
            and token.name == "table"
            and not is_all_whitespace("".join(self.pending_table_text))
        ):
            # If a misnested <a> exists only in the active formatting list, html5lib
            # reports the implied close when the table ends.
            if self.active_formatting:
                for idx in range(len(self.active_formatting) - 1, -1, -1):
                    entry = self.active_formatting[idx]
                    if entry is FORMAT_MARKER:
                        break
                    if entry["name"] == "a" and entry["node"] not in self.open_elements:
                        self._parse_error("unexpected-implied-end-tag-in-table-view")
                        break
        self._flush_pending_table_text()
        original = self.table_text_original_mode or InsertionMode.IN_TABLE
        self.table_text_original_mode = None
        self.mode = original
        return ("reprocess", original, token)

    def _mode_in_caption(self, token: AnyToken) -> ModeResultTuple | None:
        if isinstance(token, CharacterTokens):
            return self._mode_in_body(token)
        if isinstance(token, CommentToken):
            self._append_comment(token.data)
            return None
        if isinstance(token, Tag):
            name = token.name
            if token.kind == Tag.START:
                if name in {"caption", "col", "colgroup", "tbody", "tfoot", "thead", "tr", "td", "th"}:
                    self._parse_error("unexpected-start-tag-implies-end-tag", tag_name=name)
                    if self._close_caption_element():
                        return ("reprocess", InsertionMode.IN_TABLE, token)
                    # Fragment parsing with caption context: caption not on stack, ignore table structure elements
                    return None
                if name == "table":
                    self._parse_error("unexpected-start-tag-implies-end-tag", tag_name=name)
                    if self._close_caption_element():
                        return ("reprocess", InsertionMode.IN_TABLE, token)
                    # Fragment parsing: no caption on stack - handle in body mode
                    return self._mode_in_body(token)
                return self._mode_in_body(token)
            if name == "caption":
                if not self._close_caption_element():
                    return None
                return None
            if name == "table":
                if self._close_caption_element():
                    return ("reprocess", InsertionMode.IN_TABLE, token)
                return None
            if name in {"tbody", "tfoot", "thead"}:
                # These elements are never in table scope when in caption -
                # caption closes any open tbody/tfoot/thead when created
                self._parse_error("unexpected-end-tag", tag_name=name)
                return None
            return self._mode_in_body(token)
        assert isinstance(token, EOFToken), f"Unexpected token type: {type(token)}"
        return self._mode_in_body(token)

    def _close_caption_element(self) -> bool:
        if not self._has_in_table_scope("caption"):
            self._parse_error("unexpected-end-tag", tag_name="caption")
            return False
        self._generate_implied_end_tags()
        # Caption verified in scope above
        while self.open_elements:  # pragma: no branch
            node = self._pop_current()
            if node.name == "caption":
                break
        self._clear_active_formatting_up_to_marker()
        self.mode = InsertionMode.IN_TABLE
        return True

    def _mode_in_column_group(self, token: AnyToken) -> ModeResultTuple | None:
        current = self.open_elements[-1] if self.open_elements else None
        if isinstance(token, CharacterTokens):
            data = token.data or ""
            # Find first non-whitespace character
            stripped = data.lstrip(" \t\n\r\f")

            if len(stripped) < len(data):
                # Has leading whitespace - insert it
                ws = data[: len(data) - len(stripped)]
                self._append_text(ws)

            # Continue processing non-whitespace with a new token
            non_ws_token = CharacterTokens(stripped)
            if current and current.name == "html":
                # Fragment parsing with colgroup context: drop non-whitespace characters
                # (This is the only way html can be current in IN_COLUMN_GROUP mode)
                self._parse_error("unexpected-characters-in-column-group")
                return None
            # In a template, non-whitespace characters are parse errors - ignore them
            if current and current.name == "template":
                self._parse_error("unexpected-characters-in-template-column-group")
                return None
            self._parse_error("unexpected-characters-in-column-group")
            self._pop_current()
            self.mode = InsertionMode.IN_TABLE
            return ("reprocess", InsertionMode.IN_TABLE, non_ws_token)
        if isinstance(token, CommentToken):
            self._append_comment(token.data)
            return None
        if isinstance(token, Tag):
            name = token.name
            if token.kind == Tag.START:
                if name == "html":
                    return self._mode_in_body(token)
                if name == "col":
                    self._insert_element(token, push=True)
                    self._pop_current()  # push=True always adds to stack
                    return None
                if name == "template":
                    # Template is handled by delegating to IN_HEAD
                    return self._mode_in_head(token)
                if name == "colgroup":
                    self._parse_error("unexpected-start-tag-implies-end-tag", tag_name=name)
                    # Don't pop template element - only pop actual colgroup
                    if current and current.name == "colgroup":
                        self._pop_current()
                        self.mode = InsertionMode.IN_TABLE
                        return ("reprocess", InsertionMode.IN_TABLE, token)
                    return None
                if (
                    self.fragment_context
                    and self.fragment_context.tag_name.lower() == "colgroup"
                    and not self._has_in_table_scope("table")
                ):
                    self._parse_error("unexpected-start-tag-in-column-group", tag_name=name)
                    return None
                # Anything else: if we're in a colgroup, pop it and switch to IN_TABLE
                if current and current.name == "colgroup":
                    self._pop_current()
                    self.mode = InsertionMode.IN_TABLE
                    return ("reprocess", InsertionMode.IN_TABLE, token)
                # In template column group context (via <col> in template), ignore non-column content
                # At this point current is template - the only other case after colgroup fragment
                # and colgroup element are handled
                self._parse_error("unexpected-start-tag-in-template-column-group", tag_name=name)
                return None
            if name == "colgroup":
                if current and current.name == "colgroup":
                    self._pop_current()
                    self.mode = InsertionMode.IN_TABLE
                else:
                    self._parse_error("unexpected-end-tag", tag_name=token.name)
                return None
            if name == "col":
                self._parse_error("unexpected-end-tag", tag_name=name)
                return None
            if name == "template":
                # Template end tag needs proper handling
                return self._mode_in_head(token)
            if current and current.name != "html":  # pragma: no branch
                self._pop_current()
                self.mode = InsertionMode.IN_TABLE
            return ("reprocess", InsertionMode.IN_TABLE, token)
        assert isinstance(token, EOFToken), f"Unexpected token type: {type(token)}"
        if current and current.name == "colgroup":
            self._pop_current()
            self.mode = InsertionMode.IN_TABLE
            return ("reprocess", InsertionMode.IN_TABLE, token)
        if current and current.name == "template":
            # In template, delegate EOF handling to IN_TEMPLATE
            return self._mode_in_template(token)
        return None
        # Per spec: EOF when current is html - implicit None return

    def _mode_in_table_body(self, token: AnyToken) -> ModeResultTuple | None:
        if isinstance(token, CharacterTokens) or isinstance(token, CommentToken):
            return self._mode_in_table(token)
        if isinstance(token, Tag):
            name = token.name
            if token.kind == Tag.START:
                if name == "tr":
                    self._clear_stack_until(_CLEAR_STACK_UNTIL_TBODY_TFOOT_THEAD_TEMPLATE_HTML)
                    self._insert_element(token, push=True)
                    self.mode = InsertionMode.IN_ROW
                    return None
                if name in {"td", "th"}:
                    self._parse_error("unexpected-cell-in-table-body")
                    self._clear_stack_until(_CLEAR_STACK_UNTIL_TBODY_TFOOT_THEAD_TEMPLATE_HTML)
                    implied = Tag(Tag.START, "tr", {}, False)
                    self._insert_element(implied, push=True)
                    self.mode = InsertionMode.IN_ROW
                    return ("reprocess", InsertionMode.IN_ROW, token)
                if name in {"caption", "col", "colgroup", "tbody", "tfoot", "thead", "table"}:
                    current = self.open_elements[-1] if self.open_elements else None
                    # When in a template, these tags create invalid structure - treat as "anything else"
                    if current and current.name == "template":
                        self._parse_error("unexpected-start-tag-in-template-table-context", tag_name=name)
                        return None
                    # In fragment parsing with tbody/tfoot/thead context and no tbody on stack, ignore these tags
                    if (
                        self.fragment_context
                        and current
                        and current.name == "html"
                        and self.fragment_context.tag_name.lower() in {"tbody", "tfoot", "thead"}
                    ):
                        self._parse_error("unexpected-start-tag")
                        return None
                    # Pop tbody/tfoot/thead (stack always has elements here in normal parsing)
                    if self.open_elements:
                        self._pop_current()
                        self.mode = InsertionMode.IN_TABLE
                        return ("reprocess", InsertionMode.IN_TABLE, token)
                    # Empty stack edge case - go directly to IN_TABLE without reprocess
                    self.mode = InsertionMode.IN_TABLE  # pragma: no cover
                    return None  # pragma: no cover
                return self._mode_in_table(token)
            if name in {"tbody", "tfoot", "thead"}:
                if not self._has_in_table_scope(name):
                    self._parse_error("unexpected-end-tag", tag_name=name)
                    return None
                self._clear_stack_until(_CLEAR_STACK_UNTIL_TBODY_TFOOT_THEAD_TEMPLATE_HTML)
                self._pop_current()
                self.mode = InsertionMode.IN_TABLE
                return None
            if name == "table":
                current = self.open_elements[-1] if self.open_elements else None
                # In a template, reject </table> as there's no table element
                if current and current.name == "template":
                    self._parse_error("unexpected-end-tag", tag_name=token.name)
                    return None
                # In fragment parsing with tbody/tfoot/thead context and no tbody on stack, ignore </table>
                if (
                    self.fragment_context
                    and current
                    and current.name == "html"
                    and self.fragment_context.tag_name.lower() in {"tbody", "tfoot", "thead"}
                ):
                    self._parse_error("unexpected-end-tag", tag_name=token.name)
                    return None
                if current and current.name in {"tbody", "tfoot", "thead"}:
                    self._pop_current()
                self.mode = InsertionMode.IN_TABLE
                return ("reprocess", InsertionMode.IN_TABLE, token)
            if name in {"caption", "col", "colgroup", "td", "th", "tr"}:
                self._parse_error("unexpected-end-tag", tag_name=name)
                return None
            return self._mode_in_table(token)
        assert isinstance(token, EOFToken), f"Unexpected token type: {type(token)}"
        return self._mode_in_table(token)

    def _mode_in_row(self, token: AnyToken) -> ModeResultTuple | None:
        if isinstance(token, CharacterTokens) or isinstance(token, CommentToken):
            return self._mode_in_table(token)
        if isinstance(token, Tag):
            name = token.name
            if token.kind == Tag.START:
                if name in {"td", "th"}:
                    self._clear_stack_until(_CLEAR_STACK_UNTIL_TR_TEMPLATE_HTML)
                    self._insert_element(token, push=True)
                    self._push_formatting_marker()
                    self.mode = InsertionMode.IN_CELL
                    return None
                if name in {"caption", "col", "colgroup", "tbody", "tfoot", "thead", "tr", "table"}:
                    if not self._has_in_table_scope("tr"):
                        self._parse_error("unexpected-start-tag-implies-end-tag", tag_name=name)
                        return None
                    self._end_tr_element()
                    return ("reprocess", self.mode, token)
                previous = self.insert_from_table
                self.insert_from_table = True
                try:
                    return self._mode_in_body(token)
                finally:
                    self.insert_from_table = previous
            else:
                if name == "tr":
                    if not self._has_in_table_scope("tr"):
                        self._parse_error("unexpected-end-tag", tag_name=name)
                        return None
                    self._end_tr_element()
                    return None
                if name in {"table", "tbody", "tfoot", "thead"}:
                    if self._has_in_table_scope(name):
                        self._end_tr_element()
                        return ("reprocess", self.mode, token)
                    self._parse_error("unexpected-end-tag", tag_name=name)
                    return None
                if name in {"caption", "col", "group", "td", "th"}:
                    self._parse_error("unexpected-end-tag", tag_name=name)
                    return None
                previous = self.insert_from_table
                self.insert_from_table = True
                try:
                    return self._mode_in_body(token)
                finally:
                    self.insert_from_table = previous
        assert isinstance(token, EOFToken), f"Unexpected token type: {type(token)}"
        return self._mode_in_table(token)

    def _end_tr_element(self) -> None:
        self._clear_stack_until(_CLEAR_STACK_UNTIL_TR_TEMPLATE_HTML)
        # Pop tr if on top (may not be if stack was exhausted)
        if self.open_elements and self.open_elements[-1].name == "tr":
            self._pop_current()
        # When in a template, restore template mode; otherwise use IN_TABLE_BODY
        if self.template_modes:
            self.mode = self.template_modes[-1]
        else:
            self.mode = InsertionMode.IN_TABLE_BODY

    def _mode_in_cell(self, token: AnyToken) -> ModeResultTuple | None:
        if isinstance(token, CharacterTokens):
            previous = self.insert_from_table
            self.insert_from_table = False
            try:
                return self._mode_in_body(token)
            finally:
                self.insert_from_table = previous
        if isinstance(token, CommentToken):
            self._append_comment(token.data)
            return None
        if isinstance(token, Tag):
            name = token.name
            if token.kind == Tag.START:
                if name in {"caption", "col", "colgroup", "tbody", "td", "tfoot", "th", "thead", "tr"}:
                    if self._close_table_cell():
                        return ("reprocess", self.mode, token)
                    # Per spec: if we reach here in IN_CELL mode with no cell to close,
                    # we're in a fragment context with td/th as context element and no table structure.
                    # Issue parse error and ignore the token.
                    self._parse_error("unexpected-start-tag-in-cell-fragment", tag_name=name)
                    return None
                previous = self.insert_from_table
                self.insert_from_table = False
                try:
                    return self._mode_in_body(token)
                finally:
                    self.insert_from_table = previous
            else:
                if name in {"td", "th"}:
                    if not self._has_in_table_scope(name):
                        self._parse_error("unexpected-end-tag", tag_name=name)
                        return None
                    self._end_table_cell(name)
                    return None
                if name in {"table", "tbody", "tfoot", "thead", "tr"}:
                    # Per HTML5 spec: only close cell if the element is actually in scope
                    # Otherwise it's a parse error and we ignore the token
                    if not self._has_in_table_scope(name):
                        self._parse_error("unexpected-end-tag", tag_name=name)
                        return None
                    self._close_table_cell()
                    return ("reprocess", self.mode, token)
                previous = self.insert_from_table
                self.insert_from_table = False
                try:
                    return self._mode_in_body(token)
                finally:
                    self.insert_from_table = previous
        assert isinstance(token, EOFToken), f"Unexpected token type: {type(token)}"
        if self._close_table_cell():
            return ("reprocess", self.mode, token)
        return self._mode_in_table(token)

    def _mode_in_select(self, token: AnyToken) -> ModeResultTuple | None:
        if isinstance(token, CharacterTokens):
            data = token.data or ""
            if "\x00" in data:
                data = data.replace("\x00", "")
            if data:
                self._reconstruct_active_formatting_elements()
                self._append_text(data)
            return None
        if isinstance(token, CommentToken):
            self._append_comment(token.data)
            return None
        if isinstance(token, Tag):
            name = token.name
            if token.kind == Tag.START:
                if name == "html":
                    return ("reprocess", InsertionMode.IN_BODY, token)
                if name == "option":
                    if self.open_elements and self.open_elements[-1].name == "option":
                        self._pop_current()
                    self._reconstruct_active_formatting_elements()
                    self._insert_element(token, push=True)
                    return None
                if name == "optgroup":
                    if self.open_elements and self.open_elements[-1].name == "option":
                        self._pop_current()
                    if self.open_elements and self.open_elements[-1].name == "optgroup":
                        self._pop_current()
                    self._reconstruct_active_formatting_elements()
                    self._insert_element(token, push=True)
                    return None
                if name == "select":
                    self._parse_error("unexpected-select-in-select")
                    # select is always in scope in IN_SELECT mode
                    self._pop_until_inclusive("select")
                    self._reset_insertion_mode()
                    return None
                if name in {"input", "textarea"}:
                    self._parse_error("unexpected-start-tag-in-select", tag_name=name)
                    # select is always in scope in IN_SELECT mode
                    self._pop_until_inclusive("select")
                    self._reset_insertion_mode()
                    return ("reprocess", self.mode, token)
                if name == "keygen":
                    self._reconstruct_active_formatting_elements()
                    self._insert_element(token, push=False)
                    return None
                if name in {"caption", "col", "colgroup", "tbody", "td", "tfoot", "th", "thead", "tr", "table"}:
                    self._parse_error("unexpected-start-tag-in-select", tag_name=name)
                    # select is always in scope in IN_SELECT mode
                    self._pop_until_inclusive("select")
                    self._reset_insertion_mode()
                    return ("reprocess", self.mode, token)
                if name in {"script", "template"}:
                    return self._mode_in_head(token)
                if name in {"svg", "math"}:
                    # For foreign elements, honor the self-closing flag
                    self._reconstruct_active_formatting_elements()
                    self._insert_element(token, push=not token.self_closing, namespace=name)
                    return None
                if name in FORMATTING_ELEMENTS:
                    self._reconstruct_active_formatting_elements()
                    node = self._insert_element(token, push=True)
                    self._append_active_formatting_entry(name, token.attrs, node)
                    return None
                if name == "hr":
                    self._parse_error("unexpected-start-tag-in-select", tag_name=name)
                    # Per spec: pop option and optgroup before inserting hr (makes hr sibling, not child)
                    if self.open_elements and self.open_elements[-1].name == "option":
                        self._pop_current()
                    if self.open_elements and self.open_elements[-1].name == "optgroup":
                        self._pop_current()
                    self._reconstruct_active_formatting_elements()
                    self._insert_element(token, push=False)
                    return None
                if name == "menuitem":
                    self._parse_error("unexpected-start-tag-in-select", tag_name=name)
                    self._reconstruct_active_formatting_elements()
                    self._insert_element(token, push=True)
                    return None
                # Allow common HTML elements in select (newer spec)
                if name in {"p", "div", "span", "button", "datalist", "selectedcontent"}:
                    self._parse_error("unexpected-start-tag-in-select", tag_name=name)
                    self._reconstruct_active_formatting_elements()
                    self._insert_element(token, push=not token.self_closing)
                    return None
                if name in {"br", "img"}:
                    self._parse_error("unexpected-start-tag-in-select", tag_name=name)
                    self._reconstruct_active_formatting_elements()
                    self._insert_element(token, push=False)
                    return None
                if name == "plaintext":
                    # Per spec: plaintext element is inserted in select (consumes all remaining text)
                    self._parse_error("unexpected-start-tag-in-select", tag_name=name)
                    self._reconstruct_active_formatting_elements()
                    self._insert_element(token, push=True)
                    return None
                # Any other start tag: parse error, ignore.
                self._parse_error("unexpected-start-tag-in-select", tag_name=name)
                return None
            if name == "optgroup":
                if self.open_elements and self.open_elements[-1].name == "option":
                    self._pop_current()
                if self.open_elements and self.open_elements[-1].name == "optgroup":
                    self._pop_current()
                else:
                    self._parse_error("unexpected-end-tag-in-select", tag_name=token.name)
                return None
            if name == "option":
                if self.open_elements and self.open_elements[-1].name == "option":
                    self._pop_current()
                else:
                    self._parse_error("unexpected-end-tag-in-select", tag_name=token.name)
                return None
            if name == "select":
                # In IN_SELECT mode, select is always in scope - pop to it
                self._pop_until_inclusive("select")
                self._reset_insertion_mode()
                return None
            # Handle end tags for allowed HTML elements in select
            if name == "a" or name in FORMATTING_ELEMENTS:
                # select is always on stack in IN_SELECT mode
                select_node = self._find_last_on_stack("select")
                fmt_index = self._find_active_formatting_index(name)
                if fmt_index is None:
                    self._parse_error("unexpected-end-tag-in-select", tag_name=name)
                    return None
                target = self.active_formatting[fmt_index]["node"]
                if target in self.open_elements:  # pragma: no branch
                    select_index = self.open_elements.index(select_node)
                    target_index = self.open_elements.index(target)
                    if target_index < select_index:
                        self._parse_error("unexpected-end-tag-in-select", tag_name=name)
                        return None
                self._adoption_agency(name)
                return None
            if name in {"p", "div", "span", "button", "datalist", "selectedcontent"}:
                self._parse_error("unexpected-end-tag-in-select", tag_name=name)
                # Per HTML5 spec: these end tags in select mode close the element if it's on the stack.
                # But we must not pop across the select boundary (i.e., don't pop elements BEFORE select).
                select_idx = None
                target_idx = None
                for i, node in enumerate(self.open_elements):
                    if node.name == "select" and select_idx is None:
                        select_idx = i
                    if node.name == name:
                        target_idx = i  # Track the LAST occurrence
                # Only pop if target exists and is AFTER (or at same level as) select
                # i.e., the target is inside the select or there's no select
                if target_idx is not None and (select_idx is None or target_idx > select_idx):
                    while True:
                        popped = self._pop_current()
                        if popped.name == name:
                            break
                return None
            if name in {"caption", "col", "colgroup", "tbody", "td", "tfoot", "th", "thead", "tr", "table"}:
                self._parse_error("unexpected-end-tag-in-select", tag_name=name)
                # select is always in scope in IN_SELECT mode
                self._pop_until_inclusive("select")
                self._reset_insertion_mode()
                return ("reprocess", self.mode, token)
            # Any other end tag: parse error, ignore
            self._parse_error("unexpected-end-tag", tag_name=name)
            return None
        assert isinstance(token, EOFToken), f"Unexpected token type: {type(token)}"
        return self._mode_in_body(token)

    def _mode_in_template(self, token: AnyToken) -> ModeResultTuple | None:
        # § The "in template" insertion mode
        # https://html.spec.whatwg.org/multipage/parsing.html#parsing-main-intemplate
        if isinstance(token, CharacterTokens):
            return self._mode_in_body(token)
        if isinstance(token, CommentToken):
            return self._mode_in_body(token)
        if isinstance(token, Tag):
            if token.kind == Tag.START:
                # Table-related tags switch template mode
                if token.name in {"caption", "colgroup", "tbody", "tfoot", "thead"}:
                    self.template_modes.pop()
                    self.template_modes.append(InsertionMode.IN_TABLE)
                    self.mode = InsertionMode.IN_TABLE
                    return ("reprocess", InsertionMode.IN_TABLE, token)
                if token.name == "col":
                    self.template_modes.pop()
                    self.template_modes.append(InsertionMode.IN_COLUMN_GROUP)
                    self.mode = InsertionMode.IN_COLUMN_GROUP
                    return ("reprocess", InsertionMode.IN_COLUMN_GROUP, token)
                if token.name == "tr":
                    self.template_modes.pop()
                    self.template_modes.append(InsertionMode.IN_TABLE_BODY)
                    self.mode = InsertionMode.IN_TABLE_BODY
                    return ("reprocess", InsertionMode.IN_TABLE_BODY, token)
                if token.name in {"td", "th"}:
                    self.template_modes.pop()
                    self.template_modes.append(InsertionMode.IN_ROW)
                    self.mode = InsertionMode.IN_ROW
                    return ("reprocess", InsertionMode.IN_ROW, token)
                # Default: pop template mode and push IN_BODY
                if token.name not in {
                    "base",
                    "basefont",
                    "bgsound",
                    "link",
                    "meta",
                    "noframes",
                    "script",
                    "style",
                    "template",
                    "title",
                }:
                    self.template_modes.pop()
                    self.template_modes.append(InsertionMode.IN_BODY)
                    self.mode = InsertionMode.IN_BODY
                    return ("reprocess", InsertionMode.IN_BODY, token)
            if token.kind == Tag.END and token.name == "template":
                return self._mode_in_head(token)
            # Head-related tags process in InHead
            if token.name in {
                "base",
                "basefont",
                "bgsound",
                "link",
                "meta",
                "noframes",
                "script",
                "style",
                "template",
                "title",
            }:
                return self._mode_in_head(token)
        if isinstance(token, EOFToken):
            # Check if template is on the stack (don't use _in_scope as table blocks it)
            has_template = any(node.name == "template" for node in self.open_elements)
            if not has_template:
                return None
            # Parse error for EOF in template
            self._parse_error("expected-closing-tag-but-got-eof", tag_name="template")
            # Pop until template, then handle EOF in reset mode
            self._pop_until_inclusive("template")
            self._clear_active_formatting_up_to_marker()
            # template_modes is always non-empty when template is on stack
            self.template_modes.pop()
            self._reset_insertion_mode()
            return ("reprocess", self.mode, token)
        return None

    def _mode_after_body(self, token: AnyToken) -> ModeResultTuple | None:
        if isinstance(token, CharacterTokens):
            if is_all_whitespace(token.data):
                # Whitespace is processed using InBody rules (appended to body)
                # but we stay in AfterBody mode
                self._mode_in_body(token)
                return None
            return ("reprocess", InsertionMode.IN_BODY, token)
        if isinstance(token, CommentToken):
            self._append_comment(token.data, parent=self.open_elements[0])
            return None
        if isinstance(token, Tag):
            if token.kind == Tag.START and token.name == "html":
                return ("reprocess", InsertionMode.IN_BODY, token)
            if token.kind == Tag.END and token.name == "html":
                self.mode = InsertionMode.AFTER_AFTER_BODY
                return None
            return ("reprocess", InsertionMode.IN_BODY, token)
        assert isinstance(token, EOFToken), f"Unexpected token type: {type(token)}"
        return None

    def _mode_after_after_body(self, token: AnyToken) -> ModeResultTuple | None:
        if isinstance(token, CharacterTokens):
            if is_all_whitespace(token.data):
                # Per spec: whitespace characters are inserted using the rules for the "in body" mode
                # Process with InBody rules but stay in AfterAfterBody mode
                self._mode_in_body(token)
                return None
            # Non-whitespace character: parse error, reprocess in IN_BODY
            self._parse_error("unexpected-char-after-body")
            return ("reprocess", InsertionMode.IN_BODY, token)
        if isinstance(token, CommentToken):
            if self.fragment_context is not None:
                # html is always on stack in fragment parsing
                html_node = self._find_last_on_stack("html")
                html_node.append_child(Comment(data=token.data))
                return None
            self._append_comment_to_document(token.data)
            return None
        if isinstance(token, Tag):
            if token.kind == Tag.START and token.name == "html":
                return ("reprocess", InsertionMode.IN_BODY, token)
            # Any other tag: parse error, reprocess in IN_BODY
            self._parse_error("unexpected-token-after-body")
            return ("reprocess", InsertionMode.IN_BODY, token)
        assert isinstance(token, EOFToken), f"Unexpected token type: {type(token)}"
        return None

    def _mode_in_frameset(self, token: AnyToken) -> ModeResultTuple | None:
        # Per HTML5 spec §13.2.6.4.16: In frameset insertion mode
        if isinstance(token, CharacterTokens):
            # Only whitespace characters allowed; ignore all others
            whitespace = "".join(ch for ch in token.data if ch in "\t\n\f\r ")
            if whitespace:
                self._append_text(whitespace)
            return None
        if isinstance(token, CommentToken):
            self._append_comment(token.data)
            return None
        if isinstance(token, Tag):
            if token.kind == Tag.START and token.name == "html":
                return ("reprocess", InsertionMode.IN_BODY, token)
            if token.kind == Tag.START and token.name == "frameset":
                self._insert_element(token, push=True)
                return None
            if token.kind == Tag.END and token.name == "frameset":
                if self.open_elements and self.open_elements[-1].name == "html":
                    self._parse_error("unexpected-end-tag", tag_name=token.name)
                    return None
                self._pop_current()
                if self.open_elements and self.open_elements[-1].name != "frameset":
                    self.mode = InsertionMode.AFTER_FRAMESET
                return None
            if token.kind == Tag.START and token.name == "frame":
                self._insert_element(token, push=True)
                self._pop_current()
                return None
            if token.kind == Tag.START and token.name == "noframes":
                # Per spec: use IN_HEAD rules but preserve current mode for TEXT restoration
                self._insert_element(token, push=True)
                self.original_mode = self.mode
                self.mode = InsertionMode.TEXT
                return None
        if isinstance(token, EOFToken):
            if self.open_elements and self.open_elements[-1].name != "html":
                self._parse_error("expected-closing-tag-but-got-eof", tag_name=self.open_elements[-1].name)
            return None
        self._parse_error("unexpected-token-in-frameset")
        return None

    def _mode_after_frameset(self, token: AnyToken) -> ModeResultTuple | None:
        # Per HTML5 spec §13.2.6.4.17: After frameset insertion mode
        if isinstance(token, CharacterTokens):
            # Only whitespace characters allowed; non-whitespace is a parse error.
            data = token.data or ""
            whitespace = "".join(ch for ch in data if ch in "\t\n\f\r ")
            if any(ch not in "\t\n\f\r " for ch in data):
                self._parse_error("unexpected-token-after-frameset")
            if whitespace:
                self._append_text(whitespace)
            return None
        if isinstance(token, CommentToken):
            self._append_comment(token.data)
            return None
        if isinstance(token, Tag):
            if token.kind == Tag.START and token.name == "html":
                return ("reprocess", InsertionMode.IN_BODY, token)
            if token.kind == Tag.END and token.name == "html":
                self.mode = InsertionMode.AFTER_AFTER_FRAMESET
                return None
            if token.kind == Tag.END and token.name == "frameset":
                self._parse_error("unexpected-token-after-frameset")
                return None
            if token.kind == Tag.START and token.name == "noframes":
                # Insert noframes element directly and switch to TEXT mode
                self._insert_element(token, push=True)
                self.original_mode = self.mode
                self.mode = InsertionMode.TEXT
                return None
        if isinstance(token, EOFToken):
            return None
        self._parse_error("unexpected-token-after-frameset")
        self.mode = InsertionMode.IN_FRAMESET
        return ("reprocess", InsertionMode.IN_FRAMESET, token)

    def _mode_after_after_frameset(self, token: AnyToken) -> ModeResultTuple | None:
        # Per HTML5 spec §13.2.6.4.18: After after frameset insertion mode
        if isinstance(token, CharacterTokens):
            # Whitespace is processed using InBody rules
            # but we stay in AfterAfterFrameset mode
            if is_all_whitespace(token.data):
                self._mode_in_body(token)
                return None
            # Non-whitespace falls through to "Anything else"
        if isinstance(token, CommentToken):
            self._append_comment_to_document(token.data)
            return None
        if isinstance(token, Tag):
            if token.kind == Tag.START and token.name == "html":
                return ("reprocess", InsertionMode.IN_BODY, token)
            if token.kind == Tag.START and token.name == "noframes":
                # Insert noframes element directly and switch to TEXT mode
                self._insert_element(token, push=True)
                self.original_mode = self.mode
                self.mode = InsertionMode.TEXT
                return None
            # Other tags fall through to "Anything else"
        if isinstance(token, EOFToken):
            return None
        # Anything else: parse error, reprocess in IN_FRAMESET
        self._parse_error("unexpected-token-after-after-frameset")
        self.mode = InsertionMode.IN_FRAMESET
        return ("reprocess", InsertionMode.IN_FRAMESET, token)

    # Helpers ----------------------------------------------------------------

    _MODE_HANDLERS: list[Callable[[TreeBuilderModesMixin, AnyToken], ModeResultTuple | None]] = [
        _mode_initial,
        _mode_before_html,
        _mode_before_head,
        _mode_in_head,
        _mode_in_head_noscript,
        _mode_after_head,
        _mode_text,
        _mode_in_body,
        _mode_after_body,
        _mode_after_after_body,
        _mode_in_table,
        _mode_in_table_text,
        _mode_in_caption,
        _mode_in_column_group,
        _mode_in_table_body,
        _mode_in_row,
        _mode_in_cell,
        _mode_in_frameset,
        _mode_after_frameset,
        _mode_after_after_frameset,
        _mode_in_select,
        _mode_in_template,
    ]

    _BODY_TOKEN_HANDLERS: dict[type[AnyToken], Callable[[TreeBuilderModesMixin, Any], ModeResultTuple | None]] = {
        CharacterTokens: _handle_characters_in_body,
        CommentToken: _handle_comment_in_body,
        Tag: _handle_tag_in_body,
        EOFToken: _handle_eof_in_body,
    }

    _BODY_START_HANDLERS: dict[str, Callable[[TreeBuilderModesMixin, Tag], ModeResultTuple | None]] = {
        "a": _handle_body_start_a,
        "address": _handle_body_start_block_with_p,
        "applet": _handle_body_start_applet_like,
        "area": _handle_body_start_void_with_formatting,
        "article": _handle_body_start_block_with_p,
        "aside": _handle_body_start_block_with_p,
        "b": _handle_body_start_formatting,
        "base": _handle_body_start_in_head,
        "basefont": _handle_body_start_in_head,
        "bgsound": _handle_body_start_in_head,
        "big": _handle_body_start_formatting,
        "blockquote": _handle_body_start_block_with_p,
        "body": _handle_body_start_body,
        "br": _handle_body_start_br,
        "button": _handle_body_start_button,
        "caption": _handle_body_start_table_parse_error,
        "center": _handle_body_start_block_with_p,
        "code": _handle_body_start_formatting,
        "col": _handle_body_start_col_or_frame,
        "colgroup": _handle_body_start_structure_ignored,
        "dd": _handle_body_start_dd_dt,
        "details": _handle_body_start_block_with_p,
        "dialog": _handle_body_start_block_with_p,
        "dir": _handle_body_start_block_with_p,
        "div": _handle_body_start_block_with_p,
        "dl": _handle_body_start_block_with_p,
        "dt": _handle_body_start_dd_dt,
        "em": _handle_body_start_formatting,
        "embed": _handle_body_start_void_with_formatting,
        "fieldset": _handle_body_start_block_with_p,
        "figcaption": _handle_body_start_block_with_p,
        "figure": _handle_body_start_block_with_p,
        "font": _handle_body_start_formatting,
        "footer": _handle_body_start_block_with_p,
        "form": _handle_body_start_form,
        "frame": _handle_body_start_col_or_frame,
        "frameset": _handle_body_start_frameset,
        "h1": _handle_body_start_heading,
        "h2": _handle_body_start_heading,
        "h3": _handle_body_start_heading,
        "h4": _handle_body_start_heading,
        "h5": _handle_body_start_heading,
        "h6": _handle_body_start_heading,
        "head": _handle_body_start_head,
        "header": _handle_body_start_block_with_p,
        "hgroup": _handle_body_start_block_with_p,
        "html": _handle_body_start_html,
        "i": _handle_body_start_formatting,
        "image": _handle_body_start_image,
        "img": _handle_body_start_void_with_formatting,
        "input": _handle_body_start_input,
        "keygen": _handle_body_start_void_with_formatting,
        "li": _handle_body_start_li,
        "link": _handle_body_start_in_head,
        "listing": _handle_body_start_pre_listing,
        "main": _handle_body_start_block_with_p,
        "marquee": _handle_body_start_applet_like,
        "math": _handle_body_start_math,
        "menu": _handle_body_start_block_with_p,
        "meta": _handle_body_start_in_head,
        "nav": _handle_body_start_block_with_p,
        "nobr": _handle_body_start_formatting,
        "noscript": _handle_body_start_noscript,
        "noframes": _handle_body_start_in_head,
        "object": _handle_body_start_applet_like,
        "ol": _handle_body_start_block_with_p,
        "optgroup": _handle_body_start_optgroup,
        "option": _handle_body_start_option,
        "p": _handle_body_start_paragraph,
        "param": _handle_body_start_simple_void,
        "plaintext": _handle_body_start_plaintext_xmp,
        "pre": _handle_body_start_pre_listing,
        "rb": _handle_body_start_rb_rtc,
        "rp": _handle_body_start_rp_rt,
        "rt": _handle_body_start_rp_rt,
        "rtc": _handle_body_start_rb_rtc,
        "s": _handle_body_start_formatting,
        "script": _handle_body_start_in_head,
        "search": _handle_body_start_block_with_p,
        "section": _handle_body_start_block_with_p,
        "select": _handle_body_start_select,
        "small": _handle_body_start_formatting,
        "source": _handle_body_start_simple_void,
        "strike": _handle_body_start_formatting,
        "strong": _handle_body_start_formatting,
        "style": _handle_body_start_in_head,
        "summary": _handle_body_start_block_with_p,
        "svg": _handle_body_start_svg,
        "table": _handle_body_start_table,
        "tbody": _handle_body_start_structure_ignored,
        "td": _handle_body_start_structure_ignored,
        "template": _handle_body_start_in_head,
        "textarea": _handle_body_start_textarea,
        "tfoot": _handle_body_start_structure_ignored,
        "th": _handle_body_start_structure_ignored,
        "thead": _handle_body_start_structure_ignored,
        "title": _handle_body_start_in_head,
        "tr": _handle_body_start_structure_ignored,
        "track": _handle_body_start_simple_void,
        "tt": _handle_body_start_formatting,
        "u": _handle_body_start_formatting,
        "ul": _handle_body_start_block_with_p,
        "wbr": _handle_body_start_void_with_formatting,
        "xmp": _handle_body_start_plaintext_xmp,
    }
    _BODY_END_HANDLERS: dict[str, Callable[[TreeBuilderModesMixin, Tag], ModeResultTuple | None]] = {
        "address": _handle_body_end_block,
        "applet": _handle_body_end_applet_like,
        "article": _handle_body_end_block,
        "aside": _handle_body_end_block,
        "blockquote": _handle_body_end_block,
        "body": _handle_body_end_body,
        "button": _handle_body_end_block,
        "center": _handle_body_end_block,
        "dd": _handle_body_end_dd_dt,
        "details": _handle_body_end_block,
        "dialog": _handle_body_end_block,
        "dir": _handle_body_end_block,
        "div": _handle_body_end_block,
        "dl": _handle_body_end_block,
        "dt": _handle_body_end_dd_dt,
        "fieldset": _handle_body_end_block,
        "figcaption": _handle_body_end_block,
        "figure": _handle_body_end_block,
        "footer": _handle_body_end_block,
        "form": _handle_body_end_form,
        "h1": _handle_body_end_heading,
        "h2": _handle_body_end_heading,
        "h3": _handle_body_end_heading,
        "h4": _handle_body_end_heading,
        "h5": _handle_body_end_heading,
        "h6": _handle_body_end_heading,
        "header": _handle_body_end_block,
        "hgroup": _handle_body_end_block,
        "html": _handle_body_end_html,
        "li": _handle_body_end_li,
        "listing": _handle_body_end_block,
        "main": _handle_body_end_block,
        "marquee": _handle_body_end_applet_like,
        "menu": _handle_body_end_block,
        "nav": _handle_body_end_block,
        "object": _handle_body_end_applet_like,
        "ol": _handle_body_end_block,
        "p": _handle_body_end_p,
        "pre": _handle_body_end_block,
        "search": _handle_body_end_block,
        "section": _handle_body_end_block,
        "summary": _handle_body_end_block,
        "table": _handle_body_end_block,
        "template": _handle_body_end_template,
        "ul": _handle_body_end_block,
    }
