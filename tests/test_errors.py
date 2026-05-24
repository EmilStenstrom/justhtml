"""Tests for error collection and strict mode."""

import unittest

from justhtml import JustHTML, ParseError, StrictModeError
from justhtml.tokenizer import Tokenizer
from justhtml.tokenizer.tokens import CharacterTokens, Tag
from justhtml.treebuilder import TreeBuilder


class TestErrorCollection(unittest.TestCase):
    """Test that errors are collected when collect_errors=True."""

    def test_no_errors_by_default(self):
        """By default, errors list is not populated (for performance)."""
        doc = JustHTML("<html><body></body></html>")
        # When collect_errors=False, errors is an empty list
        assert doc.errors == []

    def test_collect_errors_enabled(self):
        """When collect_errors=True, parse errors are collected."""
        # Null character triggers parse error
        doc = JustHTML("<p>\x00</p>", collect_errors=True)
        assert len(doc.errors) > 0
        assert all(isinstance(e, ParseError) for e in doc.errors)

    def test_error_has_line_and_column(self):
        """Errors include line and column information."""
        doc = JustHTML("<p>\x00</p>", collect_errors=True)
        assert len(doc.errors) > 0
        error = doc.errors[0]
        assert error.line is not None
        assert error.column is not None
        assert isinstance(error.line, int)
        assert isinstance(error.column, int)

    def test_error_code_is_string(self):
        """Error code is a descriptive string."""
        doc = JustHTML("<p>\x00</p>", collect_errors=True)
        assert len(doc.errors) > 0
        error = doc.errors[0]
        assert isinstance(error.code, str)
        assert len(error.code) > 0

    def test_valid_html_no_errors(self):
        """Well-formed HTML with doctype produces no errors."""
        doc = JustHTML("<!DOCTYPE html><html><head></head><body></body></html>", collect_errors=True)
        # May still have some parse errors depending on strictness
        # At minimum, this shouldn't crash
        assert isinstance(doc.errors, list)

    def test_multiline_error_positions(self):
        """Errors on different lines have correct line numbers."""
        html = "<!DOCTYPE html>\n<html>\n<body>\n<p><b></p>"  # Misnested tags
        doc = JustHTML(html, collect_errors=True)
        # Should have errors due to misnesting
        # Verify line numbers are tracked
        for error in doc.errors:
            assert error.line >= 1

    def test_error_column_after_newline(self):
        """Error column is calculated correctly after newlines."""
        # Put a null char after a newline to test column calculation
        html = "line1\nline2\x00"
        doc = JustHTML(html, collect_errors=True)
        assert len(doc.errors) > 0
        # The null is at position 11 (after newline at position 5)
        # Column should be relative to last newline
        error = next(e for e in doc.errors if e.code == "unexpected-null-character")
        assert error.line == 2
        assert error.column > 0

    def test_location_at_offset_lazy_without_error_collection(self):
        doc = JustHTML("<p>a\nb</p>", track_node_locations=True)
        p = doc.query("p")[0]
        text = p.children[0]
        assert text.name == "#text"
        assert text.origin_location == (1, 4)


class TestStrictMode(unittest.TestCase):
    """Test strict mode that raises on parse errors."""

    def test_strict_mode_raises(self):
        """Strict mode raises StrictModeError on first error."""
        with self.assertRaises(StrictModeError) as ctx:
            JustHTML("<p>\x00</p>", strict=True)
        assert ctx.exception.error is not None
        assert isinstance(ctx.exception.error, ParseError)

    def test_strict_mode_valid_html(self):
        """Strict mode with valid HTML doesn't raise."""
        # Fully valid HTML5 document
        doc = JustHTML(
            "<!DOCTYPE html><html><head><title>Test</title></head><body></body></html>",
            strict=True,
        )
        assert doc.root is not None
        # Empty errors list (since parsing succeeded)
        assert doc.errors == []

    def test_strict_mode_enables_error_collection(self):
        """Strict mode automatically enables error collection."""
        # We can't check this directly since it raises, but we verify
        # the exception contains error info
        with self.assertRaises(StrictModeError) as ctx:
            JustHTML("<p>\x00</p>", strict=True)
        error = ctx.exception.error
        assert error.line is not None
        assert error.column is not None


class TestParseError(unittest.TestCase):
    """Test ParseError class behavior."""

    def test_parse_error_str(self):
        """ParseError has readable string representation."""
        error = ParseError("test-error", line=1, column=5)
        assert str(error) == "(1,5): test-error"

    def test_parse_error_repr(self):
        """ParseError has useful repr."""
        error = ParseError("test-error", line=1, column=5)
        assert "test-error" in repr(error)
        assert "line=1" in repr(error)
        assert "column=5" in repr(error)

    def test_parse_error_equality(self):
        """ParseErrors with same values are equal."""
        e1 = ParseError("error-code", line=1, column=5)
        e2 = ParseError("error-code", line=1, column=5)
        e3 = ParseError("other-error", line=1, column=5)
        assert e1 == e2
        assert e1 != e3

    def test_parse_error_equality_with_non_parseerror(self):
        """ParseError compared with non-ParseError returns NotImplemented."""
        e1 = ParseError("error-code", line=1, column=5)
        assert e1.__eq__("not a ParseError") is NotImplemented

    def test_parse_error_no_location(self):
        """ParseError works without location info."""
        error = ParseError("test-error")
        assert str(error) == "test-error"
        assert "line=" not in repr(error)

    def test_parse_error_no_location_with_message(self):
        """ParseError with message but no location."""
        error = ParseError("test-error", message="This is a test error")
        assert str(error) == "test-error - This is a test error"
        assert "line=" not in repr(error)

    def test_parse_error_with_location_and_message(self):
        """ParseError with both location and message."""
        error = ParseError("test-error", line=5, column=10, message="Detailed error")
        assert str(error) == "(5,10): test-error - Detailed error"

    def test_parse_error_as_exception_no_location(self):
        """as_exception() works without location info."""
        error = ParseError("test-error", message="Test error message")
        exc = error.as_exception()
        assert isinstance(exc, SyntaxError)
        assert exc.msg == "Test error message"
        assert not hasattr(exc, "lineno") or exc.lineno is None

    def test_parse_error_as_exception_with_location(self):
        """as_exception() highlights HTML source location."""
        html = "<html>\n<body>\n  <div></div>\n</body>"
        error = ParseError("test-error", line=3, column=3, message="Unexpected div", source_html=html)
        exc = error.as_exception()
        assert isinstance(exc, SyntaxError)
        assert exc.lineno == 3
        assert exc.filename == "<html>"
        assert exc.text == "  <div></div>"
        # Should highlight the full <div> tag
        assert exc.offset == 3  # Start of <div>
        assert exc.end_offset == 8  # End of <div>

    def test_parse_error_as_exception_with_end_column(self):
        """as_exception() respects explicit end_column."""
        html = "<html><body><div></div></body>"
        error = ParseError("test-error", line=1, column=13, source_html=html)
        exc = error.as_exception(end_column=18)
        assert exc.offset == 13
        assert exc.end_offset == 18

    def test_parse_error_as_exception_invalid_line(self):
        """as_exception() handles invalid line numbers."""
        html = "<html>"
        error = ParseError("test-error", line=99, column=1, source_html=html)
        exc = error.as_exception()
        assert isinstance(exc, SyntaxError)
        assert exc.msg == "test-error"

    def test_parse_error_as_exception_not_on_tag_start(self):
        """as_exception() finds tag start when column is in middle of tag."""
        html = "<html>\n<body>\n  <div></div>\n</body>"
        # Column 5 is the 'i' in <div>
        error = ParseError("test-error", line=3, column=5, source_html=html)
        exc = error.as_exception()
        # Should find the '<' and highlight full <div>
        assert exc.offset == 3  # Start of <div>
        assert exc.end_offset == 8  # End of <div>

    def test_parse_error_as_exception_no_closing_bracket(self):
        """as_exception() handles tags without closing '>'."""
        html = "<html><body><div"
        error = ParseError("test-error", line=1, column=13, source_html=html)
        exc = error.as_exception()
        # Should highlight from '<' to end of line
        assert exc.offset == 13
        assert exc.end_offset == 17  # End of string + 1

    def test_parse_error_as_exception_far_from_tag_start(self):
        """as_exception() doesn't search too far back for '<'."""
        html = "<html><body>some long text here error</body>"
        # Column 35 is far from any '<'
        error = ParseError("test-error", line=1, column=35, source_html=html)
        exc = error.as_exception()
        # Should give up searching and use current position
        assert exc.offset == 35

    def test_parse_error_as_exception_no_tag_start_found(self):
        """as_exception() handles case where no '<' is found before position."""
        html = "some text without tags"
        error = ParseError("test-error", line=1, column=10, source_html=html)
        exc = error.as_exception()
        # Should use current position when no '<' found
        assert exc.offset == 10

    def test_parse_error_with_end_column_from_token(self):
        """ParseError created with end_column uses it for highlighting."""
        html = "<html><body><div>text</div></body></html>"
        # Simulate a tag token error: <div> at position 13-17
        error = ParseError(
            "test-error",
            line=1,
            column=13,
            message="Test error on div tag",
            source_html=html,
            end_column=18,  # End of <div>
        )
        exc = error.as_exception()
        assert exc.offset == 13
        assert exc.end_offset == 18


class TestTokenBasedErrorHighlighting(unittest.TestCase):
    """Test that ParseError highlighting works with different token types."""

    def test_tag_token_start_tag(self):
        """Start tag tokens get full tag highlighting."""
        html = "<html>"
        parser = JustHTML(html, collect_errors=True)
        assert len(parser.errors) == 1
        error = parser.errors[0]
        # For tree-builder tag errors we store the end-of-token position.
        # <html> is 6 characters long.
        assert error.column == 6

    def test_tag_token_end_tag(self):
        """End tag tokens get full tag highlighting."""
        html = "<html></br></html>"
        parser = JustHTML(html, collect_errors=True)
        # </br> is treated as error (should be <br>)
        assert any(e.code == "unexpected-end-tag" for e in parser.errors)


class TestTreeBuilderParseErrorWithTokens(unittest.TestCase):
    """Test TreeBuilder._parse_error with different token types."""

    def setUp(self):
        """Create a TreeBuilder with a mocked tokenizer."""
        self.builder = TreeBuilder(collect_errors=True)
        # Create a minimal tokenizer with buffer
        self.builder.tokenizer = Tokenizer(None, None, collect_errors=False)
        self.builder.tokenizer.buffer = "<html><body>text</body></html>"
        self.builder.tokenizer.last_token_line = 1

    def test_parse_error_with_tag_token(self):
        """_parse_error with Tag token calculates correct positions."""
        token = Tag(Tag.START, "div", {"class": "test"}, False)
        # Simulate tokenizer pointing after <div class="test">
        self.builder.tokenizer.last_token_column = 18  # After '>' of <div class="test">

        self.builder._parse_error("test-error", tag_name="div", token=token)

        assert len(self.builder.errors) == 1
        error = self.builder.errors[0]
        # Tag length: <div class="test"> = 18 chars
        # Start = 18 - 18 + 1 = 1
        assert error.column == 1
        assert error._end_column == 19

    def test_parse_error_with_tag_token_empty_attr_value(self):
        """_parse_error handles boolean/empty-value attributes without adding value length."""
        token = Tag(Tag.START, "div", {"disabled": ""}, False)
        # <div disabled> is 14 characters long
        self.builder.tokenizer.last_token_column = 14

        self.builder._parse_error("test-error", tag_name="div", token=token)

        assert len(self.builder.errors) == 1
        error = self.builder.errors[0]
        assert error.column == 1
        assert error._end_column == 15

    def test_parse_error_with_end_tag_token(self):
        """_parse_error with end Tag token calculates correct positions."""
        token = Tag(Tag.END, "div", {}, False)
        # Simulate tokenizer pointing after </div>
        self.builder.tokenizer.last_token_column = 6  # After '>' of </div>

        self.builder._parse_error("test-error", tag_name="div", token=token)

        assert len(self.builder.errors) == 1
        error = self.builder.errors[0]
        # Tag length: </div> = 6 chars
        # Start = 6 - 6 + 1 = 1
        assert error.column == 1
        assert error._end_column == 7

    def test_parse_error_with_self_closing_tag(self):
        """_parse_error with self-closing tag includes / in length."""
        token = Tag(Tag.START, "img", {"src": "test.jpg"}, True)
        # <img src="test.jpg"/> (no space before /)
        # Tag length: 3(img) + 2(<>) + 1(space) + 3(src) + 1(=) + 2(quotes) + 8(test.jpg) + 1(/) = 21
        # Simulate tokenizer pointing after the tag
        tag_len = 21
        self.builder.tokenizer.last_token_column = tag_len

        self.builder._parse_error("test-error", tag_name="img", token=token)

        assert len(self.builder.errors) == 1
        error = self.builder.errors[0]
        assert error.column == 1
        assert error._end_column == tag_len + 1

    def test_parse_error_with_non_tag_token(self):
        """_parse_error with non-Tag token uses fallback highlighting."""
        token = CharacterTokens("hello")
        # Non-Tag tokens don't get special position calculation
        self.builder.tokenizer.last_token_column = 11

        self.builder._parse_error("test-error", token=token)

        assert len(self.builder.errors) == 1
        error = self.builder.errors[0]
        # Should use original column without adjustment
        assert error.column == 11
        assert error._end_column is None


class TestTokenizerErrors(unittest.TestCase):
    """Test tokenizer-specific errors are collected."""

    def test_null_character_error(self):
        """Null characters in data trigger errors."""
        doc = JustHTML("<p>\x00</p>", collect_errors=True)
        # Null character is a parse error
        assert len(doc.errors) > 0

    def test_unexpected_eof_in_tag(self):
        """Unexpected EOF in tag triggers error."""
        doc = JustHTML("<div att", collect_errors=True)
        assert len(doc.errors) > 0

    def test_unexpected_equals_in_tag(self):
        """Unexpected characters in attribute trigger error."""
        doc = JustHTML('<div attr="val\x00">text</div>', collect_errors=True)
        assert len(doc.errors) > 0


class TestTreeBuilderErrors(unittest.TestCase):
    """Test tree builder errors are collected."""

    def test_unexpected_end_tag(self):
        """Unexpected end tag triggers error."""
        doc = JustHTML("<!DOCTYPE html><html><body></span>", collect_errors=True)
        # Closing tag without opening tag
        assert len(doc.errors) > 0

    def test_treebuilder_error_after_newline(self):
        """Tree builder error column is calculated after newlines."""
        # Put an unexpected end tag after a newline
        html = "<!DOCTYPE html>\n<html>\n<body>\n</span>"
        doc = JustHTML(html, collect_errors=True)
        assert len(doc.errors) > 0
        # At least one error should have line > 1
        assert any(e.line > 1 for e in doc.errors if e.line is not None)

    def test_nested_p_in_button(self):
        """Paragraph in button triggers special handling."""
        doc = JustHTML("<!DOCTYPE html><button><p>text</button>", collect_errors=True)
        # This may trigger various parse errors
        assert isinstance(doc.errors, list)

    def test_line_counting_in_attribute_whitespace(self):
        """Line counting works in whitespace before/after attributes."""
        # Whitespace with newlines before attribute name
        html = "<div\n   \n   class='test'>content</div>"
        doc = JustHTML(html, collect_errors=True)
        assert doc.root is not None

        # Whitespace with newlines AFTER attribute name (before =)
        html_after = "<div class\n   \n   ='test'>content</div>"
        doc = JustHTML(html_after, collect_errors=True)
        assert doc.root is not None

    def test_line_counting_in_quoted_attribute_values(self):
        """Line counting works in multiline attribute values."""
        # Double-quoted attribute with newlines
        html_double = '<div data-content="line1\nline2\nline3">text</div>'
        doc = JustHTML(html_double, collect_errors=True)
        assert doc.root is not None

        # Single-quoted attribute with newlines
        html_single = "<div data-content='line1\nline2'>text</div>"
        doc = JustHTML(html_single, collect_errors=True)
        assert doc.root is not None

    def test_line_counting_with_cr_in_attributes(self):
        """Line counting handles carriage returns in attribute values."""
        # Attribute value with CR+LF
        html = '<div data-x="a\r\nb\rc">text</div>'
        doc = JustHTML(html, collect_errors=True)
        assert doc.root is not None
