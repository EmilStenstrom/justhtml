import unittest

from justhtml import stream
from justhtml.core.text import ascii_lower
from justhtml.parser.scanner import (
    ascii_find,
    ascii_rfind,
    ascii_startswith,
    find_rawtext_end_tag,
    find_script_end_tag,
)
from justhtml.parser.stream import _StreamScanner


class TestStream(unittest.TestCase):
    def test_basic_stream(self):
        html = '<div class="container">Hello <b>World</b></div>'
        events = list(stream(html))

        expected = [
            ("start", ("div", {"class": "container"})),
            ("text", "Hello "),
            ("start", ("b", {})),
            ("text", "World"),
            ("end", "b"),
            ("end", "div"),
        ]
        assert events == expected

    def test_comments(self):
        html = "<!-- comment -->"
        events = list(stream(html))
        expected = [("comment", " comment ")]
        assert events == expected

    def test_doctype(self):
        html = "<!DOCTYPE html>"
        events = list(stream(html))
        # Doctype token structure: (name, public_id, system_id)
        expected = [("doctype", ("html", None, None))]
        assert events == expected

    def test_void_elements(self):
        html = "<br><hr>"
        events = list(stream(html))
        expected = [
            ("start", ("br", {})),
            # The event stream does not synthesize end tags for void elements.
            ("start", ("hr", {})),
        ]
        assert events == expected

    def test_text_coalescing(self):
        # The scanner may find adjacent text spans. Stream should coalesce.
        html = "abc"
        events = list(stream(html))
        expected = [("text", "abc")]
        assert events == expected

    def test_script_rawtext(self):
        html = "<script>console.log('<');</script>"
        events = list(stream(html))
        expected = [
            ("start", ("script", {})),
            ("text", "console.log('<');"),
            ("end", "script"),
        ]
        assert events == expected

    def test_ascii_lower_preserves_full_document_search_offsets_and_ascii_markers(self):
        # The parser searches this folded full-document copy for ASCII markup
        # markers, then reuses the match offsets against the original string.
        # It must only fold ASCII A-Z, not perform Unicode case folding.
        assert ascii_lower("café</SCRIPT>") == "café</script>"
        assert ascii_lower("İ</SCRIPT>") == "İ</script>"

        # U+212A lowercases to ASCII "k" under Python's Unicode lowercasing,
        # but HTML marker matching is only ASCII case-insensitive. Keep it as
        # text so future searches for ASCII markers containing "k" cannot
        # accidentally match non-ASCII content.
        assert ascii_lower("\u212a</SCRIPT>") == "\u212a</script>"

    def test_svg_cdata_streams_as_text(self):
        html = "<svg><![CDATA[x<y]]></svg>"
        events = list(stream(html))
        expected = [
            ("start", ("svg", {})),
            ("text", "x<y"),
            ("end", "svg"),
        ]
        assert events == expected

    def test_math_cdata_streams_as_text(self):
        html = "<math><![CDATA[x<y]]></math>"
        events = list(stream(html))
        expected = [
            ("start", ("math", {})),
            ("text", "x<y"),
            ("end", "math"),
        ]
        assert events == expected

    def test_svg_script_does_not_switch_to_rawtext(self):
        html = "<svg><script><b></script></svg>"
        events = list(stream(html))
        expected = [
            ("start", ("svg", {})),
            ("start", ("script", {})),
            ("start", ("b", {})),
            ("end", "script"),
            ("end", "svg"),
        ]
        assert events == expected

    def test_foreign_font_breakout_depends_on_attrs(self):
        html_breakout = "<svg><font color=x></font><script><b></script>"
        events_breakout = list(stream(html_breakout))
        assert events_breakout == [
            ("start", ("svg", {})),
            ("start", ("font", {"color": "x"})),
            ("end", "font"),
            ("start", ("script", {})),
            ("text", "<b>"),
            ("end", "script"),
        ]

        html_late_breakout_attr = "<svg><font data-x=1 size=2></font><script><b></script>"
        events_late_breakout_attr = list(stream(html_late_breakout_attr))
        assert events_late_breakout_attr == [
            ("start", ("svg", {})),
            ("start", ("font", {"data-x": "1", "size": "2"})),
            ("end", "font"),
            ("start", ("script", {})),
            ("text", "<b>"),
            ("end", "script"),
        ]

        html_foreign = "<svg><font></font><script><b></script>"
        events_foreign = list(stream(html_foreign))
        assert events_foreign == [
            ("start", ("svg", {})),
            ("start", ("font", {})),
            ("end", "font"),
            ("start", ("script", {})),
            ("start", ("b", {})),
            ("end", "script"),
        ]

    def test_self_closing_foreign_tags_do_not_leave_stream_in_foreign_context(self):
        events_svg = list(stream("<svg/><script><b></script>"))
        assert events_svg == [
            ("start", ("svg", {})),
            ("start", ("script", {})),
            ("text", "<b>"),
            ("end", "script"),
        ]

        events_math = list(stream("<math/><script><b></script>"))
        assert events_math == [
            ("start", ("math", {})),
            ("start", ("script", {})),
            ("text", "<b>"),
            ("end", "script"),
        ]

    def test_foreign_end_tags_pop_matching_stream_context(self):
        html = "<svg><g></svg><script><b></script>"
        events = list(stream(html))
        assert events == [
            ("start", ("svg", {})),
            ("start", ("g", {})),
            ("end", "svg"),
            ("start", ("script", {})),
            ("text", "<b>"),
            ("end", "script"),
        ]

    def test_stream_integration_points_use_html_text_parsing(self):
        html = "<svg><foreignObject><textarea><b></textarea></foreignObject></svg>"
        events = list(stream(html))
        assert events == [
            ("start", ("svg", {})),
            ("start", ("foreignobject", {})),
            ("start", ("textarea", {})),
            ("text", "<b>"),
            ("end", "textarea"),
            ("end", "foreignobject"),
            ("end", "svg"),
        ]

        html_math = "<math><mi><style><b></style></mi></math>"
        events_math = list(stream(html_math))
        assert events_math == [
            ("start", ("math", {})),
            ("start", ("mi", {})),
            ("start", ("style", {})),
            ("text", "<b>"),
            ("end", "style"),
            ("end", "mi"),
            ("end", "math"),
        ]

        html_annotation = '<math><annotation-xml encoding="text/html"><script><b></script></annotation-xml></math>'
        events_annotation = list(stream(html_annotation))
        assert events_annotation == [
            ("start", ("math", {})),
            ("start", ("annotation-xml", {"encoding": "text/html"})),
            ("start", ("script", {})),
            ("text", "<b>"),
            ("end", "script"),
            ("end", "annotation-xml"),
            ("end", "math"),
        ]

        html_annotation_late_encoding = (
            '<math><annotation-xml data-x=1 encoding="application/xhtml+xml">'
            "<script><b></script></annotation-xml></math>"
        )
        events_annotation_late_encoding = list(stream(html_annotation_late_encoding))
        assert events_annotation_late_encoding == [
            ("start", ("math", {})),
            ("start", ("annotation-xml", {"data-x": "1", "encoding": "application/xhtml+xml"})),
            ("start", ("script", {})),
            ("text", "<b>"),
            ("end", "script"),
            ("end", "annotation-xml"),
            ("end", "math"),
        ]

    def test_stream_annotation_xml_without_html_encoding_keeps_foreign_context(self):
        html = "<math><annotation-xml><script><b></script></annotation-xml></math>"
        events = list(stream(html))
        assert events == [
            ("start", ("math", {})),
            ("start", ("annotation-xml", {})),
            ("start", ("script", {})),
            ("start", ("b", {})),
            ("end", "script"),
            ("end", "annotation-xml"),
            ("end", "math"),
        ]

    def test_stream_annotation_xml_allows_svg_without_html_encoding(self):
        html = "<math><annotation-xml><svg><![CDATA[x<y]]></svg></annotation-xml></math>"
        events = list(stream(html))
        assert events == [
            ("start", ("math", {})),
            ("start", ("annotation-xml", {})),
            ("start", ("svg", {})),
            ("text", "x<y"),
            ("end", "svg"),
            ("end", "annotation-xml"),
            ("end", "math"),
        ]

    def test_stream_foreign_p_end_tag_breaks_out_of_foreign_context(self):
        html = "<svg><g></p><script><b></script>"
        events = list(stream(html))
        assert events == [
            ("start", ("svg", {})),
            ("start", ("g", {})),
            ("end", "p"),
            ("start", ("script", {})),
            ("text", "<b>"),
            ("end", "script"),
        ]

    def test_stream_unmatched_foreign_end_tag_pops_one_context(self):
        html = "<svg><g></foo><script><b></script>"
        events = list(stream(html))
        assert events == [
            ("start", ("svg", {})),
            ("start", ("g", {})),
            ("end", "foo"),
            ("start", ("script", {})),
            ("start", ("b", {})),
            ("end", "script"),
        ]

    def test_stream_unmatched_html_end_tag_with_stack_pops_one_context(self):
        html = "<div></span><script><b></script>"
        events = list(stream(html))
        assert events == [
            ("start", ("div", {})),
            ("end", "span"),
            ("start", ("script", {})),
            ("text", "<b>"),
            ("end", "script"),
        ]

    def test_unmatched_end_tag(self):
        html = "</div>"
        events = list(stream(html))
        expected = [("end", "div")]
        assert events == expected

    def test_unmatched_end_tag_stops_at_nearer_html_element(self):
        # </foo> targets the outer <foo>, but the nearer <bar> (also HTML
        # namespace) isn't a match, so only the innermost open element is
        # popped and <foo>/<bar> remain unmatched.
        html = "<foo><bar><svg></svg></foo>"
        events = list(stream(html))
        assert events == [
            ("start", ("foo", {})),
            ("start", ("bar", {})),
            ("start", ("svg", {})),
            ("end", "svg"),
            ("end", "foo"),
        ]

    def test_trailing_less_than_and_non_tag_openers_are_text(self):
        assert list(stream("abc<")) == [("text", "abc<")]
        assert list(stream("<1x")) == [("text", "<1x")]

    def test_bogus_end_tags_and_processing_instructions_are_comments(self):
        assert list(stream("</>")) == []
        assert list(stream("</!x>")) == [("comment", "!x")]
        assert list(stream("</!x")) == [("comment", "!x")]
        assert list(stream("<?x>")) == [("comment", "?x")]
        assert list(stream("<?x")) == [("comment", "?x")]

    def test_unclosed_tags_are_ignored(self):
        assert list(stream("<div")) == []
        assert list(stream("</div")) == []
        assert list(stream("<x q='unterminated")) == []

    def test_plaintext_and_unclosed_rawtext(self):
        assert list(stream("<plaintext>a<b>")) == [
            ("start", ("plaintext", {})),
            ("text", "a<b>"),
        ]
        assert list(stream("<style>x")) == [
            ("start", ("style", {})),
            ("text", "x"),
        ]
        assert list(stream("<style>x</style")) == [
            ("start", ("style", {})),
            ("text", "x</style"),
        ]

    def test_comment_edge_cases(self):
        assert list(stream("<!-->")) == [("comment", "")]
        assert list(stream("<!--->")) == [("comment", "")]
        assert list(stream("<!--x--!>")) == [("comment", "x")]
        assert list(stream("<!--x--")) == [("comment", "x")]
        assert list(stream("<!--x\0-->")) == [("comment", "x\ufffd")]

    def test_doctype_edge_cases(self):
        assert list(stream("<!DOCTYPE")) == [("doctype", (None, None, None))]
        assert list(stream("<!DOCTYPE >")) == [("doctype", (None, None, None))]
        assert list(stream("<!DoCtYpE HTML>")) == [("doctype", ("html", None, None))]

    def test_doctype_identifiers_and_malformed_quotes(self):
        assert list(stream('<!DOCTYPE html SYSTEM "x">Hello')) == [
            ("doctype", ("html", None, "x")),
            ("text", "Hello"),
        ]
        assert list(stream('<!DOCTYPE html PUBLIC "x" "y">Hello')) == [
            ("doctype", ("html", "x", "y")),
            ("text", "Hello"),
        ]
        assert list(stream('<!DOCTYPE potato taco "ddd>Hello')) == [
            ("doctype", ("potato", None, None)),
            ("text", "Hello"),
        ]
        assert list(stream("<!DOCTYPE potato PUBLIC 'go'of'>Hello")) == [
            ("doctype", ("potato", "go", None)),
            ("text", "Hello"),
        ]
        assert list(stream("<!DOCTYPE h\0 PUBLIC 'p\0' 's\0'>")) == [
            ("doctype", ("h\ufffd", "p\ufffd", "s\ufffd")),
        ]

    def test_null_handling_matches_stream_compatibility(self):
        assert list(stream("a\0b")) == [("text", "a\0b")]
        assert list(stream("<svg><![CDATA[a\0b]]></svg>")) == [
            ("start", ("svg", {})),
            ("text", "a\0b"),
            ("end", "svg"),
        ]
        assert list(stream("<style>a\0b</style>")) == [
            ("start", ("style", {})),
            ("text", "a\ufffdb"),
            ("end", "style"),
        ]
        assert list(stream("<!--a\0b-->")) == [("comment", "a\ufffdb")]

    def test_unclosed_cdata_and_bogus_declarations(self):
        assert list(stream("<svg><![CDATA[x")) == [
            ("start", ("svg", {})),
            ("text", "x"),
        ]
        assert list(stream("<!foo")) == [("comment", "foo")]
        assert list(stream("<!foo>")) == [("comment", "foo")]

    def test_attribute_edge_cases(self):
        assert list(stream("<DIV A\0B=x\0&amp; A=dup>")) == [
            ("start", ("div", {"a\ufffdb": "x\ufffd&", "a": "dup"})),
        ]
        assert list(stream('<x a = unquoted b="v&amp;">')) == [
            ("start", ("x", {"a": "unquoted", "b": "v&"})),
        ]
        assert list(stream("<x / >")) == [("start", ("x", {}))]
        assert list(stream("<x/>")) == [("start", ("x", {}))]
        assert list(stream("<x a=first a=second>")) == [
            ("start", ("x", {"a": "first"})),
        ]

    def test_bytes_input(self):
        assert list(stream(b"<p>x</p>")) == [
            ("start", ("p", {})),
            ("text", "x"),
            ("end", "p"),
        ]

    def test_scanner_defensive_helpers(self):
        empty = _StreamScanner("")
        assert empty._parse_start_tag(0, 0) is None
        assert empty._parse_end_tag(0, 0) is None
        assert empty._in_foreign_context() is False

        spaces = _StreamScanner(" ")
        assert spaces._parse_attrs(0, 1) == ({}, False, 1, False)
        bare_attribute = _StreamScanner("a")
        assert bare_attribute._parse_attrs(0, 1) == ({"a": ""}, False, 1, False)

        text_buffer: list[str] = []
        empty._append_text(text_buffer, "")
        empty._append_text(text_buffer, "\r\n\0&amp;")
        assert text_buffer == ["\n\0&"]
        assert list(empty._flush_text([""])) == []

    def test_ascii_marker_searches_fold_only_ascii_candidates(self):
        html = "<sty\u212ae><STYLE>x</style>"

        assert ascii_startswith(html, "<style", 7, len(html)) is True
        assert ascii_startswith(html, "<style", 0, len(html)) is False
        assert ascii_startswith("DOCTYPE", "doctype", 0, 3) is False
        assert ascii_find(html, "<style", 0, len(html)) == 7
        assert ascii_find(html, "<script", 0, len(html)) == -1
        assert ascii_rfind(html, "</style", 0, len(html)) == 15
        assert ascii_rfind(html, "</script", 0, len(html)) == -1
        assert ascii_startswith("doctype", "DOCTYPE", 0, 7) is True
        assert ascii_find("xDOCTYPEdoctype", "doctype", 0, 15) == 1
        assert ascii_find("DOCTYPE", "doctype", 0, 7) == 0
        assert ascii_find("doctypeDOCTYPE", "doctype", 0, 14) == 0
        assert ascii_find("doctype", "DOCTYPE", 0, 7) == 0
        assert ascii_rfind("DOCTYPEdoctype", "DOCTYPE", 0, 14) == 7
        assert ascii_rfind("DOCTYPE", "doctype", 0, 7) == 0
        assert ascii_rfind("DOCTYPE", "DOCTYPE", 0, 7) == 0

    def test_script_scanner_rejects_nested_invalid_end_markers(self):
        invalid_inner = "<!--<script X </scriptx </script>"
        assert find_script_end_tag(invalid_inner, 0, len(invalid_inner)) == (
            None,
            len(invalid_inner),
        )

        invalid_quoted = '<!--<script X </script foo="</scriptx">tail</script>'
        assert find_script_end_tag(invalid_quoted, 0, len(invalid_quoted)) == (43, 52)

        # A terminated inner end marker inside the outer end tag's quoted
        # attributes leaves the double-escaped state before that inner marker.
        valid_quoted = '<!--<script X </script foo="</script>">tail</script>'
        assert find_script_end_tag(valid_quoted, 0, len(valid_quoted)) == (28, 37)

        # A "<script" immediately followed by "</script>" is not a
        # script-data-double-escape-start (its following character "<" is not a
        # terminator), so the trailing "</script>" must still close the element.
        unterminated_marker = "<!--<script</script>"
        assert find_script_end_tag(unterminated_marker, 0, len(unterminated_marker)) == (
            11,
            20,
        )

        # "<!-->" closes the escaped comment immediately (the "<!--" dashes count
        # toward the "-->"), so the following "<script>" is plain script data and
        # does not start a double escape; the "</script>" still closes it.
        short_comment = "<!--><script></script>"
        assert find_script_end_tag(short_comment, 0, len(short_comment)) == (13, 22)

    def test_rawtext_end_tag_slash_terminated_without_gt(self):
        # "</style/x" terminates the end-tag name with "/", so at EOF the end tag
        # is emitted (attributes dropped) and the raw text ends, even though no
        # ">" ever appears.
        slashed = "</style/x"
        assert find_rawtext_end_tag(slashed, "style", 0, len(slashed)) == (0, len(slashed))
        # A bare "</style" at EOF with no terminator stays raw text.
        bare = "</style"
        assert find_rawtext_end_tag(bare, "style", 0, len(bare)) == (None, len(bare))

    def test_rawtext_end_tag_after_length_changing_case_char(self):
        # "İ" (U+0130) lowers to two characters, so a str.lower() copy of the
        # input drifts out of index alignment with the original. The rawtext
        # scan must keep finding the real end tag after such a character.
        assert list(stream("<div>İ</div><script>y</script>0")) == [
            ("start", ("div", {})),
            ("text", "İ"),
            ("end", "div"),
            ("start", ("script", {})),
            ("text", "y"),
            ("end", "script"),
            ("text", "0"),
        ]
        assert list(stream('<script>var x = "İ";</script>0')) == [
            ("start", ("script", {})),
            ("text", 'var x = "İ";'),
            ("end", "script"),
            ("text", "0"),
        ]

    def test_doctype_after_length_changing_case_char(self):
        assert list(stream("İ<!DOCTYPE html>")) == [
            ("text", "İ"),
            ("doctype", ("html", None, None)),
        ]

    def test_foreign_cdata_after_length_changing_case_char(self):
        assert list(stream("<svg>İ<![CDATA[x]]></svg>")) == [
            ("start", ("svg", {})),
            ("text", "İ"),
            ("text", "x"),
            ("end", "svg"),
        ]
