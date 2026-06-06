import unittest

from justhtml import stream


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
            # Tokenizer does not emit end tags for void elements automatically
            ("start", ("hr", {})),
        ]
        assert events == expected

    def test_text_coalescing(self):
        # Tokenizer might emit multiple character tokens. Stream should coalesce.
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
