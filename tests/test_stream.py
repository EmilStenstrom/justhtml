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

    def test_unmatched_end_tag(self):
        html = "</div>"
        events = list(stream(html))
        expected = [("end", "div")]
        assert events == expected
