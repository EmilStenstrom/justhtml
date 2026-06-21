import unittest
from dataclasses import replace
from pathlib import Path

from justhtml import JustHTML
from justhtml.dom import Element
from justhtml.parser.context import FragmentContext
from justhtml.parser.engine import ParseEngine, compile_default_engine_plan, compile_raw_engine_plan
from justhtml.parser.options import ParserOptions
from justhtml.sanitizer import DEFAULT_DOCUMENT_POLICY, DEFAULT_POLICY, UrlPolicy
from tests.harness.tree import TestRunner


class TestParserEngineIntegrationCoverage(unittest.TestCase):
    def assert_parses_to(self, html: str, expected: str, **kwargs: object) -> JustHTML:
        document = JustHTML(html, **kwargs)
        assert document.to_html(pretty=False) == expected
        return document

    def test_malformed_markup_and_document_shell(self) -> None:
        cases = [
            ("abc<", "<html><head></head><body>abc&lt;</body></html>"),
            ("a</!x>b", "<html><head></head><body>ab</body></html>"),
            ("a<?x>b", "<html><head></head><body>ab</body></html>"),
            ("a<!x>b", "<html><head></head><body>ab</body></html>"),
            ("a<div", "<html><head></head><body>a</body></html>"),
            ("<p>a</p", "<html><head></head><body><p>a</p></body></html>"),
            ("<DIV CLASS=x>A</DIV>", '<html><head></head><body><div class="x">A</div></body></html>'),
            (
                "<!doctype html><html x=1><head><title>x</title></head><body y=2>z</body></html><!--t-->",
                "<!DOCTYPE html><html><head><title>x</title></head><body>z</body></html>",
            ),
            ("<html><body>x</body><!--c--></html><!--d-->", "<html><head></head><body>x</body></html>"),
        ]
        for html, expected in cases:
            with self.subTest(html=html):
                self.assert_parses_to(html, expected)

    def test_frameset_table_and_template_recovery(self) -> None:
        cases = [
            (
                "<!doctype html><html><head></head><frameset><frame src=x></frameset></html>",
                "<!DOCTYPE html><html><head></head></html>",
            ),
            (
                "<!doctype html><frameset> x <noframes><p>fallback</p></noframes></frameset>",
                "<!DOCTYPE html><html><head></head>  &lt;p&gt;fallback&lt;/p&gt;</html>",
            ),
            (
                "<table>text<tr><td>A<td>B</table>tail",
                "<html><head></head><body>text<table><tbody><tr><td>A</td><td>B</td></tr></tbody></table>"
                "tail</body></html>",
            ),
            (
                "<table><caption>x<table><tr><td>y</table>z",
                "<html><head></head><body><table><caption>x"
                "<table><tbody><tr><td>y</td></tr></tbody></table>z"
                "</caption></table></body></html>",
            ),
            (
                "<table><form><input type=hidden><tr><td>x</form></table>",
                "<html><head></head><body><table><tbody><tr><td>x</td></tr></tbody></table></body></html>",
            ),
            (
                "<template><table><tr><td>x</template>y",
                "<html><head><table><tbody><tr><td>x</td></tr></tbody></table></head><body>y</body></html>",
            ),
            ("<template><select><option>x</template>y", "<html><head>x</head><body>y</body></html>"),
        ]
        for html, expected in cases:
            with self.subTest(html=html):
                self.assert_parses_to(html, expected)

    def test_formatting_lists_select_and_foreign_content(self) -> None:
        cases = [
            (
                "<p><b><i>x</b>y</i>z",
                "<html><head></head><body><p><b><i>x</i></b><i>y</i>z</p></body></html>",
            ),
            (
                "<table><b><tr><td>x</b></table>",
                "<html><head></head><body><b></b><table><tbody><tr><td>x</td></tr></tbody></table></body></html>",
            ),
            (
                "<ul><li>a<li>b</ul><dl><dt>x<dd>y</dl>",
                "<html><head></head><body><ul><li>a</li><li>b</li></ul>xy</body></html>",
            ),
            ("<select><option>a<optgroup><option selected>b</select>", "<html><head></head><body>ab</body></html>"),
            ("<select><div>x</div><input><hr></select>", "<html><head></head><body><div>x</div><hr></body></html>"),
            (
                "<svg><g><foreignObject><p>x</p></foreignObject><font color=red>y</font></svg>",
                "<html><head></head><body>y</body></html>",
            ),
            ("<svg><![CDATA[x<y]]><title><b>z</title></svg>", "<html><head></head><body></body></html>"),
            (
                "<math><mi><b>x</b></mi><annotation-xml encoding=text/html><p>y</p></annotation-xml></math>",
                "<html><head></head><body></body></html>",
            ),
        ]
        for html, expected in cases:
            with self.subTest(html=html):
                self.assert_parses_to(html, expected)

    def test_text_modes_and_projected_attributes(self) -> None:
        cases = [
            (
                "<style>a<b&c</style><script><!--<script>x</script>--></script>",
                "<html><head></head><body></body></html>",
            ),
            ("<plaintext>a<b>&amp;", "<html><head></head><body>a&lt;b&gt;&amp;amp;</body></html>"),
            ("<textarea>\n&amp;<b></textarea>", "<html><head></head><body>&amp;&lt;b&gt;</body></html>"),
            ('<div A=1 a=2 bad="unterminated', "<html><head></head><body></body></html>"),
            (
                '<a href="//example.com" title=x onclick=y>x</a><img src="https://x">',
                '<html><head></head><body><a href="https://example.com" title="x">x</a><img></body></html>',
            ),
        ]
        for html, expected in cases:
            with self.subTest(html=html):
                self.assert_parses_to(html, expected)

    def test_fragment_context_modes(self) -> None:
        cases = [
            ("<tr><td>x<td>y", FragmentContext("tbody"), "<tr><td>x</td><td>y</td></tr>"),
            ("<option>a<optgroup><option>b", FragmentContext("select"), "ab"),
            ("<tr><td>x", FragmentContext("template"), "x"),
            ("<![CDATA[x<y]]><g/>", FragmentContext("svg", "svg"), ""),
            ("<head><title>x</title><body>y", FragmentContext("html"), "xy"),
        ]
        for html, context, expected in cases:
            with self.subTest(html=html, context=context):
                self.assert_parses_to(html, expected, fragment_context=context)

    def test_fragment_unwraps_disallowed_template_contents(self) -> None:
        self.assert_parses_to("<template>x</template>", "x", fragment=True)

    def test_raw_div_fragment_ignores_head_wrapper(self) -> None:
        self.assert_parses_to(
            "<head><title>x</title></head>",
            "<title>x</title>",
            fragment=True,
            sanitize=False,
        )

    def test_diagnostic_location_and_xml_modes(self) -> None:
        located = self.assert_parses_to(
            "<p a=1>x</p>",
            '<html><head></head><body><p a="1">x</p></body></html>',
            sanitize=False,
            track_node_locations=True,
        )
        paragraph = located.query("p")[0]
        assert paragraph.origin_location == (1, 1)

        diagnosed = self.assert_parses_to(
            "<!doctype><p>\0</x>",
            "<!DOCTYPE><html><head></head><body><p></p></body></html>",
            sanitize=False,
            collect_errors=True,
        )
        assert [error.code for error in diagnosed.errors] == [
            "unknown-doctype",
            "unexpected-null-character",
            "unexpected-end-tag",
        ]

        self.assert_parses_to(
            "<!--a--b--><p>\0\x01</p>",
            "<!--a- -b--><html><head></head><body><p>\x01</p></body></html>",
            sanitize=False,
            _parser_opts=ParserOptions(xml_coercion=True),
        )

    def test_scripting_disabled_head_noscript(self) -> None:
        self.assert_parses_to(
            "<head><noscript><link href=x><p>y</noscript></head>",
            "<html><head></head><body><p>y</p></body></html>",
            scripting_enabled=False,
        )

    def test_compiled_safe_start_and_end_recovery(self) -> None:
        cases = [
            ("</", "<html><head></head><body>&lt;/</body></html>"),
            ("x</!y", "<html><head></head><body>x</body></html>"),
            ("<DIV>x</DIV>", "<html><head></head><body><div>x</div></body></html>"),
            ("</p>", "<html><head></head><body></body></html>"),
            ("<div></body>x", "<html><head></head><body><div>x</div></body></html>"),
            (
                "<table><tr><td>x</body>y",
                "<html><head></head><body><table><tbody><tr><td>xy</td></tr></tbody></table></body></html>",
            ),
            ("<head></head><title>x</title>", "<html><head><title>x</title></head><body></body></html>"),
            (
                "<table><colgroup></colgroup><tr><td>x",
                "<html><head></head><body><table><tbody><tr><td>x</td></tr></tbody></table></body></html>",
            ),
            ("</h1>x", "<html><head></head><body>x</body></html>"),
            ("x</p>y", "<html><head></head><body>x<p></p>y</body></html>"),
            ("<audio><span></audio>x", "<html><head></head><body><span></span>x</body></html>"),
            ("<head></br>x", "<html><head></head><body><br>x</body></html>"),
            ("<head></div><title>x</title>", "<html><head><title>x</title></head><body></body></html>"),
            ("<image src=x>", '<html><head></head><body><img src="x"></body></html>'),
            ("<table><colgroup><col><div>x", "<html><head></head><body><table><div>x</div></table></body></html>"),
            (
                "<!doctype html><frameset></frameset><html x=1>",
                "<!DOCTYPE html><html><head></head></html>",
            ),
            ("<head><div>x", "<html><head></head><body><div>x</div></body></html>"),
            ("<head></head><div>x", "<html><head></head><body><div>x</div></body></html>"),
            ("<div> </div><frameset><frame></frameset>", "<html><head></head></html>"),
            ("<p>x</p><frameset>", "<html><head></head><body><p>x</p></body></html>"),
            ("<frameset><noframes>a<b></noframes></frameset>", "<html><head></head>a&lt;b&gt;</html>"),
            ("<frame><p>x", "<html><head></head><body><p>x</p></body></html>"),
            ("<select><option>x<select><p>y", "<html><head></head><body>x<p>y</p></body></html>"),
            ("<xmp>a<b>c</xmp>", "<html><head></head><body>a&lt;b&gt;c</body></html>"),
            ("<button>a<button>b", "<html><head></head><body>ab</body></html>"),
            (
                "<table><table><tr><td>x",
                "<html><head></head><body><table></table><table><tbody><tr><td>x</td></tr></tbody></table></body></html>",
            ),
            ("<div><caption>x", "<html><head></head><body><div>x</div></body></html>"),
            ("<div><colgroup><col>x", "<html><head></head><body><div>x</div></body></html>"),
            ("<div><tbody><tr><td>x", "<html><head></head><body><div>x</div></body></html>"),
            ("<div><tr><td>x", "<html><head></head><body><div>x</div></body></html>"),
            ("<div><td>x", "<html><head></head><body><div>x</div></body></html>"),
            ("<pre>\nx</pre>", "<html><head></head><body><pre>x</pre></body></html>"),
            ("<b><menuitem>x</b>", "<html><head></head><body><b>x</b></body></html>"),
            (
                "<table><b>x<div>y</div></b></table>",
                "<html><head></head><body><b>x<div>y</div></b><table></table></body></html>",
            ),
        ]
        for html, expected in cases:
            with self.subTest(html=html):
                self.assert_parses_to(html, expected)

    def test_additional_head_noscript_recovery(self) -> None:
        cases = [
            ("<head><noscript></noscript><title>x</title>", "<html><head></head><body><title>x</title></body></html>"),
            ("<head><noscript></div>x</noscript>", "<html><head></head><body>x</body></html>"),
            ("<head><noscript></br>x", "<html><head></head><body><br>x</body></html>"),
        ]
        for html, expected in cases:
            with self.subTest(html=html):
                self.assert_parses_to(html, expected, scripting_enabled=False)

    def test_additional_fragment_recovery(self) -> None:
        cases = [
            ("<div>x</body>y", FragmentContext("html"), "<div>x</div>y"),
            ("x</p>y", FragmentContext("div"), "x<p></p>y"),
            ("<col><div>x", FragmentContext("colgroup"), ""),
            ("<colgroup><col><caption>x", FragmentContext("table"), "<caption>x</caption>"),
            ("<tr><td>x<table><tr><td>y", FragmentContext("caption"), "xy<table></table>"),
            (
                "<td>x<tr><td>y<table><tr><td>z",
                FragmentContext("tbody"),
                "<tr><td>x</td></tr><tr><td>y<table><tbody><tr><td>z</td></tr></tbody></table></td></tr>",
            ),
            ("<td>x<tr><td>y", FragmentContext("tr"), "<td>x</td><td>y</td>"),
            ("<frameset><frame></frameset>", FragmentContext("html"), ""),
        ]
        for html, context, expected in cases:
            with self.subTest(html=html, context=context):
                self.assert_parses_to(html, expected, fragment_context=context)

    def test_url_projection_variants(self) -> None:
        anchor_cases = [
            ("", "<a>x</a>"),
            ("\\bad", "<a>x</a>"),
            ("#frag", '<a href="#frag">x</a>'),
            ("//example.com", '<a href="https://example.com">x</a>'),
            ("relative/path", '<a href="relative/path">x</a>'),
            ("http://x", '<a href="http://x">x</a>'),
            ("https:foo", '<a href="https:foo">x</a>'),
            ("mailto:a@b", '<a href="mailto:a@b">x</a>'),
            ("javascript:alert(1)", "<a>x</a>"),
            ("\x01https://x", "<a>x</a>"),
            ("https://é.example", '<a href="https://é.example">x</a>'),
        ]
        for value, expected_anchor in anchor_cases:
            with self.subTest(tag="a", value=value):
                self.assert_parses_to(
                    f'<a href="{value}">x</a>',
                    f"<html><head></head><body>{expected_anchor}</body></html>",
                )

        image_cases = [
            ("", "<img>"),
            ("\\bad", "<img>"),
            ("#frag", '<img src="#frag">'),
            ("//example.com", "<img>"),
            ("relative/path", '<img src="relative/path">'),
            ("http://x", "<img>"),
            ("https:foo", "<img>"),
            ("mailto:a@b", "<img>"),
            ("javascript:alert(1)", "<img>"),
            ("\x01https://x", "<img>"),
            ("https://é.example", "<img>"),
        ]
        for value, expected_image in image_cases:
            with self.subTest(tag="img", value=value):
                self.assert_parses_to(
                    f'<img src="{value}">',
                    f"<html><head></head><body>{expected_image}</body></html>",
                )

    def test_policy_planner_fallbacks(self) -> None:
        base = DEFAULT_DOCUMENT_POLICY
        source = '<!--c--><custom><p onclick=x href=https://x style="color:red">x</p></custom>'
        cases = [
            (replace(base, unsafe_handling="collect"), "<html><head></head><body><p>x</p></body></html>"),
            (
                replace(base, disallowed_tag_handling="escape"),
                "<html><head></head><body>&lt;!--c--&gt;&lt;custom&gt;<p>x</p>&lt;/custom&gt;</body></html>",
            ),
            (
                replace(base, allowed_tags=frozenset(tag for tag in base.allowed_tags if tag != "body")),
                "<html><head></head><p>x</p></html>",
            ),
            (replace(base, drop_comments=False), "<!--c--><html><head></head><body><p>x</p></body></html>"),
            (
                replace(base, force_link_rel=frozenset({"noopener"})),
                "<html><head></head><body><p>x</p></body></html>",
            ),
            (
                replace(
                    base,
                    allowed_tags=base.allowed_tags | {"style"},
                    allowed_css_properties=frozenset({"color"}),
                ),
                "<html><head></head><body><p>x</p></body></html>",
            ),
            (
                replace(base, allowed_tags=base.allowed_tags | {"svg"}),
                "<html><head></head><body><p>x</p></body></html>",
            ),
            (
                replace(base, allowed_tags=base.allowed_tags | {"custom"}),
                "<html><head></head><body><custom><p>x</p></custom></body></html>",
            ),
            (
                replace(base, allowed_tags=frozenset(tag for tag in base.allowed_tags if tag != "template")),
                "<html><head></head><body><p>x</p></body></html>",
            ),
            (
                replace(base, allowed_attributes={**base.allowed_attributes, "custom": {"class"}}),
                "<html><head></head><body><p>x</p></body></html>",
            ),
            (
                replace(
                    base,
                    allowed_attributes={
                        **base.allowed_attributes,
                        "p": set(base.allowed_attributes["*"]) | {"onclick"},
                    },
                ),
                '<html><head></head><body><p onclick="x">x</p></body></html>',
            ),
            (
                replace(
                    base,
                    allowed_attributes={
                        **base.allowed_attributes,
                        "p": set(base.allowed_attributes["*"]) | {"href"},
                    },
                ),
                "<html><head></head><body><p>x</p></body></html>",
            ),
            (
                replace(
                    base,
                    allowed_attributes={
                        **base.allowed_attributes,
                        "p": set(base.allowed_attributes["*"]) | {"style"},
                    },
                    allowed_css_properties=frozenset({"color"}),
                ),
                '<html><head></head><body><p style="color: red">x</p></body></html>',
            ),
        ]
        for policy, expected in cases:
            with self.subTest(policy=policy):
                self.assert_parses_to(source, expected, policy=policy)

        fragment_policy = replace(
            DEFAULT_POLICY,
            allowed_tags=frozenset(tag for tag in DEFAULT_POLICY.allowed_tags if tag != "template"),
        )
        self.assert_parses_to("<p>x</p>", "<p>x</p>", fragment=True, policy=fragment_policy)

    def test_raw_text_and_fragment_modes(self) -> None:
        raw_cases = [
            ("<", "<html><head></head><body>&lt;</body></html>"),
            ("</", "<html><head></head><body>&lt;/</body></html>"),
            ("<div", "<html><head></head><body>&lt;div</body></html>"),
            (
                "<style>a\r\nb\fc\0</style>",
                "<html><head><style>a\nb\fc�</style></head><body></body></html>",
            ),
            (
                "<plaintext>a\r\nb\fc\0<q>",
                "<html><head></head><body><plaintext>a\nb\fc�<q></plaintext></body></html>",
            ),
            (
                "<script><!--<script>x</script>--></script>",
                "<html><head><script><!--<script>x&lt;/script>--></script></head><body></body></html>",
            ),
            (
                "<html a=1><head b=2></head><body c=3>x</body></html><p>y",
                '<html a="1"><head b="2"></head><body c="3">x<p>y</p></body></html>',
            ),
            ("<svg><g></svg><p>x", "<html><head></head><body><svg><g></g></svg><p>x</p></body></html>"),
            ("<math><mi></math><p>x", "<html><head></head><body><math><mi></mi></math><p>x</p></body></html>"),
            (
                "<frameset><frame></frameset>",
                "<html><head></head><frameset><frame></frame></frameset></html>",
            ),
        ]
        for html, expected in raw_cases:
            with self.subTest(html=html):
                self.assert_parses_to(html, expected, sanitize=False, track_node_locations=True)

        fragment_cases = [
            ("x</textarea>y", FragmentContext("textarea"), "x&lt;/textarea&gt;y"),
            ("a<b", FragmentContext("plaintext"), "a&lt;b"),
            ("a</style>b", FragmentContext("style"), "a&lt;/style&gt;b"),
            ("<frame>", FragmentContext("frameset"), "<frame></frame>"),
            ("<input><option>x", FragmentContext("select"), "<option>x</option>"),
        ]
        for html, context, expected in fragment_cases:
            with self.subTest(html=html, context=context):
                self.assert_parses_to(html, expected, sanitize=False, fragment_context=context)

    def test_raw_xml_text_modes(self) -> None:
        options = ParserOptions(xml_coercion=True)
        cases = [
            (
                "<style>a\r\nb\fc\0</style>",
                "<html><head><style>a\nb c�</style></head><body></body></html>",
            ),
            (
                "<plaintext>a\r\nb\fc\0",
                "<html><head></head><body><plaintext>a\nb c�</plaintext></body></html>",
            ),
        ]
        for html, expected in cases:
            with self.subTest(html=html):
                self.assert_parses_to(html, expected, sanitize=False, _parser_opts=options)

    def test_untracked_raw_parser_modes(self) -> None:
        cases = [
            ("<?x>", "<!--?x--><html><head></head><body></body></html>"),
            ("<?x", "<!--?x--><html><head></head><body></body></html>"),
            ("</!x>", "<!--!x--><html><head></head><body></body></html>"),
            ("</!x", "<!--!x--><html><head></head><body></body></html>"),
            ("<!x>", "<!--x--><html><head></head><body></body></html>"),
            ("<!x", "<!--x--><html><head></head><body></body></html>"),
            ("<!--x", "<!--x--><html><head></head><body></body></html>"),
            ("<!--x--!>", "<!--x--><html><head></head><body></body></html>"),
            ("<script>x", "<html><head><script>x</script></head><body></body></html>"),
            ("<style>x", "<html><head><style>x</style></head><body></body></html>"),
            ("<body><script>x", "<html><head></head><body><script>x</script></body></html>"),
            ("<body><style>x", "<html><head></head><body><style>x</style></body></html>"),
            ("<xmp>x", "<html><head></head><body><xmp>x</xmp></body></html>"),
            ("<iframe>x", "<html><head></head><body><iframe>x</iframe></body></html>"),
            ("<noembed>x", "<html><head></head><body><noembed>x</noembed></body></html>"),
            ("<noframes>x", "<html><head><noframes>x</noframes></head><body></body></html>"),
            (
                "<table><form><input>x",
                "<html><head></head><body><input>x<table><form></form></table></body></html>",
            ),
            ("<select><b>x</b></select>", "<html><head></head><body><select><b>x</b></select></body></html>"),
            ('<div a="unterminated', "<html><head></head><body></body></html>"),
            ("<DIV A=1 a=2>x</DIV>", '<html><head></head><body><div a="1">x</div></body></html>'),
            ("<svg><g/>x</svg>", "<html><head></head><body><svg><g></g>x</svg></body></html>"),
            ("<math><mi>x</mi></math>", "<html><head></head><body><math><mi>x</mi></math></body></html>"),
            ("<p><b>x</p>y</b>", "<html><head></head><body><p><b>x</b></p><b>y</b></body></html>"),
            (
                "<frameset><frame></frameset><frameset>",
                "<html><head></head><frameset><frame></frame></frameset></html>",
            ),
            ("<frameset>x", "<html><head></head><frameset></frameset></html>"),
            ("<frameset> \r\n", "<html><head></head><frameset> \n</frameset></html>"),
        ]
        for html, expected in cases:
            with self.subTest(html=html):
                self.assert_parses_to(html, expected, sanitize=False)

    def test_empty_fragment_and_multiline_error_paths(self) -> None:
        self.assert_parses_to("\n", "", fragment_context=FragmentContext("textarea"))
        self.assert_parses_to("", "", fragment_context=FragmentContext("style"), sanitize=False)
        self.assert_parses_to(
            "<head></head> \n",
            "<html><head></head> \n<body></body></html>",
            sanitize=False,
        )

        document = JustHTML("<!doctype html><div></span\n>", sanitize=False, collect_errors=True)
        assert [error.code for error in document.errors] == ["unexpected-end-tag"]

    def test_template_insertion_modes(self) -> None:
        cases = [
            ("<template><colgroup><col><col></colgroup></template>", "<html><head></head><body></body></html>"),
            (
                "<template><table><form><tr><td>x</table></template>",
                "<html><head><table><tbody><tr><td>x</td></tr></tbody></table></head><body></body></html>",
            ),
            (
                "<template><table><td>x</td></table></template>",
                "<html><head><table><tbody><tr><td>x</td></tr></tbody></table></head><body></body></html>",
            ),
            (
                "<template><table><tbody><tr><td>x</td></tr></tbody></table></template>",
                "<html><head><table><tbody><tr><td>x</td></tr></tbody></table></head><body></body></html>",
            ),
            (
                "<template><table><thead><tr><th>x<tbody><tr><td>y</table></template>",
                "<html><head><table><thead><tr><th>x</th></tr></thead>"
                "<tbody><tr><td>y</td></tr></tbody></table></head><body></body></html>",
            ),
            (
                "<template><table><tr><td>x<tr><td>y</table></template>",
                "<html><head><table><tbody><tr><td>x</td></tr><tr><td>y</td></tr></tbody></table>"
                "</head><body></body></html>",
            ),
            (
                "<template><table><tr><td>x</tbody><td>y</table></template>",
                "<html><head>y<table><tbody><tr><td>x</td></tr></tbody></table></head><body></body></html>",
            ),
            (
                "<template><table><caption>x<tbody><tr><td>y</table></template>",
                "<html><head><table><caption>x</caption><tbody><tr><td>y</td></tr></tbody></table>"
                "</head><body></body></html>",
            ),
            (
                "<template><table><col><tbody><tr><td>x</table></template>",
                "<html><head><table><tbody><tr><td>x</td></tr></tbody></table></head><body></body></html>",
            ),
            (
                "<template><tbody><td>x<tfoot><tr><td>y</template>",
                "<html><head><tbody><tr><td>x</td></tr></tbody><tfoot><tr><td>y</td></tr></tfoot>"
                "</head><body></body></html>",
            ),
            (
                "<template><tr><th>x</th><caption>y</template>",
                "<html><head><tr><th>x</th></tr>y</head><body></body></html>",
            ),
            (
                "<template><td>x</td><tr><td>y</template>",
                "<html><head><td>x</td><td>y</td></head><body></body></html>",
            ),
            (
                "<template><pre>\nx</pre><div>y</template>",
                "<html><head><pre>x</pre><div>y</div></head><body></body></html>",
            ),
            ("<template><html><head><body>x</template>", "<html><head>x</head><body></body></html>"),
            (
                "<template><template><table><tr><td>x</template>y</template>",
                "<html><head><table><tbody><tr><td>x</td></tr></tbody></table>y</head><body></body></html>",
            ),
        ]
        for html, expected in cases:
            with self.subTest(html=html):
                self.assert_parses_to(html, expected)

    def test_targeted_recovery_regressions(self) -> None:
        default_cases = [
            ("<script>x", "<html><head></head><body></body></html>"),
            ("<table><template>x</table></template>", "<html><head></head><body><table>x</table></body></html>"),
            ("<svg><noframes>x</svg></noframes>", "<html><head></head><body></body></html>"),
            ("<template><colgroup>x</colgroup></template>", "<html><head></head><body></body></html>"),
            ("<table><colgroup>x</table></colgroup>", "<html><head></head><body>x<table></table></body></html>"),
            ("<nobr/>", "<html><head></head><body></body></html>"),
            ("<svg><plaintext>x</svg></plaintext>", "<html><head></head><body></body></html>"),
            ("<noframes>x</noframes>", "<html><head>x</head><body></body></html>"),
            ("<noframes/>", "<html><head></head><body></body></html>"),
            ("<li><li>x</li></li>", "<html><head></head><body><li></li><li>x</li></body></html>"),
            ("<dd><dd>x</dd></dd>", "<html><head></head><body>x</body></html>"),
            ("<head><svg>x</head></svg>", "<html><head></head><body></body></html>"),
            ("<nobr><p>x</nobr></p>", "<html><head></head><body><p>x</p></body></html>"),
            ("<frameset><body>x</frameset></body>", "<html><head></head></html>"),
        ]
        for html, expected in default_cases:
            with self.subTest(html=html):
                self.assert_parses_to(html, expected)

        raw_cases = [
            ("<h1>x</h1>", "<html><head></head><body><h1>x</h1></body></html>"),
            ("<svg><b>x</svg></b>", "<html><head></head><body><svg></svg><b>x</b></body></html>"),
        ]
        for html, expected in raw_cases:
            with self.subTest(html=html):
                self.assert_parses_to(html, expected, sanitize=False, track_node_locations=True)

        fragment_cases = [
            ("<svg>x", FragmentContext("table"), ""),
            ("<p><plaintext>x</p></plaintext>", FragmentContext("table"), "<p></p>x&lt;/p&gt;&lt;/plaintext&gt;"),
            ("<html>x</html>", FragmentContext("html"), "x"),
            ("<frameset>x</frameset>", FragmentContext("html"), ""),
            ("<h1><h1>x</h1></h1>", FragmentContext("tbody"), "<h1></h1><h1>x</h1>"),
            ("<html><template>x</html></template>", FragmentContext("table"), "x"),
            ("<span><frameset>x</span></frameset>", FragmentContext("html"), "<span>x</span>"),
        ]
        for html, context, expected in fragment_cases:
            with self.subTest(html=html, context=context):
                self.assert_parses_to(html, expected, fragment_context=context)

    def test_coverage_guided_malformed_recovery(self) -> None:
        default_cases = [
            ("<svg><div><frameset>x</svg></div></frameset>", "<html><head></head></html>"),
            (
                "<table><a><caption>x</table></a></caption>",
                "<html><head></head><body><a></a><table><caption>x</caption></table></body></html>",
            ),
            ("<xmp>\r\f\0</xmp>", "<html><head></head><body>\n\f\0</body></html>"),
            ('<nobr a\0b="unterminated>x</nobr>', "<html><head></head><body></body></html>"),
            ("<plaintext>\r\f\0", "<html><head></head><body>\n\f�</body></html>"),
            ("<x =x>", "<html><head></head><body></body></html>"),
            ('<form a\0b="unterminated>x</form>', "<html><head></head><body></body></html>"),
            ("<x a = >", "<html><head></head><body></body></html>"),
            ("<x a", "<html><head></head><body></body></html>"),
        ]
        for html, expected in default_cases:
            with self.subTest(html=html):
                self.assert_parses_to(html, expected)

        self.assert_parses_to(
            "<!DOCTYPE",
            "<html><head></head><body>&lt;!DOCTYPE</body></html>",
            sanitize=False,
            track_node_locations=True,
        )
        self.assert_parses_to(
            "<x a = >",
            "<html><head></head><body><x a></x></body></html>",
            sanitize=False,
            track_node_locations=True,
        )

        raw_null_attr = JustHTML("<x a\0b=x\0>", sanitize=False, track_node_locations=True)
        element = raw_null_attr.query("x")[0]
        assert element.attrs == {"a�b": "x�"}

        fragment_cases = [
            ("<template><td><option>x</template></td></option>", FragmentContext("tbody"), "<tr><td>x</td></tr>"),
            (
                "<a><tbody><span>x</a></tbody></span>",
                FragmentContext("table"),
                "<a></a><a><span>x</span></a><tbody></tbody>",
            ),
            (
                "<svg><table><caption>x</svg></table></caption>",
                FragmentContext("table"),
                "<table><caption>x</caption></table>",
            ),
            (
                "<template><caption><td>x</template></caption></td>",
                FragmentContext("table"),
                "<caption><td>x</td></caption>",
            ),
            (
                "<template><colgroup><plaintext>x</template></colgroup></plaintext>",
                FragmentContext("table"),
                "x&lt;/template&gt;&lt;/colgroup&gt;&lt;/plaintext&gt;",
            ),
            (
                "<template><caption><tbody>x</template></caption></tbody>",
                FragmentContext("table"),
                "x<caption><tbody></tbody></caption>",
            ),
        ]
        for html, context, expected in fragment_cases:
            with self.subTest(html=html, context=context):
                self.assert_parses_to(html, expected, fragment_context=context)

    def test_foreign_integration_attribute_projection(self) -> None:
        values = [
            "",
            "\\bad",
            "#frag",
            "//example.com",
            "relative/path",
            "http://x",
            "https:foo",
            "mailto:a@b",
            "javascript:alert(1)",
            "\x01https://x",
            "https://é.example",
        ]
        for value in values:
            html = (
                "<svg><foreignObject>"
                f'<a href="{value}" class=x onclick=y>x</a>'
                f'<img src="{value}" alt=x>'
                "</foreignObject></svg>"
            )
            with self.subTest(value=value):
                self.assert_parses_to(html, "<html><head></head><body></body></html>")

    def test_compiled_url_policy_variants(self) -> None:
        base = DEFAULT_DOCUMENT_POLICY
        base_rules = dict(base.url_policy.allow_rules)
        base_anchor_rule = base_rules[("a", "href")]

        variants = [
            (replace(base_anchor_rule, allow_fragment=False), "#frag", "<a>x</a>"),
            (
                replace(base_anchor_rule, resolve_protocol_relative=None),
                "//example.com",
                "<a>x</a>",
            ),
            (
                replace(base_anchor_rule, resolve_protocol_relative="ftp"),
                "//example.com",
                "<a>x</a>",
            ),
            (replace(base_anchor_rule, allow_relative=False), "relative/path", "<a>x</a>"),
            (replace(base_anchor_rule, allow_relative=False), "https:foo", "<a>x</a>"),
            (replace(base_anchor_rule, allowed_schemes={"https"}), "mailto:a@b", "<a>x</a>"),
        ]
        for rule, value, expected_anchor in variants:
            rules = {**base_rules, ("a", "href"): rule}
            policy = replace(base, url_policy=UrlPolicy(default_handling="allow", allow_rules=rules))
            with self.subTest(rule=rule, value=value):
                self.assert_parses_to(
                    f'<a href="{value}">x</a>',
                    f"<html><head></head><body>{expected_anchor}</body></html>",
                    policy=policy,
                )

                self.assert_parses_to(
                    f'<svg><foreignObject><a href="{value}">x</a></foreignObject></svg>',
                    "<html><head></head><body></body></html>",
                    policy=policy,
                )

    def test_deep_recovery_interactions(self) -> None:
        self.assert_parses_to(
            "<template type=hidden><nobr selected>x<select selected>\n"
            "<tbody type=hidden></nobr></select>\n</template>",
            "<html><head>x\n\n</head><body></body></html>",
        )
        self.assert_parses_to(
            "<a href=x><rb a=1> <rtc><table class=x id=y><select href=x>x</table><ruby class=x id=y>&amp;",
            '<html><head></head><body><a href="x"><rb a="1"> <rtc><select href="x">x</select>'
            '<table class="x" id="y"></table><ruby class="x" id="y">&amp;</ruby></rtc></rb></a></body></html>',
            sanitize=False,
            track_node_locations=True,
        )
        self.assert_parses_to(
            "<h1 type=hidden><ruby></ruby>\n<h2 href=x>x</h1>x<th a=1><svg type=hidden></th>",
            '<html><head></head><body><h1 type="hidden"><ruby></ruby>\n</h1>'
            '<h2 href="x">x</h2>x<svg type="hidden"></svg></body></html>',
            sanitize=False,
            track_node_locations=True,
        )
        self.assert_parses_to(
            "<svg type=hidden></rp></svg><frameset class=x id=y> </ruby><rb href=x></span>\n",
            "<html><head></head> \n</html>",
        )
        self.assert_parses_to(
            "<th class=x><template class=x id=y></th>x<caption class=x id=y>x"
            "<tr selected><address>&amp;<span href=x><ul type=hidden> ",
            '&amp;<span><ul> </ul></span><tbody><tr><th class="x">x</th></tr></tbody>'
            '<caption class="x" id="y">x<tr></tr></caption>',
            fragment_context=FragmentContext("table"),
        )
        self.assert_parses_to(
            "<tr selected> <svg href=x><table href=x></table><math a=1> </svg>\n</tr></math>\n<blockquote a=1>",
            "<tbody><tr> </tr></tbody><table></table> \n\n<blockquote></blockquote>",
            fragment_context=FragmentContext("table"),
        )
        self.assert_parses_to(
            "<svg selected>\n<caption class=x id=y>&amp;<rtc a=1> <rb class=x id=y>"
            "</rtc></svg> <rt href=x>\n<frameset href=x>x<h1 a=1>&amp;<math>x",
            '<html><head></head><body><svg selected>\n<caption class="x" id="y">&amp;'
            '<rtc a="1"> <rb class="x" id="y"></rb></rtc></caption></svg> '
            '<rt href="x">\nx<h1 a="1">&amp;<math>x</math></h1></rt></body></html>',
            sanitize=False,
            track_node_locations=True,
        )

    def test_text_normalization_states(self) -> None:
        cases = [
            ("<p>a\r\nb</p>", "<html><head></head><body><p>a\nb</p></body></html>", {}),
            ("<head>  x", "<html><head>  </head><body>x</body></html>", {}),
            ("<head></head>  <body>x", "<html><head></head>  <body>x</body></html>", {}),
            (
                "<table><colgroup>  x<tr><td>y",
                "<html><head></head><body>x<table>  <tbody><tr><td>y</td></tr></tbody></table></body></html>",
                {},
            ),
            ("<table><colgroup>  x", "<html><head></head><body>x<table>  </table></body></html>", {}),
            ("<pre>\nx", "<html><head></head><body><pre>x</pre></body></html>", {}),
            (
                "<head><noscript>  \n</noscript><p>x",
                "<html><head>  \n</head><body><p>x</p></body></html>",
                {"scripting_enabled": False},
            ),
            (
                "<svg><g>a\0b</g></svg>",
                "<html><head></head><body><svg><g>a�b</g></svg></body></html>",
                {"sanitize": False},
            ),
        ]
        for html, expected, kwargs in cases:
            with self.subTest(html=html):
                self.assert_parses_to(html, expected, **kwargs)

        self.assert_parses_to(
            "\n a\r\fb\0&amp;",
            " a\n\fb�&amp;",
            fragment_context=FragmentContext("textarea"),
        )
        self.assert_parses_to(
            "<p>a\ufdd0b\ufffec</p>",
            "<html><head></head><body><p>a�b�c</p></body></html>",
            sanitize=False,
            _parser_opts=ParserOptions(xml_coercion=True),
        )

    def test_coverage_guided_frameset_and_attr_recovery(self) -> None:
        self.assert_parses_to(
            "</body>x",
            "<html><head></head><body>x</body></html>",
            sanitize=False,
        )
        self.assert_parses_to(
            "</body>x<!--c-->",
            "<html><head></head><body>x<!--c--></body></html>",
            sanitize=False,
        )
        self.assert_parses_to(
            "<frameset></frameset></body>",
            "<html><head></head><frameset></frameset></html>",
            sanitize=False,
        )
        self.assert_parses_to(
            "<frameset><frame><frame>",
            "<html><head></head><frameset><frame></frame><frame></frame></frameset></html>",
            sanitize=False,
        )
        self.assert_parses_to(
            "x<frameset>",
            "<head></head><body>x</body>",
            fragment_context=FragmentContext("html"),
            sanitize=False,
        )
        self.assert_parses_to(
            "<input><frameset>",
            "<html><head></head><body><input></body></html>",
            sanitize=False,
        )
        self.assert_parses_to(
            "<button><frameset>",
            "<html><head></head><body></body></html>",
        )

        self.assert_parses_to(
            '<x/"><p>',
            "<html><head></head><body><p></p></body></html>",
        )
        self.assert_parses_to(
            "<x ?=y>",
            "<html><head></head><body></body></html>",
        )
        self.assert_parses_to(
            '<x/"><p>',
            '<html><head></head><body><x "><p></p></x></body></html>',
            sanitize=False,
            track_node_locations=True,
        )

    def test_direct_engine_branch_coverage(self) -> None:
        raw_plan = compile_raw_engine_plan(fragment=False)
        raw_fragment_plan = compile_raw_engine_plan(fragment=True)
        default_plan = compile_default_engine_plan(fragment=False)

        cases = [
            (
                ParseEngine("</body>x<!--c-->", fragment=False, plan=raw_plan),
                "<html><head></head><body>x<!--c--></body></html>",
            ),
            (
                ParseEngine("</body>x", fragment=False, plan=raw_plan),
                "<html><head></head><body>x</body></html>",
            ),
            (
                ParseEngine("<frameset></frameset></body>", fragment=False, plan=raw_plan),
                "<html><head></head><frameset></frameset></html>",
            ),
            (
                ParseEngine("<frameset><frame><frame>", fragment=False, plan=raw_plan),
                "<html><head></head><frameset><frame></frame><frame></frame></frameset></html>",
            ),
            (
                ParseEngine(
                    "x<frameset>",
                    fragment=True,
                    fragment_context=FragmentContext("html"),
                    plan=raw_fragment_plan,
                ),
                "<head></head><body>x</body>",
            ),
            (
                ParseEngine("<input><frameset>", fragment=False, plan=raw_plan),
                "<html><head></head><body><input></body></html>",
            ),
            (
                ParseEngine("<button><frameset>", fragment=False, plan=default_plan),
                "<html><head></head><body></body></html>",
            ),
            (
                ParseEngine('<x/"><p>', fragment=False, plan=default_plan),
                "<html><head></head><body><p></p></body></html>",
            ),
            (
                ParseEngine("<x ?=y>", fragment=False, plan=raw_plan, track_node_locations=True),
                '<html><head></head><body><x ?="y"></x></body></html>',
            ),
        ]

        for engine, expected in cases:
            with self.subTest(html=engine._html_input, fragment=engine._fragment):
                assert engine.parse().to_html(pretty=False) == expected
        self.assert_parses_to(
            "<x ?=y>",
            '<html><head></head><body><x ?="y"></x></body></html>',
            sanitize=False,
            track_node_locations=True,
        )

    def test_direct_engine_private_helper_coverage(self) -> None:
        raw_plan = compile_raw_engine_plan(fragment=False)
        raw_fragment_plan = compile_raw_engine_plan(fragment=True)
        default_plan = compile_default_engine_plan(fragment=False)

        comment_engine = ParseEngine("</body>x<!--c-->", fragment=False, plan=raw_plan)
        comment_engine.parse()
        comment_engine._after_body = True
        comment_engine._after_document = True
        comment_engine._after_html = False
        comment_engine._body_mode_seen = False
        comment_engine._append_comment("c", source_pos=8)
        self.assertFalse(comment_engine._after_body)
        self.assertFalse(comment_engine._after_document)
        self.assertFalse(comment_engine._after_html)
        self.assertTrue(comment_engine._body_mode_seen)

        text_engine = ParseEngine("</body>x", fragment=False, plan=raw_plan)
        text_engine.parse()
        text_engine._after_body = True
        text_engine._after_document = True
        text_engine._after_html = False
        text_engine._body_mode_seen = False
        text_engine._stack = [text_engine._doc, text_engine._html]  # type: ignore[list-item]
        text_engine._append_text("x", source_pos=7)
        self.assertFalse(text_engine._after_body)
        self.assertFalse(text_engine._after_document)
        self.assertFalse(text_engine._after_html)
        self.assertTrue(text_engine._body_mode_seen)

        end_tag_engine = ParseEngine("</body>", fragment=False, plan=raw_plan)
        end_tag_engine.parse()
        end_tag_engine._frameset_seen = True
        end_tag_engine._body_explicit = False
        end_tag_engine._stack = [end_tag_engine._doc, end_tag_engine._html, end_tag_engine._head]  # type: ignore[list-item]
        end_tag_engine._parse_end_tag(2, len(end_tag_engine._html_input))
        self.assertTrue(end_tag_engine._after_document)
        self.assertFalse(end_tag_engine._after_html)

        frame_engine = ParseEngine("<frame>", fragment=False, plan=raw_plan)
        frame_engine.parse()
        frameset = Element("frameset", {}, "html")
        frame_engine._append(frame_engine._html, frameset)
        frame_engine._frameset_seen = True
        frame_engine._body_explicit = False
        frame_engine._stack = [frame_engine._doc, frame_engine._html, frameset]  # type: ignore[list-item]
        frame_engine._parse_start_tag(1, len(frame_engine._html_input))
        self.assertEqual([child.name for child in frameset.children], ["frame"])

        blocked_frame_engine = ParseEngine("<frame>", fragment=False, plan=default_plan)
        blocked_frame_engine.parse()
        blocked_frameset = Element("frameset", {}, "html")
        blocked_frame_engine._append(blocked_frame_engine._html, blocked_frameset)
        blocked_frame_engine._frameset_seen = True
        blocked_frame_engine._body_explicit = False
        blocked_frame_engine._stack = [blocked_frame_engine._doc, blocked_frame_engine._html, blocked_frameset]  # type: ignore[list-item]
        blocked_frame_engine._parse_start_tag(1, len(blocked_frame_engine._html_input))
        self.assertEqual(blocked_frameset.children, [])

        skip_attrs_engine = ParseEngine('/">', fragment=False, plan=default_plan)
        assert skip_attrs_engine._skip_attrs(0, len(skip_attrs_engine._html_input)) == ({}, False, 3, True)

        parse_attrs_engine = ParseEngine('/">', fragment=False, plan=default_plan)
        action = default_plan.tag_actions["a"]
        assert parse_attrs_engine._parse_attrs_for_action(action, 0, len(parse_attrs_engine._html_input)) == (
            {},
            False,
            3,
            True,
        )

        fragment_engine = ParseEngine(
            "",
            fragment=True,
            fragment_context=FragmentContext("html"),
            plan=raw_fragment_plan,
        )
        fragment_engine.parse()
        fragment_engine._append(fragment_engine._body, Element("input", {}, "html"))
        self.assertFalse(fragment_engine._accept_fragment_frameset())

        parser_only_engine = ParseEngine("", fragment=False, plan=raw_plan)
        parser_only_engine.parse()
        parser_only_engine._append(parser_only_engine._body, Element("button", {}, "justhtml-parser-only"))
        self.assertFalse(parser_only_engine._body_allows_frameset(parser_only_engine._body))

        visible_input_engine = ParseEngine("", fragment=False, plan=raw_plan)
        visible_input_engine.parse()
        visible_input_engine._append(visible_input_engine._body, Element("input", {}, "html"))
        self.assertFalse(visible_input_engine._body_allows_frameset(visible_input_engine._body))

    def test_upstream_inputs_across_diagnostic_modes(self) -> None:
        config = {
            "fail_fast": False,
            "test_specs": [],
            "exclude_html": None,
            "filter_html": None,
            "exclude_errors": None,
            "filter_errors": None,
            "exclude_files": None,
        }
        runner = TestRunner(Path("tests/html5lib-tests-tree"), config)

        parsed = 0
        for file_path, tests in runner.load_tests():
            for index, test in enumerate(tests):
                if not runner._should_run_test(file_path.name, index, test):
                    continue
                scripting_enabled = test.script_directive != "script-off"

                located = JustHTML(
                    test.data,
                    fragment_context=test.fragment_context,
                    scripting_enabled=scripting_enabled,
                    sanitize=False,
                    collect_errors=True,
                    track_node_locations=True,
                )
                assert located.root is not None

                xml = JustHTML(
                    test.data,
                    fragment_context=test.fragment_context,
                    scripting_enabled=scripting_enabled,
                    sanitize=False,
                    _parser_opts=ParserOptions(xml_coercion=True),
                )
                assert xml.root is not None

                projected = JustHTML(
                    test.data,
                    fragment_context=test.fragment_context,
                    scripting_enabled=scripting_enabled,
                    track_node_locations=True,
                )
                assert projected.root is not None
                parsed += 1

        assert parsed >= 1700
