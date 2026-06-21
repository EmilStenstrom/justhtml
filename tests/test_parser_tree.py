import unittest

from justhtml import JustHTML, SanitizationPolicy
from justhtml.parser.context import FragmentContext
from justhtml.parser.options import ParserOptions
from justhtml.sanitizer import UrlPolicy


class TestParserTreeConstruction(unittest.TestCase):
    def test_chromium_null_and_line_end_preprocessing_in_markup_states(self) -> None:
        cases = {
            "<!--a\0b-->": "<!--a�b--><html><head></head><body></body></html>",
            "<!--a-": "<!--a--><html><head></head><body></body></html>",
            "<!x\r\ny>": "<!--x\ny--><html><head></head><body></body></html>",
            "<\0div>": "<html><head></head><body>&lt;�div&gt;</body></html>",
            "</>": "<html><head></head><body></body></html>",
            "\0\fA": "<html><head></head><body>A</body></html>",
        }

        for html, expected in cases.items():
            with self.subTest(html=html):
                assert JustHTML(html, sanitize=False).to_html(pretty=False) == expected

        document = JustHTML("<!DOCTYPE\0html>", sanitize=False)
        doctype = document.root.children[0]
        assert doctype.name == "!doctype"
        assert doctype.data.name == "�html"

    def test_absent_digit_numeric_reference_follows_the_standard_despite_chromium(self) -> None:
        document = JustHTML("&#x;", sanitize=False)

        assert document.to_text(strip=False) == "&#x;"

    def test_chromium_preserves_preamble_order_and_replacement_attribute_names(self) -> None:
        document = JustHTML("<! first><?second", sanitize=False)
        assert [(node.name, node.data) for node in document.root.children[:2]] == [
            ("#comment", " first"),
            ("#comment", "?second"),
        ]

        document = JustHTML("<!--first--><!DOCTYPE\r\nhtml>", sanitize=False)
        assert [node.name for node in document.root.children[:3]] == ["#comment", "!doctype", "html"]
        assert document.root.children[1].data.name == "html"

        document = JustHTML("<div \0name=value>", sanitize=False)
        div = document.query("div")[0]
        assert div.attrs == {"�name": "value"}
        assert document.to_html(pretty=False) == ('<html><head></head><body><div �name="value"></div></body></html>')

        document = JustHTML("<source\0 x=y>", sanitize=False)
        source = document.query("source�")[0]
        assert source.attrs == {"x": "y"}
        assert document.to_html(pretty=False) == ('<html><head></head><body><source� x="y"></source�></body></html>')

        document = JustHTML("<html 1name='\0'>", sanitize=False)
        assert document.root.children[-1].attrs == {"1name": "�"}

        document = JustHTML("<textarea\v/>x", sanitize=False)
        textarea = document.query("body")[0].children[0]
        assert textarea.name == "textarea\v"
        assert textarea.to_text(strip=False) == "x"
        assert document.to_html(pretty=False) == ("<html><head></head><body><textarea\v>x</textarea\v></body></html>")

        document = JustHTML("<source\0 '='<script>alert(1)</script>'>", sanitize=False)
        assert document.query("source�")[0].attrs == {"'": "<script>alert(1)</script>"}

        attribute_cases = {
            "<h\n=>": {"=": ""},
            "<h\n=9>": {"=9": ""},
            "<h\n==>": {"=": ""},
            "<p ==>": {"=": ""},
            "<m\fi=''=\">": {"i": "", '="': ""},
            "<r/Y>": {"y": ""},
            '<x/">': {'"': ""},
        }
        for html, expected in attribute_cases.items():
            with self.subTest(html=html):
                element = JustHTML(html, sanitize=False).query("body")[0].children[0]
                assert element.attrs == expected

    def test_chromium_foreign_template_table_and_frameset_regressions(self) -> None:
        cases = {
            "<svg><image href='x'></svg>": (
                '<html><head></head><body><svg><image href="x"></image></svg></body></html>'
            ),
            "<table><colgroup></colgroup><colgroup></colgroup></table>": (
                "<html><head></head><body><table><colgroup></colgroup><colgroup></colgroup></table></body></html>"
            ),
            "<template><title>x</title><base><link></template>": (
                "<html><head><template><title>x</title><base><link></template></head><body></body></html>"
            ),
            "<frameset><body>x</body></frameset>": ("<html><head></head><frameset></frameset></html>"),
            "<frameset></frameset></html><!--a--><script><!--b--></script>": (
                "<html><head></head><frameset></frameset></html><!--a--><!--b-->"
            ),
            "<frameset></frameset><frame>": "<html><head></head><frameset></frameset></html>",
            "<frameset></frameset><frameset>": "<html><head></head><frameset></frameset></html>",
            "<frameset></body><!": "<html><head></head><frameset><!----></frameset></html>",
            "<template></html><//": ("<html><head><template><!--/--></template></head><body></body></html>"),
        }

        for html, expected in cases.items():
            with self.subTest(html=html):
                document = JustHTML(html, sanitize=False)
                assert document.to_html(pretty=False) == expected

        document = JustHTML("<math><html>", sanitize=False)
        math = document.query("math")[0]
        assert math.children[0].name == "html"
        assert math.children[0].namespace == "math"

        document = JustHTML("<rt><rt>", sanitize=False)
        first_rt = document.query("rt")[0]
        assert first_rt.children[0].name == "rt"

        document = JustHTML("<select><table><input><select>", sanitize=False)
        select = document.query("select")[0]
        assert [child.name for child in select.children] == ["input", "select", "table"]

        document = JustHTML("<select><table><d><select>", sanitize=False)
        select = document.query("select")[0]
        assert [child.name for child in select.children] == ["d", "table"]
        assert select.children[0].children[0].name == "select"

        cases = {
            "</html><script>": "<html><head></head><body><script></script></body></html>",
            "</head></html><y>": "<html><head></head><body><y></y></body></html>",
            "</html><</ ": "<html><head></head><body>&lt;<!-- --></body></html>",
            "<button><p><math></button>>": (
                "<html><head></head><body><button><p><math></math></p></button>&gt;</body></html>"
            ),
            "<i></body></f><": "<html><head></head><body><i>&lt;</i></body></html>",
            "</>&#9": "<html><head></head><body></body></html>",
            "<!>&#xD": "<!----><html><head></head><body></body></html>",
            "<!DOCTYPE>&#xD": "<!DOCTYPE><html><head></head><body></body></html>",
            "</e>&#9": "<html><head></head><body></body></html>",
            "<template><tr>d": "<html><head><template><tr></tr>d</template></head><body></body></html>",
        }
        for html, expected in cases.items():
            with self.subTest(html=html):
                assert JustHTML(html, sanitize=False).to_html(pretty=False) == expected

        for html, element_name, text in (
            ("<math><source>>", "source", ">"),
            ("<svg><input>x", "input", "x"),
            ("<math><frame><", "frame", "<"),
        ):
            with self.subTest(html=html):
                element = JustHTML(html, sanitize=False).query(element_name)[0]
                assert element.to_text(strip=False) == text

        for html in ('<o\fe="\r">', '<d 2="\r">'):
            with self.subTest(html=html):
                element = JustHTML(html, sanitize=False).query("body")[0].children[0]
                assert next(iter(element.attrs.values())) == "\n"

        document = JustHTML("<a 8=&#13\r>", sanitize=False)
        assert document.query("a")[0].attrs == {"8": "\r"}

        document = JustHTML("<frameset>&#9", sanitize=False)
        assert document.query("frameset")[0].to_text(strip=False) == "\t"

        document = JustHTML("<svg><body>", sanitize=False)
        body = document.query("body")[0]
        assert [child.name for child in body.children] == ["svg"]

        cases = {
            "<!><template></template><?": (
                "<!----><html><head><template></template><!--?--></head><body></body></html>"
            ),
            "<template><d><table><tr>": (
                "<html><head><template><d><table><tbody><tr></tr></tbody></table></d></template>"
                "</head><body></body></html>"
            ),
            "<table><c><col>": (
                "<html><head></head><body><c></c><table><colgroup><col></colgroup></table></body></html>"
            ),
            "<optgroup><optgroup>": (
                "<html><head></head><body><optgroup><optgroup></optgroup></optgroup></body></html>"
            ),
        }
        for html, expected in cases.items():
            with self.subTest(html=html):
                assert JustHTML(html, sanitize=False).to_html(pretty=False) == expected

        document = JustHTML("</head></html>2</t></-", sanitize=False)
        body = document.query("body")[0]
        assert body.to_text(strip=False) == "2"
        assert body.children[-1].name == "#comment"

    def test_unknown_body_element_blocks_frameset_per_standard_despite_chromium(self) -> None:
        document = JustHTML("<f><frameset>", sanitize=False)

        assert document.to_html(pretty=False) == "<html><head></head><body><f></f></body></html>"

    def test_chromium_nested_template_and_recovery_regressions(self) -> None:
        cases = {
            "<template><table><tr><table>": (
                "<html><head><template><table><tbody><tr></tr></tbody></table><table></table>"
                "</template></head><body></body></html>"
            ),
            "<template><table><col>": (
                "<html><head><template><table><colgroup><col></colgroup></table></template></head><body></body></html>"
            ),
            "<template><colgroup>X": (
                "<html><head><template><colgroup></colgroup>X</template></head><body></body></html>"
            ),
            "<template><colgroup><v>": (
                "<html><head><template><colgroup></colgroup><v></v></template></head><body></body></html>"
            ),
            "<template><col><": "<html><head><template><col></template></head><body></body></html>",
            "<template><table><table><": (
                "<html><head><template><table></table>&lt;<table></table></template></head><body></body></html>"
            ),
            "<template><td><table><tr>": (
                "<html><head><template><td><table><tbody><tr></tr></tbody></table></td></template>"
                "</head><body></body></html>"
            ),
            "<template><table><body><td>": (
                "<html><head><template><table><tbody><tr><td></td></tr></tbody></table></template>"
                "</head><body></body></html>"
            ),
            "<template><tr><form>V": (
                "<html><head><template><tr><form></form></tr>V</template></head><body></body></html>"
            ),
            "<template><tbody><e><tr>": (
                "<html><head><template><tbody><tr></tr></tbody><e></e></template></head><body></body></html>"
            ),
            "<template><tr><u><td>": (
                "<html><head><template><tr><td></td></tr><u></u></template></head><body></body></html>"
            ),
            "<table><template><n><form><": (
                "<html><head></head><body><table><template><n><form>&lt;</form></n></template></table></body></html>"
            ),
            "<a><dialog><a>": ("<html><head></head><body><a><dialog></dialog></a><a></a></body></html>"),
            "<table><a><th></th></br>": (
                "<html><head></head><body><a></a><a><br></a><table><tbody><tr><th></th></tr>"
                "</tbody></table></body></html>"
            ),
            '</r\n=="><': "<html><head></head><body></body></html>",
            "<table></> </l>\n<": ("<html><head></head><body>\n&lt;<table> </table></body></html>"),
            "<table></>\n<": "<html><head></head><body>\n&lt;<table></table></body></html>",
            "<table><template><tr>s": (
                "<html><head></head><body><table><template><tr></tr>s</template></table></body></html>"
            ),
            "<template><tr><script>": (
                "<html><head><template><tr><script></script></tr></template></head><body></body></html>"
            ),
            "<h3><svg><title><d></h3><": (
                "<html><head></head><body><h3><svg><title><d>&lt;</d></title></svg></h3></body></html>"
            ),
            "<p><a><xmp>": ("<html><head></head><body><p><a></a></p><a><xmp></xmp></a></body></html>"),
            "<div><strong></div><table><colgroup>H": (
                "<html><head></head><body><div><strong></strong></div><strong>H</strong>"
                "<table><colgroup></colgroup></table></body></html>"
            ),
        }
        for html, expected in cases.items():
            with self.subTest(html=html):
                assert JustHTML(html, sanitize=False).to_html(pretty=False) == expected

        document = JustHTML("<noscript><script>", sanitize=False, scripting_enabled=False)
        assert document.to_html(pretty=False) == (
            "<html><head><noscript></noscript><script></script></head><body></body></html>"
        )

        assert JustHTML("<!doctype html><summary><p>foo</summary>bar").to_html(pretty=False) == (
            "<!DOCTYPE html><html><head></head><body><p>foo</p>bar</body></html>"
        )
        assert JustHTML("<summary><span></summary>").to_html(pretty=False) == (
            "<html><head></head><body><span></span></body></html>"
        )
        assert JustHTML("<table><a><th></th></br>").to_html(pretty=False) == (
            "<html><head></head><body><a></a><a><br></a><table><tbody><tr><th></th></tr></tbody></table></body></html>"
        )

    def test_chromium_duplicate_body_and_nested_caption_table_regressions(self) -> None:
        document = JustHTML("<body id='a'><div><body id='b'>x</div>", sanitize=False)
        assert document.to_html(pretty=False) == ('<html><head></head><body id="a"><div>x</div></body></html>')

        document = JustHTML("<div><li>x<body class='late'>y", sanitize=False)
        assert document.to_html(pretty=False) == (
            '<html><head></head><body class="late"><div><li>xy</li></div></body></html>'
        )

        document = JustHTML("<div>x<head><title>ignored</title></head>y", sanitize=False)
        assert document.to_html(pretty=False) == (
            "<html><head></head><body><div>x<title>ignored</title>y</div></body></html>"
        )

        document = JustHTML(
            "<table><caption>x<table><tr><td>nested</td></tr></table></caption></table>",
            sanitize=False,
        )
        assert document.to_html(pretty=False) == (
            "<html><head></head><body><table><caption>x"
            "<table><tbody><tr><td>nested</td></tr></tbody></table>"
            "</caption></table></body></html>"
        )

    def test_compiled_safe_path_merges_duplicate_shell_attributes_and_colgroups(self) -> None:
        document = JustHTML("<html id='first'><html id='second'>")
        assert document.query("html")[0].attrs == {"id": "first"}

        document = JustHTML("<body id='first'><div><body id='second'>x</div>")
        assert document.query("body")[0].attrs == {"id": "first"}
        assert document.query("div")[0].to_text() == "x"

        document = JustHTML("<div><li>x<body class='late'>y")
        assert document.query("body")[0].attrs == {"class": "late"}
        assert document.query("li")[0].to_text() == "xy"

        document = JustHTML("<div>x<head><title>ignored</title></head>y")
        assert document.query("div")[0].to_text(strip=False) == "x ignored y"

        policy = SanitizationPolicy(
            allowed_tags={"html", "head", "body", "table", "colgroup"},
            allowed_attributes={"*": set()},
        )
        document = JustHTML(
            "<table><colgroup></colgroup><colgroup></colgroup></table>",
            sanitize=policy,
        )
        assert document.to_html(pretty=False) == "<html><head></head><body><table></table></body></html>"

    def test_deep_adoption_agency_keeps_the_standard_three_node_limit(self) -> None:
        cases = [
            "<b><em><foo><foob><fooc><aside></b></em>",
            "<b><em><foo><foo><foo><aside></b></em>",
            "<b><em><foo><foo><foo><foo><foo><aside></b></em>",
        ]

        for html in cases:
            with self.subTest(html=html):
                document = JustHTML(html, fragment=True, sanitize=False)
                aside = document.query("aside")[0]
                assert aside.to_html(pretty=False) == "<aside><b></b></aside>"

    def test_active_formatting_reconstructs_before_ordinary_and_foreign_elements(self) -> None:
        document = JustHTML("<div><b>x</div><span>y</span><svg>z</svg>", sanitize=False)

        assert document.to_html(pretty=False) == (
            "<html><head></head><body><div><b>x</b></div><b><span>y</span><svg>z</svg></b></body></html>"
        )

    def test_head_noscript_comment_follows_the_standard_despite_chromium(self) -> None:
        document = JustHTML(
            '<head><noscript><head class="foo"><!--foo--></noscript>',
            sanitize=False,
            scripting_enabled=False,
        )

        assert (
            document.to_html(pretty=False) == "<html><head><noscript><!--foo--></noscript></head><body></body></html>"
        )

    def test_command_remains_an_ordinary_element_despite_chromium(self) -> None:
        document = JustHTML("<!DOCTYPE html><body><command>A", sanitize=False)

        assert document.to_html(pretty=False) == (
            "<!DOCTYPE html><html><head></head><body><command>A</command></body></html>"
        )

    def test_select_fragment_ignores_input_per_standard_despite_chromium(self) -> None:
        document = JustHTML(
            "<input><option>",
            fragment_context=FragmentContext("select"),
            sanitize=False,
        )

        assert document.to_html(pretty=False) == "<option></option>"

    def test_parser_options_copy_is_independent(self) -> None:
        options = ParserOptions(
            discard_bom=False,
            emit_bogus_markup_as_text=True,
            scripting_enabled=False,
            xml_coercion=True,
        )

        copied = options.copy()
        copied.discard_bom = True

        self.assertFalse(options.discard_bom)
        self.assertTrue(copied.discard_bom)
        self.assertTrue(copied.emit_bogus_markup_as_text)
        self.assertFalse(copied.scripting_enabled)
        self.assertTrue(copied.xml_coercion)

    def test_finish_handles_deeply_nested_html_without_recursion(self) -> None:
        html = "<div>" * 1200 + "x" + "</div>" * 1200

        doc = JustHTML(html, sanitize=False)

        self.assertEqual(doc.to_text(strip=False), "x")

    def test_selectedcontent_population_handles_deep_selected_option(self) -> None:
        html = (
            "<select><option selected>"
            + "<div>" * 1200
            + "x"
            + "</div>" * 1200
            + "</option><selectedcontent></selectedcontent></select>"
        )

        doc = JustHTML(html, fragment=True, sanitize=False)

        selectedcontent = None
        stack = [doc.root]
        while stack:
            node = stack.pop()
            if node.name == "selectedcontent":
                selectedcontent = node
                break
            template_content = getattr(node, "template_content", None)
            if template_content is not None:
                stack.append(template_content)
            stack.extend(reversed(getattr(node, "children", None) or []))

        self.assertIsNotNone(selectedcontent)
        assert selectedcontent is not None
        self.assertTrue(selectedcontent.children)
        self.assertEqual(selectedcontent.children[0].name, "div")

    def test_selectedcontent_without_options_does_not_crash(self) -> None:
        doc = JustHTML("<select><selectedcontent></selectedcontent></select>", sanitize=False)

        self.assertEqual(
            doc.to_html(pretty=False),
            "<html><head></head><body><select><selectedcontent></selectedcontent></select></body></html>",
        )

    def test_selectedcontent_population_handles_multiple_selectedcontent_nodes(self) -> None:
        doc = JustHTML(
            "<select><option selected>x</option><selectedcontent></selectedcontent>"
            "<selectedcontent></selectedcontent></select>",
            fragment=True,
            sanitize=False,
        )

        self.assertEqual(
            doc.to_html(pretty=False),
            "<select><option selected>x</option><selectedcontent>x</selectedcontent>"
            "<selectedcontent>x</selectedcontent></select>",
        )

    def test_selectedcontent_population_replaces_fallback_content(self) -> None:
        doc = JustHTML(
            "<select><selectedcontent>fallback</selectedcontent><option>real</option></select>",
            fragment=True,
            sanitize=False,
        )

        self.assertEqual(
            doc.to_html(pretty=False),
            "<select><selectedcontent>real</selectedcontent><option>real</option></select>",
        )

    def test_selectedcontent_population_uses_fallback_options_like_chromium(self) -> None:
        doc = JustHTML(
            "<select><selectedcontent><option selected>fallback</option></selectedcontent>"
            "<option>real</option></select>",
            fragment=True,
            sanitize=False,
        )

        self.assertEqual(
            doc.to_html(pretty=False),
            "<select><selectedcontent>fallback</selectedcontent><option>real</option></select>",
        )

    def test_selectedcontent_population_uses_last_selected_option_like_chromium(self) -> None:
        doc = JustHTML(
            "<select><option selected>one</option><option selected>two</option>"
            "<selectedcontent></selectedcontent></select>",
            fragment=True,
            sanitize=False,
        )

        self.assertEqual(
            doc.to_html(pretty=False),
            "<select><option selected>one</option><option selected>two</option>"
            "<selectedcontent>two</selectedcontent></select>",
        )

    def test_selectedcontent_population_uses_first_selected_option_for_multiple_like_chromium(self) -> None:
        doc = JustHTML(
            "<select multiple><option selected>one</option><option selected>two</option>"
            "<selectedcontent></selectedcontent></select>",
            fragment=True,
            sanitize=False,
        )

        self.assertEqual(
            doc.to_html(pretty=False),
            "<select multiple><option selected>one</option><option selected>two</option>"
            "<selectedcontent>one</selectedcontent></select>",
        )

    def test_selectedcontent_population_skips_disabled_fallback_options_like_chromium(self) -> None:
        cases = {
            "<select><option disabled>first</option><option>second</option>"
            "<selectedcontent></selectedcontent></select>": (
                "<select><option disabled>first</option><option>second</option>"
                "<selectedcontent>second</selectedcontent></select>"
            ),
            "<select><optgroup disabled><option>grouped</option></optgroup><option>second</option>"
            "<selectedcontent></selectedcontent></select>": (
                "<select><optgroup disabled><option>grouped</option></optgroup><option>second</option>"
                "<selectedcontent>second</selectedcontent></select>"
            ),
            "<select><selectedcontent><option disabled>fallback</option></selectedcontent>"
            "<option>real</option></select>": (
                "<select><selectedcontent>real</selectedcontent><option>real</option></select>"
            ),
        }

        for html, expected in cases.items():
            with self.subTest(html=html):
                doc = JustHTML(html, fragment=True, sanitize=False)
                self.assertEqual(doc.to_html(pretty=False), expected)

    def test_selectedcontent_population_clears_when_no_option_source_like_chromium(self) -> None:
        cases = {
            "<select><selectedcontent>fallback</selectedcontent></select>": (
                "<select><selectedcontent></selectedcontent></select>"
            ),
            "<select><option disabled>first</option><selectedcontent>fallback</selectedcontent></select>": (
                "<select><option disabled>first</option><selectedcontent></selectedcontent></select>"
            ),
        }

        for html, expected in cases.items():
            with self.subTest(html=html):
                doc = JustHTML(html, fragment=True, sanitize=False)
                self.assertEqual(doc.to_html(pretty=False), expected)

    def test_selectedcontent_population_does_not_self_clone_like_chromium(self) -> None:
        cases = {
            "<select><option selected><selectedcontent></selectedcontent></option></select>": (
                "<select><option selected><selectedcontent></selectedcontent></option></select>"
            ),
            "<select><option selected>before<selectedcontent></selectedcontent>after</option></select>": (
                "<select><option selected>before<selectedcontent></selectedcontent>after</option></select>"
            ),
        }

        for html, expected in cases.items():
            with self.subTest(html=html):
                doc = JustHTML(html, fragment=True, sanitize=False)
                self.assertEqual(doc.to_html(pretty=False), expected)

    def test_selectedcontent_population_ignores_datalist_options_like_chromium(self) -> None:
        cases = {
            "<select><datalist><option>one</option></datalist><selectedcontent></selectedcontent></select>": (
                "<select><datalist><option>one</option></datalist><selectedcontent></selectedcontent></select>"
            ),
            "<select><datalist><option selected>one</option></datalist><option>two</option>"
            "<selectedcontent></selectedcontent></select>": (
                "<select><datalist><option selected>one</option></datalist><option>two</option>"
                "<selectedcontent>two</selectedcontent></select>"
            ),
        }

        for html, expected in cases.items():
            with self.subTest(html=html):
                doc = JustHTML(html, fragment=True, sanitize=False)
                self.assertEqual(doc.to_html(pretty=False), expected)

    def test_select_start_tags_close_p_like_chromium(self) -> None:
        cases = {
            "<select><p>x<option>y</select>": "<select><p>x</p><option>y</option></select>",
            "<select><p>x<optgroup><option>y</select>": (
                "<select><p>x</p><optgroup><option>y</option></optgroup></select>"
            ),
            "<select><p>x<hr><option>y</select>": "<select><p>x</p><hr><option>y</option></select>",
            "<select><p><i>x<hr>y</select>": "<select><p><i>x</i></p><hr><i>y</i></select>",
            "<select><p><i>x<div>y<option>z</select>": (
                "<select><p><i>x</i></p><div><i>y<option>z</option></i></div></select>"
            ),
            "<select><p><i>x<option>y</select>": ("<select><p><i>x<option>y</option></i></p></select>"),
            "<select><p>x<span>y<option>z</select>": ("<select><p>x<span>y<option>z</option></span></p></select>"),
        }

        for html, expected in cases.items():
            with self.subTest(html=html):
                doc = JustHTML(html, fragment=True, sanitize=False)
                self.assertEqual(doc.to_html(pretty=False), expected)

    def test_li_start_tag_stops_before_special_listing_per_standard_and_chromium(self) -> None:
        doc = JustHTML(
            "<listing>a<li><slot></slot><nextid><article></article></nextid><listing><li><audio><select>x<th><pre><th>&#-1;</li></pre>y</select>",
            fragment=True,
            sanitize=False,
        )

        outer_listing = doc.query("listing")[0]
        outer_li = outer_listing.children[1]

        self.assertEqual(outer_li.name, "li")
        self.assertEqual(outer_li.children[2].name, "listing")
        self.assertEqual(outer_li.children[2].children[0].name, "li")

    def test_menuitem_is_not_special_for_li_start_tag_scanning(self) -> None:
        doc = JustHTML("<!DOCTYPE html><li><menuitem><li>", sanitize=False)

        body_children = doc.query("body")[0].children
        self.assertEqual([child.name for child in body_children], ["li", "li"])
        self.assertEqual(body_children[0].children[0].name, "menuitem")

    def test_heading_end_tag_closes_across_button_scope_like_chromium(self) -> None:
        doc = JustHTML("<object><h2><button><ul></ul></h2><iframe>x", sanitize=False)

        obj = doc.query("object")[0]
        self.assertEqual(obj.children[0].name, "h2")
        self.assertEqual(obj.children[0].children[0].name, "button")
        self.assertEqual(obj.children[1].name, "iframe")

    def test_hr_in_table_context_uses_in_body_void_element_rules(self) -> None:
        cases = {
            "<table><tr><td>before<hr>after</td></tr></table>": (
                "<table><tbody><tr><td>before<hr>after</td></tr></tbody></table>"
            ),
            "<table><tr><th>a<hr>b</th></tr></table>": ("<table><tbody><tr><th>a<hr>b</th></tr></tbody></table>"),
            "<table><caption>a<hr>b</caption></table>": "<table><caption>a<hr>b</caption></table>",
            "<table><tr><td><p>a<hr>b</td></tr></table>": (
                "<table><tbody><tr><td><p>a</p><hr>b</td></tr></tbody></table>"
            ),
        }

        for html, expected in cases.items():
            with self.subTest(html=html):
                doc = JustHTML(html, fragment=True, sanitize=False)
                self.assertEqual(doc.to_html(pretty=False), expected)

    def test_self_closing_hr_in_table_context_is_acknowledged(self) -> None:
        doc = JustHTML(
            "<table><tr><td><hr/></td></tr></table>",
            fragment=True,
            sanitize=False,
            collect_errors=True,
        )

        self.assertNotIn(
            "non-void-html-element-start-tag-with-trailing-solidus",
            [error.code for error in doc.errors],
        )

    def test_leading_lf_skip_does_not_leak_past_non_character_tokens(self) -> None:
        cases = {
            "<div><pre></pre>\nX</div>": "<div><pre></pre>\nX</div>",
            "<div><pre><code></code></pre>\nX</div>": "<div><pre><code></code></pre>\nX</div>",
            "<div><pre><!--x--></pre>\nX</div>": "<div><pre><!--x--></pre>\nX</div>",
            "<div><listing></listing>\nX</div>": "<div><listing></listing>\nX</div>",
            "<div><textarea></textarea>\nX</div>": "<div><textarea></textarea>\nX</div>",
            "<div><pre>\nX</pre></div>": "<div><pre>X</pre></div>",
            "<div><listing>\nX</listing></div>": "<div><listing>X</listing></div>",
            "<div><textarea>\nX</textarea></div>": "<div><textarea>X</textarea></div>",
            "<div><pre></": "<div><pre>&lt;/</pre></div>",
            "<div><pre></x": "<div><pre></pre></div>",
        }

        for html, expected in cases.items():
            with self.subTest(html=html):
                doc = JustHTML(html, fragment=True, sanitize=False)
                self.assertEqual(doc.to_html(pretty=False), expected)

    def test_pre_and_listing_in_select_use_their_in_body_start_rules(self) -> None:
        cases = {
            "<select><pre>\nX</pre></select>": "<select><pre>X</pre></select>",
            "<select><listing>\nX</listing></select>": "<select><listing>X</listing></select>",
            "<select><p>P<pre>\nX</pre></select>": "<select><p>P</p><pre>X</pre></select>",
            "<select><p>P<listing>\nX</listing></select>": "<select><p>P</p><listing>X</listing></select>",
            "<select><pre><code></code>\nX</pre></select>": ("<select><pre><code></code>\nX</pre></select>"),
        }

        for html, expected in cases.items():
            with self.subTest(html=html):
                doc = JustHTML(html, fragment=True, sanitize=False)
                self.assertEqual(doc.to_html(pretty=False), expected)

    def test_form_feed_is_preserved_in_html_text_like_chromium(self) -> None:
        cases = {
            "<div>A\fB</div>": "<div>A\fB</div>",
            "<pre>A\fB</pre>": "<pre>A\fB</pre>",
            "<textarea>A\fB</textarea>": "<textarea>A\fB</textarea>",
            "<style>A\fB</style>": "<style>A\fB</style>",
            "<script>A\fB</script>": "<script>A\fB</script>",
            "<table>A\fB<tr><td>C\fD</td></tr></table>": ("A\fB<table><tbody><tr><td>C\fD</td></tr></tbody></table>"),
            "<svg><text>A\fB</text></svg>": "<svg><text>A\fB</text></svg>",
        }

        for html, expected in cases.items():
            with self.subTest(html=html):
                doc = JustHTML(html, fragment=True, sanitize=False)
                self.assertEqual(doc.to_html(pretty=False), expected)

    def test_form_feed_is_still_coerced_for_xml_output(self) -> None:
        doc = JustHTML(
            "<div>A\fB</div>",
            fragment=True,
            sanitize=False,
            _parser_opts=ParserOptions(xml_coercion=True),
        )

        self.assertEqual(doc.to_html(pretty=False), "<div>A B</div>")

    def test_select_end_tags_create_p_and_close_custom_children_like_chromium(self) -> None:
        cases = {
            "<select>x</p>": "<select>x<p></p></select>",
            "<select><option></p>": "<select><option><p></p></option></select>",
            "<select><b></br>x": "<select><b><br>x</b></select>",
            "<select><p>x</p>y": "<select><p>x</p>y</select>",
            "<select><form></form>x": "<select><form></form>x</select>",
            "<select><unknown></unknown>x": "<select><unknown></unknown>x</select>",
        }

        for html, expected in cases.items():
            with self.subTest(html=html):
                doc = JustHTML(html, fragment=True, sanitize=False)
                self.assertEqual(doc.to_html(pretty=False), expected)

    def test_select_custom_start_tags_stay_inside_select_like_chromium(self) -> None:
        cases = {
            "<select><form>x": "<select><form>x</form></select>",
            "<select><fieldset>x": "<select><fieldset>x</fieldset></select>",
            "<select><unknown>x": "<select><unknown>x</unknown></select>",
            "<select><foreignObject>x": "<select><foreignobject>x</foreignobject></select>",
            "<select><mi>x": "<select><mi>x</mi></select>",
            "<select><textarea>hi</textarea>x": "<select><textarea>hi</textarea>x</select>",
        }

        for html, expected in cases.items():
            with self.subTest(html=html):
                doc = JustHTML(html, fragment=True, sanitize=False)
                self.assertEqual(doc.to_html(pretty=False), expected)

    def test_select_foreign_start_tags_adjust_names_like_chromium(self) -> None:
        cases = {
            '<select><svg viewbox="0 0"></svg></select>': '<select><svg viewBox="0 0"></svg></select>',
            '<select><math definitionurl="x"></math></select>': '<select><math definitionURL="x"></math></select>',
        }

        for html, expected in cases.items():
            with self.subTest(html=html):
                doc = JustHTML(html, fragment=True, sanitize=False)
                self.assertEqual(doc.to_html(pretty=False), expected)

    def test_select_plaintext_switches_tokenizer_state_like_chromium(self) -> None:
        doc = JustHTML("<select><plaintext><b>x</b></select><p>y", fragment=True, sanitize=False)

        self.assertEqual(
            doc.to_html(pretty=False),
            "<select><plaintext><b>x</b></select><p>y</plaintext></select>",
        )

    def test_template_end_tag_closes_template_from_select_mode(self) -> None:
        cases = {
            "<template><select></template>y": "<template><select></select></template>y",
            "<template><select><option>x</template>y": ("<template><select><option>x</option></select></template>y"),
            "<template><select><optgroup><option>x</template>y": (
                "<template><select><optgroup><option>x</option></optgroup></select></template>y"
            ),
        }

        for html, expected in cases.items():
            with self.subTest(html=html):
                doc = JustHTML(html, fragment=True, sanitize=False)
                self.assertEqual(doc.to_html(pretty=False), expected)

    def test_colgroup_whitespace_preserves_current_colgroup_like_chromium(self) -> None:
        cases = {
            "<table><colgroup> <col></colgroup></table>": "<table><colgroup> <col></colgroup></table>",
            "<table><colgroup> \n\t<col></colgroup></table>": "<table><colgroup> \n\t<col></colgroup></table>",
        }

        for html, expected in cases.items():
            with self.subTest(html=html):
                doc = JustHTML(html, fragment=True, sanitize=False)
                self.assertEqual(doc.to_html(pretty=False), expected)

    def test_colgroup_end_tag_in_row_is_ignored_like_chromium(self) -> None:
        doc = JustHTML("<table><tr></colgroup><td>x</td></tr></table>", fragment=True, sanitize=False)

        self.assertEqual(
            doc.to_html(pretty=False),
            "<table><tbody><tr><td>x</td></tr></tbody></table>",
        )

    def test_escape_disallowed_rawtext_end_tags_preserve_source_order(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags={"p"},
            allowed_attributes={"*": set(), "p": set()},
            url_policy=UrlPolicy(allow_rules={}),
            drop_content_tags=set(),
            disallowed_tag_handling="escape",
        )

        cases = {
            "<script>x</script>": "&lt;script&gt;x&lt;/script&gt;",
            "<script>x</script/>": "&lt;script&gt;x&lt;/script/&gt;",
            "<script DATA-X=1>hi</SCRIPT foo=bar>": "&lt;script DATA-X=1&gt;hi&lt;/SCRIPT foo=bar&gt;",
            "<script><!--x</script>": "&lt;script&gt;&lt;!--x&lt;/script&gt;",
            "<script><!--x</script foo=bar>": "&lt;script&gt;&lt;!--x&lt;/script foo=bar&gt;",
            "<title>x</title>": "&lt;title&gt;x&lt;/title&gt;",
            "<title>x</title/>": "&lt;title&gt;x&lt;/title/&gt;",
            "<title DATA-X=1>hi</TITLE foo=bar>": "&lt;title DATA-X=1&gt;hi&lt;/TITLE foo=bar&gt;",
        }

        for html, expected in cases.items():
            with self.subTest(html=html):
                doc = JustHTML(html, fragment=True, policy=policy)
                self.assertEqual(doc.to_html(pretty=False), expected)

    def test_title_fragment_context_uses_rcdata(self) -> None:
        doc = JustHTML(
            "a&amp;b</title><p>x",
            fragment_context=FragmentContext("title"),
            sanitize=False,
        )

        self.assertEqual(doc.to_html(pretty=False), "a&amp;b&lt;/title&gt;&lt;p&gt;x")

    def test_script_fragment_context_uses_rawtext_without_context_end_tag(self) -> None:
        doc = JustHTML(
            "a</script><p>x",
            fragment_context=FragmentContext("script"),
            sanitize=False,
        )

        self.assertEqual(doc.to_html(pretty=False), "a&lt;/script&gt;&lt;p&gt;x")

    def test_noscript_fragment_context_respects_scripting_flag(self) -> None:
        disabled = JustHTML(
            "<b>x</b>",
            fragment_context=FragmentContext("noscript"),
            scripting_enabled=False,
            sanitize=False,
        )
        enabled = JustHTML(
            "<b>x</b>",
            fragment_context=FragmentContext("noscript"),
            scripting_enabled=True,
            sanitize=False,
        )

        self.assertEqual(disabled.to_html(pretty=False), "<b>x</b>")
        self.assertEqual(enabled.to_html(pretty=False), "&lt;b&gt;x&lt;/b&gt;")

    def test_integration_point_text_like_elements_use_html_text_parsing(self) -> None:
        cases = {
            "<svg><foreignObject><script><b></script></foreignObject></svg>": (
                "<svg><foreignObject><script><b></script></foreignObject></svg>"
            ),
            "<svg><desc><style><b></style></desc></svg>": "<svg><desc><style><b></style></desc></svg>",
            "<svg><title><textarea><b></textarea></title></svg>": (
                "<svg><title><textarea>&lt;b&gt;</textarea></title></svg>"
            ),
            "<math><mi><textarea><b></textarea></mi></math>": ("<math><mi><textarea>&lt;b&gt;</textarea></mi></math>"),
            "<math><mtext><style><b></style></mtext></math>": ("<math><mtext><style><b></style></mtext></math>"),
            '<math><annotation-xml encoding="text/html"><script><b></script></annotation-xml></math>': (
                '<math><annotation-xml encoding="text/html"><script><b></script></annotation-xml></math>'
            ),
        }

        for html, expected in cases.items():
            with self.subTest(html=html):
                doc = JustHTML(html, fragment=True, sanitize=False)
                self.assertEqual(doc.to_html(pretty=False), expected)

    def test_mathml_text_integration_end_tag_breakouts_stay_inside_integration_point(self) -> None:
        cases = {
            "<math><mi></p>": "<math><mi><p></p></mi></math>",
            "<math><mtext></p>": "<math><mtext><p></p></mtext></math>",
            "<math><mi></br>x": "<math><mi><br>x</mi></math>",
        }

        for html, expected in cases.items():
            with self.subTest(html=html):
                doc = JustHTML(html, fragment=True, sanitize=False)
                self.assertEqual(doc.to_html(pretty=False), expected)

    def test_annotation_xml_without_html_encoding_does_not_use_rawtext(self) -> None:
        cases = {
            "<math><annotation-xml><script><b></script></annotation-xml></math>": (
                "<math><annotation-xml><script></script></annotation-xml></math><b></b>"
            ),
            "<math><annotation-xml data-x=1><script><b></script></annotation-xml></math>": (
                '<math><annotation-xml data-x="1"><script></script></annotation-xml></math><b></b>'
            ),
        }

        for html, expected in cases.items():
            with self.subTest(html=html):
                doc = JustHTML(html, fragment=True, sanitize=False)
                self.assertEqual(doc.to_html(pretty=False), expected)

    def test_form_end_tag_generates_implied_end_tags(self) -> None:
        cases = {
            "<form><p></form><input>": "<form><p></p></form><input>",
            "<form><li></form><input>": "<form><li></li></form><input>",
            "<form><dd></form><input>": "<form><dd></dd></form><input>",
        }

        for html, expected in cases.items():
            with self.subTest(html=html):
                doc = JustHTML(html, fragment=True, sanitize=False)
                self.assertEqual(doc.to_html(pretty=False), expected)

    def test_forms_in_template_do_not_set_form_pointer(self) -> None:
        cases = {
            "<template><form></template><form><input>": ("<template><form></form></template><form><input></form>"),
            "<template></form></template>": "<template></template>",
            "<template><div></form></template>": "<template><div></div></template>",
            "<template><form><div></form></template>": "<template><form><div></div></form></template>",
            "<form><template><form><input></template><input name=outer>": (
                '<form><template><form><input></form></template><input name="outer"></form>'
            ),
            "<template><table><form><tr><td>x</template><form><input>": (
                "<template><table><form></form><tbody><tr><td>x</td></tr></tbody></table></template>"
                "<form><input></form>"
            ),
        }

        for html, expected in cases.items():
            with self.subTest(html=html):
                doc = JustHTML(html, fragment=True, sanitize=False)
                self.assertEqual(doc.to_html(pretty=False), expected)

    def test_fragment_eof_inside_nested_template_does_not_crash(self) -> None:
        doc = JustHTML("<template></script><template>", fragment=True)

        self.assertEqual(doc.to_html(pretty=False), "")

    def test_fragment_eof_inside_malformed_rawtext_template_does_not_crash(self) -> None:
        html = "<template></script></td></p><template><svg><foreignObject><textarea><style></mi>"
        doc = JustHTML(html, fragment=True)

        self.assertEqual(doc.to_html(pretty=False), "")

    def test_fragment_after_head_template_placeholder_does_not_crash(self) -> None:
        doc = JustHTML("<template></style><template></style>", fragment=True)

        self.assertEqual(doc.to_html(pretty=False), "")

    def test_null_in_body_text_is_removed(self) -> None:
        doc = JustHTML("<body>a\x00b</body>", collect_errors=True)
        text = doc.to_text(strip=False)
        self.assertEqual(text, "ab")
        self.assertNotIn("\x00", text)

    def test_only_null_in_body_text_becomes_empty(self) -> None:
        doc = JustHTML("<body>\x00</body>", collect_errors=True)
        text = doc.to_text(strip=False)
        self.assertEqual(text, "")
