import unittest

from justhtml import JustHTML, SanitizationPolicy
from justhtml.dom import Element
from justhtml.parser.context import FragmentContext
from justhtml.sanitizer import UrlPolicy
from justhtml.tokenizer import Tokenizer, TokenizerOpts
from justhtml.treebuilder import InsertionMode, TreeBuilder


def _set_open_elements(tree_builder, elements):
    tree_builder.open_elements = elements
    tree_builder._open_p_elements = 0
    for element in elements:
        tree_builder._note_open_element_pushed(element)


class TestTreeBuilder(unittest.TestCase):
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

    def test_scope_checks_skip_placeholder_stack_entries(self) -> None:
        tree_builder = TreeBuilder()
        html = tree_builder._create_element("html", None, {})
        body = tree_builder._create_element("body", None, {})
        p = tree_builder._create_element("p", None, {})
        _set_open_elements(tree_builder, [html, None, body, None, p])

        self.assertTrue(tree_builder._has_element_in_scope("body"))
        self.assertTrue(tree_builder._has_element_in_scope("p"))

    def test_any_other_end_tag_skips_placeholder_stack_entries(self) -> None:
        tree_builder = TreeBuilder(collect_errors=True)
        html = tree_builder._create_element("html", None, {})
        body = tree_builder._create_element("body", None, {})
        span = tree_builder._create_element("span", None, {})
        _set_open_elements(tree_builder, [html, None, body, span, None])

        tree_builder._any_other_end_tag("span")

        self.assertEqual(tree_builder.open_elements, [html, None, body])

    def test_generate_implied_end_tags_stops_at_placeholder_stack_entry(self) -> None:
        tree_builder = TreeBuilder()
        html = tree_builder._create_element("html", None, {})
        p = tree_builder._create_element("p", None, {})
        _set_open_elements(tree_builder, [html, None, p, None])

        tree_builder._generate_implied_end_tags()

        self.assertEqual(tree_builder.open_elements, [html, None, p, None])

    def test_null_in_body_text_is_removed(self) -> None:
        doc = JustHTML("<body>a\x00b</body>", collect_errors=True)
        text = doc.to_text(strip=False)
        self.assertEqual(text, "ab")
        self.assertNotIn("\x00", text)

    def test_only_null_in_body_text_becomes_empty(self) -> None:
        doc = JustHTML("<body>\x00</body>", collect_errors=True)
        text = doc.to_text(strip=False)
        self.assertEqual(text, "")

    def test_process_characters_strips_null_and_appends(self) -> None:
        tree_builder = TreeBuilder(collect_errors=True)
        tree_builder.mode = InsertionMode.IN_BODY
        tree_builder.open_elements.append(Element("body", {}, None))

        tree_builder.process_characters("a\x00b")
        body = tree_builder.open_elements[-1]
        self.assertEqual(len(body.children), 1)
        self.assertEqual(body.children[0].data, "ab")
        self.assertEqual([error.code for error in tree_builder.errors], ["invalid-codepoint"])

    def test_process_characters_only_null_returns_continue(self) -> None:
        tree_builder = TreeBuilder(collect_errors=True)
        tree_builder.mode = InsertionMode.IN_BODY
        tree_builder.open_elements.append(Element("body", {}, None))

        tree_builder.process_characters("\x00")
        body = tree_builder.open_elements[-1]
        self.assertEqual(body.children, [])
        self.assertEqual([error.code for error in tree_builder.errors], ["invalid-codepoint"])

    def test_process_characters_empty_returns_continue(self) -> None:
        tree_builder = TreeBuilder(collect_errors=True)
        tree_builder.mode = InsertionMode.IN_BODY
        tree_builder.open_elements.append(Element("body", {}, None))

        tree_builder.process_characters("")
        body = tree_builder.open_elements[-1]
        self.assertEqual(body.children, [])

    def test_append_comment_tracking_when_start_pos_unknown(self) -> None:
        tree_builder = TreeBuilder(collect_errors=False)
        tokenizer = Tokenizer(
            tree_builder,
            TokenizerOpts(),
            collect_errors=False,
            track_node_locations=True,
        )
        tokenizer.initialize("")
        tokenizer.last_token_start_pos = None
        tree_builder.tokenizer = tokenizer

        tree_builder._append_comment_to_document("x")
        assert tree_builder.document.children is not None
        node = tree_builder.document.children[-1]
        assert node.name == "#comment"
        assert node.origin_offset is None
        assert node.origin_location is None

    def test_append_comment_inside_element_start_pos_unknown(self) -> None:
        tree_builder = TreeBuilder(collect_errors=False)

        html = tree_builder._create_element("html", None, {})
        body = tree_builder._create_element("body", None, {})
        tree_builder.document.append_child(html)
        html.append_child(body)
        _set_open_elements(tree_builder, [html, body])

        tokenizer = Tokenizer(
            tree_builder,
            TokenizerOpts(),
            collect_errors=False,
            track_node_locations=True,
        )
        tokenizer.initialize("")
        tokenizer.last_token_start_pos = None
        tree_builder.tokenizer = tokenizer

        tree_builder._append_comment("x", parent=body)
        assert body.children
        node = body.children[-1]
        assert node.name == "#comment"
        assert node.origin_offset is None
        assert node.origin_location is None

    def test_append_text_foster_parenting_start_pos_unknown(self) -> None:
        tree_builder = TreeBuilder(collect_errors=False)

        html = tree_builder._create_element("html", None, {})
        body = tree_builder._create_element("body", None, {})
        table = tree_builder._create_element("table", None, {})
        tree_builder.document.append_child(html)
        html.append_child(body)
        body.append_child(table)
        _set_open_elements(tree_builder, [html, body, table])

        tokenizer = Tokenizer(
            tree_builder,
            TokenizerOpts(),
            collect_errors=False,
            track_node_locations=True,
        )
        tokenizer.initialize("")
        tokenizer.last_token_start_pos = None
        tree_builder.tokenizer = tokenizer

        tree_builder._append_text("hi")

        def walk(n):
            yield n
            children = getattr(n, "children", None)
            if children:
                for c in children:
                    yield from walk(c)

        texts = [
            n
            for n in walk(tree_builder.document)
            if getattr(n, "name", None) == "#text" and getattr(n, "data", None) == "hi"
        ]
        assert texts
        assert texts[0].origin_offset is None
        assert texts[0].origin_location is None

    def test_append_text_fast_path_start_pos_unknown(self) -> None:
        tree_builder = TreeBuilder(collect_errors=False)

        html = tree_builder._create_element("html", None, {})
        body = tree_builder._create_element("body", None, {})
        div = tree_builder._create_element("div", None, {})
        tree_builder.document.append_child(html)
        html.append_child(body)
        body.append_child(div)
        _set_open_elements(tree_builder, [html, body, div])

        tokenizer = Tokenizer(
            tree_builder,
            TokenizerOpts(),
            collect_errors=False,
            track_node_locations=True,
        )
        tokenizer.initialize("")
        tokenizer.last_token_start_pos = None
        tree_builder.tokenizer = tokenizer

        tree_builder._append_text("hi")
        assert div.children
        node = div.children[0]
        assert node.name == "#text"
        assert node.data == "hi"
        assert node.origin_offset is None
        assert node.origin_location is None
