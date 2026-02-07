import textwrap
import unittest

from justhtml import HTMLContext, UrlRule
from justhtml import JustHTML as _JustHTML
from justhtml.context import FragmentContext
from justhtml.node import Comment, DocumentFragment, Node, Template, Text
from justhtml.serialize import (
    _can_unquote_attr_value,
    _choose_attr_quote,
    _collapse_html_whitespace,
    _escape_attr_value,
    _escape_text,
    _is_blocky_element,
    _is_formatting_whitespace_text,
    _is_layout_blocky_element,
    _normalize_formatting_whitespace,
    _should_pretty_indent_children,
    serialize_end_tag,
    serialize_start_tag,
    to_html,
    to_test_format,
)


def JustHTML(*args, **kwargs):  # noqa: N802
    if "sanitize" not in kwargs and "safe" not in kwargs:
        kwargs["sanitize"] = False
    return _JustHTML(*args, **kwargs)


class TestSerialize(unittest.TestCase):
    def test_basic_document(self):
        html = "<!DOCTYPE html><html><head><title>Test</title></head><body><p>Hello</p></body></html>"
        doc = JustHTML(html)
        output = doc.to_html()
        assert "<!DOCTYPE html>" in output
        assert "<title>Test</title>" in output
        assert "<p>Hello</p>" in output

    def test_safe_document_serialization_preserves_document_wrappers(self):
        doc = JustHTML("<p>Hi</p>")
        output = doc.to_html(pretty=False)
        assert output == "<html><head></head><body><p>Hi</p></body></html>"

    def test_fragment_parameter_default_context(self):
        doc = JustHTML("<p>Hi</p>", fragment=True)
        assert doc.root.name == "#document-fragment"

        output = doc.to_html(pretty=False)
        assert "<html>" not in output
        assert output == "<p>Hi</p>"

    def test_fragment_parameter_respects_explicit_fragment_context(self):
        # <tr> only parses correctly in table-related fragment contexts.
        doc = JustHTML(
            "<tr><td>cell</td></tr>",
            fragment=True,
            fragment_context=FragmentContext("tbody"),
        )
        output = doc.to_html(pretty=False)
        assert output == "<tr><td>cell</td></tr>"

    def test_pretty_template_serializes_template_content(self):
        frag = DocumentFragment()
        template = Template("template", namespace="html")
        frag.append_child(template)

        p = Node("p")
        p.append_child(Text("x"))
        assert template.template_content is not None
        template.template_content.append_child(p)

        out = to_html(frag, pretty=True)
        assert "<template>" in out
        assert "<p>" in out
        assert out.index("<template>") < out.index("<p>") < out.index("</template>")

    def test_collapse_html_whitespace_vertical_tab(self):
        # \v is not HTML whitespace, so it should be preserved as a non-whitespace character
        # while surrounding whitespace is collapsed.
        text = "  a  \v  b  "
        # Expected: "a \v b" because \v is treated as a regular character,
        # so "  a  " -> "a ", "\v", "  b  " -> " b"
        # Wait, let's trace the logic:
        # "  a  \v  b  "
        # parts: [" ", "a", " ", "\v", " ", "b", " "] (roughly)
        # joined: " a \v b " -> stripped: "a \v b"
        self.assertEqual(_collapse_html_whitespace(text), "a \v b")

    def test_can_unquote_attr_value_coverage(self):
        self.assertFalse(_can_unquote_attr_value(None))
        self.assertTrue(_can_unquote_attr_value("foo"))
        self.assertFalse(_can_unquote_attr_value("foo bar"))
        self.assertFalse(_can_unquote_attr_value("foo=bar"))
        self.assertFalse(_can_unquote_attr_value("foo'bar"))
        self.assertFalse(_can_unquote_attr_value('foo"bar'))
        # < is allowed in unquoted attribute values in HTML5
        self.assertTrue(_can_unquote_attr_value("foo<bar"))

    def test_attributes(self):
        html = '<div id="test" class="foo" data-val="x&y"></div>'
        doc = JustHTML(html)
        output = doc.to_html()
        assert 'id="test"' in output
        assert 'class="foo"' in output
        assert 'data-val="x&amp;y"' in output  # Check escaping

    def test_text_escaping(self):
        frag = DocumentFragment()
        div = Node("div")
        frag.append_child(div)
        div.append_child(Text("a<b&c"))
        output = to_html(frag, pretty=False)
        assert output == "<div>a&lt;b&amp;c</div>"

    def test_to_html_context_url(self):
        # Test serialization with HTMLContext.URL
        # Case 1: Simple text node
        node = Text("https://example.com/foo bar")
        output = to_html(node, context=HTMLContext.URL)
        # Expect percent encoding of space
        self.assertEqual(output, "https://example.com/foo%20bar")

        # Case 2: Element with text content
        div = Node("div")
        div.append_child(Text("  /path/to/thing?q=a b  "))
        # to_html with URL context should strip whitespace and encode
        output = to_html(div, context=HTMLContext.URL)
        self.assertEqual(output, "/path/to/thing?q=a%20b")

    def test_to_html_inner_html_js_context(self):
        output = _JustHTML.escape_html_text_in_js_string('<div a="b">Hi &amp; <b>world</b></div>')
        # & escapes to \u0026
        # " escapes to \"
        # <, > were already escaped to &lt;, &gt; by html escaping, then the & in those is escaped
        # but let's see. &lt; -> \u0026lt;
        # Actually, let's just make it exact.
        expected = r"&lt;div a=\"b\"&gt;Hi &amp;amp; &lt;b&gt;world&lt;/b&gt;&lt;/div&gt;"
        assert output == expected

    def test_to_html_inner_html_js_context_single_quote(self):
        output = _JustHTML.escape_html_text_in_js_string("<p>It's ok</p>", quote="'")
        expected = r"&lt;p&gt;It\'s ok&lt;/p&gt;"
        assert output == expected

    def test_escape_html_text_in_js_string_noop_when_no_html_chars(self):
        # Covers the fast-path in _escape_html_text().
        assert _JustHTML.escape_html_text_in_js_string("plain text") == "plain text"

    def test_to_html_js_string_context(self):
        doc = JustHTML("<b>Hi</b>", fragment=True)
        output = doc.to_html(pretty=False, context=HTMLContext.JS_STRING)
        # < -> \u003c, > -> \u003e
        assert output == r"\u003cb\u003eHi\u003c/b\u003e"

    def test_escape_url_in_js_string(self):
        output = _JustHTML.escape_url_in_js_string("/path with space?x=1&y=2")
        # = -> \u003d, & -> \u0026
        assert output == r"/path%20with%20space?x=1&y=2"

    def test_clean_url_value_requires_rule(self):
        rule = UrlRule(allowed_schemes={"https"})
        assert _JustHTML.clean_url_value(value="https://example.com/x", url_rule=rule) == "https://example.com/x"

        assert _JustHTML.clean_url_value(value="javascript:alert(1)", url_rule=rule) is None

    def test_clean_url_in_js_string(self):
        rule = UrlRule(allowed_schemes={"https"})
        output = _JustHTML.clean_url_in_js_string(
            value="https://example.com/path with space?x=1&y=2",
            url_rule=rule,
        )
        assert output == r"https://example.com/path%20with%20space?x=1&y=2"

        # Test disallowed URL returns None
        assert _JustHTML.clean_url_in_js_string(value="javascript:alert(1)", url_rule=rule) is None

    def test_escape_attr_value(self):
        output = _JustHTML.escape_attr_value('" onerror="alert(1)', quote='"')
        assert output == "&quot; onerror=&quot;alert(1)"

    def test_to_html_html_attr_value_context(self):
        doc = JustHTML("<b>Hi</b>", fragment=True)
        output = doc.to_html(pretty=False, context=HTMLContext.HTML_ATTR_VALUE, quote='"')
        assert output == "&lt;b&gt;Hi&lt;/b&gt;"

    def test_to_html_html_attr_value_context_escapes_quotes(self):
        doc = JustHTML('<p class="x">Hi</p>', fragment=True)
        output = doc.to_html(pretty=False, context=HTMLContext.HTML_ATTR_VALUE, quote='"')
        assert output == "&lt;p class=&quot;x&quot;&gt;Hi&lt;/p&gt;"

    def test_to_html_html_js_string_context_escapes_quotes(self):
        doc = JustHTML('<p class="x">Hi</p>', fragment=True)
        output = doc.to_html(pretty=False, context=HTMLContext.JS_STRING)
        # <u003c p class=\u003d\"x\"\u003e...
        assert output == r"\u003cp class=\"x\"\u003eHi\u003c/p\u003e"

    def test_to_html_unknown_context_raises(self):
        doc = JustHTML("<p>Hi</p>", fragment=True)
        with self.assertRaises(TypeError):
            _ = doc.to_html(pretty=False, context="nope")

    def test_void_elements(self):
        html = "<br><hr><img>"
        doc = JustHTML(html)
        output = doc.to_html()
        assert "<br>" in output
        assert "<hr>" in output
        assert "<img>" in output
        assert "</br>" not in output

    def test_comments(self):
        html = "<!-- hello world -->"
        doc = JustHTML(html)
        output = doc.to_html()
        assert "<!-- hello world -->" in output

    def test_document_fragment(self):
        # Manually create a document fragment since parser returns Document
        frag = DocumentFragment()
        child = Node("div")
        frag.append_child(child)
        output = to_html(frag)
        assert "<div></div>" in output

    def test_text_only_children(self):
        html = "<div>Text only</div>"
        doc = JustHTML(html)
        output = doc.to_html()
        assert "<div>Text only</div>" in output

    def test_pretty_text_only_element_collapses_whitespace(self):
        doc = JustHTML("<h3>\n\nSorry to interrupt, but we're short on time to hit our goal.\n</h3>")
        h3 = doc.query("h3")[0]
        assert h3.to_html(pretty=True) == ("<h3>Sorry to interrupt, but we're short on time to hit our goal.</h3>")

    def test_pretty_text_only_div_collapses_whitespace(self):
        html = (
            '<div class="overlay-banner-main-footer-cta">'
            "\n\nWe ask you, sincerely: don't skip this, join the 2% of readers who give.\n"
            "</div>"
        )
        doc = JustHTML(html)
        div = doc.query("div")[0]
        assert div.to_html(pretty=True) == (
            '<div class="overlay-banner-main-footer-cta">'
            "We ask you, sincerely: don't skip this, join the 2% of readers who give."
            "</div>"
        )

    def test_pretty_block_container_splits_on_formatting_whitespace_runs(self):
        # Wikipedia-like: large spacing/newlines between inline-ish siblings should become line breaks,
        # even when there is some non-whitespace text in the flow (which disables the simpler
        # "all-children-are-elements" multiline rule).
        div = Node("div")
        div.append_child(Node("span"))
        div.append_child(Text("      "))
        div.append_child(Node("a"))
        div.append_child(Text("\n  \t"))
        div.append_child(Text("Search"))
        div.append_child(Text("     "))
        div.append_child(Node("ul"))

        output = div.to_html(pretty=True)
        expected = textwrap.dedent(
            """\
            <div>
              <span></span>
              <a></a>
              Search
              <ul></ul>
            </div>
            """
        ).strip("\n")
        assert output == expected

    def test_pretty_block_container_falls_back_when_run_contains_blocky_and_inline(self):
        # If a "run" contains a blocky element + other nodes, we skip the smart
        # run-splitting layout and fall back to the existing compact-pretty logic.
        div = Node("div")
        div.append_child(Node("span"))
        div.append_child(Text("     "))
        div.append_child(Node("ul"))
        div.append_child(Node("span"))  # Adjacent to <ul> (no whitespace) => same run
        div.append_child(Text("\n  \t"))
        div.append_child(Node("a"))
        div.append_child(Text("     "))
        div.append_child(Text("Tail"))

        output = div.to_html(pretty=True)
        expected = textwrap.dedent(
            """\
            <div>
              <span></span>
              <ul></ul>
              <span></span>
              <a></a>
              Tail
            </div>
            """
        ).strip("\n")
        assert output == expected

    def test_pretty_block_container_keeps_inline_run_on_one_line(self):
        div = Node("div")
        div.append_child(Text("Hello"))
        div.append_child(Text(" "))
        div.append_child(Node("em"))
        div.append_child(Text(" world"))
        assert div.to_html(pretty=True) == "<div>Hello <em></em> world</div>"

    def test_pretty_block_container_breaks_before_block_child_even_with_small_spaces(self):
        # Wikipedia-like: block element begins after only small spacing, so the
        # "formatting whitespace separator" heuristic doesn't trigger.
        # We still want the block subtree (and its indentation) to be preserved.
        div = Node("div")
        div.append_child(Text("Main menu  Navigation  "))

        ul = Node("ul")
        li = Node("li")
        li.append_child(Text("Item"))
        ul.append_child(li)
        div.append_child(ul)

        assert div.to_html(pretty=True) == textwrap.dedent(
            """\
            <div>
              Main menu  Navigation
              <ul>
                <li>Item</li>
              </ul>
            </div>
            """
        ).strip("\n")

    def test_is_formatting_whitespace_text(self):
        assert _is_formatting_whitespace_text("") is False
        assert _is_formatting_whitespace_text("  ") is False
        assert _is_formatting_whitespace_text("   ") is True
        assert _is_formatting_whitespace_text("\n") is True

    def test_is_blocky_element_returns_false_for_text_comment_and_doctype_nodes(self):
        assert _is_blocky_element(Text("x")) is False
        assert _is_blocky_element(Comment(data="c")) is False
        assert _is_blocky_element(Node("!doctype")) is False

    def test_is_blocky_element_returns_false_for_objects_without_children(self):
        class _NameOnly:
            name = "span"

        assert _is_blocky_element(_NameOnly()) is False

    def test_is_layout_blocky_element_returns_false_for_objects_without_name(self):
        assert _is_layout_blocky_element(object()) is False

    def test_is_layout_blocky_element_returns_false_for_comment_node(self):
        assert _is_layout_blocky_element(Comment(data="c")) is False

    def test_is_layout_blocky_element_true_for_layout_blocks_and_descendants(self):
        assert _is_layout_blocky_element(Node("div")) is True

        # Descendant scan: wrapper is inline, but contains a layout block (<ul>).
        wrapper = Node("span")
        inner = Node("span")
        inner.append_child(Node("ul"))
        # Put None last so it gets popped first (covers the None-skip branch).
        wrapper.children = [inner, None]
        assert _is_layout_blocky_element(wrapper) is True

        # Grandchild scan path that does not find a layout block.
        outer = Node("span")
        mid = Node("span")
        mid.append_child(Node("em"))
        outer.append_child(mid)
        assert _is_layout_blocky_element(outer) is False

    def test_is_layout_blocky_element_returns_false_for_objects_without_children(self):
        class _NameOnly:
            name = "span"

        assert _is_layout_blocky_element(_NameOnly()) is False

    def test_pretty_block_container_smart_mode_keeps_small_spacing_inside_run(self):
        # Ensure smart run-splitting covers whitespace-only nodes inside runs
        # (" ") and renders inline elements via _node_to_html(..., indent=0).
        div = Node("div")
        div.append_child(Node("span"))
        div.append_child(Text("\n  "))
        div.append_child(Text("Hello"))
        div.append_child(Text(" "))
        div.append_child(Node("em"))
        div.append_child(Text(" world"))
        div.append_child(Text("\n"))
        div.append_child(Node("a"))

        assert div.to_html(pretty=True) == textwrap.dedent(
            """\
            <div>
              <span></span>
              Hello <em></em> world
              <a></a>
            </div>
            """
        ).strip("\n")

    def test_pretty_block_container_smart_mode_ignores_leading_and_trailing_formatting_whitespace(self):
        # Covers trimming of leading/trailing formatting whitespace in smart-mode.
        div = Node("div")
        div.append_child(Text("\n"))
        div.append_child(Node("span"))
        div.append_child(Text("\n"))
        div.append_child(Node("a"))
        div.append_child(Text(" "))
        div.append_child(Text("Tail"))
        div.append_child(Text("\n"))

        assert div.to_html(pretty=True) == textwrap.dedent(
            """\
            <div>
              <span></span>
              <a></a> Tail
            </div>
            """
        ).strip("\n")

    def test_pretty_block_container_smart_mode_collapses_consecutive_separators(self):
        # Two adjacent formatting-whitespace text nodes should act like a single separator.
        div = Node("div")

        # Note: append_child() may merge adjacent text nodes, so set children directly
        # to guarantee two distinct formatting-whitespace nodes.
        div.children = [
            Node("span"),
            Text("\n"),
            Text("\n  \t"),
            Node("a"),
            # Make the container mixed-content so we actually take the smart
            # run-splitting path (the simpler multiline indentation path is disabled
            # when there is non-whitespace text).
            Text(" Tail"),
        ]

        assert div.to_html(pretty=True) == textwrap.dedent(
            """\
            <div>
              <span></span>
              <a></a> Tail
            </div>
            """
        ).strip("\n")

    def test_pretty_block_container_multiline_mode_skips_none_and_edge_whitespace(self):
        # Exercise None-child skipping and leading/trailing whitespace-only handling.
        div = Node("div")
        ul = Node("ul")
        ul.append_child(Node("li"))
        div.children = [
            None,
            Text("   "),
            Text("Text"),
            ul,
            Text("   "),
            None,
        ]

        assert div.to_html(pretty=True) == textwrap.dedent(
            """\
            <div>
              Text
              <ul>
                <li></li>
              </ul>
            </div>
            """
        ).strip("\n")

    def test_pretty_text_only_element_empty_text_is_dropped(self):
        h3 = Node("h3")
        h3.append_child(Text(""))
        assert h3.to_html(pretty=True) == "<h3></h3>"

    def test_pretty_text_only_script_preserves_whitespace(self):
        doc = JustHTML("<script>\n  var x = 1;\n\n  var y = 2;\n</script>")
        script = doc.query("script")[0]
        assert script.to_html(pretty=True) == ("<script>\n  var x = 1;\n\n  var y = 2;\n</script>")

    def test_compact_mode_does_not_normalize_script_text_children(self):
        # Artificial tree to cover serializer branch: skip normalization inside rawtext elements.
        script = Node("script")
        script.append_child(Text("a\nb"))
        script.append_child(Node("span"))
        assert script.to_html(pretty=True) == "<script>a\nb<span></span></script>"

    def test_mixed_children(self):
        html = "<div>Text <span>Span</span></div>"
        doc = JustHTML(html)
        div = doc.query("div")[0]
        output = div.to_html(pretty=True)
        assert output == "<div>Text <span>Span</span></div>"

    def test_mixed_children_normalizes_newlines_in_text_nodes(self):
        html = '<span>100+\n <span class="jsl10n">articles</span></span>'
        doc = JustHTML(html)
        outer = doc.query("span")[0]
        output = outer.to_html(pretty=True)
        assert "100+ <span" in output
        assert "100+\n" not in output

        html2 = '<span class="text">\n100+\n <span class="jsl10n">articles</span></span>'
        doc2 = JustHTML(html2)
        outer2 = doc2.query("span")[0]
        output2 = outer2.to_html(pretty=True)
        assert ">100+ <span" in output2

    def test_mixed_children_preserves_pure_space_runs(self):
        html = "<p>Hello  <b>world</b></p>"
        doc = JustHTML(html)
        p = doc.query("p")[0]
        assert p.to_html(pretty=True) == "<p>Hello  <b>world</b></p>"

    def test_normalize_formatting_whitespace_empty(self):
        assert _normalize_formatting_whitespace("") == ""

    def test_normalize_formatting_whitespace_no_formatting_chars(self):
        assert _normalize_formatting_whitespace("a  b") == "a  b"

    def test_normalize_formatting_whitespace_preserves_double_spaces(self):
        # Contains a newline (so normalization runs), but the double-space run does
        # not include formatting whitespace and should be preserved.
        assert _normalize_formatting_whitespace("a  b\nc") == "a  b c"

    def test_normalize_formatting_whitespace_collapses_newlines_and_tabs(self):
        assert _normalize_formatting_whitespace("a\n\t  b") == "a b"

    def test_normalize_formatting_whitespace_trims_edges_from_newlines(self):
        assert _normalize_formatting_whitespace("\n100+\n ") == "100+ "
        assert _normalize_formatting_whitespace(" hi\n") == " hi"

    def test_pretty_print_does_not_insert_spaces_in_inline_mixed_content(self):
        html = (
            '<code class="constructorsynopsis cpp">'
            '<span class="methodname">BApplication</span>'
            '(<span class="methodparam">'
            '<span class="modifier">const </span>'
            '<span class="type">char* </span>'
            '<span class="parameter">signature</span>'
            "</span>);"
            "</code>"
        )
        doc = JustHTML(html)
        code = doc.query("code")[0]

        pretty_html = code.to_html(pretty=True)
        assert "</span>(<span" in pretty_html

        rendered_text = JustHTML(pretty_html).to_text(separator="", strip=False)
        assert rendered_text == "BApplication(const char* signature);"

    def test_empty_attributes(self):
        html = "<input disabled>"
        doc = JustHTML(html)
        output = doc.to_html()
        assert "<input disabled>" in output

    def test_none_attributes(self):
        # Manually create node with None attribute value
        node = Node("div")
        node.attrs = {"data-test": None}
        output = to_html(node)
        assert "<div data-test></div>" in output

    def test_empty_string_attribute(self):
        html = '<div data-val=""></div>'
        doc = JustHTML(html)
        output = doc.to_html()
        assert "<div data-val></div>" in output

    def test_boolean_attribute_value_case_insensitive_minimization(self):
        # Attribute values that match the attribute name case-insensitively should be minimized.
        assert serialize_start_tag("input", {"disabled": "DISABLED"}) == "<input disabled>"

    def test_serialize_start_tag_quotes(self):
        # Prefer single quotes if the value contains a double quote but no single quote
        tag = serialize_start_tag("span", {"title": 'foo"bar'})
        assert tag == "<span title='foo\"bar'>"

        # Otherwise use double quotes and escape embedded double quotes
        tag = serialize_start_tag("span", {"title": "foo'bar\"baz"})
        assert tag == '<span title="foo\'bar&quot;baz">'

        # Always quote normal attribute values
        assert serialize_start_tag("span", {"title": "foo"}) == '<span title="foo">'

    def test_serialize_start_tag_unquoted_mode_and_escape_lt(self):
        # In unquoted mode, escape '&' and '<'.
        assert (
            serialize_start_tag(
                "span",
                {"title": "a<b&c"},
                quote_attr_values=False,
            )
            == "<span title=a&lt;b&amp;c>"
        )

    def test_serialize_start_tag_none_attr_non_minimized(self):
        # When boolean minimization is disabled, None becomes an explicit empty value.
        assert (
            serialize_start_tag(
                "span",
                {"disabled": None},
                minimize_boolean_attributes=False,
            )
            == '<span disabled="">'
        )

    def test_serialize_end_tag(self):
        assert serialize_end_tag("span") == "</span>"

    def test_serializer_private_helpers_none(self):
        assert _escape_text(None) == ""
        assert _choose_attr_quote(None) == '"'
        assert _escape_attr_value(None, '"') == ""

        # Covered for branch completeness: unquote check rejects None.
        assert _can_unquote_attr_value(None) is False

    def test_mixed_content_whitespace(self):
        html = "<div>   <p></p></div>"
        doc = JustHTML(html)
        output = doc.to_html()
        assert "<div>" in output
        assert "<p></p>" in output

    def test_pretty_indent_skips_whitespace_text_nodes(self):
        div = Node("div")
        div.append_child(Text("\n  "))
        div.append_child(Node("p"))
        div.append_child(Text("\n"))
        output = div.to_html(pretty=True)
        expected = textwrap.dedent(
            """\
            <div>
              <p></p>
            </div>
            """
        ).strip("\n")
        assert output == expected

    def test_pretty_indent_children_indents_inline_elements_in_block_containers(self):
        div = Node("div")
        div.append_child(Node("span"))
        output = div.to_html(pretty=True)
        assert output == "<div>\n  <span></span>\n</div>"

    def test_pretty_indent_children_indents_adjacent_inline_elements_in_block_containers(self):
        div = Node("div")
        div.append_child(Node("span"))
        div.append_child(Node("em"))
        output = div.to_html(pretty=True)
        assert output == "<div>\n  <span></span>\n  <em></em>\n</div>"

    def test_compact_pretty_collapses_large_whitespace_gaps(self):
        # Two adjacent inline children are rendered in compact mode.
        # For block-ish containers with whitespace separators, compact pretty upgrades
        # to a multiline layout for readability.
        div = Node("div")
        div.append_child(Node("span"))
        div.append_child(Text("     "))
        div.append_child(Node("span"))
        output = div.to_html(pretty=True)
        expected = textwrap.dedent(
            """\
            <div>
              <span></span>
              <span></span>
            </div>
            """
        ).strip("\n")
        assert output == expected

    def test_compact_pretty_collapses_newline_whitespace_to_single_space(self):
        # Newlines/tabs between inline siblings upgrade to multiline layout.
        div = Node("div")
        div.append_child(Node("span"))
        div.append_child(Text("\n  \t"))
        div.append_child(Node("span"))
        output = div.to_html(pretty=True)
        expected = textwrap.dedent(
            """\
            <div>
              <span></span>
              <span></span>
            </div>
            """
        ).strip("\n")
        assert output == expected

    def test_should_pretty_indent_children_no_element_children(self):
        children = [Text("  "), Text("\n")]
        assert _should_pretty_indent_children(children) is True

    def test_should_pretty_indent_children_skips_none_children(self):
        # Defensive: children lists can contain None.
        children = [None, Text(" "), None]
        assert _should_pretty_indent_children(children) is True

    def test_should_pretty_indent_children_single_special_element_child(self):
        # Single block-ish child should allow indentation.
        children = [Node("div")]
        assert _should_pretty_indent_children(children) is True

    def test_should_pretty_indent_children_single_inline_element_child(self):
        # Single inline/phrasing child should *not* allow indentation.
        children = [Node("span")]
        assert _should_pretty_indent_children(children) is False

    def test_should_pretty_indent_children_single_inline_child_with_block_descendant(self):
        # Inline wrapper that contains block-ish content should be treated as block-ish.
        wrapper = Node("span")
        wrapper.append_child(Node("div"))
        assert _should_pretty_indent_children([wrapper]) is True

    def test_should_pretty_indent_children_allows_adjacent_blocky_inline_children(self):
        # Two inline elements that both contain block-ish descendants should allow indentation.
        a1 = Node("a")
        a1.append_child(Node("div"))
        a2 = Node("a")
        a2.append_child(Node("div"))
        assert _should_pretty_indent_children([a1, a2]) is True

    def test_is_blocky_element_returns_false_for_objects_without_name(self):
        assert _is_blocky_element(object()) is False

    def test_is_blocky_element_scans_grandchildren(self):
        # Ensure the descendant scanning path is exercised even when no block-ish
        # elements are found.
        outer = Node("span")
        mid = Node("span")
        mid.append_child(Node("em"))
        outer.append_child(mid)
        assert _is_blocky_element(outer) is False

    def test_should_pretty_indent_children_single_anchor_with_special_grandchild(self):
        # Anchor wrapping block-ish content should be treated as block-ish.
        link = Node("a")
        link.children = [None, Node("div")]
        assert _should_pretty_indent_children([link]) is True

    def test_compact_pretty_preserves_small_spacing_exactly(self):
        # When there is already whitespace separating element siblings in a block-ish
        # container, prefer multiline formatting over preserving exact whitespace.
        div = Node("div")
        div.append_child(Node("span"))
        div.append_child(Text("  "))
        div.append_child(Node("span"))
        output = div.to_html(pretty=True)
        expected = textwrap.dedent(
            """\
            <div>
              <span></span>
              <span></span>
            </div>
            """
        ).strip("\n")
        assert output == expected

    def test_compact_pretty_skips_none_children(self):
        div = Node("div")
        # Manually inject a None child to exercise defensive branch.
        div.children = [Node("span"), None, Node("em")]
        output = div.to_html(pretty=True)
        assert output == "<div>\n  <span></span>\n  <em></em>\n</div>"

    def test_compact_pretty_drops_empty_text_children(self):
        div = Node("div")
        div.append_child(Node("span"))
        div.append_child(Text(""))
        div.append_child(Node("span"))
        output = div.to_html(pretty=True)
        expected = textwrap.dedent(
            """\
            <div>
              <span></span>
              <span></span>
            </div>
            """
        ).strip("\n")
        assert output == expected

    def test_compact_pretty_drops_leading_and_trailing_whitespace(self):
        div = Node("div")
        div.append_child(Text(" \n"))
        div.append_child(Node("a"))
        div.append_child(Text(" \t\n"))
        output = div.to_html(pretty=True)
        assert output == "<div>\n  <a></a>\n</div>"

    def test_blockish_compact_multiline_not_used_with_comment_children(self):
        div = Node("div")
        div.append_child(Node("span"))
        div.append_child(Text(" "))
        div.append_child(Comment(data="x"))
        div.append_child(Text(" "))
        div.append_child(Node("span"))
        output = div.to_html(pretty=True)
        assert output == "<div><span></span> <!--x--> <span></span></div>"

    def test_blockish_compact_multiline_not_used_with_non_whitespace_text(self):
        div = Node("div")
        div.append_child(Node("span"))
        div.append_child(Text("hi"))
        div.append_child(Text(" "))
        div.append_child(Node("span"))
        output = div.to_html(pretty=True)
        assert output == "<div><span></span>hi <span></span></div>"

    def test_blockish_compact_multiline_allows_trailing_text(self):
        div = Node("div")
        div.append_child(Node("div"))
        div.append_child(Text(" "))
        div.append_child(Node("div"))
        div.append_child(Text(" Collapse\n"))

        output = div.to_html(pretty=True)
        expected = textwrap.dedent(
            """\
            <div>
              <div></div>
              <div></div>
              Collapse
            </div>
            """
        ).strip("\n")
        assert output == expected

    def test_blockish_compact_multiline_skips_when_inner_lines_are_empty(self):
        # Exercise the multiline eligibility path where it matches, but each
        # child renders to an empty string (so we fall back to compact mode).
        div = Node("div")
        frag1 = DocumentFragment()
        frag1.append_child(Text("  "))
        frag2 = DocumentFragment()
        frag2.append_child(Text("\n"))
        div.append_child(frag1)
        div.append_child(Text(" "))
        div.append_child(frag2)

        output = div.to_html(pretty=True)
        assert output == "<div> </div>"

    def test_blockish_compact_multiline_skips_none_children(self):
        div = Node("div")
        div.children = [Node("span"), Text(" "), None, Node("span")]
        output = div.to_html(pretty=True)
        expected = textwrap.dedent(
            """\
            <div>
              <span></span>
              <span></span>
            </div>
            """
        ).strip("\n")
        assert output == expected

    def test_blockish_compact_multiline_skips_none_children_with_trailing_text(self):
        # Exercise the multiline-eligibility path that allows trailing non-whitespace text,
        # while also skipping None children in both scanning and rendering loops.
        div = Node("div")
        div.children = [
            Node("span"),
            Text(" "),
            None,
            Node("span"),
            Text(" trailing\n"),
        ]
        output = div.to_html(pretty=True)
        expected = textwrap.dedent(
            """\
            <div>
              <span></span>
              <span></span>
              trailing
            </div>
            """
        ).strip("\n")
        assert output == expected

    def test_compact_mode_collapses_newline_whitespace_for_inline_container(self):
        # For non-block containers, we stay in compact mode and collapse
        # formatting whitespace to a single space.
        outer = Node("span")
        outer.append_child(Node("span"))
        outer.append_child(Text("\n  \t"))
        outer.append_child(Node("span"))
        output = outer.to_html(pretty=True)
        assert output == "<span><span></span> <span></span></span>"

    def test_compact_mode_drops_edge_whitespace_with_none_children(self):
        # Ensure the compact-mode whitespace edge trimming is robust with None children.
        div = Node("div")
        div.children = [
            Text(" \n"),
            Node("a"),
            Comment(data="x"),
            Text(" \t\n"),
            None,
        ]
        output = div.to_html(pretty=True)
        assert output == "<div><a></a><!--x--></div>"

    def test_pretty_indentation_skips_whitespace_text_nodes(self):
        # Hit the indentation-mode branch that skips whitespace-only text nodes.
        outer = Node("span")
        outer.children = [Node("div"), Text(" \n\t"), Node("div")]
        output = outer.to_html(pretty=True)
        assert output == "<span>\n  <div></div>\n  <div></div>\n</span>"

    def test_indents_single_anchor_child_when_anchor_wraps_block_elements(self):
        container = Node("div")
        link = Node("a")
        link.append_child(Node("div"))
        link.append_child(Node("div"))
        container.append_child(link)

        output = container.to_html(pretty=True)
        expected = textwrap.dedent(
            """\
            <div>
              <a>
                <div></div>
                <div></div>
              </a>
            </div>
            """
        ).strip("\n")
        assert output == expected

    def test_single_anchor_child_not_blockish_without_special_grandchildren(self):
        link = Node("a")
        link.children = [None, Node("span")]
        assert _should_pretty_indent_children([link]) is False

    def test_single_anchor_child_not_blockish_without_grandchildren(self):
        link = Node("a")
        link.children = []
        assert _should_pretty_indent_children([link]) is False

    def test_pretty_indent_children_does_not_indent_comments(self):
        div = Node("div")
        div.append_child(Comment(data="x"))
        div.append_child(Node("p"))
        output = div.to_html(pretty=True)
        assert output == "<div><!--x--><p></p></div>"

    def test_whitespace_in_fragment(self):
        frag = DocumentFragment()
        # Node constructor: name, attrs=None, data=None, namespace=None
        text_node = Text("   ")
        frag.append_child(text_node)
        output = to_html(frag)
        assert output == ""

    def test_text_node_pretty_strips_and_renders(self):
        frag = DocumentFragment()
        frag.append_child(Text("  hi  "))
        output = to_html(frag, pretty=True)
        assert output == "hi"

    def test_empty_text_node_is_dropped_when_not_pretty(self):
        div = Node("div")
        div.append_child(Text(""))
        output = to_html(div, pretty=False)
        assert output == "<div></div>"

    def test_element_with_nested_children(self):
        # Test serialize.py line 82->86: all_text branch when NOT all text
        html = "<div><span>inner</span></div>"
        doc = JustHTML(html)
        output = doc.to_html()
        assert "<div>" in output
        assert "<span>inner</span>" in output
        assert "</div>" in output

    def test_element_without_attributes(self):
        # Test serialize.py line 82->86: attr_parts is empty (no attributes)
        node = Node("div")
        text_node = Text("hello")
        node.append_child(text_node)
        output = to_html(node)
        assert output == "<div>hello</div>"

    def test_to_test_format_single_element(self):
        # Test to_test_format on non-document node (line 102)
        node = Node("div")
        output = to_test_format(node)
        assert output == "| <div>"

    def test_to_test_format_template_with_attributes(self):
        # Test template with attributes (line 126)
        template = Template("template", namespace="html")
        template.attrs = {"id": "t1"}
        child = Node("p")
        template.template_content.append_child(child)
        output = to_test_format(template)
        assert "| <template>" in output
        assert '|   id="t1"' in output
        assert "|   content" in output
        assert "|     <p>" in output

    def test_escape_js_string_invalid_quote(self):
        with self.assertRaises(ValueError):
            _JustHTML.escape_js_string("test", quote="x")

    def test_escape_attr_value_invalid_quote(self):
        with self.assertRaises(ValueError):
            _JustHTML.escape_attr_value("test", quote="x")

    def test_to_html_html_attr_value_context_invalid_quote(self):
        doc = JustHTML("<b>Hi</b>", fragment=True)
        with self.assertRaises(ValueError):
            doc.to_html(pretty=False, context=HTMLContext.HTML_ATTR_VALUE, quote="x")

    def test_clean_url_value_not_urlrule(self):
        with self.assertRaises(TypeError):
            _JustHTML.clean_url_value(value="https://example.com/", url_rule="not a rule")

    def test_clean_url_value_proxy_without_config(self):
        with self.assertRaises(ValueError):
            rule = UrlRule(allowed_schemes={"https"}, handling="proxy")
            _JustHTML.clean_url_value(value="https://example.com/", url_rule=rule)

    def test_escape_js_string_special_chars(self):
        # Test all special escape sequences
        result = _JustHTML.escape_js_string("\\back\nline\rret\ttab\bbell\fform\u2028ls\u2029ps", quote='"')
        assert result == "\\\\back\\nline\\rret\\ttab\\bbell\\fform\\u2028ls\\u2029ps"

        # Test empty string
        assert _JustHTML.escape_js_string("", quote='"') == ""

    def test_escape_html_text_in_js_string_empty(self):
        assert _JustHTML.escape_html_text_in_js_string("", quote='"') == ""

    def test_escape_url_value_empty(self):
        assert _JustHTML.escape_url_value("") == ""


if __name__ == "__main__":
    unittest.main()
