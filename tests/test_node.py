import unittest

from justhtml import JustHTML
from justhtml.node import (
    Comment,
    Document,
    Element,
    Node,
    Template,
    Text,
    _markdown_code_span,
    _markdown_link_destination,
    _MarkdownBuilder,
    _to_markdown_walk,
)
from justhtml.sanitize import DEFAULT_POLICY, SanitizationPolicy


class TestNode(unittest.TestCase):
    def test_simple_dom_text_node_text_property(self):
        node = Text("Hi")
        assert node.text == "Hi"

    def test_node_text_property_for_text_name(self):
        node = Node("#text", data="Hi")
        assert node.text == "Hi"

    def test_node_text_property_for_text_name_none(self):
        node = Node("#text", data=None)
        assert node.text == ""

    def test_append_child_noop_for_comment_node(self):
        parent = Comment(data="comment")
        child = Node("span")
        parent.append_child(child)
        assert child.parent is None

    def test_remove_child_noop_for_comment_node(self):
        parent = Comment(data="comment")
        child = Node("span")
        parent.remove_child(child)
        assert child.parent is None

    def test_text_property_simple(self):
        node = Node("div")
        text = Text("Hello")
        node.append_child(text)
        assert node.text == ""
        assert text.text == "Hello"
        assert node.to_text() == "Hello"

    def test_text_property_nested(self):
        root = Node("div")
        span = Node("span")
        text1 = Text("Hello ")
        text2 = Text("World")

        root.append_child(text1)
        root.append_child(span)
        span.append_child(text2)

        assert root.text == ""
        assert span.text == ""
        assert root.to_text() == "Hello World"
        assert span.to_text() == "World"

    def test_text_property_empty(self):
        node = Node("div")
        assert node.text == ""

    def test_text_property_comment(self):
        node = Comment(data="comment")
        assert node.text == ""

    def test_to_text_matches_textcontent(self):
        root = Node("div")
        span = Node("span")
        root.append_child(Text("Hello "))
        root.append_child(span)
        span.append_child(Text("World"))

        assert root.to_text() == "Hello World"
        assert span.to_text() == "World"
        assert root.to_text(separator="", strip=False) == "Hello World"
        assert root.to_text(separator="", strip=True) == "HelloWorld"

    def test_to_text_skips_empty_and_whitespace_segments_by_default(self):
        root = Node("div")
        root.append_child(Text(""))
        root.append_child(Text("   "))
        root.append_child(Text("A"))
        assert root.to_text() == "A"

    def test_to_text_empty_subtree(self):
        root = Node("div")
        assert root.to_text() == ""

    def test_textnode_to_text_strip_false(self):
        t = Text("  A  ")
        assert t.to_text(strip=False) == "  A  "
        assert t.to_text(strip=True) == "A"

    def test_textnode_to_text_none_data(self):
        t = Text(None)
        assert t.to_text() == ""

    def test_to_text_includes_template_content(self):
        template = Template("template", namespace="html")
        template.template_content.append_child(Text("Inside"))

        # `.text` only sees direct children, while `to_text()` includes template content.
        assert template.text == ""
        assert template.to_text() == "Inside"

    def test_to_text_simple_dom_text_node_branch(self):
        node = Text("Hi")
        assert node.to_text() == "Hi"

    def test_justhtml_to_text(self):
        doc = JustHTML("<p>Hello</p><p>World</p>")
        assert doc.to_text() == "Hello World"
        assert doc.to_text(separator="", strip=True) == "HelloWorld"

    def test_to_text_sanitizes_by_default(self):
        doc = JustHTML("<p>ok</p><script>alert(1)</script>")
        assert doc.to_text() == "ok"

    def test_to_text_safe_false_includes_script_text(self):
        doc = JustHTML("<p>ok</p><script>alert(1)</script>", sanitize=False)
        assert doc.to_text() == "ok alert(1)"

    def test_noscript_scripting_enabled_parses_as_rawtext(self):
        doc = JustHTML(
            "<noscript><b>Hi</b></noscript>",
            fragment=True,
            sanitize=False,
            scripting_enabled=True,
        )
        root = doc.root
        assert root.children is not None
        assert len(root.children) == 1
        noscript = root.children[0]
        assert noscript.name == "noscript"
        assert len(noscript.children) == 1
        child = noscript.children[0]
        assert child.name == "#text"
        assert child.data == "<b>Hi</b>"

    def test_noscript_scripting_disabled_parses_html(self):
        doc = JustHTML(
            "<noscript><b>Hi</b></noscript>",
            fragment=True,
            sanitize=False,
            scripting_enabled=False,
        )
        root = doc.root
        assert root.children is not None
        assert len(root.children) == 1
        noscript = root.children[0]
        assert noscript.name == "noscript"
        assert len(noscript.children) == 1
        child = noscript.children[0]
        assert child.name == "b"

    def test_head_noscript_scripting_enabled_parses_as_rawtext(self):
        doc = JustHTML(
            "<head><noscript><b>Hi</b></noscript></head>",
            sanitize=False,
            scripting_enabled=True,
        )
        head = doc.query("head")[0]
        noscript = next(c for c in head.children if c.name == "noscript")
        assert len(noscript.children) == 1
        child = noscript.children[0]
        assert child.name == "#text"
        assert child.data == "<b>Hi</b>"

    def test_head_noscript_scripting_disabled_parses_html(self):
        doc = JustHTML(
            "<head><noscript><b>Hi</b></noscript></head>",
            sanitize=False,
            scripting_enabled=False,
        )
        head = doc.query("head")[0]
        noscript = next(c for c in head.children if c.name == "noscript")
        assert noscript.children == []
        assert doc.query("b")

    def test_to_text_policy_override_can_preserve_script_text(self):
        # With a custom policy that *doesn't* treat <script> as a drop-content tag,
        # the sanitizer will strip the <script> element but keep its children.
        policy = SanitizationPolicy(
            allowed_tags=DEFAULT_POLICY.allowed_tags,
            allowed_attributes=DEFAULT_POLICY.allowed_attributes,
            url_policy=DEFAULT_POLICY.url_policy,
            drop_content_tags=set(),
        )
        doc = JustHTML("<p>ok</p><script>alert(1)</script>", policy=policy)
        assert doc.to_text() == "ok alert(1)"

    def test_node_origin_offset_and_location_helpers(self):
        doc = JustHTML("<p>hi</p>", track_node_locations=True)
        p = doc.query("p")[0]
        assert p.origin_offset == 0
        assert p.origin_location == (1, 1)
        assert p.origin_line == 1
        assert p.origin_col == 1

        text = p.children[0]
        assert text.name == "#text"
        assert text.origin_offset == 3
        assert text.origin_location == (1, 4)
        assert text.origin_line == 1
        assert text.origin_col == 4

    def test_node_origin_location_is_none_by_default(self):
        doc = JustHTML("<p>hi</p>")
        p = doc.query("p")[0]
        assert p.origin_offset is None
        assert p.origin_location is None

        text = p.children[0]
        assert text.name == "#text"
        assert text.origin_location is None

    def test_textnode_origin_location_is_none_if_unset(self):
        node = Text("x")
        assert node.origin_location is None

    def test_node_origin_location_for_comment(self):
        doc = JustHTML("<!--x--><p>y</p>", track_node_locations=True, sanitize=False)
        assert doc.root.children is not None
        comment = doc.root.children[0]
        assert comment.name == "#comment"
        assert comment.origin_offset == 0
        assert comment.origin_location == (1, 1)

    def test_node_origin_location_for_comment_inside_element(self):
        doc = JustHTML("<p><!--x--></p>", track_node_locations=True, sanitize=False)
        p = doc.query("p")[0]
        comment = p.children[0]
        assert comment.name == "#comment"
        assert comment.origin_offset is not None
        assert comment.origin_location == (1, comment.origin_offset + 1)

    def test_pre_ignores_single_leading_lf(self):
        # Start tag <pre> sets ignore_lf, and the very next leading LF is dropped.
        doc = JustHTML("<pre>\n</pre>")
        pre = doc.query("pre")[0]
        assert pre.to_text(strip=False) == ""

    def test_pre_ignores_only_first_lf(self):
        doc = JustHTML("<pre>\nX</pre>")
        pre = doc.query("pre")[0]
        assert pre.to_text(strip=False) == "X"

    def test_pre_does_not_ignore_non_lf(self):
        # ignore_lf only drops an initial LF, not other characters.
        doc = JustHTML("<pre>X</pre>")
        pre = doc.query("pre")[0]
        assert pre.to_text(strip=False) == "X"

    def test_adoption_agency_preserves_origin_for_replacement_nodes(self):
        # Mis-nested formatting triggers the adoption agency algorithm which replaces
        # formatting elements. With tracking enabled, replacement nodes should keep
        # origin_offset/origin_location.
        html = "<b><i><p>1</b>2</i>"
        doc = JustHTML(html, track_node_locations=True)
        bolds = doc.query("b")
        italics = doc.query("i")
        assert bolds
        assert italics

        for node in bolds + italics:
            assert node.origin_offset is not None
            assert node.origin_location == (1, node.origin_offset + 1)

    def test_text_in_table_tracks_origin_in_foster_parenting_path(self):
        doc = JustHTML("<table>hi</table>", track_node_locations=True)

        def walk(n):
            yield n
            children = getattr(n, "children", None)
            if children:
                for c in children:
                    yield from walk(c)

        texts = [n for n in walk(doc.root) if getattr(n, "name", None) == "#text" and getattr(n, "data", None) == "hi"]
        assert texts
        assert texts[0].origin_offset is not None
        assert texts[0].origin_location == (1, texts[0].origin_offset + 1)

    def test_reconstruct_active_formatting_preserves_origin(self):
        # This triggers active formatting reconstruction where the new formatting node
        # has no token start_pos and must copy its origin from the formatting entry.
        html = "<p><b>1</p>2"
        doc = JustHTML(html, track_node_locations=True)
        bolds = doc.query("b")
        assert len(bolds) >= 2
        assert bolds[0].origin_offset is not None
        assert bolds[1].origin_offset == bolds[0].origin_offset
        assert bolds[1].origin_location == bolds[0].origin_location

    def test_to_markdown_headings_paragraphs_and_inline(self):
        doc = JustHTML("<h1>Title</h1><p>Hello <b>world</b> <em>ok</em> <a href='https://e.com'>link</a> a*b</p>")

        md = doc.to_markdown()
        assert md.startswith("# Title\n\n")
        assert "Hello **world** *ok* [link](https://e.com) a\\*b" in md

    def test_to_markdown_code_inline_and_block(self):
        doc = JustHTML("<pre>code`here\n</pre><p>inline <code>a`b</code></p>")
        md = doc.to_markdown()
        assert "```\ncode`here\n```" in md
        # Inline code uses a longer fence when content contains backticks.
        assert "inline ``a`b``" in md

    def test_to_markdown_blockquote_and_br(self):
        doc = JustHTML("<blockquote><p>Q<br>R</p></blockquote>")
        assert doc.to_markdown() == "> Q\n> R"

    def test_to_markdown_lists(self):
        doc = JustHTML("<ul><li>One</li><li>Two</li></ul><ol><li>A</li><li>B</li></ol>")
        md = doc.to_markdown()
        assert "- One\n- Two" in md
        assert "1. A\n2. B" in md

    def test_to_markdown_tables_and_images_are_html(self):
        doc = JustHTML("<p>Hi<img src=x alt=y>there</p><table><tr><td>A</td></tr></table>")
        md = doc.to_markdown()
        assert '<img src="x" alt="y">' in md
        # HTML5 parsing inserts <tbody>; ensure the table subtree is preserved as HTML.
        assert "<table" in md
        assert "<td>A</td>" in md
        assert "</table>" in md

    def test_to_markdown_ignores_comment_and_doctype(self):
        root = Node("div")
        root.append_child(Comment(data="nope"))
        root.append_child(Node("!doctype", data="html"))
        root.append_child(Text("ok"))
        assert root.to_markdown() == "ok"

    def test_to_markdown_preserves_script_whitespace(self):
        # script/style are preserved as raw HTML blocks in markdown when passthrough is on.
        root = Node("div")
        script = Node("script")
        # Include a trailing newline to exercise raw-newline tracking.
        script.append_child(Text("var x = 1;\nvar y = 2;\n"))
        root.append_child(script)
        assert root.to_markdown(html_passthrough=True) == "<script>var x = 1;\nvar y = 2;\n</script>"

    def test_to_markdown_empty_script_still_outputs_tags(self):
        root = Node("div")
        root.append_child(Node("script"))
        assert root.to_markdown() == ""

    def test_to_markdown_empty_script_passthrough(self):
        root = Node("div")
        root.append_child(Node("script"))
        assert root.to_markdown(html_passthrough=True) == "<script></script>"

    def test_to_markdown_script_drops_content_by_default(self):
        root = Node("div")
        script = Node("script")
        script.append_child(Text("alert(1);"))
        root.append_child(script)
        assert root.to_markdown() == ""

    def test_to_markdown_textnode_method(self):
        t = Text("a*b")
        assert t.to_markdown() == "a\\*b"

    def test_to_markdown_empty_textnode(self):
        # Exercises empty-string handling in markdown helpers and builder.
        t = Text("")
        assert t.to_markdown() == ""

    def test_to_markdown_ignores_empty_inline_formatting(self):
        root = Node("div")
        root.append_child(Node("i"))
        root.append_child(Node("b"))
        assert root.to_markdown() == ""

    def test_to_markdown_br_on_empty_buffer_and_multiple_newlines(self):
        # Exercises newline logic when buffer is empty and when newline_count is already >= 2.
        doc = JustHTML("<br><br><br>")
        assert doc.to_markdown() == ""

    def test_to_markdown_empty_blocks_and_hr(self):
        doc = JustHTML("<hr><h2></h2><p></p><pre></pre><blockquote></blockquote>")
        md = doc.to_markdown()
        assert "---" in md
        assert "##" in md
        assert "```\n```" in md

    def test_to_markdown_list_skips_non_li_children(self):
        # Newlines between list items become text nodes; list renderer should skip them.
        doc = JustHTML("<ul>\n<li>One</li>\n</ul>")
        assert doc.to_markdown() == "- One"

    def test_to_markdown_link_without_href(self):
        doc = JustHTML("<p><a>text</a></p>")
        assert doc.to_markdown() == "[text]"

    def test_to_markdown_link_destination_wrapped_when_parentheses(self):
        doc = JustHTML("<p><a href='https://e.com/a(b)c'>x</a></p>")
        assert doc.to_markdown() == "[x](<https://e.com/a(b)c>)"

    def test_to_markdown_link_destination_wrapped_when_whitespace(self):
        # Whitespace in href should not be able to break Markdown formatting.
        doc = JustHTML("<p><a href='https://e.com/a b'>x</a></p>")
        assert doc.to_markdown() == "[x](<https://e.com/a%20b>)"

    def test_to_markdown_in_link_br_and_paragraph_spacing(self):
        a = Node("a", attrs={"href": "https://e.com"})
        a.append_child(Text("A"))
        a.append_child(Node("br"))
        a.append_child(Text("B"))
        p = Node("p")
        p.append_child(Text("C"))
        a.append_child(p)
        a.append_child(Text("D"))
        assert a.to_markdown() == "[A BC D](https://e.com)"

    def test_to_markdown_in_link_block_elements_are_flattened(self):
        a = Node("a", attrs={"href": "https://e.com"})
        bq = Node("blockquote")
        p = Node("p")
        p.append_child(Text("Q"))
        bq.append_child(p)
        a.append_child(bq)

        ul = Node("ul")
        li1 = Node("li")
        li1.append_child(Text("One"))
        li2 = Node("li")
        li2.append_child(Text("Two"))
        ul.append_child(li1)
        ul.append_child(li2)
        a.append_child(ul)

        assert a.to_markdown() == "[Q One Two](https://e.com)"

    def test_to_markdown_in_link_table_heading_pre_and_hr(self):
        a = Node("a", attrs={"href": "https://e.com"})
        a.append_child(Node("hr"))

        h2 = Node("h2")
        h2.append_child(Text("T"))
        a.append_child(h2)

        pre = Node("pre")
        pre.append_child(Text("code"))
        a.append_child(pre)

        table = Node("table")
        tr = Node("tr")
        td = Node("td")
        td.append_child(Text("A"))
        tr.append_child(td)
        table.append_child(tr)
        a.append_child(table)

        md = a.to_markdown()
        assert md.startswith("[")
        assert md.endswith("](https://e.com)")
        assert "T" in md
        assert "`code`" in md
        assert "<table" in md
        assert "---" not in md

    def test_to_markdown_in_link_blockquote_empty_and_list_skips_non_li(self):
        a = Node("a", attrs={"href": "https://e.com"})

        # Covers blockquote in-link branch with no children.
        a.append_child(Node("blockquote"))

        # Covers list-in-link branch where a non-li child is skipped.
        ul = Node("ul")
        ul.append_child(Text("\n"))
        li = Node("li")
        li.append_child(Text("One"))
        ul.append_child(li)
        a.append_child(ul)

        assert a.to_markdown() == "[One](https://e.com)"

    def test_markdown_code_span_edge_cases(self):
        # Cover helper edge cases (None input and leading/trailing backticks).
        assert _markdown_code_span(None) == "``"
        assert _markdown_code_span("`x") == "`` `x ``"
        assert _markdown_code_span("x`") == "`` x` ``"
        # Exercise backtick runs that don't increase the longest run.
        assert _markdown_code_span("a`b`") == "`` a`b` ``"

    def test_markdown_link_destination_helper_edge_cases(self):
        assert _markdown_link_destination("") == ""
        assert _markdown_link_destination("   ") == ""
        assert _markdown_link_destination("https://e.com/x") == "https://e.com/x"

    def test_to_markdown_pre_rstrips_trailing_spaces_before_newline(self):
        doc = JustHTML("<pre>X   \n</pre>")
        assert doc.to_markdown() == "```\nX\n```"

    def test_to_markdown_document_container_direct(self):
        doc = Document()
        doc.append_child(Node("p"))
        assert doc.to_markdown() == ""

    def test_markdown_builder_text_preserve_whitespace_branch(self):
        b = _MarkdownBuilder()
        b.text("x\n", preserve_whitespace=True)
        assert b.finish() == "x"

    def test_to_markdown_walk_preserves_whitespace_for_text_nodes(self):
        b = _MarkdownBuilder()
        _to_markdown_walk(Text("a\nb"), b, preserve_whitespace=True, list_depth=0)
        assert b.finish() == "a\nb"

    def test_markdown_builder_text_leading_whitespace_does_not_add_space(self):
        # Covers the branch where pending whitespace exists but we are at start of output.
        b = _MarkdownBuilder()
        b.text("   a")
        assert b.finish() == "a"

    def test_to_markdown_raw_with_internal_newline_no_trailing_newline(self):
        # Covers raw() newline handling when the string contains a newline but doesn't end with one.
        root = Node("div")
        style = Node("style")
        style.append_child(Text("a {\n  b: c; }"))
        root.append_child(style)
        assert "a {\n  b: c; }" in root.to_markdown(html_passthrough=True)

    def test_to_markdown_unknown_container_walks_children(self):
        doc = JustHTML("<span>Hi</span>")
        assert doc.to_markdown() == "Hi"

    def test_markdown_builder_raw_inserts_pending_space(self):
        b = _MarkdownBuilder()
        b.text("a ")
        b.raw("**")
        b.raw("b")
        assert b.finish() == "a **b"

    def test_markdown_builder_raw_does_not_insert_space_before_newline(self):
        # Covers the branch where pending space exists but raw output starts with whitespace.
        b = _MarkdownBuilder()
        b.text("a ")
        b.raw("\n")
        assert b.finish() == "a"

    def test_markdown_walk_document_children_loop(self):
        b = _MarkdownBuilder()
        doc = Document()
        doc.append_child(Text("Hi"))
        _to_markdown_walk(doc, b, preserve_whitespace=False, list_depth=0)
        assert b.finish() == "Hi"

    def test_markdown_walk_document_without_children(self):
        # Covers the document-container branch when there are no children.
        doc = Document()
        assert doc.to_markdown() == ""

    def test_to_markdown_includes_template_content(self):
        template = Template("template", namespace="html")
        template.template_content.append_child(Text("T"))
        assert template.to_markdown() == "T"

    def test_markdown_walk_unknown_tag_children_loop(self):
        b = _MarkdownBuilder()
        span = Node("span")
        span.append_child(Text("Hi"))
        _to_markdown_walk(span, b, preserve_whitespace=False, list_depth=0)
        assert b.finish() == "Hi"

    def test_insert_before(self):
        parent = Node("div")
        child1 = Node("span", attrs={"id": "1"})
        child2 = Node("span", attrs={"id": "2"})

        parent.append_child(child1)
        parent.insert_before(child2, child1)

        assert parent.children == [child2, child1]
        assert child2.parent == parent

    def test_insert_before_none(self):
        parent = Node("div")
        child1 = Node("span", attrs={"id": "1"})
        child2 = Node("span", attrs={"id": "2"})

        parent.append_child(child1)
        parent.insert_before(child2, None)

        assert parent.children == [child1, child2]
        assert child2.parent == parent

    def test_insert_before_invalid_reference(self):
        parent = Node("div")
        child1 = Node("span", attrs={"id": "1"})
        child2 = Node("span", attrs={"id": "2"})
        other = Node("div")

        parent.append_child(child1)

        with self.assertRaises(ValueError):
            parent.insert_before(child2, other)

    def test_insert_before_no_children_allowed(self):
        comment = Comment(data="foo")
        node = Node("div")

        with self.assertRaises(ValueError):
            comment.insert_before(node, None)

    def test_text_node_none(self):
        text = Text(None)
        assert text.text == ""

    def test_simple_dom_node_text_none(self):
        node = Text(None)
        assert node.text == ""

    def test_replace_child(self):
        parent = Node("div")
        child1 = Node("span", attrs={"id": "1"})
        child2 = Node("span", attrs={"id": "2"})
        new_child = Node("p")

        parent.append_child(child1)
        parent.append_child(child2)

        replaced = parent.replace_child(new_child, child1)

        assert replaced == child1
        assert parent.children == [new_child, child2]
        assert new_child.parent == parent
        assert child1.parent is None

    def test_replace_child_invalid(self):
        parent = Node("div")
        child1 = Node("span")
        other = Node("p")

        parent.append_child(child1)

        with self.assertRaises(ValueError):
            parent.replace_child(other, other)

    def test_replace_child_no_children_allowed(self):
        comment = Comment(data="foo")
        node = Node("div")

        with self.assertRaises(ValueError):
            comment.replace_child(node, node)

    def test_has_child_nodes(self):
        parent = Node("div")
        assert not parent.has_child_nodes()

        parent.append_child(Node("span"))
        assert parent.has_child_nodes()

    def test_clone_node_shallow(self):
        node = Node("div", attrs={"class": "foo"}, namespace="html")
        child = Node("span")
        node.append_child(child)

        clone = node.clone_node(deep=False)

        assert clone.name == "div"
        assert clone.attrs == {"class": "foo"}
        assert clone.namespace == "html"
        assert clone.children == []
        assert clone is not node
        assert clone.attrs is not node.attrs

    def test_clone_node_simple(self):
        node = Node("div", attrs={"id": "1"})
        clone = node.clone_node()
        assert clone.name == "div"
        assert clone.attrs == {"id": "1"}
        assert clone is not node
        assert clone.children == []

    def test_clone_node_deep(self):
        parent = Node("div")
        child = Node("span")
        parent.append_child(child)

        clone = parent.clone_node(deep=True)
        assert len(clone.children) == 1
        assert clone.children[0].name == "span"
        assert clone.children[0] is not child
        assert clone.children[0].parent == clone

    def test_clone_text_node(self):
        text = Text("hello")
        clone = text.clone_node()
        assert clone.data == "hello"
        assert clone is not text

    def test_clone_template_node(self):
        template = Template("template", namespace="html")
        content_child = Node("div")
        template.template_content.append_child(content_child)

        clone = template.clone_node(deep=True)
        assert clone is not template
        assert clone.template_content is not template.template_content
        assert len(clone.template_content.children) == 1
        assert clone.template_content.children[0].name == "div"

    def test_clone_template_node_with_children(self):
        template = Template("template", namespace="html")
        child = Node("span")
        template.append_child(child)

        clone = template.clone_node(deep=True)
        assert len(clone.children) == 1
        assert clone.children[0].name == "span"
        assert clone.children[0] is not child
        assert clone.children[0].parent == clone

    def test_clone_element_node(self):
        element = Element("div", attrs={"class": "foo"}, namespace="html")
        child = Node("span")
        element.append_child(child)

        # Shallow clone
        clone_shallow = element.clone_node(deep=False)
        assert isinstance(clone_shallow, Element)
        assert clone_shallow.children == []

        # Deep clone
        clone_deep = element.clone_node(deep=True)
        assert len(clone_deep.children) == 1
        assert clone_deep.children[0].name == "span"
        assert clone_deep.children[0] is not child
        assert clone_deep.children[0].parent == clone_deep

    def test_clone_node_empty_attrs(self):
        node = Node("div")
        clone = node.clone_node()
        assert clone.attrs == {}

    def test_clone_comment_node(self):
        node = Comment(data="foo")
        clone = node.clone_node()
        assert clone.attrs is None
        assert clone.data == "foo"

    def test_clone_template_node_non_html(self):
        template = Template("template", namespace="svg")
        assert template.template_content is None
        # Add a child to exercise the for loop even when template_content is None
        child = Node("g")
        template.append_child(child)

        clone = template.clone_node(deep=True)
        assert clone.template_content is None
        assert clone.namespace == "svg"
        assert len(clone.children) == 1
        assert clone.children[0].name == "g"

    def test_clone_template_node_shallow(self):
        template = Template("template", namespace="html")
        child = Node("div")
        template.append_child(child)

        clone = template.clone_node(deep=False)
        assert clone.name == "template"
        assert clone.namespace == "html"
        # Shallow clone should not copy children
        assert len(clone.children) == 0

    def test_clone_doctype(self):
        node = Node("!doctype", data="html")
        clone = node.clone_node()
        assert clone.name == "!doctype"
        assert clone.attrs is None

    def test_clone_document(self):
        node = Document()
        clone = node.clone_node()
        assert clone.name == "#document"
        assert clone.children == []
        assert clone.attrs == {}

    def test_clone_document_deep(self):
        node = Document()
        child = Node("div")
        node.append_child(child)
        clone = node.clone_node(deep=True)
        assert len(clone.children) == 1
        assert clone.children[0].name == "div"
        assert clone.children[0] is not child
        assert clone.children[0].parent is clone

    def test_remove_child(self):
        parent = Node("div")
        child = Node("span")
        parent.append_child(child)

        parent.remove_child(child)
        assert parent.children == []
        assert child.parent is None

    def test_remove_child_not_found(self):
        parent = Node("div")
        child = Node("span")
        with self.assertRaises(ValueError):
            parent.remove_child(child)

    def test_to_html_method(self):
        node = Node("div")
        output = node.to_html()
        assert "<div>" in output

    def test_query_method(self):
        parent = Node("div")
        child = Node("span")
        parent.append_child(child)
        results = parent.query("span")
        assert len(results) == 1
        assert results[0].name == "span"

    def test_template_node_clone_with_content(self):
        template = Template("template", namespace="html")
        inner = Node("div")
        template.template_content.append_child(inner)
        # Also add a direct child to cover line 180-181
        direct_child = Node("span")
        template.append_child(direct_child)

        clone = template.clone_node(deep=True)
        assert len(clone.template_content.children) == 1
        assert clone.template_content.children[0].name == "div"
        assert len(clone.children) == 1
        assert clone.children[0].name == "span"

    def test_text_node_children_and_has_child_nodes(self):
        text = Text("hello")
        assert text.children == []
        assert not text.has_child_nodes()
