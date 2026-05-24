import unittest

from justhtml.core.constants import FORMAT_MARKER
from justhtml.tokenizer.tokens import Tag
from justhtml.treebuilder import TreeBuilder
from justhtml.treebuilder.utils import InsertionMode


class TestPrecommitCoverageHotspots(unittest.TestCase):
    def test_flush_pending_table_text_whitespace_branch(self):
        builder = TreeBuilder(collect_errors=False)
        builder.open_elements.append(builder._create_element("div", None, {}))
        builder.pending_table_text.append("   ")

        builder._flush_pending_table_text()

        div = builder.open_elements[-1]
        assert div.children
        assert getattr(div.children[-1], "data", None) == "   "

    def test_in_head_end_template_without_template_on_stack(self):
        builder = TreeBuilder(collect_errors=True)
        html = builder._create_element("html", None, {})
        head = builder._create_element("head", None, {})
        builder.open_elements = [html, head]
        builder.head_element = head

        token = Tag(Tag.END, "template", {}, False)
        builder._mode_in_head(token)

    def test_after_head_end_template_with_template_on_stack(self):
        builder = TreeBuilder(collect_errors=True)
        html = builder._create_element("html", None, {})
        head = builder._create_element("head", None, {})
        template = builder._create_element("template", None, {})
        builder.open_elements = [html, head, template]
        builder.head_element = head
        builder.template_modes = [InsertionMode.IN_TEMPLATE]
        builder.mode = InsertionMode.AFTER_HEAD

        token = Tag(Tag.END, "template", {}, False)
        builder._mode_after_head(token)

    def test_in_table_text_breaks_on_format_marker(self):
        builder = TreeBuilder(collect_errors=True)
        builder.open_elements.append(builder._create_element("div", None, {}))
        builder.mode = InsertionMode.IN_TABLE_TEXT
        builder.table_text_original_mode = InsertionMode.IN_TABLE
        builder.active_formatting = [FORMAT_MARKER]
        builder.pending_table_text.append("x")

        token = Tag(Tag.END, "table", {}, False)
        builder._mode_in_table_text(token)

    def test_in_select_end_a_with_formatting_entry_not_on_stack(self):
        builder = TreeBuilder(collect_errors=True)
        html = builder._create_element("html", None, {})
        body = builder._create_element("body", None, {})
        select = builder._create_element("select", None, {})
        builder.open_elements = [html, body, select]
        builder.mode = InsertionMode.IN_SELECT

        a_node = builder._create_element("a", None, {})
        builder.active_formatting = [FORMAT_MARKER, {"name": "a", "node": a_node}]

        token = Tag(Tag.END, "a", {}, False)
        builder._mode_in_select(token)

    def test_in_frameset_end_frameset_when_only_html_on_stack(self):
        builder = TreeBuilder(collect_errors=True)
        html = builder._create_element("html", None, {})
        builder.open_elements = [html]
        builder.mode = InsertionMode.IN_FRAMESET

        token = Tag(Tag.END, "frameset", {}, False)
        builder._mode_in_frameset(token)
