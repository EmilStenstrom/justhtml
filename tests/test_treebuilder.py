import unittest

from justhtml import JustHTML
from justhtml.node import Element
from justhtml.tokenizer import Tokenizer, TokenizerOpts
from justhtml.treebuilder import InsertionMode, TreeBuilder


class TestTreeBuilder(unittest.TestCase):
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

    def test_process_characters_only_null_returns_continue(self) -> None:
        tree_builder = TreeBuilder(collect_errors=True)
        tree_builder.mode = InsertionMode.IN_BODY
        tree_builder.open_elements.append(Element("body", {}, None))

        tree_builder.process_characters("\x00")
        body = tree_builder.open_elements[-1]
        self.assertEqual(body.children, [])

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
        tree_builder.open_elements = [html, body]

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
        tree_builder.open_elements = [html, body, table]

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
        tree_builder.open_elements = [html, body, div]

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
