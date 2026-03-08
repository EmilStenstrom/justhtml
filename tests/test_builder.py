import unittest

from justhtml import JustHTML as _JustHTML
from justhtml.builder import (
    _normalize_attrs,
    _parse_element_name,
    comment,
    doctype,
    element,
    text,
)
from justhtml.node import Comment, Document, DocumentFragment, Element, Template, Text


def JustHTML(*args, **kwargs):  # noqa: N802
    if "sanitize" not in kwargs and "safe" not in kwargs:
        kwargs["sanitize"] = False
    return _JustHTML(*args, **kwargs)


class TestBuilder(unittest.TestCase):
    def test_text_factory_returns_text_node(self):
        node = text("Hello")
        self.assertIsInstance(node, Text)
        self.assertEqual(node.data, "Hello")

    def test_text_factory_rejects_non_string(self):
        with self.assertRaises(TypeError):
            text(1)

    def test_comment_factory_returns_comment_node(self):
        node = comment("Hello")
        self.assertIsInstance(node, Comment)
        self.assertEqual(node.data, "Hello")

    def test_comment_factory_rejects_non_string(self):
        with self.assertRaises(TypeError):
            comment(1)

    def test_doctype_factory_returns_doctype_node(self):
        node = doctype()
        self.assertEqual(node.name, "!doctype")
        self.assertIsNotNone(node.data)
        self.assertEqual(node.data.name, "html")

    def test_doctype_factory_rejects_invalid_argument_types(self):
        with self.assertRaises(TypeError):
            doctype(1)
        with self.assertRaises(TypeError):
            doctype("html", public_id=1)
        with self.assertRaises(TypeError):
            doctype("html", system_id=1)

    def test_doctype_factory_accepts_public_and_system_identifiers(self):
        node = doctype(
            "svg", public_id="-//W3C//DTD SVG 1.1//EN", system_id="https://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd"
        )
        self.assertEqual(node.data.name, "svg")
        self.assertEqual(node.data.public_id, "-//W3C//DTD SVG 1.1//EN")
        self.assertEqual(node.data.system_id, "https://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd")

    def test_element_factory_returns_element(self):
        node = element("div", {"id": "x"}, "Hello")
        self.assertIsInstance(node, Element)
        self.assertEqual(node.name, "div")
        self.assertEqual(node.attrs, {"id": "x"})
        self.assertEqual(len(node.children), 1)
        self.assertEqual(node.children[0].data, "Hello")

    def test_element_factory_treats_second_positional_non_mapping_as_child(self):
        node = element("div", "Hello", " world")
        self.assertEqual([child.data for child in node.children], ["Hello", " world"])

    def test_element_factory_uses_template_node_for_html_template(self):
        node = element("template", None, "Hello")
        self.assertIsInstance(node, Template)
        self.assertIsNotNone(node.template_content)
        self.assertEqual(len(node.children), 0)
        self.assertEqual(len(node.template_content.children), 1)
        self.assertEqual(node.template_content.children[0].data, "Hello")

    def test_element_factory_uses_plain_element_for_foreign_template(self):
        node = element("template", None, "Hello", namespace="svg")
        self.assertIsInstance(node, Element)
        self.assertNotIsInstance(node, Template)
        self.assertEqual(len(node.children), 1)

    def test_element_factory_parses_attribute_shorthand(self):
        node = element("input[type=email][required]")
        self.assertEqual(node.attrs, {"type": "email", "required": None})

    def test_element_factory_attr_dict_overlaps_shorthand_is_error(self):
        with self.assertRaises(ValueError):
            element('a[href="/wrong"]', {"href": "/right"})

    def test_element_factory_duplicate_shorthand_attribute_is_error(self):
        with self.assertRaises(ValueError):
            element('input[required][required="required"]')

    def test_element_factory_rejects_special_names(self):
        with self.assertRaises(ValueError):
            element("#text")

    def test_element_factory_rejects_non_string_name(self):
        with self.assertRaises(TypeError):
            element(1)

    def test_element_factory_flattens_iterables_and_ignores_none_and_false(self):
        node = element("div", None, ["a", None, False, ("b", ["c"])])
        self.assertEqual([child.data for child in node.children], ["a", "b", "c"])

    def test_element_factory_rejects_numeric_child_values(self):
        with self.assertRaises(TypeError):
            element("div", None, 1)

    def test_element_factory_stringifies_attribute_values(self):
        node = element("input", {"maxlength": 10, "checked": None})
        self.assertEqual(node.attrs, {"maxlength": "10", "checked": None})

    def test_element_factory_rejects_invalid_attribute_names(self):
        with self.assertRaises(TypeError):
            element("div", {1: "x"})
        with self.assertRaises(ValueError):
            element("div", {"": "x"})

    def test_element_factory_rejects_non_string_namespace(self):
        with self.assertRaises(TypeError):
            element("div", namespace=object())

    def test_element_factory_none_namespace_defaults_to_html(self):
        node = element("div", namespace=None)
        self.assertEqual(node.namespace, "html")

    def test_element_factory_accepts_internal_math_namespace_name(self):
        node = element("math", namespace="math")
        self.assertEqual(node.namespace, "math")

    def test_element_factory_accepts_mathml_namespace_alias(self):
        node = element("math", element("mi", "x", namespace="mathml"), namespace="mathml")
        self.assertEqual(node.namespace, "math")
        self.assertEqual(node.children[0].namespace, "math")

    def test_element_factory_rejects_non_html5_namespaces(self):
        with self.assertRaises(ValueError):
            element("box", "Hello", namespace="custom")

    def test_element_factory_rejects_invalid_child_types(self):
        with self.assertRaises(TypeError):
            element("div", None, True)
        with self.assertRaises(TypeError):
            element("div", None, {"child": "value"})
        with self.assertRaises(TypeError):
            element("div", None, b"bytes")
        with self.assertRaises(TypeError):
            element("div", None, object())

    def test_element_factory_accepts_explicit_text_node_child(self):
        node = element("div", text("Hello"))
        self.assertIsInstance(node.children[0], Text)
        self.assertEqual(node.children[0].data, "Hello")

    def test_element_factory_parses_quoted_single_value_shorthand(self):
        node = element("a[href='/docs']")
        self.assertEqual(node.attrs, {"href": "/docs"})

    def test_element_factory_parses_boolean_shorthand_attribute(self):
        node = element("option[selected]")
        self.assertEqual(node.attrs, {"selected": None})

    def test_element_factory_rejects_invalid_name_and_shorthand_forms(self):
        invalid_names = [
            "",
            "two words",
            "[attr]",
            "div[",
            "div[]",
            "div[attr='unterminated]",
            'div[attr="unterminated]',
            'div[attr="value"x]',
            r'div[attr="a\"b"]',
            "div[attr=[nested]]",
            "div[attr][attr]",
        ]

        for name in invalid_names:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    element(name)

    def test_normalize_attrs_rejects_non_mapping(self):
        with self.assertRaises(TypeError):
            _normalize_attrs([("id", "x")])

    def test_parse_element_name_rejects_missing_value_in_shorthand(self):
        with self.assertRaises(ValueError):
            _parse_element_name("div[attr=")

    def test_parse_element_name_rejects_invalid_attribute_shorthand_prefix(self):
        with self.assertRaises(ValueError):
            _parse_element_name("div[attr]oops")

    def test_parse_element_name_rejects_nested_unquoted_value(self):
        with self.assertRaises(ValueError):
            _parse_element_name("div[attr=a[b]]")

    def test_parse_element_name_rejects_unclosed_unquoted_value(self):
        with self.assertRaises(ValueError):
            _parse_element_name("div[attr=value")

    def test_justhtml_accepts_built_element_in_fragment_mode(self):
        doc = JustHTML(element("p", "Hello"), fragment=True)
        self.assertEqual(doc.to_html(pretty=False), "<p>Hello</p>")

    def test_justhtml_accepts_built_document_fragment(self):
        root = DocumentFragment()
        root.append_child(element("p", "Hello"))

        doc = JustHTML(root, fragment=True)
        self.assertEqual(doc.to_html(pretty=False), "<p>Hello</p>")

    def test_justhtml_accepts_built_document(self):
        document = Document()
        document.append_child(doctype())
        document.append_child(
            element(
                "html",
                element("head", element("title", "Hi")),
                element("body", element("p", "Hello")),
            )
        )

        doc = JustHTML(document)
        self.assertEqual(
            doc.to_html(pretty=False),
            "<!DOCTYPE html><html><head><title>Hi</title></head><body><p>Hello</p></body></html>",
        )

    def test_justhtml_accepts_built_doctype_node(self):
        doc = JustHTML(doctype())
        self.assertEqual(doc.to_html(pretty=False), "<!DOCTYPE html><html><head></head><body></body></html>")

    def test_justhtml_preserves_custom_doctype_name(self):
        doc = JustHTML(doctype("svg"), sanitize=False)
        self.assertEqual(doc.to_html(pretty=False), "<!DOCTYPE svg><html><head></head><body></body></html>")

    def test_justhtml_preserves_custom_doctype_identifiers(self):
        doc = JustHTML(
            doctype(
                "html",
                public_id="-//W3C//DTD HTML 4.01//EN",
                system_id="http://www.w3.org/TR/html4/strict.dtd",
            ),
            sanitize=False,
        )
        self.assertEqual(
            doc.to_html(pretty=False),
            '<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd"><html><head></head><body></body></html>',
        )

    def test_svg_namespace_survives_when_html5_can_reconstruct_it(self):
        doc = JustHTML(
            element("svg", element("title", "Hello", namespace="svg"), namespace="svg"),
            fragment=True,
            sanitize=False,
        )
        self.assertEqual(doc.root.children[0].namespace, "svg")
        self.assertEqual(doc.root.children[0].children[0].namespace, "svg")

    def test_mathml_namespace_alias_survives_when_html5_can_reconstruct_it(self):
        doc = JustHTML(
            element("math", element("mi", "x", namespace="mathml"), namespace="mathml"),
            fragment=True,
            sanitize=False,
        )
        self.assertEqual(doc.root.children[0].namespace, "math")
        self.assertEqual(doc.root.children[0].children[0].namespace, "math")
