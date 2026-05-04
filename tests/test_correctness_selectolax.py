from benchmarks.correctness import _selectolax_doctype_content, _selectolax_walk


class FakeSelectolaxNode:
    def __init__(self, tag, *, attrs=None, child=None, next_node=None, html=None, pretty=None):
        self.tag = tag
        self.attributes = attrs or {}
        self.child = child
        self.next = next_node
        self.html = html
        self._pretty = pretty

    def html_pretty(self, **kwargs):
        return self._pretty


def test_selectolax_doctype_content_formats_full_system_id():
    assert _selectolax_doctype_content('<!DOCTYPE potato SYSTEM "taco">') == 'potato "" "taco"'


def test_selectolax_doctype_content_formats_full_public_and_system_ids():
    doctype = '<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">'

    assert (
        _selectolax_doctype_content(doctype)
        == 'html "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd"'
    )


def test_selectolax_walk_uses_pretty_namespace_for_foreign_elements_and_attrs():
    node = FakeSelectolaxNode(
        "math",
        attrs={"title": "tip", "show": None},
        pretty='<math:math xlink:title="tip" xlink:show="">\n</math>\n',
    )

    assert _selectolax_walk(node, 0) == [
        "| <math math>",
        '|   xlink show=""',
        '|   xlink title="tip"',
    ]


def test_selectolax_walk_keeps_html_namespace_colon_attrs_unchanged():
    node = FakeSelectolaxNode(
        "body",
        attrs={"xlink:href": "foo"},
        pretty='<body xlink:href="foo">\n</body>\n',
    )

    assert _selectolax_walk(node, 0) == [
        "| <body>",
        '|   xlink:href="foo"',
    ]
