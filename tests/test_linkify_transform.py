from __future__ import annotations

import unittest

from justhtml import JustHTML as _JustHTML
from justhtml.node import Element, Text
from justhtml.transforms import Linkify, apply_compiled_transforms, compile_transforms


class TestLinkifyTransform(unittest.TestCase):
    def _parse(self, html: str, **kwargs) -> _JustHTML:
        if "safe" not in kwargs:
            kwargs["safe"] = False
        return _JustHTML(html, **kwargs)

    def test_linkify_wraps_fuzzy_domain_in_text_node(self) -> None:
        doc = self._parse("<p>See example.com</p>", transforms=[Linkify()])
        out = doc.to_html(pretty=False)
        assert '<a href="http://example.com">example.com</a>' in out

    def test_linkify_wraps_email_in_text_node(self) -> None:
        doc = self._parse("<p>Mail me: test@example.com</p>", transforms=[Linkify()])
        out = doc.to_html(pretty=False)
        assert '<a href="mailto:test@example.com">test@example.com</a>' in out

    def test_linkify_does_not_linkify_inside_existing_anchor(self) -> None:
        doc = self._parse('<p><a href="/x">example.com</a></p>', transforms=[Linkify()])
        out = doc.to_html(pretty=False)
        assert out == '<html><head></head><body><p><a href="/x">example.com</a></p></body></html>'

    def test_linkify_skips_pre_by_default(self) -> None:
        doc = self._parse("<pre>example.com</pre><p>example.com</p>", transforms=[Linkify()])
        out = doc.to_html(pretty=False)
        assert "<pre>example.com</pre>" in out
        assert '<p><a href="http://example.com">example.com</a></p>' in out

    def test_linkify_handles_ampersands_in_query_strings(self) -> None:
        doc = self._parse("<p>http://a.co?b=1&amp;c=2</p>", transforms=[Linkify()])
        out = doc.to_html(pretty=False)
        assert 'href="http://a.co?b=1&amp;c=2"' in out
        assert ">http://a.co?b=1&amp;c=2<" in out

    def test_linkify_preserves_trailing_text_after_match(self) -> None:
        doc = self._parse("<p>See example.com now</p>", transforms=[Linkify()])
        out = doc.to_html(pretty=False)
        assert '<a href="http://example.com">example.com</a> now' in out

    def test_linkify_runs_inside_template_content(self) -> None:
        doc = self._parse("<template>example.com</template>", transforms=[Linkify()])
        out = doc.to_html(pretty=False)
        assert '<template><a href="http://example.com">example.com</a></template>' in out

    def test_apply_compiled_transforms_handles_empty_text_nodes(self) -> None:
        root = Element("div", {}, "html")
        root.append_child(Text(""))

        compiled = compile_transforms([Linkify()])
        apply_compiled_transforms(root, compiled)

        assert root.to_html(pretty=False) == "<div></div>"
