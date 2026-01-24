from __future__ import annotations

import unittest

from justhtml import JustHTML, SetAttrs
from justhtml.context import FragmentContext
from justhtml.transforms import Linkify


class TestTransformsSanitizeIntegration(unittest.TestCase):
    def test_constructor_time_sanitization_strips_unsafe_attrs(self) -> None:
        safe_doc = JustHTML(
            "<p>example.com</p>",
            fragment_context=FragmentContext("div"),
            transforms=[Linkify(), SetAttrs("a", onclick="x()")],
        )

        assert safe_doc.to_html(pretty=False) == '<p><a href="http://example.com">example.com</a></p>'

        unsafe_doc = JustHTML(
            "<p>example.com</p>",
            sanitize=False,
            fragment_context=FragmentContext("div"),
            transforms=[Linkify(), SetAttrs("a", onclick="x()")],
        )
        assert unsafe_doc.to_html(pretty=False) == '<p><a href="http://example.com" onclick="x()">example.com</a></p>'

    def test_safe_alias_still_works(self) -> None:
        doc = JustHTML("<p onclick='x()'>ok</p>", fragment=True, safe=False)
        assert doc.to_html(pretty=False) == '<p onclick="x()">ok</p>'

    def test_safe_and_sanitize_conflict_raises(self) -> None:
        with self.assertRaises(ValueError):
            JustHTML("<p>ok</p>", fragment=True, sanitize=True, safe=False)

    def test_constructor_time_sanitization_strips_disallowed_href_schemes_from_linkify(self) -> None:
        unsafe_doc = JustHTML(
            "<p>ftp://example.com</p>",
            sanitize=False,
            fragment_context=FragmentContext("div"),
            transforms=[Linkify()],
        )

        assert unsafe_doc.to_html(pretty=False) == ('<p><a href="ftp://example.com">ftp://example.com</a></p>')

        safe_doc = JustHTML(
            "<p>ftp://example.com</p>",
            fragment_context=FragmentContext("div"),
            transforms=[Linkify()],
        )
        assert safe_doc.to_html(pretty=False) == "<p><a>ftp://example.com</a></p>"

    def test_constructor_time_sanitization_resolves_protocol_relative_links(self) -> None:
        unsafe_doc = JustHTML(
            "<p>//example.com</p>",
            sanitize=False,
            fragment_context=FragmentContext("div"),
            transforms=[Linkify()],
        )

        assert unsafe_doc.to_html(pretty=False) == '<p><a href="//example.com">//example.com</a></p>'

        safe_doc = JustHTML(
            "<p>//example.com</p>",
            fragment_context=FragmentContext("div"),
            transforms=[Linkify()],
        )
        assert safe_doc.to_html(pretty=False) == '<p><a href="https://example.com">//example.com</a></p>'
