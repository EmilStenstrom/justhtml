from __future__ import annotations

import unittest
from dataclasses import replace

from justhtml import DEFAULT_POLICY, JustHTML, SetAttrs
from justhtml.context import FragmentContext
from justhtml.node import DocumentFragment, Element, Template
from justhtml.parser import JustHTML as ParserJustHTML
from justhtml.sanitize import SanitizationPolicy, UrlPolicy, UrlRule
from justhtml.transforms import Linkify, Sanitize


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

    def test_constructor_time_default_sanitization_strips_invisible_unicode(self) -> None:
        invisible = "\u200b\u202e\ue000"
        safe_doc = JustHTML(
            f'<p><a href="java{invisible}script:alert(1)">x{invisible}y</a></p>',
            fragment=True,
        )
        assert safe_doc.to_html(pretty=False) == "<p><a>xy</a></p>"

    def test_explicit_sanitize_collect_policy_does_not_leak_stale_errors(self) -> None:
        policy = replace(DEFAULT_POLICY, unsafe_handling="collect")

        unsafe_doc = JustHTML(
            "<script>x</script>",
            fragment=True,
            transforms=[Sanitize(policy=policy)],
            collect_errors=True,
        )
        assert unsafe_doc.to_html(pretty=False) == ""
        assert [e.message for e in unsafe_doc.errors if e.category == "security"] == [
            "Unsafe tag 'script' (dropped content)"
        ]

        clean_doc = JustHTML(
            "<p>ok</p>",
            fragment=True,
            transforms=[Sanitize(policy=policy)],
            collect_errors=True,
        )
        assert clean_doc.to_html(pretty=False) == "<p>ok</p>"
        assert [e for e in clean_doc.errors if e.category == "security"] == []
        assert policy.collected_security_errors() == []

    def test_disabled_explicit_sanitize_does_not_merge_stale_collected_errors(self) -> None:
        policy = replace(DEFAULT_POLICY, unsafe_handling="collect")
        policy.handle_unsafe("stale finding")

        doc = JustHTML(
            "<p>ok</p>",
            fragment=True,
            transforms=[Sanitize(policy=policy, enabled=False)],
            collect_errors=True,
        )

        assert doc.to_html(pretty=False) == "<p>ok</p>"
        assert [e for e in doc.errors if e.category == "security"] == []

    def test_explicit_sanitize_disables_implicit_final_sanitize(self) -> None:
        doc = JustHTML(
            '<img src="/x">',
            fragment=True,
            transforms=[Sanitize(), SetAttrs("img", onerror="alert(1)")],
        )

        assert doc.to_html(pretty=False) == '<img src="/x" onerror="alert(1)">'
        assert doc.to_markdown() == '<img src="/x" onerror="alert(1)">'

    def test_explicit_sanitize_allows_later_transforms_to_reintroduce_unsafe_attrs(self) -> None:
        doc = JustHTML(
            "<p>x</p>",
            fragment=True,
            transforms=[Sanitize(), SetAttrs("p", onclick="alert(1)")],
        )

        assert doc.to_html(pretty=False) == '<p onclick="alert(1)">x</p>'

    def test_constructor_time_sanitization_stabilizes_foreign_namespace_mxss(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags={"form", "math", "mtext", "mglyph", "style", "img"},
            allowed_attributes={"*": set(), "img": {"src"}},
            url_policy=UrlPolicy(
                allow_rules={("img", "src"): UrlRule(allowed_schemes={"http", "https"}, allow_relative=True)}
            ),
            drop_foreign_namespaces=False,
            drop_content_tags=set(),
        )

        doc = JustHTML(
            "<form><math><mtext></form><form><mglyph><style></math><img src onerror=alert(1)>",
            fragment=True,
            policy=policy,
        )

        assert doc.to_html(pretty=False) == "<form><math><mtext></mtext></math></form>"

        reparsed = JustHTML(doc.to_html(pretty=False), fragment=True, sanitize=False)
        imgs = reparsed.query("img")
        assert imgs == []
        assert "onerror" not in doc.to_markdown(html_passthrough=True)

    def test_terminal_sanitize_policy_returns_none_for_empty_or_non_terminal_sanitize(self) -> None:
        policy = SanitizationPolicy(allowed_tags={"p"}, allowed_attributes={})

        assert ParserJustHTML._terminal_sanitize_policy([], default_policy=policy) is None
        assert ParserJustHTML._terminal_sanitize_policy([Linkify()], default_policy=policy) is None
        assert (
            ParserJustHTML._terminal_sanitize_policy(
                [Sanitize(enabled=False)],
                default_policy=policy,
            )
            is None
        )

    def test_has_foreign_nodes_handles_templates_and_pure_html(self) -> None:
        root = DocumentFragment()
        html_div = Element("div", {}, "html")
        root.append_child(html_div)

        assert ParserJustHTML._has_foreign_nodes(root) is False

        template = Template("template", namespace="html")
        assert template.template_content is not None
        svg = Element("svg", {}, "svg")
        template.template_content.append_child(svg)
        root.append_child(template)

        assert ParserJustHTML._has_foreign_nodes(root) is True
