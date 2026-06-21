"""Compatibility tests for parser diagnostics and malformed raw markup."""

import unittest
from dataclasses import replace

from justhtml import JustHTML, StrictModeError
from justhtml.parser.context import FragmentContext
from justhtml.sanitizer import DEFAULT_POLICY
from justhtml.transforms import Escape


class TestV240ErrorCompatibility(unittest.TestCase):
    def test_unclosed_div_is_strict_error_at_eof(self) -> None:
        html = "<!doctype html><div>"

        with self.assertRaises(StrictModeError) as ctx:
            JustHTML(html, strict=True)

        error = ctx.exception.error
        assert (error.code, error.line, error.column, error.category) == (
            "expected-closing-tag-but-got-eof",
            1,
            20,
            "treebuilder",
        )

    def test_rawtext_eof_uses_named_closing_tag_error(self) -> None:
        document = JustHTML("<!doctype html><title>x", collect_errors=True, sanitize=False)

        assert [(error.code, error.line, error.column) for error in document.errors] == [
            ("expected-named-closing-tag-but-got-eof", 1, 23)
        ]

    def test_malformed_doctype_spacing_matches_v240(self) -> None:
        document = JustHTML("<!DOCTYPEhtml>Hello", collect_errors=True, sanitize=False)

        assert [(error.code, error.line, error.column) for error in document.errors] == [
            ("missing-whitespace-before-doctype-name", 1, 10)
        ]

    def test_public_and_system_ids_require_separating_whitespace(self) -> None:
        html = '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN""http://www.w3.org/TR/html4/strict.dtd">'
        document = JustHTML(html, collect_errors=True, sanitize=False)

        assert [(error.code, error.line, error.column) for error in document.errors] == [
            ("missing-whitespace-between-doctype-public-and-system-identifiers", 1, 50)
        ]

    def test_diagnostic_scanner_edge_states(self) -> None:
        cases = [
            "<!doctype html></span\n>",
            "<div/'quoted'>",
            "<div 'quoted'>",
            "<div a=   >",
            "<?x>   ",
            "<?x><div>",
            "<frameset></frameset>x<noframes>",
            "<frameset></frameset></html>x<p>",
        ]
        for html in cases:
            with self.subTest(html=html):
                assert JustHTML(html, sanitize=False, collect_errors=True).root is not None

        colgroup = JustHTML(
            "x",
            fragment_context=FragmentContext("colgroup"),
            sanitize=False,
            collect_errors=True,
        )
        assert any(error.code == "unexpected-characters-in-column-group" for error in colgroup.errors)


class TestMalformedRawTagCompatibility(unittest.TestCase):
    def test_carriage_returns_delimit_tag_and_attribute_names(self) -> None:
        html = "<!doctype html><div\rclass\r=\r'a'\r>w</div>"

        assert JustHTML(html, sanitize=False).to_html(pretty=False) == (
            '<!DOCTYPE html><html><head></head><body><div class="a">w</div></body></html>'
        )

    def test_carriage_return_before_solidus_does_not_enter_element_name(self) -> None:
        html = "<!doctype html><div\r/>x"

        assert JustHTML(html, sanitize=False).to_html(pretty=False) == (
            "<!DOCTYPE html><html><head></head><body><div>x</div></body></html>"
        )

    def test_malformed_names_after_frameset_do_not_reach_serializer(self) -> None:
        expected = "<html><head></head><frameset><frame></frame></frameset></html>"
        cases = [
            "<html><frameset><frame></frameset></html><linkN\v\0href=x>",
            "<html><frameset><frame></frameset></html><a0<>",
        ]

        for html in cases:
            with self.subTest(html=html):
                assert JustHTML(html, sanitize=False).to_html(pretty=False) == expected

    def test_malformed_null_name_in_frameset_is_ignored(self) -> None:
        document = JustHTML("<frameset><div\0>x", sanitize=False)

        assert document.to_html(pretty=False) == "<html><head></head><frameset></frameset></html>"


class TestEngineBehaviorCompatibility(unittest.TestCase):
    def test_fragment_template_unwrap_preserves_text_boundaries(self) -> None:
        document = JustHTML("<div>a<template>b</template>c</div>", fragment=True)

        assert document.to_html(pretty=False) == "<div>abc</div>"
        assert document.to_text() == "a b c"

    def test_safe_fragment_unwraps_template_contents(self) -> None:
        document = JustHTML("<template>x</template>", fragment=True)

        assert document.to_html(pretty=False) == "x"

    def test_raw_div_fragment_ignores_head_wrapper(self) -> None:
        document = JustHTML("<head><title>x</title></head>", fragment=True, sanitize=False)

        assert document.to_html(pretty=False) == "<title>x</title>"

    def test_escape_transform_preserves_doctype(self) -> None:
        document = JustHTML("<!DOCTYPE html><p>x</p>", transforms=[Escape("nosuch")])

        assert document.to_html(pretty=False) == "<!DOCTYPE html><html><head></head><body><p>x</p></body></html>"

    def test_escape_policy_includes_synthetic_document_closers(self) -> None:
        policy = replace(DEFAULT_POLICY, disallowed_tag_handling="escape")
        document = JustHTML("<p>x</p>", policy=policy)

        assert document.to_html(pretty=False) == (
            "&lt;html&gt;&lt;head&gt;&lt;body&gt;<p>x</p>&lt;/body&gt;&lt;/html&gt;"
        )

    def test_escape_policy_escapes_fragment_comments(self) -> None:
        policy = replace(DEFAULT_POLICY, disallowed_tag_handling="escape")
        document = JustHTML("<!--c--><p>x</p>", fragment=True, policy=policy)

        assert document.to_html(pretty=False) == "&lt;!--c--&gt;<p>x</p>"


if __name__ == "__main__":
    unittest.main()
