"""Regression tests for the unified parser."""

import unittest
from dataclasses import replace

from justhtml import JustHTML
from justhtml.sanitizer import DEFAULT_POLICY
from justhtml.transforms import Escape


class TestMalformedRawTagRegressions(unittest.TestCase):
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


class TestEngineBehaviorRegressions(unittest.TestCase):
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
