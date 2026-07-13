"""Regression tests for the unified parser."""

import unittest
from dataclasses import replace

from justhtml import JustHTML
from justhtml.sanitizer import DEFAULT_POLICY
from justhtml.transforms import Escape


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


class TestAsciiCaseFoldingRegressions(unittest.TestCase):
    """The lowered scan copy must stay index-aligned with the original input.

    "İ" (U+0130) is the only character whose str.lower() expands to two
    characters, shifting every later index of a Unicode-lowered copy. Markup
    matching is ASCII case-insensitive per §13.2.5, so only A-Z may fold.
    """

    def test_rawtext_end_tag_found_after_length_changing_case_char(self) -> None:
        cases = [
            # Before the raw-text element.
            "<div>İ</div><script>y</script>0",
            "<div>İ</div><title>y</title>0",
            "<div>İ</div><textarea>a</textarea>b",
            "<div>İİİ</div><style>.a{}</style>tail",
            # Inside the raw-text content itself.
            '<script>var x = "İ";</script>0',
        ]

        for html in cases:
            with self.subTest(html=html):
                assert JustHTML(html, fragment=True, sanitize=False).to_html(pretty=False) == html

    def test_kelvin_sign_document_scans_with_strict_ascii_fold(self) -> None:
        # U+212A KELVIN SIGN is the only non-ASCII character whose str.lower()
        # is an ASCII letter ("k"), so it must take the strict A-Z fold and
        # come through scanning untouched.
        html = "<div>\u212a</div><script>y</script>0"

        assert JustHTML(html, fragment=True, sanitize=False).to_html(pretty=False) == html

    def test_foreign_cdata_after_length_changing_case_char_stays_text(self) -> None:
        document = JustHTML("<svg>İ<![CDATA[x]]></svg>", fragment=True, sanitize=False)

        assert document.to_html(pretty=False) == "<svg>İx</svg>"


if __name__ == "__main__":
    unittest.main()
