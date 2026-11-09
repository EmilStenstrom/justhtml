import unittest

from justhtml import JustHTML, stream
from justhtml.core.entities import decode_entities_in_text, decode_numeric_entity
from tests.harness.scaling import assert_scales_linearly


class TestEntities(unittest.TestCase):
    def test_legacy_named_entities_before_non_ascii_characters(self):
        for suffix in ("²", "\uff19", "\U0001d7da", "é", "中", "²;", "é;", "\u0301", "\u00a0"):
            source = f"&copy{suffix}"
            expected = f"©{suffix}"
            with self.subTest(suffix=suffix):
                for in_attribute in (False, True):
                    errors: list[str] = []
                    assert (
                        decode_entities_in_text(source, in_attribute=in_attribute, report_error=errors.append)
                        == expected
                    )
                    assert errors == ["missing-semicolon-after-character-reference"]
                assert list(stream(source)) == [("text", expected)]
                html = f'<a title="{source}">{source}</a>'
                assert list(stream(html)) == [
                    ("start", ("a", {"title": expected})),
                    ("text", expected),
                    ("end", "a"),
                ]
                for sanitize in (False, True):
                    doc = JustHTML(html, fragment=True, sanitize=sanitize)
                    assert doc.query_one("a").children[0].data == expected
                    assert doc.query_one("a").attrs["title"] == expected

    def test_numeric_references_accept_only_ascii_digits(self):
        # Include Unicode decimal digits that int() accepts as well as digits
        # such as superscripts that would raise ValueError during conversion.
        for digit in "¹²³⁰₀₁₂₃٠١٢۳४９𝟚":
            for semicolon in ("", ";"):
                for prefix, decoded in (("", "&#"), ("65", "A"), ("x", "&#x"), ("x41", "A")):
                    source = f"&#{prefix}{digit}{semicolon}"
                    expected = f"{decoded}{digit}{semicolon}"
                    with self.subTest(source=source):
                        for in_attribute in (False, True):
                            errors: list[str] = []
                            assert (
                                decode_entities_in_text(source, in_attribute=in_attribute, report_error=errors.append)
                                == expected
                            )
                            assert errors == (
                                ["missing-semicolon-after-character-reference"] if decoded == "A" else []
                            )

                        assert list(stream(source)) == [("text", expected)]
                        attribute_source = f'<a title="{source}">x</a>'
                        assert list(stream(attribute_source)) == [
                            ("start", ("a", {"title": expected})),
                            ("text", "x"),
                            ("end", "a"),
                        ]
                        for sanitize in (False, True):
                            assert JustHTML(source, fragment=True, sanitize=sanitize).to_text() == expected
                            doc = JustHTML(attribute_source, fragment=True, sanitize=sanitize)
                            assert doc.query_one("a").attrs["title"] == expected

    def test_unicode_digit_references_scale_linearly(self):
        for reference in ("&#²;", "&#65²;", "&#65;"):
            assert_scales_linearly(
                lambda size, reference=reference: reference * (size * 20),
                decode_entities_in_text,
                label="numeric character references",
            )

    def test_numeric_entity_invalid_ranges_and_replacements(self):
        assert decode_numeric_entity("110000", is_hex=True) == "\ufffd"
        assert decode_numeric_entity("D800", is_hex=True) == "\ufffd"
        assert decode_numeric_entity("80", is_hex=True) == "\u20ac"
        assert decode_numeric_entity("41", is_hex=True) == "A"

    def test_numeric_entity_rejects_digit_strings_too_long_for_int_conversion(self):
        # Python caps decimal string->int conversion length (CVE-2020-10735).
        # A numeric character reference this long is already guaranteed to be
        # out of Unicode's valid range, so it must short-circuit before int()
        # rather than crash with an uncaught ValueError.
        assert decode_numeric_entity("1" * 10000) == "\ufffd"
        assert decode_numeric_entity("f" * 10000, is_hex=True) == "\ufffd"

        # Leading zeros must not trip the length guard for otherwise-valid references.
        assert decode_numeric_entity("0" * 30 + "65") == "A"

    def test_numeric_entity_reports_controls_and_noncharacters(self):
        errors: list[str] = []

        assert decode_numeric_entity("1", report_error=errors.append) == "\x01"
        assert decode_numeric_entity("FDD0", is_hex=True, report_error=errors.append) == "\ufdd0"
        assert decode_numeric_entity("1FFFF", is_hex=True, report_error=errors.append) == "\U0001ffff"
        assert errors == [
            "control-character-reference",
            "noncharacter-character-reference",
            "noncharacter-character-reference",
        ]

    def test_entity_error_reporting_and_invalid_references(self):
        errors: list[str] = []

        assert decode_entities_in_text("a&#65 &#x41; &#; &;", report_error=errors.append) == "aA A &#; &;"
        assert errors == ["missing-semicolon-after-character-reference"]

    def test_named_entity_edge_cases(self):
        errors: list[str] = []

        assert decode_entities_in_text("&amp;&notit;&copy &unknown;", report_error=errors.append) == "&¬it;© &unknown;"
        assert decode_entities_in_text("&copy", report_error=errors.append) == "©"
        assert decode_entities_in_text("&notit", report_error=errors.append) == "¬it"
        assert errors == [
            "missing-semicolon-after-character-reference",
            "missing-semicolon-after-character-reference",
            "missing-semicolon-after-character-reference",
            "missing-semicolon-after-character-reference",
        ]

    def test_attribute_entity_legacy_rules(self):
        assert decode_entities_in_text("&copy=x", in_attribute=True) == "&copy=x"
        assert decode_entities_in_text("&notit", in_attribute=True) == "&notit"
        assert decode_entities_in_text("&copy ", in_attribute=True) == "© "

    def test_long_invalid_named_entity_is_handled_without_unbounded_prefix_search(self):
        suffix = "a" * 100_000
        assert decode_entities_in_text("&" + suffix) == "&" + suffix
        assert decode_entities_in_text("&not" + suffix) == "¬" + suffix
