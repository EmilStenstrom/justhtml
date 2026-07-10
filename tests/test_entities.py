import unittest

from justhtml.core.entities import decode_entities_in_text, decode_numeric_entity


class TestEntities(unittest.TestCase):
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
