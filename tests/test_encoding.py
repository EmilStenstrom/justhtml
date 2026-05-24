import unittest

from justhtml import JustHTML
from justhtml.parser import encoding as enc
from justhtml.parser.encoding import decode_html, normalize_encoding_label, sniff_html_encoding
from justhtml.parser.stream import stream


class TestEncoding(unittest.TestCase):
    def test_normalize_encoding_label(self):
        self.assertIsNone(normalize_encoding_label(None))
        self.assertIsNone(normalize_encoding_label(""))
        self.assertIsNone(normalize_encoding_label("   "))

        self.assertEqual(normalize_encoding_label(b"UTF-8"), "utf-8")
        self.assertEqual(normalize_encoding_label("utf7"), "windows-1252")
        self.assertEqual(normalize_encoding_label("iso-8859-1"), "windows-1252")
        self.assertEqual(normalize_encoding_label("iso8859-2"), "iso-8859-2")
        self.assertIsNone(normalize_encoding_label("koi8-r"))

    def test_sniff_transport_overrides(self):
        data = b"<meta charset=iso8859-2>"
        enc_name, bom_len = sniff_html_encoding(data, transport_encoding="utf-8")
        self.assertEqual(enc_name, "utf-8")
        self.assertEqual(bom_len, 0)

    def test_sniff_bom_utf16(self):
        self.assertEqual(sniff_html_encoding(b"\xff\xfeh\x00i\x00")[0], "utf-16le")
        self.assertEqual(sniff_html_encoding(b"\xfe\xff\x00h\x00i")[0], "utf-16be")

    def test_extract_charset_from_content(self):
        self.assertIsNone(enc._extract_charset_from_content(b""))

        # Ensure the ASCII lowercasing branch runs.
        self.assertEqual(enc._extract_charset_from_content(b"TEXT/HTML; CHARSET=UTF-8"), b"utf-8")

        self.assertIsNone(enc._extract_charset_from_content(b"text/html"))
        self.assertIsNone(enc._extract_charset_from_content(b"charset"))
        self.assertIsNone(enc._extract_charset_from_content(b"charset;"))

        self.assertEqual(enc._extract_charset_from_content(b"text/html; charset=iso8859-2"), b"iso8859-2")
        self.assertEqual(enc._extract_charset_from_content(b"text/html; charset='utf-8'"), b"utf-8")
        self.assertEqual(enc._extract_charset_from_content(b'text/html; charset="utf-8"'), b"utf-8")

        # Unterminated quote is ignored.
        self.assertIsNone(enc._extract_charset_from_content(b"text/html; charset='utf-8"))

    def test_prescan_edge_cases(self):
        self.assertIsNone(enc._prescan_for_meta_charset(b"<!--"))

        # End tags are skipped (including quotes, even though it's invalid HTML).
        self.assertIsNone(enc._prescan_for_meta_charset(b"</a title='x'>"))

        # End tag without '>' exercises the skip loop exit.
        self.assertIsNone(enc._prescan_for_meta_charset(b'</a title="x"'))

        # Non-meta tags are skipped so '<' inside an attribute value doesn't start a new tag.
        self.assertIsNone(enc._prescan_for_meta_charset(b'<p title="x><meta charset=iso8859-2>'))

        # Unclosed attribute quote in a meta tag causes the meta to be ignored.
        self.assertIsNone(enc._prescan_for_meta_charset(b'<meta charset="utf-8'))

        # Attribute name/value that runs until EOF exercises loop exit edges.
        self.assertIsNone(enc._prescan_for_meta_charset(b"<meta charset"))
        self.assertIsNone(enc._prescan_for_meta_charset(b"<meta charset=utf-8"))

    def test_decode_html_branches(self):
        text, name = decode_html(b"\x80")
        self.assertEqual(text, "\u20ac")
        self.assertEqual(name, "windows-1252")

        text, name = decode_html(b"abc", transport_encoding="iso-8859-2")
        self.assertEqual(text, "abc")
        self.assertEqual(name, "iso-8859-2")

        text, name = decode_html(b"abc", transport_encoding="euc-jp")
        self.assertEqual(text, "abc")
        self.assertEqual(name, "euc-jp")

        text, name = decode_html(b"\xff\xfeh\x00i\x00")
        self.assertEqual(text, "hi")
        self.assertEqual(name, "utf-16le")

        text, name = decode_html(b"\xfe\xff\x00h\x00i")
        self.assertEqual(text, "hi")
        self.assertEqual(name, "utf-16be")

        text, name = decode_html(b"\xff\xfeh\x00i\x00", transport_encoding="utf-16")
        self.assertEqual(text, "hi")
        self.assertEqual(name, "utf-16")

        text, name = decode_html(b"hi", transport_encoding="utf-8")
        self.assertEqual(text, "hi")
        self.assertEqual(name, "utf-8")

    def test_internal_helpers(self):
        self.assertIsNone(enc._strip_ascii_whitespace(None))

        self.assertIsNone(enc._extract_charset_from_content(b"charset   utf-8"))
        self.assertEqual(enc._extract_charset_from_content(b"charset   =utf-8"), b"utf-8")
        self.assertEqual(enc._extract_charset_from_content(b"charset=   utf-8"), b"utf-8")
        self.assertIsNone(enc._extract_charset_from_content(b"charset=   "))

        # http-equiv path where extracted encoding label is invalid and does not return.
        self.assertIsNone(
            enc._prescan_for_meta_charset(b'<meta http-equiv="Content-Type" content="text/html; charset=bogus">')
        )

    def test_parser_accepts_bytes(self):
        doc = JustHTML(b"<p>hi</p>")
        self.assertEqual(doc.root.children[0].name, "html")

    def test_stream_accepts_bytes(self):
        events = list(stream(b"<p>hi</p>"))
        self.assertTrue(any(e[0] == "start" and e[1][0] == "p" for e in events))


if __name__ == "__main__":
    unittest.main()
