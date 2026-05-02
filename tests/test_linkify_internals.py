from __future__ import annotations

import unittest
from time import perf_counter

from justhtml import linkify as linkify_mod
from justhtml.linkify import LinkifyConfig, find_links, find_links_with_config


class TestLinkifyInternals(unittest.TestCase):
    def test_helper_functions_cover_branches(self) -> None:
        assert linkify_mod._is_valid_tld("", extra_tlds=frozenset()) is False
        assert linkify_mod._is_valid_tld("se", extra_tlds=frozenset()) is True
        assert linkify_mod._is_valid_tld("xn--p1ai", extra_tlds=frozenset()) is True

        assert linkify_mod._split_domain_for_tld("") is None
        assert linkify_mod._split_domain_for_tld("localhost") == ("localhost", "")
        assert linkify_mod._split_domain_for_tld("nodot") is None

        cfg = LinkifyConfig.with_extra_tlds(["EXAMPLE"])
        assert cfg.extra_tlds == frozenset({"example"})

        assert linkify_mod._is_valid_ipv4("1.2.3") is False
        assert linkify_mod._is_valid_ipv4("1..3.4") is False
        assert linkify_mod._is_valid_ipv4("1.2.3.a") is False
        assert linkify_mod._is_valid_ipv4("999.2.3.4") is False

        # Trigger the UnicodeError branch.
        assert linkify_mod._punycode_host("\ud800") == "\ud800"

        assert linkify_mod._strip_wrapping("<x>") == ("x", 1, 1)
        assert linkify_mod._strip_wrapping("x") == ("x", 0, 0)

        assert linkify_mod._trim_trailing("") == ""

        # Cover the "unknown prefix" branch and IPv6 early return.
        assert linkify_mod._punycode_href("gopher://example.com") == "gopher//example.com"
        assert linkify_mod._punycode_href("http://[::1]/") == "http://[::1]/"

        assert find_links("") == []
        assert find_links_with_config("", LinkifyConfig()) == []

    def test_find_links_validation_branches(self) -> None:
        # Broken schema: fuzzy domain immediately after :// should be skipped.
        assert find_links_with_config("hppt://example.com", LinkifyConfig()) == []

        # Protocol-relative host parsing branches.
        assert find_links_with_config("//[::1]", LinkifyConfig()) == []
        assert find_links_with_config("//a_b.com", LinkifyConfig()) == []

        # Fuzzy email validation branches.
        assert find_links_with_config("a@b_c.com", LinkifyConfig()) == []
        assert find_links_with_config("a@b.example", LinkifyConfig()) == []

        cfg = LinkifyConfig.with_extra_tlds(["example"])
        m = find_links_with_config("a@b.example", cfg)
        assert len(m) == 1
        assert m[0].href == "mailto:a@b.example"

        # Scheme URL host/port validation branches.
        assert find_links_with_config("http://example.com:99999", LinkifyConfig()) == []
        assert find_links_with_config("http://exa_mple.com", LinkifyConfig()) == []
        assert find_links_with_config("http://999.999.999.999", LinkifyConfig()) == []

        # Protocol-relative URLs with userinfo and ports.
        m2 = find_links_with_config("See //user@a.com:8080/x", LinkifyConfig())
        assert any(x.href == "//user@a.com:8080/x" for x in m2)

    def test_punctuation_runs_do_not_trigger_quadratic_email_scan(self) -> None:
        for ch in ('"', "-", ".", "!"):
            with self.subTest(ch=ch):
                start = perf_counter()
                assert find_links_with_config(ch * 20_000, LinkifyConfig()) == []
                assert perf_counter() - start < 0.25

    def test_unmatched_closing_brackets_do_not_trigger_quadratic_trimming(self) -> None:
        for ch in (")", "]", "}", ">"):
            with self.subTest(ch=ch):
                start = perf_counter()
                out = find_links_with_config("http://example.com/" + ch * 20_000, LinkifyConfig())
                assert len(out) == 1
                assert out[0].text == "http://example.com/"
                assert perf_counter() - start < 0.25

    def test_overlap_and_unreachable_continues_via_stubs(self) -> None:
        class _FakeMatch:
            __slots__ = ("_cand", "_end2", "_start2")

            def __init__(self, cand: str, *, start2: int, end2: int) -> None:
                self._cand = cand
                self._start2 = start2
                self._end2 = end2

            def group(self, n: int) -> str:
                assert n == 2
                return self._cand

            def start(self, n: int) -> int:
                assert n == 2
                return self._start2

            def end(self, n: int) -> int:
                assert n == 2
                return self._end2

        class _FakePattern:
            __slots__ = ("_matches",)

            def __init__(self, matches: list[_FakeMatch]) -> None:
                self._matches = matches

            def finditer(self, _text: str):
                yield from self._matches

        orig_re = linkify_mod._CANDIDATE_RE
        orig_trim = linkify_mod._trim_trailing
        orig_strip = linkify_mod._strip_wrapping
        try:
            # Force an empty candidate after trimming -> covers the early continue.
            linkify_mod._CANDIDATE_RE = _FakePattern([_FakeMatch("...", start2=1, end2=4)])  # type: ignore[assignment]
            assert find_links_with_config("...", LinkifyConfig()) == []

            # Force the markdown termination -> empty after the second trim.
            def fake_trim(s: str) -> str:
                if s == "http://":
                    return ""
                return orig_trim(s)

            linkify_mod._trim_trailing = fake_trim  # type: ignore[assignment]
            linkify_mod._CANDIDATE_RE = _FakePattern([_FakeMatch("http://)[x", start2=1, end2=10)])  # type: ignore[assignment]
            assert find_links_with_config("http://)[x", LinkifyConfig()) == []

            # Force the "leading quote becomes empty" continue.
            linkify_mod._trim_trailing = lambda s: s  # type: ignore[assignment]
            linkify_mod._strip_wrapping = lambda s: (s, 0, 0)  # type: ignore[assignment]
            linkify_mod._CANDIDATE_RE = _FakePattern([_FakeMatch("'", start2=1, end2=2)])  # type: ignore[assignment]
            assert find_links_with_config("'", LinkifyConfig()) == []

            linkify_mod._CANDIDATE_RE = _FakePattern([_FakeMatch("'example.com", start2=1, end2=13)])  # type: ignore[assignment]
            quoted_out = find_links_with_config("'example.com", LinkifyConfig())
            assert len(quoted_out) == 1
            assert quoted_out[0].text == "example.com"

            # Force the broken-schema slice (start-3:start == "://") branch.
            linkify_mod._strip_wrapping = orig_strip  # type: ignore[assignment]
            linkify_mod._trim_trailing = orig_trim  # type: ignore[assignment]
            linkify_mod._CANDIDATE_RE = _FakePattern([_FakeMatch("example.com", start2=7, end2=18)])  # type: ignore[assignment]
            assert find_links_with_config("abc://example.com", LinkifyConfig()) == []

            # Force parts-is-None branches in domain and email validation.
            linkify_mod._CANDIDATE_RE = _FakePattern([_FakeMatch("nodot", start2=1, end2=6)])  # type: ignore[assignment]
            assert find_links_with_config("nodot", LinkifyConfig()) == []

            linkify_mod._CANDIDATE_RE = _FakePattern([_FakeMatch("a@nodot", start2=1, end2=8)])  # type: ignore[assignment]
            assert find_links_with_config("a@nodot", LinkifyConfig()) == []

            # Force overlapping matches so the overlap filter path is exercised.
            linkify_mod._trim_trailing = orig_trim  # type: ignore[assignment]
            linkify_mod._CANDIDATE_RE = _FakePattern(
                [
                    _FakeMatch("http://a.com", start2=1, end2=13),
                    _FakeMatch("http://a.com", start2=5, end2=17),
                ]
            )  # type: ignore[assignment]
            out = find_links_with_config("http://a.com", LinkifyConfig())
            assert len(out) == 1
            assert out[0].text == "http://a.com"
        finally:
            linkify_mod._CANDIDATE_RE = orig_re  # type: ignore[assignment]
            linkify_mod._trim_trailing = orig_trim  # type: ignore[assignment]
            linkify_mod._strip_wrapping = orig_strip  # type: ignore[assignment]
