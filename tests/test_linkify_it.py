from __future__ import annotations

import unittest
from pathlib import Path

from justhtml.transforms.linkify_core import LinkifyConfig, find_links_with_config


def _iter_fixture_cases(path: str | Path) -> list[tuple[int, str, str]]:
    out: list[tuple[int, str, str]] = []
    with Path(path).open(encoding="utf-8") as f:
        lines = f.read().splitlines()

    skip_next = False
    for idx, raw_line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue

        line = raw_line.split("%", 1)[0]
        if not line.strip():
            continue

        next_line = ""
        if idx + 1 < len(lines):
            next_line = lines[idx + 1].split("%", 1)[0]

        if next_line.strip():
            out.append((idx + 1, line, next_line))
            skip_next = True
        else:
            out.append((idx + 1, line, line))

    return out


class TestLinkifyItFixtures(unittest.TestCase):
    def test_links_fixture(self) -> None:
        cases = _iter_fixture_cases("tests/linkify-it/fixtures/links.txt")
        config = LinkifyConfig(fuzzy_ip=True)

        for line_no, text, expected in cases:
            with self.subTest(line=line_no):
                matches = find_links_with_config(text, config)
                assert matches, f"Expected a match for fixture line {line_no!r}: {text!r}"
                assert matches[0].text == expected

    def test_not_links_fixture(self) -> None:
        cases = _iter_fixture_cases("tests/linkify-it/fixtures/not_links.txt")
        config = LinkifyConfig()

        for line_no, text, _expected in cases:
            with self.subTest(line=line_no):
                matches = find_links_with_config(text, config)
                assert not matches, f"Expected no match for fixture line {line_no!r}: {text!r}"
