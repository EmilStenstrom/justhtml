#!/usr/bin/env python3
"""Benchmark parser work on malformed and deeply nested HTML shapes."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from statistics import median
from time import perf_counter

from justhtml import JustHTML

Scenario = Callable[[int], tuple[str, dict[str, object]]]


def _foster(size: int) -> tuple[str, dict[str, object]]:
    return "<!doctype html><table>" + "<br>" * size + "</table>", {"sanitize": False}


def _deep_absent_template(size: int) -> tuple[str, dict[str, object]]:
    source = "<div>" * size + "<table>" + "<br>" * size + "</table>" + "</div>" * size
    return source, {"fragment": True, "sanitize": False}


def _deep_open_template(size: int) -> tuple[str, dict[str, object]]:
    source = "<template>" + "<div>" * size + "<table>" + "<br>" * size + "</table>" + "</div>" * size + "</template>"
    return source, {"fragment": True, "sanitize": False}


def _absent_ruby(size: int) -> tuple[str, dict[str, object]]:
    return "<div>" * size + "</ruby>" * size, {"fragment": True, "sanitize": False}


def _collect_errors(size: int) -> tuple[str, dict[str, object]]:
    source = "<x>" * size + "</missing>" * size + "</x>" * size
    return source, {"fragment": True, "sanitize": False, "collect_errors": True}


def _parser_only_templates(size: int) -> tuple[str, dict[str, object]]:
    return "<template>" * size + "x" + "</template>" * size, {}


SCENARIOS: dict[str, Scenario] = {
    "foster": _foster,
    "deep-absent-template": _deep_absent_template,
    "deep-open-template": _deep_open_template,
    "absent-ruby": _absent_ruby,
    "collect-errors": _collect_errors,
    "parser-only-templates": _parser_only_templates,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1_000, 2_000, 4_000])
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats < 1 or any(size < 1 for size in args.sizes):
        raise SystemExit("sizes and repeats must be positive")

    for scenario in SCENARIOS.values():
        source, options = scenario(min(args.sizes))
        JustHTML(source, **options)

    print(f"{'scenario':24s} {'elements':>10s} {'bytes':>10s} {'median':>10s}")
    for size in args.sizes:
        for name, scenario in SCENARIOS.items():
            source, options = scenario(size)
            samples = []
            for _ in range(args.repeats):
                start = perf_counter()
                JustHTML(source, **options)
                samples.append(perf_counter() - start)
            print(f"{name:24s} {size:10d} {len(source):10d} {median(samples):9.6f}s")


if __name__ == "__main__":
    main()
