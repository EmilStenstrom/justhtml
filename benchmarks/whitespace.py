#!/usr/bin/env python3
"""Measure whitespace-sensitive queries and serialization on pre-parsed DOMs.

Run against each checkout with the same interpreter and PYTHONPATH=src.
Results are median seconds per operation; parsing and fixture setup are untimed.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import timeit
from typing import TYPE_CHECKING

from justhtml import JustHTML

if TYPE_CHECKING:
    from collections.abc import Callable


def query_cases() -> dict[str, Callable[[], object]]:
    cases: dict[str, Callable[[], object]] = {}
    for shape in ("repeated", "unique", "unicode", "many_tokens", "many_attributes"):
        for size in (10, 1_000):
            paragraphs = []
            for index in range(size):
                if shape == "many_tokens":
                    classes = " ".join(f"c{token}" for token in range(100)) + " active"
                elif shape == "unicode":
                    classes = f"c{index}\u00a0inner active"
                elif shape in {"unique", "many_attributes"}:
                    classes = f"card item active unique{index}"
                else:
                    classes = "card item active selected"
                attrs = f' id="p{index}" data-index="{index}" title="example"' if shape == "many_attributes" else ""
                paragraphs.append(f'<p class="{classes}"{attrs}>x</p>')
            document = JustHTML("<main>" + "".join(paragraphs) + "</main>", sanitize=False)
            for selector in (".active", "p.card.item.active", '[class~="active"]', "p:nth-child(2n + 1)"):
                cases[f"{shape}/{size}/{selector}"] = lambda doc=document, sel=selector: doc.query(sel)
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    cases = query_cases()
    document = JustHTML("<main>" + "<p> alpha  beta <span>word</span> text </p>" * 1_000 + "</main>", sanitize=False)
    cases["pretty/1000"] = lambda: document.to_html(pretty=True)
    cases["compact/1000"] = lambda: document.to_html(pretty=False)
    results = {}
    for label, operation in cases.items():
        operation()
        gc.collect()
        samples = timeit.repeat(operation, number=args.iterations, repeat=args.repeats)
        results[label] = statistics.median(samples) / args.iterations
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
