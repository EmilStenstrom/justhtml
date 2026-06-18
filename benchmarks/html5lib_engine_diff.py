#!/usr/bin/env python3
"""Differential scorecard for the default-safe engine on html5lib tree cases.

This compares the compiled public default-safe path:

    JustHTML(html)

against the canonical full parse followed by the public sanitizer:

    document = JustHTML(html, sanitize=False)
    sanitize_dom(document.root)

It is intentionally a default-safe semantic-equivalence scorecard, not an
upstream html5lib pass/fail runner.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for _path in (str(_ROOT), str(_SRC)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from justhtml import JustHTML, sanitize_dom  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@dataclass(slots=True)
class Example:
    file: str
    index: int
    kind: str
    input_html: str
    current: str | None = None
    reference: str | None = None
    current_exception: str | None = None
    reference_exception: str | None = None


def _default_tree_dir() -> Path:
    return _ROOT / "tests" / "html5lib-tests-tree"


def _runner_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "fail_fast": False,
        "test_specs": args.test_specs or [],
        "quiet": True,
        "write_summary": False,
        "exclude_errors": None,
        "exclude_files": args.exclude_files or None,
        "exclude_html": None,
        "filter_html": None,
        "filter_errors": None,
        "verbosity": 0,
        "regressions": False,
        "check_errors": False,
        "suite": "tree",
    }


def _new_runner(test_dir: Path, config: dict[str, object]) -> Any:
    tree_module = importlib.import_module("tests.harness.tree")
    return tree_module.TestRunner(test_dir, config)


def _skip_reason(test: Any) -> str | None:
    if test.script_directive == "script-on":
        return "script_on"
    if test.xml_coercion:
        return "xml_coercion"
    if test.iframe_srcdoc:
        return "iframe_srcdoc"
    return None


def _scripting_enabled(test: Any) -> bool:
    return bool(test.script_directive != "script-off")


def _render_current(test: Any) -> str:
    return JustHTML(
        test.data,
        fragment_context=test.fragment_context,
        scripting_enabled=_scripting_enabled(test),
    ).to_html(pretty=False)


def _render_reference(test: Any) -> str:
    document = JustHTML(
        test.data,
        fragment_context=test.fragment_context,
        scripting_enabled=_scripting_enabled(test),
        sanitize=False,
    )
    sanitize_dom(document.root)
    return document.to_html(pretty=False)


def _try_render(render: Callable[[Any], str], test: Any) -> tuple[str | None, str | None]:
    try:
        return render(test), None
    except Exception as exc:  # noqa: BLE001 - scorecard must keep scanning after parser failures
        return None, f"{type(exc).__name__}: {exc}"


def _iter_cases(runner: Any) -> Iterator[tuple[Path, int, Any]]:
    for file_path, tests in runner.load_tests():
        for index, test in enumerate(tests):
            if runner._should_run_test(file_path.name, index, test):
                yield file_path, index, test


def _preview(value: str | None, *, limit: int = 220) -> str:
    if value is None:
        return "<none>"
    if len(value) <= limit:
        return repr(value)
    return repr(value[: limit - 3] + "...")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree-dir", type=Path, default=_default_tree_dir())
    parser.add_argument("--limit", type=int, default=None, help="Stop after this many scored cases")
    parser.add_argument("--examples", type=int, default=5, help="Number of mismatch/exception examples to print")
    parser.add_argument("--worst-files", type=int, default=10, help="Number of lowest-match files to print")
    parser.add_argument("--total-cases", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--test-specs", nargs="*", default=None, help="Same file[:indices] filter as run_tests.py")
    parser.add_argument("--exclude-files", nargs="*", default=None, help="Skip files containing these substrings")
    parser.add_argument(
        "--fail-under-rate",
        type=float,
        default=None,
        help="Fail when exact/scored rate is below this decimal threshold, e.g. 0.75",
    )
    parser.add_argument(
        "--fail-on-current-exceptions",
        action="store_true",
        help="Fail if the new engine raises where the reference path serializes",
    )
    args = parser.parse_args()

    runner = _new_runner(args.tree_dir, _runner_config(args))
    counts: Counter[str] = Counter()
    excluded: Counter[str] = Counter()
    per_file: defaultdict[str, Counter[str]] = defaultdict(Counter)
    examples: list[Example] = []

    for file_path, index, test in _iter_cases(runner):
        reason = _skip_reason(test)
        if reason is not None:
            excluded[reason] += 1
            continue

        if args.limit is not None and counts["scored"] >= args.limit:
            break

        file_key = file_path.name
        counts["scored"] += 1
        per_file[file_key]["scored"] += 1

        current, current_exc = _try_render(_render_current, test)
        reference, reference_exc = _try_render(_render_reference, test)

        if current_exc is not None and reference_exc is not None and current_exc == reference_exc:
            kind = "exact"
            counts["compared"] += 1
            counts[kind] += 1
            counts["matching_exceptions"] += 1
            per_file[file_key]["compared"] += 1
            per_file[file_key][kind] += 1
            per_file[file_key]["matching_exceptions"] += 1
        elif current_exc is not None and reference_exc is not None:
            kind = "both_exceptions"
            counts[kind] += 1
            per_file[file_key][kind] += 1
        elif current_exc is not None:
            kind = "current_exceptions"
            counts[kind] += 1
            per_file[file_key][kind] += 1
        elif reference_exc is not None:
            kind = "reference_exceptions"
            counts[kind] += 1
            per_file[file_key][kind] += 1
        else:
            counts["compared"] += 1
            per_file[file_key]["compared"] += 1
            if current == reference:
                kind = "exact"
                counts[kind] += 1
                per_file[file_key][kind] += 1
            else:
                kind = "mismatches"
                counts[kind] += 1
                per_file[file_key][kind] += 1

        if kind != "exact" and len(examples) < args.examples:
            examples.append(
                Example(
                    file=file_key,
                    index=index,
                    kind=kind,
                    input_html=test.data,
                    current=current,
                    reference=reference,
                    current_exception=current_exc,
                    reference_exception=reference_exc,
                )
            )

    scored = counts["scored"]
    compared = counts["compared"]
    exact = counts["exact"]
    exact_rate = exact / scored if scored else 0.0
    compared_rate = exact / compared if compared else 0.0

    print(f"tree_dir: {args.tree_dir}")
    print(f"scored_cases: {scored}")
    print(f"compared_cases: {compared}")
    print(f"exact_matches: {exact}")
    print(f"mismatches: {counts['mismatches']}")
    print(f"current_exceptions: {counts['current_exceptions']}")
    print(f"reference_exceptions: {counts['reference_exceptions']}")
    print(f"both_exceptions: {counts['both_exceptions']}")
    print(f"matching_exceptions: {counts['matching_exceptions']}")
    print(f"exact_rate: {exact_rate:.2%}")
    print(f"exact_rate_compared: {compared_rate:.2%}")
    if excluded:
        print("excluded: " + ", ".join(f"{reason}={count}" for reason, count in sorted(excluded.items())))

    if per_file and args.worst_files:
        print("\nworst_files:")
        worst = sorted(
            per_file.items(),
            key=lambda item: (
                item[1]["exact"] / item[1]["scored"] if item[1]["scored"] else 1.0,
                -item[1]["scored"],
                item[0],
            ),
        )
        for file_key, stats in worst[: args.worst_files]:
            rate = stats["exact"] / stats["scored"] if stats["scored"] else 0.0
            print(
                f"  {file_key}: {stats['exact']}/{stats['scored']} exact ({rate:.1%}), "
                f"mismatches={stats['mismatches']}, current_exceptions={stats['current_exceptions']}, "
                f"reference_exceptions={stats['reference_exceptions']}, both_exceptions={stats['both_exceptions']}"
            )

    if examples:
        print("\nexamples:")
        for example in examples:
            print(f"  {example.file}:{example.index} {example.kind}")
            print(f"    input:     {_preview(example.input_html)}")
            if example.current_exception is not None:
                print(f"    current:   {example.current_exception}")
            else:
                print(f"    current:   {_preview(example.current)}")
            if example.reference_exception is not None:
                print(f"    reference: {example.reference_exception}")
            else:
                print(f"    reference: {_preview(example.reference)}")

    if args.fail_on_current_exceptions and counts["current_exceptions"]:
        print("FAIL: new engine raised where reference path serialized", file=sys.stderr)
        return 1
    if args.fail_under_rate is not None and exact_rate < args.fail_under_rate:
        print(
            f"FAIL: exact rate {exact_rate:.2%} is below required {args.fail_under_rate:.2%}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
