#!/usr/bin/env python3
"""Compare JustHTML parse trees with headless Chromium.

Run with an environment containing Playwright, for example:

    PYENV_VERSION=justhtml-html5lib-tests-bench \
      python benchmarks/chromium_parser_diff.py --generated 5000
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_ROOT), str(_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from justhtml import JustHTML  # noqa: E402
from justhtml.dom import Template  # noqa: E402

_KNOWN_CHROMIUM_DIVERGENCES = frozenset(
    {
        "adoption01.dat:17",
        "noscript01.dat:12",
        "tests25.dat:7",
        "tests_innerHTML_1.dat:75",
        "webkit02.dat:11",
        "webkit02.dat:15",
        "webkit02.dat:16",
        "webkit02.dat:17",
    }
)
_NON_ATOMIC_FUZZ_STRATEGIES = frozenset({"fuzz_attribute", "fuzz_nested_structure", "fuzz_open_tag"})


@dataclass(slots=True)
class Case:
    source: str
    html: str
    context_name: str | None = None
    context_namespace: str | None = None
    scripting_enabled: bool = True


def _runner_config() -> dict[str, object]:
    return {
        "fail_fast": False,
        "test_specs": [],
        "quiet": True,
        "write_summary": False,
        "exclude_errors": None,
        "exclude_files": None,
        "exclude_html": None,
        "filter_html": None,
        "filter_errors": None,
        "verbosity": 0,
        "regressions": False,
        "check_errors": False,
        "suite": "tree",
    }


def _html5lib_cases(limit: int | None) -> list[Case]:
    if limit == 0:
        return []
    tree_module = importlib.import_module("tests.harness.tree")
    runner = tree_module.TestRunner(_ROOT / "tests/html5lib-tests-tree", _runner_config())
    cases: list[Case] = []
    for path, tests in runner.load_tests():
        for index, test in enumerate(tests):
            if test.script_directive == "script-on" or test.xml_coercion or test.iframe_srcdoc:
                continue
            context = test.fragment_context
            cases.append(
                Case(
                    source=f"{path.name}:{index}",
                    html=test.data,
                    context_name=context.tag_name if context else None,
                    context_namespace=context.namespace if context else None,
                    scripting_enabled=test.script_directive != "script-off",
                )
            )
            if limit is not None and len(cases) >= limit:
                return cases
    return cases


def _generated_cases(count: int, seed: int) -> list[Case]:
    fuzz = importlib.import_module("benchmarks.fuzz")
    random.seed(seed)
    return [
        Case(source=f"generated:{index}", html=fuzz.generate_fuzzed_html(), scripting_enabled=False)
        for index in range(count)
    ]


def _atomic_generated_cases(count_per_strategy: int, seed: int) -> list[Case]:
    fuzz = importlib.import_module("benchmarks.fuzz")
    random.seed(seed)
    strategies = []
    for name, value in vars(fuzz).items():
        if not name.startswith("fuzz_") or name in _NON_ATOMIC_FUZZ_STRATEGIES or not callable(value):
            continue
        signature = inspect.signature(value)
        if any(
            parameter.default is inspect.Parameter.empty
            and parameter.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
            for parameter in signature.parameters.values()
        ):
            continue
        strategies.append((name, value))

    cases = []
    for name, strategy in sorted(strategies):
        cases.extend(
            [
                Case(
                    source=f"atomic:{name}:{index}",
                    html=strategy(),
                    scripting_enabled=False,
                )
                for index in range(count_per_strategy)
            ]
        )
    return cases


def _canonical_node(node: Any) -> list[Any]:
    name = node.name
    if name == "#text":
        return ["text", node.data or ""]
    if name == "#comment":
        return ["comment", node.data or ""]
    if name == "!doctype":
        doctype = node.data
        return ["doctype", doctype.name or "", doctype.public_id or "", doctype.system_id or ""]

    namespace = node.namespace or "html"
    attrs = [[name, value or ""] for name, value in sorted((node.attrs or {}).items())]
    children = [_canonical_node(child) for child in node.children or ()]
    content = None
    if type(node) is Template and node.template_content is not None:
        content = [_canonical_node(child) for child in node.template_content.children]
    return ["element", namespace, name, attrs, children, content]


def _render_justhtml(case: Case) -> list[Any]:
    context = None
    if case.context_name is not None:
        context_module = importlib.import_module("justhtml.parser.context")
        context = context_module.FragmentContext(case.context_name, case.context_namespace)
    document = JustHTML(
        case.html,
        fragment_context=context,
        scripting_enabled=case.scripting_enabled,
        sanitize=False,
    )
    return [_canonical_node(child) for child in document.root.children or ()]


def _is_known_divergence(case: Case) -> bool:
    if case.source in _KNOWN_CHROMIUM_DIVERGENCES:
        return True
    if case.html.startswith("\ufeff"):
        # DOMParser receives an already-decoded string and preserves this
        # character. JustHTML applies its byte-stream-compatible BOM policy.
        return True
    return case.source.startswith(("atomic:fuzz_scope_terminators:", "atomic:fuzz_formatting_boundary:")) and (
        "<button><button>" in case.html
    )


_CHROMIUM_RENDER = r"""
(cases) => {
  const namespaceUri = {
    html: "http://www.w3.org/1999/xhtml",
    svg: "http://www.w3.org/2000/svg",
    math: "http://www.w3.org/1998/Math/MathML",
    mathml: "http://www.w3.org/1998/Math/MathML",
  };
  const namespaceName = (uri) => {
    if (!uri || uri === namespaceUri.html) return "html";
    if (uri === namespaceUri.svg) return "svg";
    if (uri === namespaceUri.math) return "math";
    return uri;
  };
  const canonical = (node) => {
    if (node.nodeType === Node.TEXT_NODE) return ["text", node.data];
    if (node.nodeType === Node.COMMENT_NODE) return ["comment", node.data];
    if (node.nodeType === Node.DOCUMENT_TYPE_NODE) {
      return ["doctype", node.name || "", node.publicId, node.systemId];
    }
    const attrs = Array.from(node.attributes || [], (attr) => [attr.name, attr.value])
      .sort((a, b) => a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0);
    const children = Array.from(node.childNodes, canonical);
    const content = node.localName === "template" && node.namespaceURI === namespaceUri.html
      ? Array.from(node.content.childNodes, canonical)
      : null;
    return ["element", namespaceName(node.namespaceURI), node.localName, attrs, children, content];
  };
  return cases.map((item) => {
    let nodes;
    if (item.context_name === null) {
      const document = new DOMParser().parseFromString(item.html, "text/html");
      nodes = Array.from(document.childNodes);
    } else {
      const owner = document.implementation.createHTMLDocument("");
      const namespace = namespaceUri[item.context_namespace || "html"] || item.context_namespace;
      const context = owner.createElementNS(namespace, item.context_name);
      context.innerHTML = item.html;
      nodes = item.context_name.toLowerCase() === "template" && namespace === namespaceUri.html
        ? Array.from(context.content.childNodes)
        : Array.from(context.childNodes);
    }
    return nodes.map(canonical);
  });
}
"""


def _render_chromium(cases: list[Case], executable: str, chunk_size: int) -> list[list[Any]]:
    try:
        from playwright.sync_api import sync_playwright  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on benchmark environment
        raise SystemExit("Playwright is required for the Chromium differential") from exc

    output: list[list[Any]] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=executable, headless=True)
        page = browser.new_page()
        for start in range(0, len(cases), chunk_size):
            payload = [
                {
                    "html": case.html,
                    "context_name": case.context_name,
                    "context_namespace": case.context_namespace,
                }
                for case in cases[start : start + chunk_size]
            ]
            rendered = page.evaluate(f"(cases) => JSON.stringify(({_CHROMIUM_RENDER})(cases))", payload)
            output.extend(json.loads(rendered))
        browser.close()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chrome", default="/usr/bin/google-chrome")
    parser.add_argument("--html5lib-limit", type=int)
    parser.add_argument("--generated", type=int, default=0)
    parser.add_argument("--atomic-generated", type=int, default=0, help="Cases per individual fuzz strategy")
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--examples", type=int, default=20)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    cases = _html5lib_cases(args.html5lib_limit)
    cases.extend(_generated_cases(args.generated, args.seed))
    cases.extend(_atomic_generated_cases(args.atomic_generated, args.seed))
    chromium = _render_chromium(cases, args.chrome, args.chunk_size)

    mismatches: list[dict[str, Any]] = []
    known_divergences: list[dict[str, Any]] = []
    exceptions: list[dict[str, str]] = []
    for case, expected in zip(cases, chromium, strict=True):
        try:
            actual = _render_justhtml(case)
        except Exception as exc:  # noqa: BLE001 - differential must continue
            exceptions.append({"source": case.source, "html": case.html, "exception": f"{type(exc).__name__}: {exc}"})
            continue
        if actual != expected:
            mismatch = {
                "source": case.source,
                "html": case.html,
                "context_name": case.context_name,
                "context_namespace": case.context_namespace,
                "justhtml": actual,
                "chromium": expected,
            }
            (known_divergences if _is_known_divergence(case) else mismatches).append(mismatch)

    print(f"chromium: {args.chrome}")
    print(f"cases: {len(cases)}")
    print(f"exact: {len(cases) - len(mismatches) - len(known_divergences) - len(exceptions)}")
    print(f"known_chromium_divergences: {len(known_divergences)}")
    print(f"mismatches: {len(mismatches)}")
    print(f"exceptions: {len(exceptions)}")
    for mismatch in mismatches[: args.examples]:
        print(f"\n{mismatch['source']}: {mismatch['html']!r}")
        print(f"  justhtml: {json.dumps(mismatch['justhtml'], ensure_ascii=False)}")
        print(f"  chromium: {json.dumps(mismatch['chromium'], ensure_ascii=False)}")
    for exception in exceptions[: args.examples]:
        print(f"\n{exception['source']}: {exception['html']!r}")
        print(f"  exception: {exception['exception']}")

    if args.json_output is not None:
        args.json_output.write_text(
            json.dumps(
                {
                    "known_chromium_divergences": known_divergences,
                    "mismatches": mismatches,
                    "exceptions": exceptions,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return int(bool(mismatches or exceptions))


if __name__ == "__main__":
    raise SystemExit(main())
