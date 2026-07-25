#!/usr/bin/env python3
"""Scaling guard for parser, DOM, and serializer complexity regressions.

`parser_adversarial.py` reports absolute timings for a handful of hostile
shapes. This benchmark answers the different question a complexity fix needs:
*did the growth curve change, and is the change larger than run-to-run noise?*

Each scenario is measured at several input sizes. The harness fits an exponent
`k` such that `time ~ n**k` over those sizes, then compares it against the
complexity the scenario is expected to have. A scenario that is meant to be
linear fails the guard when its exponent drifts upward, which is what catches a
newly reintroduced quadratic path.

Typical use around a fix:

    python benchmarks/scaling_guard.py --save before.json
    # ... implement the fix ...
    python benchmarks/scaling_guard.py --compare before.json

`--compare` reports the per-scenario speedup at the largest shared size and
exits non-zero if a scenario got materially slower or grew a worse exponent, so
the command works as a CI gate as well as a development aid.

Scenarios carrying a known quadratic path are declared `expected="quadratic"`,
which reports their exponent without failing the run. **When a fix lands, flip
that scenario to `expected="linear"`** so the guard starts enforcing the new
complexity and a later regression fails instead of being reported as normal.

Use the same interpreter, sizes, and repeat count on both sides of a comparison,
and do not compare timings taken on different machines.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import pathlib
import platform
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from justhtml import JustHTML
from justhtml.parser.stream import stream

# Deep recovery shapes recurse through the tree; keep headroom so that the
# frameset limit probe measures the parser rather than the ambient limit.
_RECURSION_HEADROOM = 100_000

# Exponent above which a scenario expected to be linear is reported as failing.
# Linear scenarios measure around 1.0 and quadratic ones around 2.0, so the gap
# is wide; this threshold sits clear of both measurement noise and the
# log-linear component real parsers carry.
_LINEAR_MAX_EXPONENT = 1.35

# A comparison run must beat these margins before it is called a real change.
_REGRESSION_RATIO = 1.25
_IMPROVEMENT_RATIO = 0.80
_EXPONENT_REGRESSION = 0.25

Setup = Callable[[int], Any]
Run = Callable[[Any], Any]


@dataclass(frozen=True)
class Scenario:
    """One measurable shape.

    `setup` builds the payload (parsing an input tree, composing a source
    string) and is never timed. `run` performs the operation under test.
    """

    name: str
    issue: str
    expected: str  # "linear" or "quadratic"; a fixed scenario becomes "linear" so the guard enforces it
    run: Run
    setup: Setup = lambda size: size
    note: str = ""

    @property
    def is_control(self) -> bool:
        return self.issue == "control"


@dataclass
class Measurement:
    size: int
    best: float  # primary statistic: minimum per-operation time across samples
    median: float
    rsd: float  # relative standard deviation, as a fraction of the median
    inner: int  # operations per timed sample


@dataclass
class Result:
    scenario: Scenario
    measurements: list[Measurement] = field(default_factory=list)
    error: str | None = None

    @property
    def exponent(self) -> float | None:
        """Least-squares slope of log(time) against log(size)."""
        points = [(m.size, m.best) for m in self.measurements if m.best > 0]
        if len(points) < 2:
            return None
        xs = [math.log(size) for size, _ in points]
        ys = [math.log(value) for _, value in points]
        mean_x = statistics.fmean(xs)
        mean_y = statistics.fmean(ys)
        denominator = sum((x - mean_x) ** 2 for x in xs)
        if denominator == 0:
            return None
        return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / denominator

    @property
    def verdict(self) -> str:
        if self.error is not None:
            return "ERROR"
        exponent = self.exponent
        if exponent is None:
            return "n/a"
        if self.scenario.expected == "linear":
            return "ok" if exponent <= _LINEAR_MAX_EXPONENT else "SUPERLINEAR"
        # Scenarios still carrying a known quadratic path. They "pass" in the
        # sense that the guard does not fail the run, but a drop to linear is
        # the outcome a fix is aiming for and is reported as such.
        return "linear now" if exponent <= _LINEAR_MAX_EXPONENT else "quadratic"


# --------------------------------------------------------------------------
# Scenario builders
#
# Each adversarial shape is paired with a control that exercises the same code
# path without the pathological property, so that a fix can be shown to remove
# the quadratic term rather than to slow the control down to match.
# --------------------------------------------------------------------------


def _parse_sized(template: Callable[[int], str], **options: Any) -> tuple[Setup, Run]:
    def setup(size: int) -> str:
        return template(size)

    def run(source: str) -> Any:
        return JustHTML(source, **options)

    return setup, run


def _scenario(
    name: str,
    issue: str,
    expected: str,
    template: Callable[[int], str],
    note: str = "",
    **options: Any,
) -> Scenario:
    setup, run = _parse_sized(template, **options)
    return Scenario(name=name, issue=issue, expected=expected, run=run, setup=setup, note=note)


def _stream_scenario(name: str, issue: str, expected: str, template: Callable[[int], str], note: str = "") -> Scenario:
    def run(source: str) -> None:
        for _event in stream(source):
            pass

    return Scenario(name=name, issue=issue, expected=expected, run=run, setup=template, note=note)


def _pretty_scenario(name: str, issue: str, expected: str, template: Callable[[int], str], note: str = "") -> Scenario:
    def setup(size: int) -> Any:
        return JustHTML(template(size), sanitize=False).root

    return Scenario(
        name=name,
        issue=issue,
        expected=expected,
        run=lambda root: root.to_html(pretty=True),
        setup=setup,
        note=note,
    )


def _clone_scenario(name: str, issue: str, expected: str, template: Callable[[int], str], note: str = "") -> Scenario:
    def setup(size: int) -> Any:
        return JustHTML(template(size), sanitize=False).root

    return Scenario(
        name=name,
        issue=issue,
        expected=expected,
        run=lambda root: root.clone_node(deep=True),
        setup=setup,
        note=note,
    )


def _annotation_xml_source(size: int) -> str:
    attrs = " ".join(f"a{index}" for index in range(size))
    return f"<math><annotation-xml {attrs} encoding=text/html>" + "<svg/>" * size + "</annotation-xml></math>"


def _annotation_xml_control_source(size: int) -> str:
    """The same attribute and child counts on an element that is never an integration point.

    `mrow` is not classified, so this exercises attribute projection and foreign
    child insertion at the same scale without the `encoding` lookup.
    """
    attrs = " ".join(f"a{index}" for index in range(size))
    return f"<math><mrow {attrs} encoding=text/html>" + "<svg/>" * size + "</mrow></math>"


def _distinct_formatting(size: int) -> str:
    return "".join(f"<b id={index}>" for index in range(size))


SCENARIOS: list[Scenario] = [
    # --- Issue 1: document-shell and trailing comment anchors -------------
    _scenario(
        "comment-pre-root",
        "1",
        "linear",
        lambda n: "<!doctype html>" + "<!---->" * n,
        note="scans document children for the html anchor per comment",
        sanitize=False,
    ),
    _scenario(
        "comment-pre-head",
        "1",
        "linear",
        lambda n: "<html>" + "<!---->" * n,
        note="children.index(head) per comment",
        sanitize=False,
    ),
    _scenario(
        "comment-pre-body",
        "1",
        "linear",
        lambda n: "<head></head>" + "<!---->" * n,
        note="children.index(body) per comment",
        sanitize=False,
    ),
    _scenario(
        "comment-after-body",
        "1",
        "linear",
        lambda n: "<!doctype html><html><head></head><body></body></html>" + "<!---->" * n,
        note="ascii_rfind back to </body> per comment",
        sanitize=False,
    ),
    _scenario(
        "comment-in-body",
        "control",
        "linear",
        lambda n: "<!doctype html><body>" + "<!---->" * n,
        note="comments with no shell anchor search",
        sanitize=False,
    ),
    # --- Issue 2: boundary-qualified open-element stack lookups -----------
    _scenario(
        "scope-ordinary",
        "2",
        "linear",
        lambda n: "<x><div>" + "<span>" * n + "</x>" * n,
        note="target present below a special boundary; token ignored, stack unchanged",
        sanitize=False,
    ),
    _scenario(
        "scope-button-p",
        "2",
        "linear",
        lambda n: "<p><button>" + "<b>" * n + "</p>" * n,
        note="button scope boundary for p",
        sanitize=False,
    ),
    _scenario(
        "scope-foreign",
        "2",
        "linear",
        lambda n: "<div><svg><foreignObject><svg>" + "<g>" * n + "</div>" * n,
        note="integration-point boundary walk",
        sanitize=False,
    ),
    _scenario(
        "scope-adoption-table",
        "2",
        "linear",
        lambda n: "<b><table>" + "<span>" * n + "</b>" * n,
        note="formatting target below a table boundary",
        sanitize=False,
    ),
    _scenario(
        "scope-alternating-mutation",
        "2",
        "linear",
        lambda n: "<x><div>" + "<b>" * n + "<i></i></x>" * n,
        note="interleaves stack mutation with the ignored end tag; defeats a whole-stack version cache",
        sanitize=False,
    ),
    _scenario(
        "scope-errors-collected",
        "2",
        "linear",
        lambda n: "<x><div>" + "<b>" * n + "</x>" * n,
        note="same shape with diagnostics enabled",
        sanitize=False,
        collect_errors=True,
    ),
    _stream_scenario(
        "scope-stream",
        "2",
        "linear",
        lambda n: "<div><span><svg>" + "<g>" * n + "</div>" * n,
        note="streaming scanner repeats the same boundary walk",
    ),
    _scenario(
        "scope-object-control",
        "control",
        "linear",
        lambda n: "<b><object>" + "<span>" * n + "</b>" * n,
        note="same shape where the boundary check short-circuits",
        sanitize=False,
    ),
    _scenario(
        "scope-absent-control",
        "control",
        "linear",
        lambda n: "<div>" * n + "</ruby>" * n,
        note="absent target; covered by the existing name-count guard",
        fragment=True,
        sanitize=False,
    ),
    # --- Issue 3: active-formatting list lookups --------------------------
    _scenario(
        "afe-absent-name",
        "3",
        "linear",
        lambda n: _distinct_formatting(n) + "</i>" * n,
        note="reverse scan to the marker for a name that is never present",
    ),
    _scenario(
        "afe-reconstruct-table",
        "9",
        "linear",
        lambda n: _distinct_formatting(n) + "<table>" + "<td>x</td>" * 5,
        note="entry.node in self._stack membership per entry",
        sanitize=False,
    ),
    _scenario(
        "afe-misnested-anchor",
        "9",
        "linear",
        lambda n: "<a><b>" * n,
        note="_refresh_active_formatting_dirty membership scan per token",
        sanitize=False,
    ),
    _scenario(
        "afe-noah-ark-control",
        "control",
        "linear",
        lambda n: "<b>" * n + "x",
        note="duplicate detection is already indexed",
        sanitize=False,
    ),
    # --- Issue 4: MathML annotation-xml integration classification --------
    _scenario(
        "annotation-xml",
        "4",
        "linear",
        _annotation_xml_source,
        note="case-insensitive attribute rescan per child token",
        sanitize=False,
    ),
    _stream_scenario(
        "annotation-xml-stream",
        "4",
        "linear",
        _annotation_xml_source,
        note="same rescan in the streaming scanner",
    ),
    _scenario(
        "annotation-xml-control",
        "control",
        "linear",
        _annotation_xml_control_source,
        note="same attribute and child counts on an element that is never classified",
        sanitize=False,
    ),
    _stream_scenario(
        "annotation-xml-stream-control",
        "control",
        "linear",
        _annotation_xml_control_source,
        note="the same control through the streaming scanner",
    ),
    # --- Issue 6: pretty serialization ------------------------------------
    _pretty_scenario(
        "pretty-nested-inline",
        "6",
        "linear",
        lambda n: "<span>" * n + "x",
        note="_is_blocky_element re-walks each inline subtree per ancestor; pretty=True is the to_html default",
    ),
    _pretty_scenario(
        "pretty-nested-block-control",
        "control",
        "linear",
        lambda n: "<div>" * n + "x",
        note="block names short-circuit the descendant walk",
    ),
    _pretty_scenario(
        "pretty-wide-control",
        "control",
        "linear",
        lambda n: "<div>" + "<span>x</span>" * n + "</div>",
        note="wide but shallow",
    ),
    # --- Issue 7: deep clone of template-bearing trees --------------------
    _clone_scenario(
        "clone-template-deep",
        "7",
        "linear",
        lambda n: "<div><template></template>" * n,
        note="template clones miss the detached-leaf fast path and walk the ancestor chain",
    ),
    _clone_scenario(
        "clone-plain-deep-control",
        "control",
        "linear",
        lambda n: "<div>" * n + "x",
        note="same depth without templates",
    ),
    # --- Issue 8: selectedcontent projection ------------------------------
    _scenario(
        "selectedcontent-deep",
        "8",
        "linear",
        lambda n: (
            "<select><option selected>x</option>" + "<div><selectedcontent></selectedcontent>" * n + "</select>"
        ),
        note="_is_descendant_of walks a growing parent chain per marker",
        sanitize=False,
    ),
    _scenario(
        "selectedcontent-wide-control",
        "control",
        "linear",
        lambda n: "<select><option selected>x</option><selectedcontent></selectedcontent></select>" * n,
        note="many shallow selects",
        sanitize=False,
    ),
    # --- Issue 10: node-identity membership on the open-element stack -----
    _scenario(
        "foreign-parent-membership",
        "10",
        "linear",
        lambda n: "<svg>" + "<g>" * n + "<div>x",
        note="`parent not in self._stack` list scan per foreign insertion",
        sanitize=False,
    ),
    _scenario(
        "foreign-shallow-control",
        "control",
        "linear",
        lambda n: "<svg><g></g></svg>" * n,
        note="the same count of foreign insertions at constant depth, where the membership scan is short",
        sanitize=False,
    ),
    # --- Shapes fixed earlier in this branch, kept so a regression is caught --
    _scenario(
        "foster-parenting",
        "prior",
        "linear",
        lambda n: "<!doctype html><table>" + "<br>" * n + "</table>",
        note="locating the table for each fostered node",
        sanitize=False,
    ),
    _scenario(
        "deep-absent-template",
        "prior",
        "linear",
        lambda n: "<div>" * n + "<table>" + "<br>" * n + "</table>" + "</div>" * n,
        note="template lookup under deep nesting, with no template open",
        fragment=True,
        sanitize=False,
    ),
    _scenario(
        "deep-open-template",
        "prior",
        "linear",
        lambda n: "<template>" + "<div>" * n + "<table>" + "<br>" * n + "</table>" + "</div>" * n + "</template>",
        note="template lookup under deep nesting, with one open",
        fragment=True,
        sanitize=False,
    ),
    _scenario(
        "parser-only-templates",
        "prior",
        "linear",
        lambda n: "<template>" * n + "x" + "</template>" * n,
        note="current-parent lookup through a deep run of parser-only markers",
    ),
    _scenario(
        "collect-errors-unmatched",
        "prior",
        "linear",
        lambda n: "<x>" * n + "</missing>" * n + "</x>" * n,
        note="diagnostic open-tag tracking across unmatched end tags",
        fragment=True,
        sanitize=False,
        collect_errors=True,
    ),
    # --- Hot-path controls: these must not regress -------------------------
    _scenario(
        "hotpath-nested-divs",
        "control",
        "linear",
        lambda n: "<div>" * n + "x" + "</div>" * n,
        note=(
            "well-formed deep nesting; always past the index threshold, so this "
            "is where the index's maintenance cost shows up -- about 23% against "
            "an unindexed walk, flat in depth"
        ),
    ),
    _scenario(
        "hotpath-shallow-nesting",
        "control",
        "linear",
        lambda n: ("<div>" * 16 + "x" + "</div>" * 16) * (n // 16 or 1),
        note="the same element count at a depth real documents reach, where no index is built",
    ),
    _scenario(
        "hotpath-paragraphs",
        "control",
        "linear",
        lambda n: "<p>text</p>" * n,
        note="flat well-formed document",
    ),
    _scenario(
        "hotpath-table",
        "control",
        "linear",
        lambda n: "<table>" + "<tr><td>cell</td></tr>" * n + "</table>",
        note="well-formed table",
    ),
    _scenario(
        "hotpath-attributes",
        "control",
        "linear",
        lambda n: "".join(f'<div id=d{i} class="c{i}" data-x="{i}">t</div>' for i in range(n)),
        note="attribute projection and sanitization",
    ),
]


# --------------------------------------------------------------------------
# Recursion-limit probe (issue 5)
#
# Deep frameset eligibility is an availability cliff rather than a scaling
# curve: it raises RecursionError instead of getting slow. It is reported as a
# maximum supported depth so a fix shows up as a larger number.
# --------------------------------------------------------------------------

LIMIT_PROBES: dict[str, Callable[[int], Any]] = {
    "frameset-eligibility": lambda depth: JustHTML("<div>" * depth + "<frameset>", sanitize=False),
    "frameset-foreign-subtree": lambda depth: JustHTML(
        "<svg>" * depth + "</svg>" * depth + "<frameset>", sanitize=False
    ),
}


def probe_depth_limit(probe: Callable[[int], Any], *, ceiling: int) -> int:
    """Return the largest power-of-two-refined depth the probe survives."""
    low, high = 0, 1
    while high <= ceiling:
        try:
            probe(high)
        except RecursionError:
            break
        except Exception:  # noqa: BLE001 - a non-recursion failure is not a depth limit
            return -1
        low = high
        high *= 2
    else:
        return low
    while low + 1 < high:
        middle = (low + high) // 2
        try:
            probe(middle)
        except RecursionError:
            high = middle
        else:
            low = middle
    return low


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


def measure(scenario: Scenario, size: int, repeats: int, *, min_sample: float) -> Measurement:
    """Time one scenario at one size.

    Two things keep the numbers comparable across runs. Each sample repeats the
    operation enough times to last at least `min_sample`, so that fast shapes
    are not dominated by clock granularity and scheduler jitter; and the garbage
    collector is disabled inside the timed region, since a collection triggered
    by unrelated allocation is the single largest source of run-to-run spread
    here. The minimum across samples is the reported statistic, because noise on
    this kind of measurement is additive: the fastest observation is the one
    least contaminated by it.
    """
    payload = scenario.setup(size)
    scenario.run(payload)  # warm caches, imports, and any lazily built tables

    start = perf_counter()
    scenario.run(payload)
    single = perf_counter() - start
    inner = 1 if single >= min_sample else max(1, min(1000, int(min_sample / single) + 1))

    samples: list[float] = []
    for _ in range(repeats):
        gc.collect()
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            start = perf_counter()
            for _ in range(inner):
                scenario.run(payload)
            elapsed = perf_counter() - start
        finally:
            if gc_was_enabled:
                gc.enable()
        samples.append(elapsed / inner)

    median = statistics.median(samples)
    rsd = (statistics.stdev(samples) / median) if len(samples) > 1 and median > 0 else 0.0
    return Measurement(size=size, best=min(samples), median=median, rsd=rsd, inner=inner)


def run_scenarios(
    scenarios: list[Scenario],
    sizes: list[int],
    repeats: int,
    *,
    budget: float,
    min_sample: float,
) -> list[Result]:
    results: list[Result] = []
    for scenario in scenarios:
        result = Result(scenario=scenario)
        for size in sizes:
            try:
                measurement = measure(scenario, size, repeats, min_sample=min_sample)
            except RecursionError:
                result.error = f"RecursionError at n={size}"
                break
            except Exception as exc:  # noqa: BLE001 - report and keep going
                result.error = f"{type(exc).__name__}: {exc}"
                break
            result.measurements.append(measurement)
            if measurement.best > budget:
                # A known-quadratic shape would take minutes at the next size.
                # Two points are enough to fit an exponent.
                break
        results.append(result)
    return results


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def format_results(results: list[Result], sizes: list[int]) -> str:
    lines = []
    header = f"{'scenario':30s} {'iss':>4s} " + "".join(f"{f'n={size}':>12s}" for size in sizes)
    lines.append(header + f"{'exp':>7s}  verdict")
    lines.append("-" * len(header + f"{'exp':>7s}  verdict"))
    for result in results:
        cells = ""
        by_size = {m.size: m for m in result.measurements}
        for size in sizes:
            measurement = by_size.get(size)
            cells += f"{measurement.best * 1000:11.2f}m" if measurement else f"{'-':>12s}"
        exponent = result.exponent
        exponent_text = f"{exponent:7.2f}" if exponent is not None else f"{'-':>7s}"
        detail = result.error or result.verdict
        lines.append(f"{result.scenario.name:30s} {result.scenario.issue:>4s} {cells}{exponent_text}  {detail}")
    return "\n".join(lines)


def to_payload(results: list[Result], sizes: list[int], repeats: int) -> dict[str, Any]:
    return {
        "version": 1,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "sizes": sizes,
        "repeats": repeats,
        "scenarios": {
            result.scenario.name: {
                "issue": result.scenario.issue,
                "expected": result.scenario.expected,
                "error": result.error,
                "exponent": result.exponent,
                "measurements": [
                    {"size": m.size, "best": m.best, "median": m.median, "rsd": m.rsd, "inner": m.inner}
                    for m in result.measurements
                ],
            }
            for result in results
        },
    }


def compare(results: list[Result], baseline: dict[str, Any]) -> tuple[str, bool]:
    """Report per-scenario change against a saved run. Returns (text, failed)."""
    lines = [
        "",
        f"{'scenario':30s} {'size':>7s} {'before':>11s} {'after':>11s} {'ratio':>8s} {'exp':>14s}  status",
        "-" * 100,
    ]
    failed = False
    stored = baseline.get("scenarios", {})
    if baseline.get("python") != platform.python_version():
        lines.append(f"! baseline python {baseline.get('python')} != current {platform.python_version()}")
    for result in results:
        entry = stored.get(result.scenario.name)
        if entry is None:
            lines.append(f"{result.scenario.name:30s} {'':>7s} {'':>11s} {'':>11s} {'':>8s} {'':>14s}  new")
            continue
        before_by_size = {m["size"]: m.get("best", m.get("median")) for m in entry["measurements"]}
        after_by_size = {m.size: m.best for m in result.measurements}
        shared = sorted(set(before_by_size) & set(after_by_size))
        if not shared:
            lines.append(f"{result.scenario.name:30s} {'':>7s} {'':>11s} {'':>11s} {'':>8s} {'':>14s}  no shared size")
            continue
        size = shared[-1]
        before = before_by_size[size]
        after = after_by_size[size]
        ratio = after / before if before > 0 else math.inf
        before_exponent = entry.get("exponent")
        after_exponent = result.exponent
        exponent_text = (
            f"{before_exponent:5.2f} -> {after_exponent:5.2f}"
            if before_exponent is not None and after_exponent is not None
            else f"{'-':>14s}"
        )

        status = "same"
        if ratio >= _REGRESSION_RATIO:
            status = "REGRESSED"
            failed = True
        elif ratio <= _IMPROVEMENT_RATIO:
            status = "improved"
        if (
            before_exponent is not None
            and after_exponent is not None
            and after_exponent - before_exponent >= _EXPONENT_REGRESSION
        ):
            status = "EXPONENT REGRESSED"
            failed = True
        lines.append(
            f"{result.scenario.name:30s} {size:7d} {before * 1000:10.2f}m {after * 1000:10.2f}m "
            f"{ratio:8.2f} {exponent_text}  {status}"
        )
    return "\n".join(lines), failed


def guard_failures(results: list[Result]) -> list[str]:
    """Scenarios that are supposed to be linear but are not."""
    failures = []
    for result in results:
        if result.error is not None:
            failures.append(f"{result.scenario.name}: {result.error}")
        elif result.scenario.expected == "linear" and result.verdict == "SUPERLINEAR":
            failures.append(f"{result.scenario.name}: exponent {result.exponent:.2f} > {_LINEAR_MAX_EXPONENT}")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[500, 1000, 2000, 4000])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--budget",
        type=float,
        default=2.0,
        help="stop growing a scenario once one measurement exceeds this many seconds (default: 2.0)",
    )
    parser.add_argument(
        "--min-sample",
        type=float,
        default=0.01,
        help="repeat fast scenarios until a timed sample lasts this many seconds (default: 0.01)",
    )
    parser.add_argument("--only", nargs="+", help="substring filters on scenario name")
    parser.add_argument("--issues", nargs="+", help="restrict to these issue ids (e.g. 2 6 control)")
    parser.add_argument("--controls-only", action="store_true", help="run only the linear control scenarios")
    parser.add_argument("--save", type=pathlib.Path, help="write results as a JSON baseline")
    parser.add_argument("--compare", type=pathlib.Path, help="compare against a saved JSON baseline")
    parser.add_argument("--skip-limits", action="store_true", help="skip the recursion-depth probes")
    parser.add_argument(
        "--limit-ceiling",
        type=int,
        default=200_000,
        help="stop the depth probe once this depth succeeds (default: 200000)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when a linear scenario is superlinear (implied by --compare)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeats < 1 or any(size < 1 for size in args.sizes):
        raise SystemExit("sizes and repeats must be positive")
    sizes = sorted(args.sizes)

    scenarios = SCENARIOS
    if args.controls_only:
        scenarios = [s for s in scenarios if s.is_control]
    if args.issues:
        wanted = set(args.issues)
        scenarios = [s for s in scenarios if s.issue in wanted]
    if args.only:
        scenarios = [s for s in scenarios if any(f in s.name for f in args.only)]
    if not scenarios:
        raise SystemExit("no scenarios matched the given filters")

    sys.setrecursionlimit(_RECURSION_HEADROOM)

    print(f"python {platform.python_version()} on {platform.platform()}")
    print(f"sizes={sizes} repeats={args.repeats} scenarios={len(scenarios)}\n")

    results = run_scenarios(scenarios, sizes, args.repeats, budget=args.budget, min_sample=args.min_sample)
    print(format_results(results, sizes))

    noisy = [f"{r.scenario.name}@{m.size} rsd={m.rsd:.0%}" for r in results for m in r.measurements if m.rsd > 0.15]
    if noisy:
        print(f"\nnoisy measurements (raise --repeats or quiet the machine): {', '.join(noisy)}")

    if not args.skip_limits:
        print("\nrecursion-depth limits (issue 5; larger is better)")
        # The probes deliberately run against the interpreter default so the
        # number reflects what an embedding application would actually hit.
        sys.setrecursionlimit(1000)
        try:
            for name, probe in LIMIT_PROBES.items():
                depth = probe_depth_limit(probe, ceiling=args.limit_ceiling)
                text = (
                    "unsupported"
                    if depth < 0
                    else (f">={args.limit_ceiling}" if depth >= args.limit_ceiling else str(depth))
                )
                print(f"  {name:30s} max depth {text}")
        finally:
            sys.setrecursionlimit(_RECURSION_HEADROOM)

    failed = False
    if args.compare:
        baseline = json.loads(args.compare.read_text())
        text, comparison_failed = compare(results, baseline)
        print(text)
        failed = failed or comparison_failed

    if args.save:
        args.save.write_text(json.dumps(to_payload(results, sizes, args.repeats), indent=2) + "\n")
        print(f"\nbaseline written to {args.save}")

    if args.strict or args.compare:
        failures = guard_failures(results)
        if failures:
            print("\nlinear scenarios that are not linear:")
            for failure in failures:
                print(f"  {failure}")
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
