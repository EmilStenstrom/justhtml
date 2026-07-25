"""Shared timing assertion for the complexity-regression tests.

Several suites assert that an operation stays linear as its input doubles. Every
shape they cover is quadratic without its fix, so a reintroduced scan shows up as
roughly four times the runtime rather than two. That gap is what makes a timing
assertion viable here at all: the bound only has to separate 2x from 4x.

Two avoidable sources of spread used to eat that margin, because each copy of
this helper took the median of three samples with the collector live:

- A garbage collection triggered by unrelated allocation landing inside a
  measured region. At these sizes a single collection can add more than the
  signal being measured, and it lands in whichever region happens to cross a
  generation threshold -- so it inflates the ratio as often as it deflates it.
- One unlucky sample out of three, which a median of three cannot outvote.

`benchmarks/scaling_guard.py` handles both, and this mirrors it: collect first
and disable the collector while timing, then keep the fastest sample rather than
the middle one, since noise on this kind of measurement is additive and the
fastest observation is the one least contaminated by it.

Without that, `test_annotation_xml_integration_checks_scale_linearly` failed
reproducibly on some CPython builds and under `coverage` instrumentation, while
the code it guards measured a clean exponent of 1.0 over a decade of input
sizes.
"""

import gc
from time import perf_counter

#: Input sizes to compare. Large enough that a quadratic term dominates, small
#: enough that the suite stays fast.
SMALL_SIZE = 2_000
LARGE_SIZE = 4_000

#: Ratio ceiling for a doubling. Linear measures ~2.0 and quadratic ~4.0, so
#: this sits midway, tolerating noise without tolerating a lost fast path.
MAX_GROWTH = 3.0

#: Timed runs per size. The minimum of these is the reported figure.
SAMPLES = 5


def _fastest(prepare, run, size: int, *, reusable: bool) -> float:
    payload = prepare(size)
    run(payload)  # warm caches, imports, and any lazily built tables
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        best = float("inf")
        for _ in range(SAMPLES):
            if not reusable:
                # The operation consumes its payload, so each sample needs its
                # own. Building it stays outside the timed region.
                payload = prepare(size)
            start = perf_counter()
            run(payload)
            best = min(best, perf_counter() - start)
        return best
    finally:
        if was_enabled:
            gc.enable()


def assert_scales_linearly(prepare, run, *, label: str = "input", reusable: bool = True) -> None:
    """Assert `run` stays linear as the input `prepare` builds doubles.

    `prepare(size)` builds the payload and is not timed; `run(payload)` is. Pass
    `reusable=False` when `run` consumes or mutates its payload, so that each
    sample gets a fresh one.
    """
    small = _fastest(prepare, run, SMALL_SIZE, reusable=reusable)
    large = _fastest(prepare, run, LARGE_SIZE, reusable=reusable)
    growth = large / small
    assert growth < MAX_GROWTH, (
        f"doubling {label} took {growth:.2f}x as long "
        f"({SMALL_SIZE}: {small * 1000:.2f}ms, {LARGE_SIZE}: {large * 1000:.2f}ms); "
        f"linear is ~2x and quadratic ~4x"
    )
