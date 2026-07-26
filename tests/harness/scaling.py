# coverage: exclude file
"""Shared timing assertion for complexity-regression tests.

The guarded input shapes are quadratic when their fast path regresses, so
doubling the input takes roughly four times as long instead of two. This helper
keeps enough margin between those outcomes by collecting before measurement,
disabling the collector during it, and retaining the fastest sample.

Reusable operations are batched when a single call is very short. That avoids
letting timer, scheduler, or JIT noise dominate the measurement.

MAX_GROWTH sits at the quadratic marker rather than halfway between 2x and
4x: on PyPy, whose generational GC amortizes differently than CPython's
refcounting, doubling a fast, allocation-heavy operation like a deep clone
routinely times at 2.5-3x on its own merits, before any CI noise. A tighter
threshold flags that as a regression on every few runs; empirically even
heavy CPU contention on a real regression-free run tops out under 4x, so
this keeps the check meaningful without being a coin flip on PyPy.
"""

import gc
import math
from time import perf_counter

SMALL_SIZE = 2_000
LARGE_SIZE = 4_000
MAX_GROWTH = 4.0
SAMPLES = 5
MIN_SAMPLE_SECONDS = 0.01
MAX_BATCH_RUNS = 128


def _fastest(prepare, run, size: int, *, reusable: bool) -> float:
    payload = prepare(size)
    run(payload)
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        batch_runs = 1
        if reusable:
            start = perf_counter()
            run(payload)
            elapsed = perf_counter() - start
            if elapsed < MIN_SAMPLE_SECONDS:
                batch_runs = min(
                    MAX_BATCH_RUNS,
                    max(2, math.ceil(MIN_SAMPLE_SECONDS / max(elapsed, 1e-9))),
                )

        best = float("inf")
        for _ in range(SAMPLES):
            if not reusable:
                payload = prepare(size)
            start = perf_counter()
            for _ in range(batch_runs):
                run(payload)
            best = min(best, (perf_counter() - start) / batch_runs)
        return best
    finally:
        if was_enabled:
            gc.enable()


def assert_scales_linearly(prepare, run, *, label: str = "input", reusable: bool = True) -> None:
    """Assert that doubling a prepared input keeps the operation linear."""
    small = _fastest(prepare, run, SMALL_SIZE, reusable=reusable)
    large = _fastest(prepare, run, LARGE_SIZE, reusable=reusable)
    growth = large / small
    assert growth < MAX_GROWTH, (
        f"doubling {label} took {growth:.2f}x as long "
        f"({SMALL_SIZE}: {small * 1000:.2f}ms, {LARGE_SIZE}: {large * 1000:.2f}ms); "
        f"linear is ~2x and quadratic ~4x"
    )
