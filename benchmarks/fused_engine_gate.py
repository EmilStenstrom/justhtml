#!/usr/bin/env python3
"""Performance gate for the fused parse-engine effort.

This benchmark intentionally measures the public default-safe constructor:

    JustHTML(html)

It uses a fixed prefix of the local web100k corpus when available. The command
does not write files; pass a measured baseline with ``--baseline-seconds`` to
turn it into a speedup gate.
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import tarfile
import time

import zstandard as zstd

from justhtml import JustHTML


def _default_web100k_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2] / "web100k"


def _load_web100k_html(root: pathlib.Path, *, limit: int) -> list[str]:
    dict_bytes = (root / "html.dict").read_bytes()
    batch_path = root / "batches" / "web100k-batch-001.tar.zst"
    tar_dctx = zstd.ZstdDecompressor()
    html_dctx = zstd.ZstdDecompressor(dict_data=zstd.ZstdCompressionDict(dict_bytes))
    htmls: list[str] = []

    with batch_path.open("rb") as batch_file:
        with tar_dctx.stream_reader(batch_file) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                for member in tar:
                    if not member.isfile() or not member.name.endswith(".html.zst"):
                        continue
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        continue
                    compressed_html = extracted.read()
                    htmls.append(html_dctx.decompress(compressed_html).decode("utf-8", errors="replace"))
                    if len(htmls) >= limit:
                        break

    if not htmls:
        raise RuntimeError(f"No HTML files found in {batch_path}")
    return htmls


def _measure(htmls: list[str], *, iterations: int) -> list[float]:
    # Warm imports, transform compilation, and branch predictors outside timing.
    for html in htmls[: min(3, len(htmls))]:
        _ = JustHTML(html).root

    timings: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        for html in htmls:
            _ = JustHTML(html).root
        timings.append(time.perf_counter() - start)
    return timings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web100k-dir", type=pathlib.Path, default=_default_web100k_dir())
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--baseline-seconds", type=float, default=None)
    parser.add_argument("--fail-under-speedup", type=float, default=2.0)
    args = parser.parse_args()

    htmls = _load_web100k_html(args.web100k_dir, limit=args.limit)
    timings = _measure(htmls, iterations=args.iterations)
    median = statistics.median(timings)
    mean = statistics.fmean(timings)

    print(f"files: {len(htmls)}")
    print(f"bytes: {sum(len(html) for html in htmls):,}")
    print(f"iterations: {args.iterations}")
    print(f"median_seconds: {median:.6f}")
    print(f"mean_seconds: {mean:.6f}")
    print(f"ms_per_file_median: {median / len(htmls) * 1000:.3f}")

    if args.baseline_seconds is None:
        print("baseline_seconds: not provided")
        return 0

    speedup = args.baseline_seconds / median if median else float("inf")
    target_seconds = args.baseline_seconds / args.fail_under_speedup
    print(f"baseline_seconds: {args.baseline_seconds:.6f}")
    print(f"target_seconds: {target_seconds:.6f}")
    print(f"speedup: {speedup:.3f}x")

    if speedup < args.fail_under_speedup:
        print(
            f"FAIL: speedup {speedup:.3f}x is below required {args.fail_under_speedup:.3f}x",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
