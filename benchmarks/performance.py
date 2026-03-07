#!/usr/bin/env python3
"""
Performance benchmark for JustHTML against other HTML parsers.
Uses the web100k dataset:
    https://github.com/EmilStenstrom/web100k

Defaults assume a sibling-folder layout (next to the repo):
    ../web100k/batches/
    ../web100k/html.dict

Override with `--batches-dir` / `--dict` or set `WEB100K_DIR`.
Decompresses at runtime (no disk writes) using html.dict for optimal performance.
"""

# ruff: noqa: PERF203, PLC0415, BLE001, S110

import argparse
import multiprocessing
import os  # MEMORY: added
import pathlib
import sys
import tarfile
import threading  # MEMORY: added
import time


def _default_web100k_dir() -> pathlib.Path:
    """Resolve a portable default location for the web100k dataset."""
    env = os.environ.get("WEB100K_DIR") or os.environ.get("WEB100K_PATH")
    if env:
        return pathlib.Path(env)
    # Default to a sibling folder next to the repo: <repo_parent>/web100k
    return pathlib.Path(__file__).resolve().parents[2] / "web100k"


try:
    import zstandard as zstd
except ImportError:
    print("ERROR: zstandard is required. Install with: pip install zstandard")
    sys.exit(1)

try:
    import psutil
except ImportError:
    print("ERROR: psutil is required. Install with: pip install psutil")
    sys.exit(1)


# MEMORY: lightweight RSS monitor using psutil
class MemoryMonitor:
    def __init__(self, pid=None, sample_interval=0.01):
        """
        pid: process ID to monitor (default: current process).
        sample_interval: seconds between samples (default 10ms).
        """
        self.sample_interval = sample_interval
        self._stop = threading.Event()
        self._thread = None
        target_pid = pid if pid is not None else os.getpid()
        self._proc = psutil.Process(target_pid)
        self.start_rss = None
        self.end_rss = None
        self.peak_rss = None
        self.last_rss = None
        self.samples = 0

    def _get_rss(self):
        try:
            return self._proc.memory_info().rss
        except Exception:
            return None

    def start(self):
        self.start_rss = self._get_rss()
        self.peak_rss = self.start_rss
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            rss = self._get_rss()
            if rss is not None:
                self.last_rss = rss
                if self.peak_rss is None or rss > self.peak_rss:
                    self.peak_rss = rss
                self.samples += 1
            self._stop.wait(self.sample_interval)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

        # Try to get current RSS, if fail (process dead), use last seen
        current = self._get_rss()
        if current and current > 0:
            self.end_rss = current
        else:
            self.end_rss = self.last_rss

    def to_dict(self):
        def mb(x):
            return (x or 0) / (1024 * 1024)

        start_mb = mb(self.start_rss)
        end_mb = mb(self.end_rss)
        peak_mb = mb(self.peak_rss)
        delta_mb = end_mb - start_mb if (self.end_rss is not None and self.start_rss is not None) else 0.0
        return {
            "rss_start_mb": start_mb,
            "rss_end_mb": end_mb,
            "rss_delta_mb": delta_mb,
            "rss_peak_mb": peak_mb,
            "mem_samples": self.samples,
        }


def load_dict(dict_path):
    """Load the zstd dictionary required for decompression."""
    if not dict_path.exists():
        print(f"ERROR: Dictionary not found at {dict_path}")
        sys.exit(1)
    return dict_path.read_bytes()


def iter_html_from_batch(
    batch_path,
    dict_bytes,
    limit=None,
):
    """
    Stream HTML files from a compressed batch without writing to disk.
    Yields (filename, html_content) tuples.
    """
    if not batch_path.exists():
        print(f"ERROR: Batch file not found at {batch_path}")
        sys.exit(1)
    tar_dctx = zstd.ZstdDecompressor()
    with batch_path.open("rb") as batch_file:
        with tar_dctx.stream_reader(batch_file) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                html_dctx = zstd.ZstdDecompressor(
                    dict_data=zstd.ZstdCompressionDict(dict_bytes),
                )
                count = 0
                for member in tar:
                    if not member.isfile() or not member.name.endswith(".html.zst"):
                        continue
                    if limit and count >= limit:
                        return
                    compressed_html = tar.extractfile(member).read()
                    html_content = html_dctx.decompress(compressed_html).decode(
                        "utf-8",
                        errors="replace",
                    )
                    yield (member.name, html_content)
                    count += 1


def iter_html_from_downloaded(
    downloaded_dir,
    dict_bytes,
    limit=None,
):
    """
    Load HTML files from downloaded directory (*.html.zst files).
    Yields (filename, html_content) tuples.
    """
    if not downloaded_dir.exists():
        print(f"ERROR: Downloaded directory not found at {downloaded_dir}")
        sys.exit(1)
    html_dctx = zstd.ZstdDecompressor(
        dict_data=zstd.ZstdCompressionDict(dict_bytes),
    )
    html_files = sorted(downloaded_dir.glob("*.html.zst"))
    if limit:
        html_files = html_files[:limit]
    for file_path in html_files:
        try:
            compressed = file_path.read_bytes()
            html_content = html_dctx.decompress(compressed).decode("utf-8", errors="replace")
            yield (file_path.name, html_content)
        except Exception as e:
            print(f"Warning: Failed to decompress {file_path.name}: {e}")
            continue


def iter_html_from_all_batches(
    batches_dir,
    dict_bytes,
    limit=None,
):
    """
    Stream HTML files from all batch files in a directory.
    Yields (filename, html_content) tuples.
    """
    if not batches_dir.exists():
        print(f"ERROR: Batches directory not found at {batches_dir}")
        sys.exit(1)
    batch_files = sorted(batches_dir.glob("web100k-batch-*.tar.zst"))
    if not batch_files:
        print(f"ERROR: No batch files found in {batches_dir}")
        sys.exit(1)
    count = 0
    for batch_file in batch_files:
        print(f"  Loading {batch_file.name}...")
        for item in iter_html_from_batch(batch_file, dict_bytes, limit=None):
            yield item
            count += 1
            if limit and count >= limit:
                return


def benchmark_justhtml(html_source, iterations=1):
    """Benchmark JustHTML parser with Rust tokenizer."""
    try:
        from justhtml import JustHTML
    except ImportError:
        return {"error": "JustHTML not importable"}
    all_times = []
    errors = 0
    error_files = []
    total_bytes = 0
    file_count = 0
    warmup_done = False
    for filename, html in html_source:
        if not warmup_done:
            try:
                JustHTML(html)
            except Exception:
                pass
            warmup_done = True
        total_bytes += len(html)
        file_count += 1
        for _ in range(iterations):
            try:
                start = time.perf_counter()
                result = JustHTML(html)
                elapsed = time.perf_counter() - start
                all_times.append(elapsed)
                _ = result.root
            except Exception as e:
                errors += 1
                error_files.append((filename, str(e)))
    return {
        "total_time": sum(all_times),
        "mean_time": sum(all_times) / len(all_times) if all_times else 0,
        "min_time": min(all_times) if all_times else 0,
        "max_time": max(all_times) if all_times else 0,
        "errors": errors,
        "success_count": len(all_times),
        "error_files": error_files,
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def benchmark_justhtml_to_html(html_source, iterations=1):
    """Benchmark JustHTML parse + serialize via to_html() (safe-by-default)."""
    try:
        from justhtml import JustHTML
    except ImportError:
        return {"error": "JustHTML not importable"}

    all_times = []
    errors = 0
    error_files = []
    total_bytes = 0
    file_count = 0
    warmup_done = False

    for filename, html in html_source:
        if not warmup_done:
            try:
                JustHTML(html).to_html(pretty=False)
            except Exception:
                pass
            warmup_done = True

        total_bytes += len(html)
        file_count += 1
        for _ in range(iterations):
            try:
                start = time.perf_counter()
                out = JustHTML(html).to_html(pretty=False)
                elapsed = time.perf_counter() - start
                all_times.append(elapsed)
                _ = len(out)
            except Exception as e:
                errors += 1
                error_files.append((filename, str(e)))

    return {
        "total_time": sum(all_times),
        "mean_time": sum(all_times) / len(all_times) if all_times else 0,
        "min_time": min(all_times) if all_times else 0,
        "max_time": max(all_times) if all_times else 0,
        "errors": errors,
        "success_count": len(all_times),
        "error_files": error_files,
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def benchmark_html5lib(html_source, iterations=1):
    """Benchmark html5lib parser."""
    try:
        import html5lib
    except ImportError:
        return {"error": "html5lib not installed (pip install html5lib)"}
    all_times = []
    errors = 0
    total_bytes = 0
    file_count = 0
    warmup_done = False
    for _, html in html_source:
        if not warmup_done:
            try:
                html5lib.parse(html)
            except Exception:
                pass
            warmup_done = True
        total_bytes += len(html)
        file_count += 1
        for _ in range(iterations):
            try:
                start = time.perf_counter()
                result = html5lib.parse(html)
                elapsed = time.perf_counter() - start
                all_times.append(elapsed)
                _ = result
            except Exception:
                errors += 1
    return {
        "total_time": sum(all_times),
        "mean_time": sum(all_times) / len(all_times) if all_times else 0,
        "min_time": min(all_times) if all_times else 0,
        "max_time": max(all_times) if all_times else 0,
        "errors": errors,
        "success_count": len(all_times),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def benchmark_lxml(html_source, iterations=1):
    """Benchmark lxml parser."""
    try:
        from lxml import html as lxml_html
    except ImportError:
        return {"error": "lxml not installed (pip install lxml)"}
    times = []
    errors = 0
    total_bytes = 0
    file_count = 0
    warmup_done = False
    for _, content in html_source:
        if not warmup_done:
            try:
                lxml_html.fromstring(content)
            except Exception:
                pass
            warmup_done = True
        total_bytes += len(content)
        file_count += 1
        for _ in range(iterations):
            try:
                start = time.perf_counter()
                result = lxml_html.fromstring(content)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                _ = result
            except Exception:
                errors += 1
    return {
        "total_time": sum(times),
        "mean_time": sum(times) / len(times) if times else 0,
        "min_time": min(times) if times else 0,
        "max_time": max(times) if times else 0,
        "errors": errors,
        "success_count": len(times),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def benchmark_bs4(html_source, iterations=1):
    """Benchmark BeautifulSoup4 parser."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return {"error": "beautifulsoup4 not installed (pip install beautifulsoup4)"}
    times = []
    errors = 0
    total_bytes = 0
    file_count = 0
    warmup_done = False
    for _, html in html_source:
        if not warmup_done:
            try:
                BeautifulSoup(html, "html.parser")
            except Exception:
                pass
            warmup_done = True
        total_bytes += len(html)
        file_count += 1
        for _ in range(iterations):
            try:
                start = time.perf_counter()
                result = BeautifulSoup(html, "html.parser")
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                _ = result.name
            except Exception:
                errors += 1
    return {
        "total_time": sum(times),
        "mean_time": sum(times) / len(times) if times else 0,
        "min_time": min(times) if times else 0,
        "max_time": max(times) if times else 0,
        "errors": errors,
        "success_count": len(times),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def benchmark_html_parser(html_source, iterations=1):
    """Benchmark stdlib html.parser."""
    try:
        from html.parser import HTMLParser
    except ImportError:
        return {"error": "html.parser not available (stdlib)"}

    class SimpleHTMLParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.data = []

        def handle_starttag(self, tag, attrs):
            self.data.append(("start", tag, attrs))

        def handle_endtag(self, tag):
            self.data.append(("end", tag))

        def handle_data(self, data):
            self.data.append(("data", data))

    times = []
    errors = 0
    total_bytes = 0
    file_count = 0
    warmup_done = False
    for _, html in html_source:
        if not warmup_done:
            try:
                parser = SimpleHTMLParser()
                parser.feed(html)
            except Exception:
                pass
            warmup_done = True
        total_bytes += len(html)
        file_count += 1
        for _ in range(iterations):
            try:
                start = time.perf_counter()
                parser = SimpleHTMLParser()
                parser.feed(html)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                _ = parser.data
            except Exception:
                errors += 1
    return {
        "total_time": sum(times),
        "mean_time": sum(times) / len(times) if times else 0,
        "min_time": min(times) if times else 0,
        "max_time": max(times) if times else 0,
        "errors": errors,
        "success_count": len(times),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def benchmark_selectolax(html_source, iterations=1):
    """Benchmark selectolax parser."""
    try:
        from selectolax.parser import HTMLParser
    except ImportError:
        return {"error": "selectolax not installed (pip install selectolax)"}
    times = []
    errors = 0
    total_bytes = 0
    file_count = 0
    warmup_done = False
    for _, html in html_source:
        if not warmup_done:
            try:
                HTMLParser(html)
            except Exception:
                pass
            warmup_done = True
        total_bytes += len(html)
        file_count += 1
        for _ in range(iterations):
            try:
                start = time.perf_counter()
                result = HTMLParser(html)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                _ = result.root
            except Exception:
                errors += 1
    return {
        "total_time": sum(times),
        "mean_time": sum(times) / len(times) if times else 0,
        "min_time": min(times) if times else 0,
        "max_time": max(times) if times else 0,
        "errors": errors,
        "success_count": len(times),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def benchmark_gumbo(html_source, iterations=1):
    """Benchmark Gumbo parser (via html5-parser)."""
    try:
        import html5_parser
    except ImportError:
        return {"error": "html5-parser not installed (pip install html5-parser)"}
    times = []
    errors = 0
    total_bytes = 0
    file_count = 0
    warmup_done = False
    for _, html in html_source:
        if not warmup_done:
            try:
                html5_parser.parse(html)
            except Exception:
                pass
            warmup_done = True
        total_bytes += len(html)
        file_count += 1
        for _ in range(iterations):
            try:
                start = time.perf_counter()
                result = html5_parser.parse(html)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                _ = result.tag
            except Exception:
                errors += 1
    return {
        "total_time": sum(times),
        "mean_time": sum(times) / len(times) if times else 0,
        "min_time": min(times) if times else 0,
        "max_time": max(times) if times else 0,
        "errors": errors,
        "success_count": len(times),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def benchmark_markupever(html_source, iterations=1):
    """Benchmark markupever parser."""
    try:
        from markupever import parse
    except ImportError:
        return {"error": "markupever not installed (pip install markupever)"}
    times = []
    errors = 0
    total_bytes = 0
    file_count = 0
    warmup_done = False
    for _, html in html_source:
        if not warmup_done:
            try:
                parse(html)
            except Exception:
                pass
            warmup_done = True
        total_bytes += len(html)
        file_count += 1
        for _ in range(iterations):
            try:
                start = time.perf_counter()
                result = parse(html)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                _ = result.root()
            except Exception:
                errors += 1
    return {
        "total_time": sum(times),
        "mean_time": sum(times) / len(times) if times else 0,
        "min_time": min(times) if times else 0,
        "max_time": max(times) if times else 0,
        "errors": errors,
        "success_count": len(times),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def _benchmark_worker(bench_fn, html_files, iterations, queue):
    """Worker function to run benchmark in a separate process."""
    try:
        res = bench_fn(html_files, iterations)
        queue.put(res)
    except Exception as e:
        queue.put({"error": str(e)})


def run_benchmark_isolated(bench_fn, html_files, iterations, args):
    """Run benchmark in a separate process to isolate memory usage."""
    # Materialize generator to list (required for multiprocessing pickling)
    if not isinstance(html_files, list):
        html_files = list(html_files)

    if args.no_mem:
        return bench_fn(html_files, iterations)

    # Force GC in parent to minimize COW overhead (though fork handles it)
    import gc

    gc.collect()

    queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_benchmark_worker,
        args=(bench_fn, html_files, iterations, queue),
    )
    p.start()

    # Monitor the child process
    mon = MemoryMonitor(pid=p.pid, sample_interval=max(0.0005, args.mem_sample_ms / 1000.0))
    mon.start()

    res = None
    try:
        res = queue.get()
    finally:
        mon.stop()
        p.join()

    if res and "error" not in res:
        res.update(mon.to_dict())
    return res


def print_results(results, file_count, iterations=1):
    """Pretty print benchmark results."""
    print("\n" + "=" * 100)
    if iterations > 1:
        print(f"BENCHMARK RESULTS ({file_count} HTML files x {iterations} iterations)")
    else:
        print(f"BENCHMARK RESULTS ({file_count} HTML files)")
    print("=" * 100)
    parsers = [
        "justhtml",
        "justhtml_to_html",
        "html5lib",
        "lxml",
        "bs4",
        "html.parser",
        "selectolax",
        "gumbo",
        "markupever",
    ]

    # Combined header
    header = f"\n{'Parser':<15} {'Total (s)':<10} {'Mean (ms)':<10} {'Peak (MB)':<10} {'Delta (MB)':<10}"
    print(header)
    print("-" * 100)

    justhtml_time = results.get("justhtml", {}).get("total_time", 0)

    for parser in parsers:
        if parser not in results:
            continue
        result = results[parser]
        if "error" in result:
            print(f"{parser:<15} {result['error']}")
            continue

        total = result["total_time"]
        mean_ms = result["mean_time"] * 1000

        # Memory stats
        peak_mb = result.get("rss_peak_mb", 0)
        delta_mb = result.get("rss_delta_mb", 0)
        mem_str = f"{peak_mb:>10.1f} {delta_mb:>10.1f}" if "rss_peak_mb" in result else f"{'n/a':>10} {'n/a':>10}"

        speedup = ""
        if parser != "justhtml" and justhtml_time > 0 and total > 0:
            speedup_factor = justhtml_time / total
            if speedup_factor > 1:
                speedup = f" ({speedup_factor:.2f}x faster)"
            else:
                speedup = f" ({1 / speedup_factor:.2f}x slower)"

        print(f"{parser:<15} {total:<10.3f} {mean_ms:<10.3f} {mem_str} {speedup}")

    print("\n" + "=" * 100)

    # Error details
    for parser in parsers:
        if parser not in results:
            continue
        result = results[parser]
        error_files = result.get("error_files", [])
        if error_files:
            print(f"\nErrors for {parser}:")
            for filename, error_msg in error_files:
                print(f"  {filename}: {error_msg}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark HTML parsers using web100k dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    default_web100k = _default_web100k_dir()
    parser.add_argument("--batch", type=pathlib.Path, help="Path to single batch file")
    parser.add_argument(
        "--batches-dir",
        type=pathlib.Path,
        default=default_web100k / "batches",
        help="Path to directory containing all batch files (default: ../web100k/batches; override with WEB100K_DIR)",
    )
    parser.add_argument("--downloaded", type=pathlib.Path, help="Path to downloaded directory with .html.zst files")
    parser.add_argument("--all-batches", action="store_true", help="Process all batch files in batches-dir")
    parser.add_argument(
        "--dict",
        type=pathlib.Path,
        default=default_web100k / "html.dict",
        help="Path to html.dict file (default: ../web100k/html.dict; override with WEB100K_DIR)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Limit number of files to test (default: 100, use 0 for all)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of iterations to run for averaging (default: 5)",
    )
    parser.add_argument(
        "--parsers",
        nargs="+",
        choices=[
            "justhtml",
            "justhtml_to_html",
            "html5lib",
            "lxml",
            "bs4",
            "html.parser",
            "selectolax",
            "gumbo",
            "markupever",
        ],
        default=["justhtml", "html5lib", "lxml", "bs4", "html.parser", "selectolax", "gumbo", "markupever"],
        help="Parsers to benchmark (default: all)",
    )
    # MEMORY: options
    parser.add_argument("--no-mem", action="store_true", help="Disable memory measurement (RSS sampling)")
    parser.add_argument(
        "--mem-sample-ms",
        type=float,
        default=10.0,
        help="Memory sampling interval in milliseconds (default: 10ms)",
    )

    args = parser.parse_args()

    # Load dictionary
    print(f"Loading dictionary from {args.dict}...")
    dict_bytes = load_dict(args.dict)

    # Create a factory function that returns fresh generators for each benchmark
    limit = args.limit if args.limit > 0 else None
    if args.downloaded:
        print(f"Will stream HTML files from {args.downloaded}...")

        def html_source_factory():
            return iter_html_from_downloaded(args.downloaded, dict_bytes, limit)
    elif args.all_batches:
        print(f"Will stream HTML files from all batches in {args.batches_dir}...")

        def html_source_factory():
            return iter_html_from_all_batches(args.batches_dir, dict_bytes, limit)
    elif args.batch:
        print(f"Will stream HTML files from {args.batch}...")

        def html_source_factory():
            return iter_html_from_batch(args.batch, dict_bytes, limit)
    else:
        default_batch = args.batches_dir / "web100k-batch-001.tar.zst"
        print(f"Will stream HTML files from {default_batch}...")

        def html_source_factory():
            return iter_html_from_batch(default_batch, dict_bytes, limit)

    # Helper: run a benchmark with optional memory measurement
    def run_with_memory(bench_fn, html_source_factory, iterations):
        # Use isolated process runner
        return run_benchmark_isolated(bench_fn, html_source_factory(), iterations, args=args)

    # Run benchmarks
    results = {}
    benchmarks = {
        "justhtml": benchmark_justhtml,
        "justhtml_to_html": benchmark_justhtml_to_html,
        "html5lib": benchmark_html5lib,
        "lxml": benchmark_lxml,
        "bs4": benchmark_bs4,
        "html.parser": benchmark_html_parser,
        "selectolax": benchmark_selectolax,
        "gumbo": benchmark_gumbo,
        "markupever": benchmark_markupever,
    }

    file_count = 0
    total_bytes = 0
    for parser_name in args.parsers:
        print(f"\nBenchmarking {parser_name}...", end="", flush=True)
        res = run_with_memory(benchmarks[parser_name], html_source_factory, args.iterations)
        results[parser_name] = res
        if "error" in res:
            print(f" SKIPPED ({res['error']})")
        else:
            print(f" DONE ({res['total_time']:.3f}s)")
            # Track file count and bytes from first successful benchmark
            if file_count == 0:
                file_count = res.get("file_count", 0)
                total_bytes = res.get("total_bytes", 0)

    if file_count > 0:
        print(f"\nProcessed {file_count} HTML files ({total_bytes / 1024 / 1024:.2f} MB)")

    # Print results
    print_results(results, file_count, args.iterations)


if __name__ == "__main__":
    main()
