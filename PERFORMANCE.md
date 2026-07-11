# Performance Guide

This guide is for improving JustHTML's parsing speed on real-world HTML while
preserving its HTML5 recovery and sanitization behavior.

## Benchmark data and setup

The main benchmark uses the [web100k](https://github.com/EmilStenstrom/web100k)
corpus. By default it expects the dataset beside this repository:

```text
../web100k/
├── html.dict
└── batches/
    └── web100k-batch-001.tar.zst
```

Set `WEB100K_DIR` to use another location, or pass `--batches-dir` and
`--dict` explicitly. Install the benchmark dependencies first:

```bash
pip install -e ".[benchmark]"
```

## Measure the right pipeline

The benchmark has separate modes for parsing alone and parsing followed by
serialization. Start with the narrowest mode that covers the change:

```bash
# Parser throughput on 100 real-world documents, without RSS sampling noise.
python benchmarks/performance.py --parsers justhtml --iterations 5

# Parse plus default HTML serialization.
python benchmarks/performance.py --parsers justhtml_to_html --iterations 5

# Compare selected installed parsers on a larger corpus sample.
python benchmarks/performance.py --parsers justhtml html5lib lxml --limit 1000 --iterations 3
```

Use `--all-batches` for a corpus-wide measurement, `--batch PATH` for one
archive, or `--downloaded DIR` for a directory of decompressed-source files.
The benchmark reports total throughput and per-document timing. Compare the
same command before and after a change on the same machine; do not compare
absolute timings across machines.

## Find the hot path

Profile a representative web100k batch before optimizing:

```bash
python benchmarks/profile.py --mode parse
```

Available modes are `parse`, `fragment`, `compact-html`, `pretty-html`, and
`text`. Use the mode closest to the public operation you are improving. Focus
on functions that dominate cumulative work across many documents, not an
isolated slow malformed sample.

## Make a speed improvement

- Preserve parser and sanitizer semantics. Run the parser differential suite
  before relying on a throughput result.
- Optimize work on the normal path: token scanning, tree construction,
  attribute projection, and common serialization cases.
- Prefer direct local data access, reused compiled plans, and fewer temporary
  allocations in per-character and per-node loops.
- Measure realistic HTML before and after the change. Include a focused
  microbenchmark only when it explains the real-world result.
- Keep benchmarks honest: warm up once, run multiple iterations, and report
  the exact command, corpus size, and pipeline mode in the pull request.

## Validate and document

Add correctness tests for behavior affected by an optimization. Run the full
project gate before submitting:

```bash
pre-commit run --all-files
```

Update `CHANGELOG.md` with a `Performance` entry that states the affected
pipeline and the avoided work. If the change also prevents disproportionate
resource use from hostile input, add a `Security` entry as well.
