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

`html5-parser` (the Gumbo benchmark) and `lxml` must use the same libxml2
implementation. The `html5-parser` wheel uses the system libxml2, while a
current `lxml` wheel embeds a newer one. Remove that wheel and rebuild only
`lxml` against the local library:

```bash
pip uninstall -y lxml
pip install --no-cache-dir --no-binary=lxml 'lxml==6.1.1'
```

Confirm that they agree before running the benchmark:

```bash
python -c 'from lxml import etree; import html5_parser; print(etree.LIBXML_VERSION)'
```

If pip cannot build lxml, install your platform's libxml2 development package
and C compiler, then repeat the second command. Reinstalling the project extra
after this step is unnecessary and can reselect the binary lxml wheel.

The latest MarkupEver and TurboHTML wheels do not currently load on Python
3.15 pre-releases. On that interpreter they are reported as unavailable while
the remaining benchmark parsers, including Gumbo, still run. Use a supported
stable Python release when those comparisons are required.

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

## Check adversarial scaling

Absolute throughput benchmarks can hide complexity regressions. For parser,
sanitizer, DOM, and serializer changes, also measure structurally hostile input
at several sizes and compare the growth per doubling. Linear work should take
about twice as long when the input doubles; growth approaching four times
usually indicates a repeated depth or sibling scan.

Focused tests can use `tests.harness.scaling.assert_scales_linearly`:

```python
assert_scales_linearly(
    lambda size: "<span>" * size + "x",
    lambda source: JustHTML(source, sanitize=False),
)
```

Pair a hostile shape with a shallow or well-formed control where practical.
Keep error collection and sanitization settings identical between the shape
being investigated and its control, because either pipeline can expose a
different repeated walk.

## Make a speed improvement

- Preserve parser and sanitizer semantics. Run the parser differential suite
  before relying on a throughput result.
- Optimize work on the normal path: token scanning, tree construction,
  attribute projection, and common serialization cases.
- Prefer direct local data access, reused compiled plans, and fewer temporary
  allocations in per-character and per-node loops.
- For deep parser state, prefer adaptive indexes that activate only after a
  fixed depth. This keeps ordinary shallow documents on the simple list path
  while bounding lookups for hostile nesting.
- Treat indexes and caches as derived state. Keep the list or tree authoritative,
  update derived structures incrementally for every mutation shape, and test
  threshold transitions plus mid-structure insertions, replacements, moves,
  and truncations. Rebuilding an entire index inside a mutation can restore
  correctness while reintroducing quadratic behavior.
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
