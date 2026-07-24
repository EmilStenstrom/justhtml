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

For parser availability work, run the focused hostile-input benchmark as well:

```bash
python benchmarks/parser_adversarial.py --sizes 1000 2000 4000
```

It warms each shape once and reports the median of repeated runs. Use the same
Python version, options, sizes, and repeat count for before/after comparisons.

## Guard against complexity regressions

`parser_adversarial.py` reports absolute timings. When the change is about
*complexity* rather than throughput, use the scaling guard instead:

```bash
python benchmarks/scaling_guard.py --save before.json
# ... implement the change ...
python benchmarks/scaling_guard.py --compare before.json
```

It measures each shape at several input sizes, fits an exponent `k` such that
`time ~ n**k`, and reports it next to the complexity the shape is expected to
have. Every adversarial shape is paired with a control that exercises the same
code path without the pathological property, so a fix has to remove the
quadratic term rather than slow the control down to match. `--compare` prints
the per-shape speedup at the largest shared size and exits non-zero when a shape
gets materially slower or grows a worse exponent, which makes it usable as a
gate.

Measurements are auto-ranged (fast shapes repeat until a sample lasts at least
`--min-sample` seconds) and taken with the garbage collector disabled, then
reported as the minimum across `--repeats` samples. Shapes whose spread stays
above 15% are listed separately; raise `--repeats` or quiet the machine before
trusting those rows.

Useful filters: `--controls-only` for a fast regression check on well-formed
input, `--issues N` to focus one problem, `--only SUBSTRING` for one shape, and
`--skip-limits` to drop the recursion-depth probes.

Scaling behavior and real-world throughput are different questions. Confirm both
before landing a complexity fix, since indexing work moves cost onto the normal
path:

```bash
python benchmarks/performance.py --parsers justhtml --limit 1000 --iterations 1
```

## Open-element scope lookups

`_CountingStack` answers three questions the parser asks constantly: where the
last element with a given name sits, whether that position is above or below a
scope boundary, and whether a particular node is on the stack at all. All three
are answered from positions maintained as the stack mutates, not from a walk:

- `_html_positions` / `_other_positions` — ascending indices per tag name, split
  by namespace because scope boundaries only count for HTML elements while the
  target may be in any namespace.
- `_html_all` / `_foreign_all` / `_parser_only_all` — ascending indices per
  namespace group, for the innermost element of a kind.
- `_boundaries` — ascending indices of foreign integration points, classified
  once on the way in by `_is_open_foreign_boundary()`.
- `_index_of` — node identity to index, for membership and position queries.

A scope check is then a comparison: take the target's position, take the nearest
boundary's, and prefer the target on a tie, because the walk these replace tests
the target name before the boundary names. Getting that tie wrong silently
changes parse results, so `_find_open_index_before_boundary()`,
`_find_open_special_end_index()`, `_find_open_heading_index()`,
`_close_until_before_boundary()`, `_close_open_li_for_start()`,
`_find_open_table_scoped_end_index()`, `_has_node_in_scope()` and
`_end_tag_stays_in_foreign_context()` all use `>=`.

The index is only built once the stack reaches `_STACK_INDEX_THRESHOLD`. Below
it a reverse walk is bounded by that constant, so it cannot be quadratic, and
the stack keeps nothing at all — the parser pushes with `list.append` directly.
Roughly one document in three hundred of the web100k corpus ever crosses the
threshold, so ordinary parsing pays nothing for what only hostile nesting needs.

That split means several helpers carry both a bounded single-pass walk and an
indexed comparison. The two must agree exactly, or a document would parse
differently once its stack grew past the threshold.
`TestCountingStack.test_indexed_and_scanned_lookups_agree_on_varied_stacks` and
`test_engine_scope_helpers_agree_across_indexed_and_scanned_stacks` check that
correspondence directly; extend them when adding a lookup.

`stream.py` keeps the same shape in miniature: `_name_positions` and
`_html_positions` bound its end-tag walk, which otherwise rescans the whole
foreign suffix per token.

Reverse loops in `_find_open_index()`, `_find_open_html_index()`,
`_last_open_index_of_any()`, template lookup, table-scope lookup, and
current-template-scope lookup are also compatibility fallbacks for tests that
replace the private `_CountingStack` with a plain list.

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
