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
have. Each adversarial family carries a control that exercises the same code path
without the pathological property, so a fix has to remove the quadratic term
rather than slow the control down to match. The five shapes tagged `prior` are the
exception: they were fixed before the harness existed and are kept only so a
regression is caught, so they share the controls of the families they belong to
rather than carrying their own. `--compare` prints the per-shape speedup at the
largest shared size and exits non-zero when a shape gets materially slower or
grows a worse exponent, which makes it usable as a gate.

A control has to be checked against the unfixed tree, not just written. A shape
that looks like a control can turn out to exercise a second defect and be
quadratic itself — `"<div>" * n` under `sanitize=False` is quadratic on `main`
for reasons unrelated to whatever it was meant to isolate, which makes it useless
as a baseline.

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
About one document in thirty of the web100k corpus crosses the threshold: 310 of
batch 001's 10,000 documents build an index, with a median peak stack depth of 32
and a maximum of 52.

Maintaining the index is nonetheless not where this work costs throughput.
Raising `_STACK_INDEX_THRESHOLD` so that no document ever indexes leaves corpus
parse time unchanged, within run-to-run noise. What the corpus pays for is the
*shallow* path — the code every document runs — so tune that and measure the
index separately:

- On well-formed markup deep enough to cross the threshold, indexing costs a flat
  step of roughly 20-25%, not a slope: at a constant element count and depths of
  16, 32, 256 and 4,000, `main` takes 3.00, 3.30, 2.98 and 3.21 ms against 3.00,
  3.94, 3.79 and 4.05 ms here. Below the threshold there is no difference.
- On the corpus the same shapes are a rounding error, because almost nothing
  reaches that depth.

The lesson generalizes: when a fix adds a fast path *and* a data structure,
attribute the cost by ablation before writing it down. A profiler will hand you
the wrong answer here — `cProfile`'s per-call overhead makes any newly introduced
helper look expensive in proportion to how often it is called.

That split means several helpers carry both a bounded single-pass walk and an
indexed comparison. The two must agree exactly, or a document would parse
differently once its stack grew past the threshold.
`TestCountingStack.test_indexed_and_scanned_lookups_agree_on_varied_stacks` and
`test_engine_scope_helpers_agree_across_indexed_and_scanned_stacks` check that
correspondence directly; extend them when adding a lookup.

`stream.py` keeps the same shape in miniature: `_name_positions` and
`_html_positions` bound its end-tag walk, which otherwise rescans the whole
foreign suffix per token.

### Mutations below the top of the stack

Pushes and pops maintain every list above in constant time, including the suffix
truncation that closes a run of open elements. A mutation in the middle is
different: the positions above it all shift by one, and rebuilding the index to
account for that costs the whole depth of the stack.

That case is not rare. The adoption agency reparents a misnested formatting
element by removing it from the stack and re-pushing its clone above the furthest
block, and the element it removes sits one slot below the top — over
`"<a><div>" * n`, every one of the n removals has exactly one node above it. One
rebuild per token against a stack that grows by two per token is a quadratic term
with no constant left to tune: at n=4,000 those rebuilds re-indexed 8,009,570
nodes, or exactly n²/2.

Only the nodes that moved need new numbers, so `_renumber_from()` discards their
recorded positions and re-notes them one slot along, at the cost of a binary
search per list. The work is then proportional to the distance from the mutation
to the top of the stack rather than to the depth below it, which is what makes the
shape linear rather than merely cheaper: `"<a><div>" * 8000` took 1.58 s on `main`
and takes 47 ms here.

`remove()` is the other half of the same call, and it was still quadratic once the
rebuilds were gone: `list.remove` finds its argument by scanning up from index 0,
which is the entire depth. It reads the position from `_index_of` instead and
raises `ValueError` itself when the node is not open.

### The name that carries the shallow path

`count_of()` is constant time for `p` and for any name on an indexed stack, and a
threshold-bounded walk otherwise. That asymmetry is load-bearing rather than
incidental: `p` is the target of about two thirds of all scope checks — 30,756 of
47,640 calls to `_find_open_index_before_boundary()` over 400 corpus documents,
against 13,151 for `tr` and a long tail below 2% each. Answering those from
`_p_count` keeps them off the stack entirely, and `_find_open_index_before_boundary()`
guards itself with `count_of()` for exactly that reason.

Dropping the `if name == "p"` branch from `count_of()`, or the `count_of()` guard
from the scope check, puts every one of those calls back on a walk and costs
about a point of corpus parse throughput. Neither reads like a hot path; both are.

Per-name counts for *every* name are not the answer — maintaining them on each
push and pop costs more than the walks they save, measured at roughly twice the
throughput the guard recovers.

## Unwrapping disallowed elements

Sanitization keeps the children of an element it removes, so every disallowed
element is recorded and unwrapped after the parse. Splicing one node's children
into its parent at a time is quadratic when those nodes nest: each splice carries
everything that accumulated below it up one more level, so `"<section>x" * n`
moves n²/2 children. `_unwrap_recorded_nodes()` therefore expands a whole chain in
one walk and rebuilds each surviving parent once, which makes every child move
exactly once whatever order the nodes were recorded in.

Two properties are worth keeping if this is rewritten. It must not recurse — the
chain is as deep as the input nests — and it must not depend on a parent being
recorded before its children, because the adoption agency can clone a formatting
element into an ancestor position after its descendants were recorded.

This one costs rather than saves on ordinary documents, so it is measured by
ablation like the stack index. Real pages nest disallowed elements constantly:
over batch 001's first 1,000 documents, 254 unwrap batches cross
`_UNWRAP_BATCH_THRESHOLD` and 15,417 of the 40,125 nodes in them sit inside
another recorded node. Chains that shallow are cheap to re-move, so the
bookkeeping the general case needs — one set of the recorded nodes and one pass
over them — is a net loss there: the phase takes 19.6 ms against 15.4 ms, or
about 0.15% of corpus parse time, for removing a quadratic that a 20-byte
repetition can reach. Variants that avoid the pass were measured and lost: deciding the
grouping from the recorded order instead of per parent costs 20.3 ms, building
the same bookkeeping with C-level set and dict operations costs 23.3 ms, and
merely reversing the order the nodes are visited in costs 20.0 ms.

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
