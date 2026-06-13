# Fused Engine Findings

This branch has tested two increasingly aggressive approaches to default-safe
parsing:

1. `FusedDefaultTreeBuilder`, which keeps the existing tokenizer and
   treebuilder but moves default sanitizer work into tree construction.
2. `DefaultSafeEngine`, a proof-of-concept one-pass parser for the narrow
   `JustHTML(str)` default-safe path. This path does not use the existing
   tokenizer or treebuilder.

## Fused Treebuilder Result

- Unsafe comments, doctypes, URL/style attributes, invisible Unicode, and
  dropped-content containers such as `script`/`style` are handled as token
  events arrive.
- Foreign roots such as `svg`/`math` are skipped as dropped subtrees before DOM
  insertion for the default policy.
- Ordinary disallowed tags still enter tree construction so parser-sensitive
  elements such as `template` preserve HTML5 behavior; the same fused builder
  unwraps those recorded nodes at finish.
- The tokenizer has a default-path callback for simple `<tag>` and `</tag>`
  events, avoiding tag-buffer assembly for those events when error/location
  tracking is disabled.
- Custom policies and explicit transform pipelines still use the generic path.
- Benchmark result: roughly `1.04x` to `1.05x` over the recorded baseline.

This showed that removing the final generic sanitizer pass is not enough.
Tokenizer state dispatch, token construction, and treebuilder insertion-mode
dispatch remain the dominant cost.

## DefaultSafeEngine PoC

`DefaultSafeEngine` is intentionally narrower and less compatible. It is routed
only for plain-string `JustHTML(html)` with default sanitization, no custom
policy, no explicit transforms, no strict/error collection, no location
tracking, and no iframe `srcdoc` mode.

The engine combines scanning, DOM construction, and default sanitizer decisions
in a single loop:

- Comments are dropped while scanning; document doctypes are preserved because
  `DEFAULT_DOCUMENT_POLICY` allows them.
- `script`/`style` raw text and `svg`/`math` subtrees are skipped before DOM
  insertion.
- Default tag and attribute allowlists are applied before node creation.
- URL sink attributes use the existing URL sanitizer helpers.
- Text and attribute entity decoding happens inline.
- Full-document mode creates the default `html/head/body` shell up front.
- Parser-sensitive structure is handled directly for the common recovery cases
  listed below.

## HTML Recovery Pass

A second PoC pass added direct recovery for common malformed real-world HTML:

- Lightweight doctype parsing for `html`, `PUBLIC`, and `SYSTEM` doctypes.
- Leading pre-document whitespace suppression while preserving whitespace
  inside `head` and after an explicit `body`.
- `head` to `body` transition when body content starts inside an explicit
  `head`.
- RCDATA/rawtext-as-text handling for `title`, `textarea`, and `noscript`.
- Implicit closing for `p`, `li`, `dd`/`dt`, `option`/`optgroup`, headings,
  and nested `a` tags.
- Table repair for implicit `tbody`, `tr`, adjacent cells, and foster-parented
  non-table text/content.

Focused malformed samples for those cases now match the existing parser's
sanitized serialization.

## EnginePlan Split

A third pass introduced an explicit `EnginePlan`/executor split. The default
document and fragment plans are compiled once and cached, then `JustHTML` passes
the selected plan into `DefaultSafeEngine`.

The current plan contains the policy-derived sanitizer data and the recovery
sets the executor needs in the hot path:

- Allowed tags and per-tag/global attribute allowlists.
- URL policy and URL sink rules.
- Doctype/comment/drop-content behavior.
- RCDATA/rawtext-as-text handling sets.
- Head/body, implied-end-tag, and table-recovery sets.
- Invisible-Unicode stripping and void-element behavior.

The executor copies plan fields into slots during construction. This keeps the
planner boundary explicit without adding `self._plan...` lookups to every hot
path branch.

## Compliance Pass

A fourth pass targeted the largest remaining differential buckets:

- After-head whitespace is now inserted before the pre-created `body` element,
  matching the existing treebuilder's `AFTER_HEAD` behavior.
- Text and rawtext-as-text paths normalize CR/CRLF line endings to LF.
- `title` is only forced into `head` before body content starts; body-time
  `title` elements stay in body content.
- `li` and `dd`/`dt` implicit closes now stop at their list/definition
  boundaries instead of closing outer ancestors.
- Ignored initial LF handling was added for `pre`/`listing`.
- Table section/row/cell starts now require the right table context, so orphan
  `td`/`tr` tags are ignored like the treebuilder.
- Nested-table row and cell repair is boundary-aware, avoiding accidental
  closure of outer table rows/cells.

Focused probes for after-head whitespace, nested lists, body-time `title`,
`pre` initial LF, orphan table cells, adjacent table cells, and nested tables
now match the existing parser.

## html5lib Differential Scorecard

A reusable differential runner now targets the html5lib tree-construction
fixtures directly:

```bash
PYTHONPATH=src python benchmarks/html5lib_engine_diff.py \
  --examples 5 \
  --worst-files 12
```

This compares `JustHTML(html)` through `DefaultSafeEngine` against the current
tokenizer/treebuilder path forced with `JustHTML(html, collect_errors=True)`.
It is not the upstream html5lib pass/fail result because default-safe
sanitization changes observable output. The official tree harness still passes
against the existing parser path because it runs with `sanitize=False`.

Current result:

- Eligible full-document cases: `1564`.
- Exact matches: `942`.
- Mismatches: `620`.
- New-engine-only exceptions: `0`.
- Reference-path exceptions: `2` malformed-doctype serializations.
- Exact/eligible rate: `60.23%`.
- Exact/compared rate: `60.31%`.
- Skipped unsupported modes: `192` fragment-context cases and `35`
  scripting-directive cases.

The largest remaining buckets are adoption-agency/active-formatting behavior,
`plaintext`, select-like insertion modes, deeper table/template cases, and
quirks around malformed inline structure. Malformed doctype names are now
normalized in `DefaultSafeEngine` so the new engine does not produce unsafe
names that later fail serialization.

## Benchmark Result

Command:

```bash
PYTHONPATH=src python benchmarks/fused_engine_gate.py \
  --iterations 5 \
  --limit 100 \
  --baseline-seconds 1.073689 \
  --fail-under-speedup 1.7
```

Result:

- Baseline median: `1.073689s` for 100 `web100k` files.
- Raw `DefaultSafeEngine` median before recovery: `0.382541s`.
- Recovery-enabled median before `EnginePlan`: `0.471169s`.
- EnginePlan-enabled median: `0.467711s`.
- Compliance-pass median: `0.470205s`.
- Compliance-pass speedup: `2.283x`.
- html5lib-scorecard pass median: `0.469793s`.
- html5lib-scorecard pass speedup: `2.285x`.
- Required continuation threshold: `1.7x`.
- Required final target: `2.0x`.

The 2x target is therefore feasible in pure Python, but only with a specialized
default-safe engine that collapses tokenizer, tree construction, and sanitizer
work into one hot path.

## Parity Status

The PoC is not production-compatible yet. After the compliance pass, a 100-file
differential smoke check completed without crashes and `75/100` serialized
outputs exactly matched the existing parser. This is up from `2/100` for the
raw one-pass parser and `20/100` after the first recovery pass.

The broader html5lib differential scorecard is now the main compliance driver:
`942/1564` eligible full-document cases match the existing default-safe path,
and the new engine has no current-only serialization exceptions in that suite.

Remaining diffs are now more varied: exact whitespace counts, broader
formatting-element/adoption behavior, malformed inline links, deeper table
corner cases, select/template handling, foreign-content integration points, and
some sanitizer edge cases around unusual attributes or escaped source.

## Current Conclusion

The viable path is not a lightly fused version of the existing html5ever-shaped
pipeline. It is a new default-safe parser with its own small set of direct
handlers, then incremental parity work driven by differential fixtures.

The recovery pass is the strongest signal so far that this can remain viable:
adding a meaningful subset of real HTML error handling moved the benchmark from
`2.807x` to `2.283x`, still above the final `2.0x` target. The first
`EnginePlan` split did not add measurable overhead, and the next compliance pass
raised exact corpus matches from `20/100` to `75/100` while keeping most of the
speed margin.

The next engineering step is to keep `DefaultSafeEngine` as the PoC target and
fill in only the HTML5 behaviors that materially affect sanitized output for
real-world default-safe parsing. The remaining speed budget is approximately
`0.067s` on this 100-file benchmark before falling below `2x`.
