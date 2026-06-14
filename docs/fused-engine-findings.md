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

`FusedDefaultTreeBuilder` has now been deleted. It was useful as a benchmark
probe, but keeping it would leave three construction paths: the new engine, the
fused tokenizer/treebuilder path, and the legacy tokenizer/treebuilder path.
The release direction is one production parser path, so the intermediate fused
builder should not survive.

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
- Per-tag `TagAction` records for parser categories, output attrs,
  parser-state-only attrs, and URL attr rules.
- Doctype/comment/drop-content behavior.
- RCDATA/rawtext-as-text handling sets.
- Head/body, implied-end-tag, and table-recovery sets.
- Invisible-Unicode stripping and void-element behavior.

The executor copies plan fields into slots during construction. This keeps the
planner boundary explicit without adding `self._plan...` lookups to every hot
path branch. The latest pass also moved attr filtering into the scanner: start
tags now resolve one `TagAction`, then parse only attrs the plan can output or
needs for parser state. The old raw-attrs dict plus second sanitizer pass is no
longer in the default-engine hot path. A follow-up pass pushed projection
further down by skipping discarded attr values without slicing or decoding them,
then added direct end-tag scanning and a stack-top fast close for ordinary
matched end tags.

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

After deleting `FusedDefaultTreeBuilder`, this runner's forced reference path is
no longer a stable snapshot of the previous construction-sanitized behavior:
`collect_errors=True` now necessarily uses the generic tokenizer/treebuilder
plus post-parse sanitizer path. Further deletion work should either freeze
expected serialized outputs or teach the harness an explicit expected-output
mode instead of treating the remaining legacy parser as the reference.

Current result:

- Scored html5lib tree cases: `1783`.
- Exact behavior matches: `1783`.
- Serialized output mismatches: `0`.
- New-engine-only exceptions: `0`.
- Unmatched reference-path exceptions: `0`.
- Matching exceptions: `0`.
- Exact/scored rate: `100.00%`.
- Exact/compared rate: `100.00%`.
- Excluded unsupported modes: `8` `script-on` cases that require JavaScript
  execution semantics.

Incremental progress in this pass:

- Start: `942/1783` scored (`52.83%`).
- Dropped-to-EOF shell cleanup: `1113/1783` scored (`62.42%`),
  `2.301x` speedup.
- Null, plaintext, and `pre`/`listing` newline handling: `1142/1783`
  scored (`64.05%`), `2.198x` speedup.
- Frameset/`noframes` shell behavior: `1210/1783` scored (`67.86%`),
  `2.210x` speedup.
- Bogus markup, `</br>`, and `<image>` recovery: `1228/1783` scored
  (`68.87%`), `2.196x` speedup.
- Active formatting/adoption agency for default-safe kept formatting tags:
  `1275/1783` scored (`71.51%`), `2.067x` speedup.
- Production-safe generic recovery cleanup: `1308/1783` scored (`73.36%`),
  `2.053x` speedup.
- Document-mode state and parser-only button scope nodes: `1324/1783` scored
  (`74.26%`), `2.030x` speedup.
- Decoded `ignore_lf` state for `pre`, `listing`, and `textarea`:
  `1328/1783` scored (`74.48%`), `2.041x` speedup.
- Parser-only template scopes, template insertion modes, stricter rawtext end
  tags, script escaped-state scanning, and disallowed rawtext-as-text handling:
  `1461/1783` scored (`81.94%`), `1.986x` speedup.
- Fragment contexts and `script-off` cases included in the differential
  runner, all fragment contexts routed through `DefaultSafeEngine`, and
  script-disabled `noscript` behavior compiled into the plan:
  `1669/1783` scored (`93.61%`), `1.917x` speedup.
- Compiled `TagAction` records, direct start-tag scanning, and direct
  sanitized attr scanning remove the raw attr dict plus `_sanitize_attrs`
  handoff from the hot path while preserving the same score:
  `1669/1783` scored (`93.61%`), `2.131x` speedup.
- Attr projection pushdown and direct end-tag fast close keep the same score
  while increasing the speed margin:
  `1669/1783` scored (`93.61%`), `2.216x` speedup.
- Parser-mode/table-scope repair, unsafe unwrap-node construction for
  parser-sensitive disallowed tags, shared scope terminators, heading end-tag
  handling, body-time head-content placement, and skipped-menuitem active
  formatting reconstruction:
  `1710/1783` scored (`95.91%`), `2.071x` speedup.
- Text-path lazy whitespace classification, foster-parent fast checks,
  attr-scanner constant reuse, and stack-top mode checks recover much of the
  lost speed while preserving the compliance gains:
  `1711/1783` scored (`95.96%`), `2.174x` speedup.
- Incremental html5lib compliance pass for self-closing non-void tags,
  after-head head content, dropped rawtext whitespace, escaped-script EOF
  shell retention, selectedcontent projection, foreign-content subtree
  skipping, plaintext `p` closure, colgroup/table repairs, fragment nested
  tables, and table-scope end-tag boundaries:
  `1744/1783` scored (`97.81%`), `2.088x` speedup.
- Active-formatting markers, parser-only template adoption moves, template
  table-mode section transitions, table foster whitespace/reconstruction
  ordering, end-body stack retention, and start-tag hot-path cleanup:
  `1781/1783` scored (`99.89%`), `2.050x` speedup.
- Safe malformed-doctype serialization for both parser paths:
  `1783/1783` scored (`100.00%`), `2.021x` speedup.

The final two cases preserve the legacy tokenizer's malformed doctype names in
the tree, then omit those unsafe names during HTML serialization. Both parser
paths now emit safe `<!DOCTYPE>` output instead of raising.

## Benchmark Result

Command:

```bash
PYTHONPATH=src python benchmarks/fused_engine_gate.py \
  --iterations 9 \
  --limit 100 \
  --baseline-seconds 1.073689 \
  --fail-under-speedup 2.0
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
- Latest html5lib-targeting median: `0.540501s`.
- Latest html5lib-targeting speedup: `1.986x`.
- Expanded fragment/script-off median: `0.560081s`.
- Expanded fragment/script-off speedup: `1.917x`.
- Compiled tag/attr-action median: `0.503856s`.
- Compiled tag/attr-action speedup: `2.131x`.
- Attr projection/end-fast-close median: `0.484585s`.
- Attr projection/end-fast-close speedup: `2.216x`.
- Unsafe unwrap/scope compliance median: `0.518448s`.
- Unsafe unwrap/scope compliance speedup: `2.071x`.
- Hot-path recovery median: `0.493839s`.
- Hot-path recovery speedup: `2.174x`.
- Latest compliance median: `0.514311s`.
- Latest compliance speedup: `2.088x`.
- Frameset/table-rawtext compliance median: `0.516347s`.
- Frameset/table-rawtext compliance speedup: `2.079x`.
- Adoption/table-scope compliance median: `0.522134s`.
- Adoption/table-scope compliance speedup: `2.056x`.
- Template/foster/body-stack parity median: `0.523710s`.
- Template/foster/body-stack parity speedup: `2.050x`.
- Safe malformed-doctype serialization median: `0.531217s`.
- Safe malformed-doctype serialization speedup: `2.021x`.
- Fused treebuilder deletion/custom policy planner median: `0.529293s`.
- Fused treebuilder deletion/custom policy planner speedup: `2.029x`.
- Engine error pre-scan/srcdoc routing median: `0.533418s`.
- Engine error pre-scan/srcdoc routing speedup: `2.013x`.
- Required continuation threshold: `1.7x`.
- Required final target: `2.0x`.

The 2x target remains above the required gate, but the latest compliance work
spent part of the previous margin. The next productionization pass should keep
pairing new parser-state work with hot-path profiling. The remaining hot areas
are start-tag/attr dispatch, text append/cleaning, DOM insertion, URL
sanitization, and active-formatting reconstruction.

## Parity Status

The engine is not production-compatible as the only parser yet, mainly because
the remaining product surface extends beyond this differential scorecard:
custom policy features, explicit transforms, raw/trusted `sanitize=False`
parsing, location tracking, and broader application-level compatibility still
need promotion work.

The latest deletion pass removes the obsolete fused treebuilder path and starts
moving custom policies into the engine planner. `DefaultSafeEngine` can now
compile conservative safe-policy subsets: strip-mode unsafe handling,
unwrap-mode disallowed tags, dropped comments, no forced link-rel merge, no
allowed style/SVG/MathML surface, default-policy tag/attribute subsets, and URL
sink attrs only when the policy supplies explicit URL rules. Unsupported policy
features deliberately stay on the remaining legacy path until they have engine
semantics.

Default-safe `collect_errors`, `strict`, `debug`, and iframe `srcdoc` no longer
force the tokenizer/treebuilder path for compilable engine plans. Error
collection currently uses a separate lightweight pre-scan for null characters,
EOF-in-tag/comment cases, missing initial doctype, and obvious unexpected end
tags. That keeps diagnostic work out of the normal parse hot path while moving
the public constructor route onto `DefaultSafeEngine`; full html5lib parse-error
parity is still separate work.

The broader html5lib differential scorecard is now the main compliance driver:
`1783/1783` scored cases match the existing default-safe path, with `0`
serialized-output mismatches, `0` unmatched current exceptions, and `0`
unmatched reference exceptions in that suite. The two formerly exceptional
malformed-doctype cases now serialize safely on both parser paths. The excluded
tree-construction fixtures are the `script-on` cases that require JavaScript
execution semantics; `script-off` cases now exercise `DefaultSafeEngine`.

## Productionization Pivot

The next phase should stop chasing isolated fixture wins unless the change is a
general parser rule with a clear owner in the engine architecture. Two rules
should govern new work:

- Keep generic behavior that maps to tokenizer, insertion-mode, sanitizer, or
  serializer semantics.
- Avoid one-off fixture shortcuts; add explicit engine state instead, such as a
  compatibility-mode flag or script-data state machine, when the behavior needs
  real state.

The latest cleanup follows that rule. It keeps general fixes for block-end `p`
closure, malformed comment endings, late doctypes, tolerant quoted doctype IDs,
and rawtext end-tag boundaries. It deliberately does not include a narrow quirks
shortcut for table-in-`p` behavior or a partial script double-escaped heuristic;
those should be implemented later as proper compatibility-mode and script-data
state handling.

The next productionizing pass added that compatibility-mode state through the
same `doctype_error_and_quirks` helper used by `TreeBuilder`, then introduced
parser-only stack nodes for disallowed scoped elements. The first scoped element
is `button`: it affects button-scope decisions while DOM insertion skips the
parser-only node, matching the eventual sanitizer unwrap behavior without
creating unsafe output nodes.

The decoded `ignore_lf` state follows the same model: the engine now strips the
first decoded LF text token after `pre`, `listing`, and `textarea`, matching the
treebuilder's token-level behavior rather than skipping only a raw source
newline.

The latest compliance pass extends that architecture from parser-only stack
nodes to recorded unsafe unwrap nodes. Safe tags still go straight to the final
DOM, but parser-sensitive disallowed tags can temporarily participate in stack
scope, adoption-agency, and formatting reconstruction. The engine then unwraps
those recorded nodes at finish, preserving sanitized output while avoiding the
largest correctness loss from dropping disallowed tags before tree construction.
Column/form/void/head-only tags remain skipped in the specialized path where
retaining them would require insertion modes this engine has not implemented.

The frameset/table-rawtext pass keeps the same productionizing bias. Parser-only
`frame` tags now affect frameset parsing without producing sanitized output,
frameset text preservation follows the treebuilder rule of retaining whitespace
while ignoring non-whitespace, and dropped `script`/`style` content records when
the next table whitespace token must still be foster-parented. A short hot-path
cleanup kept that extra table whitespace check out of ordinary text appends,
preserving the `2x` speed gate.

The adoption/table-scope pass moved more treebuilder rules into explicit engine
state: ordinary adoption agency now checks default HTML scope before rewriting,
table cells clear only the active-formatting entries created inside the cell,
malformed section end tags no longer close cells unless their row/section is in
table scope, and generic end tags stop at special elements. Adoption-created
formatting clones also retain the existing parser's compatibility behavior for
disallowed formatting tags and their state-only attributes.

## Current Conclusion

The viable path is not a lightly fused version of the existing html5ever-shaped
pipeline. It is a new default-safe parser with its own small set of direct
handlers, then incremental parity work driven by differential fixtures.

The strongest signal so far is that the parser can now match every scored
html5lib tree-construction case in the default-safe differential runner while
still clearing the `2x` benchmark gate, with no scorecard exceptions.

The next engineering step is to keep `DefaultSafeEngine` as the productionizing
target: define state boundaries, add focused golden tests for each promoted
rule, preserve the benchmark gate above `2x`, and move the remaining public API
surface from fallback behavior onto explicit engine plans.
