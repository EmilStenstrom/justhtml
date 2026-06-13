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
- Recovery-enabled `DefaultSafeEngine` median: `0.471169s`.
- Recovery-enabled speedup: `2.279x`.
- Required continuation threshold: `1.7x`.
- Required final target: `2.0x`.

The 2x target is therefore feasible in pure Python, but only with a specialized
default-safe engine that collapses tokenizer, tree construction, and sanitizer
work into one hot path.

## Parity Status

The PoC is not production-compatible yet. After the recovery pass, a 100-file
differential smoke check completed without crashes and `20/100` serialized
outputs exactly matched the existing parser. This is up from `2/100` for the
raw one-pass parser.

Remaining early diffs are now often whitespace and head/body placement details,
plus deeper HTML5 behavior such as formatting-element reconstruction, richer
table insertion modes, select/template handling, foreign-content integration
points, and malformed-markup recovery outside the common cases above.

## Current Conclusion

The viable path is not a lightly fused version of the existing html5ever-shaped
pipeline. It is a new default-safe parser with its own small set of direct
handlers, then incremental parity work driven by differential fixtures.

The recovery pass is the strongest signal so far that this can remain viable:
adding a meaningful subset of real HTML error handling moved the benchmark from
`2.807x` to `2.279x`, still above the final `2.0x` target.

The next engineering step is to keep `DefaultSafeEngine` as the PoC target and
fill in only the HTML5 behaviors that materially affect sanitized output for
real-world default-safe parsing. The remaining speed budget is approximately
`0.066s` on this 100-file benchmark before falling below `2x`.
