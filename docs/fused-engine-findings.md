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

- Comments and doctypes are dropped while scanning.
- `script`/`style` raw text and `svg`/`math` subtrees are skipped before DOM
  insertion.
- Default tag and attribute allowlists are applied before node creation.
- URL sink attributes use the existing URL sanitizer helpers.
- Text and attribute entity decoding happens inline.
- Full-document mode creates the default `html/head/body` shell up front.
- A small amount of parser-sensitive structure is handled directly, currently
  including `title` placement and basic `table > tr` `tbody` insertion.

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
- Current `DefaultSafeEngine` median: `0.382541s` for 100 `web100k` files.
- Speedup: `2.807x`.
- Required continuation threshold: `1.7x`.
- Required final target: `2.0x`.

The 2x target is therefore feasible in pure Python, but only with a specialized
default-safe engine that collapses tokenizer, tree construction, and sanitizer
work into one hot path.

## Parity Status

The PoC is not production-compatible yet. A 100-file differential smoke check
completed without crashes, but only `2/100` serialized outputs exactly matched
the existing parser. The largest gaps are expected HTML5 treebuilder behavior:
doctype handling, foster parenting, formatting-element reconstruction, richer
table insertion modes, rawtext/RCDATA corner cases, foreign-content integration
points, and malformed-markup recovery.

## Current Conclusion

The viable path is not a lightly fused version of the existing html5ever-shaped
pipeline. It is a new default-safe parser with its own small set of direct
handlers, then incremental parity work driven by differential fixtures.

The next engineering step is to keep `DefaultSafeEngine` as the PoC target and
fill in only the HTML5 behaviors that materially affect sanitized output for
real-world default-safe parsing. If those parity additions keep the median under
roughly `0.54s` on this benchmark, the branch can still clear the final 2x
target with margin.
