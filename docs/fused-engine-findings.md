# Fused Engine Findings

This branch now tests an event-time default sanitizer built into the parser
sink. The default `JustHTML(html)` path uses `FusedDefaultTreeBuilder`, which
receives tokenizer events, sanitizes them before normal tree insertion where it
can, and skips the final generic `Sanitize(...)` transform.

## What Is Fused

- Unsafe comments, doctypes, URL/style attributes, invisible Unicode, and
  dropped-content containers such as `script`/`style` are handled as token
  events arrive.
- Foreign roots such as `svg`/`math` are skipped as dropped subtrees before DOM
  insertion for the default policy.
- Ordinary disallowed tags still enter tree construction so parser-sensitive
  elements such as `template` preserve HTML5 behavior; the same fused builder
  unwraps those recorded nodes at finish.
- Custom policies and explicit transform pipelines still use the generic path.

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
- Current prototype median: `1.031969s`.
- Speedup: `1.040x`.
- Required continuation threshold: `1.7x`.
- Required final target: `2.0x`.

## Current Conclusion

The event-time fused sanitizer is much closer to the requested architecture
than the previous post-parse shortcut and passes the full non-coverage test
harness. It still does not materially improve speed because tokenizer and
treebuilder dispatch remain the dominant costs, and preserving HTML5 behavior
for parser-sensitive disallowed wrappers requires some finish-time unwrapping.

A real `2x` pure-Python result likely requires collapsing tokenizer state
handlers and insertion-mode handlers into fewer direct hot-path functions, not
only moving sanitizer work into the parser sink.
