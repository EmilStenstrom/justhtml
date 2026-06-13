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
- The tokenizer has a default-path callback for simple `<tag>` and `</tag>`
  events, avoiding tag-buffer assembly for those events when error/location
  tracking is disabled.
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
- Current prototype median: `1.018393s` to `1.035433s` across repeated local
  runs after the simple-tag callback and attr fast path.
- Speedup: roughly `1.04x` to `1.05x`.
- Required continuation threshold: `1.7x`.
- Required final target: `2.0x`.

## Current Conclusion

The event-time fused sanitizer is much closer to the requested architecture
than the previous post-parse shortcut and passes the full non-coverage test
harness. It still does not materially improve speed because tokenizer and
treebuilder dispatch remain the dominant costs. Attempts to inline common
`IN_BODY` insertion-mode handling inside the fused sink made the benchmark
slower, so the existing `TreeBuilder.process_token` hot path is already close
to the practical limit for this shape.

A real `2x` pure-Python result likely requires a larger rewrite than a fused
sink: the tokenizer and insertion-mode state machines would need to be
co-designed around fewer Python calls and fewer object/list/dict mutations in
the hot path.
