# Fused Engine Findings

This branch tested the first pure-Python step toward a fused parse engine:
replacing the generic post-parse default `Sanitize(...)` transform walk with a
specialized default-policy sanitizer.

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
- Prototype median: `0.942781s`.
- Speedup: `1.139x`.
- Required continuation threshold: `1.7x`.
- Required final target: `2.0x`.

## Bottleneck

After the prototype, profiling still shows the runtime dominated by the
existing tokenizer and treebuilder:

- Tokenizer: about `31.6%` of profile tottime.
- Treebuilder: about `30.6%`.
- Specialized sanitizer: about `14.6%` cumulative.

The specialized sanitizer is useful, but it is not enough to justify continuing
the rewrite under the original pure-Python `2x` target. A true `2x` speedup
would require eliminating most tokenizer/treebuilder dispatch overhead, not
only fusing sanitizer work.

## Decision

Stop this rewrite at the prototype checkpoint. The branch keeps the benchmark
gate and the behavior-preserving default sanitizer fast path, but does not
continue into a full tokenizer/treebuilder rewrite because the measured
prototype missed the plan's `1.7x` continuation threshold.
