# Project Ideas

## Treebuilder Refactors

### Open element stack helper

Replace direct `open_elements` list mutation with an `OpenElementStack` helper. The helper would own push, pop, remove, replace, and truncate operations, and it would keep cached counters for common stack queries such as open `<p>` elements, template presence, and scope-relevant names.

Expected value: high maintainability, safer future performance caches, and fewer hidden invariants around `_note_open_element_pushed()` and `_note_open_element_removed()`.

Performance requirement: no meaningful regression on the general `web100k` benchmark.

### Active formatting list helper

Move active formatting marker and entry operations into an `ActiveFormattingList` helper. It would own marker handling, duplicate detection, lookup by name, lookup by node, removal, and clear-to-marker behavior.

Expected value: lower adoption-agency complexity and safer future optimizations around active formatting reconstruction.

### Declarative insertion-mode dispatch

Generate or centralize start/end tag dispatch tables for major insertion modes, especially `IN_BODY`, table modes, and select mode. Keep fallback behavior explicit so tag priority remains easy to audit.

Expected value: possible hot-path speedup and clearer mode behavior. Needs before/after benchmark validation because dispatch overhead may vary.

### Split insertion modes into smaller modules

Break `modes.py` into mode-group modules such as head, body, table, select, template, and foreign-content handling while keeping the public `TreeBuilder` API unchanged.

Expected value: large maintainability improvement and easier auditing. Performance should be neutral if done mechanically.

### Stack index helpers

Centralize repeated reverse stack scans into helpers such as `_find_open_element_index()`, `_find_last_open_element_index()`, and `_truncate_open_elements_from()`. Add caching only where benchmarks prove it helps.

Expected value: clearer code first, optional targeted performance improvements later.

### Separate parsing decisions from DOM mutation

Create a tighter mutation API for element insertion, text insertion, foster parenting, child movement, and stack popping. Keep mode handlers focused on parser decisions.

Expected value: maintainability and auditability. Runtime effect must be measured because extra calls could slow hot paths.

### Narrow internal node types

Replace some `Any` usage around stack nodes and active formatting entries with protocols or narrower internal types.

Expected value: safer refactors and better mypy coverage. Runtime effect should be neutral.
