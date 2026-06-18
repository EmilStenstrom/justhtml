## JustHTML – Agent instructions

# Decision & Clarification Policy (Overrides)

- Replace "propose a follow-up" with "propose **and execute** the best alternative by default; ask only for destructive/irreversible choices."
- Keep preambles to a single declarative sentence ("I'm scanning the repo and then drafting a minimal fix.") — no approval requests.

### Architecture Snapshot
- Parser engine (`src/justhtml/parser/engine.py`): Plan-driven tokenization, tree construction, sanitizer projection, and HTML5 recovery behavior.
- Parser scanner (`src/justhtml/parser/scanner.py`): Shared low-level scanning for tags, rawtext, and script end markers.
- DOM (`src/justhtml/dom/`): DOM-like node tree. Use `append_child()` / `insert_before()` for public tree operations.
- Entities (`src/justhtml/core/entities.py`): HTML5 named and numeric character-reference decoding.
- Constants (`src/justhtml/core/constants.py`): HTML5 element categories, void elements, formatting elements, and namespaces.
- Transforms (`src/justhtml/transforms/`): Compiled post-parse tree transformations.

### Golden Rules
1. **Spec compliance first**: Follow WHATWG HTML5 spec exactly. No heuristics, no shortcuts.
2. **No exceptions in hot paths**: Use deterministic control flow, not try/except for branching.
3. **No reflective probing**: No `hasattr`, `getattr`, or `delattr` - all data structures used are deterministic.
4. **Minimal allocations**: Reuse buffers and avoid transient objects in parser hot paths.
5. **Node ownership**: Move nodes through DOM helpers and keep parent/child links consistent.
6. **State-machine clarity**: Keep parser recovery and insertion-mode transitions explicit.
7. **No test-specific code**: No references to test files in comments or code.

### Testing Workflow
1. **Target failures**: Use `--test-specs file:indices` to run specific tests
   ```bash
   python run_tests.py --test-specs test2.test:5,10 -v
   ```

2. **Check test output**: Use `-v` for diffs, `-vv` for debug output
   ```bash
   python run_tests.py --test-specs test3.test -vv
   ```

3. **Run full suite**: Always check for regressions
   ```bash
   python run_tests.py -q  # Quick overview
   python run_tests.py --regressions  # Check for new failures vs baseline
   ```

4. **Quick iteration**: Test snippet without full suite (full suite runs in ~1s)
   ```bash
   python -c 'from justhtml import JustHTML, to_test_format; print(to_test_format(JustHTML("<html>").root))'
   ```

5. **Benchmark performance**: After changes, verify speed impact
   ```bash
   python benchmarks/performance.py --iterations 1 --parser justhtml --no-mem
   ```

6. **Profile hotspots**: For performance optimization
   ```bash
   python benchmarks/profile.py  # Profiles on web100k dataset
   ```

### Test Runner Flags
- `--test-specs FILE[:INDICES]`: Run specific test(s), e.g., `test2.test:5,10` or `tests1.dat`
- `-v, -vv, -vvv`: Verbosity (diffs, debug output, full debug)
- `-q, --quiet`: Summary only
- `-x, --fail-fast`: Stop on first failure
- `--regressions`: Compare against HEAD baseline
- `--exclude-files`, `--exclude-errors`, `--exclude-html`: Skip tests matching patterns
- `--filter-errors`, `--filter-html`: Only run tests matching patterns

### Benchmark Flags (benchmarks/performance.py)
- `--iterations 1`: Single run (default: 5 for averaging)
- `--parser justhtml`: Benchmark only JustHTML (default: all parsers)
- `--no-mem`: Disable memory profiling (faster)
- `--limit N`: Test on N files (default: 100)

### Logging & Comments
- Comments explain **why** (spec rationale), not **what** (code is self-documenting)
- Cite spec sections when relevant (e.g., "Per §13.2.5.72")
- No historical notes ("Previously", "Fixed", "Changed") - prefer removing old code
- Debug calls: `self.debug()` / `parser.debug()` - no gating needed

### Performance Mindset
- Parser scanning and tree construction are hot paths: minimize allocations and avoid unnecessary string slicing
- Use `str.find()` for scanning, not regex when possible
- Reuse buffers: `text_buffer`, `current_tag_name`, etc.
- Infer state from structure (stacks, tree) instead of storing flags
