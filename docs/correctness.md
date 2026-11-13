[← Back to docs](index.md)

# Correctness testing

JustHTML is tested against the web platform html5 treebuilder fixtures, with coverage checks, parser comparisons, and fuzzing.

## The web platform tests

The web platform html5 treebuilder tests live in [web-platform-tests/wpt](https://github.com/web-platform-tests/wpt/tree/57bdb7f88a04f09d441ecb4eeebf23664884aa06/html/syntax/parsing/resources). Serializer and encoding fixtures remain in [html5lib-tests](https://github.com/html5lib/html5lib-tests).

Verified on 2026-10-04 against WPT commit [`57bdb7f88a`](https://github.com/web-platform-tests/wpt/commit/57bdb7f88a04f09d441ecb4eeebf23664884aa06): JustHTML passed all 1,917 non-script treebuilder cases.

The external fixture inputs contain:

- **62 treebuilder test files** - Testing how the parser builds the DOM tree
- **5 serializer fixture files** - Testing how token streams are serialized back to HTML
- **Encoding sniffing tests** - Testing BOM/meta charset/transport overrides and legacy fallbacks
- **1,959 treebuilder cases** - Covering edge cases, error recovery, and spec compliance

### What the tests cover

The tests verify correct handling of:

- **Malformed HTML** - Missing closing tags, misnested elements, invalid attributes
- **Implicit element creation** - `<html>`, `<head>`, and `<body>` are auto-inserted
- **Adoption agency algorithm** - Complex handling of misnested formatting elements
- **Foster parenting** - Content in wrong places (like text directly in `<table>`)
- **Foreign content** - SVG and MathML embedded in HTML
- **Character references** - Named entities (`&amp;`), numeric (`&#65;`), and edge cases
- **Script/style handling** - RAWTEXT and RCDATA content models
- **DOCTYPE parsing** - Quirks mode detection
- **Encoding sniffing** - BOM detection, `<meta charset=...>`, transport overrides (`encoding=`), and `windows-1252` fallback

### Example test case

Here's what a test case looks like (from `tests1.dat`):

```
#data
<b><p></b></i>

#errors
(1:9) Unexpected end tag </i>

#document
| <html>
|   <head>
|   <body>
|     <b>
|     <p>
|       <b>
```

This tests the adoption agency algorithm. When the parser encounters `</b>` inside `<p>`, it splits the formatting across the block element boundary.

## Compliance comparison

We run the same test suite against other Python parsers to compare compliance. The current cross-parser scores, browser results, and their benchmark-specific notes live in the [comparison guide](comparison.md).

The Python-parser scores come from a strict tree comparison against the expected output in the treebuilder fixtures, excluding `#script-on` / `#script-off` cases. Unsupported parser capabilities count as failures. The numbers will not match the `html5lib` project’s own reported totals, because `html5lib` runs the suite in multiple configurations and also has its own skip/xfail lists. Run `python benchmarks/correctness.py` to reproduce the Python-parser benchmark.

## Our testing strategy

### 1. Official and project test suite

We run the complete html5lib test suite on every commit:

```bash
python run_tests.py
```

To run only a single suite (useful for faster iteration), use `--suite`:

```bash
python run_tests.py --suite tree
python run_tests.py --suite justhtml
python run_tests.py --suite serializer
python run_tests.py --suite encoding
python run_tests.py --suite unit
```

<!-- justhtml: output -->
```
PASSED: 3464/3464 passed (100.0%)
```

There are also 6 expected skips, including scripted (`#script-on`) cases that
require JavaScript execution during parsing.

Per-file results are also written to `test-summary.txt`, with suite prefixes like `html5lib-tests-tree/...`, `html5lib-tests-serializer/...`, `html5lib-tests-encoding/...`, and `justhtml-tests/...`.

The encoding coverage comes from both:

- The official `html5lib-tests/encoding` fixtures (exposed in this repo as `tests/html5lib-tests-encoding/...`).
- JustHTML's own unit tests (see `tests/test_encoding.py`) which exercise byte input, encoding label normalization, BOM handling, and meta charset prescanning.

### 2. Coverage and parser differential checks

The test suite enforces 100% combined line and branch coverage, including the parser engine:

```bash
coverage run run_tests.py && coverage report --fail-under=100
```

The parser engine is additionally checked behaviorally:

```bash
PYTHONPATH=src python benchmarks/html5lib_engine_diff.py \
  --fail-under-rate 1.0 \
  --fail-on-current-exceptions
```

This requires exact agreement with the reference parser path across every scored web platform html5 treebuilder case.

### 3. Fuzz testing (millions of cases)

We generate random malformed HTML to find crashes and hangs:

```bash
python benchmarks/fuzz.py -n 3000000
```

<!-- justhtml: output -->
```
============================================================
FUZZING RESULTS: justhtml
============================================================
Total tests:    3000000
Successes:      3000000
Crashes:        0
Hangs (>5s):    0
Total time:     928s
Tests/second:   3232
```

The fuzzer generates malformed input such as:
- Deeply nested elements
- Invalid character references (`&#xFFFFFFFF;`)
- Mismatched tags (`<b><p></b></i>`)
- CDATA in wrong contexts
- Null bytes and control characters
- Malformed doctypes
- SVG/MathML interleaved with HTML

### 4. Custom edge case tests

We maintain additional tests in `tests/justhtml-tests/` for:
- Branch coverage gaps found during development
- Edge cases discovered by fuzzing
- XML coercion handling
- iframe srcdoc parsing
- Empty stack edge cases

## Running the tests

### Quick start

```bash
# Clone the test suites (one-time setup)
cd ..
git clone --filter=blob:none --sparse https://github.com/web-platform-tests/wpt.git
cd wpt
git sparse-checkout set html/syntax/parsing/resources
cd ..
git clone https://github.com/html5lib/html5lib-tests.git
cd justhtml

# Create symlinks
cd tests
ln -s ../../wpt/html/syntax/parsing/resources html5lib-tests-tree
ln -s ../../html5lib-tests/serializer html5lib-tests-serializer
ln -s ../../html5lib-tests/encoding html5lib-tests-encoding
cd ..

# Run all tests
python run_tests.py
```

### Test runner options

```bash
# Verbose output with diffs
python run_tests.py -v

# Run specific test file
python run_tests.py --test-specs test2.test:5,10

# Stop on first failure
python run_tests.py -x

# Check for regressions against baseline
python run_tests.py --regressions
```

### Correctness benchmark

Compare against other parsers:

```bash
python benchmarks/correctness.py
```

<a id="why-100-matters"></a>

## Coverage of HTML5 parsing rules

The HTML5 parsing specification includes:
- Rawtext/RCDATA/script scanning states
- 23 tree construction insertion modes
- The "adoption agency algorithm" (called "the most complicated part of the tree builder" by Firefox's HTML5 parser author)
- Foster parenting for misplaced table content
- "Noah's Ark" clause limiting identical elements to 3

JustHTML passed all 1,917 non-script treebuilder cases in the 2026-10-04 verification described above. Passing these fixtures does not establish complete HTML5 compliance. Browser results and differences in the scored cases are documented in the [comparison guide](comparison.md).

## Error diagnostics

The html5lib suite verifies tree output, not a standardized diagnostic stream.
JustHTML reports a limited set of parser errors. Enable `collect_errors=True`
to collect them:

```python
doc = JustHTML("<!doctype html><!--", collect_errors=True)
for error in doc.errors:
    print(f"{error.line}:{error.column} {error.code}")
# Output: 1:19 eof-in-comment
```

Error collection is optional and adds work. Strict mode raises on the earliest
supported diagnostic, but is not a complete HTML conformance validator.

See [Error Codes](errors.md) for the supported set and stability contract.
