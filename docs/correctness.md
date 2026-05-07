[← Back to docs](index.md)

# Correctness Testing

JustHTML is the only pure-Python HTML5 parser that passes 100% of the official html5lib test suite. This page explains how we verify and maintain that compliance.

## The html5lib Test Suite

The [html5lib-tests](https://github.com/html5lib/html5lib-tests) repository is the gold standard for HTML5 parsing compliance. It's used by browser vendors to verify their implementations against the [WHATWG HTML5 specification](https://html.spec.whatwg.org/).

The suite contains:
- **56 tree-construction test files** - Testing how the parser builds the DOM tree
- **14 tokenizer test files** - Testing lexical analysis of HTML
- **5 serializer fixture files** - Testing how token streams are serialized back to HTML
- **Encoding sniffing tests** - Testing BOM/meta charset/transport overrides and legacy fallbacks
- **9k+ individual test cases** - Covering edge cases, error recovery, and spec compliance

### What the Tests Cover

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

### Example Test Case

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

This tests the adoption agency algorithm - when `</b>` is encountered inside `<p>`, the browser doesn't just close `<b>`. Instead, it splits the formatting across the block element boundary.

## Compliance Comparison

We run the same test suite against other Python parsers to compare compliance:

| Parser | Tests Passed | Compliance | Notes |
|--------|-------------|------------|-------|
| **JustHTML** | 1743/1743 | **100%** | Full spec compliance |
| selectolax | 1742/1743 | 99.9% | C-based (Lexbor), fast and nearly spec-compliant with dev `html5test` output API |
| markupever | 1545/1743 | 89% | Rust-based (html5ever), mostly correct |
| html5lib | 1496/1743 | 86% | Reference implementation, but incomplete |
| html5_parser | 862/1743 | 49% | C-based (Gumbo), fast but loses exposed tree information |
| BeautifulSoup | 6/1743 | <1% | Uses html.parser, not HTML5 compliant |
| html.parser | 6/1743 | <1% | Python stdlib, basic error recovery only |
| lxml | 5/1743 | <1% | XML-based, not HTML5 compliant |

*Run `python benchmarks/correctness.py` to reproduce these results. The selectolax score requires its dev `html5test` output and fragment-context APIs.*

These numbers come from a strict tree comparison against the expected output in the `html5lib-tests` tree-construction fixtures (excluding `#script-on` / `#script-off` cases). Unsupported parser capabilities count as failures for this compliance table. The numbers will not match the `html5lib` project’s own reported totals, because `html5lib` runs the suite in multiple configurations and also has its own skip/xfail lists.

## Our Testing Strategy

### 1. Official Test Suite (9k+ tests)

We run the complete html5lib test suite on every commit:

```bash
python run_tests.py
```

To run only a single suite (useful for faster iteration), use `--suite`:

```bash
python run_tests.py --suite tree
python run_tests.py --suite justhtml
python run_tests.py --suite tokenizer
python run_tests.py --suite serializer
python run_tests.py --suite encoding
python run_tests.py --suite unit
```

Output:
```
PASSED: 9k+ tests (100%), a few skipped
```

The skipped tests are scripted (`#script-on`) cases that require JavaScript execution during parsing.

Per-file results are also written to `test-summary.txt`, with suite prefixes like `html5lib-tests-tree/...`, `html5lib-tests-tokenizer/...`, `html5lib-tests-serializer/...`, `html5lib-tests-encoding/...`, and `justhtml-tests/...`.

The encoding coverage comes from both:

- The official `html5lib-tests/encoding` fixtures (exposed in this repo as `tests/html5lib-tests-encoding/...`).
- JustHTML's own unit tests (see `tests/test_encoding.py`) which exercise byte input, encoding label normalization, BOM handling, and meta charset prescanning.

### 2. 100% Code Coverage

Every line and branch of code is covered by tests. We enforce this in CI:

```bash
coverage run run_tests.py && coverage report --fail-under=100
```

This isn't just vanity - during development, we discovered that uncovered code was often dead code. Removing it made the parser faster and cleaner.

### 3. Fuzz Testing (millions of cases)

We generate random malformed HTML to find crashes and hangs:

```bash
python benchmarks/fuzz.py -n 3000000
```

Output:
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

The fuzzer generates truly nasty edge cases:
- Deeply nested elements
- Invalid character references (`&#xFFFFFFFF;`)
- Mismatched tags (`<b><p></b></i>`)
- CDATA in wrong contexts
- Null bytes and control characters
- Malformed doctypes
- SVG/MathML interleaved with HTML

### 4. Custom Edge Case Tests

We maintain additional tests in `tests/justhtml-tests/` for:
- Branch coverage gaps found during development
- Edge cases discovered by fuzzing
- XML coercion handling
- iframe srcdoc parsing
- Empty stack edge cases

## Running the Tests

### Quick Start

```bash
# Clone the test suite (one-time setup)
cd ..
git clone https://github.com/html5lib/html5lib-tests.git
cd justhtml

# Create symlinks
cd tests
ln -s ../../html5lib-tests/tokenizer html5lib-tests-tokenizer
ln -s ../../html5lib-tests/tree-construction html5lib-tests-tree
ln -s ../../html5lib-tests/serializer html5lib-tests-serializer
ln -s ../../html5lib-tests/encoding html5lib-tests-encoding
cd ..

# Run all tests
python run_tests.py
```

### Test Runner Options

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

### Correctness Benchmark

Compare against other parsers:

```bash
python benchmarks/correctness.py
```

## Why 100% Matters

HTML5 parsing is notoriously complex. The spec describes an intricate state machine with:
- 80+ tokenizer states
- 23 tree builder insertion modes
- The "adoption agency algorithm" (called "the most complicated part of the tree builder" by Firefox's HTML5 parser author)
- Foster parenting for misplaced table content
- "Noah's Ark" clause limiting identical elements to 3

Getting 99% compliance means you're still breaking on real-world edge cases. Browsers pass 100% because they have to - and now JustHTML does too.

## Standardizing Error Codes

Beyond tree construction, we're working to standardize parse error reporting. The HTML5 spec defines specific error codes for malformed input, but:

- The html5lib test suite focuses on tree output, not error codes
- Different parsers report errors inconsistently (or not at all)
- Error messages vary wildly between implementations

JustHTML uses **kebab-case error codes** matching the WHATWG spec where possible:

```python
doc = JustHTML("<p>Hello", collect_errors=True)
for error in doc.errors:
    print(f"{error.line}:{error.column} {error.code}")
# Output: 1:9 expected-closing-tag-but-got-eof
```

Our error codes are centralized in `src/justhtml/errors.py` with human-readable messages. This makes it possible to:

1. **Lint HTML** - Report all parse errors with source locations
2. **Strict mode** - Reject malformed HTML entirely
3. **Compare implementations** - Verify error detection matches the spec

See [Error Codes](errors.md) for the complete list.
