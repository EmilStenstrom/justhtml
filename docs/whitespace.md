# Whitespace rules

HTML and CSS syntax use five whitespace characters: TAB (U+0009), LF
(U+000A), FF (U+000C), CR (U+000D), and SPACE (U+0020). Python's `split()`,
`strip()`, `isspace()`, and regular-expression `\s` recognize additional
characters. These operations are not interchangeable when interpreting HTML.

For example, `class="alpha beta"` with a nonbreaking space contains one
class. It must not match `.alpha`. The nonbreaking space also remains part
of an identifier at the end of `.alpha ` or inside `:not(.alpha )`.

## Implementation

Use `HTML_SPACE_CHARACTERS` when trimming HTML/CSS syntax. Use
`core.whitespace.split_html_whitespace()` for token lists. The helper uses
Python's native splitter for printable text, where SPACE is the only whitespace,
and an explicit HTML character class for other input. There are no repeated
replacement passes, Python loops over characters, or Unicode-space tables.

Selector token sets are shared by attribute value within one query. Normalized
attribute dictionaries and per-attribute token dictionaries are created only
when needed. The caches do not survive a query, and byte accounting runs
before looking up a shared token set. Mixed-case duplicate attribute names
retain their existing last-key precedence. Class selectors and `[class~=value]`
use the same class-token cache.

`An+B` formulas validate their syntax before integer conversion. Whitespace
may surround the offset sign, but cannot split digits or separate a
coefficient from `n`. Unicode digits and Python integer conveniences such as
underscores are not CSS syntax. Existing invalid-formula behavior is retained:
an unsupported formula matches no elements.

## Library audit

| Area | Rule and disposition |
| --- | --- |
| Parser and shared scanner | Existing ASCII delimiter sets retained; DOM and event-stream tests verify that non-HTML whitespace remains in tag names. |
| Streamed doctypes | Explicit HTML whitespace replaces Unicode `\s`; names agree with DOM parsing. |
| Byte encoding labels | Trim only HTML whitespace; preserve non-ASCII bytes so an invalid label cannot become valid by dropping bytes. UTF-7 rejection remains in place. |
| Class and attribute word selectors | Split only on HTML whitespace, including when using direct `matches()` and selector-targeted transforms. |
| Selector edges, pseudo arguments, and transform fast paths | Trim only HTML whitespace. |
| Structural selectors | Non-HTML whitespace counts as content for `:empty`; numeric formulas validate ASCII syntax. |
| HTML output | Compact output preserves text; pretty output trims and collapses only HTML whitespace, including whitespace-only nodes and mixed block/inline content. |
| `CollapseWhitespace` | Existing HTML-only rules retained. |
| `PruneEmpty` | Non-HTML whitespace counts as content; `strip_whitespace=False` also preserves ordinary HTML whitespace. |
| `MergeAttrs` | Existing and supplied tokens preserve non-HTML whitespace. Existing case normalization is unchanged. |
| CSS declarations | Trim only HTML/CSS whitespace around property names and values. Conservative resource-function and escape rejection remains in place. |
| URL lists and `srcset` | Split list tokens and descriptors using HTML whitespace. Controls are still rejected by URL policy, rather than converted into separators. Width descriptors require ASCII digits. |
| Standalone URL checks, URL serialization, refresh guards, URL proxy configuration | Existing conservative normalization retained. URL acceptance is a security-policy decision; these operations do not define HTML token boundaries. |
| Plain-text extraction and `:contains()` | Existing Unicode trimming retained for normalized human-readable text. Use `to_text(strip=False)` when preserving text characters is required. |
| Linkification | Existing Unicode word boundaries retained for discovering URLs in human-readable text. |
| CLI, policy/glob configuration, builder shorthand | Existing Unicode trimming/rejection retained for trusted application configuration rather than parsed HTML syntax. |

## Validation

Behavior-specific tests share an exhaustive enumeration of Python's Unicode
whitespace characters, with separate positive controls for all five HTML
separators. They cover token interiors and edges, direct and nested selectors,
transform matching, both sanitization execution paths, compact and pretty
output, encoding labels, and both DOM and stream parsing. Tests also verify
cache reuse, attribute precedence, resource budgets, and linear scaling with
unique class values.

Run the full correctness gates with `pre-commit run --all-files`. The coverage
gate includes the external HTML fixtures and requires 100% line and branch
coverage. The parser differential gate compares tree construction against the
reference implementation.

For performance, compare the same interpreter and machine before and after:

```bash
PYTHONPATH=src python benchmarks/whitespace.py --iterations 20 --repeats 7
python benchmarks/performance.py --parsers justhtml justhtml_to_html --iterations 5
```

The focused benchmark times queries on pre-parsed DOMs with repeated, unique,
Unicode, long-list, and multi-attribute class values, plus compact and pretty
serialization. Use the current benchmark script with each checkout's
`PYTHONPATH` when comparing revisions. The corpus benchmark measures the
default parser and parse-plus-HTML pipelines on web100k. Timing comparisons
describe the measured workloads and interpreter; they cannot guarantee equal
performance for every application or Python implementation.

## Standards

- [Infra: ASCII whitespace and splitting](https://infra.spec.whatwg.org/#split-on-ascii-whitespace)
- [HTML: space-separated tokens](https://html.spec.whatwg.org/multipage/common-microsyntaxes.html#space-separated-tokens)
- [Selectors: whitespace](https://www.w3.org/TR/selectors-4/#whitespace)
- [Selectors: `:empty`](https://www.w3.org/TR/selectors-4/#the-empty-pseudo)
- [CSS Syntax: `An+B`](https://www.w3.org/TR/css-syntax-3/#anb-microsyntax)
- [Encoding: label matching](https://encoding.spec.whatwg.org/#concept-encoding-get)
