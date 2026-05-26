[← Back to docs](index.md)

# Comparison

Use JustHTML when you want browser-grade HTML parsing, safe-by-default sanitization, CSS selectors, transforms, text extraction, and serialization in one pure-Python package.

Use a different tool when one narrow requirement matters more than the whole pipeline: maximum throughput, a BeautifulSoup-specific API, XPath-heavy XML work, or integration with an existing lxml tree.

## At a Glance

| Tool | HTML5 parsing [1][2] | Speed | Query | Build | Sanitize | Notes |
|------|------------------------------------------|-------|----------|-------|------------------|-------|
| **JustHTML**<br>Pure Python | ✅&nbsp;100% | ⚡ Fast | ✅ CSS selectors | ✅ `element()` | ✅ Built-in | Correct, secure, easy to install, and fast enough. |
| **`selectolax`**<br>Python wrapper of C-based Lexbor | ✅&nbsp;100% | 🚀 Very Fast | ✅ CSS selectors | ✅ `create_node()` | ❌ Needs sanitization | Very fast and spec-compliant. |
| **Chromium**<br>browser engine | ✅&nbsp;99.5% | 🚀&nbsp;Very&nbsp;Fast | — | — | — | — |
| **WebKit**<br>browser engine | ✅ 98.4% | 🚀 Very Fast | — | — | — | — |
| **Firefox**<br>browser engine | ✅ 97.6% | 🚀 Very Fast | — | — | — | — |
| **`markupever`**<br>Python wrapper of Rust-based html5ever | 🟡 89% | 🚀 Very Fast | ✅ CSS selectors | ✅ `TreeDom .create_*()` | ❌ Needs sanitization | Fast and mostly correct, but missing benchmarked capabilities count against compliance. |
| **`html5lib`**<br>Pure Python | 🟡 86% | 🐢 Slow | 🟡 XPath (lxml) | 🟡 Tree API | 🔴 [Deprecated](https://github.com/html5lib/html5lib-python/issues/443) | Unmaintained reference implementation; incomplete coverage of the tree-construction fixtures. |
| **`html5_parser`**<br>Python wrapper of C-based Gumbo | 🔴 49% | 🚀 Very Fast | 🟡 XPath (lxml) | 🟡 `etree` (lxml) | ❌ Needs sanitization | Fast, but its public tree API loses information needed by many fixtures. |
| **`BeautifulSoup`**<br>Pure Python | 🔴 <1% (default) | 🐢 Slow | 🟡 Custom API | ✅ `new_tag()` API | ❌ Needs sanitization | Wraps `html.parser` (default). Can use lxml or html5lib. |
| **`html.parser`**<br>Python stdlib | 🔴 <1% | ⚡ Fast | ❌ None | ❌ None | ❌ Needs sanitization | Standard library. Chokes on malformed HTML. |
| **`lxml`**<br>Python wrapper of C-based libxml2 | 🔴 <1% | 🚀 Very Fast | 🟡 XPath | ✅ `etree` / E-factory | ❌ Needs sanitization | Fast but not HTML5 compliant. Context-fragment cases are skipped; supported cases still perform poorly. Don't use the old lxml.html.clean module! |

[1]: Parser compliance scores are from a strict run of the [html5lib-tests](https://github.com/html5lib/html5lib-tests) tree-construction fixtures (1,743 non-script tests). The score is `pass / (pass + fail + error)`; unsupported public API capabilities count as failures rather than being faked. The benchmark may compose multiple public APIs from the same parser, but does not use testcase-specific shims or synthetic adapters when an API surface is missing. See [Correctness Testing](correctness.md) for details.

[2]: Browser numbers are from a local rerun of [`justhtml-html5lib-tests-bench`](https://github.com/EmilStenstrom/justhtml-html5lib-tests-bench) against this repo's `tests/html5lib-tests-tree/*.dat` corpus: Chromium 1762/1770, WebKit 1742/1770, Firefox 1728/1770, with 12 skipped scripting-enabled cases per engine.

## Why JustHTML

Most Python HTML projects start simple and then accumulate extra tools:

- a parser for broken HTML
- a sanitizer for user input
- a selector engine
- a serializer
- linkification or cleanup filters
- text or Markdown extraction

JustHTML keeps those operations on one DOM. That makes the behavior easier to reason about, especially when the input is untrusted.

```python
from justhtml import JustHTML

doc = JustHTML("<p>Hello<script>alert(1)</script><a href='javascript:x'>link</a></p>", fragment=True)

print(doc.to_html(pretty=False))
# <p>Hello<a>link</a></p>
```

Sanitization happens before you query or serialize unless you explicitly disable it with `sanitize=False`.

## When to Choose Another Tool

Choose **selectolax** when raw speed is the main requirement and the HTML is trusted or sanitized elsewhere.

Choose **markupever** or **html5_parser** when you specifically want their underlying parser engines or tree APIs and can accept their compatibility tradeoffs.

Choose **BeautifulSoup** when you want its forgiving, familiar scraping API and parser correctness is not the main risk.

Choose **lxml** when your project is already built around XPath, etree, or XML-style processing.

Choose **nh3** when you only need fast sanitization and are happy with a Rust-backed dependency.

Choose **html.parser** when you need a tiny stdlib-only script for trusted input and HTML5 correctness does not matter.

Choose **Bleach** only for existing codebases that already depend on it. For new projects, prefer an actively maintained sanitizer path. See [Migrating from Bleach](bleach-migration.md).

## Tradeoffs

JustHTML is pure Python. That makes it easy to install, inspect, debug, and run in environments like Pyodide, but it will not beat C or Rust parsers on raw throughput.

JustHTML sanitizes HTML output by default. That is the right default for user-generated content, CMS snippets, comments, scraped fragments, and transform pipelines that eventually return to a browser. If all of your input is trusted, pass `sanitize=False`.

JustHTML's sanitizer emits HTML-only output. SVG and MathML can still be parsed when sanitization is disabled, but sanitized output drops foreign-namespace content to keep the security model smaller and more reviewable.

## Related Pages

- [Correctness Testing](correctness.md)
- [Sanitization & Security](sanitization.md)
- [Migrating from Bleach](bleach-migration.md)
- [Performance Benchmark](../benchmarks/performance.py)
