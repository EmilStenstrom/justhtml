[← Back to docs](index.md)

# Comparison

Use JustHTML when you want browser-grade HTML parsing, safe-by-default sanitization, CSS selectors, transforms, text extraction, and serialization in one pure-Python package.

Use a different tool when one narrow requirement matters more than the whole pipeline: maximum throughput, a BeautifulSoup-specific API, XPath-heavy XML work, or integration with an existing lxml tree.

## At a Glance

| Tool | HTML5 parsing [1] | Speed | Query | Build | Sanitize | Notes |
|------|------------------------------------------|-------|----------|-------|------------------|-------|
| **JustHTML**<br>Pure Python | ✅ 100% | ⚡ Fast | ✅ CSS selectors | ✅ `element()` | ✅ Built-in | Correct, secure, easy to install, and fast enough. |
| **`selectolax`**<br>Python wrapper of C-based Lexbor | 🟡 95.2% [2] | 🚀 Very Fast | ✅ CSS selectors | ✅ `create_node()` | ❌ Needs sanitization | Very fast; processing-instruction fixtures remain in the score. |
| **Chromium**<br>browser engine | 🟡 94.6% [3] | 🚀 Very Fast | — | — | — | Current browser-harness result. |
| **`turbohtml`**<br>Python wrapper of a C core | 🟡 94.1% | 🚀 Very Fast | ✅ CSS selectors, XPath | ✅ `E.*` builder | ✅ Built-in | Broad, compiled alternative with parsing, querying, and sanitization. |
| **WebKit**<br>browser engine | 🟡 93.5% [3] | 🚀 Very Fast | — | — | — | Current browser-harness result. |
| **Firefox**<br>browser engine | 🟡 92.8% [3] | 🚀 Very Fast | — | — | — | Current browser-harness result. |
| **`html5lib`**<br>Pure Python | 🟡 82.2% | 🐢 Slow | 🟡 XPath (lxml) | 🟡 Tree API | 🔴 [Deprecated](https://github.com/html5lib/html5lib-python/issues/443) | Unmaintained reference implementation; incomplete coverage of the tree-construction fixtures. |
| **`markupever`**<br>Python wrapper of Rust-based html5ever | 🟡 79.2% | 🚀 Very Fast | ✅ CSS selectors | ✅ `TreeDom .create_*()` | ❌ Needs sanitization | Fast, but 107 fixture cases abort its current parser process. |
| **`html5_parser`**<br>Python wrapper of C-based Gumbo | 🔴 47.6% | 🚀 Very Fast | 🟡 XPath (lxml) | 🟡 `etree` (lxml) | ❌ Needs sanitization | Fast, but its public tree API loses information needed by many fixtures. |
| **`BeautifulSoup`**<br>Pure Python | 🔴 0.3% (default) | 🐢 Slow | 🟡 Custom API | ✅ `new_tag()` API | ❌ Needs sanitization | Wraps `html.parser` (default). Can use lxml or html5lib. |
| **`html.parser`**<br>Python stdlib | 🔴 0.3% | ⚡ Fast | ❌ None | ❌ None | ❌ Needs sanitization | Standard library. Chokes on malformed HTML. |
| **`lxml`**<br>Python wrapper of C-based libxml2 | 🔴 0.3% | 🚀 Very Fast | 🟡 XPath | ✅ `etree` / E-factory | ❌ Needs sanitization | Fast but not HTML5 compliant. Context-fragment cases are skipped; supported cases still perform poorly. Don't use the old lxml.html.clean module! |

[1]: Parser compliance scores are from strict runs of the [html5lib-tests](https://github.com/html5lib/html5lib-tests) tree-construction fixtures: 1,879 non-scripting cases, with 39 scripting cases skipped. The score is `pass / (pass + fail + error)`; unsupported public API capabilities count as failures rather than being faked. The benchmark may compose multiple public APIs from the same parser, but does not use testcase-specific shims or synthetic adapters when an API surface is missing. See [Correctness Testing](correctness.md) for details.

[2]: The Selectolax result uses its development build with Lexbor's `HTML5TEST` serializer enabled. It passed 1,789 cases; 89 processing-instruction fixtures and one recovery case remain non-passing.

[3]: Current local rerun with [`justhtml-html5lib-tests-bench`](https://github.com/EmilStenstrom/justhtml-html5lib-tests-bench): Chromium 1803/1906 (94.6%), WebKit 1783/1906 (93.5%), Firefox 1768/1906 (92.8%). The browser harness skips 12 `#script-on` cases but includes `#script-off` cases, so these scores are not directly comparable to the 1,879-case Python-parser scores above.

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

Choose **turbohtml** when you want a compiled, all-in-one HTML toolkit and are comfortable depending on a native extension. Its feature set overlaps more with JustHTML's than the parser-only alternatives do.

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
