# JustHTML

A pure Python HTML5 parser that just works. No C extensions to compile. No system dependencies to install. No complex API to learn.

[📖 Full documentation](https://emilstenstrom.github.io/justhtml/) | [🛝 Try it in the Playground](https://emilstenstrom.github.io/justhtml/playground/)

## Why use JustHTML?

### Just... Correct ✅

Spec-perfect HTML5 parsing with browser-grade error recovery. Passes the official 9k+ [html5lib-tests](https://github.com/html5lib/html5lib-tests) suite, with 100% line+branch coverage. ([Correctness](docs/correctness.md))

```python
JustHTML("<p><b>Hi<i>there</b>!", fragment=True).to_html(pretty=False)
# => <p><b>Hi<i>there</i></b><i>!</i></p>
```

**Note**: `fragment=True` parses snippets (no `<html>`/`<body>` needed).

### Just... Secure 🔒

Safe-by-default sanitization at construction time. Built-in Bleach-style allowlist sanitization on `JustHTML(...)` (disable with `sanitize=False`). Can sanitize inline CSS rules. ([Sanitization & Security](docs/sanitization.md))

```python
JustHTML(
    "<p>Hello<script>alert(1)</script> "
    "<a href=\"javascript:alert(1)\">bad</a> "
    "<a href=\"https://example.com/?a=1&b=2\">ok</a></p>",
    fragment=True,
).to_html()
# => <p>Hello <a>bad</a> <a href="https://example.com/?a=1&amp;b=2">ok</a></p>
```

### Just... Query 🔍

CSS selectors out of the box. Two methods (`query()`, `query_one()`), familiar syntax (combinators, groups, pseudo-classes), and plain Python nodes as results. ([CSS Selectors](docs/selectors.md))

```python
JustHTML(
    "<div><p class=\"x\">Hi</p><p>Bye</p></div>",
    fragment=True,
).query_one("div p.x").to_html(pretty=False)
# => <p class="x">Hi</p>
```

⚠️ **Note**: Sanitization happens before querying, so remember to disable (`sanitize=False`) if you are working with safe HTML.

### Just... Transform 🏗️

Built-in DOM transforms for dropping and unwrapping nodes, rewriting attributes, linkifying text, and composing safe pipelines. ([Transforms](docs/transforms.md))

```python
from justhtml import JustHTML, Linkify, SetAttrs, Unwrap

doc = JustHTML(
    "<p>Hello <span class=\"x\">world</span> example.com</p>",
    transforms=[
        Unwrap("span.x"),
        Linkify(),
        SetAttrs("a", rel="nofollow"),
    ],
    fragment=True,
    safe=False,
)
print(doc.to_html(pretty=False))
# => <p>Hello world <a href="http://example.com" rel="nofollow">example.com</a></p>
```

### Just... Build 🧱

Build node trees directly when Python is driving the HTML, then normalize them through the same HTML5 parser. ([Building HTML](docs/building.md))

```python
from justhtml import JustHTML
from justhtml.builder import element

doc = JustHTML(
    element(
        "article",
        {"class": "post"},
        element("h2", "JustHTML"),
        element("p", "Build nodes directly."),
        element("a", {"href": "/docs"}, "Read docs"),
    ),
    fragment=True,
    sanitize=False,
)
print(doc.to_html(pretty=False))
# => <article class="post"><h2>JustHTML</h2><p>Build nodes directly.</p><a href="/docs">Read docs</a></article>
```

### Just... Python 🐍

Pure Python, zero dependencies. No C extensions or system libraries, easy to debug, and works anywhere Python runs, including PyPy and Pyodide. ([Run in the browser](https://emilstenstrom.github.io/justhtml/playground/))

```bash
python -m pip show justhtml | grep -E '^Requires:'
# Requires: [intentionally left blank]
```

### Just... Fast Enough ⚡

Fast for the common case, and the fastest pure-Python HTML5 parser available. For terabytes, use a C/Rust parser like `html5ever`. ([Benchmarks](benchmarks/performance.py))

```bash
curl -Ls https://en.wikipedia.org/wiki/HTML -o /tmp/justhtml-bench.html
/usr/bin/time -f '%e s' python -m justhtml /tmp/justhtml-bench.html > /dev/null
# 0.22 s
```

## Comparison

| Tool | HTML5 parsing [1][2] | Speed | Query | Build | Sanitize | Notes |
|------|------------------------------------------|-------|----------|-------|------------------|-------|
| **JustHTML**<br>Pure Python | ✅&nbsp;**100%** | ⚡ Fast | ✅ CSS selectors | ✅ `element()` | ✅ Built-in | Correct, secure, easy to install, and fast enough. |
| **Chromium**<br>browser engine | ✅ **99%** | 🚀&nbsp;Very&nbsp;Fast | — | — | — | — |
| **WebKit**<br>browser engine | ✅ **98%** | 🚀 Very Fast | — | — | — | — |
| **Firefox**<br>browser engine | ✅ **97%** | 🚀 Very Fast | — | — | — | — |
| **`markupever`**<br>Python wrapper of Rust-based html5ever | ✅ **95%** | 🚀 Very Fast | ✅ CSS selectors | ✅ `TreeDom. create_*()` | ❌ Needs sanitization | Fast and correct. |
| **`html5lib`**<br>Pure Python | 🟡 88% | 🐢 Slow | 🟡 XPath (lxml) | 🟡 Tree API | 🔴 [Deprecated](https://github.com/html5lib/html5lib-python/issues/443) | Unmaintained. Reference implementation; Correct but quite slow. |
| **`html5_parser`**<br>Python wrapper of C-based Gumbo | 🟡 84% | 🚀 Very Fast | 🟡 XPath (lxml) | 🟡 `etree` (lxml) | ❌ Needs sanitization | Fast and mostly correct. |
| **`selectolax`**<br>Python wrapper of C-based Lexbor | 🟡 74% | 🚀 Very Fast | ✅ CSS selectors | ✅ `create_node()` | ❌ Needs sanitization | Very fast; score reflects this benchmark's selectolax public-API serializer. |
| **`BeautifulSoup`**<br>Pure Python | 🔴 4% (default) | 🐢 Slow | 🟡 Custom API | ✅ `new_tag()` API | ❌ Needs sanitization | Wraps `html.parser` (default). Can use lxml or html5lib. |
| **`html.parser`**<br>Python stdlib | 🔴 4% | ⚡ Fast | ❌ None | ❌ None | ❌ Needs sanitization | Standard library. Chokes on malformed HTML. |
| **`lxml`**<br>Python wrapper of C-based libxml2 | 🔴 1% | 🚀 Very Fast | 🟡 XPath | ✅ `etree` / E-factory | ❌ Needs sanitization | Fast but not HTML5 compliant. Don't use the old lxml.html.clean module! |

[1]: Parser compliance scores are from a strict run of the [html5lib-tests](https://github.com/html5lib/html5lib-tests) tree-construction fixtures (1,743 non-script tests). See [docs/correctness.md](docs/correctness.md) for details.

[2]: Browser numbers are from [`justhtml-html5lib-tests-bench`](https://github.com/EmilStenstrom/justhtml-html5lib-tests-bench) on the upstream `html5lib-tests/tree-construction` corpus (excluding 12 scripting-enabled cases).


## Installation

```bash
pip install justhtml
```

Next: [Quickstart Guide](docs/quickstart.md), [CSS Selectors](docs/selectors.md), [Sanitization & Security](docs/sanitization.md), or [try the Playground](https://emilstenstrom.github.io/justhtml/playground/).

Requires Python 3.10 or later.

## Quick Example

```python
from justhtml import JustHTML

doc = JustHTML("<html><body><p class='intro'>Hello!</p></body></html>")

# Query with CSS selectors
for p in doc.query("p.intro"):
    print(p.name)        # "p"
    print(p.attrs)       # {"class": "intro"}
    print(p.to_html())   # <p class="intro">Hello!</p>
```

See the **[Quickstart Guide](docs/quickstart.md)** for more examples including tree traversal, streaming, and strict mode.

## Command Line

If you installed JustHTML (for example with `pip install justhtml` or `pip install -e .`), you can use the `justhtml` command.
If you don't have it available, use the equivalent `python -m justhtml ...` form instead.

```bash
# Pretty-print an HTML file
justhtml index.html

# Parse from stdin
curl -s https://example.com | justhtml -

# Select nodes and output text
justhtml index.html --selector "main p" --format text

# Select nodes and output Markdown (subset of GFM)
justhtml index.html --selector "article" --format markdown

# Select nodes and output HTML
justhtml index.html --selector "a" --format html
```

```bash
# Example: extract Markdown from GitHub README HTML
curl -s https://github.com/EmilStenstrom/justhtml/ | justhtml - --selector '.markdown-body' --format markdown --unsafe | head -n 8
```

Output:

```text
# JustHTML

[](#justhtml)

A pure Python HTML5 parser that just works. No C extensions to compile. No system dependencies to install. No complex API to learn.

[📖 Full documentation](https://emilstenstrom.github.io/justhtml/) | [🛝 Try it in the Playground](https://emilstenstrom.github.io/justhtml/playground/)
```

## Security

For security policy and vulnerability reporting, please see [SECURITY.md](SECURITY.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## Acknowledgments

JustHTML started as a Python port of [html5ever](https://github.com/servo/html5ever), the HTML5 parser from Mozilla's Servo browser engine. While the codebase has since evolved significantly, html5ever's clean architecture and spec-compliant approach were invaluable as a starting point. Thank you to the Servo team for their excellent work.

Correctness and conformance work is heavily guided by the [html5lib](https://github.com/html5lib/html5lib-python) ecosystem and especially the official [html5lib-tests](https://github.com/html5lib/html5lib-tests) fixtures used across implementations.

The sanitization API and threat-model expectations are informed by established Python sanitizers like [Bleach](https://github.com/mozilla/bleach) and [nh3](https://github.com/messense/nh3).

The CSS selector query API is inspired by the ergonomics of [lxml.cssselect](https://lxml.de/cssselect.html).

## License

MIT. Free to use both for commercial and non-commercial use.
