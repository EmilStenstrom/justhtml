[← Back to docs](index.md)

# Sanitization & Security

JustHTML includes a built-in, **policy-driven HTML sanitizer** intended for rendering *untrusted HTML safely*.

JustHTML’s sanitizer is validated against the [`justhtml-xss-bench`](https://github.com/EmilStenstrom/justhtml-xss-bench) suite (a headless-browser harness), currently covering **7,000+ real-world XSS vectors**. The benchmark can be used to compare output with established sanitizers like `nh3` and `bleach`.

The sanitizer is **DOM-based** (it runs on the parsed JustHTML tree), and JustHTML is **safe-by-default** at construction time.

## Guides

- [HTML Cleaning](html-cleaning.md): tags/attributes allowlists and inline styles.
- [URL Cleaning](url-cleaning.md): validation of URL-valued attributes (`href`, `src`, `srcset`, …), plus strip/proxy rules.
- [Unsafe Handling](unsafe-handling.md): what happens when unsafe input is encountered (strip/collect/raise).
- [Migrating from Bleach](bleach-migration.md): mapping Bleach cleaner/filter pipelines to JustHTML.

## Quickstart

Most real-world untrusted HTML is a **snippet** (a fragment) rather than a full document. In that case, pass `fragment=True` to avoid implicit document wrappers.

If you *are* sanitizing a full HTML document, the default policy keeps the document structure (it preserves `<html>`, `<head>`, and `<body>` wrappers).

By default, construction sanitizes:

```python
from justhtml import JustHTML

doc = JustHTML('<p>Hello <b>world</b> <script>alert(1)</script></p>', fragment=True)
print(doc.to_html())
```

Output:

```html
<p>Hello <b>world</b></p>
```

For a deeper dive, continue in [HTML Cleaning](html-cleaning.md) and [URL Cleaning](url-cleaning.md).

## Sanitizing the in-memory DOM with `Sanitize(...)`

Safe-by-default construction (`JustHTML(..., sanitize=True)`) sanitizes the in-memory tree once, after parsing and transforms run.

Sanitization is appended automatically after any custom transforms. If you want to run transforms *after* sanitization, add `Sanitize(...)` to your transform list and put additional transforms after it:

```python
from justhtml import JustHTML, Sanitize

doc = JustHTML(user_html, fragment=True, transforms=[Sanitize(), PruneEmpty()])
clean_root = doc.root
```

See also: [Transforms](transforms.md) (especially [`Sanitize(...)`](transforms.md#sanitizepolicynone-enabledtrue) and [`Stage([...])`](transforms.md#advanced-stages)).

## Why `Sanitize(...)` is reviewable

`Sanitize(...)` is not a hidden “black box” pass. Internally, it compiles the sanitization policy into a concrete, readable pipeline of smaller transforms (drop content tags, unwrap disallowed tags, drop dangerous attributes, validate URLs, sanitize styles, enforce link `rel`, …).

This is a strong security property:

- The sanitizer becomes a list of explicitly named operations.
- Each step is narrow and testable in isolation.
- Policy changes map directly to a small set of predictable pipeline steps.

If you’re evaluating or auditing sanitization behavior, the [`Sanitize(...)` transform documentation](transforms.md#sanitizepolicynone-enabledtrue) summarizes the pipeline at a high level.

## Threat model

The goal of sanitization is to take **untrusted HTML** and clean it into output that is safe enough to be **embedded as markup into a normal (safe) HTML page**.

In scope:

- Preventing script execution and markup-based XSS when rendering the sanitized output as HTML.
- Safely handling URL-valued attributes (like `href`, `src`, and `srcset`) so they can’t be used to execute script or unexpectedly load remote resources.

Out of scope (you must handle these separately):

- Using sanitized output in non-HTML contexts (JavaScript strings, CSS contexts, URL contexts, etc.).
- Content security beyond markup execution (e.g. phishing / UI redress).
- Security policies like CSP, sandboxing, and permissions (recommended for defense-in-depth).

See [HTML Cleaning](html-cleaning.md) for tag/attribute rules and unsafe handling, and [URL Cleaning](url-cleaning.md) for URL handling (`default_handling`) and URL validation rules.
