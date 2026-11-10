[← Back to docs](index.md)

# Sanitization & security

JustHTML includes a built-in HTML sanitizer. `JustHTML(html)` sanitizes input by default according to a [strict allowlist policy](#configuration). You do not need a separate sanitization library.

## Threat model

JustHTML protects against:
- **XSS**: Script execution via tags (`<script>`), attributes (`onload`), and schemes (`javascript:`).
- **Markup Injection**: Breaking out of intended HTML structure.
- **Remote URL requests**: Only URLs allowed by the policy are kept.

The sanitizer is tested against [`justhtml-xss-bench`](https://github.com/EmilStenstrom/justhtml-xss-bench/), which contains 7,000+ real-world XSS vectors. Use the benchmark to compare JustHTML's output with sanitizers such as nh3 and Bleach.

<a id="important-context-is-king"></a>

## Choose the output context

Output safety depends on where you embed the result. A string that is safe inside a `<div>` might be dangerous inside a `<script>` tag or an HTML attribute.

Choose the output method for the target context:

1. **[HTML Body](#1-safety-for-standard-html)**: `JustHTML(html, fragment=True).to_html()` for [snippets](#fragments-snippets), or `JustHTML(html).to_html()` for [full page documents](#full-documents). (Safe by default)
2. **[JavaScript](#2-safety-for-javascript--dynamic-contexts)**: Use `JustHTML.escape_js_string()` for strings, `clean_url_in_js_string()` for URLs.
3. **[HTML Attributes](#3-safety-for-html-attributes)**: Use `JustHTML.escape_attr_value()` for values, `clean_url_value()` for URLs.

### 1. Safety for standard HTML

For HTML rendered into a page body, `JustHTML(...)` sanitizes input by default. Its conservative policy strips scripts and allows safe tags.

In a custom transform pipeline, `Sanitize(...)` sanitizes the tree at its position in the list. Later transforms can reintroduce unsafe content. See [Transforms](transforms.md#safety-model).

#### Fragments (snippets)

Most real-world untrusted HTML is a snippet (a comment, a bio, a post) meant to be inserted into an existing page. Pass `fragment=True` to parse it without adding `<html>`/`<body>` wrappers.

```python
from justhtml import JustHTML

# Safe for embedding in a <div>
html = JustHTML(user_html, fragment=True).to_html()
# Output: <p>User content...</p>
```

#### Full documents

If you are sanitizing a complete HTML document (e.g. an uploaded file or email), omit `fragment=True`. JustHTML will preserve or add the necessary `<html>`, `<head>`, and `<body>` structure.

```python
# Safe full document
doc = JustHTML(full_html_string).to_html()
# Output: <html><head></head><body>...</body></html>
```

### 2. Safety for JavaScript & dynamic contexts

If you embed HTML or data into `<script>` blocks or event handlers, use escaping for that JavaScript context. HTML cleaning alone does not make the result safe there.

#### Embedding HTML markup in a JS string

If you need to put a block of HTML into a variable (e.g., for a client-side template):

```python
from justhtml import JustHTML, HTMLContext

# Safe for: const myHtml = "${safe}";
safe = JustHTML(user_html, fragment=True).to_html(context=HTMLContext.JS_STRING)
```

#### Embedding text in `innerHTML` via JS

Use `element.textContent = ...` to insert plain text. The browser treats the value as text instead of parsing it as HTML. If you embed that value in a JavaScript string literal, use `escape_js_string()` (see below).

If you **must** assign text to `.innerHTML` (e.g. legacy code), you need to escape HTML entities *and* JS characters:

```python
from justhtml import JustHTML

# Safe for: element.innerHTML = "${safe}";
safe = JustHTML.escape_html_text_in_js_string(user_text)
```

#### Embedding plain strings in JS

If the string is just data (not HTML):

```python
from justhtml import JustHTML

# Safe for: const label = "${safe}";
safe = JustHTML.escape_js_string(user_text)
```

#### Embedding URLs in JS

If you are setting `location.href` or similar from untrusted input, you must **clean** the URL (validate scheme/host) before escaping it.

```python
from justhtml import JustHTML, UrlRule

rule = UrlRule(allowed_schemes={"https"})

# Safe for: location.href = "${clean}";
clean = JustHTML.clean_url_in_js_string(value=user_url, url_rule=rule)
```

### 3. Safety for HTML attributes

When inserting data into HTML attributes (like `title="..."` or `class="..."`), you must escape double quotes to prevent the value from closing the attribute early.

```python
from justhtml import JustHTML

# Safe for: <input value="${safe}">
safe = JustHTML.escape_attr_value(user_input)
```

#### URLs in HTML attributes

If the attribute expects a URL, validate it before escaping. This applies to attributes such as `href`, `src`, and `action`. Escaping alone cannot block `javascript:alert(1)`, which contains no quotes.

```python
from justhtml import JustHTML, UrlRule

rule = UrlRule(allowed_schemes={"https", "mailto"})
cleaned_url = JustHTML.clean_url_value(value=user_url, url_rule=rule)

if cleaned_url:
    # cleaned_url is a validated, percent-encoded URL, not an HTML-attribute-safe
    # string: it deliberately keeps "&" literal so multi-parameter query strings
    # stay usable as plain text (e.g. in a JS string). Escape it like any other
    # attribute value before embedding it in HTML.
    safe_url = JustHTML.escape_attr_value(cleaned_url)
    # <a href="${safe_url}">
```

## Configuration

JustHTML is secure by default, but you can customize the allowlist to fit your application's needs.

*   **[HTML Cleaning](html-cleaning.md)**: Configure allowed tags, attributes, and CSS styles.
*   **[URL Cleaning](url-cleaning.md)**: Configure allowed URL schemes (e.g. `mailto:`, `tel:`) and hosts.
