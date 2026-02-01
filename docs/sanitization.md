[← Back to docs](index.md)

# Sanitization & Security

**JustHTML includes a built-in HTML sanitizer.** When you parse HTML with `JustHTML(html)`, it is sanitized by default according to a [strict allowlist policy](#configuration). You do not need a separate library.

## Threat Model

JustHTML protects against:
- **XSS**: Script execution via tags (`<script>`), attributes (`onload`), and schemes (`javascript:`).
- **Markup Injection**: Breaking out of intended HTML structure.
- **Remote url requests**: Only URL:s that are allowed are let through.

The sanitizer is validated against [`justhtml-xss-bench`](https://github.com/EmilStenstrom/justhtml-xss-bench/) (7,000+ real-world XSS vectors). The benchmark can be used to compare JustHTML's output against  established sanitizers like nh3 and bleach.

## Important: Context is King

Preventing Cross-Site Scripting (XSS) requires more than just cleaning HTML tags. Security depends entirely on **context**. A string that is safe to display in a `<div>` might be dangerous inside a `<script>` tag or an HTML attribute.

Therefore, you must choose the correct output method when embedding data in contexts other than standard HTML.

1. **[HTML Body](#1-safety-for-standard-html)**: `JustHTML(html, fragment=True).to_html()` for [snippets](#fragments-snippets), or `JustHTML(html).to_html()` for [full page documents](#full-documents). (Safe by default)
2. **[JavaScript](#2-safety-for-javascript--dynamic-contexts)**: Use `JustHTML.escape_js_string()` for strings, `clean_url_in_js_string()` for URLs.
3. **[HTML Attributes](#3-safety-for-html-attributes)**: Use `JustHTML.escape_attr_value()` for values, `clean_url_value()` for URLs.

### 1. Safety for Standard HTML

If you are rendering HTML into the body of a page (server-side rendering), JustHTML safeguards you automatically. By default, `JustHTML(...)` sanitizes input using a conservative policy (strips scripts, allowlists safe tags).

#### Fragments (Snippets)

Most real-world untrusted HTML is a snippet (a comment, a bio, a post) meant to be inserted into an existing page. Pass `fragment=True` to parse it without adding `<html>`/`<body>` wrappers.

```python
from justhtml import JustHTML

# Safe for embedding in a <div>
html = JustHTML(user_html, fragment=True).to_html()
# Output: <p>User content...</p>
```

#### Full Documents

If you are sanitizing a complete HTML document (e.g. an uploaded file or email), omit `fragment=True`. JustHTML will preserve or add the necessary `<html>`, `<head>`, and `<body>` structure.

```python
# Safe full document
doc = JustHTML(full_html_string).to_html()
# Output: <html><head></head><body>...</body></html>
```

### 2. Safety for JavaScript & Dynamic Contexts

This is where most XSS vulnerabilities occur. If you embed HTML or data into `<script>` blocks or event handlers, simple HTML cleaning is **not enough**. You need specialized escaping.

#### Embedding HTML markup in a JS string

If you need to put a block of HTML into a variable (e.g., for a client-side template):

```python
from justhtml import JustHTML, HTMLContext

# Safe for: const myHtml = "${safe}";
safe = JustHTML(user_html, fragment=True).to_html(context=HTMLContext.JS_STRING)
```

#### Embedding Text in `innerHTML` via JS

**Recommendation**: Use `element.textContent = ...` instead. It safer and handles escaping automatically. You only need `escape_js_string()` for that (see below).

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

### 3. Safety for HTML Attributes

When inserting data into HTML attributes (like `title="..."` or `class="..."`), you must escape double quotes to prevent the value from closing the attribute early.

```python
from justhtml import JustHTML

# Safe for: <input value="${safe}">
safe = JustHTML.escape_attr_value(user_input)
```

#### URLs in HTML Attributes

If the attribute expects a URL (like `href`, `src`, or `action`), simple escaping is safer than nothing but often insufficient (e.g. `javascript:alert(1)` contains no quotes). You must **validate** the URL first.

```python
from justhtml import JustHTML, UrlRule

rule = UrlRule(allowed_schemes={"https", "mailto"})
cleaned_url = JustHTML.clean_url_value(value=user_url, url_rule=rule)

if cleaned_url:
    # URL is safe to use.
    # Note: clean_url_value performs URL encoding, so no extra escape needed if using double quotes in HTML.
    # <a href="${cleaned_url}">
    pass
```

## Configuration

JustHTML is secure by default, but you can customize the allowlist to fit your application's needs.

*   **[HTML Cleaning](html-cleaning.md)**: Configure allowed tags, attributes, and CSS styles.
*   **[URL Cleaning](url-cleaning.md)**: Configure allowed URL schemes (e.g. `mailto:`, `tel:`) and hosts.
