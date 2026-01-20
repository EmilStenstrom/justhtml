[← Back to docs](index.md)

# HTML Cleaning

JustHTML includes a built-in, **policy-driven HTML sanitizer** intended for rendering *untrusted HTML safely*.

This page focuses on **HTML cleaning**: tags, attributes, and inline styles. For URL validation and rewriting, see [URL Cleaning](url-cleaning.md).

On this page:

- [DOM vs construction](#important-dom-vs-construction)
- [Safe-by-default construction](#safe-by-default-construction)
- [Sanitizing a DOM directly](#sanitizing-a-dom-directly)
- [Disable sanitization](#disable-sanitization)
- [Default policy](#default-sanitization-policy)
- [Inline styles](#inline-styles-optional)
- [Custom policy](#use-a-custom-sanitization-policy)
- [Reporting issues](#reporting-issues)

## Important: DOM vs construction

The parsed DOM is **sanitized by default** at construction time (`JustHTML(..., safe=True)`), and serialization is a pure output step.

If you want to sanitize **after** other transforms or after direct DOM edits, apply the `Sanitize(...)` transform to sanitize the in-memory tree.

## Safe-by-default construction

By default, construction removes all dangerous html:

```python
from justhtml import JustHTML

user_html = '<p>Hello <b>world</b> <script>alert(1)</script> <a href="javascript:alert(1)">bad</a> <a href="https://example.com/?a=1&b=2">ok</a></p>'
doc = JustHTML(user_html, fragment=True)

print(doc.to_html())
print()
print(doc.to_markdown())
```

Output:

```html
<p>Hello <b>world</b>  <a>bad</a> <a href="https://example.com/?a=1&amp;b=2">ok</a></p>

Hello **world** [bad] [ok](https://example.com/?a=1&b=2)
```

## Sanitizing the in-memory DOM

If you will be working with the DOM and want a clean slate to work from, add `Sanitize(...)` to your transform pipeline.

If you want explicit pass boundaries (advanced use), you can group transforms using [`Stage([...])`](transforms.md#advanced-stages).

```python
from justhtml import JustHTML, Sanitize

user_html = '<p>Hello <b>world</b> <script>alert(1)</script> <a href="javascript:alert(1)">bad</a> <a href="https://example.com/?a=1&b=2">ok</a></p>'
doc = JustHTML(user_html, fragment=True, transforms=[Sanitize()])

# The DOM is now sanitized in-memory.
print(doc.root.to_html(pretty=False))
# => <p>Hello <b>world</b>  <a>bad</a> <a href="https://example.com/?a=1&amp;b=2">ok</a></p>
```

## Disable sanitization

If you want to (dangerously) disable sanitization, because you know that your trusted HTML can't contain malicious code:

```python
from justhtml import JustHTML

user_html = '<p>Hello <b>world</b> <script>init_page_view_tracker()</script> <a href="javascript:track_pageview()">ok</a></p>'
doc = JustHTML(user_html, fragment=True, safe=False)

print(doc.to_html())
```

Output:

```html
<p>Hello <b>world</b> <script>init_page_view_tracker()</script> <a href="javascript:track_pageview()">ok</a></p>
```

## Default sanitization policy

The built-in default is `DEFAULT_POLICY` (a conservative allowlist).

The default URL policy is conservative about remote loads: by default `a[href]` allows common link schemes, while `img[src]` only allows relative URLs (so images won't load from remote hosts unless you opt in via a custom policy). For details, see [URL Cleaning](url-cleaning.md).

High-level behavior:

- Disallowed tags are stripped (their children may be kept) but dangerous containers like `script`/`style` have their content dropped.
- Comments and doctypes are dropped.
- Foreign namespaces (SVG/MathML) are dropped.
- Event handlers (`on*`), `srcdoc`, and namespace-style attributes (anything with `:`) are removed.
- Inline styles are disabled by default.

### Disallowed tags

Disallowed tag handling is controlled by `SanitizationPolicy(disallowed_tag_handling=...)`:

- `"unwrap"` (default): remove the disallowed tag, keep/sanitize its children
- `"escape"`: emit the disallowed tag’s start/end tags as escaped text, keep/sanitize its children
- `"drop"`: drop the entire disallowed subtree

Default allowlists:

- Allowed tags: `a`, `img`, common text/structure tags, headings, lists, and tables (`table`, `thead`, `tbody`, `tfoot`, `tr`, `th`, `td`).
- Allowed attributes:
  - Global: `class`, `id`, `title`, `lang`, `dir`
  - `a`: `href`, `title`
  - `img`: `src`, `alt`, `title`, `width`, `height`, `loading`, `decoding`
  - `th`/`td`: `colspan`, `rowspan`

## Inline styles (optional)

Inline styles are disabled by default. To allow them you must:

1) Allow the `style` attribute for the relevant tag via `allowed_attributes`, and
2) Provide a non-empty allowlist via `allowed_css_properties`.

Even then, JustHTML is conservative: it rejects declarations that look like they can load external resources (such as values containing `url(` or `image-set(`), as well as legacy constructs like `expression(`.

To avoid "footgun" policies, you can start from the built-in preset `CSS_PRESET_TEXT`.

```python
from justhtml import CSS_PRESET_TEXT, JustHTML, SanitizationPolicy, UrlPolicy

policy = SanitizationPolicy(
    allowed_tags=["p"],
    allowed_attributes={"*": [], "p": ["style"]},
    url_policy=UrlPolicy(allow_rules={}),
    allowed_css_properties=CSS_PRESET_TEXT | {"width"},
)

html = '<p style="color: red; background-image: url(https://evil.test/x); width: expression(alert(1));">Hi</p>'
print(JustHTML(html, policy=policy).to_html())
```

Output:

```html
<p style="color: red">Hi</p>

```

## Use a custom sanitization policy

You are encouraged to write your own `SanitizationPolicy`, and not rely on the default one. This makes it easier for future developers to understand what's being cleaned, without having to look it up in justhtml's documentation.

When expanding the default policy, prefer adding small, explicit allowlists.

Treat these as a separate security review if you plan to allow them:

- `iframe`, `object`, `embed`
- `meta`, `link`, `base`
- form elements and submission-related attributes

For URL-related risks and controls, see [URL Cleaning](url-cleaning.md).

```python
from justhtml import JustHTML, SanitizationPolicy, UrlPolicy, UrlRule

user_html = '<p>Hello <b>world</b> <script>alert(1)</script> <a href="javascript:alert(1)">bad</a> <a href="https://example.com/?a=1&b=2">ok</a></p>'

policy = SanitizationPolicy(
    allowed_tags=["p", "b", "a"],
    allowed_attributes={"*": [], "a": ["href"]},
    url_policy=UrlPolicy(
        default_handling="strip",
        allow_rules={
            ("a", "href"): UrlRule(allowed_schemes=["https"]),
        }
    ),
)

doc = JustHTML(user_html, fragment=True)
doc = JustHTML(user_html, fragment=True, policy=policy)
print(doc.to_html())
```

Output:

```html
<p>Hello <b>world</b>  <a>bad</a> <a href="https://example.com/?a=1&amp;b=2">ok</a></p>
```

## Reporting issues

If you find a sanitizer bypass, please report it responsibly (see [SECURITY.md](../SECURITY.md)).
