[← Back to docs](index.md)

# HTML Cleaning

JustHTML includes a built-in, **policy-driven HTML sanitizer** intended for rendering *untrusted HTML safely*.

This page focuses on **HTML cleaning**: tags, attributes, and inline styles. For URL validation and rewriting, see [URL Cleaning](url-cleaning.md).

On this page:

- [Safe-by-default construction](#safe-by-default-construction)
- [Default policy](#default-sanitization-policy)
- [Custom policy](#use-a-custom-sanitization-policy)
- [Sanitizing a DOM directly](#sanitizing-a-dom-directly)
- [Inline styles](#inline-styles-optional)
- [Advanced policy options](#advanced-policy-options)
- [Disable sanitization](#disable-sanitization)
- [Reporting issues](#reporting-issues)

## Safe-by-default construction

By default, construction removes all dangerous html:

### HTML output

```python
from justhtml import JustHTML

user_html = '<p>Hello <b>world</b> <script>alert(1)</script> <a href="javascript:alert(1)">bad</a> <a href="https://example.com/?a=1&b=2">ok</a></p>'
doc = JustHTML(user_html, fragment=True)
print(doc.to_html())
```

<!-- justhtml: output -->

```html
<p>Hello <b>world</b>  <a>bad</a> <a href="https://example.com/?a=1&amp;b=2">ok</a></p>
```

### Markdown output

```python
from justhtml import JustHTML

user_html = '<p>Hello <b>world</b> <script>alert(1)</script> <a href="javascript:alert(1)">bad</a> <a href="https://example.com/?a=1&b=2">ok</a></p>'
doc = JustHTML(user_html, fragment=True)
print(doc.to_markdown())
```

<!-- justhtml: output -->

```markdown
Hello **world** [bad] [ok](https://example.com/?a=1&b=2)
```

## Default sanitization policy

The built-in default is `DEFAULT_POLICY` (a conservative allowlist).

The default URL policy is conservative about remote loads: by default `a[href]` allows common link schemes, while `img[src]` only allows relative URLs (so images won't load from remote hosts unless you opt in via a custom policy). For details, see [URL Cleaning](url-cleaning.md).

High-level behavior:

- Disallowed tags are stripped (their children may be kept) but dangerous containers like `script`/`style` have their content dropped.
- Comments and doctypes are dropped.
- Foreign namespaces (SVG/MathML) are always dropped by the sanitizer, even if the tags are included in a custom allowlist. Parsing and treebuilding still support foreign content when `sanitize=False`.
- Invisible Unicode commonly used for obfuscation, including variation selectors, zero-width/bidi controls, and private-use characters, is stripped from text and attributes before other sanitizer checks run.
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

## Use a custom sanitization policy

Start with the smallest policy that matches the HTML you want to accept. This makes the allowed output clear to future maintainers.

```python
from justhtml import JustHTML, SanitizationPolicy, UrlPolicy, UrlRule

policy = SanitizationPolicy(
    allowed_tags={"p", "strong", "a"},
    allowed_attributes={"a": {"href"}},
    url_policy=UrlPolicy(
        allow_rules={
            ("a", "href"): UrlRule(allowed_schemes={"https", "mailto"}),
        }
    ),
)

safe_html = JustHTML(user_html, fragment=True, policy=policy).to_html()
```

The three settings work together:

- `allowed_tags` controls which elements remain.
- `allowed_attributes` controls which attribute names remain. An omitted tag has no allowed attributes, so you only need to list tags that accept attributes.
- URL-valued attributes such as `href` also need a `UrlRule`; allowing the attribute name alone is not enough.

For URL schemes, hosts, proxies, `srcset`, and other URL-specific controls, continue with [URL Cleaning](url-cleaning.md).

## Sanitizing the in-memory DOM

The parsed DOM is sanitized by default at construction time (`JustHTML(..., sanitize=True)`), and serialization is a pure output step.

If you want to sanitize after other transforms or after direct DOM edits, add `Sanitize(...)` at the point where the tree should become safe. Later transforms can reintroduce unsafe content. For explicit pass boundaries (advanced use), see [`Stage([...])`](transforms.md#advanced-stages).

```python
from justhtml import JustHTML, Sanitize

doc = JustHTML(user_html, fragment=True, transforms=[Sanitize()])
print(doc.to_html(pretty=False))
```

## Inline styles (optional)

Inline styles are disabled by default. To allow them you must:

1. Allow the `style` attribute for the relevant tag via `allowed_attributes`.
2. Provide a non-empty allowlist via `allowed_css_properties`.

Even then, JustHTML rejects declarations that look like they can load external resources (such as values containing `url(` or `image-set(`), as well as legacy constructs like `expression(`. Start from the conservative `CSS_PRESET_TEXT` preset.

```python
from justhtml import CSS_PRESET_TEXT, JustHTML, SanitizationPolicy, UrlPolicy

policy = SanitizationPolicy(
    allowed_tags={"p"},
    allowed_attributes={"p": {"style"}},
    url_policy=UrlPolicy(allow_rules={}),
    allowed_css_properties=CSS_PRESET_TEXT | {"width"},
)

html = '<p style="color: red; background-image: url(https://evil.test/x); width: expression(alert(1));">Hi</p>'
print(JustHTML(html, policy=policy).to_html())
```

## Advanced policy options

Selector limits are for trusted pipelines that need to accept unusually large selectors or match budgets:

```python
from justhtml import SanitizationPolicy
from justhtml.selector import SelectorLimits

policy = SanitizationPolicy(
    allowed_tags={"div", "p"},
    allowed_attributes={"*": {"class"}},
    selector_limits=SelectorLimits(max_length=20_000),
)
```

Treat policies that allow active content as a separate security review: `iframe`, `object`, `embed`, `meta`, `link`, `base`, form elements, and their active attributes are preserved when you explicitly allow them.

## Disable sanitization

Only disable sanitization for HTML you fully trust:

```python
doc = JustHTML(trusted_html, fragment=True, sanitize=False)
```

## Reporting issues

If you find a sanitizer bypass, please report it responsibly (see [SECURITY.md](../SECURITY.md)).
