[← Back to docs](index.md)

# Fragment Parsing

Parse HTML fragments as if they were inserted into a specific context element. This is essential for WYSIWYG editors, sanitizers, and template systems.

## Why Fragment Parsing?

HTML parsing rules depend on context. The same markup can produce different results depending on where it appears:

```python
from justhtml import JustHTML

# "<tr>" at document level gets moved outside tables
doc = JustHTML("<tr><td>cell</td></tr>")
print(doc.to_html(indent_size=4))
```

Output:

```html
<html>
    <head></head>
    <body>cell</body>
</html>
```

But if we tell the parser this HTML will be inserted into a `<tbody>`:

```python
from justhtml import JustHTML
from justhtml.context import FragmentContext

html = "<tr><td>cell</td></tr>"
ctx = FragmentContext("tbody")
doc = JustHTML(html, fragment_context=ctx)
print(doc.to_html(indent_size=4))
```

Output:

```html
<tr>
    <td>cell</td>
</tr>
```

## Basic Usage

```python
from justhtml import JustHTML

# Parse as if inside a <div> (common for WYSIWYG editors)
html = "<p>User's <b>content</b></p>"
doc = JustHTML(html, fragment=True)

# The root is #document-fragment, not #document
print(doc.root.name)  # "#document-fragment"

# No implicit <html>, <head>, or <body> are inserted
print(doc.to_html())

# Query and serialize work normally
paragraphs = doc.query("p")
```

Output:

```html
#document-fragment
<p>User's <b>content</b></p>
```

## Common Use Cases

### WYSIWYG Editor Content

When users edit HTML in a rich text editor, the content will typically be inserted into a container like `<div>` or `<article>`:

```python
# User's editor content
user_html = "<p>Hello</p><ul><li>Item 1</li><li>Item 2</li></ul>"

# Parse as fragment inside a div
doc = JustHTML(user_html, fragment=True)

# Sanitize, transform, or validate...
clean_html = doc.to_html()
```

### Table Cell Content

Parsing content that will go inside a table cell:

```python
cell_content = "Price: <b>$99</b>"
ctx = FragmentContext("td")
doc = JustHTML(cell_content, fragment_context=ctx)
```

### List Item Content

```python
item_html = "Buy <a href='/milk'>milk</a>"
ctx = FragmentContext("li")
doc = JustHTML(item_html, fragment_context=ctx)
```

### Select Options

```python
options_html = "<option>Red</option><option>Blue</option>"
ctx = FragmentContext("select")
doc = JustHTML(options_html, fragment_context=ctx)
```

## Context Elements

The context element affects parsing rules:

| Context | Use Case |
|---------|----------|
| `div`, `article`, `section` | General HTML content |
| `tbody`, `thead`, `tfoot` | Table rows |
| `tr` | Table cells |
| `td`, `th` | Cell content |
| `ul`, `ol` | List items |
| `select` | Option elements |
| `textarea` | Raw text (no HTML parsing) |
| `title` | Raw text (no HTML parsing) |

### Raw Text Contexts

Some elements treat their content as raw text:

```python
from justhtml import JustHTML
from justhtml.context import FragmentContext

# Content in <textarea> is not parsed as HTML
ctx = FragmentContext("textarea")
doc = JustHTML("<b>not bold</b>", fragment_context=ctx)
print(doc.to_html())
```

Output:

```html
&lt;b&gt;not bold&lt;/b&gt;
```

## SVG and MathML Fragments

Parse fragments in foreign namespaces:

```python
# SVG fragment
svg_content = "<circle cx='50' cy='50' r='40'/>"
ctx = FragmentContext("svg", namespace="svg")
doc = JustHTML(svg_content, fragment_context=ctx)

# MathML fragment
math_content = "<mi>x</mi><mo>=</mo><mn>5</mn>"
ctx = FragmentContext("math", namespace="math")
doc = JustHTML(math_content, fragment_context=ctx)
```

## FragmentContext API

```python
from justhtml.context import FragmentContext

# HTML namespace (default)
ctx = FragmentContext("div")
ctx = FragmentContext("div", namespace=None)

# SVG namespace
ctx = FragmentContext("svg", namespace="svg")

# MathML namespace
ctx = FragmentContext("math", namespace="math")
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `tag_name` | `str` | The context element's tag name |
| `namespace` | `str \| None` | `None` for HTML, `"svg"` for SVG, `"math"` for MathML |

## Document vs Fragment

| Feature | `JustHTML(html)` | `JustHTML(html, fragment_context=ctx)` |
|---------|------------------|----------------------------------------|
| Root node | `#document` | `#document-fragment` |
| Implicit elements | `<html>`, `<head>`, `<body>` auto-inserted | None - only your content |
| Children | Full document structure | Just the parsed content |
| Use case | Complete HTML documents | Partial HTML snippets |

## Example: HTML Sanitizer

```python
from justhtml import JustHTML
from justhtml.context import FragmentContext

from justhtml import Sanitize, SanitizationPolicy, UrlPolicy, UrlRule

def sanitize_fragment(html: str) -> str:
    """Sanitize user HTML as if it was inserted into a <div>."""
    ctx = FragmentContext("div")
    policy = SanitizationPolicy(
        allowed_tags={"p", "b", "i", "a", "ul", "ol", "li"},
        allowed_attributes={"*": [], "a": ["href"]},
        url_policy=UrlPolicy(
            default_allow_relative=True,
            allow_rules={("a", "href"): UrlRule(allowed_schemes=["https"])},
        ),
    )

    # Sanitize the in-memory DOM by applying a Sanitize transform.
    # Put Sanitize at the end if you want a sanitized DOM.
    doc = JustHTML(html, fragment_context=ctx, transforms=[Sanitize(policy)])
    return doc.to_html(pretty=False)

# Usage
dirty = '<p>Hello</p><script>alert("xss")</script><b>world</b>'
print(sanitize_fragment(dirty))
```

Output:

```html
<p>Hello</p><b>world</b>
```
