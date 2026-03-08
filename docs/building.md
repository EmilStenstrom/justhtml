[← Back to docs](index.md)

# Building HTML

Build HTML programmatically with explicit node factories, then normalize the
result through `JustHTML(...)`.

This guide is for the case where HTML is assembled from Python values,
conditions, loops, and helper functions.

If your markup is already known and mostly static, plain HTML strings are still
the simpler API:

```python
from justhtml import JustHTML

doc = JustHTML("""
<p>Hello <strong>world</strong></p>
""", fragment=True)
```

Use the builder when you want to construct HTML as data rather than hand-write
HTML text.

## The Model

The builder module creates nodes directly:

```python
from justhtml.builder import comment, doctype, element, text
```

`JustHTML(...)` is still the parser and normalizer.

That means the usual flow is:

1. build an attempted node tree
2. pass that node tree to `JustHTML(...)`
3. let the HTML5 parser normalize it

```python
from justhtml import JustHTML
from justhtml.builder import element

doc = JustHTML(element("p", "Hello"), fragment=True)
doc.to_html(pretty=False)  # => <p>Hello</p>
```

## Start Small

The core factory is `element()`:

```python
from justhtml.builder import element

node = element("p", "Hello")
node.to_html(pretty=False)  # => <p>Hello</p>
```

That creates an element node with one text child.

You can pass attributes as a dict:

```python
from justhtml.builder import element

link = element("a", {"href": "/docs"}, "Read docs")
link.to_html(pretty=False)  # => <a href="/docs">Read docs</a>
```

And you can nest elements directly:

```python
from justhtml.builder import element

card = element(
    "article",
    {"class": "post"},
    element("h2", "JustHTML"),
    element("p", "Build nodes directly."),
    element("a", {"href": "/docs"}, "Read docs"),
)

card.to_html(pretty=False)  # => <article class="post"><h2>JustHTML</h2><p>Build nodes directly.</p><a href="/docs">Read docs</a></article>
```

## Text, Comments, and Doctype

Use the dedicated factories when you want those exact node types:

```python
from justhtml.builder import comment, doctype, element, text

html = element(
    "html",
    element("body",
        comment("page content starts here"),
        element("p", text("Hello")),
    ),
)

html.to_html(pretty=False)  # => <html><body><!--page content starts here--><p>Hello</p></body></html>
```

For a document doctype:

```python
from justhtml.builder import doctype

dt = doctype()
dt.to_html(pretty=False)  # => <!DOCTYPE html>
```

You can also provide explicit identifiers:

```python
from justhtml.builder import doctype

dt = doctype(
    "html",
    public_id="-//W3C//DTD HTML 4.01//EN",
    system_id="http://www.w3.org/TR/html4/strict.dtd",
)

dt.to_html(pretty=False)  # => <!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">
```

## Parse the Built Nodes

Once you have a built node, hand it to `JustHTML(...)`.

### Fragment mode

```python
from justhtml import JustHTML
from justhtml.builder import element

doc = JustHTML(
    element("li", "One"),
    fragment=True,
)

doc.root.name  # => #document-fragment
doc.to_html(pretty=False)  # => <li>One</li>
```

### Full document mode

```python
from justhtml import JustHTML
from justhtml.builder import element

doc = JustHTML(
    element(
        "html",
        element("head", element("title", "Example")),
        element("body", element("p", "Hello")),
    )
)

doc.to_html(pretty=False)  # => <html><head><title>Example</title></head><body><p>Hello</p></body></html>
```

This example has no doctype because `JustHTML(...)` follows normal parser
behavior: it will insert missing `html`, `head`, and `body` structure, but it
will not invent a doctype token that was never provided.

If you want a doctype, provide one explicitly:

```python
from justhtml import JustHTML
from justhtml.builder import doctype, element
from justhtml.node import Document

document = Document()
document.append_child(doctype())
document.append_child(
    element(
        "html",
        element("head", element("title", "Example")),
        element("body", element("p", "Hello")),
    )
)

doc = JustHTML(document)
doc.to_html(pretty=False)  # => <!DOCTYPE html><html><head><title>Example</title></head><body><p>Hello</p></body></html>
```

In document mode, the default document sanitization policy preserves doctypes,
so `sanitize=False` is not required here.

Even if the attempted tree is awkward or incomplete, final structure is defined
by normal HTML5 parsing behavior.

## Build Dynamic Content

The builder becomes useful when Python is deciding what HTML to emit.

### Conditions

`None` and `False` are ignored in child positions.

```python
from justhtml.builder import element

is_admin = True

header = element(
    "header",
    element("h1", "Dashboard"),
    is_admin and element("a", {"href": "/admin"}, "Admin"),
)

print(header.to_html(pretty=False))
# => <header><h1>Dashboard</h1><a href="/admin">Admin</a></header>
```

If `is_admin = False`, the link disappears completely because `False` is ignored
in child positions.

```html
<header><h1>Dashboard</h1></header>
```

### Loops

Iterables of child values are flattened.

```python
from justhtml.builder import element

items = ["One", "Two", "Three"]

listing = element(
    "ul",
    (element("li", item) for item in items),
)

print(listing.to_html(pretty=False))
# => <ul><li>One</li><li>Two</li><li>Three</li></ul>
```

### Helper functions

You can return nodes from helper functions and compose them normally.

```python
from justhtml.builder import element

def user_card(user: dict[str, str]):
    return element(
        "article",
        {"class": "user-card"},
        element("h2", user["name"]),
        element("p", user["email"]),
    )


page = element(
    "section",
    user_card({"name": "Ada", "email": "ada@example.com"}),
    user_card({"name": "Linus", "email": "linus@example.com"}),
)

print(page.to_html(pretty=False))
# => <section><article class="user-card"><h2>Ada</h2><p>ada@example.com</p></article><article class="user-card"><h2>Linus</h2><p>linus@example.com</p></article></section>
```

## Attribute Dicts First

The explicit attrs dict is the canonical form:

```python
from justhtml.builder import element

element("a", {"href": "/docs", "target": "_blank"}, "Docs").to_html(pretty=False)
# => <a href="/docs" target="_blank">Docs</a>
```

This is usually the clearest style, especially when values are long or contain
special characters.

The builder also allows a restricted shorthand in the tag name:

```python
from justhtml.builder import element

element("input[type=email][required]").to_html(pretty=False)
# => <input type="email" required>
element("a[href=/docs][target=_blank]", "Docs").to_html(pretty=False)
# => <a href="/docs" target="_blank">Docs</a>
```

Supported forms:

- `[attr]`
- `[attr=value]`
- `[attr="value"]`
- `[attr='value']`

Use the shorthand when it stays short. Switch back to an attrs dict when it
starts to feel like a tiny language.

## What Gets Coerced

Children are intentionally strict.

- strings become text nodes
- iterables are flattened
- `None` and `False` are ignored
- numbers are rejected as children

Example:

```python
from justhtml.builder import element

element("p", "Hello ", element("strong", "world")).to_html(pretty=False)
# => <p>Hello <strong>world</strong></p>
```

And numeric child values are rejected:

```python
from justhtml.builder import element

element("p", 1)
```

This raises `TypeError`.

Attributes are slightly more forgiving.

- `None` means a present boolean attribute
- other values are converted to strings

```python
from justhtml.builder import element

element("input", {"maxlength": 10, "required": None}).to_html(pretty=False)
# => <input maxlength="10" required>
```

## Template Elements

`element("template", ...)` creates a template node.

Its children are written into `template_content`, which matches how users think
about `<template>...</template>` content.

```python
from justhtml.builder import element

node = element(
    "template",
    element("p", "Hello from a template"),
)

node.to_html(pretty=False)  # => <template><p>Hello from a template</p></template>
```

## Fragment Context Still Belongs to JustHTML

The builder does not decide fragment parsing context. `JustHTML(...)` still does.

```python
from justhtml import JustHTML
from justhtml.context import FragmentContext
from justhtml.builder import element

row = element("tr", element("td", "cell"))

doc = JustHTML(
    row,
    fragment=True,
    fragment_context=FragmentContext("tbody"),
)

doc.to_html(pretty=False)  # => <tr><td>cell</td></tr>
```

That keeps parsing behavior in one place.

Only HTML5 namespaces are accepted by the builder: HTML, SVG, and MathML.

Foreign namespaces are preserved when normal HTML5 parsing can reconstruct them,
such as content inside `<svg>` or `<math>`. Arbitrary custom namespaces are
rejected by `element()` instead of being silently lost during normalization.

## Common Patterns

### Build a fragment from one top-level node

```python
from justhtml import JustHTML
from justhtml.builder import element

doc = JustHTML(element("p", "Hello"), fragment=True)
doc.to_html(pretty=False)  # => <p>Hello</p>
```

### Build a whole page

```python
from justhtml import JustHTML
from justhtml.builder import element

doc = JustHTML(
    element(
        "html",
        element("head", element("title", "Page")),
        element("body", element("p", "Hello")),
    )
)

doc.to_html(pretty=False)  # => <html><head><title>Page</title></head><body><p>Hello</p></body></html>
```

### Build first, query later

```python
from justhtml import JustHTML
from justhtml.builder import element

doc = JustHTML(
    element(
        "div",
        element("p", {"class": "lead"}, "Hello"),
        element("p", "World"),
    ),
    fragment=True,
)

lead = doc.query_one("p.lead")
lead.to_html(pretty=False)  # => <p class="lead">Hello</p>
```

## When Not to Use the Builder

Don’t use the builder just because it exists.

Plain HTML is still better when:

- the markup is mostly static
- you want the code to look exactly like HTML
- there is little or no conditional logic

Use the builder when Python structure is already driving the output.

## Next Steps

- [API Reference](api.md) - Full builder and parser reference
- [Fragment Parsing](fragments.md) - How fragment context affects parsing
- [Transforms](transforms.md) - Modify the normalized DOM after parsing
