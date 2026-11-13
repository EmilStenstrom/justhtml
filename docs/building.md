[← Back to docs](index.md)

# Building HTML

Build HTML programmatically with explicit node factories, then normalize the
result through `JustHTML(...)`.

Use the builder to assemble HTML from Python values, conditions, loops, and
helper functions.

For mostly static markup, use plain HTML strings:

```python
from justhtml import JustHTML

doc = JustHTML("""
<p>Hello <strong>world</strong></p>
""", fragment=True)
```

<a id="the-model"></a>

## Node construction

The builder module creates nodes directly:

```python
from justhtml.dom.builder import comment, doctype, element, text
```

`JustHTML(...)` parses and normalizes the built tree:

1. Build a node tree.
2. Pass the tree to `JustHTML(...)`.
3. Use the normalized tree returned by the HTML5 parser.

```python
from justhtml import JustHTML
from justhtml.dom.builder import element

doc = JustHTML(element("p", "Hello"), fragment=True)
doc.to_html(pretty=False)  # => <p>Hello</p>
```

<a id="start-small"></a>

## Create an element

The core factory is `element()`:

```python
from justhtml.dom.builder import element

node = element("p", "Hello")
node.to_html(pretty=False)  # => <p>Hello</p>
```

That creates an element node with one text child.

You can pass attributes as a dict:

```python
from justhtml.dom.builder import element

link = element("a", {"href": "/docs"}, "Read docs")
link.to_html(pretty=False)  # => <a href="/docs">Read docs</a>
```

And you can nest elements directly:

```python
from justhtml.dom.builder import element

card = element(
    "article",
    {"class": "post"},
    element("h2", "JustHTML"),
    element("p", "Build nodes directly."),
    element("a", {"href": "/docs"}, "Read docs"),
)

card.to_html(pretty=False)  # => <article class="post"><h2>JustHTML</h2><p>Build nodes directly.</p><a href="/docs">Read docs</a></article>
```

## Text, comments, and doctype

Use the dedicated factories when you want those exact node types:

```python
from justhtml.dom.builder import comment, doctype, element, text

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
from justhtml.dom.builder import doctype

dt = doctype()
dt.to_html(pretty=False)  # => <!DOCTYPE html>
```

You can also provide explicit identifiers:

```python
from justhtml.dom.builder import doctype

dt = doctype(
    "html",
    public_id="-//W3C//DTD HTML 4.01//EN",
    system_id="http://www.w3.org/TR/html4/strict.dtd",
)

dt.to_html(pretty=False)  # => <!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">
```

## Parse the built nodes

Pass the built node to `JustHTML(...)` to parse and normalize it.

### Fragment mode

```python
from justhtml import JustHTML
from justhtml.dom.builder import element

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
from justhtml.dom.builder import element

doc = JustHTML(
    element(
        "html",
        element("head", element("title", "Example")),
        element("body", element("p", "Hello")),
    )
)

doc.to_html(pretty=False)  # => <html><head><title>Example</title></head><body><p>Hello</p></body></html>
```

`JustHTML(...)` inserts missing `html`, `head`, and `body` elements. It adds
a doctype only when the input includes one, so this example has no doctype.

If you want a doctype, provide one explicitly:

```python
from justhtml import JustHTML
from justhtml.dom.builder import doctype, element
from justhtml.dom import Document

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

HTML5 parsing rules determine the final structure, including recovery from
misnested or incomplete input trees.

## Build dynamic content

Use conditions, loops, and helper functions to choose which nodes to create.

### Conditions

`None` and `False` are ignored in child positions.

```python
from justhtml.dom.builder import element

is_admin = True

header = element(
    "header",
    element("h1", "Dashboard"),
    is_admin and element("a", {"href": "/admin"}, "Admin"),
)

print(header.to_html(pretty=False))
# => <header><h1>Dashboard</h1><a href="/admin">Admin</a></header>
```

If `is_admin = False`, the builder omits the link because it ignores `False`
in child positions.

```html
<header><h1>Dashboard</h1></header>
```

### Loops

Iterables of child values are flattened.

```python
from justhtml.dom.builder import element

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
from justhtml.dom.builder import element

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

<a id="attribute-dicts-first"></a>

## Attribute dictionaries

The explicit attrs dict is the canonical form:

```python
from justhtml.dom.builder import element

element("a", {"href": "/docs", "target": "_blank"}, "Docs").to_html(pretty=False)
# => <a href="/docs" target="_blank">Docs</a>
```

This is usually the clearest style, especially when values are long or contain
special characters.

The builder also allows a restricted shorthand in the tag name:

```python
from justhtml.dom.builder import element

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

Use shorthand for short attribute lists. Use an attrs dict for longer lists
or values that contain special characters.

<a id="what-gets-coerced"></a>

## Child and attribute value conversion

The builder accepts these child values:

- strings become text nodes
- iterables are flattened
- `None` and `False` are ignored
- numbers are rejected as children

Example:

```python
from justhtml.dom.builder import element

element("p", "Hello ", element("strong", "world")).to_html(pretty=False)
# => <p>Hello <strong>world</strong></p>
```

And numeric child values are rejected:

```python
from justhtml.dom.builder import element

element("p", 1)
```

This raises `TypeError`.

For attribute values:

- `None` means a present boolean attribute
- other values are converted to strings

```python
from justhtml.dom.builder import element

element("input", {"maxlength": 10, "required": None}).to_html(pretty=False)
# => <input maxlength="10" required>
```

## Template elements

`element("template", ...)` creates a template node.

The builder stores `<template>...</template>` children in `template_content`.

```python
from justhtml.dom.builder import element

node = element(
    "template",
    element("p", "Hello from a template"),
)

node.to_html(pretty=False)  # => <template><p>Hello from a template</p></template>
```

<a id="fragment-context-still-belongs-to-justhtml"></a>

## Fragment parsing context

Set the fragment parsing context on `JustHTML(...)` when normalizing built nodes.
The builder creates the nodes without choosing a parsing context.

```python
from justhtml import JustHTML
from justhtml.dom.builder import element
from justhtml.parser.context import FragmentContext

row = element("tr", element("td", "cell"))

doc = JustHTML(
    row,
    fragment=True,
    fragment_context=FragmentContext("tbody"),
)

doc.to_html(pretty=False)  # => <tr><td>cell</td></tr>
```

Only HTML5 namespaces are accepted by the builder: HTML, SVG, and MathML.

Foreign namespaces are preserved when normal HTML5 parsing can reconstruct them,
such as content inside `<svg>` or `<math>`. Arbitrary custom namespaces are
rejected by `element()` instead of being silently lost during normalization.

## Common patterns

### Build a fragment from one top-level node

```python
from justhtml import JustHTML
from justhtml.dom.builder import element

doc = JustHTML(element("p", "Hello"), fragment=True)
doc.to_html(pretty=False)  # => <p>Hello</p>
```

### Build a whole page

```python
from justhtml import JustHTML
from justhtml.dom.builder import element

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
from justhtml.dom.builder import element

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

### Clone from template DOMs before appending

DOM insertion methods adopt existing nodes. If you append a node that already
belongs to another tree, it is removed from its old parent and moved into the new
tree.

When you use one parsed DOM as a template for another document, clone the source
node before appending it:

```python
from justhtml import JustHTML
from justhtml.dom.builder import element

template = JustHTML('<section><p class="lead">Hello</p></section>', fragment=True, sanitize=False)
output = JustHTML(element("main"), fragment=True, sanitize=False)

lead = template.query_one("p.lead")
assert lead is not None
main = output.query_one("main")
assert main is not None

main.append_child(lead.clone_node(deep=True))

template.to_html(pretty=False)  # => <section><p class="lead">Hello</p></section>
output.to_html(pretty=False)  # => <main><p class="lead">Hello</p></main>
```

<a id="when-not-to-use-the-builder"></a>

## Static HTML strings

Prefer plain HTML strings when:

- the markup is mostly static
- you want the code to look exactly like HTML
- there is little or no conditional logic

## Next steps

- [API Reference](api.md) - Full builder and parser reference
- [Fragment Parsing](fragments.md) - How fragment context affects parsing
- [Transforms](transforms.md) - Modify the normalized DOM after parsing
