[← Back to docs](index.md)

# Quickstart

Get up and running with JustHTML in 5 minutes.

## Installation

```bash
pip install justhtml
```

## Basic Parsing

```python
from justhtml import JustHTML

html = "<html><body><div id='main'><p>Hello, <b>world</b>!</p></div></body></html>"
doc = JustHTML(html)
```

## HTML Snippets (Fragments)

If your input is an HTML **snippet** (like user generated content from a WYSIWYG editor), you usually want fragment parsing to avoid implicit `<html>`, `<head>`, and `<body>` insertion:

```python
from justhtml import JustHTML

snippet = "<p>Hello <b>world</b></p>"
doc = JustHTML(snippet, fragment=True)

print(doc.to_html())
```

Output:

```html
<p>Hello <b>world</b></p>
```

## Parsing Bytes (Encoding Sniffing)

If you pass bytes (for example from a file), JustHTML decodes them using HTML encoding sniffing. If no encoding is found, it falls back to `windows-1252` for browser compatibility.

```python
from justhtml import JustHTML
from pathlib import Path

data = Path("page.html").read_bytes()
doc = JustHTML(data)
print(doc.encoding)
```

See [Encoding & Byte Input](encoding.md) for details and how to override with `encoding=...`.

## Traversing the Tree

The parser returns a tree of `Node` objects:

```python
from justhtml import JustHTML

html = "<html><body><div id='main'><p>Hello, <b>world</b>!</p></div></body></html>"
doc = JustHTML(html)

root = doc.root              # #document
html_node = root.children[0] # <html>
body = html_node.children[1] # <body> (children[0] is <head>)
div = body.children[0]       # <div>

# Each node has:
print(div.name)                                # "div"
print(div.attrs)                               # {"id": "main"}
print([child.name for child in div.children])  # ['p']
print(div.parent.name)                         # body
```

## Querying with CSS Selectors

Use familiar CSS syntax to find elements:

```python
# Find all paragraphs
paragraphs = doc.query("p")

# Find by ID
main_div = doc.query("#main")[0]

# Complex selectors
links = doc.query("nav > ul li a.active")

# Multiple selectors
headings = doc.query("h1, h2, h3")
```

## Serializing to HTML

Convert any node back to HTML:

```python
from justhtml import JustHTML

html = "<html><body><div id='main'><p>Hello, <b>world</b>!</p></div></body></html>"
doc = JustHTML(html)
div = doc.query("#main")[0]

print(div.to_html(indent_size=4))
```

Output:

```html
<div id="main">
    <p>Hello, <b>world</b>!</p>
</div>
```

## Strict Mode

Reject malformed HTML instead of silently fixing it:

```python
from justhtml import JustHTML

JustHTML("<!doctype html><p></div>", strict=True)  # doctest: skip
```

Output:

```text
Traceback (most recent call last):
    File "snippet.py", line 3, in <module>
        JustHTML("<!doctype html><p></div>", strict=True)
        ~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    File "parser.py", line 127, in __init__
        raise StrictModeError(self.errors[0])

    File "<html>", line 1
        <!doctype html><p></div>
                          ^^^^^^
justhtml.parser.StrictModeError: Unexpected </div> end tag
```

## Streaming API

For large files or when you don't need the full DOM:

```python
from justhtml import stream

html = "<p>Hello</p><p>world</p>"

for event, data in stream(html):
    if event == "start":
        tag, attrs = data
        print(f"Start: {tag}")
    elif event == "text":
        print(f"Text: {data}")
    elif event == "end":
        print(f"End: {data}")
```

Output:

```html
Start: p
Text: Hello
End: p
Start: p
Text: world
End: p
```

## Next Steps

- [API Reference](api.md) - Complete API documentation
- [CSS Selectors](selectors.md) - All supported selectors
- [Error Codes](errors.md) - Understanding parse errors
