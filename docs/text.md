[← Back to docs](index.md)

# Extracting Text

JustHTML gives you a few ways to get text out of a parsed document, depending on whether you want a fast concatenation, or something structured.

## 1) `to_text()` (concatenated text)

Use `to_text()` when you want the concatenated text from a whole subtree:

- Traverses descendants.
- Joins text nodes using `separator` (default: a single space).
- Strips each text node by default (`strip=True`) and drops empty segments.
- Includes `<template>` contents (via `template_content`).
- Sanitizes untrusted HTML by default (safe-by-default at construction).

```python
from justhtml import JustHTML

doc = JustHTML("<div><h1>Title</h1><p>Hello <b>world</b></p></div>", fragment=True)
print(doc.to_text())
# => Title Hello world
```

```python
from justhtml import JustHTML

untrusted = JustHTML("<p>Hello<script>alert(1)</script>World</p>", fragment=True)
print(untrusted.to_text())
# => Hello World
```

```python
from justhtml import JustHTML

untrusted = JustHTML("<p>Hello<script>alert(1)</script>World</p>", fragment=True, safe=False)
print(untrusted.to_text())
# => Hello alert(1) World
```

```python
from justhtml import JustHTML

doc = JustHTML("<p>Hello <b>world</b></p>", fragment=True)
print(doc.root.to_text(separator="", strip=False))
# => Hello world
```

The default `separator=" "` avoids accidentally smashing words together when the HTML splits text across nodes:

```python
from justhtml import JustHTML

doc = JustHTML("<p>Hello<b>world</b></p>")

print(doc.to_text())
print(doc.to_text(separator="", strip=True))
# => Hello world
# => Helloworld
```

## 2) `to_markdown()` (GitHub Flavored Markdown)

`to_markdown()` outputs a pragmatic subset of GitHub Flavored Markdown (GFM) that aims to be readable and stable for common HTML.

- Converts common elements like headings, paragraphs, lists, emphasis, links, and code.
- Keeps tables (`<table>`) and images (`<img>`) as raw HTML.
- Drops `<script>`, `<style>`, and `<textarea>` by default; pass
  `html_passthrough=True` to include them and their contents.

```python
from justhtml import JustHTML

doc = JustHTML("<h1>Title</h1><p>Hello <b>world</b></p>")
print(doc.to_markdown())
# => # Title
# =>
# => Hello **world**
```

Example:

```python
from justhtml import JustHTML

html = """
<div>
  <h1>Title</h1>
  <p>Hello <b>world</b> and <a href="https://example.com">links</a>.</p>
  <ul>
    <li>First item</li>
    <li>Second item</li>
  </ul>
  <pre>code block</pre>
</div>
"""

doc = JustHTML(html)
print(doc.to_markdown())
```

Output:

````html
# Title

Hello **world** and [links](https://example.com).

- First item
- Second item

```
code block
```
````

## Which should I use?
- Use `to_text()` for the raw concatenated text of a subtree (textContent semantics).
- Use `to_markdown()` when you want readable, structured Markdown.
