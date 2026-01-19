[← Back to docs](index.md)

# API Reference

Complete documentation for the JustHTML public API.

## JustHTML

The main parser class.

```python
from justhtml import JustHTML
```

### Constructor

```python
JustHTML(html, *, safe=True, policy=None, collect_errors=False, track_node_locations=False, debug=False, encoding=None, fragment=False, fragment_context=None, iframe_srcdoc=False, strict=False, tokenizer_opts=None, tree_builder=None, transforms=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `html` | `str \| bytes \| bytearray \| memoryview` | required | HTML input to parse. Bytes are decoded using HTML encoding sniffing. |
| `safe` | `bool` | `True` | Sanitize untrusted HTML during construction |
| `policy` | `SanitizationPolicy \| None` | `None` | Override the default sanitization policy |
| `collect_errors` | `bool` | `False` | Collect all parse errors (enables `errors` property) |
| `track_node_locations` | `bool` | `False` | Track line/column positions for nodes (slower) |
| `debug` | `bool` | `False` | Enable debug mode (internal) |
| `encoding` | `str \| None` | `None` | Transport-supplied encoding label used as an override for byte input. See [Encoding & Byte Input](encoding.md). |
| `fragment` | `bool` | `False` | Parse as a fragment in a default `<div>` context (convenience). |
| `fragment_context` | `FragmentContext` | `None` | Parse as fragment inside this context element |
| `strict` | `bool` | `False` | Raise `StrictModeError` on the earliest parse error by source position |
| `transforms` | `list[Transform] \| None` | `None` | Optional DOM transforms applied after parsing. See [Transforms](transforms.md). |
| `iframe_srcdoc` | `bool` | `False` | Parse whole document as if it's inside an iframe `srcdoc` (HTML parsing quirk) |
| `tokenizer_opts` | `TokenizerOpts \| None` | `None` | Advanced tokenizer configuration |
| `tree_builder` | `TreeBuilder \| None` | `None` | Supply a custom tree builder |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `root` | `Document \| DocumentFragment` | The document root |
| `errors` | `list[ParseError]` | Parse errors, ordered by source position (only if `collect_errors=True`) |

### Methods

#### `to_text()`

Return the document's concatenated text.

```python
doc = JustHTML("<p>Hello <b>world</b></p>")
doc.to_text()  # "Hello world"  (separator=" ", strip=True)
```

Parameters:

- `separator` (default: `" "`): join string between text nodes
- `strip` (default: `True`): strip each text node and drop empties

Sanitization happens at construction time. Use `JustHTML(..., safe=False)` for trusted input or `JustHTML(..., policy=...)` to customize the policy.

#### `to_markdown(html_passthrough=False)`

Return a pragmatic subset of GitHub Flavored Markdown (GFM).

Tables (`<table>`) and images (`<img>`) are preserved as raw HTML. Raw HTML tags like
`<script>`, `<style>`, and `<textarea>` are dropped by default; pass
`html_passthrough=True` to preserve them (including their contents).

```python
doc = JustHTML("<h1>Title</h1><p>Hello <b>world</b></p>")
doc.to_markdown()  # "# Title\n\nHello **world**"
doc.to_markdown(html_passthrough=True)
```

Sanitization happens at construction time. Use `JustHTML(..., safe=False)` for trusted input or `JustHTML(..., policy=...)` to customize the policy.

#### `query(selector)`

Find all elements matching a CSS selector.

```python
doc.query("div.container > p")  # Returns list of matching nodes
```

---

## Node

Base type for all DOM nodes.

Node types:

- `Document`: the root for full-document parses
- `DocumentFragment`: the root for fragment parses
- `Element`: normal HTML/SVG/MathML elements
- `Text`: text nodes (`#text`)
- `Comment`: comment nodes (`#comment`)
- `Template`: `<template>` elements with `template_content`

`Template` nodes expose a `template_content` document fragment (HTML namespace only),
which holds the template’s children.

### Recommended mutation path

Direct DOM edits are supported, but transforms are the preferred way to make
changes because they preserve ordering semantics and make sanitization explicit.
See [Transforms](transforms.md) for the recommended workflow. If you mutate the
DOM after construction, sanitization has already happened; re-sanitize by using
`sanitize_dom(...)` or rebuild the document with a `Sanitize(...)` transform in
the construction pipeline.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Tag name (e.g., `"div"`) or `"#text"`, `"#comment"`, `"#document"` |
| `attrs` | `dict \| None` | Attribute dictionary (None for comments/doctypes) |
| `children` | `list \| None` | Child nodes (None for comments/doctypes) |
| `parent` | `Node` | Parent node (or `None` for root) |
| `text` | `str` | Node-local text value. For text nodes this is the node data, otherwise `""`. Use `to_text()` for textContent semantics. |
| `namespace` | `str \| None` | Namespace for the node (`"html"` by default for elements). |

### Methods

#### `to_html(indent=2)`

Serialize the node to HTML string.

```python
node.to_html()                 # Pretty-printed HTML
node.to_html(indent=0)         # Compact HTML
node.to_html(indent=4)         # 4-space indent

# Safety happens at construction time:
# - default: JustHTML(..., safe=True)
# - raw/trusted: JustHTML(..., safe=False)
# - custom policy: JustHTML(..., policy=policy)
```

#### `query(selector)`

Find descendants matching a CSS selector.

```python
div.query("p.intro")  # Search within this node
```

#### `to_text()`

Return the node's concatenated text.

```python
node.to_text()
```

Text extraction is safe-by-default when you build documents with `JustHTML(..., safe=True)` (the default). Use `safe=False` at construction for trusted input.

#### `to_markdown(html_passthrough=False)`

Return a pragmatic subset of GitHub Flavored Markdown (GFM) for this subtree.

```python
node.to_markdown()
node.to_markdown(html_passthrough=True)
```

Markdown output is safe-by-default when you build documents with `JustHTML(..., safe=True)` (the default). Use `safe=False` at construction for trusted input.

#### `append_child(node)`

Append a child node to this node.

#### `insert_before(node, reference_node)`

Insert `node` before `reference_node` (or append if `reference_node` is `None`).

#### `remove_child(node)`

Remove a direct child node.

#### `replace_child(new_node, old_node)`

Replace a direct child node with a new node.

#### `clone_node(deep=False, override_attrs=None)`

Clone this node. If `deep=True`, children are cloned recursively.

#### `has_child_nodes()`

Return `True` if this node has children.

---

## Sanitization

JustHTML includes a built-in, policy-driven HTML sanitizer.

Guides:

- [Sanitization overview](sanitization.md)
- [HTML Cleaning](html-cleaning.md)
- [URL Cleaning](url-cleaning.md)
- [Unsafe Handling](unsafe-handling.md)

```python
from justhtml import DEFAULT_POLICY, SanitizationPolicy, UrlPolicy, UrlProxy, UrlRule, sanitize_dom
```

### Sanitizing output vs sanitizing the DOM

- Construction sanitization is the default: `JustHTML(..., safe=True)` sanitizes once, right after parsing.
- If you want to sanitize *after* other transforms or direct DOM edits, add `Sanitize(...)` to your transform pipeline.
    - If you care about explicit transform passes, group transforms using [`Stage([...])`](transforms.md#advanced-stages).
    - For details on how `Sanitize(...)` works (and why it’s reviewable), see [Transforms](transforms.md#sanitizepolicynone-enabledtrue).

```python
from justhtml import JustHTML, Sanitize

doc = JustHTML(user_html, fragment=True, transforms=[Sanitize()])
clean_root = doc.root
```

### `sanitize_dom(node, *, policy=None, errors=None)`

Re-sanitize a DOM tree after direct edits. For document roots (`#document` or
`#document-fragment`), this mutates the tree in place. For other nodes, the
node is sanitized as if it were the only child of a document fragment; the
returned node may need to be reattached by the caller.

```python
from justhtml import sanitize_dom

sanitize_dom(doc.root)  # In-place for document roots
```

### `DEFAULT_POLICY`

Conservative built-in policy used for safe-by-default sanitization.

### `DEFAULT_DOCUMENT_POLICY`

Conservative built-in policy used when sanitizing full documents (preserves `<html>`, `<head>`, and `<body>` wrappers).

### `SanitizationPolicy`

Defines allowlists for tags and attributes, URL validation rules, and optional inline-style allowlisting.

Notable options:

- `unsafe_handling`: "strip" (default), "raise", or "collect"
- `disallowed_tag_handling`: "unwrap" (default), "escape", or "drop"
- `url_policy`: controls URL validation and URL handling ("allow", "strip", or "proxy")

### `UrlPolicy`

Wraps URL rules and controls what happens to URL-valued attributes.

```python
UrlPolicy(
    default_handling="strip",  # or "allow" / "proxy"
    default_allow_relative=True,
    allow_rules={},
    url_filter=None,
    proxy=None,
)
```

### `UrlProxy`

Proxy rewrite configuration used when `default_handling="proxy"`.

```python
UrlProxy(
    url="/proxy",
    param="url",
)
```

### `UrlRule`

Controls how URL-valued attributes like `a[href]` and `img[src]` are validated.

```python
UrlRule(
    allow_fragment=True,
    resolve_protocol_relative="https",
    allowed_schemes=set(),
    allowed_hosts=None,
    proxy=None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `allow_fragment` | `bool` | `True` | Allow fragment-only URLs (e.g. `#anchor`) |
| `resolve_protocol_relative` | `str \| None` | `"https"` | Scheme to resolve protocol-relative URLs (`//...`) to before checking. If `None`, they are dropped. |
| `allowed_schemes` | `set[str]` | `set()` | Allowed schemes for absolute URLs (e.g. `{"https", "mailto"}`) |
| `allowed_hosts` | `set[str] \| None` | `None` | If set, only allow these hosts (e.g. `{"example.com"}`) |
| `proxy` | `UrlProxy \| None` | `None` | Per-rule proxy override used when `UrlPolicy.default_handling="proxy"` |

---

## stream

Memory-efficient streaming parser.

```python
from justhtml import stream

for event, data in stream(html):
    ...
```

`stream()` accepts the same input types as `JustHTML`. If you pass bytes, it will decode using HTML encoding sniffing.
To override the encoding for byte input, pass `encoding=...`.

### Events

| Event | Data | Description |
|-------|------|-------------|
| `"start"` | `(tag_name, attrs_dict)` | Opening tag |
| `"end"` | `tag_name` | Closing tag |
| `"text"` | `text_content` | Text content |
| `"comment"` | `comment_text` | HTML comment |
| `"doctype"` | `doctype_name` | DOCTYPE declaration |

---

## FragmentContext

Specifies the context element for fragment parsing. See [Fragment Parsing](fragments.md) for detailed usage.

```python
from justhtml.context import FragmentContext
```

### Constructor

```python
FragmentContext(tag_name, namespace=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tag_name` | `str` | required | Context element tag name (e.g., `"div"`, `"tbody"`) |
| `namespace` | `str \| None` | `None` | `None` for HTML, `"svg"` for SVG, `"math"` for MathML |

### Example

```python
from justhtml import JustHTML
from justhtml.context import FragmentContext

# Parse table rows in correct context
ctx = FragmentContext("tbody")
doc = JustHTML("<tr><td>cell</td></tr>", fragment_context=ctx)
```

---

## ParseError

Represents a parse error with location information.

```python
from justhtml import ParseError
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `code` | `str` | Error code (e.g., `"eof-in-tag"`) |
| `line` | `int` | Line number (1-indexed) |
| `column` | `int` | Column number (1-indexed) |
| `message` | `str` | Human-readable error message |

### Methods

#### `as_exception()`

Convert to a `SyntaxError` with source highlighting (Python 3.11+).

```python
error.as_exception()  # Returns SyntaxError
```

---

## StrictModeError

Exception raised when parsing with `strict=True`.

```python
from justhtml import StrictModeError
```

Inherits from `SyntaxError`, so it displays source location in tracebacks.

---

## Standalone Functions

### `query(node, selector)`

Query a node without using the method syntax.

```python
from justhtml import query
results = query(doc.root, "div.main")
```

### `matches(node, selector)`

Check if a node matches a selector.

```python
from justhtml import matches
if matches(node, "div.active"):
    ...
```

### `to_html(node, indent=2)`

Serialize a node to HTML.

```python
from justhtml import to_html
html_string = to_html(node)
```

---

## SelectorError

Exception raised for invalid CSS selectors.

```python
from justhtml import SelectorError

try:
    doc.query("div[invalid")
except SelectorError as e:
    print(e)
```
