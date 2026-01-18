[← Back to docs](index.md)

# Transforms

JustHTML supports optional **transforms** to modify the parsed DOM tree right after parsing.

This is intended as a migration path for Bleach/html5lib filter pipelines, but implemented as DOM transforms (tree-aware and HTML5-treebuilder-correct).

If you're migrating an existing Bleach setup, see [Migrating from Bleach](bleach-migration.md).

Transforms are applied during construction via the `transforms` keyword argument:

```python
from justhtml import JustHTML

doc = JustHTML("<p>Hello</p>", transforms=[...])
```

## Quick example

```python
from justhtml import JustHTML, Drop, SetAttrs

doc = JustHTML(
    "<p>Hello</p><script>alert(1)</script>",
    transforms=[
        SetAttrs("p", id="greeting"),
        Drop("script"),
    ],
)

# The tree is transformed in memory
print(doc.root.to_html())

# Output is still safe by default
print(doc.to_html(pretty=False))
```

## Safety model

- Transforms run **once**, right after parsing, and mutate `doc.root`.
- JustHTML is safe-by-default by sanitizing at construction (`JustHTML(..., safe=True)`).
- Serialization (`to_html`/`to_text`/`to_markdown`) is serialize-only; earlier versions accepted `safe=` or `policy=` when serializing. This is no longer needed.

Raw output is available by disabling sanitization:

```python
doc = JustHTML("<p>Hello</p><script>alert(1)</script>", safe=False)
doc.to_html(pretty=False)
doc.root.to_html(pretty=False)
```

Sanitization can remove or rewrite transform results (for example, unsafe tags, event handler attributes, or unsafe URLs in `href`).

## Ordering

Transforms run left-to-right, but JustHTML may **batch compatible transforms** into a single tree walk for performance.

Batching preserves left-to-right ordering, but it is still a single walk with a moving cursor.
If a transform inserts or moves nodes **before** the current cursor, later transforms in the same walk may not visit those nodes.

If you need explicit pass boundaries (to make multi-pass pipelines easier to read, or to avoid cross-transform batching effects), use `Stage([...])` (see “Advanced: Stages” below).

```python
from justhtml import JustHTML, Drop, SetAttrs

doc = JustHTML(
    "<p>Hello</p>",
    transforms=[
        SetAttrs("p", id="x"),
        Drop("p"),
    ],
)

```

## Advanced: Stages

`Stage([...])` lets you explicitly split transforms into **separate passes**.

Use it when you want to make a multi-pass pipeline clearer, or when you want to avoid cross-transform batching effects.

Stages also matter for semantics when earlier transforms insert/move nodes “behind” the current walk position.
Splitting into stages forces a new walk, so later transforms see the updated tree.

- Stages can be nested; nested stages are flattened.
- If at least one `Stage` is present at the top level, any top-level transforms around it are automatically grouped into implicit stages.

Example: Let's a Edit() transform create new nodes and then set attributes

```python
from justhtml import Edit, JustHTML, SetAttrs, Stage
from justhtml.node import Node, Text


def insert_marker(p):
    # Insert a new sibling *before* the current node.
    # Without an explicit stage boundary, later transforms in the same walk
    # may not visit nodes inserted before the current cursor.
    marker = Node("span")
    marker.append_child(Text("NEW "))
    # If this was insert_after, SetAttrs would have seen the node.
    p.parent.insert_before(marker, p)

doc = JustHTML(
    "<p>one</p><p>two</p>",
    fragment=True,
    transforms=[
        # Without Stage, SetAttrs will miss the inserted <span>.
        Edit("p:first-child", insert_marker),
        SetAttrs("span", id="marker"),
    ],
)

# With Stage, the second pass sees the inserted <span>:
doc2 = JustHTML(
    "<p>one</p><p>two</p>",
    fragment=True,
    safe=False,
    transforms=[
        Stage([Edit("p:first-child", insert_marker)]),
        Stage([SetAttrs("span", id="marker")]),
    ],
)

print(doc.to_html(pretty=False))
print(doc2.to_html(pretty=False))
```

Output:

```html
<span>NEW </span><p>one</p><p>two</p>
<span id="marker">NEW </span><p>one</p><p>two</p>
```


## Tree shape

Transforms operate on the HTML5 treebuilder result, not the original token stream.

This means elements may already be inserted, moved, or normalized according to HTML parsing rules (for example, `<template>` elements end up in `<head>` in a full document).

## Performance

JustHTML **compiles transforms before applying them**:

- CSS selectors are parsed once up front.
- The tree is then walked with the compiled transforms.

Transforms are applied with a single in-place traversal that supports structural edits.

Transforms are optional; omitting `transforms` keeps constructor behavior unchanged.

## Validation

Transform selectors are validated during construction.
Invalid selectors raise `SelectorError` early, before the document is exposed.

Only the built-in transform objects are supported.
Unsupported transform objects raise `TypeError`.

## Scope

Selector-based transforms (`SetAttrs`, `Drop`, `Unwrap`, `Empty`, `Edit`) apply only to element nodes.
They never match document containers, text nodes, comments, or doctypes.

`Linkify` is different: it scans **text nodes** and wraps detected URLs/emails in `<a>` elements.
It never touches attributes, existing tags, comments, or doctypes.

## Enabling/disabling transforms

All built-in transforms have an `enabled` flag.

- If `enabled=False`, the transform is skipped at compile time (it does not run and does not affect ordering).
- `Stage([...], enabled=False)` is treated as if it was not present.

## Hooks

All built-in transforms share the same optional keyword parameters:

- `enabled=True` — if false, the transform is skipped at compile time (it does not run and does not affect ordering).
- `callback=None` — a node hook, invoked as `callback(node)` when the transform performs its action for a node.
- `report=None` — a reporting hook, invoked as `report(msg, node=...)` with a human-readable description of what happened.

Some transforms require an additional function argument (for example `Edit(..., func)`), which is documented in their signatures below.

## Built-in transforms

- [`Linkify(...)`](linkify.md) — Scan text nodes and convert URLs/emails into `<a>` elements.
- `CollapseWhitespace(skip_tags=(...))` — Collapse whitespace runs in text nodes (html5lib-like).
- `Sanitize(policy=None)` — Sanitize the in-memory tree (reviewable pipeline).
- `PruneEmpty(selector, strip_whitespace=True)` — Recursively drop empty elements.
- `Stage([...])` — Split transforms into explicit passes (advanced).

Core selector transforms:

- `SetAttrs(selector, attributes=None, **attrs)` — Set/overwrite attributes on matching elements.
- `Drop(selector)` — Remove matching nodes.
- `Unwrap(selector)` — Remove the element but keep its children.
- `Empty(selector)` — Remove all children of matching elements.
- `Edit(selector, func)` — Run custom logic for matching elements.

Advanced building blocks (useful for policy-driven pipelines):

- `EditDocument(func)` — Run once on the root container.
- `Decide(selector, func)` — Keep/drop/unwrap/empty based on a callback.
- `EditAttrs(selector, func)` — Rewrite attributes based on a callback (`RewriteAttrs` is an alias).
- `DropComments()` — Drop `#comment` nodes.
- `DropDoctype()` — Drop `!doctype` nodes.
- `DropForeignNamespaces()` — Drop elements in foreign namespaces (SVG/MathML).
- `DropAttrs(selector, patterns=())` — Drop attributes matching glob-like patterns.
- `AllowlistAttrs(selector, allowed_attributes=...)` — Keep only allowlisted attributes.
- `DropUrlAttrs(selector, url_policy=...)` — Validate/rewrite/drop URL-valued attributes.
- `AllowStyleAttrs(selector, allowed_css_properties=...)` — Sanitize inline `style` attributes.
- `MergeAttrs(tag, attr=..., tokens=...)` — Merge tokens into a whitespace-delimited attribute.

### `Linkify(...)`

See [`Linkify(...)`](linkify.md) for full documentation and examples.

### `CollapseWhitespace(skip_tags=(...), enabled=True, callback=None, report=None)`

Collapses runs of HTML whitespace characters in text nodes to a single space.

This is similar to `html5lib.filters.whitespace.Filter`.

By default it skips `<pre>`, `<textarea>`, `<code>`, `<title>`, `<script>`, and `<style>`.

```python
from justhtml import CollapseWhitespace, JustHTML

doc = JustHTML(
    "<p>Hello \n\t world</p><pre>a  b</pre>",
    fragment=True,
    transforms=[CollapseWhitespace()],
)

print(doc.to_html(pretty=False))
```

Output:

```html
<p>Hello world</p><pre>a  b</pre>
```

### `Sanitize(policy=None, enabled=True, callback=None, report=None)`

Sanitizes the in-memory DOM tree using the same sanitizer as `safe=True` output.

- This is useful if you want to traverse/modify a clean DOM.
- `Sanitize(...)` runs at its position in the pipeline. If you add transforms after it, be careful not to reintroduce unsafe content.

#### Reviewability (security argument)

`Sanitize(...)` is implemented as an explicit pipeline of smaller, focused transforms (like `DropComments`, `DropAttrs`, `DropUrlAttrs`, …).
This makes it easier to audit: the sanitizer behavior is a readable list of operations rather than a monolithic “magic” pass.

The `Sanitize(...)` pipeline compiles to this ordered list of transforms (some may be disabled by policy):

- `Drop("tag1, tag2, ...", callback=..., report=...)` — Drops dangerous content containers like `script`/`style` (drops the *entire* subtree).
- `DropComments(callback=..., report=...)` — Drops comments.
- `DropDoctype(callback=..., report=...)` — Drops doctypes.
- `DropForeignNamespaces(callback=..., report=...)` — Drops elements in foreign namespaces (SVG/MathML) when enabled by policy.
- `Unwrap(":not(allowed_tags)", callback=..., report=...)` — Unwraps disallowed elements (keeps their children) for non-container tags.
- `DropAttrs("*", patterns=("on*", "srcdoc", "*:*"), callback=..., report=...)` — Drops dangerous attributes (`on*`, `srcdoc`, and namespaced attributes like `xlink:href`).
- `AllowlistAttrs("*", allowed_attributes=..., callback=..., report=...)` — Applies tag/attribute allowlists.
- `DropUrlAttrs("*", url_policy=..., callback=..., report=...)` — Validates and rewrites URL-valued attributes (`href`, `src`, `srcset`, …) according to `UrlPolicy`.
- `AllowStyleAttrs("[style]", allowed_css_properties=..., callback=..., report=...)` — Optionally sanitizes inline styles using an allowlist of CSS properties.
- `MergeAttrs("a", attr="rel", tokens=..., callback=..., report=...)` — Optionally enforces `rel` tokens on links.

For policy details, see [Sanitization & Security](sanitization.md).

`PruneEmpty(...)` after `Sanitize(...)` is useful if sanitization removes unsafe children (for example `<script>`) and leaves a now-empty wrapper element.

`CollapseWhitespace(...)` after `Sanitize(...)` is useful if you want an already-sanitized in-memory tree, but still normalize whitespace (similar to `html5lib.filters.whitespace.Filter`).

### `PruneEmpty(selector, strip_whitespace=True, enabled=True, callback=None, report=None)`

Recursively drops elements that are empty after transforms have run.

"Empty" means there are no element children and no non-whitespace text.

If you want whitespace-only text nodes to count as content (so `<p> </p>` is kept), pass `strip_whitespace=False`.

`PruneEmpty(...)` runs as a post-order walk over the tree and removes elements that are empty at that point in the transform pipeline.

If you want to prune after all other transforms, put `PruneEmpty(...)` at the end (or immediately before `Sanitize(...)`).

Example: remove empty paragraphs after dropping unwanted tags:

```python
from justhtml import Drop, JustHTML, PruneEmpty

doc = JustHTML(
    "<p></p><p><img></p><p><img src=\"/x\"></p>",
    fragment=True,
    transforms=[
        Drop('img:not([src]), img[src=""]'),
        PruneEmpty("p"),
    ],
)

print(doc.to_html(pretty=False))
```

Output:

```html
<p><img src="/x"></p>
```

### `SetAttrs(selector, enabled=True, callback=None, report=None, attributes=None, **attrs)`

Sets/overwrites attributes on matching elements.

Attribute values are converted to strings.
Passing `None` creates a boolean attribute (serialized in minimized form by default).

`enabled` controls whether the transform runs.

If you need to set an attribute that collides with a keyword-only parameter name (like `enabled`, `callback`, or `report`), use the `attributes=` dict.

```python
SetAttrs("a", rel="nofollow", target="_blank")
SetAttrs("input", disabled=None)
SetAttrs("div", attributes={"enabled": "true"})
```

### `Drop(selector, enabled=True, callback=None, report=None)`

Removes matching elements and their contents.

Optional: pass `callback(node)` / `report(msg, node=...)` to run hooks right before the node is dropped.

`Drop(...)` supports any selector that the JustHTML selector engine supports, including comma-separated selectors and attribute selectors.

Example: remove scripts and styles:

```python
from justhtml import JustHTML, Drop

doc = JustHTML(
    "<p>Hello</p><script>alert(1)</script><style>p{}</style>",
    fragment=True,
    transforms=[Drop("script, style")],
)

print(doc.to_html(pretty=False))
```

Output:

```html
<p>Hello</p>
```

Example: drop elements by class:

```python
from justhtml import JustHTML, Drop

doc = JustHTML(
    '<p>One</p><div class="ad">Buy</div><p>Two</p>',
    fragment=True,
    transforms=[Drop(".ad")],
)

print(doc.to_html(pretty=False))
```

Output:

```html
<p>One</p><p>Two</p>
```

Example: drop only some elements based on attributes:

```python
from justhtml import JustHTML, Drop

doc = JustHTML(
    '<p><img><img src=""><img src="/x"></p>',
    fragment=True,
    transforms=[Drop('img:not([src]), img[src=""]')],
)

print(doc.to_html(pretty=False))
```

Output:

```html
<p><img src="/x"></p>
```

### `Unwrap(selector, enabled=True, callback=None, report=None)`

Removes the element but keeps its children (hoists contents).

Optional: pass `callback(node)` / `report(msg, node=...)` to run hooks right before the node is unwrapped.

```python
Unwrap("span")
Unwrap("div.wrapper")
```

### `Empty(selector, enabled=True, callback=None, report=None)`

Keeps the element but removes its children.

This also clears `<template>` contents.

Optional: pass `callback(node)` / `report(msg, node=...)` to run hooks when a node is emptied.

```python
Empty("pre")
Empty("template")
```

### `Edit(selector, func, enabled=True, callback=None, report=None)`

Escape hatch for custom logic. Runs `func(node)` for each matching element.

Optional: pass `callback(node)` / `report(msg, node=...)` to run hooks when the transform is applied.

## Advanced transform building blocks

These are lower-level transforms that are primarily useful for building policy-driven pipelines (including the built-in `Sanitize(...)` pipeline).

### `EditDocument(func, enabled=True, callback=None, report=None)`

Runs `func(root)` exactly once with the document root (`#document` or `#document-fragment`).

Use this for transforms that need access to the root container node (which selector-based transforms do not visit).

### `Decide(selector, func, enabled=True, callback=None, report=None)`

General-purpose structural transform.

- For a normal selector (not `"*"`), the callback is invoked only for matching element/template nodes.
- For selector `"*"`, the callback is invoked for every node type (including text/comment/doctype and document containers).

The callback must return one of:

- `Decide.KEEP` — keep node
- `Decide.DROP` — drop node
- `Decide.UNWRAP` — remove node but keep its children (and template contents)
- `Decide.EMPTY` — keep node but remove its children (and template contents)

### `EditAttrs(selector, func, enabled=True, callback=None, report=None)`

Rewrite element attributes using a callback.

- Return `None` to leave attributes unchanged.
- Return a `dict[str, str | None]` to replace the node’s attributes.

`RewriteAttrs` is a backwards-compatible alias for `EditAttrs`.

### `DropComments(enabled=True, callback=None, report=None)`

Drops comment nodes (`#comment`).

### `DropDoctype(enabled=True, callback=None, report=None)`

Drops doctype nodes (`!doctype`).

### `DropForeignNamespaces(enabled=True, callback=None, report=None)`

Drops elements in non-HTML namespaces (for example SVG/MathML) when enabled.

If provided, `callback(node)` / `report(msg, node=...)` is called when a foreign element is dropped.

### `DropAttrs(selector, patterns=(), enabled=True, callback=None, report=None)`

Drops attributes whose names match patterns like `"on*"` or `"*:*"`.

Patterns support `*` and `?` wildcards.
If provided, `callback(node)` / `report(msg, node=...)` is called for each dropped attribute.

### `AllowlistAttrs(selector, allowed_attributes=..., enabled=True, callback=None, report=None)`

Keeps only allowlisted attributes.

`allowed_attributes` is a mapping like:

- `{"*": {"id", "class"}, "a": {"href", "rel"}}`

If provided, `callback(node)` / `report(msg, node=...)` is called when an attribute is dropped for not being allowlisted.

### `DropUrlAttrs(selector, url_policy=..., enabled=True, callback=None, report=None)`

Validates and rewrites/drops URL-valued attributes (`href`, `src`, `srcset`, …) according to a `UrlPolicy`.

If provided, `callback(node)` / `report(msg, node=...)` is called when a URL is dropped.

### `AllowStyleAttrs(selector, allowed_css_properties=..., enabled=True, callback=None, report=None)`

Sanitizes inline `style` attributes using an allowlist of CSS properties.

If provided, `callback(node)` / `report(msg, node=...)` is called when a `style` attribute is dropped.

### `MergeAttrs(tag, attr=..., tokens=..., enabled=True, callback=None, report=None)`

Merges tokens into a whitespace-delimited attribute without removing existing tokens.

This is used by the sanitizer to enforce `rel` tokens on links (e.g. `noopener`).

`Edit` is useful for transformations that don’t fit the built-ins, such as removing attributes, rewriting URLs, or conditionally dropping nodes.

```python
from justhtml import JustHTML, Edit


def strip_tracking_params(a):
    href = a.attrs.get("href")
    if href:
        a.attrs["href"] = href.split("?", 1)[0]


doc = JustHTML(
    "<a href=\"https://e.com/?utm_source=x\">x</a>",
    transforms=[Edit("a", strip_tracking_params)],
)
```

Removing attributes is typically done via `Edit`:

```python
from justhtml import JustHTML, Edit


def drop_inline_handlers(node):
    node.attrs.pop("onclick", None)
    node.attrs.pop("onload", None)


doc = JustHTML("<a onclick=\"x()\">x</a>", transforms=[Edit("a", drop_inline_handlers)])
```

## Recipes

### Add `rel` to all links

```python
from justhtml import JustHTML, SetAttrs

doc = JustHTML(
    "<p><a href=\"https://example.com\">Example</a></p>",
    transforms=[
        SetAttrs("a", rel="nofollow noreferrer"),
    ],
)
```

### Strip elements but keep their text

```python
from justhtml import JustHTML, Unwrap

doc = JustHTML(
    "<p>Hello <span class=\"x\">world</span></p>",
    transforms=[Unwrap("span.x")],
)
```

### Drop unsafe elements early, while still keeping safe-by-default sanitization

```python
from justhtml import JustHTML, Drop

doc = JustHTML(
    "<p>ok</p><script>alert(1)</script>",
    transforms=[Drop("script")],
)
print(doc.to_html(pretty=False))
```

## Acknowledgements

JustHTML's Linkify behavior is validated against the upstream `linkify-it` fixture suite (MIT licensed).

- Fixtures: `tests/linkify-it/fixtures/`
- License: `tests/linkify-it/LICENSE.txt`
