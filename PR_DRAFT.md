# PR: Expose pre-sanitization DOM and add `SANITIZE_NONE` convenience constant

## Problem

When `sanitize=True` (the default), `JustHTML` drops elements not in the
`DEFAULT_DOCUMENT_POLICY.allowed_tags` list. This includes commonly needed
elements like `<input>`, `<meta>`, `<link>`, and `<form>` — elements that
are essential for web scraping, form parsing, and metadata extraction.

```python
from justhtml import JustHTML

html = '<html><head><meta charset="utf-8"></head><body><input id="x" value="1"><p>hi</p></body></html>'

doc = JustHTML(html)  # sanitize=True by default
doc.query("meta")     # [] — meta dropped!
doc.query("input")    # [] — input dropped!
doc.query("form")     # [] — form dropped!

doc = JustHTML(html, sanitize=False)
doc.query("meta")     # [<Element meta>] — works!
doc.query("input")    # [<Element input>] — works!
```

Users discover this only after debugging mysterious empty `query()` results.
The `sanitize=False` workaround exists but is not discoverable, and it
disables ALL sanitization (XSS protection, style stripping, etc.) when
the user only needs a few extra elements.

## Solution

Two additions:

### 1. `JustHTML.raw_root` property

Exposes the pre-sanitization DOM tree. When `sanitize=False`, `raw_root`
is identical to `root`. When `sanitize=True`, `raw_root` contains ALL
elements (including `<input>`, `<meta>`, etc.) while `root` contains
the sanitized version.

```python
doc = JustHTML(html)  # sanitize=True
doc.query("meta")        # [] (sanitized)
doc.raw_root.query("meta")  # [<Element meta>] (raw!)
```

### 2. `justhtml.ALLOW_ALL_TAGS` constant

A pre-built `SanitizationPolicy` that allows all HTML5 elements. Useful
for web scraping where you want URL sanitization and XSS protection
but don't want elements dropped:

```python
from justhtml import JustHTML, ALLOW_ALL_TAGS

doc = JustHTML(html, policy=ALLOW_ALL_TAGS)
doc.query("meta")     # [<Element meta>] — preserved!
doc.query("input")    # [<Element input>] — preserved!
```

## Changes

### `parser/__init__.py`

Store the raw root before transforms are applied:

```python
# After treebuilder finishes, before transforms:
self._raw_root = self.root  # ← NEW

# After transforms:
# self.root is now sanitized (existing behavior unchanged)
```

Add property:

```python
@property
def raw_root(self) -> Document | DocumentFragment:
    """The pre-sanitization DOM tree.

    When ``sanitize=False``, this is identical to ``root``.
    When ``sanitize=True``, this contains ALL elements — including
    ``<input>``, ``<meta>``, ``<link>``, ``<form>`` — that the
    sanitizer would otherwise drop.

    Use this when you need to query elements that the default
    sanitizer removes, without disabling sanitization entirely.
    """
    return self._raw_root
```

### `sanitizer/policy_defaults.py`

Add `ALLOW_ALL_TAGS` after `DEFAULT_POLICY`:

```python
def _build_allow_all_tags() -> SanitizationPolicy:
    """A policy that allows all standard HTML5 elements.

    Drops only ``<script>`` and ``<style>`` content (same as default),
    but preserves structural and form elements (``<input>``, ``<meta>``,
    ``<form>``, ``<link>``, etc.) that the default content policy drops.

    Use for web scraping where you want XSS protection but need
    access to all DOM elements.
    """
    from justhtml.sanitizer.policy import SanitizationPolicy

    # All standard HTML5 elements, minus script/style (content dropped)
    ALL_TAGS = {
        # Document structure
        "html", "head", "body", "title",
        # Metadata
        "meta", "link", "base",
        # Sections
        "article", "aside", "footer", "header", "hgroup", "main", "nav", "section",
        # Headings
        "h1", "h2", "h3", "h4", "h5", "h6",
        # Block
        "address", "blockquote", "dd", "div", "dl", "dt", "figcaption", "figure",
        "hr", "li", "ol", "p", "pre", "ul",
        # Inline
        "a", "abbr", "b", "bdi", "bdo", "br", "cite", "code", "data", "dfn", "em",
        "i", "kbd", "mark", "q", "rp", "rt", "rtc", "ruby", "s", "samp", "small",
        "span", "strong", "sub", "sup", "time", "u", "var", "wbr",
        # Media
        "area", "audio", "img", "map", "track", "video",
        # Embedded
        "embed", "iframe", "object", "param", "picture", "source",
        # Table
        "caption", "col", "colgroup", "table", "tbody", "td", "tfoot", "th",
        "thead", "tr",
        # Form
        "button", "datalist", "fieldset", "form", "input", "label", "legend",
        "meter", "optgroup", "option", "output", "progress", "select", "textarea",
        # Interactive
        "details", "dialog", "menu", "summary",
        # Scripting (content dropped, but element preserved)
        "noscript", "canvas", "template",
        # Deprecated but still in the wild
        "center", "font", "big", "strike", "tt",
        # Web components
        "slot",
    }

    return SanitizationPolicy(
        allowed_tags=sorted(ALL_TAGS),
        allowed_attributes={
            "*": ["class", "id", "title", "lang", "dir", "role", "tabindex"],
            "a": ["href", "title", "target", "rel"],
            "img": ["src", "alt", "title", "width", "height", "loading", "decoding", "srcset", "sizes"],
            "input": ["type", "name", "value", "placeholder", "checked", "disabled", "readonly", "required", "min", "max", "step", "pattern", "accept", "autocomplete", "autofocus"],
            "form": ["action", "method", "enctype", "novalidate"],
            "textarea": ["name", "rows", "cols", "placeholder", "disabled", "readonly", "required"],
            "select": ["name", "multiple", "disabled", "required"],
            "option": ["value", "selected", "disabled"],
            "button": ["type", "name", "value", "disabled"],
            "meta": ["charset", "name", "content", "http-equiv", "property"],
            "link": ["rel", "href", "type", "sizes", "media"],
            "th": ["colspan", "rowspan", "scope"],
            "td": ["colspan", "rowspan"],
            "iframe": ["src", "width", "height", "frameborder", "allowfullscreen"],
            "video": ["src", "controls", "autoplay", "loop", "muted", "poster", "width", "height"],
            "audio": ["src", "controls", "autoplay", "loop", "muted"],
            "source": ["src", "type", "srcset", "sizes"],
            "time": ["datetime"],
            "data": ["value"],
            "meter": ["value", "min", "max", "low", "high", "optimum"],
            "progress": ["value", "max"],
            "output": ["for", "form", "name"],
            "details": ["open"],
            "dialog": ["open"],
        },
        url_policy=UrlPolicy(
            default_handling="strip",
            allow_rules={
                ("a", "href"): UrlRule(
                    allowed_schemes=["http", "https", "mailto", "tel"],
                    handling="allow",
                    resolve_protocol_relative="https",
                ),
                ("img", "src"): UrlRule(
                    allowed_schemes=[],
                    handling="allow",
                    resolve_protocol_relative=None,
                ),
                ("form", "action"): UrlRule(
                    allowed_schemes=["http", "https"],
                    handling="allow",
                    resolve_protocol_relative="https",
                ),
            },
        ),
        allowed_css_properties=set(),
    )


ALLOW_ALL_TAGS = _build_allow_all_tags()
```

### `__init__.py`

Export the new constant:

```python
from .sanitizer import (
    ALLOW_ALL_TAGS,  # ← NEW
    CSS_PRESET_TEXT,
    DEFAULT_DOCUMENT_POLICY,
    DEFAULT_POLICY,
    ...
)
```

## Usage examples

### Web scraping (metadata extraction)

```python
from justhtml import JustHTML

html = requests.get(url).text
doc = JustHTML(html)  # sanitize=True (default)

# These work with the default policy:
doc.query("title")     # [<Element title>]
doc.query("p")         # [<Element p>, ...]
doc.query("a[href]")   # [<Element a>, ...]

# These need raw_root or sanitize=False:
doc.raw_root.query("meta[property='og:title']")  # [<Element meta>]
doc.raw_root.query_one("input[name='csrf']").attrs["value"]  # "token123"
```

### Form parsing

```python
from justhtml import JustHTML, ALLOW_ALL_TAGS

html = open("form.html").read()
doc = JustHTML(html, policy=ALLOW_ALL_TAGS)

# All form elements preserved
for inp in doc.query("input[name]"):
    print(inp.attrs.get("name"), inp.attrs.get("value"))
```

## Backward compatibility

- `sanitize=True` behavior is unchanged (default policy still drops the same elements)
- `sanitize=False` behavior is unchanged
- `raw_root` is a new property — no existing code breaks
- `ALLOW_ALL_TAGS` is a new constant — no existing code breaks
