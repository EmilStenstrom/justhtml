[← Back to docs](index.md)

# Migrating from Bleach

JustHTML’s sanitization and transform pipeline were heavily inspired by Bleach’s real-world ergonomics.

Bleach has helped a lot of projects ship safer HTML over the years, and a lot of that is thanks to the hard work of [@willkg](https://github.com/willkg) building and maintaining it.

In 2023, Bleach’s maintainer announced that **Bleach is deprecated** (but will continue to receive security updates, new Python version support, and fixes for egregious bugs). See: https://github.com/mozilla/bleach/issues/698

This guide covers common migration patterns.

## Mental model differences

- Bleach takes a string and returns a cleaned string.
- JustHTML parses into a DOM and sanitizes by default at construction time:
    - `JustHTML(html)` sanitizes by default (`sanitize=True`).
    - `JustHTML(html, sanitize=False)` disables sanitization (trusted input only). (`safe` is a backwards-compatible alias.)

JustHTML also supports constructor-time **transforms** (a DOM equivalent of Bleach/html5lib filter pipelines): see [Transforms](transforms.md).

## Equivalent of `bleach.clean(...)`

A typical Bleach call:

```python
import bleach

clean = bleach.clean(
    user_html,
    tags=["p", "b", "a"],
    attributes={"a": ["href"]},
    protocols=["http", "https"],
    strip=True,
)
```

In JustHTML you typically configure a `SanitizationPolicy`:

```python
from justhtml import JustHTML, SanitizationPolicy, UrlPolicy, UrlRule

policy = SanitizationPolicy(
    allowed_tags=["p", "b", "a"],
    allowed_attributes={"*": [], "a": ["href"]},
    url_policy=UrlPolicy(
        default_handling="strip",
        allow_rules={
            ("a", "href"): UrlRule(allowed_schemes=["http", "https"]),
        },
    ),
)

doc = JustHTML(user_html, fragment=True, policy=policy)
clean = doc.to_html()
```

Notes:

- Prefer `fragment=True` for user-generated snippets. That avoids adding `<html>`, `<head>`, and `<body>` tags.
- JustHTML sanitizes *at construction time* by default. If you need to sanitize again after later DOM edits, add `Sanitize(...)` at the end of your transform pipeline (see [HTML Cleaning](html-cleaning.md)).

## Bleach filters → JustHTML transforms

Bleach supports html5lib filters and helper utilities (like linkifying text).

In JustHTML, you compose transforms (applied once, right after parsing):

- `bleach.linkify(...)` → `Linkify(...)` (see [Linkify](linkify.md))
- `html5lib.filters.whitespace.Filter` → `CollapseWhitespace(...)`
- "Strip tag but keep contents" → `Unwrap(selector)`
- "Drop tag and contents" → `Drop(selector)`
- "Remove children" → `Empty(selector)`
- "Set attributes" → `SetAttrs(selector, **attrs)`
- "Custom rewrite" → `Edit(selector, func)`

Example: linkify text, then add safe link attributes:

```python
from justhtml import JustHTML, Linkify, SetAttrs

doc = JustHTML(
    "<p>See example.com</p>",
    fragment=True,
    transforms=[
        Linkify(),
        SetAttrs("a", rel="nofollow noopener", target="_blank"),
    ],
)

# Still sanitized by default (construction time)
print(doc.to_html(pretty=False))
```

## URLs and protocols

Bleach’s `protocols=[...]` concept maps to JustHTML’s URL policy rules.

- Configure allowed schemes per attribute via `UrlRule(allowed_schemes=[...])`.
- Under `sanitize=True`, URLs can be rewritten or stripped according to policy (see [URL Cleaning](url-cleaning.md)).

## What to do with “strip=True/False”

Bleach’s `strip` option controls whether disallowed tags are removed entirely or escaped.

JustHTML’s sanitizer is allowlist-based and focuses on producing safe markup. Disallowed tags are handled by `SanitizationPolicy(disallowed_tag_handling=...)`, and dangerous containers (like `script`/`style`) drop their contents.

Mapping:

- Bleach `strip=True` → `disallowed_tag_handling="unwrap"` (default)
    - The disallowed tag is removed, but its children are kept (and sanitized).
- Bleach `strip=False` → `disallowed_tag_handling="escape"`
    - The disallowed tag’s start/end tags are emitted as text (escaped), and its children are kept (and sanitized).

JustHTML also supports `disallowed_tag_handling="drop"` to drop the entire disallowed subtree.

If you need to display untrusted HTML *as text*, render it as text instead of HTML (for example via `to_text()` / `to_markdown()`), or escape it before embedding it into an HTML page.

If you need additional structural cleanup beyond policy decisions, prefer doing it explicitly with transforms.

## Suggested migration approach

1. Start by using JustHTML’s default sanitizer (`JustHTML(...).to_html()`).
2. Add a `SanitizationPolicy` that matches your allowlist and URL requirements.
3. Add transforms to replace any Bleach filter/linkify behavior.
4. Add tests for your exact input corpus and output expectations (especially around URLs and allowed attributes).
