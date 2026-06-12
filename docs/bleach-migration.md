[← Back to docs](index.md)

# Migrating from Bleach

JustHTML’s sanitization and transform pipeline were heavily inspired by Bleach’s real-world ergonomics.

Bleach has helped a lot of projects ship safer HTML over the years, and a lot of that is thanks to the hard work of [@willkg](https://github.com/willkg) building and maintaining it.

In 2023, Bleach’s maintainer announced that **Bleach is deprecated** (but will continue to receive security updates, new Python version support, and fixes for egregious bugs). See: https://github.com/mozilla/bleach/issues/698

This guide covers common migration patterns.

## Real-world migration pattern: add a local wrapper first

A large application migration often works best in two steps:

1. Add a small application-local `clean(...)` wrapper that accepts the Bleach-shaped options your codebase already uses.
2. Migrate call sites from `bleach.clean(...)` / `bleach.linkify(...)` to that wrapper.

That keeps most call-site changes mechanical while leaving the security policy in one place.

Example wrapper:

```python
import re
from collections.abc import Collection, Mapping

from justhtml import JustHTML, Linkify, SanitizationPolicy, SetAttrs, UrlPolicy, UrlRule

URL_LIKE_ATTRS = {
    "href",
    "src",
    "srcset",
    "poster",
    "action",
    "formaction",
    "data",
    "cite",
    "background",
    "ping",
}


def build_url_policy(
    allowed_tags: Collection[str],
    allowed_attributes: Mapping[str, Collection[str]],
) -> UrlPolicy:
    rules = {}

    global_attrs = allowed_attributes.get("*", ())
    for tag in allowed_tags:
        for attr in global_attrs:
            if attr in URL_LIKE_ATTRS:
                rules[(tag, attr)] = UrlRule(
                    allowed_schemes={"http", "https", "mailto", "tel"},
                    allow_relative=True,
                )

    for tag, attrs in allowed_attributes.items():
        if tag == "*":
            continue
        for attr in attrs:
            if attr in URL_LIKE_ATTRS:
                rules[(tag, attr)] = UrlRule(
                    allowed_schemes={"http", "https", "mailto", "tel"},
                    allow_relative=True,
                )
    return UrlPolicy(allow_rules=rules)


def clean(
    html: str,
    *,
    tags: Collection[str] = (),
    attributes: Mapping[str, Collection[str]] | None = None,
    css_properties: Collection[str] = (),
    strip: bool = True,
    strip_comments: bool = True,
) -> str:
    attrs = dict(attributes or {})
    disallowed_tag_handling = "unwrap" if strip else "escape"

    # Bleach defaults to strip_comments=True. If you use escape mode
    # (`strip=False`) and still want comments removed rather than displayed,
    # remove comments before parsing.
    if strip_comments and disallowed_tag_handling == "escape":
        html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)

    policy = SanitizationPolicy(
        allowed_tags=tags,
        allowed_attributes=attrs,
        allowed_css_properties=css_properties,
        url_policy=build_url_policy(tags, attrs),
        drop_comments=strip_comments,
        disallowed_tag_handling=disallowed_tag_handling,
    )
    return JustHTML(html, fragment=True, policy=policy).to_html(pretty=False)


def linkify(text: str, *, nofollow: bool = False) -> str:
    transforms = [Linkify()]
    if nofollow:
        transforms.append(SetAttrs("a", rel="nofollow"))
    return JustHTML(
        text,
        fragment=True,
        sanitize=False,
        transforms=transforms,
    ).to_html(pretty=False)
```

Treat this as a compatibility shim, not a universal policy. Tighten `build_url_policy(...)` for your application, especially for attributes that load remote resources such as `img[src]`.

## Mental model differences

- Bleach takes a string and returns a cleaned string.
- JustHTML parses into a DOM and sanitizes by default at construction time:
    - `JustHTML(html)` sanitizes by default (`sanitize=True`).
    - `JustHTML(html, sanitize=False)` disables sanitization (trusted input only).

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
        default_handling="allow",
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
- URL-like attributes need explicit `(tag, attr)` allow rules even when the attribute itself is allowed.
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

If you need to display untrusted HTML with no HTML in the output, prefer `to_text()`, or escape the output before embedding it into an HTML page. `to_markdown()` runs on the sanitized DOM when `sanitize=True` (the default), but the value it returns is still Markdown source, not escaped HTML. Render it with a compliant Markdown renderer before embedding it into a page, or escape it first if you want to show the raw Markdown source. It may still include sanitized raw HTML for elements such as tables and images.

If you need additional structural cleanup beyond policy decisions, prefer doing it explicitly with transforms.

## Migration test expectations

Expect some output differences even when the security behavior is equivalent:

- Inline styles may be normalized, for example `vertical-align:top;` can serialize as `vertical-align: top`.
- Malformed HTML can serialize differently because JustHTML parses with its own HTML5 tree builder.
- `strip=False` maps to escaped disallowed tags. If your old Bleach call also relied on `strip_comments=True`, add an explicit test for comments.
- Linkification should be tested separately from sanitization when your old code used both `bleach.linkify(...)` and `bleach.clean(...)`.

## Suggested migration approach

1. Inventory existing `bleach.clean(...)`, `bleach.Cleaner(...)`, and `bleach.linkify(...)` call shapes.
2. For a large app, add a local compatibility wrapper so most call sites can move mechanically.
3. Start by using JustHTML’s default sanitizer (`JustHTML(...).to_html()`) for new code.
4. Add a `SanitizationPolicy` that matches your allowlist and URL requirements.
5. Add transforms to replace any Bleach filter/linkify behavior.
6. Add tests for your exact input corpus and output expectations (especially around URLs, allowed attributes, comments, and malformed HTML).
