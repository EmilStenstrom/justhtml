[← Back to docs](index.md)

# Migrating from Bleach

JustHTML’s sanitization and transform pipeline were heavily inspired by Bleach’s real-world ergonomics.

Bleach has helped a lot of projects ship safer HTML over the years, and a lot of that is thanks to the hard work of [@willkg](https://github.com/willkg) building and maintaining it.

In June 2026, Bleach’s maintainer [announced Bleach’s final release](https://bluesock.org/~willkg/blog/dev/bleach_6_4_0_final_release.html). The announcement identifies JustHTML as an easy migration path and explicitly recommends swapping Bleach out for it.

This guide starts with Bleach's defaults, then builds up to a copyable compatibility wrapper for a gradual migration.

## Start with the defaults

Bleach and JustHTML are both safe by default, but their default policies are intentionally different. Do not replace `bleach.clean(html)` with `JustHTML(html).to_html()` and expect byte-for-byte identical output.

Bleach's default allowlist keeps a small set of formatting tags such as `b`, `em`, `i`, `a`, `code`, and lists. JustHTML's default policy allows a broader set of ordinary document tags and applies its own URL policy.

The wrapper below reproduces Bleach 6.x's default tags, attributes, and protocols without making Bleach a runtime dependency. It is a compatibility baseline, not a recommended final policy.

## Copyable `clean(...)` compatibility wrapper

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

BLEACH_DEFAULT_TAGS = frozenset(
    {"a", "abbr", "acronym", "b", "blockquote", "code", "em", "i", "li", "ol", "strong", "ul"}
)
BLEACH_DEFAULT_ATTRIBUTES = {
    "a": {"href", "title"},
    "abbr": {"title"},
    "acronym": {"title"},
}
BLEACH_DEFAULT_PROTOCOLS = frozenset({"http", "https", "mailto"})


def build_url_policy(
    allowed_tags: Collection[str],
    allowed_attributes: Mapping[str, Collection[str]],
    protocols: Collection[str],
) -> UrlPolicy:
    rules = {}

    global_attrs = allowed_attributes.get("*", ())
    for tag in allowed_tags:
        for attr in global_attrs:
            if attr in URL_LIKE_ATTRS:
                rules[(tag, attr)] = UrlRule(
                    allowed_schemes=protocols,
                    allow_relative=True,
                )

    for tag, attrs in allowed_attributes.items():
        if tag == "*":
            continue
        for attr in attrs:
            if attr in URL_LIKE_ATTRS:
                rules[(tag, attr)] = UrlRule(
                    allowed_schemes=protocols,
                    allow_relative=True,
                )
    return UrlPolicy(allow_rules=rules)


def clean(
    html: str,
    *,
    tags: Collection[str] = BLEACH_DEFAULT_TAGS,
    attributes: Mapping[str, Collection[str]] = BLEACH_DEFAULT_ATTRIBUTES,
    protocols: Collection[str] = BLEACH_DEFAULT_PROTOCOLS,
    css_properties: Collection[str] = (),
    strip: bool = False,
    strip_comments: bool = True,
) -> str:
    attrs = {tag: set(values) for tag, values in attributes.items()}
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
        url_policy=build_url_policy(tags, attrs, protocols),
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

Use the wrapper with Bleach-compatible defaults:

```python
# Replace existing bleach.clean(user_html) calls with clean(user_html).
safe_html = clean(user_html)
```

## Customize the wrapper gradually

Keep the call shape familiar while making the policy explicit at the call site:

```python
safe_html = clean(
    user_html,
    tags={"p", "b", "a"},
    attributes={"a": {"href"}},
    protocols={"http", "https"},
    strip=True,
)
```

`fragment=True` inside the wrapper keeps user-generated snippets free of `<html>`, `<head>`, and `<body>` wrappers. For URL rules with host restrictions, proxies, `srcset`, and remote-resource controls, extend the local `build_url_policy(...)` function; see [URL Cleaning](url-cleaning.md).

## Intentional compatibility differences

The wrapper matches Bleach's ordinary allowlist behavior, but it deliberately does not preserve every historical output detail:

- JustHTML drops the contents of dangerous containers such as `script` and `style`; Bleach with `strip=True` can leave their text content behind.
- JustHTML normalizes a protocol-relative URL such as `//example.com/x` to `https://example.com/x`; Bleach preserves the original spelling.
- JustHTML parses into a DOM and sanitizes at construction time. Add `Sanitize(...)` after later DOM edits if the tree needs to become safe again.

JustHTML also supports constructor-time transforms, which replace Bleach/html5lib filter pipelines; see [Transforms](transforms.md).

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
