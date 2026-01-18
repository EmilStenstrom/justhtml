[← Back to docs](index.md)

# Unsafe Handling

This page focuses on **what happens when unsafe input is encountered** during sanitization.

For tag/attribute allowlists and inline styles, see [HTML Cleaning](html-cleaning.md).
For URL validation and URL handling, see [URL Cleaning](url-cleaning.md).

Unsafe behavior is controlled by `SanitizationPolicy(unsafe_handling=...)`.

## Strip (default) — unsafe_handling="strip"

Unsafe constructs are removed/dropped and sanitization continues.

Use this mode when you want best-effort cleaned output and don't need a list of what was removed.

## Collect (optional) — unsafe_handling="collect"

If you want to keep sanitizing but also get a list of what was removed, set `unsafe_handling="collect"`.

Collected findings are exposed as parse-style errors with `category == "security"` and are available in `doc.errors` after construction.

Tip: pass `track_node_locations=True` to `JustHTML(...)` to include `line`/`column` information in the collected findings (otherwise location may be missing). This comes with a slight performance penalty, which is why it's disabled by default.

```python
from justhtml import JustHTML, SanitizationPolicy, UrlPolicy

policy = SanitizationPolicy(
    allowed_tags=["p"],
    allowed_attributes={"*": []},
    url_policy=UrlPolicy(allow_rules={}),
    unsafe_handling="collect",
)

doc = JustHTML(
    "<p>ok</p><script>alert(1)</script>",
    fragment=True,
    track_node_locations=True,
    policy=policy,
)
for e in doc.errors:
    if e.category == "security":
        print(e.message)
```

Output:

```text
Unsafe tag 'script' (dropped content)
```

## Raise (optional) — unsafe_handling="raise"

If you want to treat unsafe HTML as an error instead of stripping it, set `unsafe_handling="raise"`.

In this mode, the sanitizer raises `UnsafeHtmlError` at the first unsafe construct it encounters.

```python
from justhtml import JustHTML, SanitizationPolicy, UnsafeHtmlError, UrlPolicy

doc = JustHTML("<p>ok</p><script>alert(1)</script>", fragment=True)

policy = SanitizationPolicy(
    allowed_tags=["p"],
    allowed_attributes={"*": []},
    url_policy=UrlPolicy(allow_rules={}),
    unsafe_handling="raise",
)

try:
    _ = JustHTML("<p>ok</p><script>alert(1)</script>", fragment=True, policy=policy)
except UnsafeHtmlError as e:
    print(e)
```

Output:

```text
Unsafe tag 'script' (dropped content)
```
