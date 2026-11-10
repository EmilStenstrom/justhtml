[← Back to docs](index.md)

# Unsafe handling

The sanitizer can strip unsafe content, collect findings, or raise an error.

For tag/attribute allowlists and inline styles, see [HTML Cleaning](html-cleaning.md).
For URL validation and URL handling, see [URL Cleaning](url-cleaning.md).

Choose the behavior with `SanitizationPolicy(unsafe_handling=...)`.

<a id="strip-default--unsafe_handlingstrip"></a>

## Strip (default): unsafe_handling="strip"

The sanitizer removes unsafe constructs and continues processing.

Use this mode when you want best-effort cleaned output and don't need a list of what was removed.

<a id="collect-optional--unsafe_handlingcollect"></a>

## Collect (optional): unsafe_handling="collect"

If you want to keep sanitizing but also get a list of what was removed, set `unsafe_handling="collect"`.

Collected findings are exposed as parse-style errors with `category == "security"` and are available in `doc.errors` after construction.

To include `line`/`column` information in collected findings, pass `track_node_locations=True` to `JustHTML(...)`. Otherwise, locations may be missing. Location tracking adds some processing time and is disabled by default.

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

<!-- justhtml: output -->

```text
Unsafe tag 'script' (dropped content)
```

<a id="raise-optional--unsafe_handlingraise"></a>

## Raise (optional): unsafe_handling="raise"

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

<!-- justhtml: output -->

```text
Unsafe tag 'script' (dropped content)
```
