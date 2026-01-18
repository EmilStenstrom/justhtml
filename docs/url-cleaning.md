[← Back to docs](index.md)

# URL Cleaning

This page focuses on **URL cleaning**: how JustHTML validates and rewrites URL-valued attributes like `a[href]` or `img[src]`.

For tag/attribute allowlists, inline styles, and unsafe-handling modes, see [HTML Cleaning](html-cleaning.md).


On this page:

- [Key idea: URL-like attributes require explicit rules](#key-idea-url-like-attributes-require-explicit-rules)
- [How URL cleaning works (in order)](#how-url-cleaning-works-in-order)
- [UrlPolicy: URL allowlisting and defaults](#urlpolicy-url-allowlisting-and-defaults)
    - [Allow all URL-like attributes (default_handling=`"allow"`)](#default_handling-allow)
    - [Default: Strip all URL-like attributes (default_handling=`"strip"`)](#default_handling-strip)
    - [Proxy all URL-like attributes (default_handling=`"proxy"`)](#default_handling-proxy)
    - [Protocol-relative URLs](#protocol-relative-urls)
    - [Special handling: srcset](#special-handling-srcset)
    - [url_filter hook](#url_filter-hook)
- [UrlRule: validation for a single (tag, attr)](#urlrule-validation-for-a-single-tag-attr)
    - [Common URL rules](#common-url-rules)

## Key idea: URL-like attributes require explicit rules

JustHTML treats a set of attributes as *URL-like* (including `href`, `src`, `srcset`, `action`, and a few others).

The reason is that these attributes can trigger navigation or resource loading (and in some cases script execution via unsafe schemes like `javascript:`). Different attributes also have different security expectations: for example, allowing `a[href]` is often fine, while allowing `img[src]` can cause remote requests/tracking. Requiring an explicit `(tag, attr)` rule forces you to opt in and define what is considered a valid URL for that specific attribute.

For safety, these attributes are **only kept** if there is an explicit matching rule in `UrlPolicy(allow_rules=...)` for the `(tag, attr)` pair.

```python
from justhtml import JustHTML, SanitizationPolicy

policy = SanitizationPolicy(
    allowed_tags=["img"],
    allowed_attributes={"img": ["src"]},
)

print(JustHTML("""
    <img src="https://example.com">
    <img src="https://attacker.com">
""", fragment=True, policy=policy).to_html())
```

Output:

```html
<img>
<img>
```

Since no urlpolicy was set, the default kicked in, and deleted any URL-like attribute. It's not enough to allow an attribute if it's "URL-like", you need to add a url_policy, matching what you want to allow:

```python
from justhtml import JustHTML, SanitizationPolicy, UrlPolicy, UrlRule

policy = SanitizationPolicy(
    allowed_tags=["img"],
    allowed_attributes={"img": ["src"]},
    url_policy=UrlPolicy(
        allow_rules={
            ("img", "src"): UrlRule(
                allowed_schemes={"https"},
                allowed_hosts=["example.com"],
            ),
        }
    )
)

print(JustHTML("""
    <img src="https://example.com">
    <img src="http://example.com">
    <img src="https://attacker.com">
""", fragment=True, policy=policy).to_html())
```

Output:

```html
<img src="https://example.com">
<img>
<img>
```

## How URL cleaning works (in order)

For a URL-like attribute (like `img[src]` or `a[href]`), JustHTML applies these steps:

1. The tag must be allowed by `SanitizationPolicy.allowed_tags`.
2. The attribute name must be allowed by `SanitizationPolicy.allowed_attributes`.
3. The attribute must have an explicit matching rule in `UrlPolicy(allow_rules=...)`.
4. (If configured: `UrlPolicy.url_filter` runs and can rewrite or drop the value here).
5. The value is normalized and validated by the matching `UrlRule`.
6. If it validates the *effective* URL handling is applied:
    - if `UrlRule.handling` is set, it is applied
    - otherwise the URL is kept ("allow")

## UrlPolicy: URL allowlisting and defaults

URL behavior is controlled by `UrlPolicy`:

- `default_handling`: the default action for URL-like attributes.
- `default_allow_relative`: whether **relative** URLs (like `/path`, `./path`, `../path`, `?q`) are allowed by default.

For URL-like attributes that match an explicit `(tag, attr)` rule in `allow_rules`, URLs are kept by default once they pass validation. To strip or proxy a specific attribute, set `UrlRule.handling`.

Note: URL validation is always enforced by `UrlRule`.

```python
from justhtml import UrlPolicy

UrlPolicy(
    default_handling="strip",  # or "allow" / "proxy"
    default_allow_relative=True,
    allow_rules={},
    url_filter=None,
    proxy=None,
)
```

<a id="default_handling-allow"></a>
### Allow all URL-like attributes (default_handling=`"allow"`)

This is the “keep validated URLs” behavior.

For URL-like attributes that match an explicit `(tag, attr)` rule in `UrlPolicy(allow_rules=...)`, a validated URL is kept by default unless you override handling with `UrlRule.handling`.

<a id="default_handling-strip"></a>
### Default: Strip all URL-like attributes (default_handling=`"strip"`)

Some renderers (notably email clients) want to avoid loading remote resources by default.

The built-in `DEFAULT_POLICY` already blocks remote image loads by default (`img[src]` only allows relative URLs).

To strip URL-valued attributes, either omit the `(tag, attr)` rule (so the attribute is dropped), or set `UrlRule(handling="strip")` for that attribute.

```python
from justhtml import JustHTML, SanitizationPolicy, UrlPolicy, UrlRule

policy = SanitizationPolicy(
    allowed_tags=["img"],
    allowed_attributes={"*": [], "img": ["src"]},
    url_policy=UrlPolicy(
        allow_rules={("img", "src"): UrlRule(handling="strip", allowed_schemes={"http", "https"})},
    ),
)

print(JustHTML('<img src="https://example.com/x">', fragment=True, policy=policy).to_html())
print(JustHTML('<img src="/x">', fragment=True, policy=policy).to_html())
```

Output:

```html
<img>
<img>
```

If you instead want to block remote loads but allow relative image loads, configure the rule:

```python
from justhtml import JustHTML, SanitizationPolicy, UrlPolicy, UrlRule

policy = SanitizationPolicy(
    allowed_tags=["img"],
    allowed_attributes={"*": [], "img": ["src"]},
    url_policy=UrlPolicy(
        allow_rules={
            ("img", "src"): UrlRule(
                allow_relative=True,
                allowed_schemes=set(),
                resolve_protocol_relative=None,
            )
        },
    ),
)

print(JustHTML('<img src="https://example.com/x">', fragment=True, policy=policy).to_html())
print(JustHTML('<img src="/x">', fragment=True, policy=policy).to_html())
```

Output:

```html
<img>
<img src="/x">
```

<a id="default_handling-proxy"></a>
### Proxy all URL-like attributes (default_handling=`"proxy"`)

Instead of keeping URLs, you can rewrite them through a proxy endpoint:

```python
from justhtml import JustHTML, SanitizationPolicy, UrlPolicy, UrlProxy, UrlRule

policy = SanitizationPolicy(
    allowed_tags=["a"],
    allowed_attributes={"*": [], "a": ["href"]},
    url_policy=UrlPolicy(
        proxy=UrlProxy(url="/proxy", param="url"),
        allow_rules={
            ("a", "href"): UrlRule(handling="proxy", allowed_schemes={"https"}),
        },
    ),
)

print(JustHTML('<a href="https://example.com/?a=1&b=2">link</a>', policy=policy).to_html())
```

Output:

```html
<a href="/proxy?url=https%3A%2F%2Fexample.com%2F%3Fa%3D1%26b%3D2">link</a>
```

Notes:

- URL validation still happens before rewriting (schemes/hosts are still enforced).
- In proxy mode, relative URLs are also rewritten if the effective `allow_relative=True`.
- In proxy mode, a proxy must be configured either globally (`UrlPolicy.proxy`) or per rule (`UrlRule.proxy`).

Example: using a per-rule proxy override:

```python
from justhtml import JustHTML, SanitizationPolicy, UrlPolicy, UrlProxy, UrlRule

policy = SanitizationPolicy(
    allowed_tags=["a"],
    allowed_attributes={"*": [], "a": ["href"]},
    url_policy=UrlPolicy(
        allow_rules={
            ("a", "href"): UrlRule(
                handling="proxy",
                allowed_schemes={"https"},
                proxy=UrlProxy(url="/p", param="u"),
            )
        },
    ),
)

print(JustHTML('<a href="https://example.com/?a=1&b=2">link</a>', policy=policy).to_html())
```

Output:

```html
<a href="/p?u=https%3A%2F%2Fexample.com%2F%3Fa%3D1%26b%3D2">link</a>
```

### Protocol-relative URLs

Protocol-relative URLs start with `//`, and are relatively unknown. Browsers resolve them to "https" if you are on a https-enabled site, and "http" otherwise.

By default, justhtml resolves them to `https` before validation. This ensures they are checked against allowed schemes and prevents inheriting an insecure protocol from the embedding page.

You can configure this behavior per rule:

```python
from justhtml import UrlRule

# Default behavior: resolve to https
rule = UrlRule(allowed_schemes=["https"], resolve_protocol_relative="https")

# Resolve to http
rule = UrlRule(allowed_schemes=["http", "https"], resolve_protocol_relative="http")

# Disallow protocol-relative URLs entirely
rule = UrlRule(allowed_schemes=["https"], resolve_protocol_relative=None)
```

There is currently no way to leave protocol relative URLs untouched. If this is something you need, open an issue with a desciption of your use-case.

### Special handling: srcset

`srcset` contains **multiple URLs**, so it requires special care.

JustHTML parses the comma-separated candidates and sanitizes each candidate URL using the matching `UrlRule` for `(tag, "srcset")`.

If any candidate is unsafe, the entire attribute is dropped.

### url_filter hook

`UrlPolicy.url_filter` lets you apply a last-mile filter/rewrite (or drop) based on `(tag, attr, value)`.

- Return a string to keep it (possibly rewritten).
- Return `None` to drop the attribute.

This runs before validation.

Example: drop URLs to a blocked host:

```python
from justhtml import JustHTML, SanitizationPolicy, UrlPolicy, UrlRule


def url_filter(tag: str, attr: str, value: str) -> str | None:
    if "attacker.com" in value:
        return None
    return value


policy = SanitizationPolicy(
    allowed_tags=["a"],
    allowed_attributes={"*": [], "a": ["href"]},
    url_policy=UrlPolicy(
        url_filter=url_filter,
        allow_rules={
            ("a", "href"): UrlRule(
                allowed_schemes={"https"},
            )
        },
    ),
)

html = '<a href="https://example.com/">ok</a>\n<a href="https://attacker.com/">bad</a>'
print(JustHTML(html, fragment=True, policy=policy).to_html())
```

Output:

```html
<a href="https://example.com/">ok</a>
<a>bad</a>
```



## UrlRule: validation for a single (tag, attr)

A `UrlRule` controls how a single URL-valued attribute is validated:

```python
from justhtml import UrlRule

UrlRule(
    allow_fragment=True,
    resolve_protocol_relative="https",
    allowed_schemes=set(),
    allowed_hosts=None,
    handling=None,
    allow_relative=None,
    proxy=None,
)
```

Field reference:

- `allow_fragment` (default: `True`): allow same-document fragments like `#section`.
- `resolve_protocol_relative` (default: `"https"`): how to resolve protocol-relative URLs like `//example.com` before validation; set to `None` to reject them.
- `allowed_schemes` (default: `set()`): allowed schemes for absolute URLs (lowercased), e.g. `{"https"}`; empty means disallow all absolute URLs.
- `allowed_hosts` (default: `None`): optional host allowlist for absolute URLs; if set, the parsed host must be in this set.
- `handling` (default: `None`): optional handling override for an allowlisted attribute; `"strip"` drops it, `"proxy"` rewrites it, and `None` keeps it after validation.
- `allow_relative` (default: `None`): optional override for `UrlPolicy.default_allow_relative` (relative URLs like `/x`, `./x`, `?q`).
- `proxy` (default: `None`): optional per-rule proxy config used when effective handling is `"proxy"` (overrides `UrlPolicy.proxy`).

### Common URL rules

These are small `UrlRule(...)` building blocks that you can use in `UrlPolicy(allow_rules={...})` for a specific `(tag, attr)` pair.

- Allow all HTTPS links:

```python
UrlRule(allowed_schemes={"https"})
```

- Allow HTTP and HTTPS links:

```python
UrlRule(allowed_schemes={"http", "https"})
```

- Allow only your own host:

```python
UrlRule(allowed_schemes={"https"}, allowed_hosts={"example.com"})
```

- Allow your main host and a CDN host:

```python
UrlRule(
    allowed_schemes={"https"},
    allowed_hosts={"example.com", "static.example.com"},
)
```

- Allow only relative URLs (block remote loads):

```python
UrlRule(allow_relative=True)
```

- Allow only fragments (e.g. `#section`) and drop everything else:

```python
UrlRule(allow_fragment=True)
```

- Allow HTTPS but disallow same-document fragments:

```python
UrlRule(allowed_schemes={"https"}, allow_fragment=False)
```

- Allow relative URLs and HTTPS to a specific host:

```python
UrlRule(
    allow_relative=True,
    allowed_schemes={"https"},
    allowed_hosts={"example.com"},
)
```

- Allow HTTPS but disallow protocol-relative URLs (`//example.com`) entirely:

```python
UrlRule(
    allowed_schemes={"https"},
    resolve_protocol_relative=None,
)
```

- Allow only `mailto:` links:

```python
UrlRule(allowed_schemes={"mailto"})
```

- Allow only `tel:` links:

```python
UrlRule(allowed_schemes={"tel"})
```

- Allow `https:` and `mailto:` (common for `a[href]`):

```python
UrlRule(allowed_schemes={"https", "mailto"}, resolve_protocol_relative="https")
```

- Strip a URL-valued attribute even if it validates (explicit drop):

```python
UrlRule(handling="strip", allowed_schemes={"https"})
```

- Proxy validated URLs:

```python
# Uses UrlPolicy.proxy (global proxy config)
UrlRule(handling="proxy", allowed_schemes={"https"})
```

```python
from justhtml import UrlProxy

# Uses a per-rule proxy override (UrlRule.proxy takes precedence over UrlPolicy.proxy)
UrlRule(
    handling="proxy",
    allowed_schemes={"https"},
    proxy=UrlProxy(url="/proxy", param="url"),
)
```
