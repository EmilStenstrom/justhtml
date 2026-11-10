[← Back to docs](index.md)

# HTML cleaning

JustHTML uses a policy to sanitize untrusted HTML before rendering.

HTML cleaning controls which tags, attributes, and inline styles the sanitizer allows. For URL validation and rewriting, see [URL Cleaning](url-cleaning.md).

On this page:

- [Safe-by-default construction](#safe-by-default-construction)
- [Default policy](#default-sanitization-policy)
- [Custom policy](#use-a-custom-sanitization-policy)
- [Sanitizing a DOM directly](#sanitizing-a-dom-directly)
- [Inline styles](#inline-styles-optional)
- [Advanced policy options](#advanced-policy-options)
- [Disable sanitization](#disable-sanitization)
- [Reporting issues](#reporting-issues)

## Safe-by-default construction

By default, JustHTML sanitizes input using its built-in allowlist policy:

### HTML output

```python
from justhtml import JustHTML

user_html = '<p>Hello <b>world</b> <script>alert(1)</script> <a href="javascript:alert(1)">bad</a> <a href="https://example.com/?a=1&b=2">ok</a></p>'
doc = JustHTML(user_html, fragment=True)
print(doc.to_html())
```

<!-- justhtml: output -->

```html
<p>Hello <b>world</b>  <a>bad</a> <a href="https://example.com/?a=1&amp;b=2">ok</a></p>
```

### Markdown output

```python
from justhtml import JustHTML

user_html = '<p>Hello <b>world</b> <script>alert(1)</script> <a href="javascript:alert(1)">bad</a> <a href="https://example.com/?a=1&b=2">ok</a></p>'
doc = JustHTML(user_html, fragment=True)
print(doc.to_markdown())
```

<!-- justhtml: output -->

```markdown
Hello **world** [bad] [ok](https://example.com/?a=1&b=2)
```

## Default sanitization policy

The built-in default is `DEFAULT_POLICY` (a conservative allowlist).

The default URL policy allows common link schemes for `a[href]`. For `img[src]`, it allows only relative URLs. To allow remote image URLs, configure a custom policy. See [URL Cleaning](url-cleaning.md).

High-level behavior:

- Disallowed tags are stripped (their children may be kept) but dangerous containers like `script`/`style` have their content dropped.
- Comments and doctypes are dropped.
- Foreign namespaces (SVG/MathML) are always dropped by the sanitizer, even if the tags are included in a custom allowlist. Parsing and treebuilding still support foreign content when `sanitize=False`.
- Invisible Unicode commonly used for obfuscation, including variation selectors, zero-width/bidi controls, and private-use characters, is stripped from text and attributes before other sanitizer checks run.
- Event handlers (`on*`), `srcdoc`, and namespace-style attributes (anything with `:`) are removed.
- Inline styles are disabled by default.

### Disallowed tags

Disallowed tag handling is controlled by `SanitizationPolicy(disallowed_tag_handling=...)`:

- `"unwrap"` (default): remove the disallowed tag, keep/sanitize its children
- `"escape"`: emit the disallowed tag’s start/end tags as escaped text, keep/sanitize its children
- `"drop"`: drop the entire disallowed subtree

Default allowlists:

- Allowed tags: `a`, `img`, common text/structure tags, headings, lists, and tables (`table`, `thead`, `tbody`, `tfoot`, `tr`, `th`, `td`).
- Allowed attributes:
  - Global: `class`, `id`, `title`, `lang`, `dir`
  - `a`: `href`, `title`
  - `img`: `src`, `alt`, `title`, `width`, `height`, `loading`, `decoding`
  - `th`/`td`: `colspan`, `rowspan`

## Use a custom sanitization policy

Start with the smallest policy that matches the HTML you want to accept. This makes the allowed output clear to future maintainers.

```python
from justhtml import JustHTML, SanitizationPolicy, UrlPolicy, UrlRule

policy = SanitizationPolicy(
    allowed_tags={"p", "strong", "a"},
    allowed_attributes={"a": {"href"}},
    url_policy=UrlPolicy(
        allow_rules={
            ("a", "href"): UrlRule(allowed_schemes={"https", "mailto"}),
        }
    ),
)

safe_html = JustHTML(user_html, fragment=True, policy=policy).to_html()
```

The three settings work together:

- `allowed_tags` controls which elements remain.
- `allowed_attributes` controls which attribute names remain. An omitted tag has no allowed attributes, so you only need to list tags that accept attributes.
- URL-valued attributes such as `href` also need a `UrlRule`; allowing the attribute name alone is not enough.

For URL schemes, hosts, proxies, `srcset`, and other URL-specific controls, continue with [URL Cleaning](url-cleaning.md).

### Restrict attribute values

For each tag in `allowed_attributes`, use a set of attribute names for the normal
sanitizer checks, or a mapping to sets of permitted literal values and
`Regex(...)` whole-value patterns:

```python
from justhtml import JustHTML, Regex, SanitizationPolicy

policy = SanitizationPolicy(
    allowed_tags={"p", "a"},
    allowed_attributes={
        "p": {
            "id": {Regex(r"note-[0-9]+")},
            "dir": {"ltr", "rtl", "auto"},
        },
        "a": {"title"},
    },
)
print(JustHTML(
    '<p id="note-42" dir="sideways" title="example">Hello</p>',
    fragment=True,
    policy=policy,
).to_html())
```

<!-- justhtml: output -->

```html
<p id="note-42">Hello</p>
```

Plain strings match exactly, including case and whitespace. Patterns match the
entire attribute value, including an entire `class` list rather than individual
tokens. HTML character references are decoded before matching, and the policy's
invisible-Unicode stripping runs first when enabled. Attribute values are not
trimmed. An empty rule set permits no values; `{""}` permits an empty or valueless
attribute. Attributes remain optional.

Global (`"*"`) and tag-specific permissions are additive. If both constrain an
attribute, their permitted values are combined. A name-only permission in either
scope permits any value through the normal checks; an empty set in the other
scope does not revoke it. Avoid listing an attribute globally without constraints
if you intend to restrict its values on a particular tag.

Matching does not bypass URL validation or inline-style sanitization. These
checks may subsequently remove or rewrite accepted values. `force_link_rel` may
also add tokens after value matching. Output still uses normal HTML escaping.
Regex patterns are trusted application configuration and have the same
[backtracking considerations as CSS value patterns](#restrict-css-values).

## Sanitizing the in-memory DOM

The parsed DOM is sanitized by default at construction time (`JustHTML(..., sanitize=True)`), and serialization is a pure output step.

If you want to sanitize after other transforms or after direct DOM edits, add `Sanitize(...)` at the point where the tree should become safe. Later transforms can reintroduce unsafe content. For explicit pass boundaries (advanced use), see [`Stage([...])`](transforms.md#advanced-stages).

```python
from justhtml import JustHTML, Sanitize

doc = JustHTML(user_html, fragment=True, transforms=[Sanitize()])
print(doc.to_html(pretty=False))
```

## Inline styles (optional)

Inline styles are disabled by default. To allow them you must:

1. Allow the `style` attribute for the relevant tag via `allowed_attributes`.
2. Provide a non-empty allowlist via `allowed_css_properties`.

Even then, JustHTML rejects declarations that look like they can load external resources (such as values containing `url(` or `image-set(`), as well as legacy constructs like `expression(`. Start from the conservative `CSS_PRESET_TEXT` preset.

```python
from justhtml import CSS_PRESET_TEXT, JustHTML, SanitizationPolicy, UrlPolicy

policy = SanitizationPolicy(
    allowed_tags={"p"},
    allowed_attributes={"p": {"style"}},
    url_policy=UrlPolicy(allow_rules={}),
    allowed_css_properties=CSS_PRESET_TEXT | {"width"},
)

html = '<p style="color: red; background-image: url(https://evil.test/x); width: expression(alert(1));">Hi</p>'
print(JustHTML(html, policy=policy).to_html())
```

### Restrict CSS values

Pass a mapping to restrict values for individual properties. Use `True` to allow
a property with the normal sanitizer checks and no additional value restriction.
Plain strings in value collections are literal, case-sensitive values. Wrap a
pattern in `Regex(...)` to match the whole value; JustHTML compiles it when the
`Regex` is constructed.

```python
from justhtml import JustHTML, Regex, SanitizationPolicy

policy = SanitizationPolicy(
    allowed_tags={"p"},
    allowed_attributes={"p": {"style"}},
    allowed_css_properties={
        "color": {
            "var(--primary-color)",
            Regex(r"#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?"),
        },
        "background-color": {"var(--surface-color)"},
        "font-family": True,
    },
)
print(JustHTML(
    '<p style="color: var(--primary-color); background-color: red; font-family: Arial">Hello</p>',
    fragment=True,
    policy=policy,
).to_html())
```

<!-- justhtml: output -->

```html
<p style="color: var(--primary-color); font-family: Arial">Hello</p>
```

For properties with value collections, normal colors such as `red` are rejected
unless a literal or pattern matches them. Values are matched after trimming
surrounding HTML ASCII whitespace. An empty value set allows no values for that property;
properties absent from the mapping are disallowed. The original collection form,
such as `allowed_css_properties={"color"}`, behaves like `{"color": True}`.
`True` does not authorize CSS variables or bypass URL checks. `False` and `None`
are invalid; omit a property to disallow it.

A matching rule explicitly permits CSS variable references, including fallbacks
if the rule matches them. Variable names retain their case. Permission to
reference `--primary-color` does not permit a `--primary-color: ...` declaration;
that property would need its own explicit permission.

**Variable definitions are trusted application input.** Variables resolve in the
browser after sanitization. With `background-image: var(--image)`, for example,
an external URL in the variable definition can cause a request without passing
through JustHTML's URL validation, filtering, or proxying. The application must
also trust indirect references and fallbacks in those variable definitions.
Only permit these values when the application controls their definitions and
accepts their effects. The property using the variable determines its effects,
regardless of the variable's name.

Value rules do not bypass declaration structure checks or checks on directly
supplied URLs. Matching values with comments, CSS escapes, control characters, or
unbalanced delimiters are rejected conservatively. Existing bans on constructs
such as `expression()` and `image-set()` still apply, as do the URL rules for
literal `url(...)` values. URL validation/proxying may rewrite an accepted URL.

Regex patterns are trusted configuration, but run against untrusted input.
Prefer bounded patterns such as the hex-color pattern above. Python regexes can
backtrack excessively; JustHTML does not provide a regex timeout. Apply input
size and execution limits appropriate to the application.

## Advanced policy options

Selector limits are for trusted pipelines that need to accept unusually large selectors or match budgets:

```python
from justhtml import SanitizationPolicy
from justhtml.selector import SelectorLimits

policy = SanitizationPolicy(
    allowed_tags={"div", "p"},
    allowed_attributes={"*": {"class"}},
    selector_limits=SelectorLimits(max_length=20_000),
)
```

Treat policies that allow active content as a separate security review: `iframe`, `object`, `embed`, `meta`, `link`, `base`, form elements, and their active attributes are preserved when you explicitly allow them.

## Disable sanitization

Only disable sanitization for HTML you fully trust:

```python
doc = JustHTML(trusted_html, fragment=True, sanitize=False)
```

## Reporting issues

If you find a sanitizer bypass, please report it responsibly (see [SECURITY.md](../SECURITY.md)).
