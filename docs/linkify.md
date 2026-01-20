[← Back to docs](index.md)

# Linkify

JustHTML’s `Linkify` transform scans **text nodes** and wraps detected URLs/emails in `<a>` elements.

This is a DOM transform (it does not operate on raw HTML strings), so it never rewrites tag soup or breaks markup.

## Quickstart

```python
from justhtml import JustHTML, Linkify

doc = JustHTML("<p>See example.com</p>", fragment=True, transforms=[Linkify()])
print(doc.to_html(pretty=False))
# => <p>See <a href="http://example.com">example.com</a></p>
```

## Behavior

- Operates on DOM **text nodes** only.
- Inserts `<a href="...">…</a>` nodes around matches.
- By default, skips linkification inside: `a`, `pre`, `textarea`, `code`, `script`, `style`.
- Works inside `<template>` contents.

## Unicode and punycode (IDNA)

Linkify can detect domains containing Unicode characters.

When it generates a link, it normalizes the hostname portion of `href` using IDNA (punycode).
This keeps the visible link text readable while ensuring the `href` is ASCII-only.

Example:

```python
from justhtml import JustHTML, Linkify

doc = JustHTML("<p>See bücher.de</p>", fragment=True, transforms=[Linkify()])
print(doc.to_html(pretty=False))
# => <p>See <a href="http://xn--bcher-kva.de">bücher.de</a></p>
```

Notes:

- Only the **host** is punycoded; paths/queries remain Unicode.
- Punycode normalization is applied for `http://`, `https://`, `ftp://`, and protocol-relative `//...` URLs.

## Configuration

```python
from justhtml import JustHTML, Linkify

doc = JustHTML(
    "<p>See 127.0.0.1 and example.dev</p>",
    transforms=[
        Linkify(
            fuzzy_ip=True,
            extra_tlds={"dev"},
            skip_tags={"a", "pre", "textarea", "code", "script", "style"},
        )
    ],
)
```

Options:

- `skip_tags`: iterable of tag names to skip (matched case-insensitively).
- `fuzzy_ip`: enable linkifying bare IPv4 addresses like `192.168.0.1`.
- `extra_tlds`: additional TLDs to accept for fuzzy domain/email detection.
- `enabled` (default: `True`): if set to `False`, Linkify is skipped.

## Fuzzy domains and TLD allowlist

For protocol-less “fuzzy” detection (like `example.com` or `test@example.com`), Linkify uses a TLD allowlist to reduce false positives.

This allowlist is **not** used for links that already include an explicit scheme like `http://...` (those are accepted regardless of TLD).
Similarly, `mailto:` links are accepted even when the domain doesn’t have a recognized TLD.

### Default accepted TLDs

By default, Linkify accepts:

- All valid two-letter ccTLDs (like `se`, `uk`, `de`, …).
- Any punycode TLD starting with `xn--...`.
- A small built-in set of common generic TLDs:
    `biz`, `com`, `edu`, `gov`, `net`, `org`, `pro`, `web`, `xxx`, `aero`, `asia`, `coop`, `info`, `museum`, `name`, `shop`, `рф`.

### Adding extra TLDs

If you want fuzzy matching for newer gTLDs (like `.dev`, `.app`, `.email`, …), pass them via `extra_tlds`:

```python
from justhtml import JustHTML, Linkify

doc = JustHTML(
        "<p>See example.dev and mail me@company.app</p>",
        transforms=[Linkify(extra_tlds={"dev", "app"})],
)
```

`extra_tlds` values are compared case-insensitively and should be provided without a leading dot.

## Composing with other transforms

To add attributes to generated links, compose with `SetAttrs`:

```python
from justhtml import JustHTML, Linkify, SetAttrs

doc = JustHTML(
    "<p>See example.com</p>",
    transforms=[
        Linkify(),
        SetAttrs("a", rel="nofollow", target="_blank"),
    ],
)
```

## Interaction with safe output (sanitization)

Transforms mutate the in-memory DOM. `JustHTML(..., safe=True)` appends a final `Sanitize(...)` step unless you include one yourself.

This matters for Linkify because sanitization policies can remove or rewrite attributes on the generated `<a>` when the final sanitizer runs:

- Schemes not allowed for `a[href]` are stripped (the `<a>` remains, but `href` is removed).
- Protocol-relative `//example.com` is resolved according to policy (default: `https://example.com`).

If you want Linkify output without any sanitization changes (trusted input only), use `safe=False` and avoid adding `Sanitize(...)` in transforms.

## Provenance

JustHTML’s Linkify behavior is validated against the upstream `linkify-it` fixture suite (MIT licensed).

- Fixtures: `tests/linkify-it/fixtures/`
- License: `tests/linkify-it/LICENSE.txt`
