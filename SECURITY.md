# Security Policy

## Contents

- [Supported Versions](#supported-versions)
- [For Application Developers](#for-application-developers)
- [Security Domains](#security-domains)
- [Reporting a Vulnerability](#reporting-a-vulnerability)
  - [Reporting](#reporting)
  - [Response Time](#response-time)
  - [Disclosure Policy](#disclosure-policy)
  - [Recognition](#recognition)

## Supported Versions

| Version | Supported                                |
| ------- | ---------------------------------------- |
| 1.x     | :white_check_mark: (until 2.0 is released) |
| < 1.0   | :x:                                      |

## Security Domains

This section describes the security areas we use when reviewing `justhtml`.
Each area has a short promise: what users can rely on when they use the public
API as documented.

These promises apply only to their intended context. Do not treat sanitized HTML
for a page body as automatically safe inside JavaScript strings, URL attributes,
CSS, or other contexts. See [Sanitization & Security](docs/sanitization.md) for
guidance on where different outputs are safe to use.

### 1. Untrusted HTML Ingestion

Summary: JustHTML parses malformed or hostile HTML without treating it as
trusted content.

Scope: input decoding, tokenization, tree construction, fragment/document mode,
strict mode, and parse errors.

Promise:
- JustHTML decodes bytes using HTML-style encoding detection and rejects UTF-7.
- JustHTML parses malformed HTML into a DOM using browser-style HTML5 recovery.
- The parser handles entity and character-reference edge cases.
- The parser avoids known small-input patterns that can cause excessive work.
- JustHTML reports parse errors as data unless strict mode is enabled.
- Fragment parsing and full-document parsing keep their different assumptions
  explicit.

Out of scope:
- JustHTML does not prevent denial of service from very large input. Callers
  should apply their own size, time, or memory limits when needed.
- JustHTML does not guarantee that the raw parsed DOM is safe when sanitization
  is disabled.

### 2. Sanitization Policy Enforcement

Summary: By default, JustHTML removes or neutralizes unsafe tags and attributes
before callers use the result.

Scope: allowed tags and attributes, comments, doctypes, blocked tags, rawtext
elements, templates, SVG/MathML content, and the default sanitization path.

Promise:
- `JustHTML(html)` sanitizes by default.
- The sanitizer removes, unwraps, escapes, or drops tags and attributes outside
  the allowlist according to policy.
- Custom allowlists are authoritative. If a caller explicitly allows a tag or
  attribute, JustHTML preserves that choice instead of silently overriding it
  with extra hardcoded tag or attribute bans.
- The default policy blocks event handler attributes and executable tags.
- The default policy drops the contents of dangerous rawtext containers such as
  `<script>` and `<style>`.
- The sanitizer applies specific handling for rawtext and SVG/MathML content.
- The sanitizer strips invisible Unicode commonly used for obfuscation by
  default.
- The policy controls whether JustHTML strips, raises, or collects security
  findings.

Out of scope:
- Safety after callers pass `sanitize=False`, `safe=False`, or `--unsafe`.
- Safety after caller-provided transforms change the DOM after `Sanitize` has
  already run.
- Safety of custom allowlists that allow dangerous tags, attributes, URL
  schemes, CSS properties, or foreign content.
- Protecting callers from intentionally permissive custom policies. JustHTML
  does not try to outsmart an explicit allowlist; callers who allow dangerous
  browser features are responsible for that policy decision.

### 3. URL and CSS Handling

Summary: JustHTML keeps URLs in links, image sources, forms, and styles only
when the active policy allows them.

Scope: URL attributes, `srcset`, URL lists, URL functions, inline styles,
protocol-relative URLs, host allowlists, and optional URL proxying.

Promise:
- JustHTML keeps URL attributes only when they match an explicit URL rule.
- The default policy rejects dangerous schemes such as `javascript:`.
- JustHTML rejects backslash and control-character URL tricks conservatively.
- JustHTML either rejects protocol-relative URLs or resolves them before
  checking.
- The default policy does not allow inline styles.
- The sanitizer allowlists inline CSS by property and checks for URL-loading
  behavior.

Out of scope:
- Blocking every possible browser request if a custom policy allows remote URL
  attributes such as `img[src]`, `srcset`, `ping`, CSS URLs, or forms.
- JustHTML does not decide whether an allowed external host is trustworthy.
- URL cleaning does not make a value safe for JavaScript, CSS, or other
  non-URL contexts. Use the documented helper for the target context.

### 4. Serialization and Context Escaping

Summary: JustHTML escapes output for its target context, and callers choose the
right helper outside normal HTML body content.

Scope: `to_html()`, `to_text()`, `to_markdown()`, element and attribute name
validation, rawtext serialization, and helpers for JavaScript strings, HTML
attributes, and URL values.

Promise:
- HTML serialization escapes text and attribute values for HTML output.
- Serialization rejects unsafe element and attribute names created through
  custom DOM changes.
- Serialization neutralizes rawtext end-tag sequences.
- `to_text()` returns text content rather than markup.
- `to_markdown()` escapes Markdown-sensitive text by default.
- `to_markdown()` escapes line-start Markdown markers that could change block
  structure.
- `to_markdown()` wraps or encodes link destinations that could break Markdown
  link syntax.
- `to_markdown()` uses code fences long enough to contain backticks safely.
- `to_markdown()` drops `<script>`, `<style>`, and any `<textarea>` elements
  that remain in the DOM by default. Default sanitization may unwrap disallowed
  `<textarea>` elements and preserve their text content.
- JustHTML provides specific helpers for JavaScript strings, HTML attributes, and URL
  values.

Out of scope:
- Using output made for one context in another. For example, HTML-body output is
  not safe inside a `<script>` block, an event handler, a CSS string, or an
  unquoted attribute.
- Removing all raw HTML from Markdown output. `to_markdown()` can preserve
  sanitized tables and images as raw HTML; use `to_text()` when output must
  contain no markup.
- Making `html_passthrough=True` Markdown output safe for every Markdown
  renderer.

### 5. Transform Pipeline

Summary: Transforms can change the document, so safety depends on where
sanitization runs in the transform list.

Scope: construction-time transforms, transform order, selector-based DOM edits,
linkification, cleanup transforms, and explicit `Sanitize` placement.

Promise:
- When callers enable sanitization and supply no `Sanitize` transform, JustHTML
  adds a sanitization step to the end, automatically.
- If callers supply an explicit `Sanitize` transform, its position becomes the
  sanitization point for the transform pipeline. JustHTML does not add another
  sanitization step after later transforms.
- Built-in URL-aware transforms reuse sanitizer URL cleaning where they can.
- Linkification operates on DOM text nodes rather than raw HTML strings.

Out of scope:
- Safety of arbitrary user callbacks passed to transform APIs.
- Safety of transforms intentionally run after sanitization, including built-in
  transforms. If a transform can create or change markup, attributes, URLs, or
  text after `Sanitize`, callers must place another explicit `Sanitize` later
  when they need sanitized output.
- Treating transform APIs as a sandbox.

### 6. Selector Engine

Summary: Selectors only find nodes; they do not run code or decide what a user
may access.

Scope: CSS selector parsing and matching in the query API, CLI selection, and
transform targeting.

Promise:
- Invalid selectors fail with selector errors rather than executing code.
- Selector parsing and matching have internal limits to reduce excessive work.
- Selectors choose existing nodes; they do not by themselves make unsafe markup
  safe or unsafe.

### 7. Runtime Side Effects

Summary: The Python API processes HTML locally and does not perform network,
filesystem, script, or subprocess side effects.

Scope: parsing, sanitization, selector matching, transforms, and serialization
through the Python API.

Promise:
- The core library does not execute scripts or event handlers found in input
  HTML.
- The core library does not fetch URLs found in input HTML.
- The core library does not read from or write to the filesystem.
- The core library does not start subprocesses or execute shell commands.
- Runtime `justhtml` has no required third-party Python packages.

Out of scope:
- Side effects from caller-provided transform callbacks.
- Network requests made by a browser or renderer after output leaves JustHTML,
  especially when a custom policy allows remote URLs.

### 8. Command Line Interface

Summary: The command line tool uses the same safe defaults as the Python API.

Scope: `justhtml` command-line input/output, stdin handling, output files,
`--unsafe`, `--allow-tags`, `--cleanup`, selectors, and output formats.

Promise:
- The CLI uses the same default sanitization behavior as the Python API.
- `--unsafe` is an explicit opt-out for trusted input only.
- `--cleanup` runs cleanup after sanitization so it can react to stripped
  attributes.

### 9. Documentation Playground

Summary: The playground is a static browser page. It runs code in the user's
browser and displays examples safely.

Scope: the static browser playground, Pyodide loading, local source loading,
browser-side execution, and browser-side output highlighting.

Promise:
- The playground must escape user-controlled text before using `innerHTML` for
  highlighted output.
- Error display uses text nodes rather than HTML injection.

## Reporting a Vulnerability

If you believe you have found a security vulnerability in `justhtml`, please
report it privately.

### Reporting

Please use GitHub's **Private Vulnerability Reporting** feature.

1. Go to the [Security tab](https://github.com/EmilStenstrom/justhtml/security) of the repository.
2. Click on "Report a vulnerability" to open a private advisory.

Please do **not** report security vulnerabilities in public GitHub issues.

### Response Time

We aim to respond to security reports within **48 hours**.

### Disclosure Policy

Please give us **90 days** to investigate, fix, and release a patch before
public disclosure.

### Recognition

We appreciate security research and will credit valid reports in the release
notes when appropriate. See [CHANGELOG.md](CHANGELOG.md).
