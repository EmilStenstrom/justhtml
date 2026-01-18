[← Back to docs](index.md)

# Command Line Interface

JustHTML ships with a small CLI for parsing HTML and extracting HTML/text/Markdown from selected parts of a document.

## Running

If you installed JustHTML (for example with `pip install justhtml` or `pip install -e .`), you can use the `justhtml` command.

If you don't have it available, use the equivalent `python -m justhtml ...` form.

## Basic usage

```bash
# Pretty-print an HTML file
justhtml page.html

# Read HTML from stdin
curl -s https://example.com | justhtml -
```

## Selecting nodes

Use `--selector` to choose which nodes to extract.

```bash
# Extract text from all paragraphs
justhtml page.html --selector "p" --format text

# Only output the first match
justhtml page.html --selector "main p" --format text --first
```

## Fragments

Use `--fragment` to parse the input as an HTML fragment (instead of a full document). This avoids implicit `<html>`, `<head>`, and `<body>` insertion.

```bash
echo '<li>Hi</li>' | justhtml - --fragment
```

## Output formats

`--format` controls what is printed:

- `html` (default): pretty-printed HTML for each match
- `text`: concatenated text (same semantics as `to_text(separator=" ", strip=True)`; sanitized by default)
- `markdown`: a pragmatic subset of GitHub Flavored Markdown (GFM)

Notes:

- `markdown` keeps tables (`<table>`) and images (`<img>`) as raw HTML.
- For multiple matches:
  - `html` and `text` print one result per line.
  - `markdown` prints matches separated by a blank line.

## Sanitization

By default, the CLI sanitizes output (same safe-by-default behavior as `JustHTML(..., safe=True)`).

To disable sanitization for trusted input, pass `--unsafe`.

## Text options

When using `--format text`, you can control whitespace handling:

- `--separator "..."` (default: a single space) joins text nodes
- `--strip` / `--no-strip` controls whether each text node is stripped and empty segments dropped

Example:

```bash
justhtml page.html --selector "main" --format text --separator "" --no-strip
```

## Exit codes

- `0`: success
- `1`: missing input path or no matches for the selector
- `2`: invalid selector

## Real-world example

```bash
curl -s https://github.com/EmilStenstrom/justhtml/ | justhtml - --selector '.markdown-body' --format markdown | head -n 15
```

Output:

```text
# JustHTML

[](#justhtml)

A pure Python HTML5 parser that just works. No C extensions to compile. No system dependencies to install. No complex API to learn.

**[📖 Read the full documentation here](/EmilStenstrom/justhtml/blob/main/docs/index.md)**

## Why use JustHTML?

[](#why-use-justhtml)

### 1. Just... Correct ✅

[](#1-just-correct-)
```
