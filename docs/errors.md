[← Back to docs](index.md)

# Error Codes

Parse errors that JustHTML can detect and report.

## Collecting Errors

By default, JustHTML silently recovers from errors (like browsers do). To collect errors:

```python
from justhtml import JustHTML

doc = JustHTML("<p>Hello", collect_errors=True)
for error in doc.errors:
    print(f"{error.line}:{error.column} - {error.category}:{error.code}")
```

`doc.errors` is ordered by source position (line, column), with unknown positions (if any) appearing last.

## Error Categories

Each error has a `category` field:

- `tokenizer`: lexical/scanning errors
- `treebuilder`: tree construction structure errors
- `security`: sanitizer findings (only when you opt in via `unsafe_handling="collect"`)

## Strict Mode

To reject malformed HTML entirely:

```python
from justhtml import JustHTML, StrictModeError

try:
    doc = JustHTML("<p>Hello", strict=True)
except StrictModeError as e:
    print(e)  # Shows source location
```

In strict mode, JustHTML raises on the earliest error by source position.

## Error Locations (Line/Column)

JustHTML reports a source location for each parse error as a best-effort pointer to where the parser detected the problem in the input stream.

- Coordinates are 1-based: the first character in the input is `(line=1, column=1)`.
- Scanner-detected character errors (for example `unexpected-null-character`) should point at the exact offending character within the input, even if that character is emitted as part of a larger run of text.
- Tree construction errors are associated with the markup that triggered the structure repair.
    - In practice this usually means the error points at (or near) the triggering tag location.
    - When available, JustHTML will highlight the full triggering tag range.
- EOF-related errors point to the end-of-input position where the parser realized it could not continue.

This means error locations are not universally “at the beginning” or “at the end” of a tag: character-level errors point at the character, while structure errors generally point at the triggering tag’s start.

## Node Locations (Optional)

Sometimes you want a source location for a *node*, not just for parse errors.

For performance reasons, node locations are **disabled by default**. To enable them, pass `track_node_locations=True` when parsing:

```python
from justhtml import JustHTML

doc = JustHTML("<p>hi</p>", track_node_locations=True)
p = doc.query("p")[0]

print(p.origin_location)  # (1, 1)
print(p.origin_line)      # 1
print(p.origin_col)       # 1
print(p.origin_offset)    # 0 (0-indexed)
```

Each node exposes best-effort origin metadata:

- `origin_location -> (line, col) | None` (both 1-indexed)
- `origin_line -> int | None` (1-indexed)
- `origin_col -> int | None` (1-indexed)
- `origin_offset -> int | None` (0-indexed offset into the input)

Notes:

- If `track_node_locations=False` (default), these are typically `None`.
- Locations are best-effort. When the tree builder creates or moves nodes as part of error recovery, the reported origin is the location of the token that created the node (or the closest available source position).
- Enabling node tracking adds overhead. If you only need error locations, prefer `collect_errors=True` / `strict=True`.

### Example: Reporting missing includes

```python
import sys
from pathlib import Path

from justhtml import JustHTML


with open(sys.argv[1]) as f:
    html = f.read()

doc = JustHTML(html, track_node_locations=True)
for include_node in doc.query("x-include"):
    src = include_node.attrs.get("src", "")
    if not Path(src).exists():
        line, col = include_node.origin_location or (0, 0)
        print(f"Missing include source: {src} ({sys.argv[1]}:{line}.{col})")
```

---

## Scanner Errors

Errors detected during tokenization (lexical analysis).

### DOCTYPE Errors

| Code | Description |
|------|-------------|
| `eof-in-doctype` | Unexpected end of file in DOCTYPE declaration |
| `eof-in-doctype-name` | Unexpected end of file while reading DOCTYPE name |
| `eof-in-doctype-public-identifier` | Unexpected end of file in DOCTYPE public identifier |
| `eof-in-doctype-system-identifier` | Unexpected end of file in DOCTYPE system identifier |
| `expected-doctype-name-but-got-right-bracket` | Expected DOCTYPE name but got `>` |
| `missing-whitespace-before-doctype-name` | Missing whitespace after `<!DOCTYPE` |
| `abrupt-doctype-public-identifier` | DOCTYPE public identifier ended abruptly |
| `abrupt-doctype-system-identifier` | DOCTYPE system identifier ended abruptly |
| `missing-quote-before-doctype-public-identifier` | Missing quote before DOCTYPE public identifier |
| `missing-quote-before-doctype-system-identifier` | Missing quote before DOCTYPE system identifier |
| `missing-doctype-public-identifier` | Missing DOCTYPE public identifier |
| `missing-doctype-system-identifier` | Missing DOCTYPE system identifier |
| `missing-whitespace-before-doctype-public-identifier` | Missing whitespace before DOCTYPE public identifier |
| `missing-whitespace-after-doctype-public-identifier` | Missing whitespace after DOCTYPE public identifier |
| `missing-whitespace-between-doctype-public-and-system-identifiers` | Missing whitespace between DOCTYPE identifiers |
| `missing-whitespace-after-doctype-name` | Missing whitespace after DOCTYPE name |
| `unexpected-character-after-doctype-public-keyword` | Unexpected character after PUBLIC keyword |
| `unexpected-character-after-doctype-system-keyword` | Unexpected character after SYSTEM keyword |
| `unexpected-character-after-doctype-public-identifier` | Unexpected character after public identifier |
| `unexpected-character-after-doctype-system-identifier` | Unexpected character after system identifier |

### Comment Errors

| Code | Description |
|------|-------------|
| `eof-in-comment` | Unexpected end of file in comment |
| `abrupt-closing-of-empty-comment` | Comment ended abruptly with `-->` |
| `incorrectly-closed-comment` | Comment ended with `--!>` instead of `-->` |
| `incorrectly-opened-comment` | Incorrectly opened comment |

### Tag Errors

| Code | Description |
|------|-------------|
| `eof-in-tag` | Unexpected end of file in tag |
| `eof-before-tag-name` | Unexpected end of file before tag name |
| `empty-end-tag` | Empty end tag `</>` is not allowed |
| `invalid-first-character-of-tag-name` | Invalid first character of tag name |
| `unexpected-question-mark-instead-of-tag-name` | Unexpected `?` instead of tag name |
| `unexpected-character-after-solidus-in-tag` | Unexpected character after `/` in tag |

### Attribute Errors

| Code | Description |
|------|-------------|
| `duplicate-attribute` | Duplicate attribute name |
| `missing-attribute-value` | Missing attribute value |
| `unexpected-character-in-attribute-name` | Unexpected character in attribute name |
| `unexpected-character-in-unquoted-attribute-value` | Unexpected character in unquoted attribute value |
| `missing-whitespace-between-attributes` | Missing whitespace between attributes |
| `unexpected-equals-sign-before-attribute-name` | Unexpected `=` before attribute name |

### Script Errors

| Code | Description |
|------|-------------|
| `eof-in-script-html-comment-like-text` | Unexpected end of file in script with HTML-like comment |
| `eof-in-script-in-script` | Unexpected end of file in nested script tag |

### CDATA Errors

| Code | Description |
|------|-------------|
| `eof-in-cdata` | Unexpected end of file in CDATA section |
| `cdata-in-html-content` | CDATA section only allowed in SVG/MathML content |

### Character Reference Errors

| Code | Description |
|------|-------------|
| `control-character-reference` | Invalid control character in character reference |
| `illegal-codepoint-for-numeric-entity` | Invalid codepoint in numeric character reference |
| `missing-semicolon-after-character-reference` | Missing semicolon after character reference |
| `named-entity-without-semicolon` | Named entity used without semicolon |
| `noncharacter-character-reference` | Noncharacter in character reference |

### Other Scanner Errors

| Code | Description |
|------|-------------|
| `unexpected-null-character` | Unexpected NULL character (U+0000) |
| `noncharacter-in-input-stream` | Noncharacter in input stream |

---

## Tree Builder Errors

Errors detected during tree construction.

### DOCTYPE Errors

| Code | Description |
|------|-------------|
| `unexpected-doctype` | Unexpected DOCTYPE declaration |
| `unknown-doctype` | Unknown DOCTYPE (expected `<!DOCTYPE html>`) |
| `expected-doctype-but-got-chars` | Expected DOCTYPE but got text content |
| `expected-doctype-but-got-eof` | Expected DOCTYPE but reached end of file |
| `expected-doctype-but-got-start-tag` | Expected DOCTYPE but got start tag |
| `expected-doctype-but-got-end-tag` | Expected DOCTYPE but got end tag |

### Unexpected Tag Errors

| Code | Description |
|------|-------------|
| `unexpected-start-tag` | Unexpected start tag in current context |
| `unexpected-end-tag` | Unexpected end tag in current context |
| `unexpected-end-tag-before-html` | Unexpected end tag before `<html>` |
| `unexpected-end-tag-before-head` | Unexpected end tag before `<head>` |
| `unexpected-end-tag-after-head` | Unexpected end tag after `<head>` |
| `unexpected-start-tag-ignored` | Start tag ignored in current context |
| `unexpected-start-tag-implies-end-tag` | Start tag implicitly closes previous element |

### EOF Errors

| Code | Description |
|------|-------------|
| `expected-closing-tag-but-got-eof` | Expected closing tag but reached end of file |
| `expected-named-closing-tag-but-got-eof` | Expected specific closing tag but reached end of file |

### Invalid Character Errors

| Code | Description |
|------|-------------|
| `invalid-codepoint` | Invalid character (U+0000 NULL or U+000C FORM FEED) |
| `invalid-codepoint-before-head` | Invalid character before `<head>` |
| `invalid-codepoint-in-body` | Invalid character in `<body>` |
| `invalid-codepoint-in-table-text` | Invalid character in table text |
| `invalid-codepoint-in-select` | Invalid character in `<select>` |
| `invalid-codepoint-in-foreign-content` | Invalid character in SVG/MathML content |

### Table Errors

| Code | Description |
|------|-------------|
| `foster-parenting-character` | Text content in table requires foster parenting |
| `foster-parenting-start-tag` | Start tag in table requires foster parenting |
| `unexpected-character-implies-table-voodoo` | Unexpected character in table triggers foster parenting |
| `unexpected-start-tag-implies-table-voodoo` | Start tag in table triggers foster parenting |
| `unexpected-end-tag-implies-table-voodoo` | End tag in table triggers foster parenting |
| `unexpected-implied-end-tag-in-table-view` | Unexpected implied end tag while closing table |
| `eof-in-table` | Unexpected end of file in table |
| `unexpected-cell-in-table-body` | Unexpected table cell outside of table row |
| `unexpected-form-in-table` | Form element not allowed in table context |
| `unexpected-hidden-input-in-table` | Hidden input in table triggers foster parenting |

### Frameset Errors

| Code | Description |
|------|-------------|
| `unexpected-token-in-frameset` | Unexpected content in `<frameset>` |
| `unexpected-token-after-frameset` | Unexpected content after `<frameset>` |
| `unexpected-token-after-after-frameset` | Unexpected content after frameset closed |

### After-Body Errors

| Code | Description |
|------|-------------|
| `unexpected-token-after-body` | Unexpected content after `</body>` |
| `unexpected-char-after-body` | Unexpected character after `</body>` |

### Column Group / Template Table Context Errors

| Code | Description |
|------|-------------|
| `unexpected-characters-in-column-group` | Text not allowed in `<colgroup>` |
| `unexpected-characters-in-template-column-group` | Text not allowed in template column group |
| `unexpected-start-tag-in-column-group` | Start tag not allowed in `<colgroup>` |
| `unexpected-start-tag-in-template-column-group` | Start tag not allowed in template column group |
| `unexpected-start-tag-in-template-table-context` | Start tag not allowed in template table context |

### Fragment Context Errors

| Code | Description |
|------|-------------|
| `unexpected-start-tag-in-cell-fragment` | Start tag not allowed in cell fragment context |
| `unexpected-end-tag-in-fragment-context` | End tag not allowed in fragment parsing context |

### Head/Body Context Errors

| Code | Description |
|------|-------------|
| `unexpected-hidden-input-after-head` | Unexpected hidden input after `<head>` |

### Foreign Content Errors

| Code | Description |
|------|-------------|
| `unexpected-doctype-in-foreign-content` | Unexpected DOCTYPE in SVG/MathML content |
| `unexpected-html-element-in-foreign-content` | HTML element breaks out of SVG/MathML content |
| `unexpected-end-tag-in-foreign-content` | Mismatched end tag in SVG/MathML content |

### Select Errors

| Code | Description |
|------|-------------|
| `unexpected-start-tag-in-select` | Unexpected start tag in `<select>` |
| `unexpected-end-tag-in-select` | Unexpected end tag in `<select>` |
| `unexpected-select-in-select` | Unexpected nested `<select>` in `<select>` |

### Miscellaneous Errors

| Code | Description |
|------|-------------|
| `end-tag-too-early` | End tag closed early (unclosed children) |
| `adoption-agency-1.3` | Misnested tags require adoption agency algorithm |
| `non-void-html-element-start-tag-with-trailing-solidus` | Self-closing syntax on non-void element (e.g., `<div/>`) |
| `image-start-tag` | Deprecated `<image>` tag (use `<img>` instead) |

---

## Security Errors

Errors reported by the sanitizer when you opt in via `unsafe_handling="collect"`.

| Code | Description |
|------|-------------|
| `unsafe-html` | Unsafe HTML detected by sanitization policy (see `error.message` for details) |
