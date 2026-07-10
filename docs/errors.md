[← Back to docs](index.md)

# Error Codes

JustHTML reports a deliberately small set of high-value parser diagnostics. It
does not attempt to reproduce every error emitted by the WHATWG reference
parser. Tree recovery and output remain standards-compatible even when no
diagnostic is reported.

Error collection is disabled by default because the additional diagnostic scan
has a performance cost.

## Collecting Errors

```python
from justhtml import JustHTML

doc = JustHTML("<!doctype html><!--", collect_errors=True)
for error in doc.errors:
    print(f"{error.line}:{error.column} - {error.category}:{error.code}")
```

`doc.errors` is ordered by source position (line, column), with unknown
positions appearing last.

At most 1,000 diagnostics are retained by default, including parser,
sanitizer, and transform findings produced during construction. Set a different
positive `max_errors` value when constructing `JustHTML` if your application
needs a different limit.

## Error Categories

Each error has a `category` field:

- `tokenizer`: lexical/scanning errors
- `treebuilder`: basic document-structure errors
- `security`: sanitizer findings when `unsafe_handling="collect"`
- `transform`: errors explicitly emitted by a custom transform

Custom transforms may emit application-specific codes in addition to the
built-in codes listed below.

## Strict Mode

```python
from justhtml import JustHTML, StrictModeError

try:
    doc = JustHTML("<!doctype html><!--", strict=True)
except StrictModeError as error:
    print(error)
```

Strict mode enables error collection and raises on the earliest supported
parser diagnostic. It is not a complete HTML conformance validator: malformed
input that the parser can recover from may not have a corresponding diagnostic.

## Stability

The codes listed on this page are the supported built-in diagnostic set. Exact
locations are best-effort, and new codes may be added in minor releases. Code
that needs stable behavior should match on `error.code`, not the complete
message or the exact list produced for a complex malformed document.

## Error Locations

- Coordinates are 1-based.
- Character errors point at the offending character where possible.
- Tag-related errors point at or near the triggering markup and may include an
  end position for highlighting.
- EOF errors point at the end of the input.

## Parser Errors

### Tokenizer

| Code | Description |
|------|-------------|
| `eof-in-comment` | Input ended inside an HTML comment |
| `eof-in-tag` | Input ended inside a start or end tag |
| `unexpected-null-character` | Input contains a NULL character (U+0000) |

### Tree builder

| Code | Description |
|------|-------------|
| `expected-doctype-but-got-chars` | A document started with text instead of a DOCTYPE |
| `expected-doctype-but-got-start-tag` | A document started with an element instead of a DOCTYPE |
| `unexpected-end-tag` | An end tag had no matching open element |
| `unknown-doctype` | The DOCTYPE is not the standard HTML DOCTYPE |

These diagnostics intentionally omit detailed recovery events such as
foster-parenting and adoption-agency repairs. Those events are implementation
details of successful tree construction and reproducing a legacy parser's
diagnostic stream would add substantial duplicate work.

## Security Errors

Security errors are reported by the sanitizer when its unsafe handling mode is
`collect`.

| Code | Description |
|------|-------------|
| `unsafe-html` | Content violated the sanitization policy; details are in `error.message` |
| `unsafe-rawtext-child` | A non-text child inside `<script>` or `<style>` was removed |
| `unsafe-rawtext-end-tag` | A dangerous closing-tag sequence in raw text was neutralized |
| `unsafe-style-resource` | Resource-loading CSS in a `<style>` element was removed |

## Node Locations

Node locations are separate from parse errors and disabled by default. Enable
them with `track_node_locations=True`:

```python
from justhtml import JustHTML

doc = JustHTML("<p>hi</p>", track_node_locations=True)
p = doc.query("p")[0]

print(p.origin_location)  # (1, 1)
print(p.origin_line)      # 1
print(p.origin_col)       # 1
print(p.origin_offset)    # 0
```

Locations are best-effort. Nodes created or moved during tree recovery retain
the location of the token that created them, or the closest available source
position.
