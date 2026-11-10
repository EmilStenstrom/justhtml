[← Back to docs](index.md)

# Streaming API

For large files or when you don't need the full DOM tree, use the streaming API.

## Overview

`stream()` yields events without building a DOM, which reduces tree-allocation work and memory use. Iterate over the events to process tags and text.

It does not provide the full HTML5 tree-construction recovery used by `JustHTML`. Use `JustHTML` when you need the recovered document tree.

## Basic usage

```python
from justhtml import stream

html = "<html><body><p>Hello, world!</p></body></html>"

for event, data in stream(html):
    print(event, data)
```

<!-- justhtml: output -->
```text
start ('html', {})
start ('body', {})
start ('p', {})
text Hello, world!
end p
end body
end html
```

## Byte input and encodings

`stream()` also accepts bytes (`bytes`, `bytearray`, `memoryview`). In that case, the input is decoded using HTML encoding sniffing (including a `windows-1252` fallback for legacy documents).

```python
from justhtml import stream
from pathlib import Path

data = Path("page.html").read_bytes()
for event, data in stream(data):
    ...
```

To override decoding when you already know the correct encoding:

```python
from justhtml import stream
from pathlib import Path

data = Path("page.html").read_bytes()
for event, data in stream(data, encoding="utf-8"):
    ...
```

See [Encoding & Byte Input](encoding.md) for details.

## Events

| Event | Data Type | Description |
|-------|-----------|-------------|
| `"start"` | `(tag_name, attrs_dict)` | Opening tag encountered |
| `"end"` | `tag_name` | Closing tag encountered |
| `"text"` | `str` | Text content |
| `"comment"` | `str` | HTML comment content |
| `"doctype"` | `(name, public_id, system_id)` | DOCTYPE declaration fields |

## Examples

### Extract all links

```python
from justhtml import stream
from pathlib import Path

html = Path("page.html").read_text()

for event, data in stream(html):
    if event == "start":
        tag, attrs = data
        if tag == "a" and "href" in attrs:
            print(attrs["href"])
```

### Count elements

```python
from justhtml import stream
from collections import Counter

counts = Counter()

for event, data in stream(html):
    if event == "start":
        tag, attrs = data
        counts[tag] += 1

print(counts.most_common(10))
```

### Extract text content

```python
from justhtml import stream

text_parts = []
for event, data in stream(html):
    if event == "text":
        text_parts.append(data)

full_text = " ".join(text_parts)
```

### Filter by tag

```python
from justhtml import stream

in_script = False
for event, data in stream(html):
    if event == "start" and data[0] == "script":
        in_script = True
    elif event == "end" and data == "script":
        in_script = False
    elif event == "text" and not in_script:
        print(data)  # Only non-script text
```

## When to use streaming

Use the streaming API when:
- Processing very large HTML files
- You only need specific elements (like all links)
- Memory is constrained
- You don't need to traverse the tree

Use the DOM API (`JustHTML`) when:
- You need to query with CSS selectors
- You need to traverse up/down the tree
- You need to serialize back to HTML
- The document fits in memory

## Performance

The documented timing example compares event scanning with DOM construction:

| API | Time (100 files) | Memory |
|-----|------------------|--------|
| `JustHTML()` | ~1.0s | Higher |
| `stream()` | ~0.7s | Lower |

These timings do not include a corpus or environment description. Measure your own input before relying on the speed difference, and choose the API with the structural behavior you need.
