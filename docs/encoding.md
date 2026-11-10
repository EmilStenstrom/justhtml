[← Back to docs](index.md)

# Encoding & byte input

JustHTML can parse both Unicode strings (`str`) and raw byte streams (`bytes`, `bytearray`, `memoryview`).

If you pass **bytes**, JustHTML will **sniff and decode** the input using the HTML Standard’s encoding rules.

## When encoding sniffing happens

- If `html` is a `str`: no sniffing/decoding happens (it’s already decoded).
- If `html` is bytes-like: JustHTML decodes it into a `str` before tokenization.

The chosen encoding is exposed as `doc.encoding` when you use `JustHTML(...)`.

## Why the default is `windows-1252`

If no encoding information is found, JustHTML defaults to Windows-1252 (also called "cp1252") for compatibility with legacy HTML:

- Many older documents were authored as “Latin-1” without an explicit encoding.
- Browsers historically treated this as Windows-1252, not ISO-8859-1.
- Using the same default makes JustHTML behave like browsers on real-world old documents.

## What JustHTML looks at (high level)

For byte input, JustHTML follows the standard precedence:

1. **Transport encoding override** (what you pass as `encoding=`)
2. **BOM** (byte order mark)
3. **`<meta charset=...>` / `<meta http-equiv=... content=...>` in the initial bytes
4. Fallback to **`windows-1252`**

JustHTML also treats `utf-7` labels as unsafe and falls back to `windows-1252`.

## How to control it

### 1) Let JustHTML sniff (recommended for unknown/legacy HTML)

```python
from justhtml import JustHTML
from pathlib import Path

data = Path("page.html").read_bytes()

doc = JustHTML(data)
print(doc.encoding)
```

### 2) Override with a known encoding

If you already know the correct encoding (e.g. from HTTP headers, file metadata, or your application protocol), pass it as `encoding=`.

```python
from justhtml import JustHTML
from pathlib import Path

data = Path("page.html").read_bytes()

doc = JustHTML(data, encoding="utf-8")
```

### 3) Decode yourself (when you want full control)

```python
from justhtml import JustHTML
from pathlib import Path

data = Path("page.html").read_bytes()
html = data.decode("utf-8", errors="replace")

doc = JustHTML(html)
```

## Streaming API

The streaming API supports the same byte-input behavior:

```python
from justhtml import stream
from pathlib import Path

for event, data in stream(Path("page.html").read_bytes()):
    ...
```

To override the encoding:

```python
from justhtml import stream
from pathlib import Path

for event, data in stream(Path("page.html").read_bytes(), encoding="utf-8"):
    ...
```
