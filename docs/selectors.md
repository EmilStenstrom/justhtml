[← Back to docs](index.md)

# CSS Selectors

JustHTML supports a comprehensive subset of CSS selectors for querying the DOM.

## Basic Selectors

| Selector | Example | Description |
|----------|---------|-------------|
| Tag | `div` | Elements by tag name |
| Class | `.intro` | Elements with class |
| ID | `#main` | Element with ID |
| Universal | `*` | All elements |

```python
doc.query("div")       # All <div> elements
doc.query(".intro")    # All elements with class="intro"
doc.query("#main")     # Element with id="main"
doc.query("*")         # All elements
```

## Attribute Selectors

| Selector | Example | Description |
|----------|---------|-------------|
| Has attribute | `[href]` | Elements with attribute |
| Exact match | `[type="text"]` | Exact attribute value |
| Starts with | `[href^="https"]` | Attribute starts with value |
| Ends with | `[href$=".pdf"]` | Attribute ends with value |
| Contains | `[href*="example"]` | Attribute contains value |
| Word match | `[class~="active"]` | Space-separated word match |
| Prefix match | `[lang\|="en"]` | Value or value followed by `-` |

```python
doc.query("[href]")              # All elements with href attribute
doc.query('[type="submit"]')     # Exact match
doc.query('[href^="https://"]')  # Links starting with https://
doc.query('[src$=".png"]')       # Images ending with .png
doc.query('[class*="btn"]')      # Classes containing "btn"
```

## Combinators

| Combinator | Example | Description |
|------------|---------|-------------|
| Descendant | `div p` | `<p>` anywhere inside `<div>` |
| Child | `div > p` | `<p>` direct child of `<div>` |
| Adjacent | `h1 + p` | `<p>` immediately after `<h1>` |
| Sibling | `h1 ~ p` | Any `<p>` sibling after `<h1>` |

```python
doc.query("div p")          # Paragraphs anywhere in a div
doc.query("ul > li")        # Direct list items
doc.query("h1 + p")         # First paragraph after h1
doc.query("h1 ~ p")         # All paragraphs after h1
```

## Pseudo-classes

### Structural

| Selector | Description |
|----------|-------------|
| `:first-child` | First child of parent |
| `:last-child` | Last child of parent |
| `:only-child` | Only child of parent |
| `:nth-child(n)` | Nth child (1-indexed) |
| `:nth-last-child(n)` | Nth child from end |
| `:first-of-type` | First of its type in parent |
| `:last-of-type` | Last of its type in parent |
| `:only-of-type` | Only of its type in parent |
| `:nth-of-type(n)` | Nth of its type |
| `:nth-last-of-type(n)` | Nth of its type from end |

```python
doc.query("li:first-child")       # First list item
doc.query("li:last-child")        # Last list item
doc.query("li:nth-child(2)")      # Second list item
doc.query("li:nth-child(odd)")    # Odd list items (1st, 3rd, 5th...)
doc.query("li:nth-child(even)")   # Even list items (2nd, 4th, 6th...)
doc.query("li:nth-child(3n)")     # Every 3rd item
doc.query("li:nth-child(3n+1)")   # 1st, 4th, 7th... (formula: An+B)
```

### Content

| Selector | Description |
|----------|-------------|
| `:empty` | Elements with no children |
| `:root` | Document root element |
| `:contains("text")` | Non-standard (jQuery-style): elements whose descendant text contains `text` |

```python
doc.query("p:empty")    # Empty paragraphs
doc.query(":root")      # The <html> element
```

### Negation

| Selector | Description |
|----------|-------------|
| `:not(selector)` | Elements not matching selector |

```python
doc.query("div:not(.hidden)")     # Visible divs
doc.query("input:not([disabled])")  # Enabled inputs
```

## Selector Groups

Combine multiple selectors with commas:

```python
doc.query("h1, h2, h3")           # All headings
doc.query(".btn, [type='submit']") # Buttons and submit inputs
```

## Compound Selectors

Chain selectors without spaces:

```python
doc.query("div.container")        # <div class="container">
doc.query("input[type='text']")   # <input type="text">
doc.query("li.active:first-child") # First li with class active
```

## Examples

```python
# Navigation links
doc.query("nav a")

# Active menu item
doc.query("nav li.active > a")

# Form inputs except hidden
doc.query("form input:not([type='hidden'])")

# Alternating table rows
doc.query("tr:nth-child(even)")

# First paragraph in each container
doc.query("div > p:first-of-type")

# External links
doc.query('a[href^="http"]:not([href*="mysite.com"])')
```
