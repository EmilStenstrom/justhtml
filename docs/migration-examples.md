[← Back to docs](index.md)

# Learn by examples

This page is a grab-bag of real-world “how do I…” questions from StackOverflow, rewritten to show the equivalent JustHTML approach.

The “Original HTML” blocks are excerpts from the StackOverflow question bodies (see the linked threads for full context and attribution).

Principles these examples emphasize:

- **One library, end-to-end**: parse HTML5 → query with CSS selectors → transform → serialize.
- **Safe by default**: sanitization is on unless you disable it.
- **Composable building blocks**: selectors *locate* nodes; policies and transforms do the work; `to_html()` / `to_text()` / `to_markdown()` handle output.

Note: Each example links to the original question thread for context. The code below is a fresh JustHTML rewrite (not a copy/paste of StackOverflow answers).

---

## BeautifulSoup / bs4

### Find elements by CSS class

Original thread: https://stackoverflow.com/questions/22726860/beautifulsoup-webscraping-find-all-finding-exact-match

Original HTML (excerpt):

```html
<body>
    <div class="product">Product 1</div>
    <div class="product">Product 2</div>
    <div class="product special">Product 3</div>
    <div class="product special">Product 4</div>
</body>
```

BeautifulSoup:

```python
from bs4 import BeautifulSoup

soup = BeautifulSoup(html, "html.parser")
for el in soup.select(".product.special"):
    print(el.get_text(strip=True))
```

JustHTML:

```python
from justhtml import JustHTML

doc = JustHTML(html, fragment=True, sanitize=False)

names = [el.to_text() for el in doc.root.query(".product.special")]
print(names)
# => ['Product 3', 'Product 4']
```

### Extract an attribute value

Original thread: https://stackoverflow.com/questions/5815747/beautifulsoup-getting-href

Original HTML (excerpt):

```html
<a href="some_url">next</a>
<span class="class">...</span>
```

BeautifulSoup:

```python
from bs4 import BeautifulSoup

soup = BeautifulSoup(html, "html.parser")
links = [a.get("href") for a in soup.select("a[href]")]
print(links)
```

JustHTML:

```python
from justhtml import JustHTML

doc = JustHTML(html, fragment=True, sanitize=False)

links = [a.attrs.get("href") for a in doc.root.query("a[href]")]
print(links)
# => ['some_url']
```

### Extract a subtree by id (and serialize it)

Original thread: https://stackoverflow.com/questions/16780158/search-within-tags-with-beautifulsoup-python

Original HTML (excerpt):

```html
<div id="cmeProductSlatePaginiationTop" class="cmePaginiation">
    <ul>
        <li class="disabled">
        <li class="active">
        <li class="away-1">
        <li>
    </ul>
</div>
```

BeautifulSoup:

```python
from bs4 import BeautifulSoup

soup = BeautifulSoup(html, "html.parser")
el = soup.find(id="cmeProductSlatePaginiationTop")
print(el.prettify() if el else "")
```

JustHTML:

```python
from justhtml import JustHTML

doc = JustHTML(html, fragment=True, sanitize=False)

el = next(iter(doc.root.query("#cmeProductSlatePaginiationTop")), None)
print(el.to_html(pretty=True) if el else "")
# => <div id="cmeProductSlatePaginiationTop" class="cmePaginiation">
# => ...<ul>
# => ...<li class="disabled"></li>
# => ...<li class="active"></li>
# => ...<li class="away-1"></li>
# => ...<li></li>
# => ...</ul>
# => </div>
```

### Remove `<script>` tags (content included in input HTML)

Original thread: https://stackoverflow.com/questions/25215922/remove-script-tags-inside-p-tags-using-beautifulsoup

Original HTML (excerpt):

```html
<p>
<script>
...
</script>
</p>
```

BeautifulSoup:

```python
from bs4 import BeautifulSoup

soup = BeautifulSoup(html, "html.parser")
for script in soup.select("script"):
    script.decompose()
print(str(soup))
```

JustHTML:

```python
from justhtml import Drop, JustHTML

doc = JustHTML(
    html,
    fragment=True,
    sanitize=False,
    transforms=[Drop("script")],
)

# Note: You could also set sanitize=True to apply an allowlist of tags.

print(doc.to_html())
# => <p></p>
```

### Convert HTML to readable text

Original thread: https://stackoverflow.com/questions/14694482/converting-html-to-text-with-python

Original HTML (excerpt):

```html
<div class="body"><p><strong></strong></p>
<p><strong></strong>Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Aenean commodo ligula eget dolor. Aenean massa</p>
<p>Consectetuer adipiscing elit. <a href="http://example.com/" target="_blank" class="source">Some Link</a> Aenean commodo ligula eget dolor. Aenean massa</p>
<p>Aenean massa.Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Aenean commodo ligula eget dolor. Aenean massa</p>
<p>Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Aenean commodo ligula eget dolor. Aenean massa</p>
<p>Consectetuer adipiscing elit. Aenean commodo ligula eget dolor. Aenean massa</p></div>
```

BeautifulSoup:

```python
from bs4 import BeautifulSoup

soup = BeautifulSoup(html, "html.parser")
print(soup.get_text("\n", strip=True))
```

JustHTML:

```python
from justhtml import Drop, JustHTML

doc = JustHTML(
    html,
    fragment=True,
    transforms=[
        Drop(
            "script, style, head, meta, link, noscript, iframe, object, embed, svg"
        ),
        Drop("[hidden], [aria-hidden='true']"),
    ],
)

text = doc.to_markdown().strip()
print(text)
# => Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Aenean commodo ligula eget dolor. Aenean massa
# =>
# => Consectetuer adipiscing elit. [Some Link](http://example.com/) Aenean commodo ligula eget dolor. Aenean massa
# =>
# => Aenean massa.Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Aenean commodo ligula eget dolor. Aenean massa
# =>
# => Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Aenean commodo ligula eget dolor. Aenean massa
# =>
# => Consectetuer adipiscing elit. Aenean commodo ligula eget dolor. Aenean massa
```

---

## lxml

### Extract a subtree (XPath → CSS) and pretty-print it

Original thread: https://stackoverflow.com/questions/38232584/how-to-grab-raw-all-raw-html-within-a-certain-xpath-from-a-local-file-in-python

Original HTML (excerpt):

```html
<html>

<head>
    <title>Title</title>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css">
</head>

<body>
    <div class="container">
        <div class="row">
```

lxml:

```python
from lxml import html as lxml_html

root = lxml_html.fromstring(html)
container = root.xpath("//div[contains(concat(' ', normalize-space(@class), ' '), ' container ')]")
print(lxml_html.tostring(container[0], encoding="unicode", pretty_print=True) if container else "")
```

JustHTML:

```python
from justhtml import JustHTML

doc = JustHTML(html, fragment=True, sanitize=False)

matches = doc.root.query(".container")
print(matches[0].to_html(pretty=True) if matches else "")
# => <div class="container">
# => ...<div class="row"></div>
# => </div>
```

### Select descendants by id/class

Original thread: https://stackoverflow.com/questions/31695634/xpath-descendant-and-descendant-or-self-work-completely-different

Original HTML (excerpt):

```html
<html>
    <body>
        <div id='indicator'>
            <table>
               <tbody>
                    <tr>
                        <th>1</th>
                        <th>2</th>
                        <th>3</th>
                    </tr>
```

lxml:

```python
from lxml import html as lxml_html

root = lxml_html.fromstring(html)
print(root.xpath("//*[@id='indicator']//th/text()"))
```

JustHTML:

```python
from justhtml import JustHTML

doc = JustHTML(html, fragment=True, sanitize=False)
print([th.to_text() for th in doc.root.query("#indicator th")])
# => ['1', '2', '3']
```

### Remove `data-*` attributes

Original thread: https://stackoverflow.com/questions/55098355/remove-all-data-attributes-with-etree-from-all-elements

Original HTML (excerpt):

```html
<html>
    <ul>
        <li data-i="sdfdsf">something</li>
        <li data-i="dsfd">something</li>
    </ul>
    <p data-para="cvcv">content</p>
```

lxml:

```python
from lxml import html as lxml_html

root = lxml_html.fromstring(html)
for el in root.iter():
        for name in list(el.attrib):
                if name.startswith("data-"):
                        el.attrib.pop(name)

print(lxml_html.tostring(root, encoding="unicode"))
```

JustHTML:

```python
from justhtml import JustHTML
from justhtml.transforms import DropAttrs

doc = JustHTML(
    html,
    fragment=True,
    sanitize=False,
    transforms=[DropAttrs("*", patterns=("data-*",))],
)

print(doc.to_html())
# => <ul>
# => ...<li>something</li>
# => ...<li>something</li>
# => </ul>
# => <p>content</p>
```

### Get all text inside an element

Original thread: https://stackoverflow.com/questions/24262505/lxml-xpath-how-to-get-concatenated-text-from-node

Original HTML (excerpt):

```html
<a class="someclass">
Wie
<em>Messi</em>
einen kleinen Jungen stehen lässt
</a>
```

lxml:

```python
from lxml import html as lxml_html

root = lxml_html.fromstring(html)
el = root.cssselect("a.someclass")
print("".join(el[0].itertext()) if el else "")
```

JustHTML:

```python
from justhtml import JustHTML

doc = JustHTML(html, fragment=True, sanitize=False)

matches = doc.root.query("a.someclass")
print(matches[0].to_text() if matches else "")
# => Wie Messi einen kleinen Jungen stehen lässt
```

---

## html5lib

### Parse HTML and query it (XPath → CSS selectors)

Original thread: https://stackoverflow.com/questions/2558056/how-can-i-parse-html-with-html5lib-and-query-the-parsed-html-with-xpath

Original HTML (excerpt):

```html
<html>
    <table>
        <tr><td>Header</td></tr>
        <tr><td>Want This</td></tr>
    </table>
</html>
```

html5lib:

```python
import html5lib

# Parse with html5lib, build an lxml tree, query with XPath.
doc = html5lib.parse(html, treebuilder="lxml")
print(doc.xpath("//tr[2]/td/text()"))
```

JustHTML:

```python
from justhtml import JustHTML

doc = JustHTML(html, fragment=True, sanitize=False)

# JustHTML queries with CSS selectors, not XPath.
# Example XPath: //tr[2]/td
# Rough CSS equivalent: tr:nth-child(2) td
cells = [td.to_text() for td in doc.root.query("tr:nth-child(2) td")]
print(cells)
# => ['Want This']
```

### Get source locations (line/col/offset)

Original thread: https://stackoverflow.com/questions/28728498/obtaining-position-info-when-parsing-html-in-python

Original HTML (excerpt):

```html
<p>some text
```

lxml (line number only):

```python
from lxml import html as lxml_html

root = lxml_html.fromstring(html)
for el in root.xpath("//p"):
    print(el.sourceline)
```

JustHTML:

```python
from justhtml import JustHTML

doc = JustHTML(html, fragment=True, sanitize=False, track_node_locations=True)

locs = [(el.origin_line, el.origin_col, el.origin_offset) for el in doc.root.query("p")]
print(locs)
# => [(1, 1, 0)]
```

### Convert an HTML fragment to plain text

Original thread: https://stackoverflow.com/questions/8685332/use-html5lib-to-convert-an-html-fragment-to-plain-text

Original HTML (excerpt):

```html
<p>Hello World. Greetings from <strong>Mars.</strong></p>
```

html5lib:

```python
import html5lib

frag = html5lib.parseFragment(html, treebuilder="etree")
text = "".join(frag.itertext())
print(text)
```

JustHTML:

```python
from justhtml import Drop, JustHTML

doc = JustHTML(
    html,
    fragment=True,
)

print(doc.to_text(separator=" ", strip=True))
# => Hello World. Greetings from Mars.
```

### Remove `<style>` blocks completely

Original thread: https://stackoverflow.com/questions/7538600/remove-contents-of-style-style-tags-using-html5lib-or-bleach

Original HTML (excerpt):

```html
<STYLE> st1:*{behavior:url(#ieooui) } </STYLE>
```

html5lib (via lxml tree):

```python
import html5lib
from lxml import etree

doc = html5lib.parseFragment(html, treebuilder="lxml")
for el in doc.xpath(".//style"):
    el.getparent().remove(el)

print(etree.tostring(doc, encoding="unicode"))
```

JustHTML:

```python
from justhtml import Drop, JustHTML

# Note: Sanitize (default on) drops style tags automatically
doc = JustHTML(html, fragment=True, sanitize=True)

# More explicit way to ONLY drop style tags
doc = JustHTML(html, fragment=True, sanitize=False, transforms=[Drop("style")])
out = doc.to_html()
print(out if out else "<empty>")
# => <empty>
```

---

## Bleach

### Sanitize HTML, but escape everything inside `<pre>/<code>`

Original thread: https://stackoverflow.com/questions/39192753/make-bleach-to-allow-code-tags

Original HTML (excerpt):

```html
<pre>
<code>
 for (auto a = 0; i &lt; 10; ++i) {
    echo "<p>Hello</p>"
 }
</code>
</pre>
```

Bleach:

```python
import bleach

clean = bleach.clean(html, tags=["pre", "code"], attributes={}, strip=False)
print(clean)
```

JustHTML:

```python
from justhtml import Escape, JustHTML, SanitizationPolicy, UrlPolicy, UrlRule

policy = SanitizationPolicy(
    allowed_tags=["p", "pre", "code", "b", "i", "em", "strong", "a"],
    allowed_attributes={"*": [], "a": ["href", "title"]},
    # Keep URL handling explicit when allowing links.
    url_policy=UrlPolicy(
        allow_rules={
            ("a", "href"): UrlRule(
                allowed_schemes={"http", "https", "mailto"},
                allow_relative=True,
            ),
        }
    ),
)

doc = JustHTML(
    html,
    fragment=True,
    policy=policy,
    # Turn markup inside <code> into literal text.
    transforms=[Escape("code *")],
)

print(doc.to_html())
# => <pre><code>
# => ...for (auto a = 0; i &lt; 10; ++i) {
# => ...echo "&lt;p&gt;Hello&lt;/p&gt;"
# => ...}
# => </code>
# => </pre>
```

### Sanitize but keep links (strip unknown attrs)

Original thread: https://stackoverflow.com/questions/54354764/python-bleach-inconsistent-cleaning-behaviour

Original HTML (excerpt):

```html
<p   >This <a href="book"> book </a attr="test"> will help you</p  >
```

Bleach:

```python
import bleach
clean = bleach.clean(
    html,
    tags=["p", "a"],
    attributes={"a": ["href"]},
    strip=True,
)
print(clean)
```

JustHTML:

```python
from justhtml import JustHTML, SanitizationPolicy, UrlPolicy, UrlRule

policy = SanitizationPolicy(
    allowed_tags=["p", "a"],
    allowed_attributes={"*": [], "a": ["href"]},
    url_policy=UrlPolicy(
        allow_rules={
            ("a", "href"): UrlRule(
                allowed_schemes={"http", "https", "mailto"},
                allow_relative=True,
            ),
        }
    ),
)

print(JustHTML(html, fragment=True, policy=policy).to_html())
# => <p>This <a href="book">book</a> will help you</p>
```

### Allow safe inline styles (instead of stripping them)

Original thread: https://stackoverflow.com/questions/65870475/bleach-stripping-style-that-should-be-allowed

Original HTML (excerpt):

```html
<div id="cover" style="display: block; height: 682px"><div class="cover-desktop hidden-xs" style="background-image: linear-gradient(rgba(0, 0, 0, 0.45), rgba(0, 0, 0, 0.45)), url('/site_media/covers/cover.jpg')"></div></div>
```

Bleach:

```python
import bleach
from bleach.css_sanitizer import CSSSanitizer

clean = bleach.clean(
    html,
    tags=["div"],
    attributes={"div": ["id", "class", "style"]},
    css_sanitizer=CSSSanitizer(allowed_css_properties=["background-image", "display", "height"]),
    strip=True,
)
print(clean)
```

JustHTML:

```python
from justhtml import CSS_PRESET_TEXT, JustHTML, SanitizationPolicy

policy = SanitizationPolicy(
    allowed_tags=["div"],
    allowed_attributes={"*": [], "div": ["id", "class", "style"]},
    allowed_css_properties=CSS_PRESET_TEXT | {"display", "height"},
)

# Note: JustHTML intentionally drops any inline style declaration that contains
# url(...)/image-set(...) (to prevent network requests), even if the property is
# allowlisted. In the HTML above, that means the `background-image: ... url(...)`
# declaration will be removed.

print(JustHTML(html, fragment=True, policy=policy).to_html())
# => <div id="cover" style="display: block; height: 682px">
# => ...<div class="cover-desktop hidden-xs"></div>
# => </div>
```

### Remove links but keep their text

Original thread: https://stackoverflow.com/questions/63695338/how-to-remove-links-from-html-completely-with-bleach

Original HTML (excerpt):

```html
<a href="">stays</a>
```

Bleach:

```python
import bleach

# Disallow <a>, but keep other tags; strip disallowed tags but keep their text.
allowed = set(bleach.sanitizer.ALLOWED_TAGS) - {"a"}
clean = bleach.clean(html, tags=sorted(allowed), strip=True)
print(clean)
```

JustHTML:

```python
from justhtml import JustHTML, Unwrap

doc = JustHTML(html, fragment=True, transforms=[Unwrap("a")])
print(doc.to_html())
# => stays
```

---

## xml.etree.ElementTree

These threads are often symptoms of the same underlying issue: **ElementTree is great for XML, but HTML isn’t XML**.

If the input is HTML (or “HTML-ish”), JustHTML gives you HTML5 parsing, browser-grade error recovery, CSS selectors, and built-in output formats.

### Parse HTML that contains `&nbsp;`

Original thread: https://stackoverflow.com/questions/35591478/how-to-parse-html-with-entities-such-as-nbsp-using-builtin-library-elementtree

Original HTML (excerpt):

```html
<html>
    <div>Some reasonably well-formed HTML content.</div>
    <form action="login">
    <input name="foo" value="bar"/>
    <input name="username"/><input name="password"/>

    <div>It is not unusual to see &nbsp; in an HTML page.</div>

    </form></html>
```

ElementTree:

```python
import xml.etree.ElementTree as ET

# ElementTree is an XML parser; HTML entities like &nbsp; can break parsing.
ET.fromstring(html)
```

JustHTML:

```python
from justhtml import JustHTML

doc = JustHTML(html, sanitize=False)
print(doc.to_text(separator=" ", strip=True))
# => Some reasonably well-formed HTML content. It is not unusual to see   in an HTML page.
```

### Parse XHTML entities like `&copy;`

Original thread: https://stackoverflow.com/questions/51932260/parsing-xhtml-including-standard-entities-using-elementtree

Original HTML (excerpt):

```html
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
  "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
  <head><title>&copy;</title></head>
  <body></body>
</html>
```

ElementTree:

```python
import xml.etree.ElementTree as ET

root = ET.fromstring(html)
title = root.find("{http://www.w3.org/1999/xhtml}head/{http://www.w3.org/1999/xhtml}title")
print(title.text if title is not None else "")
```

JustHTML:

```python
from justhtml import JustHTML

doc = JustHTML(html, sanitize=False)
print(doc.root.query("title")[0].to_text())
# => ©
```

### Avoid namespace-prefixed tag names (`html:html`)

Original thread: https://stackoverflow.com/questions/6088684/python-elementtree-write-to-file-without-namespaces

Original HTML (excerpt):

```html
<html:html lang="en-US" xml:lang="en-US" xmlns:html="http://www.w3.org/1999/xhtml">
<html:head>
<html:title>vocab</html:title>
<html:style type="text/css"> ...
```

ElementTree:

```python
import xml.etree.ElementTree as ET

root = ET.fromstring(html)
print(root.tag)  # Namespace-prefixed names are part of the tag in XML.
```

JustHTML:

```python
from justhtml import JustHTML

doc = JustHTML(html, sanitize=False)
print(doc.to_html(pretty=True))
# => <html>
# => ...<head></head>
# => ...<body>
# => ...vocab...
# => ...</body>
# => </html>
```

### Parse XML embedded in an HTML comment

Original thread: https://stackoverflow.com/questions/76223870/parsing-xml-within-html-using-python

Original HTML (excerpt):

```html
<!DOCTYPE html>
<html>
<body>
    <div class="panel panel-primary call__report-modal-panel">
        <div class="panel-heading text-center custom-panel-heading">
            <h2>Report</h2>
        </div>
        ...
    </div>
</body>
</html>
<!--<?xml version = "1.0" encoding="Windows-1252" standalone="yes"?>
<ROOTTAG>
  <mytag>
    <headername>BASE</headername>
    <fieldname>NAME</fieldname>
    <val><![CDATA[Testcase]]></val>
  </mytag>
  <mytag>
    <headername>BASE</headername>
    <fieldname>AGE</fieldname>
    <val><![CDATA[5]]></val>
  </mytag>
</ROOTTAG>
-->

```

ElementTree (why this is painful):

```python
import xml.etree.ElementTree as ET

# ElementTree is an XML parser; this input is HTML + a commented XML payload,
# so parsing the whole document as XML will fail.
ET.fromstring(html)
```

JustHTML:

```python
from justhtml import JustHTML


doc = JustHTML(html, sanitize=False)

# New: select HTML comments directly.
xml_text: str | None = None
for c in doc.root.query(":comment"):
    if isinstance(c.data, str):
        t = c.data.strip()
        if t.startswith("<?xml") or "<ROOTTAG" in t:
            xml_text = t
            break

rows = []
if xml_text:
    # Drop any XML declaration to make the payload more "HTML-ish".
    if xml_text.startswith("<?xml"):
        end = xml_text.find("?>")
        if end != -1:
            xml_text = xml_text[end + 2 :].lstrip()

    # JustHTML is an HTML parser; strip CDATA wrappers to keep values readable.
    xml_text = xml_text.replace("<![CDATA[", "").replace("]]>", "")

    # Parse the XML payload with JustHTML (it will treat it as markup).
    # Note: HTML parsing lowercases tag names, so query lowercased names.
    xml_doc = JustHTML(xml_text, fragment=True, sanitize=False)

    for m in xml_doc.root.query("roottag mytag"):
        field = ""
        val = ""

        field_nodes = m.query("fieldname")
        if field_nodes:
            field = field_nodes[0].to_text(strip=True)

        val_nodes = m.query("val")
        if val_nodes:
            val = val_nodes[0].to_text(strip=True)

        rows.append((field, val))

print(rows)
# => [('NAME', 'Testcase'), ('AGE', '5')]
```
