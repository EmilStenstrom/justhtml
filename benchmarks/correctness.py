#!/usr/bin/env python3
"""
Correctness benchmark: Run html5lib test suite against multiple HTML parsers.

This tests how well each parser implements the HTML5 specification by comparing
their output against the expected results from the html5lib-tests suite.
"""
# ruff: noqa: PERF401, TRY300, BLE001, PLC0415

import argparse
import os
import re
import sys
from pathlib import Path

from justhtml import JustHTML, to_test_format
from justhtml.context import FragmentContext

# Available parsers
PARSERS = ["justhtml", "html5lib", "html5_parser", "lxml", "bs4", "html.parser", "selectolax"]


def check_parser_available(parser_name):
    """Check if a parser is available."""
    if parser_name == "justhtml":
        return True  # Always available (imported above)
    if parser_name == "html5lib":
        try:
            import html5lib  # noqa: F401

            return True
        except ImportError:
            return False
    if parser_name == "lxml":
        try:
            import lxml.html  # noqa: F401

            return True
        except ImportError:
            return False
    if parser_name == "bs4":
        try:
            from bs4 import BeautifulSoup  # noqa: F401

            return True
        except ImportError:
            return False
    if parser_name == "html.parser":
        return True  # stdlib, always available
    if parser_name == "selectolax":
        try:
            from selectolax.lexbor import LexborHTMLParser  # noqa: F401

            return True
        except ImportError:
            return False
    if parser_name == "html5_parser":
        try:
            import html5_parser  # noqa: F401

            return True
        except ImportError:
            return False
    return False


def parse_dat_file(path):
    """Parse a .dat test file into test cases."""
    with path.open("r", encoding="utf-8", newline="") as f:
        content = f.read()

    tests = []
    lines = content.split("\n")

    current_test_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        current_test_lines.append(line)

        if i + 1 >= len(lines) or (i + 1 < len(lines) and lines[i + 1] == "#data"):
            if current_test_lines and any(line.strip() for line in current_test_lines):
                test = parse_single_test(current_test_lines)
                if test:
                    tests.append(test)
            current_test_lines = []
        i += 1

    return tests


def parse_single_test(lines):
    """Parse a single test from lines."""
    data = []
    document = []
    fragment_context = None
    script_directive = None
    xml_coercion = False
    iframe_srcdoc = False
    mode = None

    for line in lines:
        if line.startswith("#"):
            directive = line[1:]
            if directive in ("script-on", "script-off"):
                script_directive = directive
            elif directive == "xml-coercion":
                xml_coercion = True
            elif directive == "iframe-srcdoc":
                iframe_srcdoc = True
            else:
                mode = directive
        elif mode == "data":
            data.append(line)
        elif mode == "document":
            document.append(line)
        elif mode == "document-fragment":
            fragment_str = line.strip()
            if " " in fragment_str:
                namespace, tag_name = fragment_str.split(" ", 1)
                fragment_context = (namespace, tag_name)
            else:
                fragment_context = (None, fragment_str)

    if data or document:
        return {
            "data": "\n".join(data),
            "document": "\n".join(document),
            "fragment_context": fragment_context,
            "script_directive": script_directive,
            "xml_coercion": xml_coercion,
            "iframe_srcdoc": iframe_srcdoc,
        }
    return None


def compare_outputs(expected, actual):
    """Compare expected and actual outputs, normalizing whitespace."""

    def normalize(text):
        return "\n".join(line.rstrip() for line in text.strip().splitlines())

    return normalize(expected) == normalize(actual)


def run_test_justhtml(html, fragment_context, expected, xml_coercion=False, iframe_srcdoc=False):
    """Run a single test with JustHTML."""
    from justhtml.tokenizer import TokenizerOpts

    try:
        opts = TokenizerOpts(xml_coercion=xml_coercion)
        if fragment_context:
            namespace, tag_name = fragment_context
            ctx = FragmentContext(tag_name, namespace)
            parser = JustHTML(
                html,
                fragment_context=ctx,
                tokenizer_opts=opts,
                iframe_srcdoc=iframe_srcdoc,
                safe=False,
            )
        else:
            parser = JustHTML(html, tokenizer_opts=opts, iframe_srcdoc=iframe_srcdoc, safe=False)
        actual = to_test_format(parser.root)
        passed = compare_outputs(expected, actual)
        return passed, actual, None
    except Exception as e:
        return False, "", str(e)


def run_test_html5lib(html, fragment_context, expected, xml_coercion=False, iframe_srcdoc=False):
    """Run a single test with html5lib using its native testSerializer."""
    import html5lib
    from html5lib import getTreeBuilder

    try:
        tree_builder = getTreeBuilder("etree", fullTree=True)
        # Use namespaceHTMLElements=True to get SVG/MathML namespace prefixes
        p = html5lib.HTMLParser(tree=tree_builder, namespaceHTMLElements=True)

        if fragment_context:
            _, tag_name = fragment_context
            doc = p.parseFragment(html, container=tag_name)
        else:
            doc = p.parse(html)

        # Use html5lib's native testSerializer
        raw_output = p.tree.testSerializer(doc)

        # Convert from html5lib format to test format
        # html5lib outputs: #document\n|  <html html>\n|    <html head>...
        # Expected format:  | <html>\n|   <head>...
        actual = _convert_html5lib_test_output(raw_output, is_fragment=fragment_context is not None)

        passed = compare_outputs(expected, actual)
        return passed, actual, None
    except Exception as e:
        return False, "", str(e)


def _convert_html5lib_test_output(data, is_fragment=False):
    """Convert html5lib testSerializer output to standard test format.

    Key transformations:
    - Remove #document/#document-fragment header
    - Convert |  (pipe+2 spaces) to | (pipe+1 space), adjusting indent
    - Strip 'html ' prefix from elements (keep 'svg '/'math ' prefixes)
    - Add 'content' wrapper for template element children per HTML5 spec
    - html5lib stores template content as element.text, not a separate fragment
    """
    lines = data.split("\n")

    # Skip first line (#document, #document-fragment, or |<DOCUMENT_FRAGMENT>)
    if lines:
        first = lines[0]
        if first in ("#document", "#document-fragment") or "DOCUMENT_FRAGMENT" in first:
            lines = lines[1:]

    result = []
    # Stack of template indent levels (in original |  format)
    template_indents = []

    for i, line in enumerate(lines):
        if line.startswith("|"):
            # Get original indent (spaces after | in html5lib format)
            # html5lib: |  <html> = 2 spaces base, +2 per level
            content_after_pipe = line[1:]  # Everything after |
            stripped = content_after_pipe.lstrip()
            orig_indent = len(content_after_pipe) - len(stripped)

            # Strip 'html ' namespace prefix from elements and attributes
            # Keep 'svg ' and 'math ' prefixes
            # Patterns: <html tagname>, html attr="value"
            if stripped.startswith("<html "):
                # <html tagname> -> <tagname>
                stripped = "<" + stripped[6:]
            elif stripped.startswith("html ") and "=" in stripped:
                # html attr="value" -> attr="value"
                stripped = stripped[5:]

            # Check if this line closes any templates
            # (line is at or before template's indent level)
            while template_indents and orig_indent <= template_indents[-1]:
                template_indents.pop()

            # Calculate extra indent from template nesting
            extra_indent = len(template_indents) * 2

            # Check if this is a template opening tag
            is_template_open = stripped.startswith(("<template>", "<html template>"))
            if stripped.startswith("<html template>"):
                stripped = "<template>"

            # Build the converted line with adjusted indent
            # html5lib base indent is 2, test format base is 0
            # So subtract 2 from orig_indent, then add template nesting
            new_indent = " " * (orig_indent - 2 + extra_indent)
            converted_line = f"| {new_indent}{stripped}"
            result.append(converted_line)

            # If opening a template, add content wrapper and track it
            if is_template_open:
                # Check if there's content after this template
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if next_line.startswith("|"):
                        next_content = next_line[1:]
                        next_stripped = next_content.lstrip()
                        next_orig_indent = len(next_content) - len(next_stripped)
                        # If next line is deeper (child content)
                        if (
                            next_orig_indent > orig_indent
                            and not next_stripped.startswith("</template>")
                            and not next_stripped.startswith("</html template>")
                        ):
                            # Add content wrapper
                            content_wrapper_indent = " " * (orig_indent - 2 + extra_indent + 2)
                            result.append(f"| {content_wrapper_indent}content")
                            # Track this template for child indent adjustment
                            template_indents.append(orig_indent)

        elif line.startswith("<!DOCTYPE"):
            # DOCTYPE line doesn't have | prefix in testSerializer
            result.append("| " + line)
        else:
            result.append(line)
    return "\n".join(result)


def run_test_lxml(html, fragment_context, expected, xml_coercion=False, iframe_srcdoc=False):
    """Run a single test with lxml."""
    import lxml.html
    from lxml import etree

    try:
        if fragment_context:
            # lxml fragment parsing is limited - skip these tests
            return False, "", "lxml does not support fragment parsing with context"
        doc = lxml.html.document_fromstring(html)
        # Check if input had DOCTYPE (lxml adds default if missing)
        has_doctype = html.lstrip()[:9].upper().startswith("<!DOCTYPE")
        actual = _lxml_document_to_test_format(doc, etree, has_doctype)
        passed = compare_outputs(expected, actual)
        return passed, actual, None
    except Exception as e:
        return False, "", str(e)


def run_test_bs4(html, fragment_context, expected, xml_coercion=False, iframe_srcdoc=False):
    """Run a single test with BeautifulSoup."""
    from bs4 import BeautifulSoup

    try:
        soup = BeautifulSoup(html, "html.parser")
        actual = _bs4_to_test_format(soup)
        passed = compare_outputs(expected, actual)
        return passed, actual, None
    except Exception as e:
        return False, "", str(e)


def run_test_html_parser(html, fragment_context, expected, xml_coercion=False, iframe_srcdoc=False):
    """Run a single test with stdlib html.parser."""
    from html.parser import HTMLParser

    class TreeBuilder(HTMLParser):
        def __init__(self):
            super().__init__()
            self.root = {"name": "#document", "children": []}
            self.stack = [self.root]
            self.void_elements = {
                "area",
                "base",
                "br",
                "col",
                "embed",
                "hr",
                "img",
                "input",
                "keygen",
                "link",
                "meta",
                "param",
                "source",
                "track",
                "wbr",
            }

        def handle_starttag(self, tag, attrs):
            # Convert None attribute values to empty string
            node = {"name": tag, "attrs": {k: (v if v is not None else "") for k, v in attrs}, "children": []}
            self.stack[-1]["children"].append(node)
            if tag.lower() not in self.void_elements:
                self.stack.append(node)

        def handle_endtag(self, tag):
            if len(self.stack) > 1 and self.stack[-1]["name"] == tag:
                self.stack.pop()

        def handle_data(self, data):
            if data:
                self.stack[-1]["children"].append({"name": "#text", "data": data})

        def handle_comment(self, data):
            self.stack[-1]["children"].append({"name": "#comment", "data": data})

        def handle_decl(self, decl):
            if decl.lower().startswith("doctype"):
                self.stack[-1]["children"].append({"name": "!doctype", "data": decl[8:].strip()})

    try:
        builder = TreeBuilder()
        builder.feed(html)
        actual = _dict_to_test_format(builder.root)
        passed = compare_outputs(expected, actual)
        return passed, actual, None
    except Exception as e:
        return False, "", str(e)


def run_test_selectolax(html, fragment_context, expected, xml_coercion=False, iframe_srcdoc=False):
    """Run a single test with selectolax (Lexbor backend)."""
    from selectolax.lexbor import LexborHTMLParser

    try:
        tree = LexborHTMLParser(html)
        actual = _selectolax_to_test_format(tree)
        passed = compare_outputs(expected, actual)
        return passed, actual, None
    except Exception as e:
        return False, "", str(e)


def run_test_html5_parser(html, fragment_context, expected, xml_coercion=False, iframe_srcdoc=False):
    """Run a single test with html5-parser (Gumbo backend)."""
    import html5_parser
    from lxml import etree

    try:
        if fragment_context:
            return False, "", "html5_parser does not support fragment parsing"
        # Use namespace_elements=True to get proper SVG/MathML namespace info
        # Use sanitize_names=False to preserve invalid chars in tag/attr names
        doc = html5_parser.parse(html, treebuilder="lxml", namespace_elements=True, sanitize_names=False)
        actual = _html5_parser_to_test_format(doc, etree, html)
        passed = compare_outputs(expected, actual)
        return passed, actual, None
    except Exception as e:
        return False, "", str(e)


# =============================================================================
# Test format conversion helpers
# =============================================================================

# Namespace URIs used in HTML5
NS_HTML = "http://www.w3.org/1999/xhtml"
NS_SVG = "http://www.w3.org/2000/svg"
NS_MATHML = "http://www.w3.org/1998/Math/MathML"
NS_XLINK = "http://www.w3.org/1999/xlink"
NS_XML = "http://www.w3.org/XML/1998/namespace"
NS_XMLNS = "http://www.w3.org/2000/xmlns/"


def _extract_doctype_from_html(html):
    """Extract DOCTYPE from original HTML since Gumbo normalizes it.

    Returns a tuple (name, public_id, system_id) or None if no DOCTYPE.
    The HTML5 spec says DOCTYPE stores:
    - name: the doctype name (lowercased per spec)
    - public identifier (optional)
    - system identifier (optional)
    """
    import re

    stripped = html.lstrip()
    if not stripped[:9].upper().startswith("<!DOCTYPE"):
        return None

    # Match DOCTYPE declaration - extract everything after <!DOCTYPE until >
    match = re.match(r"<!DOCTYPE\s*([^>]*?)>", stripped, re.IGNORECASE)
    if not match:
        return None

    content = match.group(1).strip()
    if not content:
        # Empty DOCTYPE like <!DOCTYPE>
        return ("", None, None)

    # Parse the DOCTYPE content per HTML5 spec:
    # DOCTYPE name is everything up to whitespace, PUBLIC, SYSTEM, or >
    parts = content.split(None, 1)
    name = parts[0].lower() if parts else ""  # HTML5 lowercases the name
    rest = parts[1] if len(parts) > 1 else ""

    # Check for PUBLIC or SYSTEM identifiers
    public_id = None
    system_id = None

    if rest:
        rest_upper = rest.upper()
        if rest_upper.startswith("PUBLIC"):
            # Parse PUBLIC "public_id" "system_id" or PUBLIC "public_id"
            # Allow either single or double quotes, must match
            pub_match = re.match(
                r'PUBLIC\s+(["\'])([^"\']*)\1(?:\s+(["\'])([^"\']*)\3)?',
                rest,
                re.IGNORECASE,
            )
            if pub_match:
                public_id = pub_match.group(2)
                system_id = pub_match.group(4)  # May be None
        elif rest_upper.startswith("SYSTEM"):
            # Parse SYSTEM "system_id"
            sys_match = re.match(r'SYSTEM\s+(["\'])([^"\']*)\1', rest, re.IGNORECASE)
            if sys_match:
                system_id = sys_match.group(2)
        # Otherwise it's invalid content after name - per HTML5, we just keep the name

    return (name, public_id, system_id)


def _html5_parser_to_test_format(doc, etree, original_html):
    """Convert html5_parser lxml document to test format with namespace support."""
    lines = []

    # Extract DOCTYPE from original HTML (Gumbo normalizes to 'html')
    doctype_info = _extract_doctype_from_html(original_html)
    if doctype_info is not None:
        name, public_id, system_id = doctype_info
        # Test format: <!DOCTYPE name "public_id" "system_id">
        # No PUBLIC/SYSTEM keywords, just quoted strings
        if public_id is not None or system_id is not None:
            pub = public_id if public_id is not None else ""
            sys = system_id if system_id is not None else ""
            lines.append(f'| <!DOCTYPE {name} "{pub}" "{sys}">')
        elif name == "":
            lines.append("| <!DOCTYPE >")
        else:
            lines.append(f"| <!DOCTYPE {name}>")

    # Serialize the root element with namespace awareness
    lines.extend(_html5_parser_element_to_lines(doc, 0, etree))
    return "\n".join(lines)


def _html5_parser_element_to_lines(elem, indent, etree):
    """Convert lxml element to test format lines with namespace prefixes."""
    prefix = " " * indent
    lines = []

    tag = elem.tag
    if callable(tag):
        # Special node types (Comment, ProcessingInstruction)
        if tag == etree.Comment:
            lines.append(f"| {prefix}<!-- {elem.text} -->")
        return lines

    # Parse namespace and local name from Clark notation {ns}local
    ns = None
    local_name = tag
    if tag.startswith("{"):
        ns_end = tag.find("}")
        if ns_end > 0:
            ns = tag[1:ns_end]
            local_name = tag[ns_end + 1 :]

    # Format tag with namespace prefix if needed
    if ns == NS_SVG:
        tag_str = f"svg {local_name}"
    elif ns == NS_MATHML:
        tag_str = f"math {local_name}"
    elif ns == NS_HTML or ns is None:
        # HTML namespace - tag names should be lowercase per spec
        tag_str = local_name.lower()
    else:
        # Unknown namespace - use full URI
        tag_str = f"{ns} {local_name}"

    lines.append(f"| {prefix}<{tag_str}>")

    # Attributes (sorted, with namespace handling)
    if elem.attrib:
        attr_lines = []
        for name, value in elem.attrib.items():
            # Parse attribute namespace
            attr_ns = None
            attr_local = name
            if name.startswith("{"):
                ns_end = name.find("}")
                if ns_end > 0:
                    attr_ns = name[1:ns_end]
                    attr_local = name[ns_end + 1 :]

            # Format attribute with namespace prefix if needed
            if attr_ns == NS_XLINK:
                attr_str = f"xlink {attr_local}"
            elif attr_ns == NS_XML:
                attr_str = f"xml {attr_local}"
            elif attr_ns == NS_XMLNS:
                attr_str = f"xmlns {attr_local}"
            elif attr_ns is None:
                attr_str = attr_local
            else:
                attr_str = f"{attr_ns} {attr_local}"

            attr_lines.append((attr_str, value))

        # Sort by attribute name (after namespace prefix)
        for attr_name, attr_value in sorted(attr_lines):
            lines.append(f'| {prefix}  {attr_name}="{attr_value}"')

    # Check if this is a template element - needs special "content" wrapper
    is_template = local_name == "template" and (ns == NS_HTML or ns is None)

    if is_template:
        # Template always has a "content" document fragment
        lines.append(f"| {prefix}  content")

        if elem.text or len(elem) > 0:
            content_prefix = " " * (indent + 4)

            # Text content (before first child) - inside content
            if elem.text:
                lines.append(f'| {content_prefix}"{elem.text}"')

            # Children - inside content
            for child in elem:
                lines.extend(_html5_parser_element_to_lines(child, indent + 4, etree))
                # Tail text (after this child)
                if child.tail:
                    lines.append(f'| {content_prefix}"{child.tail}"')
    else:
        # Normal element handling
        # Text content (before first child)
        if elem.text:
            lines.append(f'| {prefix}  "{elem.text}"')

        # Children
        for child in elem:
            lines.extend(_html5_parser_element_to_lines(child, indent + 2, etree))
            # Tail text (after this child, at current element's indent)
            if child.tail:
                lines.append(f'| {prefix}  "{child.tail}"')

    return lines


def _lxml_document_to_test_format(doc, etree, has_doctype):
    """Convert lxml document to test format."""
    lines = []

    # Only output DOCTYPE if the input had one
    if has_doctype:
        tree = doc.getroottree()
        doctype = tree.docinfo.doctype
        if doctype and doctype.startswith("<!DOCTYPE "):
            doctype_content = doctype[10:-1].strip()  # Remove <!DOCTYPE and >
            parts = doctype_content.split(None, 1)
            name = parts[0] if parts else "html"
            if len(parts) > 1:
                # Has public/system identifier
                rest = parts[1]
                lines.append(f"| <!DOCTYPE {name} {rest}>")
            else:
                lines.append(f"| <!DOCTYPE {name}>")

    # Serialize the root element
    lines.extend(_lxml_element_to_lines(doc, 0, etree))
    return "\n".join(lines)


def _lxml_element_to_lines(elem, indent, etree):
    """Convert lxml element to test format lines."""
    prefix = " " * indent
    lines = []

    tag = elem.tag
    if callable(tag):
        # Special node types (Comment, ProcessingInstruction)
        if tag == etree.Comment:
            lines.append(f"| {prefix}<!-- {elem.text} -->")
        return lines

    lines.append(f"| {prefix}<{tag}>")

    # Attributes (sorted)
    if elem.attrib:
        for name in sorted(elem.attrib.keys()):
            value = elem.attrib[name]
            lines.append(f'| {prefix}  {name}="{value}"')

    # Text content (before first child)
    if elem.text:
        lines.append(f'| {prefix}  "{elem.text}"')

    # Children
    for child in elem:
        lines.extend(_lxml_element_to_lines(child, indent + 2, etree))
        # Tail text (after this child, at current element's indent)
        if child.tail:
            lines.append(f'| {prefix}  "{child.tail}"')

    return lines


def _bs4_to_test_format(soup):
    """Convert BeautifulSoup tree to test format."""
    from bs4 import Comment, Doctype, NavigableString, Tag

    def process_node(node, indent):
        prefix = " " * indent
        lines = []

        for child in node.children:
            if isinstance(child, Doctype):
                lines.append(f"| <!DOCTYPE {child}>")
            elif isinstance(child, Comment):
                lines.append(f"| {prefix}<!-- {child} -->")
            elif isinstance(child, NavigableString):
                text = str(child)
                if text:
                    lines.append(f'| {prefix}"{text}"')
            elif isinstance(child, Tag):
                lines.append(f"| {prefix}<{child.name}>")
                # Attributes (sorted)
                if child.attrs:
                    for name in sorted(child.attrs.keys()):
                        value = child.attrs[name]
                        if isinstance(value, list):
                            value = " ".join(value)
                        lines.append(f'| {prefix}  {name}="{value}"')
                # Recurse
                lines.extend(process_node(child, indent + 2))
        return lines

    return "\n".join(process_node(soup, 0))


def _dict_to_test_format(node):
    """Convert dict-based tree to test format."""

    def process(node, indent):
        prefix = " " * indent
        lines = []
        name = node.get("name", "")

        if name == "#document":
            for child in node.get("children", []):
                lines.extend(process(child, 0))
        elif name == "#text":
            lines.append(f'| {prefix}"{node.get("data", "")}"')
        elif name == "#comment":
            lines.append(f"| {prefix}<!-- {node.get('data', '')} -->")
        elif name == "!doctype":
            data = node.get("data", "html")
            lines.append(f"| <!DOCTYPE {data}>")
        else:
            lines.append(f"| {prefix}<{name}>")
            attrs = node.get("attrs", {})
            if attrs:
                for aname in sorted(attrs.keys()):
                    lines.append(f'| {prefix}  {aname}="{attrs[aname]}"')
            for child in node.get("children", []):
                lines.extend(process(child, indent + 2))
        return lines

    return "\n".join(process(node, 0))


def _selectolax_to_test_format(tree):
    """Convert selectolax tree to test format."""

    def walk(node, indent):
        prefix = " " * indent
        lines = []
        tag = node.tag

        if tag == "-text":
            # Text node
            text = node.text_content
            if text:
                lines.append(f'| {prefix}"{text}"')
        elif tag == "-comment":
            # Comment node - extract text from html property
            # Format: <!-- content --> (with space padding around content)
            comment_html = node.html or ""
            if comment_html.startswith("<!--") and comment_html.endswith("-->"):
                comment_text = comment_html[4:-3]  # Remove <!-- and -->
                lines.append(f"| {prefix}<!-- {comment_text} -->")
        elif tag == "-doctype":
            # DOCTYPE node - extract from html property
            doctype_html = node.html
            if doctype_html and doctype_html.startswith("<!DOCTYPE"):
                # Extract name from <!DOCTYPE name>
                content = doctype_html[9:-1].strip()  # Remove <!DOCTYPE and >
                lines.append(f"| <!DOCTYPE {content}>")
        elif tag and not tag.startswith("-"):
            # Element node
            lines.append(f"| {prefix}<{tag}>")

            # Attributes (sorted)
            if node.attributes:
                for name in sorted(node.attributes.keys()):
                    value = node.attributes[name]
                    if value is None:
                        value = ""
                    lines.append(f'| {prefix}  {name}="{value}"')

            # Children
            child = node.child
            while child:
                lines.extend(walk(child, indent + 2))
                child = child.next

        return lines

    # Start from document node (parent of root) to capture DOCTYPE
    root = tree.root
    if root is None:
        return ""

    doc = root.parent
    if doc and doc.tag == "":
        # Document node - iterate its children (DOCTYPE, html)
        lines = []
        child = doc.child
        while child:
            lines.extend(walk(child, 0))
            child = child.next
        return "\n".join(lines)

    # Fallback to just root
    return "\n".join(walk(root, 0))


# Parser dispatch
PARSER_RUNNERS = {
    "justhtml": run_test_justhtml,
    "html5lib": run_test_html5lib,
    "html5_parser": run_test_html5_parser,
    "lxml": run_test_lxml,
    "bs4": run_test_bs4,
    "html.parser": run_test_html_parser,
    "selectolax": run_test_selectolax,
}


def collect_test_files(test_dir, exclude_files=None):
    """Collect .dat test files."""
    files = []
    for root, _, filenames in os.walk(test_dir, followlinks=True):
        for filename in filenames:
            if filename.endswith(".dat"):
                files.append(Path(root) / filename)

    if exclude_files:
        files = [f for f in files if not any(excl in f.name for excl in exclude_files)]

    def natural_sort_key(path):
        def convert(text):
            return int(text) if text.isdigit() else text.lower()

        return [convert(c) for c in re.split("([0-9]+)", str(path))]

    return sorted(files, key=natural_sort_key)


def run_correctness_tests(args):
    """Run correctness tests."""
    # Determine which parsers to test
    if args.parsers:
        parser_names = [p.strip() for p in args.parsers.split(",")]
    else:
        parser_names = PARSERS  # Default to all parsers

    # Check parser availability
    available_parsers = []
    for name in parser_names:
        if name not in PARSER_RUNNERS:
            print(f"Warning: Unknown parser '{name}', skipping")
            continue
        if check_parser_available(name):
            available_parsers.append(name)
        else:
            print(f"Note: Parser '{name}' not available (not installed), skipping")

    if not available_parsers:
        print("Error: No parsers available", file=sys.stderr)
        sys.exit(1)

    # Collect test files from all test directories
    test_dirs = [
        "tests/html5lib-tests-tree",
        # "tests/justhtml-tests",
    ]

    exclude_files = args.exclude_files.split(",") if args.exclude_files else None
    test_files = []
    for test_dir in test_dirs:
        test_path = Path(test_dir)
        if test_path.exists():
            test_files.extend(collect_test_files(test_path, exclude_files))

    if not test_files:
        print("Error: No test files found", file=sys.stderr)
        sys.exit(1)

    print(f"Running {len(test_files)} test files against {len(available_parsers)} parser(s)")
    print(f"Parsers: {', '.join(available_parsers)}")
    print()

    # Results tracking per parser
    results = {name: {"passed": 0, "failed": 0, "errors": 0, "skipped": 0} for name in available_parsers}
    failures = {name: [] for name in available_parsers}
    total_tests = 0

    # Run tests
    for file_path in test_files:
        tests = parse_dat_file(file_path)
        file_name = file_path.name

        for i, test in enumerate(tests):
            # Skip script-dependent tests
            if test["script_directive"] in ("script-on", "script-off"):
                for name in available_parsers:
                    results[name]["skipped"] += 1
                continue

            total_tests += 1
            html = test["data"]
            expected = test["document"]
            fragment = test["fragment_context"]
            xml_coercion = test.get("xml_coercion", False)
            iframe_srcdoc = test.get("iframe_srcdoc", False)

            for parser_name in available_parsers:
                runner = PARSER_RUNNERS[parser_name]
                passed, actual, error = runner(
                    html,
                    fragment,
                    expected,
                    xml_coercion=xml_coercion,
                    iframe_srcdoc=iframe_srcdoc,
                )

                if error:
                    results[parser_name]["errors"] += 1
                    if args.verbose >= 2:
                        print(f"[{parser_name}] ERROR {file_name}:{i} - {error}")
                elif passed:
                    results[parser_name]["passed"] += 1
                else:
                    results[parser_name]["failed"] += 1
                    if args.verbose >= 1:
                        failures[parser_name].append(
                            {
                                "file": file_name,
                                "index": i,
                                "html": html,
                                "expected": expected,
                                "actual": actual,
                            }
                        )

        if not args.quiet:
            # Progress indicator
            print(f"\r{file_name}: done", end="", flush=True)

    print()  # Newline after progress
    print()

    # Print results table
    print("=" * 70)
    print("CORRECTNESS RESULTS")
    print("=" * 70)
    print(f"{'Parser':<15} {'Passed':>10} {'Failed':>10} {'Errors':>10} {'Skipped':>10} {'Pass Rate':>12}")
    print("-" * 70)

    for name in available_parsers:
        r = results[name]
        total = r["passed"] + r["failed"] + r["errors"]
        rate = (r["passed"] / total * 100) if total > 0 else 0
        print(f"{name:<15} {r['passed']:>10} {r['failed']:>10} {r['errors']:>10} {r['skipped']:>10} {rate:>11.2f}%")

    print("-" * 70)
    print(f"Total test cases: {total_tests}")
    print()

    # Print failures if verbose
    if args.verbose >= 1:
        for parser_name in available_parsers:
            parser_failures = failures[parser_name]
            if parser_failures:
                max_show = args.show_failures
                print(
                    f"{parser_name} failures (showing first {min(max_show, len(parser_failures))} of {len(parser_failures)}):"
                )
                print("-" * 60)
                for fail in parser_failures[:max_show]:
                    print(f"  {fail['file']}:{fail['index']}")
                    print(f"    Input: {fail['html'][:60]!r}...")
                    if args.verbose >= 2:
                        print(f"    Expected:\n{fail['expected']}")
                        print(f"    Actual:\n{fail['actual']}")
                    print()

    return results


def main():
    parser = argparse.ArgumentParser(description="Run HTML5 correctness tests against HTML parsers")
    parser.add_argument(
        "--parsers",
        help=f"Comma-separated list of parsers to test (available: {', '.join(PARSERS)})",
    )
    parser.add_argument(
        "--exclude-files",
        help="Comma-separated list of file patterns to exclude",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v for failures, -vv for diffs)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--show-failures",
        type=int,
        default=5,
        help="Number of failures to show per parser (default: 5)",
    )

    args = parser.parse_args()
    run_correctness_tests(args)


if __name__ == "__main__":
    main()
