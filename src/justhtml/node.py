from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from .selector import query
from .serialize import to_html

if TYPE_CHECKING:
    from .serialize import HTMLContext
    from .tokens import Doctype


def _markdown_escape_text(s: str) -> str:
    if not s:
        return ""
    # Pragmatic: escape the few characters that commonly change Markdown meaning.
    # Keep this minimal to preserve readability.
    out: list[str] = []
    for ch in s:
        if ch in "\\`*_[]":
            out.append("\\")
        out.append(ch)
    return "".join(out)


def _markdown_code_span(s: str | None) -> str:
    if s is None:
        s = ""
    # Use a backtick fence longer than any run of backticks inside.
    longest = 0
    run = 0
    for ch in s:
        if ch == "`":
            run += 1
            if run > longest:
                longest = run
        else:
            run = 0
    fence = "`" * (longest + 1)
    # CommonMark requires a space if the content starts/ends with backticks.
    needs_space = s.startswith("`") or s.endswith("`")
    if needs_space:
        return f"{fence} {s} {fence}"
    return f"{fence}{s}{fence}"


def _markdown_link_destination(url: str) -> str:
    """Return a Markdown-safe link destination.

    We primarily care about avoiding Markdown formatting injection and broken
    parsing for URLs that contain whitespace or parentheses.

    CommonMark supports destinations wrapped in angle brackets:
    `[text](<https://example.com/a(b)c>)`
    """

    u = (url or "").strip()
    if not u:
        return ""

    # If the destination contains characters that can terminate or confuse
    # the Markdown destination parser, wrap in <...> and percent-encode
    # whitespace and angle brackets.
    if any(ch in u for ch in (" ", "\t", "\n", "\r", "(", ")", "<", ">")):
        u = quote(u, safe=":/?#[]@!$&'*+,;=%-._~()")
        return f"<{u}>"

    return u


class _MarkdownBuilder:
    __slots__ = ("_buf", "_newline_count", "_pending_space")

    _buf: list[str]
    _newline_count: int
    _pending_space: bool

    def __init__(self) -> None:
        self._buf = []
        self._newline_count = 0
        self._pending_space = False

    def _rstrip_last_segment(self) -> None:
        if not self._buf:
            return
        last = self._buf[-1]
        stripped = last.rstrip(" \t")
        if stripped != last:
            self._buf[-1] = stripped

    def newline(self, count: int = 1) -> None:
        for _ in range(count):
            self._pending_space = False
            self._rstrip_last_segment()
            self._buf.append("\n")
            # Track newlines to make it easy to insert blank lines.
            if self._newline_count < 2:
                self._newline_count += 1

    def ensure_newlines(self, count: int) -> None:
        while self._newline_count < count:
            self.newline(1)

    def raw(self, s: str) -> None:
        if not s:
            return

        # If we've collapsed whitespace and the next output is raw (e.g. "**"),
        # we still need to emit a single separating space.
        if self._pending_space:
            first = s[0]
            if first not in " \t\n\r\f" and self._buf and self._newline_count == 0:
                self._buf.append(" ")
            self._pending_space = False

        self._buf.append(s)
        if "\n" in s:
            # Count trailing newlines (cap at 2 for blank-line semantics).
            trailing = 0
            i = len(s) - 1
            while i >= 0 and s[i] == "\n":
                trailing += 1
                i -= 1
            self._newline_count = min(2, trailing)
            if trailing:
                self._pending_space = False
        else:
            self._newline_count = 0

    def text(self, s: str, preserve_whitespace: bool = False) -> None:
        if not s:
            return

        if preserve_whitespace:
            self.raw(s)
            return

        for ch in s:
            if ch in " \t\n\r\f":
                self._pending_space = True
                continue

            if self._pending_space:
                if self._buf and self._newline_count == 0:
                    self._buf.append(" ")
                self._pending_space = False

            self._buf.append(ch)
            self._newline_count = 0

    def finish(self) -> str:
        out = "".join(self._buf)
        return out.strip(" \t\n")


# Type alias for any node type
NodeType = "Node | Element | Template | Text | Comment | Document | DocumentFragment"


def _to_text_collect(node: Any, parts: list[str], strip: bool) -> None:
    # Iterative traversal avoids recursion overhead on large documents.
    stack: list[Any] = [node]
    while stack:
        current = stack.pop()
        name: str = current.name

        if name == "#text":
            data: str | None = current.data
            if not data:
                continue
            if strip:
                data = data.strip()
                if not data:
                    continue
            parts.append(data)
            continue

        # Preserve the same traversal order as the recursive implementation:
        # children first, then template content.
        if type(current) is Template and current.template_content:
            stack.append(current.template_content)

        children = current.children
        if children:
            stack.extend(reversed(children))


_TEXT_BLOCK_ELEMENTS: frozenset[str] = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "body",
        "dd",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "html",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    }
)

_TEXT_BREAK_ELEMENTS: frozenset[str] = frozenset({"br"})


def _to_text_break(chunks: list[list[str]]) -> None:
    if chunks and chunks[-1]:
        chunks.append([])


def _to_text_collect_block_chunks(node: Any, chunks: list[list[str]], strip: bool) -> None:
    # Depth-first walk that inserts chunk boundaries for block-level elements.
    # This lets callers join chunks with a separator (e.g. "\n") without
    # introducing separators inside inline elements like <b> or <span>.
    stack: list[tuple[Any, int]] = [(node, 0)]  # (node, state), state: 0=enter, 1=exit
    while stack:
        current, state = stack.pop()
        name: str = current.name

        if state == 1:
            _to_text_break(chunks)
            continue

        if name == "#text":
            data: str | None = current.data
            if not data:
                continue
            if strip:
                data = data.strip()
                if not data:
                    continue
            chunks[-1].append(data)
            continue

        if name in _TEXT_BREAK_ELEMENTS:
            _to_text_break(chunks)
            continue

        if name in _TEXT_BLOCK_ELEMENTS:
            _to_text_break(chunks)
            stack.append((current, 1))

        # Preserve the same traversal order as the recursive implementation:
        # children first, then template content.
        if type(current) is Template and current.template_content:
            stack.append((current.template_content, 0))

        children = current.children
        if children:
            stack.extend((child, 0) for child in reversed(children))


class Node:
    __slots__ = (
        "_origin_col",
        "_origin_line",
        "_origin_pos",
        "_source_html",
        "attrs",
        "children",
        "data",
        "name",
        "namespace",
        "parent",
    )

    name: str
    parent: Node | Element | Template | None
    attrs: dict[str, str | None] | None
    children: list[Any] | None
    data: str | Doctype | None
    namespace: str | None
    _origin_pos: int | None
    _origin_line: int | None
    _origin_col: int | None
    _source_html: str | None

    def __init__(
        self,
        name: str,
        attrs: dict[str, str | None] | None = None,
        data: str | Doctype | None = None,
        namespace: str | None = None,
    ) -> None:
        self.name = name
        self.parent = None
        self.data = data
        self._source_html = None
        self._origin_pos = None
        self._origin_line = None
        self._origin_col = None

        if name.startswith("#") or name == "!doctype":
            self.namespace = namespace
            if name == "#comment" or name == "!doctype":
                self.children = None
                self.attrs = None
            else:
                self.children = []
                self.attrs = attrs if attrs is not None else {}
        else:
            self.namespace = namespace or "html"
            self.children = []
            self.attrs = attrs if attrs is not None else {}

    def append_child(self, node: Any) -> None:
        if self.children is not None:
            self.children.append(node)
            node.parent = self

    @property
    def origin_offset(self) -> int | None:
        """Best-effort origin offset (0-indexed) in the source HTML, if known."""
        return self._origin_pos

    @property
    def origin_line(self) -> int | None:
        return self._origin_line

    @property
    def origin_col(self) -> int | None:
        return self._origin_col

    @property
    def origin_location(self) -> tuple[int, int] | None:
        if self._origin_line is None or self._origin_col is None:
            return None
        return (self._origin_line, self._origin_col)

    def remove_child(self, node: Any) -> None:
        if self.children is not None:
            self.children.remove(node)
            node.parent = None

    def to_html(
        self,
        indent: int = 0,
        indent_size: int = 2,
        pretty: bool = True,
        *,
        context: HTMLContext | None = None,
        quote: str = '"',
    ) -> str:
        """Convert node to HTML string."""
        return to_html(self, indent, indent_size, pretty=pretty, context=context, quote=quote)

    def query(self, selector: str) -> list[Any]:
        """
        Query this subtree using a CSS selector.

        Args:
            selector: A CSS selector string

        Returns:
            A list of matching nodes

        Raises:
            ValueError: If the selector is invalid
        """
        result: list[Any] = query(self, selector)
        return result

    def query_one(self, selector: str) -> Any | None:
        """Return the first matching descendant for a CSS selector, or None."""
        matches = self.query(selector)
        if not matches:
            return None
        return matches[0]

    @property
    def text(self) -> str:
        """Return the node's own text value.

        For text nodes this is the node data. For other nodes this is an empty
        string. Use `to_text()` to get textContent semantics.
        """
        if self.name == "#text":
            data = self.data
            if isinstance(data, str):
                return data
            return ""
        return ""

    def to_text(
        self,
        separator: str = " ",
        strip: bool = True,
        *,
        separator_blocks_only: bool = False,
    ) -> str:
        """Return the concatenated text of this node's descendants.

        - `separator` controls how text nodes are joined (default: a single space).
        - `strip=True` strips each text node and drops empty segments.
        - `separator_blocks_only=True` only applies `separator` between block-level
          elements, avoiding separators inside inline elements (like `<b>`).
        Template element contents are included via `template_content`.
        """
        node: Any = self
        if not separator_blocks_only:
            parts: list[str] = []
            _to_text_collect(node, parts, strip=strip)
            if not parts:
                return ""
            return separator.join(parts)

        chunks: list[list[str]] = [[]]
        _to_text_collect_block_chunks(node, chunks, strip=strip)

        intra_sep = " " if strip else ""
        texts: list[str] = []
        for chunk in chunks:
            if not chunk:
                continue
            texts.append(intra_sep.join(chunk))

        if not texts:
            return ""
        return separator.join(texts)

    def to_markdown(self, html_passthrough: bool = False) -> str:
        """Return a GitHub Flavored Markdown representation of this subtree.

        This is a pragmatic HTML->Markdown converter intended for readability.
        - Tables and images are preserved as raw HTML.
        - Unknown elements fall back to rendering their children.
        """
        builder = _MarkdownBuilder()
        _to_markdown_walk(
            self,
            builder,
            preserve_whitespace=False,
            list_depth=0,
            html_passthrough=html_passthrough,
        )
        return builder.finish()

    def insert_before(self, node: Any, reference_node: Any | None) -> None:
        """
        Insert a node before a reference node.

        Args:
            node: The node to insert
            reference_node: The node to insert before. If None, append to end.

        Raises:
            ValueError: If reference_node is not a child of this node
        """
        if self.children is None:
            raise ValueError(f"Node {self.name} cannot have children")

        if reference_node is None:
            self.append_child(node)
            return

        try:
            index = self.children.index(reference_node)
            self.children.insert(index, node)
            node.parent = self
        except ValueError:
            raise ValueError("Reference node is not a child of this node") from None

    def replace_child(self, new_node: Any, old_node: Any) -> Any:
        """
        Replace a child node with a new node.

        Args:
            new_node: The new node to insert
            old_node: The child node to replace

        Returns:
            The replaced node (old_node)

        Raises:
            ValueError: If old_node is not a child of this node
        """
        if self.children is None:
            raise ValueError(f"Node {self.name} cannot have children")

        try:
            index = self.children.index(old_node)
        except ValueError:
            raise ValueError("The node to be replaced is not a child of this node") from None

        self.children[index] = new_node
        new_node.parent = self
        old_node.parent = None
        return old_node

    def has_child_nodes(self) -> bool:
        """Return True if this node has children."""
        return bool(self.children)

    def clone_node(self, deep: bool = False, override_attrs: dict[str, str | None] | None = None) -> Node:
        """
        Clone this node.

        Args:
            deep: If True, recursively clone children.
            override_attrs: Optional dictionary to use as attributes for the clone.

        Returns:
            A new node that is a copy of this node.
        """
        attrs = override_attrs if override_attrs is not None else (self.attrs.copy() if self.attrs else None)
        clone = Node(
            self.name,
            attrs,
            self.data,
            self.namespace,
        )
        clone._source_html = self._source_html
        clone._origin_pos = self._origin_pos
        clone._origin_line = self._origin_line
        clone._origin_col = self._origin_col
        if deep and self.children:
            for child in self.children:
                clone.append_child(child.clone_node(deep=True))
        return clone


class Document(Node):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("#document")

    def clone_node(self, deep: bool = False, override_attrs: dict[str, str | None] | None = None) -> Document:
        _ = override_attrs
        clone = Document()
        clone._source_html = self._source_html
        clone._origin_pos = self._origin_pos
        clone._origin_line = self._origin_line
        clone._origin_col = self._origin_col
        if deep and self.children:
            for child in self.children:
                clone.append_child(child.clone_node(deep=True))
        return clone


class DocumentFragment(Node):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("#document-fragment")

    def clone_node(self, deep: bool = False, override_attrs: dict[str, str | None] | None = None) -> DocumentFragment:
        _ = override_attrs
        clone = DocumentFragment()
        clone._source_html = self._source_html
        clone._origin_pos = self._origin_pos
        clone._origin_line = self._origin_line
        clone._origin_col = self._origin_col
        if deep and self.children:
            for child in self.children:
                clone.append_child(child.clone_node(deep=True))
        return clone


class Comment(Node):
    __slots__ = ()

    def __init__(self, data: str | None = None) -> None:
        super().__init__("#comment", data=data)

    def clone_node(self, deep: bool = False, override_attrs: dict[str, str | None] | None = None) -> Comment:
        _ = override_attrs
        _ = deep
        clone = Comment(self.data if isinstance(self.data, str) else None)
        clone._source_html = self._source_html
        clone._origin_pos = self._origin_pos
        clone._origin_line = self._origin_line
        clone._origin_col = self._origin_col
        return clone


class Element(Node):
    __slots__ = (
        "_end_tag_end",
        "_end_tag_present",
        "_end_tag_start",
        "_self_closing",
        "_start_tag_end",
        "_start_tag_start",
        "template_content",
    )

    template_content: Node | None
    children: list[Any]
    attrs: dict[str, str | None]
    _start_tag_start: int | None
    _start_tag_end: int | None
    _end_tag_start: int | None
    _end_tag_end: int | None
    _end_tag_present: bool
    _self_closing: bool

    def __init__(self, name: str, attrs: dict[str, str | None] | None, namespace: str | None) -> None:
        self.name = name
        self.parent = None
        self.data = None
        self.namespace = namespace
        self.children = []
        self.attrs = attrs if attrs is not None else {}
        self.template_content = None
        self._source_html = None
        self._origin_pos = None
        self._origin_line = None
        self._origin_col = None
        self._start_tag_start = None
        self._start_tag_end = None
        self._end_tag_start = None
        self._end_tag_end = None
        self._end_tag_present = False
        self._self_closing = False

    def clone_node(self, deep: bool = False, override_attrs: dict[str, str | None] | None = None) -> Element:
        attrs = override_attrs if override_attrs is not None else (self.attrs.copy() if self.attrs else {})
        clone = Element(self.name, attrs, self.namespace)
        clone._source_html = self._source_html
        clone._origin_pos = self._origin_pos
        clone._origin_line = self._origin_line
        clone._origin_col = self._origin_col
        clone._start_tag_start = self._start_tag_start
        clone._start_tag_end = self._start_tag_end
        clone._end_tag_start = self._end_tag_start
        clone._end_tag_end = self._end_tag_end
        clone._end_tag_present = self._end_tag_present
        clone._self_closing = self._self_closing
        if deep:
            for child in self.children:
                clone.append_child(child.clone_node(deep=True))
        return clone


class Template(Element):
    __slots__ = ()

    def __init__(
        self,
        name: str,
        attrs: dict[str, str | None] | None = None,
        data: str | None = None,
        namespace: str | None = None,
    ) -> None:
        super().__init__(name, attrs, namespace)
        if self.namespace == "html":
            self.template_content = DocumentFragment()
        else:
            self.template_content = None

    def clone_node(self, deep: bool = False, override_attrs: dict[str, str | None] | None = None) -> Template:
        attrs = override_attrs if override_attrs is not None else (self.attrs.copy() if self.attrs else {})
        clone = Template(
            self.name,
            attrs,
            None,
            self.namespace,
        )
        clone._source_html = self._source_html
        clone._origin_pos = self._origin_pos
        clone._origin_line = self._origin_line
        clone._origin_col = self._origin_col
        clone._start_tag_start = self._start_tag_start
        clone._start_tag_end = self._start_tag_end
        clone._end_tag_start = self._end_tag_start
        clone._end_tag_end = self._end_tag_end
        clone._end_tag_present = self._end_tag_present
        clone._self_closing = self._self_closing
        if deep:
            if self.template_content:
                clone.template_content = self.template_content.clone_node(deep=True)
            for child in self.children:
                clone.append_child(child.clone_node(deep=True))
        return clone


class Text:
    __slots__ = ("_origin_col", "_origin_line", "_origin_pos", "data", "name", "namespace", "parent")

    data: str | None
    name: str
    namespace: None
    parent: Node | Element | Template | None
    _origin_pos: int | None
    _origin_line: int | None
    _origin_col: int | None

    def __init__(self, data: str | None) -> None:
        self.data = data
        self.parent = None
        self.name = "#text"
        self.namespace = None
        self._origin_pos = None
        self._origin_line = None
        self._origin_col = None

    @property
    def origin_offset(self) -> int | None:
        """Best-effort origin offset (0-indexed) in the source HTML, if known."""
        return self._origin_pos

    @property
    def origin_line(self) -> int | None:
        return self._origin_line

    @property
    def origin_col(self) -> int | None:
        return self._origin_col

    @property
    def origin_location(self) -> tuple[int, int] | None:
        if self._origin_line is None or self._origin_col is None:
            return None
        return (self._origin_line, self._origin_col)

    @property
    def text(self) -> str:
        """Return the text content of this node."""
        return self.data or ""

    def to_text(
        self,
        separator: str = " ",
        strip: bool = True,
        *,
        separator_blocks_only: bool = False,
    ) -> str:
        _ = separator
        _ = separator_blocks_only
        if self.data is None:
            return ""
        if strip:
            return self.data.strip()
        return self.data

    def to_markdown(self, html_passthrough: bool = False) -> str:
        _ = html_passthrough
        builder = _MarkdownBuilder()
        builder.text(_markdown_escape_text(self.data or ""), preserve_whitespace=False)
        return builder.finish()

    @property
    def children(self) -> list[Any]:
        """Return empty list for Text (leaf node)."""
        return []

    def has_child_nodes(self) -> bool:
        """Return False for Text."""
        return False

    def clone_node(self, deep: bool = False) -> Text:
        clone = Text(self.data)
        clone._origin_pos = self._origin_pos
        clone._origin_line = self._origin_line
        clone._origin_col = self._origin_col
        return clone


_MARKDOWN_BLOCK_ELEMENTS: frozenset[str] = frozenset(
    {
        "p",
        "div",
        "section",
        "article",
        "header",
        "footer",
        "main",
        "nav",
        "aside",
        "blockquote",
        "pre",
        "ul",
        "ol",
        "li",
        "hr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "table",
    }
)


def _to_markdown_walk(
    node: Any,
    builder: _MarkdownBuilder,
    preserve_whitespace: bool,
    list_depth: int,
    in_link: bool = False,
    html_passthrough: bool = False,
) -> None:
    name: str = node.name

    if name == "#text":
        if preserve_whitespace:
            builder.raw(node.data or "")
        else:
            builder.text(_markdown_escape_text(node.data or ""), preserve_whitespace=False)
        return

    if name == "br":
        if in_link:
            builder.text(" ", preserve_whitespace=False)
        else:
            builder.newline(1)
        return

    # Comments/doctype don't contribute.
    if name == "#comment" or name == "!doctype":
        return

    # Document containers contribute via descendants.
    if name.startswith("#"):
        if node.children:
            for child in node.children:
                _to_markdown_walk(
                    child,
                    builder,
                    preserve_whitespace,
                    list_depth,
                    in_link=in_link,
                    html_passthrough=html_passthrough,
                )
        return

    tag = name.lower()

    # Metadata containers don't contribute to body text.
    if tag == "head" or tag == "title":
        return

    # Preserve <img>, <table>, and raw-text elements as HTML.
    if tag == "img":
        builder.raw(node.to_html(indent=0, indent_size=2, pretty=False))
        return

    if tag in {"table", "script", "style", "textarea"}:
        if not in_link:
            builder.ensure_newlines(2 if builder._buf else 0)
        if tag in {"script", "style", "textarea"}:
            if not html_passthrough:
                return
            builder.raw(f"<{tag}>")
            content = node.to_text(separator="", strip=False)
            if content:
                builder.raw(content)
            builder.raw(f"</{tag}>")
        else:
            builder.raw(node.to_html(indent=0, indent_size=2, pretty=False))
        if not in_link:
            builder.ensure_newlines(2)
        return

    # Headings.
    if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        if not in_link:
            builder.ensure_newlines(2 if builder._buf else 0)
            level = int(tag[1])
            builder.raw("#" * level)
            builder.raw(" ")

        if node.children:
            for child in node.children:
                _to_markdown_walk(
                    child,
                    builder,
                    preserve_whitespace=False,
                    list_depth=list_depth,
                    in_link=in_link,
                    html_passthrough=html_passthrough,
                )

        if not in_link:
            builder.ensure_newlines(2)
        return

    # Horizontal rule.
    if tag == "hr":
        if not in_link:
            builder.ensure_newlines(2 if builder._buf else 0)
            builder.raw("---")
            builder.ensure_newlines(2)
        return

    # Code blocks.
    if tag == "pre":
        if not in_link:
            builder.ensure_newlines(2 if builder._buf else 0)
            code = node.to_text(separator="", strip=False)
            builder.raw("```")
            builder.newline(1)
            if code:
                builder.raw(code.rstrip("\n"))
                builder.newline(1)
            builder.raw("```")
            builder.ensure_newlines(2)
        else:
            # Inside link, render as inline code or text
            code = node.to_text(separator="", strip=False)
            builder.raw(_markdown_code_span(code))
        return

    # Inline code.
    if tag == "code" and not preserve_whitespace:
        code = node.to_text(separator="", strip=False)
        builder.raw(_markdown_code_span(code))
        return

    # Paragraph-like blocks.
    if tag == "p":
        if not in_link:
            builder.ensure_newlines(2 if builder._buf else 0)

        if node.children:
            for child in node.children:
                _to_markdown_walk(
                    child,
                    builder,
                    preserve_whitespace=False,
                    list_depth=list_depth,
                    in_link=in_link,
                    html_passthrough=html_passthrough,
                )

        if not in_link:
            builder.ensure_newlines(2)
        else:
            builder.text(" ", preserve_whitespace=False)
        return

    # Blockquotes.
    if tag == "blockquote":
        if not in_link:
            builder.ensure_newlines(2 if builder._buf else 0)
            inner = _MarkdownBuilder()
            if node.children:
                for child in node.children:
                    _to_markdown_walk(
                        child,
                        inner,
                        preserve_whitespace=False,
                        list_depth=list_depth,
                        in_link=in_link,
                        html_passthrough=html_passthrough,
                    )
            text = inner.finish()
            if text:
                lines = text.split("\n")
                for i, line in enumerate(lines):
                    if i:
                        builder.newline(1)
                    builder.raw("> ")
                    builder.raw(line)
            builder.ensure_newlines(2)
        else:
            if node.children:
                for child in node.children:
                    _to_markdown_walk(
                        child,
                        builder,
                        preserve_whitespace=False,
                        list_depth=list_depth,
                        in_link=in_link,
                        html_passthrough=html_passthrough,
                    )
        return

    # Lists.
    if tag in {"ul", "ol"}:
        if not in_link:
            builder.ensure_newlines(2 if builder._buf else 0)
            ordered = tag == "ol"
            idx = 1
            for child in node.children or []:
                if child.name.lower() != "li":
                    continue
                if idx > 1:
                    builder.newline(1)
                indent = "  " * list_depth
                marker = f"{idx}. " if ordered else "- "
                builder.raw(indent)
                builder.raw(marker)
                # Render list item content inline-ish.
                for li_child in child.children or []:
                    _to_markdown_walk(
                        li_child,
                        builder,
                        preserve_whitespace=False,
                        list_depth=list_depth + 1,
                        in_link=in_link,
                        html_passthrough=html_passthrough,
                    )
                idx += 1
            builder.ensure_newlines(2)
        else:
            # Flatten list inside link
            for child in node.children or []:
                if child.name.lower() != "li":
                    continue
                builder.raw(" ")
                for li_child in child.children or []:
                    _to_markdown_walk(
                        li_child,
                        builder,
                        preserve_whitespace=False,
                        list_depth=list_depth + 1,
                        in_link=in_link,
                        html_passthrough=html_passthrough,
                    )
        return

    # Emphasis/strong.
    if tag in {"em", "i"}:
        inner = _MarkdownBuilder()
        for child in node.children or []:
            _to_markdown_walk(
                child,
                inner,
                preserve_whitespace=False,
                list_depth=list_depth,
                in_link=in_link,
                html_passthrough=html_passthrough,
            )
        content = inner.finish()
        if not content:
            return
        builder.raw("*")
        builder.raw(content)
        builder.raw("*")
        return

    if tag in {"strong", "b"}:
        inner = _MarkdownBuilder()
        for child in node.children or []:
            _to_markdown_walk(
                child,
                inner,
                preserve_whitespace=False,
                list_depth=list_depth,
                in_link=in_link,
                html_passthrough=html_passthrough,
            )
        content = inner.finish()
        if not content:
            return
        builder.raw("**")
        builder.raw(content)
        builder.raw("**")
        return

    # Links.
    if tag == "a":
        href = ""
        if node.attrs and "href" in node.attrs and node.attrs["href"] is not None:
            href = str(node.attrs["href"])

        # Capture inner text to strip whitespace.
        inner_builder = _MarkdownBuilder()
        for child in node.children or []:
            _to_markdown_walk(
                child,
                inner_builder,
                preserve_whitespace=False,
                list_depth=list_depth,
                in_link=True,
                html_passthrough=html_passthrough,
            )
        link_text = inner_builder.finish()

        builder.raw("[")
        builder.raw(link_text)
        builder.raw("]")
        if href:
            builder.raw("(")
            builder.raw(_markdown_link_destination(href))
            builder.raw(")")
        return

    # Containers / unknown tags: recurse into children.
    next_preserve = preserve_whitespace or (tag in {"textarea", "script", "style"})
    if node.children:
        for child in node.children:
            _to_markdown_walk(
                child,
                builder,
                next_preserve,
                list_depth,
                in_link=in_link,
                html_passthrough=html_passthrough,
            )

    if isinstance(node, Element) and node.template_content:
        _to_markdown_walk(
            node.template_content,
            builder,
            next_preserve,
            list_depth,
            in_link=in_link,
            html_passthrough=html_passthrough,
        )

    # Add spacing after block containers to keep output readable.
    if tag in _MARKDOWN_BLOCK_ELEMENTS:
        if not in_link:
            builder.ensure_newlines(2)
        else:
            builder.text(" ", preserve_whitespace=False)
