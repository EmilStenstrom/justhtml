from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias, cast

from justhtml.serializer import to_html

if TYPE_CHECKING:
    from justhtml.core.types import Doctype
    from justhtml.serializer import HTMLContext


def _to_text_collect(node: NodeType, parts: list[str], strip: bool) -> None:
    # Iterative traversal avoids recursion overhead on large documents.
    stack: list[NodeType] = [node]
    while stack:
        current = stack.pop()
        name: str = current.name

        if name == "#text":
            data = current.data if isinstance(current.data, str) else None
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


def _to_text_collect_block_chunks(node: NodeType, chunks: list[list[str]], strip: bool) -> None:
    # Depth-first walk that inserts chunk boundaries for block-level elements.
    # This lets callers join chunks with a separator (e.g. "\n") without
    # introducing separators inside inline elements like <b> or <span>.
    stack: list[tuple[NodeType, int]] = [(node, 0)]  # (node, state), state: 0=enter, 1=exit
    while stack:
        current, state = stack.pop()
        name: str = current.name

        if state == 1:
            _to_text_break(chunks)
            continue

        if name == "#text":
            data = current.data if isinstance(current.data, str) else None
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
        "_metadata",
        "attrs",
        "children",
        "data",
        "name",
        "namespace",
        "parent",
    )

    name: str
    parent: Node | None
    attrs: dict[str, str | None] | None
    children: list[Any] | None
    data: str | Doctype | None
    namespace: str | None
    _metadata: list[str | int | None] | None

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
        self._metadata = None

        if name.startswith("#") or name == "!doctype":
            self.namespace = namespace
            if name in {"#comment", "#processing-instruction"} or name == "!doctype":
                self.children = None
                self.attrs = None
            else:
                self.children = []
                self.attrs = attrs if attrs is not None else {}
        else:
            self.namespace = namespace or "html"
            self.children = []
            self.attrs = attrs if attrs is not None else {}

    def append_child(self, node: NodeType) -> None:
        if self.children is not None:
            self._adopt_child(node)
            self.children.append(node)
            node.parent = self

    def _adopt_child(self, node: NodeType) -> tuple[Node | None, int | None]:
        if node is self:
            raise ValueError("Cannot insert a node into itself")

        old_parent = node.parent
        if old_parent is None:
            # Fast path for the common builder case: a freshly created detached
            # leaf node cannot participate in a cycle.
            children = node.children
            if not children:
                if not isinstance(node, Template) or node.template_content is None:
                    return None, None

        current: Node | None = self
        while current is not None:
            if current is node:
                raise ValueError("Cannot insert an ancestor into its descendant")
            current = current.parent

        old_children = old_parent.children if old_parent is not None else None
        old_index: int | None = None
        if old_children is not None:
            try:
                old_index = old_children.index(node)
            except ValueError:
                pass
            else:
                old_children.pop(old_index)
        node.parent = None
        return old_parent, old_index

    @property
    def _source_html(self) -> str | None:
        metadata = self._metadata
        return None if metadata is None else cast("str | None", metadata[0])

    @_source_html.setter
    def _source_html(self, value: str | None) -> None:
        metadata = self._metadata
        if metadata is None:
            if value is None:
                return
            metadata = [None] * 8
            self._metadata = metadata
        metadata[0] = value

    @property
    def _origin_pos(self) -> int | None:
        metadata = self._metadata
        return None if metadata is None else cast("int | None", metadata[1])

    @_origin_pos.setter
    def _origin_pos(self, value: int | None) -> None:
        metadata = self._metadata
        if metadata is None:
            if value is None:
                return
            metadata = [None] * 8
            self._metadata = metadata
        metadata[1] = value

    @property
    def _origin_line(self) -> int | None:
        metadata = self._metadata
        return None if metadata is None else cast("int | None", metadata[2])

    @_origin_line.setter
    def _origin_line(self, value: int | None) -> None:
        metadata = self._metadata
        if metadata is None:
            if value is None:
                return
            metadata = [None] * 8
            self._metadata = metadata
        metadata[2] = value

    @property
    def _origin_col(self) -> int | None:
        metadata = self._metadata
        return None if metadata is None else cast("int | None", metadata[3])

    @_origin_col.setter
    def _origin_col(self, value: int | None) -> None:
        metadata = self._metadata
        if metadata is None:
            if value is None:
                return
            metadata = [None] * 8
            self._metadata = metadata
        metadata[3] = value

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

    def remove_child(self, node: NodeType) -> None:
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

    def query(self, selector: str) -> list[QueryMatch]:
        """
        Query this subtree using a CSS selector.

        Args:
            selector: A CSS selector string

        Returns:
            A list of matching nodes

        Raises:
            ValueError: If the selector is invalid
        """
        from justhtml.selector import query  # noqa: PLC0415

        return query(self, selector)

    def query_one(self, selector: str) -> QueryMatch | None:
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
        node: NodeType = self
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
        from justhtml.serializer.markdown import to_markdown  # noqa: PLC0415

        return to_markdown(self, html_passthrough=html_passthrough)

    def insert_before(self, node: NodeType, reference_node: NodeType | None) -> None:
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

        if node is reference_node:
            return

        try:
            index = self.children.index(reference_node)
        except ValueError:
            raise ValueError("Reference node is not a child of this node") from None

        old_parent, old_index = self._adopt_child(node)
        if old_parent is self and old_index is not None and old_index < index:
            index -= 1
        self.children.insert(index, node)
        node.parent = self

    def replace_child(self, new_node: NodeType, old_node: NodeType) -> NodeType:
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

        if new_node is old_node:
            return old_node

        old_parent, old_index = self._adopt_child(new_node)
        if old_parent is self and old_index is not None and old_index < index:
            index -= 1
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
        attrs = override_attrs.copy() if override_attrs is not None else (self.attrs.copy() if self.attrs else None)
        clone = Node(
            self.name,
            attrs,
            self.data,
            self.namespace,
        )
        clone._metadata = self._metadata.copy() if self._metadata is not None else None
        if deep:
            return _clone_subtree_iterative(self)
        return clone


class Document(Node):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("#document")

    def clone_node(self, deep: bool = False, override_attrs: dict[str, str | None] | None = None) -> Document:
        _ = override_attrs
        clone = Document()
        clone._metadata = self._metadata.copy() if self._metadata is not None else None
        if deep:
            return cast("Document", _clone_subtree_iterative(self))
        return clone


class DocumentFragment(Node):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("#document-fragment")

    def clone_node(self, deep: bool = False, override_attrs: dict[str, str | None] | None = None) -> DocumentFragment:
        _ = override_attrs
        clone = DocumentFragment()
        clone._metadata = self._metadata.copy() if self._metadata is not None else None
        if deep:
            return cast("DocumentFragment", _clone_subtree_iterative(self))
        return clone


class Comment(Node):
    __slots__ = ()

    def __init__(self, data: str | None = None) -> None:
        super().__init__("#comment", data=data)

    def clone_node(self, deep: bool = False, override_attrs: dict[str, str | None] | None = None) -> Comment:
        _ = override_attrs
        _ = deep
        clone = Comment(self.data if isinstance(self.data, str) else None)
        clone._metadata = self._metadata.copy() if self._metadata is not None else None
        return clone


class ProcessingInstruction(Node):
    __slots__ = ()

    def __init__(self, data: str | None = None) -> None:
        super().__init__("#processing-instruction", data=data)

    def clone_node(
        self, deep: bool = False, override_attrs: dict[str, str | None] | None = None
    ) -> ProcessingInstruction:
        _ = override_attrs
        _ = deep
        clone = ProcessingInstruction(self.data if isinstance(self.data, str) else None)
        clone._metadata = self._metadata.copy() if self._metadata is not None else None
        return clone


class Element(Node):
    __slots__ = (
        "_end_tag_present",
        "_self_closing",
        "template_content",
    )

    template_content: Node | None
    children: list[Any]
    attrs: dict[str, str | None]
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
        self._metadata = None
        self._end_tag_present = False
        self._self_closing = False

    @property
    def _start_tag_start(self) -> int | None:
        metadata = self._metadata
        return None if metadata is None else cast("int | None", metadata[4])

    @_start_tag_start.setter
    def _start_tag_start(self, value: int | None) -> None:
        metadata = self._metadata
        if metadata is None:
            if value is None:
                return
            metadata = [None] * 8
            self._metadata = metadata
        metadata[4] = value

    @property
    def _start_tag_end(self) -> int | None:
        metadata = self._metadata
        return None if metadata is None else cast("int | None", metadata[5])

    @_start_tag_end.setter
    def _start_tag_end(self, value: int | None) -> None:
        metadata = self._metadata
        if metadata is None:
            if value is None:
                return
            metadata = [None] * 8
            self._metadata = metadata
        metadata[5] = value

    @property
    def _end_tag_start(self) -> int | None:
        metadata = self._metadata
        return None if metadata is None else cast("int | None", metadata[6])

    @_end_tag_start.setter
    def _end_tag_start(self, value: int | None) -> None:
        metadata = self._metadata
        if metadata is None:
            if value is None:
                return
            metadata = [None] * 8
            self._metadata = metadata
        metadata[6] = value

    @property
    def _end_tag_end(self) -> int | None:
        metadata = self._metadata
        return None if metadata is None else cast("int | None", metadata[7])

    @_end_tag_end.setter
    def _end_tag_end(self, value: int | None) -> None:
        metadata = self._metadata
        if metadata is None:
            if value is None:
                return
            metadata = [None] * 8
            self._metadata = metadata
        metadata[7] = value

    def clone_node(self, deep: bool = False, override_attrs: dict[str, str | None] | None = None) -> Element:
        attrs = override_attrs.copy() if override_attrs is not None else (self.attrs.copy() if self.attrs else {})
        clone = Element(self.name, attrs, self.namespace)
        clone._metadata = self._metadata.copy() if self._metadata is not None else None
        clone._end_tag_present = self._end_tag_present
        clone._self_closing = self._self_closing
        if deep:
            return cast("Element", _clone_subtree_iterative(self))
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
            self.template_content.parent = self
        else:
            self.template_content = None

    def clone_node(self, deep: bool = False, override_attrs: dict[str, str | None] | None = None) -> Template:
        attrs = override_attrs.copy() if override_attrs is not None else (self.attrs.copy() if self.attrs else {})
        clone = Template(
            self.name,
            attrs,
            None,
            self.namespace,
        )
        clone._metadata = self._metadata.copy() if self._metadata is not None else None
        clone._end_tag_present = self._end_tag_present
        clone._self_closing = self._self_closing
        if deep:
            return cast("Template", _clone_subtree_iterative(self))
        return clone


def _clone_subtree_iterative(root: Node) -> Node:
    clone_root = root.clone_node(deep=False)
    stack: list[tuple[Node, Node]] = [(root, clone_root)]

    while stack:
        source, target = stack.pop()

        if isinstance(source, Template) and isinstance(target, Template) and source.template_content is not None:
            target.template_content = source.template_content.clone_node(deep=False)
            target.template_content.parent = target
            stack.append((source.template_content, target.template_content))

        children = source.children
        if not children:
            continue

        pending: list[tuple[Node, Node]] = []
        for child in children:
            child_clone = child.clone_node(deep=False)
            target.append_child(child_clone)
            if isinstance(child, Node) and isinstance(child_clone, Node):
                pending.append((child, child_clone))

        stack.extend(reversed(pending))

    return clone_root


class Text:
    __slots__ = ("_metadata", "data", "name", "namespace", "parent")

    data: str | None
    name: str
    namespace: None
    parent: Node | None
    _metadata: list[int | None] | None

    def __init__(self, data: str | None) -> None:
        self.data = data
        self.parent = None
        self.name = "#text"
        self.namespace = None
        self._metadata = None

    @property
    def _origin_pos(self) -> int | None:
        metadata = self._metadata
        return None if metadata is None else metadata[0]

    @_origin_pos.setter
    def _origin_pos(self, value: int | None) -> None:
        metadata = self._metadata
        if metadata is None:
            if value is None:
                return
            metadata = [None, None, None]
            self._metadata = metadata
        metadata[0] = value

    @property
    def _origin_line(self) -> int | None:
        metadata = self._metadata
        return None if metadata is None else metadata[1]

    @_origin_line.setter
    def _origin_line(self, value: int | None) -> None:
        metadata = self._metadata
        if metadata is None:
            if value is None:
                return
            metadata = [None, None, None]
            self._metadata = metadata
        metadata[1] = value

    @property
    def _origin_col(self) -> int | None:
        metadata = self._metadata
        return None if metadata is None else metadata[2]

    @_origin_col.setter
    def _origin_col(self, value: int | None) -> None:
        metadata = self._metadata
        if metadata is None:
            if value is None:
                return
            metadata = [None, None, None]
            self._metadata = metadata
        metadata[2] = value

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
        from justhtml.serializer.markdown import to_markdown  # noqa: PLC0415

        return to_markdown(self, html_passthrough=html_passthrough)

    @property
    def children(self) -> list[Any]:
        """Return empty list for Text (leaf node)."""
        return []

    def has_child_nodes(self) -> bool:
        """Return False for Text."""
        return False

    def clone_node(self, deep: bool = False) -> Text:
        _ = deep
        clone = Text(self.data)
        clone._metadata = self._metadata.copy() if self._metadata is not None else None
        return clone


# Public type aliases for users who accept or return JustHTML DOM nodes.
NodeType: TypeAlias = Node | Text
QueryMatch: TypeAlias = Element | Comment
