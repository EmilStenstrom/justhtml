"""Markdown rendering for JustHTML DOM nodes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from .constants import HTML_SPACE_CHARACTERS
from .node import Element

if TYPE_CHECKING:
    from .node import NodeType


def _markdown_escape_text(s: str) -> str:
    if not s:
        return ""
    # Escape Markdown syntax and HTML-significant characters so text content
    # cannot turn into raw HTML when rendered from Markdown.
    out: list[str] = []
    for ch in s:
        if ch == "&":
            out.append("&amp;")
            continue
        if ch == "<":
            out.append("&lt;")
            continue
        if ch in "\\`*_[]":
            out.append("\\")
        out.append(ch)
    return "".join(out)


def _markdown_code_span(s: str | None) -> str:
    if s is None:
        s = ""
    if not s:
        return ""
    # Inline code spans are inline Markdown constructs; line breaks can create
    # block boundaries in compliant renderers, so keep this representation
    # single-line.
    if "\n" in s or "\r" in s:
        s = " ".join(s.splitlines())
    # Use a backtick fence longer than any run of backticks inside.
    fence = _markdown_backtick_fence(s, minimum=1)
    # CommonMark requires a space if the content starts/ends with backticks.
    needs_space = s.startswith("`") or s.endswith("`")
    if needs_space:
        return f"{fence} {s} {fence}"
    return f"{fence}{s}{fence}"


def _markdown_backtick_fence(s: str | None, *, minimum: int) -> str:
    if s is None:
        s = ""
    longest = 0
    run = 0
    for ch in s:
        if ch == "`":
            run += 1
            if run > longest:
                longest = run
        else:
            run = 0
    return "`" * max(minimum, longest + 1)


def _markdown_thematic_or_setext_line(s: str, marker: str, *, minimum_markers: int) -> bool:
    stripped = s.rstrip(" \t")
    if not stripped:
        return False

    count = 0
    for ch in stripped:
        if ch in " \t":
            continue
        if ch != marker:
            return False
        count += 1
    return count >= minimum_markers


def _markdown_escape_line_start(s: str) -> tuple[str, int] | None:
    if not s:
        return None

    first = s[0]
    if first == "#":
        if len(s) == 1 or s[1] in " \t":
            return r"\#", 1
        return None

    if first == ">":
        return r"\>", 1

    if first == "-":
        if (len(s) > 1 and s[1] in " \t") or _markdown_thematic_or_setext_line(s, "-", minimum_markers=3):
            return r"\-", 1
        return None

    if first == "+":
        if len(s) > 1 and s[1] in " \t":
            return r"\+", 1
        return None

    if first == "=":
        if _markdown_thematic_or_setext_line(s, "=", minimum_markers=1):
            return r"\=", 1
        return None

    if first == "~":
        if len(s) >= 3 and s.startswith("~~~"):
            return r"\~", 1
        return None

    if first == "`":
        if len(s) >= 3 and s.startswith("```"):
            return r"\`", 1
        return None

    if first.isdigit():
        end = 1
        while end < len(s) and s[end].isdigit():
            end += 1
        if end < len(s) and s[end] in ".)" and end + 1 < len(s) and s[end + 1] in " \t":
            return f"{s[:end]}\\{s[end]}", end + 1

    return None


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
            if first not in HTML_SPACE_CHARACTERS and self._buf and self._newline_count == 0:
                self._buf.append(" ")
            self._pending_space = False

        # Adjacent raw backtick runs can merge into a different Markdown code
        # span delimiter and expose code text as ordinary Markdown/HTML.
        if self._buf and self._newline_count == 0 and self._buf[-1].endswith("`") and s.startswith("`"):
            self._buf.append(" ")

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

        index = 0
        length = len(s)
        while index < length:
            ch = s[index]
            if ch in HTML_SPACE_CHARACTERS:
                self._pending_space = True
                index += 1
                continue

            if self._pending_space:
                if self._buf and self._newline_count == 0:
                    self._buf.append(" ")
                self._pending_space = False

            if not self._buf or self._newline_count > 0:
                escaped_prefix = _markdown_escape_line_start(s[index:])
                if escaped_prefix is not None:
                    replacement, consumed = escaped_prefix
                    self._buf.append(replacement)
                    self._newline_count = 0
                    index += consumed
                    continue

            self._buf.append(ch)
            self._newline_count = 0
            index += 1

    def finish(self) -> str:
        out = "".join(self._buf)
        return out.strip(" \t\n")


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
    node: NodeType,
    builder: _MarkdownBuilder,
    preserve_whitespace: bool,
    list_depth: int,
    in_link: bool = False,
    html_passthrough: bool = False,
) -> None:
    tasks: list[Any] = [("visit", node, builder, preserve_whitespace, list_depth, in_link)]

    while tasks:
        task = tasks.pop()
        kind = task[0]

        if kind == "visit":
            current, current_builder, current_preserve, current_list_depth, current_in_link = (
                task[1],
                task[2],
                task[3],
                task[4],
                task[5],
            )
            name: str = current.name

            if name == "#text":
                if current_preserve:
                    current_builder.raw(current.data or "")
                else:
                    current_builder.text(_markdown_escape_text(current.data or ""), preserve_whitespace=False)
                continue

            if name == "br":
                if current_in_link:
                    current_builder.text(" ", preserve_whitespace=False)
                else:
                    current_builder.newline(1)
                continue

            if name == "#comment" or name == "!doctype":
                continue

            if name.startswith("#"):
                tasks.extend(
                    ("visit", child, current_builder, current_preserve, current_list_depth, current_in_link)
                    for child in reversed(current.children or [])
                )
                continue

            tag = name.lower()

            if tag == "head" or tag == "title":
                continue

            if tag == "img":
                current_builder.raw(current.to_html(indent=0, indent_size=2, pretty=False))
                continue

            if tag in {"table", "script", "style", "textarea"}:
                if not current_in_link:
                    current_builder.ensure_newlines(2 if current_builder._buf else 0)
                if tag == "table" or html_passthrough:
                    current_builder.raw(current.to_html(indent=0, indent_size=2, pretty=False))
                if not current_in_link:
                    current_builder.ensure_newlines(2)
                continue

            if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                if not current_in_link:
                    current_builder.ensure_newlines(2 if current_builder._buf else 0)
                    current_builder.raw("#" * int(tag[1]))
                    current_builder.raw(" ")
                tasks.append(("after_heading", current_builder, current_in_link))
                tasks.extend(
                    ("visit", child, current_builder, False, current_list_depth, current_in_link)
                    for child in reversed(current.children or [])
                )
                continue

            if tag == "hr":
                if not current_in_link:
                    current_builder.ensure_newlines(2 if current_builder._buf else 0)
                    current_builder.raw("---")
                    current_builder.ensure_newlines(2)
                continue

            if tag == "pre":
                code = current.to_text(separator="", strip=False)
                if current_in_link:
                    current_builder.raw(_markdown_code_span(code))
                else:
                    fence = _markdown_backtick_fence(code, minimum=3)
                    current_builder.ensure_newlines(2 if current_builder._buf else 0)
                    current_builder.raw(fence)
                    current_builder.newline(1)
                    if code:
                        current_builder.raw(code.rstrip("\n"))
                        current_builder.newline(1)
                    current_builder.raw(fence)
                    current_builder.ensure_newlines(2)
                continue

            if tag == "code" and not current_preserve:
                current_builder.raw(_markdown_code_span(current.to_text(separator="", strip=False)))
                continue

            if tag == "p":
                if not current_in_link:
                    current_builder.ensure_newlines(2 if current_builder._buf else 0)
                tasks.append(("after_paragraph", current_builder, current_in_link))
                tasks.extend(
                    ("visit", child, current_builder, False, current_list_depth, current_in_link)
                    for child in reversed(current.children or [])
                )
                continue

            if tag == "blockquote":
                if current_in_link:
                    tasks.extend(
                        ("visit", child, current_builder, False, current_list_depth, current_in_link)
                        for child in reversed(current.children or [])
                    )
                else:
                    inner_builder = _MarkdownBuilder()
                    tasks.append(("after_blockquote", current_builder, inner_builder))
                    tasks.extend(
                        ("visit", child, inner_builder, False, current_list_depth, current_in_link)
                        for child in reversed(current.children or [])
                    )
                continue

            if tag in {"ul", "ol"}:
                items = [child for child in current.children or () if child.name.lower() == "li"]
                if current_in_link:
                    tasks.extend(
                        ("flatten_list_item", child, current_builder, current_list_depth, html_passthrough)
                        for child in reversed(items)
                    )
                else:
                    current_builder.ensure_newlines(2 if current_builder._buf else 0)
                    ordered = tag == "ol"
                    tasks.append(("after_list", current_builder))
                    for index, child in reversed(list(enumerate(items, start=1))):
                        tasks.append(("visit_list_item", child, current_builder, current_list_depth, ordered, index))
                        if index != 1:
                            tasks.append(("list_separator", current_builder))
                continue

            if tag in {"em", "i"}:
                inner_builder = _MarkdownBuilder()
                tasks.append(("after_marker", current_builder, inner_builder, "*"))
                tasks.extend(
                    ("visit", child, inner_builder, False, current_list_depth, current_in_link)
                    for child in reversed(current.children or [])
                )
                continue

            if tag in {"strong", "b"}:
                inner_builder = _MarkdownBuilder()
                tasks.append(("after_marker", current_builder, inner_builder, "**"))
                tasks.extend(
                    ("visit", child, inner_builder, False, current_list_depth, current_in_link)
                    for child in reversed(current.children or [])
                )
                continue

            if tag == "a":
                href = ""
                if current.attrs and "href" in current.attrs and current.attrs["href"] is not None:
                    href = str(current.attrs["href"])
                inner_builder = _MarkdownBuilder()
                tasks.append(("after_link", current_builder, inner_builder, href))
                tasks.extend(
                    ("visit", child, inner_builder, False, current_list_depth, True)
                    for child in reversed(current.children or [])
                )
                continue

            next_preserve = current_preserve or (tag in {"textarea", "script", "style"})
            if tag in _MARKDOWN_BLOCK_ELEMENTS:
                if not current_in_link:
                    current_builder.ensure_newlines(2 if current_builder._buf else 0)
                tasks.append(("after_block_container", current_builder, current_in_link))
            if isinstance(current, Element) and current.template_content:
                tasks.append(
                    (
                        "visit",
                        current.template_content,
                        current_builder,
                        next_preserve,
                        current_list_depth,
                        current_in_link,
                    )
                )
            tasks.extend(
                ("visit", child, current_builder, next_preserve, current_list_depth, current_in_link)
                for child in reversed(current.children or [])
            )
            continue

        if kind == "after_heading":
            if not task[2]:
                task[1].ensure_newlines(2)
            continue

        if kind == "after_paragraph":
            if task[2]:
                task[1].text(" ", preserve_whitespace=False)
            else:
                task[1].ensure_newlines(2)
            continue

        if kind == "after_blockquote":
            parent_builder, inner_builder = task[1], task[2]
            parent_builder.ensure_newlines(2 if parent_builder._buf else 0)
            text = inner_builder.finish()
            if text:
                for index, line in enumerate(text.split("\n")):
                    if index:
                        parent_builder.newline(1)
                    parent_builder.raw("> ")
                    parent_builder.raw(line)
            parent_builder.ensure_newlines(2)
            continue

        if kind == "after_list":
            task[1].ensure_newlines(2)
            continue

        if kind == "list_separator":
            task[1].newline(1)
            continue

        if kind == "visit_list_item":
            li_node, current_builder, current_list_depth, ordered, index = task[1], task[2], task[3], task[4], task[5]
            current_builder.raw("  " * current_list_depth)
            current_builder.raw(f"{index}. " if ordered else "- ")
            tasks.extend(
                ("visit", child, current_builder, False, current_list_depth + 1, False)
                for child in reversed(li_node.children or [])
            )
            continue

        if kind == "flatten_list_item":
            li_node, current_builder, current_list_depth = task[1], task[2], task[3]
            current_builder.raw(" ")
            tasks.extend(
                ("visit", child, current_builder, False, current_list_depth + 1, True)
                for child in reversed(li_node.children or [])
            )
            continue

        if kind == "after_marker":
            parent_builder, inner_builder, marker = task[1], task[2], task[3]
            content = inner_builder.finish()
            if content:
                if "\n" in content:
                    parent_builder.ensure_newlines(2 if parent_builder._buf else 0)
                    parent_builder.raw(content)
                    continue
                parent_builder.raw(marker)
                parent_builder.raw(content)
                parent_builder.raw(marker)
            continue

        if kind == "after_link":
            parent_builder, inner_builder, href = task[1], task[2], task[3]
            link_text = inner_builder.finish()
            parent_builder.raw("[")
            parent_builder.raw(link_text)
            parent_builder.raw("]")
            if href:
                parent_builder.raw("(")
                parent_builder.raw(_markdown_link_destination(href))
                parent_builder.raw(")")
            continue

        if kind != "after_block_container":  # pragma: no cover
            raise RuntimeError(f"Unknown markdown task kind: {kind}")
        if task[2]:
            task[1].text(" ", preserve_whitespace=False)
        else:
            task[1].ensure_newlines(2)


def to_markdown(node: NodeType, *, html_passthrough: bool = False) -> str:
    builder = _MarkdownBuilder()
    _to_markdown_walk(
        node,
        builder,
        preserve_whitespace=False,
        list_depth=0,
        html_passthrough=html_passthrough,
    )
    return builder.finish()
