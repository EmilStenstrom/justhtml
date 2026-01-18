# CSS Selector implementation for JustHTML
# Supports a subset of CSS selectors for querying the DOM

from __future__ import annotations

from functools import lru_cache
from typing import Any


class SelectorError(ValueError):
    """Raised when a CSS selector is invalid."""


# Token types for the CSS selector lexer
class TokenType:
    TAG: str = "TAG"  # div, span, etc.
    ID: str = "ID"  # #foo
    CLASS: str = "CLASS"  # .bar
    UNIVERSAL: str = "UNIVERSAL"  # *
    ATTR_START: str = "ATTR_START"  # [
    ATTR_END: str = "ATTR_END"  # ]
    ATTR_OP: str = "ATTR_OP"  # =, ~=, |=, ^=, $=, *=
    STRING: str = "STRING"  # "value" or 'value' or unquoted
    COMBINATOR: str = "COMBINATOR"  # >, +, ~, or whitespace (descendant)
    COMMA: str = "COMMA"  # ,
    COLON: str = "COLON"  # :
    PAREN_OPEN: str = "PAREN_OPEN"  # (
    PAREN_CLOSE: str = "PAREN_CLOSE"  # )
    EOF: str = "EOF"


class Token:
    __slots__ = ("type", "value")

    type: str
    value: str | None

    def __init__(self, token_type: str, value: str | None = None) -> None:
        self.type = token_type
        self.value = value

    def __repr__(self) -> str:
        return f"Token({self.type}, {self.value!r})"


class SelectorTokenizer:
    """Tokenizes a CSS selector string into tokens."""

    __slots__ = ("length", "pos", "selector")

    selector: str
    pos: int
    length: int

    def __init__(self, selector: str) -> None:
        self.selector = selector
        self.pos = 0
        self.length = len(selector)

    def _peek(self, offset: int = 0) -> str:
        pos = self.pos + offset
        if pos < self.length:
            return self.selector[pos]
        return ""

    def _advance(self) -> str:
        ch = self._peek()
        self.pos += 1
        return ch

    def _skip_whitespace(self) -> None:
        while self.pos < self.length and self.selector[self.pos] in " \t\n\r\f":
            self.pos += 1

    def _is_name_start(self, ch: str) -> bool:
        # CSS identifier start: letter, underscore, or non-ASCII
        return ch.isalpha() or ch == "_" or ch == "-" or ord(ch) > 127

    def _is_name_char(self, ch: str) -> bool:
        # CSS identifier continuation: name-start or digit
        return self._is_name_start(ch) or ch.isdigit()

    def _read_name(self) -> str:
        start = self.pos
        while self.pos < self.length and self._is_name_char(self.selector[self.pos]):
            self.pos += 1
        return self.selector[start : self.pos]

    def _read_string(self, quote: str) -> str:
        # Skip opening quote
        self.pos += 1
        start = self.pos
        parts: list[str] = []

        while self.pos < self.length:
            ch = self.selector[self.pos]
            if ch == quote:
                # Append any remaining text before the closing quote
                if self.pos > start:
                    parts.append(self.selector[start : self.pos])
                self.pos += 1
                return "".join(parts)
            if ch == "\\":
                # Append text before the backslash
                if self.pos > start:
                    parts.append(self.selector[start : self.pos])
                self.pos += 1
                if self.pos < self.length:
                    # Append the escaped character
                    parts.append(self.selector[self.pos])
                    self.pos += 1
                    start = self.pos
                else:
                    start = self.pos
            else:
                self.pos += 1

        raise SelectorError(f"Unterminated string in selector: {self.selector!r}")

    def _read_unquoted_attr_value(self) -> str:
        # Read an unquoted attribute value (CSS identifier)
        start = self.pos
        while self.pos < self.length:
            ch = self.selector[self.pos]
            if ch in " \t\n\r\f]":
                break
            self.pos += 1
        return self.selector[start : self.pos]

    def tokenize(self) -> list[Token]:
        tokens: list[Token] = []
        pending_whitespace = False

        while self.pos < self.length:
            ch = self.selector[self.pos]

            # Skip whitespace but remember it for combinator detection
            if ch in " \t\n\r\f":
                pending_whitespace = True
                self._skip_whitespace()
                continue

            # Handle combinators: >, +, ~
            if ch in ">+~":
                pending_whitespace = False
                self.pos += 1
                self._skip_whitespace()
                tokens.append(Token(TokenType.COMBINATOR, ch))
                continue

            # If we had whitespace and this isn't a combinator symbol or comma,
            # it's a descendant combinator. Note: combinators and commas consume
            # trailing whitespace, so pending_whitespace is always False after them.
            if pending_whitespace and tokens and ch not in ",":
                tokens.append(Token(TokenType.COMBINATOR, " "))
            pending_whitespace = False

            # Universal selector
            if ch == "*":
                self.pos += 1
                tokens.append(Token(TokenType.UNIVERSAL))
                continue

            # ID selector
            if ch == "#":
                self.pos += 1
                name = self._read_name()
                if not name:
                    raise SelectorError(f"Expected identifier after # at position {self.pos}")
                tokens.append(Token(TokenType.ID, name))
                continue

            # Class selector
            if ch == ".":
                self.pos += 1
                name = self._read_name()
                if not name:
                    raise SelectorError(f"Expected identifier after . at position {self.pos}")
                tokens.append(Token(TokenType.CLASS, name))
                continue

            # Attribute selector
            if ch == "[":
                self.pos += 1
                tokens.append(Token(TokenType.ATTR_START))
                self._skip_whitespace()

                # Read attribute name
                attr_name = self._read_name()
                if not attr_name:
                    raise SelectorError(f"Expected attribute name at position {self.pos}")
                tokens.append(Token(TokenType.TAG, attr_name))  # Reuse TAG for attr name
                self._skip_whitespace()

                # Check for operator
                ch2 = self._peek()
                if ch2 == "]":
                    self.pos += 1
                    tokens.append(Token(TokenType.ATTR_END))
                    continue

                # Read operator
                if ch2 == "=":
                    self.pos += 1
                    tokens.append(Token(TokenType.ATTR_OP, "="))
                elif ch2 in "~|^$*":
                    op_char = ch2
                    self.pos += 1
                    if self._peek() != "=":
                        raise SelectorError(f"Expected = after {op_char} at position {self.pos}")
                    self.pos += 1
                    tokens.append(Token(TokenType.ATTR_OP, op_char + "="))
                else:
                    raise SelectorError(f"Unexpected character in attribute selector: {ch2!r}")

                self._skip_whitespace()

                # Read value
                ch3 = self._peek()
                if ch3 == '"' or ch3 == "'":
                    value = self._read_string(ch3)
                else:
                    value = self._read_unquoted_attr_value()
                tokens.append(Token(TokenType.STRING, value))

                self._skip_whitespace()
                if self._peek() != "]":
                    raise SelectorError(f"Expected ] at position {self.pos}")
                self.pos += 1
                tokens.append(Token(TokenType.ATTR_END))
                continue

            # Comma (selector grouping)
            if ch == ",":
                self.pos += 1
                self._skip_whitespace()
                tokens.append(Token(TokenType.COMMA))
                continue

            # Pseudo-class
            if ch == ":":
                self.pos += 1
                tokens.append(Token(TokenType.COLON))
                # Read pseudo-class name
                name = self._read_name()
                if not name:
                    raise SelectorError(f"Expected pseudo-class name after : at position {self.pos}")
                tokens.append(Token(TokenType.TAG, name))

                # Check for functional pseudo-class
                if self._peek() == "(":
                    self.pos += 1
                    tokens.append(Token(TokenType.PAREN_OPEN))
                    self._skip_whitespace()

                    # Special handling for :not() - can contain a selector
                    # For :nth-child() - read the expression
                    paren_depth = 1
                    arg_start = self.pos
                    while self.pos < self.length and paren_depth > 0:
                        c = self.selector[self.pos]
                        if c == "(":
                            paren_depth += 1
                        elif c == ")":
                            paren_depth -= 1
                        if paren_depth > 0:
                            self.pos += 1

                    arg = self.selector[arg_start : self.pos].strip()
                    if arg:
                        tokens.append(Token(TokenType.STRING, arg))

                    if self._peek() != ")":
                        raise SelectorError(f"Expected ) at position {self.pos}")
                    self.pos += 1
                    tokens.append(Token(TokenType.PAREN_CLOSE))
                continue

            # Tag name
            if self._is_name_start(ch):
                name = self._read_name()
                tokens.append(Token(TokenType.TAG, name.lower()))  # Tags are case-insensitive
                continue

            raise SelectorError(f"Unexpected character {ch!r} at position {self.pos}")

        tokens.append(Token(TokenType.EOF))
        return tokens


# AST Node types for parsed selectors


class SimpleSelector:
    """A single simple selector (tag, id, class, attribute, or pseudo-class)."""

    __slots__ = ("arg", "name", "operator", "type", "value")

    TYPE_TAG: str = "tag"
    TYPE_ID: str = "id"
    TYPE_CLASS: str = "class"
    TYPE_UNIVERSAL: str = "universal"
    TYPE_ATTR: str = "attr"
    TYPE_PSEUDO: str = "pseudo"

    type: str
    name: str | None
    operator: str | None
    value: str | None
    arg: str | None

    def __init__(
        self,
        selector_type: str,
        name: str | None = None,
        operator: str | None = None,
        value: str | None = None,
        arg: str | None = None,
    ) -> None:
        self.type = selector_type
        self.name = name
        self.operator = operator
        self.value = value
        self.arg = arg  # For :not() and :nth-child()

    def __repr__(self) -> str:
        parts = [f"SimpleSelector({self.type!r}"]
        if self.name:
            parts.append(f", name={self.name!r}")
        if self.operator:
            parts.append(f", op={self.operator!r}")
        if self.value is not None:
            parts.append(f", value={self.value!r}")
        if self.arg is not None:
            parts.append(f", arg={self.arg!r}")
        parts.append(")")
        return "".join(parts)


class CompoundSelector:
    """A sequence of simple selectors (e.g., div.foo#bar)."""

    __slots__ = ("selectors",)

    selectors: list[SimpleSelector]

    def __init__(self, selectors: list[SimpleSelector] | None = None) -> None:
        self.selectors = selectors or []

    def __repr__(self) -> str:
        return f"CompoundSelector({self.selectors!r})"


class ComplexSelector:
    """A chain of compound selectors with combinators."""

    __slots__ = ("parts",)

    parts: list[tuple[str | None, CompoundSelector]]

    def __init__(self) -> None:
        # List of (combinator, compound_selector) tuples
        # First item has combinator=None
        self.parts = []

    def __repr__(self) -> str:
        return f"ComplexSelector({self.parts!r})"


class SelectorList:
    """A comma-separated list of complex selectors."""

    __slots__ = ("selectors",)

    selectors: list[ComplexSelector]

    def __init__(self, selectors: list[ComplexSelector] | None = None) -> None:
        self.selectors = selectors or []

    def __repr__(self) -> str:
        return f"SelectorList({self.selectors!r})"


# Type alias for parsed selectors
ParsedSelector = ComplexSelector | SelectorList


class SelectorParser:
    """Parses a list of tokens into a selector AST."""

    __slots__ = ("pos", "tokens")

    tokens: list[Token]
    pos: int

    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> Token:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return Token(TokenType.EOF)

    def _advance(self) -> Token:
        token = self._peek()
        self.pos += 1
        return token

    def _expect(self, token_type: str) -> Token:
        token = self._peek()
        if token.type != token_type:
            raise SelectorError(f"Expected {token_type}, got {token.type}")
        return self._advance()

    def parse(self) -> ParsedSelector:
        """Parse a complete selector (possibly comma-separated list)."""
        selectors: list[ComplexSelector] = []
        # parse_selector() validates non-empty input, so first selector always exists
        first = self._parse_complex_selector()
        if first is None:  # pragma: no cover
            raise SelectorError("Empty selector")
        selectors.append(first)

        while self._peek().type == TokenType.COMMA:
            self._advance()  # consume comma
            selector = self._parse_complex_selector()
            if selector:
                selectors.append(selector)

        if self._peek().type != TokenType.EOF:
            raise SelectorError(f"Unexpected token: {self._peek()}")

        if len(selectors) == 1:
            return selectors[0]
        return SelectorList(selectors)

    def _parse_complex_selector(self) -> ComplexSelector | None:
        """Parse a complex selector (compound selectors with combinators)."""
        complex_sel = ComplexSelector()

        # First compound selector (no combinator)
        compound = self._parse_compound_selector()
        if not compound:
            return None
        complex_sel.parts.append((None, compound))

        # Parse combinator + compound selector pairs
        while self._peek().type == TokenType.COMBINATOR:
            combinator = self._advance().value
            compound = self._parse_compound_selector()
            if not compound:
                raise SelectorError("Expected selector after combinator")
            complex_sel.parts.append((combinator, compound))

        return complex_sel

    def _parse_compound_selector(self) -> CompoundSelector | None:
        """Parse a compound selector (sequence of simple selectors)."""
        simple_selectors: list[SimpleSelector] = []

        while True:
            token = self._peek()

            if token.type == TokenType.TAG:
                self._advance()
                simple_selectors.append(SimpleSelector(SimpleSelector.TYPE_TAG, name=token.value))

            elif token.type == TokenType.UNIVERSAL:
                self._advance()
                simple_selectors.append(SimpleSelector(SimpleSelector.TYPE_UNIVERSAL))

            elif token.type == TokenType.ID:
                self._advance()
                simple_selectors.append(SimpleSelector(SimpleSelector.TYPE_ID, name=token.value))

            elif token.type == TokenType.CLASS:
                self._advance()
                simple_selectors.append(SimpleSelector(SimpleSelector.TYPE_CLASS, name=token.value))

            elif token.type == TokenType.ATTR_START:
                simple_selectors.append(self._parse_attribute_selector())

            elif token.type == TokenType.COLON:
                simple_selectors.append(self._parse_pseudo_selector())

            else:
                break

        if not simple_selectors:
            return None
        return CompoundSelector(simple_selectors)

    def _parse_attribute_selector(self) -> SimpleSelector:
        """Parse an attribute selector [attr], [attr=value], etc."""
        self._expect(TokenType.ATTR_START)

        attr_name = self._expect(TokenType.TAG).value

        token = self._peek()
        if token.type == TokenType.ATTR_END:
            self._advance()
            return SimpleSelector(SimpleSelector.TYPE_ATTR, name=attr_name)

        operator = self._expect(TokenType.ATTR_OP).value
        value = self._expect(TokenType.STRING).value
        self._expect(TokenType.ATTR_END)

        return SimpleSelector(SimpleSelector.TYPE_ATTR, name=attr_name, operator=operator, value=value)

    def _parse_pseudo_selector(self) -> SimpleSelector:
        """Parse a pseudo-class selector like :first-child or :not(selector)."""
        self._expect(TokenType.COLON)
        name = self._expect(TokenType.TAG).value

        # Functional pseudo-class
        if self._peek().type == TokenType.PAREN_OPEN:
            self._advance()
            arg: str | None = None
            if self._peek().type == TokenType.STRING:
                arg = self._advance().value
            self._expect(TokenType.PAREN_CLOSE)
            return SimpleSelector(SimpleSelector.TYPE_PSEUDO, name=name, arg=arg)

        return SimpleSelector(SimpleSelector.TYPE_PSEUDO, name=name)


class SelectorMatcher:
    """Matches selectors against DOM nodes."""

    __slots__ = ()

    def _unquote_pseudo_arg(self, arg: str) -> str:
        arg = arg.strip()
        if len(arg) >= 2 and arg[0] == arg[-1] and arg[0] in ('"', "'"):
            quote = arg[0]
            # Minimal unescaping for common cases like :contains("click me")
            return arg[1:-1].replace("\\" + quote, quote).replace("\\\\", "\\")
        return arg

    def matches(self, node: Any, selector: ParsedSelector | CompoundSelector | SimpleSelector) -> bool:
        """Check if a node matches a parsed selector."""
        if isinstance(selector, SelectorList):
            return any(self.matches(node, sel) for sel in selector.selectors)
        if isinstance(selector, ComplexSelector):
            return self._matches_complex(node, selector)
        if isinstance(selector, CompoundSelector):
            return self._matches_compound(node, selector)
        if isinstance(selector, SimpleSelector):
            return self._matches_simple(node, selector)
        return False

    def _matches_complex(self, node: Any, selector: ComplexSelector) -> bool:
        """Match a complex selector (with combinators)."""
        # Work backwards from the rightmost compound selector
        parts = selector.parts
        if not parts:
            return False

        # Start with the rightmost part
        combinator, compound = parts[-1]
        if not self._matches_compound(node, compound):
            return False

        # Work backwards through the chain
        current = node
        for i in range(len(parts) - 2, -1, -1):
            combinator, compound = parts[i + 1]
            prev_compound = parts[i][1]

            if combinator == " ":  # Descendant
                found = False
                ancestor = current.parent
                while ancestor:
                    if self._matches_compound(ancestor, prev_compound):
                        current = ancestor
                        found = True
                        break
                    ancestor = ancestor.parent
                if not found:
                    return False

            elif combinator == ">":  # Child
                parent = current.parent
                if not parent or not self._matches_compound(parent, prev_compound):
                    return False
                current = parent

            elif combinator == "+":  # Adjacent sibling
                sibling = self._get_previous_sibling(current)
                if not sibling or not self._matches_compound(sibling, prev_compound):
                    return False
                current = sibling

            else:  # combinator == "~" - General sibling
                found = False
                sibling = self._get_previous_sibling(current)
                while sibling:
                    if self._matches_compound(sibling, prev_compound):
                        current = sibling
                        found = True
                        break
                    sibling = self._get_previous_sibling(sibling)
                if not found:
                    return False

        return True

    def _matches_compound(self, node: Any, compound: CompoundSelector) -> bool:
        """Match a compound selector (all simple selectors must match)."""
        return all(self._matches_simple(node, simple) for simple in compound.selectors)

    def _matches_simple(self, node: Any, selector: SimpleSelector) -> bool:
        """Match a simple selector against a node."""
        # Text nodes and other non-element nodes don't match element selectors
        if not hasattr(node, "name") or node.name.startswith("#"):
            return False

        sel_type = selector.type

        if sel_type == SimpleSelector.TYPE_UNIVERSAL:
            return True

        if sel_type == SimpleSelector.TYPE_TAG:
            # HTML tag names are case-insensitive
            return bool(node.name.lower() == (selector.name.lower() if selector.name else ""))

        if sel_type == SimpleSelector.TYPE_ID:
            node_id = node.attrs.get("id", "") if node.attrs else ""
            return node_id == selector.name

        if sel_type == SimpleSelector.TYPE_CLASS:
            class_attr = node.attrs.get("class", "") if node.attrs else ""
            classes = class_attr.split() if class_attr else []
            return selector.name in classes

        if sel_type == SimpleSelector.TYPE_ATTR:
            return self._matches_attribute(node, selector)

        if sel_type == SimpleSelector.TYPE_PSEUDO:
            return self._matches_pseudo(node, selector)

        return False

    def _matches_attribute(self, node: Any, selector: SimpleSelector) -> bool:
        """Match an attribute selector."""
        attrs = node.attrs or {}
        attr_name = (selector.name or "").lower()  # Attribute names are case-insensitive in HTML

        # Check if attribute exists (for any case)
        attr_value: str | None = None
        for name, value in attrs.items():
            if name.lower() == attr_name:
                # Attributes can be boolean (represented as None in JustHTML).
                # For selector matching, presence should still count.
                attr_value = "" if value is None else str(value)
                break

        if attr_value is None:
            return False

        # Presence check only
        if selector.operator is None:
            return True

        value = selector.value or ""
        op = selector.operator

        if op == "=":
            return attr_value == value

        if op == "~=":
            # Space-separated word match
            words = attr_value.split() if attr_value else []
            return value in words

        if op == "|=":
            # Hyphen-separated prefix match (e.g., lang="en" matches lang|="en-US")
            return attr_value == value or attr_value.startswith(value + "-")

        if op == "^=":
            # Starts with
            return attr_value.startswith(value) if value else False

        if op == "$=":
            # Ends with
            return attr_value.endswith(value) if value else False

        if op == "*=":
            # Contains
            return value in attr_value if value else False

        return False

    def _matches_pseudo(self, node: Any, selector: SimpleSelector) -> bool:
        """Match a pseudo-class selector."""
        name = (selector.name or "").lower()

        if name == "first-child":
            return self._is_first_child(node)

        if name == "last-child":
            return self._is_last_child(node)

        if name == "nth-child":
            return self._matches_nth_child(node, selector.arg)

        if name == "not":
            if not selector.arg:
                return True
            # Parse the inner selector
            inner = parse_selector(selector.arg)
            return not self.matches(node, inner)

        if name == "only-child":
            return self._is_first_child(node) and self._is_last_child(node)

        if name == "empty":
            if not node.has_child_nodes():
                return True
            # Check if all children are empty text nodes
            for child in node.children:
                if hasattr(child, "name"):
                    if child.name == "#text":
                        if child.data and child.data.strip():
                            return False
                    elif not child.name.startswith("#"):
                        return False
            return True

        if name == "root":
            # Root is the html element (or document root's first element child)
            parent = node.parent
            if parent and hasattr(parent, "name"):
                return parent.name in ("#document", "#document-fragment")
            return False

        if name == "contains":
            if selector.arg is None:
                raise SelectorError(":contains() requires a string argument")
            needle = self._unquote_pseudo_arg(selector.arg)
            if needle == "":
                return True
            # Non-standard (jQuery-style) pseudo-class: match elements whose descendant
            # text contains the substring. We use `to_text()` to approximate textContent.
            haystack: str = node.to_text(separator=" ", strip=True)
            return needle in haystack

        if name == "first-of-type":
            return self._is_first_of_type(node)

        if name == "last-of-type":
            return self._is_last_of_type(node)

        if name == "nth-of-type":
            return self._matches_nth_of_type(node, selector.arg)

        if name == "only-of-type":
            return self._is_first_of_type(node) and self._is_last_of_type(node)

        # Unknown pseudo-class - don't match
        raise SelectorError(f"Unsupported pseudo-class: :{name}")

    def _get_element_children(self, parent: Any) -> list[Any]:
        """Get only element children (exclude text, comments, etc.)."""
        if not parent or not parent.has_child_nodes():
            return []
        return [c for c in parent.children if not c.name.startswith("#")]

    def _get_previous_sibling(self, node: Any) -> Any | None:
        """Get the previous element sibling. Returns None if node is first or not found."""
        parent = node.parent
        if not parent:
            return None

        prev: Any | None = None
        for child in parent.children:
            if child is node:
                return prev
            if not child.name.startswith("#"):
                prev = child
        return None  # node not in parent.children (detached)

    def _is_first_child(self, node: Any) -> bool:
        """Check if node is the first element child of its parent."""
        parent = node.parent
        if not parent:
            return False
        elements = self._get_element_children(parent)
        return bool(elements) and elements[0] is node

    def _is_last_child(self, node: Any) -> bool:
        """Check if node is the last element child of its parent."""
        parent = node.parent
        if not parent:
            return False
        elements = self._get_element_children(parent)
        return bool(elements) and elements[-1] is node

    def _is_first_of_type(self, node: Any) -> bool:
        """Check if node is the first sibling of its type."""
        parent = node.parent
        if not parent:
            return False
        node_name = node.name.lower()
        for child in self._get_element_children(parent):
            if child.name.lower() == node_name:
                return child is node
        return False

    def _is_last_of_type(self, node: Any) -> bool:
        """Check if node is the last sibling of its type."""
        parent = node.parent
        if not parent:
            return False
        node_name = node.name.lower()
        last_of_type: Any | None = None
        for child in self._get_element_children(parent):
            if child.name.lower() == node_name:
                last_of_type = child
        return last_of_type is node

    def _parse_nth_expression(self, expr: str | None) -> tuple[int, int] | None:
        """Parse an nth-child expression like '2n+1', 'odd', 'even', '3'."""
        if not expr:
            return None

        expr = expr.strip().lower()

        if expr == "odd":
            return (2, 1)  # 2n+1
        if expr == "even":
            return (2, 0)  # 2n

        # Parse An+B syntax
        # Handle formats: n, 2n, 2n+1, -n+2, 3, etc.
        a = 0
        b = 0

        # Remove all spaces
        expr = expr.replace(" ", "")

        if "n" in expr:
            parts = expr.split("n")
            a_part = parts[0]
            b_part = parts[1] if len(parts) > 1 else ""

            if a_part == "" or a_part == "+":
                a = 1
            elif a_part == "-":
                a = -1
            else:
                try:
                    a = int(a_part)
                except ValueError:
                    return None

            if b_part:
                try:
                    b = int(b_part)
                except ValueError:
                    return None
        else:
            # Just a number
            try:
                b = int(expr)
            except ValueError:
                return None

        return (a, b)

    def _matches_nth(self, index: int, a: int, b: int) -> bool:
        """Check if 1-based index matches An+B formula."""
        if a == 0:
            return index == b
        # Solve: index = a*n + b for non-negative integer n
        # n = (index - b) / a
        diff = index - b
        if a > 0:
            return diff >= 0 and diff % a == 0
        # a < 0: need diff <= 0 and diff divisible by abs(a)
        return diff <= 0 and diff % a == 0

    def _matches_nth_child(self, node: Any, arg: str | None) -> bool:
        """Match :nth-child(An+B)."""
        parent = node.parent
        if not parent:
            return False

        parsed = self._parse_nth_expression(arg)
        if parsed is None:
            return False
        a, b = parsed

        elements = self._get_element_children(parent)
        for i, child in enumerate(elements):
            if child is node:
                return self._matches_nth(i + 1, a, b)
        return False

    def _matches_nth_of_type(self, node: Any, arg: str | None) -> bool:
        """Match :nth-of-type(An+B)."""
        parent = node.parent
        if not parent:
            return False

        parsed = self._parse_nth_expression(arg)
        if parsed is None:
            return False
        a, b = parsed

        node_name = node.name.lower()
        elements = self._get_element_children(parent)
        type_index = 0
        for child in elements:
            if child.name.lower() == node_name:
                type_index += 1
                if child is node:
                    return self._matches_nth(type_index, a, b)
        return False


def parse_selector(selector_string: str) -> ParsedSelector:
    """Parse a CSS selector string into an AST."""
    if not selector_string or not selector_string.strip():
        raise SelectorError("Empty selector")

    return _parse_selector_cached(selector_string.strip())


@lru_cache(maxsize=512)
def _parse_selector_cached(selector_string: str) -> ParsedSelector:
    tokenizer = SelectorTokenizer(selector_string)
    tokens = tokenizer.tokenize()
    parser = SelectorParser(tokens)
    return parser.parse()


# Global matcher instance
_matcher: SelectorMatcher = SelectorMatcher()


def _is_simple_tag_selector(selector: str) -> bool:
    if not selector:
        return False
    ch0 = selector[0]
    if not (ch0.isalpha() or ch0 == "_" or ch0 == "-" or ord(ch0) > 127):
        return False
    for ch in selector[1:]:
        if ch.isalnum() or ch == "_" or ch == "-" or ord(ch) > 127:
            continue
        return False
    return True


def _query_descendants_tag(node: Any, tag_lower: str, results: list[Any]) -> None:
    results_append = results.append

    stack: list[Any] = []

    root_children = node.children
    if root_children:
        stack.extend(reversed(root_children))

    if node.name == "template" and node.namespace == "html":
        template_content = node.template_content
        if template_content:
            stack.append(template_content)

    while stack:
        current = stack.pop()

        name = current.name
        if not name.startswith("#"):
            if name == tag_lower or name.lower() == tag_lower:
                results_append(current)

        children = current.children
        if children:
            stack.extend(reversed(children))

        if name == "template" and current.namespace == "html":
            template_content = current.template_content
            if template_content:
                stack.append(template_content)


def query(root: Any, selector_string: str) -> list[Any]:
    """
    Query the DOM tree starting from root, returning all matching elements.

    Searches descendants of root, not including root itself (matching browser
    behavior for querySelectorAll).

    Args:
        root: The root node to search from
        selector_string: A CSS selector string

    Returns:
        A list of matching nodes
    """
    selector_string = selector_string.strip()
    if not selector_string:
        raise SelectorError("Empty selector")

    results: list[Any] = []

    if _is_simple_tag_selector(selector_string):
        _query_descendants_tag(root, selector_string.lower(), results)
        return results

    selector = _parse_selector_cached(selector_string)
    _query_descendants(root, selector, results)
    return results


def _query_descendants(node: Any, selector: ParsedSelector, results: list[Any]) -> None:
    """Search for matching nodes in descendants."""
    matcher_matches = _matcher.matches
    results_append = results.append

    # querySelectorAll searches descendants of root, not including root itself.
    stack: list[Any] = []

    root_children = node.children
    if root_children:
        stack.extend(reversed(root_children))

    if node.name == "template" and node.namespace == "html":
        template_content = node.template_content
        if template_content:
            stack.append(template_content)

    while stack:
        current = stack.pop()

        name = current.name
        if not name.startswith("#") and matcher_matches(current, selector):
            results_append(current)

        children = current.children
        if children:
            stack.extend(reversed(children))

        if name == "template" and current.namespace == "html":
            template_content = current.template_content
            if template_content:
                stack.append(template_content)


def matches(node: Any, selector_string: str) -> bool:
    """
    Check if a node matches a CSS selector.

    Args:
        node: The node to check
        selector_string: A CSS selector string

    Returns:
        True if the node matches, False otherwise
    """
    selector = parse_selector(selector_string)
    return _matcher.matches(node, selector)
