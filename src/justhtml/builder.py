from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal, overload

from .node import Comment, Element, Node, Template, Text
from .serialize import _validate_serializable_attr_name, _validate_serializable_tag_name
from .tokens import Doctype

_SPECIAL_NODE_NAMES = {"#text", "#comment", "#document", "#document-fragment", "!doctype"}
_ALLOWED_NAMESPACES = {"html", "svg", "math"}


def text(value: str) -> Text:
    if not isinstance(value, str):
        raise TypeError("text() value must be a string")
    return Text(value)


def comment(value: str) -> Comment:
    if not isinstance(value, str):
        raise TypeError("comment() value must be a string")
    return Comment(value)


def doctype(
    name: str = "html",
    public_id: str | None = None,
    system_id: str | None = None,
    *,
    force_quirks: bool = False,
) -> Node:
    if not isinstance(name, str):
        raise TypeError("doctype() name must be a string")
    if name:
        _validate_serializable_tag_name(name)
    if public_id is not None and not isinstance(public_id, str):
        raise TypeError("doctype() public_id must be a string or None")
    if system_id is not None and not isinstance(system_id, str):
        raise TypeError("doctype() system_id must be a string or None")
    return Node(
        "!doctype",
        data=Doctype(name=name, public_id=public_id, system_id=system_id, force_quirks=force_quirks),
    )


@overload
def element(  # pragma: no cover
    name: Literal["template"],
    attrs: Mapping[str, Any] | Any | None = None,
    *children: Any,
    namespace: Literal["html"] | None = "html",
) -> Template: ...


@overload
def element(  # pragma: no cover
    name: str,
    attrs: Mapping[str, Any] | Any | None = None,
    *children: Any,
    namespace: str | None = "html",
) -> Element | Template: ...


def element(
    name: str,
    attrs: Mapping[str, Any] | Any | None = None,
    *children: Any,
    namespace: str | None = "html",
) -> Element | Template:
    if not isinstance(name, str):
        raise TypeError("element() name must be a string")

    normalized_namespace = _normalize_namespace(namespace)

    actual_children = children
    actual_attrs: Mapping[str, Any] | None
    if attrs is None or isinstance(attrs, Mapping):
        actual_attrs = attrs
    else:
        actual_attrs = None
        actual_children = (attrs, *children)

    tag_name, shorthand_attrs = _parse_element_name(name)
    if tag_name in _SPECIAL_NODE_NAMES:
        raise ValueError(f"element() cannot create special node {tag_name!r}")

    explicit_attrs = _normalize_attrs(actual_attrs)
    duplicate_attrs = set(shorthand_attrs).intersection(explicit_attrs)
    if duplicate_attrs:
        names = ", ".join(sorted(duplicate_attrs))
        raise ValueError(f"Duplicate attribute(s) across shorthand and attrs dict: {names}")

    merged_attrs = dict(shorthand_attrs)
    merged_attrs.update(explicit_attrs)

    if tag_name == "template" and normalized_namespace == "html":
        node: Element | Template = Template(tag_name, merged_attrs, namespace=normalized_namespace)
    else:
        node = Element(tag_name, merged_attrs, normalized_namespace)

    target: Any = node.template_content if isinstance(node, Template) and node.template_content is not None else node
    for child in _flatten_children(actual_children):
        target.append_child(child)

    return node


def _normalize_attrs(attrs: Mapping[str, Any] | None) -> dict[str, str | None]:
    if attrs is None:
        return {}
    if not isinstance(attrs, Mapping):
        raise TypeError("element() attrs must be a mapping")

    normalized: dict[str, str | None] = {}
    for key, value in attrs.items():
        if not isinstance(key, str):
            raise TypeError("Attribute names must be strings")
        if not key:
            raise ValueError("Attribute names must not be empty")
        _validate_serializable_attr_name(key)
        normalized[key] = None if value is None else str(value)
    return normalized


def _normalize_namespace(namespace: str | None) -> str:
    if namespace is None:
        return "html"
    if not isinstance(namespace, str):
        raise TypeError("element() namespace must be a string or None")

    normalized = namespace.lower()
    if normalized == "mathml":
        normalized = "math"

    if normalized not in _ALLOWED_NAMESPACES:
        raise ValueError("element() namespace must be one of: html, svg, mathml")

    return normalized


def _flatten_children(children: Iterable[Any]) -> list[Node | Text]:
    flattened: list[Node | Text] = []
    stack: list[Iterable[Any] | Any] = list(reversed(list(children)))

    while stack:
        current = stack.pop()

        if current is None or current is False:
            continue

        if current is True:
            raise TypeError("Boolean True is not a valid child value")

        if isinstance(current, str):
            flattened.append(Text(current))
            continue

        if isinstance(current, Text):
            flattened.append(current)
            continue

        if isinstance(current, Node):
            flattened.append(current)
            continue

        if isinstance(current, Mapping):
            raise TypeError("Mappings are not valid child values")

        if isinstance(current, (bytes, bytearray, memoryview)):
            raise TypeError("Bytes-like objects are not valid child values")

        if isinstance(current, (int, float, complex)):
            raise TypeError("Numbers are not valid child values")

        if isinstance(current, Iterable):
            stack.extend(reversed(list(current)))
            continue

        raise TypeError(f"Unsupported child value: {type(current).__name__}")

    return flattened


def _parse_element_name(value: str) -> tuple[str, dict[str, str | None]]:
    if not value:
        raise ValueError("Element name must not be empty")

    bracket_index = value.find("[")
    if bracket_index == -1:
        tag_name = value
        remainder = ""
    else:
        tag_name = value[:bracket_index]
        remainder = value[bracket_index:]

    if not tag_name:
        raise ValueError("Element name must not be empty")
    if any(ch.isspace() for ch in tag_name):
        raise ValueError("Element name must not contain whitespace")
    if tag_name not in _SPECIAL_NODE_NAMES:
        _validate_serializable_tag_name(tag_name)

    attrs: dict[str, str | None] = {}
    index = 0
    length = len(remainder)

    while index < length:
        if remainder[index] != "[":
            raise ValueError("Invalid attribute shorthand")
        index += 1

        attr_start = index
        while index < length and remainder[index] not in "=]":
            index += 1

        if index >= length:
            raise ValueError("Unclosed attribute shorthand")

        attr_name = remainder[attr_start:index]
        if not attr_name:
            raise ValueError("Attribute names must not be empty")
        _validate_serializable_attr_name(attr_name)

        if attr_name in attrs:
            raise ValueError(f"Duplicate shorthand attribute: {attr_name}")

        if remainder[index] == "]":
            attrs[attr_name] = None
            index += 1
            continue

        index += 1
        if index >= length:
            raise ValueError("Missing attribute value in shorthand")

        quote = remainder[index]
        if quote in {'"', "'"}:
            index += 1
            value_start = index
            while index < length and remainder[index] != quote:
                if remainder[index] == "\\":
                    raise ValueError("Backslash escaping is not supported in shorthand attribute values")
                index += 1
            if index >= length:
                raise ValueError("Unclosed quoted attribute value")
            attr_value = remainder[value_start:index]
            index += 1
            if index >= length or remainder[index] != "]":
                raise ValueError("Quoted attribute value must be followed by ]")
            index += 1
            attrs[attr_name] = attr_value
            continue

        value_start = index
        while index < length and remainder[index] != "]":
            if remainder[index] == "[":
                raise ValueError("Unquoted attribute values cannot contain [")
            index += 1
        if index >= length:
            raise ValueError("Unclosed attribute shorthand")
        attrs[attr_name] = remainder[value_start:index]
        index += 1

    return tag_name, attrs
