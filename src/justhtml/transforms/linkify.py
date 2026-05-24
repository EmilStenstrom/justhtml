"""Compiled Linkify transform support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from justhtml.dom import Element, Text

from .linkify_core import LinkifyConfig, find_links_with_config

if TYPE_CHECKING:
    from collections.abc import Callable

    from justhtml.dom import Node

    from .spec import Linkify


@dataclass(frozen=True, slots=True)
class CompiledLinkifyTransform:
    kind: Literal["linkify"]
    skip_tags: frozenset[str]
    config: LinkifyConfig
    callback: Any | None
    report: Any | None


def compile_linkify_transform(t: Linkify) -> CompiledLinkifyTransform:
    return CompiledLinkifyTransform(
        kind="linkify",
        skip_tags=frozenset(name.lower() for name in t.skip_tags),
        config=LinkifyConfig(fuzzy_ip=t.fuzzy_ip, extra_tlds=t.extra_tlds),
        callback=t.callback,
        report=t.report,
    )


def apply_linkify_transform(
    *,
    parent: Node,
    node: Node,
    children: list[Any],
    child_index: int,
    transform_index: int,
    transform: CompiledLinkifyTransform,
    mark_start: Callable[[object, int], None],
) -> bool:
    linkify_text = str(node.data or "")
    if not linkify_text:
        return False

    matches = find_links_with_config(linkify_text, transform.config)
    if not matches:
        return False

    if transform.callback is not None:
        transform.callback(node)
    if transform.report is not None:
        transform.report(f"Linkified {len(matches)} link(s) in text node", node=node)

    namespace = parent.namespace or "html"
    replacement: list[Any] = []
    cursor = 0
    for match in matches:
        if match.start > cursor:
            text = Text(linkify_text[cursor : match.start])
            mark_start(text, transform_index + 1)
            text.parent = parent
            replacement.append(text)

        link = Element("a", {"href": match.href}, namespace)
        link.append_child(Text(match.text))
        mark_start(link, transform_index + 1)
        link.parent = parent
        replacement.append(link)
        cursor = match.end

    if cursor < len(linkify_text):
        tail = Text(linkify_text[cursor:])
        mark_start(tail, transform_index + 1)
        tail.parent = parent
        replacement.append(tail)

    children[child_index : child_index + 1] = replacement
    node.parent = None
    return True
