"""Rawtext element hardening for sanitizer DOM passes."""

from __future__ import annotations

from typing import Any

from justhtml.core.rawtext import neutralize_rawtext_end_tag_sequences
from justhtml.core.types import ParseError

from .css import _css_value_may_load_external_resource

_RAWTEXT_SERIALIZATION_ELEMENTS: frozenset[str] = frozenset({"script", "style"})


def _neutralize_rawtext_end_tag_sequences(text: str, tag_name: str) -> tuple[str, bool]:
    return neutralize_rawtext_end_tag_sequences(text, tag_name)


def _record_rawtext_security_issue(
    *,
    policy: Any,
    errors: list[ParseError] | None,
    code: str,
    message: str,
    node: Any,
) -> None:
    policy.handle_unsafe(message, node=node)
    if errors is None:
        return
    errors.append(
        ParseError(
            code,
            line=node.origin_line,
            column=node.origin_col,
            category="security",
            message=message,
        )
    )


def _sanitize_rawtext_element_contents(
    node: Any,
    *,
    policy: Any,
    errors: list[ParseError] | None,
) -> None:
    from justhtml.dom import Template  # noqa: PLC0415

    stack: list[Any] = [node]

    while stack:
        current = stack.pop()
        raw_name = current.name
        if type(raw_name) is str:
            name = raw_name if raw_name.islower() else raw_name.lower()
        else:  # pragma: no cover
            name = str(raw_name).lower()

        if name in _RAWTEXT_SERIALIZATION_ELEMENTS:
            children = current.children
            if not children:
                continue

            text_children: list[Any] = []
            text_parts: list[str] = []
            for child in children:
                if child.name == "#text":
                    text_children.append(child)
                    text_parts.append(child.data or "")
                    continue

                _record_rawtext_security_issue(
                    policy=policy,
                    errors=errors,
                    code="unsafe-rawtext-child",
                    message=f"Unsafe non-text child inside <{name}> was dropped",
                    node=child,
                )
                child.parent = None

            if not text_children:
                current.children = []
                continue

            combined_text = "".join(text_parts)
            primary_text = text_children[0]
            sanitized_text, changed = _neutralize_rawtext_end_tag_sequences(combined_text, str(name))

            if changed:
                _record_rawtext_security_issue(
                    policy=policy,
                    errors=errors,
                    code="unsafe-rawtext-end-tag",
                    message=f"Unsafe raw text inside <{name}> contains a closing tag sequence",
                    node=primary_text,
                )
                primary_text.data = sanitized_text
                for extra_text in text_children[1:]:
                    extra_text.parent = None
                current.children = [primary_text] if sanitized_text else []
            else:
                current.children = text_children

            if name == "style" and sanitized_text and _css_value_may_load_external_resource(sanitized_text):
                _record_rawtext_security_issue(
                    policy=policy,
                    errors=errors,
                    code="unsafe-style-resource",
                    message="Unsafe CSS inside <style> contains resource-loading constructs",
                    node=primary_text,
                )
                for child in current.children:
                    child.parent = None
                current.children = []
            continue

        children = current.children
        if children:
            stack.extend(reversed(children))

        if type(current) is Template and current.template_content is not None:
            stack.append(current.template_content)
