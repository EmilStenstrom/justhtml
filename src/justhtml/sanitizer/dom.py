"""DOM entry points for sanitizer execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from .policy import SanitizationPolicy, _default_policy_for_root_name

if TYPE_CHECKING:
    from justhtml.core.types import ParseError
    from justhtml.dom import NodeType


def _sanitize(node: Any, *, policy: SanitizationPolicy | None = None) -> Any:
    """Return a sanitized clone of `node`.

    This returns a sanitized clone without mutating the original tree's
    structure, attributes, or content. One exception: in "escape" mode, the
    internal `_source_html` bookkeeping field may be propagated onto the
    original tree's descendants before cloning (see below) as a performance
    trade-off; this never affects the original tree's visible content.
    For performance, it builds the sanitized clone in a single pass.
    """

    if policy is None:
        policy = _default_policy_for_root_name(node.name)

    if policy.unsafe_handling == "collect":
        policy.reset_collected_security_errors()

    # Escape-mode tag reconstruction may need access to the original source HTML.
    # Historically we allow a child element to inherit _source_html from an
    # ancestor container; keep that behavior even though we sanitize a clone.
    if policy.disallowed_tag_handling == "escape":
        from justhtml.dom import Node, Template  # noqa: PLC0415

        root_source_html = node._source_html if isinstance(node, Node) else None
        if root_source_html:
            stack: list[Node] = [node]
            while stack:
                current = stack.pop()
                current_source_html = current._source_html or root_source_html

                children = current.children or ()
                for child in children:
                    # Text does not have _source_html.
                    if not isinstance(child, Node):
                        continue
                    if child._source_html is None:
                        child._source_html = current_source_html
                    stack.append(child)

                if type(current) is Template and current.template_content is not None:
                    tc = current.template_content
                    if tc._source_html is None:
                        tc._source_html = current_source_html
                    stack.append(tc)

    return _apply_policy_to_root(node.clone_node(deep=True), policy=policy, errors=None)


def _apply_policy_to_root(
    node: Any,
    *,
    policy: SanitizationPolicy,
    errors: list[ParseError] | None,
) -> Any:
    compiled_policy = policy.compile()

    # Transforms walk children of their root. Wrap an element so the policy can
    # also decide the fate of the supplied root itself.
    if node.name in {"#document", "#document-fragment"}:
        compiled_policy.apply_to(node, errors=errors)
        return node

    from justhtml.dom import DocumentFragment  # noqa: PLC0415

    wrapper = DocumentFragment()
    wrapper.append_child(node)
    compiled_policy.apply_to(wrapper, errors=errors)

    children = cast("list[Any]", wrapper.children)
    if len(children) == 1:
        only = children[0]
        only.parent = None
        wrapper.children = []
        return only

    return wrapper


def sanitize_dom(
    node: NodeType,
    *,
    policy: SanitizationPolicy | None = None,
    errors: list[ParseError] | None = None,
) -> NodeType:
    """Sanitize a DOM tree in place.

    For document roots (`#document` or `#document-fragment`), this mutates the
    tree in place and returns the same root. For other nodes, the node is
    sanitized as if it were the only child of a document fragment; the returned
    node may need to be reattached by the caller.
    """

    if policy is None:
        policy = _default_policy_for_root_name(node.name)

    if policy.unsafe_handling == "collect":
        policy.reset_collected_security_errors()

    result = _apply_policy_to_root(node, policy=policy, errors=errors)

    return cast("NodeType", result)
