from __future__ import annotations

import unittest

from justhtml import JustHTML
from justhtml.node import Comment, DocumentFragment, Element, Node, Template, Text
from justhtml.sanitize import SanitizationPolicy
from justhtml.transforms import (
    CollapseWhitespace,
    Drop,
    Linkify,
    PruneEmpty,
    Sanitize,
    SetAttrs,
    apply_compiled_transforms,
    compile_transforms,
)


class TestSanitizeTransform(unittest.TestCase):
    def test_compile_transforms_empty_is_ok(self) -> None:
        assert compile_transforms(()) == []

    def test_compile_transforms_allows_multiple_sanitize(self) -> None:
        compile_transforms((Sanitize(), Sanitize()))

    def test_multiple_sanitize_can_re_sanitize_after_transforms(self) -> None:
        doc = JustHTML(
            "<p><a>t</a></p>",
            fragment=True,
            transforms=[
                Sanitize(),
                SetAttrs("a", href="javascript:alert(1)"),
                Sanitize(),
            ],
        )

        # Sanitize runs again after SetAttrs introduced a new unsafe URL.
        assert doc.to_html(pretty=False) == "<p><a>t</a></p>"

    def test_sanitize_drops_comments_doctype_and_unsafe_content(self) -> None:
        html = """<!doctype html>
        <!--c-->
        <div class="ok" id="x" title="t" lang="en" dir="ltr"></div>
        <div onclick="x" srcdoc="y" data:bad="1" foo="bar"></div>
        <foo><b>ok</b></foo>
        <svg><circle /></svg>
        <a href="//example.com/path">x</a>
        <a href="javascript:alert(1)">x</a>
        <img src="https://example.com/x.png">
        """
        doc = JustHTML(html, sanitize=True)

        stack = [doc.root]
        seen = set()
        while stack:
            node = stack.pop()
            seen.add(node.name)
            stack.extend(getattr(node, "children", None) or [])
            tc = getattr(node, "template_content", None)
            if tc is not None:
                stack.append(tc)
        assert "#comment" not in seen
        assert "!doctype" not in seen

        div1 = doc.root.query("div")[0]
        assert div1.attrs == {"class": "ok", "id": "x", "title": "t", "lang": "en", "dir": "ltr"}

        div2 = doc.root.query("div")[1]
        assert div2.attrs == {}

        b = doc.root.query("b")[0]
        assert b.children[0].data == "ok"
        assert not doc.root.query("foo")

        assert not doc.root.query("svg")

        a1 = doc.root.query("a")[0]
        assert a1.attrs.get("href") == "https://example.com/path"
        a2 = doc.root.query("a")[1]
        assert "href" not in a2.attrs
        img = doc.root.query("img")[0]
        assert "src" not in img.attrs

    def test_sanitize_transform_makes_dom_safe_in_place(self) -> None:
        doc = JustHTML(
            '<p><a href="javascript:alert(1)" onclick="x()">x</a><script>alert(1)</script></p>',
            fragment=True,
            transforms=[Sanitize()],
        )

        # Tree is already sanitized.
        assert doc.to_html(pretty=False) == "<p><a>x</a></p>"

    def test_compile_transforms_allows_transforms_after_sanitize(self) -> None:
        compile_transforms((Sanitize(), Linkify()))
        compile_transforms((Sanitize(), SetAttrs("p", **{"class": "x"})))

    def test_transforms_can_run_after_sanitize(self) -> None:
        doc = JustHTML(
            '<p><a href="javascript:alert(1)" onclick="x()">x</a> https://example.com</p>',
            fragment=True,
            transforms=[Sanitize(), Linkify()],
        )

        # Existing unsafe content is removed by Sanitize, then Linkify runs.
        assert doc.to_html(pretty=False) == ('<p><a>x</a> <a href="https://example.com">https://example.com</a></p>')

    def test_pruneempty_can_run_after_sanitize(self) -> None:
        doc = JustHTML(
            "<p><script>alert(1)</script></p>",
            fragment=True,
            transforms=[Sanitize(), PruneEmpty("p")],
        )
        assert doc.to_html(pretty=False) == ""

    def test_drop_then_pruneempty_can_run_after_sanitize_in_order(self) -> None:
        doc = JustHTML(
            "<p><a>x</a></p>",
            fragment=True,
            transforms=[Sanitize(), Drop("a"), PruneEmpty("p")],
        )
        assert doc.to_html(pretty=False) == ""

    def test_collapsewhitespace_can_run_after_sanitize(self) -> None:
        doc = JustHTML(
            "<p>a  b</p>",
            fragment=True,
            transforms=[Sanitize(), CollapseWhitespace()],
        )
        assert doc.to_html(pretty=False) == "<p>a b</p>"

    def test_post_sanitize_collapsewhitespace_then_pruneempty_runs_in_order(self) -> None:
        doc = JustHTML(
            "<p>   </p><p>x</p>",
            fragment=True,
            transforms=[Sanitize(), CollapseWhitespace(), PruneEmpty("p")],
        )
        assert doc.to_html(pretty=False) == "<p>x</p>"

    def test_post_sanitize_pruneempty_then_collapsewhitespace_runs_in_order(self) -> None:
        doc = JustHTML(
            "<p>a  b</p><span> </span>",
            fragment=True,
            transforms=[Sanitize(), PruneEmpty("span"), CollapseWhitespace()],
        )
        assert doc.to_html(pretty=False) == "<p>a b</p>"

    def test_post_sanitize_consecutive_pruneempty_transforms_are_batched(self) -> None:
        doc = JustHTML(
            "<div><p></p></div>",
            fragment=True,
            transforms=[Sanitize(), PruneEmpty("p"), PruneEmpty("div")],
        )
        assert doc.to_html(pretty=False) == ""

    def test_sanitize_transform_supports_element_root(self) -> None:
        root = Element("a", {"href": "javascript:alert(1)", "onclick": "x()"}, "html")
        wrapper = DocumentFragment()
        wrapper.append_child(root)
        compiled = compile_transforms((Sanitize(),))
        apply_compiled_transforms(wrapper, compiled)

        assert root.attrs == {}

    def test_sanitize_transform_supports_template_root(self) -> None:
        root = Template("div", attrs={"onclick": "x()", "class": "ok"}, namespace="html")
        root.append_child(Element("span", {}, "html"))

        assert root.template_content is not None
        script = Element("script", {}, "html")
        script.append_child(Text("alert(1)"))
        root.template_content.append_child(script)

        wrapper = DocumentFragment()
        wrapper.append_child(root)

        compiled = compile_transforms((Sanitize(),))
        apply_compiled_transforms(wrapper, compiled)

        assert "onclick" not in root.attrs
        assert root.attrs.get("class") == "ok"
        assert all(child.parent is root for child in root.children)
        assert root.template_content is not None
        assert root.template_content.children == []

    def test_sanitize_transform_supports_text_root(self) -> None:
        wrapper = DocumentFragment()
        root = Text("hello")
        wrapper.append_child(root)
        compiled = compile_transforms((Sanitize(),))
        apply_compiled_transforms(wrapper, compiled)
        assert root.data == "hello"

    def test_sanitize_transform_supports_simpledomnode_element_root(self) -> None:
        root = Node("a", {"href": "javascript:alert(1)", "onclick": "x()"}, namespace="html")
        wrapper = DocumentFragment()
        wrapper.append_child(root)
        compiled = compile_transforms((Sanitize(),))
        apply_compiled_transforms(wrapper, compiled)
        assert root.attrs == {}

    def test_sanitize_transform_policy_override_is_used(self) -> None:
        # Covers the Sanitize(policy=...) override path.
        policy = SanitizationPolicy(allowed_tags={"p"}, allowed_attributes={"*": set()})
        root = DocumentFragment()
        compiled = compile_transforms((Sanitize(policy),))
        apply_compiled_transforms(root, compiled)

        assert root.name == "#document-fragment"

    def test_sanitize_transform_escape_disallowed_tags(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags={"b"},
            allowed_attributes={"*": set(), "b": set()},
            disallowed_tag_handling="escape",
        )
        doc = JustHTML(
            "<b>Hello <sarcasm>world</sarcasm></b>",
            fragment=True,
            transforms=[Sanitize(policy)],
        )
        assert doc.to_html(pretty=False) == "<b>Hello &lt;sarcasm&gt;world&lt;/sarcasm&gt;</b>"

    def test_sanitize_transform_escape_disallowed_self_closing_tag(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags={"p"},
            allowed_attributes={"*": set(), "p": set()},
            disallowed_tag_handling="escape",
        )
        doc = JustHTML(
            "<p>yeah right<sarcasm/></p>",
            fragment=True,
            transforms=[Sanitize(policy)],
        )
        assert doc.to_html(pretty=False) == "<p>yeah right&lt;sarcasm/&gt;</p>"

    def test_sanitize_transform_escape_disallowed_missing_end_tag(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags={"b"},
            allowed_attributes={"*": set(), "b": set()},
            disallowed_tag_handling="escape",
        )
        doc = JustHTML(
            "<b>Hello <sarcasm>world</b>",
            fragment=True,
            transforms=[Sanitize(policy)],
        )
        assert doc.to_html(pretty=False) == "<b>Hello &lt;sarcasm&gt;world</b>"

    def test_sanitize_transform_escape_disallowed_with_allowed_children(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags={"b"},
            allowed_attributes={"*": set(), "b": set()},
            disallowed_tag_handling="escape",
        )
        doc = JustHTML(
            "<sarcasm class='x'><b>world</b></sarcasm>",
            fragment=True,
            transforms=[Sanitize(policy)],
        )
        assert doc.to_html(pretty=False) == "&lt;sarcasm class='x'&gt;<b>world</b>&lt;/sarcasm&gt;"

    def test_sanitize_transform_drop_disallowed_subtree(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags={"b"},
            allowed_attributes={"*": set(), "b": set()},
            disallowed_tag_handling="drop",
        )
        doc = JustHTML(
            "<b>Hello <sarcasm>world</sarcasm></b>",
            fragment=True,
            transforms=[Sanitize(policy)],
        )
        assert doc.to_html(pretty=False) == "<b>Hello </b>"

    def test_sanitize_transform_escape_without_source_html(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags={"p"},
            allowed_attributes={"*": set(), "p": set()},
            disallowed_tag_handling="escape",
        )
        root = DocumentFragment()
        node = Element("x", {}, "html")
        node._start_tag_start = 0
        node._start_tag_end = 2
        node.append_child(Text("ok"))
        root.append_child(node)

        compiled = compile_transforms((Sanitize(policy),))
        apply_compiled_transforms(root, compiled)
        assert root.to_html(pretty=False) == "&lt;x&gt;ok"

    def test_sanitize_transform_escape_uses_raw_tokens(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags={"p"},
            allowed_attributes={"*": set(), "p": set()},
            disallowed_tag_handling="escape",
        )
        doc = JustHTML(
            "<p>Hello <x>world</x></p>",
            fragment=True,
            transforms=[Sanitize(policy)],
        )
        assert doc.to_html(pretty=False) == "<p>Hello &lt;x&gt;world&lt;/x&gt;</p>"

    def test_sanitize_transform_escape_template_content(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags={"b"},
            allowed_attributes={"*": set(), "b": set()},
            disallowed_tag_handling="escape",
        )
        doc = JustHTML(
            "<template><b>x</b></template>",
            fragment=True,
            transforms=[Sanitize(policy)],
        )
        assert doc.to_html(pretty=False) == "&lt;template&gt;<b>x</b>&lt;/template&gt;"

    def test_sanitize_transform_converts_comment_root_to_fragment_when_dropped(self) -> None:
        root = Comment(data="x")
        wrapper = DocumentFragment()
        wrapper.append_child(root)
        compiled = compile_transforms((Sanitize(),))
        apply_compiled_transforms(wrapper, compiled)

        assert wrapper.children == []

    def test_sanitize_transform_converts_doctype_root_to_fragment_when_dropped(self) -> None:
        root = Node("!doctype")
        wrapper = DocumentFragment()
        wrapper.append_child(root)
        compiled = compile_transforms((Sanitize(),))
        apply_compiled_transforms(wrapper, compiled)

        assert wrapper.children == []

    def test_sanitize_transform_drops_foreign_namespace_element_root(self) -> None:
        root = Node("p", namespace="svg")
        root.append_child(Node("span"))

        wrapper = DocumentFragment()
        wrapper.append_child(root)

        compiled = compile_transforms((Sanitize(),))
        apply_compiled_transforms(wrapper, compiled)

        assert wrapper.children == []

    def test_sanitize_transform_drops_foreign_namespace_element_root_without_children(self) -> None:
        root = Node("p", namespace="svg")

        wrapper = DocumentFragment()
        wrapper.append_child(root)

        compiled = compile_transforms((Sanitize(),))
        apply_compiled_transforms(wrapper, compiled)

        assert wrapper.children == []

    def test_sanitize_transform_drops_content_for_drop_content_tag_root(self) -> None:
        root = Node("script")
        root.append_child(Text("alert(1)"))

        wrapper = DocumentFragment()
        wrapper.append_child(root)

        compiled = compile_transforms((Sanitize(),))
        apply_compiled_transforms(wrapper, compiled)

        assert wrapper.children == []

    def test_sanitize_transform_drops_content_for_drop_content_tag_root_without_children(self) -> None:
        root = Node("script")

        wrapper = DocumentFragment()
        wrapper.append_child(root)

        compiled = compile_transforms((Sanitize(),))
        apply_compiled_transforms(wrapper, compiled)

        assert wrapper.children == []

    def test_sanitize_transform_disallowed_root_hoists_children(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=set(),
            allowed_attributes={"*": set()},
            drop_foreign_namespaces=False,
            drop_content_tags=set(),
        )
        root = Node("p")
        root.append_child(Text("x"))

        wrapper = DocumentFragment()
        wrapper.append_child(root)

        compiled = compile_transforms((Sanitize(policy),))
        apply_compiled_transforms(wrapper, compiled)
        assert wrapper.to_html(pretty=False) == "x"

    def test_sanitize_transform_disallowed_root_without_children_is_empty(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=set(),
            allowed_attributes={"*": set()},
            drop_foreign_namespaces=False,
            drop_content_tags=set(),
        )
        root = Node("p")

        wrapper = DocumentFragment()
        wrapper.append_child(root)

        compiled = compile_transforms((Sanitize(policy),))
        apply_compiled_transforms(wrapper, compiled)
        assert wrapper.to_html(pretty=False) == ""

    def test_sanitize_transform_disallowed_template_root_hoists_template_content(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags={"b"},
            allowed_attributes={"*": set()},
            drop_foreign_namespaces=False,
            drop_content_tags=set(),
        )
        root = Template("template", attrs={}, namespace="html")
        assert root.template_content is not None
        root.template_content.append_child(Node("b"))

        wrapper = DocumentFragment()
        wrapper.append_child(root)

        compiled = compile_transforms((Sanitize(policy),))
        apply_compiled_transforms(wrapper, compiled)

        assert wrapper.children is not None
        assert [c.name for c in wrapper.children] == ["b"]

    def test_sanitize_transform_disallowed_template_root_with_empty_template_content_hoists_children(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags={"b"},
            allowed_attributes={"*": set()},
            drop_foreign_namespaces=False,
            drop_content_tags=set(),
        )
        root = Template("template", attrs={}, namespace="html")
        root.append_child(Node("b"))

        wrapper = DocumentFragment()
        wrapper.append_child(root)

        compiled = compile_transforms((Sanitize(policy),))
        apply_compiled_transforms(wrapper, compiled)

        assert wrapper.children is not None
        assert [c.name for c in wrapper.children] == ["b"]

    def test_sanitize_transform_decide_handles_comments_doctype_and_containers(self) -> None:
        root = DocumentFragment()
        root.append_child(Comment(data="x"))
        root.append_child(Node("!doctype"))
        nested = DocumentFragment()
        nested.append_child(Node("p"))
        root.append_child(nested)

        compiled = compile_transforms((Sanitize(),))
        apply_compiled_transforms(root, compiled)

        assert root.to_html(pretty=False) == "<p></p>"

    def test_sanitize_transform_decide_drops_foreign_namespace_elements(self) -> None:
        root = DocumentFragment()
        root.append_child(Node("p", namespace="svg"))

        compiled = compile_transforms((Sanitize(),))
        apply_compiled_transforms(root, compiled)

        assert root.to_html(pretty=False) == ""

    def test_sanitize_transform_decide_unwraps_disallowed_elements(self) -> None:
        root = DocumentFragment()
        blink = Node("blink")
        blink.append_child(Text("x"))
        root.append_child(blink)

        compiled = compile_transforms((Sanitize(),))
        apply_compiled_transforms(root, compiled)

        assert root.to_html(pretty=False) == "x"
