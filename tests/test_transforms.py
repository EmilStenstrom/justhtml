from __future__ import annotations

import unittest
from typing import Any

from justhtml import JustHTML as _JustHTML
from justhtml import SelectorError
from justhtml.node import Comment, Document, DocumentFragment, Element, Node, Template, Text
from justhtml.sanitize import SanitizationPolicy, UrlPolicy, UrlRule
from justhtml.transforms import (
    AllowlistAttrs,
    AllowStyleAttrs,
    CollapseWhitespace,
    Decide,
    DecideAction,
    Drop,
    DropAttrs,
    DropComments,
    DropDoctype,
    DropForeignNamespaces,
    DropUrlAttrs,
    Edit,
    EditAttrs,
    EditDocument,
    Empty,
    Escape,
    Linkify,
    MergeAttrs,
    PruneEmpty,
    Sanitize,
    SetAttrs,
    Stage,
    Unwrap,
    _glob_match,
    apply_compiled_transforms,
    compile_transforms,
    emit_error,
)


def JustHTML(*args, **kwargs):  # noqa: N802
    if "sanitize" not in kwargs and "safe" not in kwargs:
        kwargs["sanitize"] = False
    return _JustHTML(*args, **kwargs)


class TestTransforms(unittest.TestCase):
    def test_glob_match_star_matches_everything(self) -> None:
        assert _glob_match("*", "anything") is True
        # Ensure the trailing-'*' consumption loop is exercised too.
        assert _glob_match("**", "") is True

    def test_glob_match_returns_false_on_wildcard_mismatch(self) -> None:
        # Exercise the internal mismatch path for wildcard patterns.
        assert _glob_match("a?c", "axd") is False

    def test_compile_transforms_rejects_unknown_transform_type(self) -> None:
        with self.assertRaises(TypeError):
            compile_transforms([object()])

    def test_rewriteattrs_selector_star_uses_all_nodes_fast_path(self) -> None:
        root = DocumentFragment()
        root.append_child(Element("div", {"a": "1"}, "html"))

        def cb(node: Node) -> dict[str, str | None] | None:
            out = dict(node.attrs)
            out["b"] = "2"
            return out

        compiled = compile_transforms([EditAttrs("*", cb)])
        apply_compiled_transforms(root, compiled)
        assert root.children[0].attrs.get("b") == "2"

    def test_compile_transforms_fuses_adjacent_rewriteattrs_with_same_selector(self) -> None:
        root = DocumentFragment()
        root.append_child(Element("div", {"a": "1"}, "html"))

        def cb1(node: Node) -> dict[str, str | None] | None:
            out = dict(node.attrs)
            out["b"] = "2"
            return out

        def cb2(node: Node) -> dict[str, str | None] | None:
            out = dict(node.attrs)
            out["c"] = "3"
            return out

        compiled = compile_transforms([EditAttrs("*", cb1), EditAttrs("*", cb2)])
        # Fused into a single rewrite-attrs transform.
        assert sum(1 for t in compiled if getattr(t, "kind", None) == "rewrite_attrs") == 1

        apply_compiled_transforms(root, compiled)
        assert root.children[0].attrs == {"a": "1", "b": "2", "c": "3"}

    def test_constructor_accepts_transforms_and_applies_setattrs(self) -> None:
        doc = JustHTML("<p>Hello</p>", transforms=[SetAttrs("p", id="x")])
        assert doc.to_html(pretty=False) == '<html><head></head><body><p id="x">Hello</p></body></html>'

    def test_constructor_compiles_selectors_and_raises_early(self) -> None:
        with self.assertRaises(SelectorError):
            JustHTML("<p>Hello</p>", transforms=[SetAttrs("div[invalid", id="x")])

    def test_drop_removes_nodes(self) -> None:
        doc = JustHTML("<p>ok</p><script>alert(1)</script>", transforms=[Drop("script")])
        assert doc.to_html(pretty=False) == "<html><head></head><body><p>ok</p></body></html>"

    def test_unwrap_hoists_children(self) -> None:
        doc = JustHTML("<p>Hello <span>world</span></p>", transforms=[Unwrap("span")])
        assert doc.to_html(pretty=False) == "<html><head></head><body><p>Hello world</p></body></html>"

    def test_unwrap_handles_empty_elements(self) -> None:
        doc = JustHTML("<div><span></span>ok</div>", transforms=[Unwrap("span")])
        assert doc.to_html(pretty=False) == "<html><head></head><body><div>ok</div></body></html>"

    def test_escape_emits_tags_as_text_and_hoists_children(self) -> None:
        seen: list[str] = []
        reports: list[str] = []

        def cb(node: Node) -> None:
            seen.append(str(node.name))

        def report(msg: str, *, node: Node | None = None) -> None:
            reports.append(msg)

        doc = JustHTML(
            "<p>Hello <span>world</span></p>",
            fragment=True,
            transforms=[Escape("span", callback=cb, report=report)],
        )

        assert doc.to_html(pretty=False) == "<p>Hello &lt;span&gt;world&lt;/span&gt;</p>"
        assert seen == ["span"]
        assert reports == ["Escaped <span> (matched selector 'span')"]

    def test_escape_reports_without_callback_and_without_report(self) -> None:
        # Exercise both branches in the Escape selector path.
        doc = JustHTML(
            "<p>Hello <span>world</span></p>",
            fragment=True,
            transforms=[Escape("span")],
        )
        assert doc.to_html(pretty=False) == "<p>Hello &lt;span&gt;world&lt;/span&gt;</p>"

    def test_empty_removes_children_but_keeps_element(self) -> None:
        doc = JustHTML("<div><b>x</b>y</div>", transforms=[Empty("div")])
        assert doc.to_html(pretty=False) == "<html><head></head><body><div></div></body></html>"

    def test_empty_also_clears_template_content(self) -> None:
        doc = JustHTML("<template><b>x</b></template>", transforms=[Empty("template")])
        assert doc.to_html(pretty=False) == "<html><head><template></template></head><body></body></html>"

    def test_edit_can_mutate_attrs(self) -> None:
        def cb(node):
            node.attrs["data-x"] = "1"

        doc = JustHTML('<a href="https://e.com">x</a>', transforms=[Edit("a", cb)])
        assert 'data-x="1"' in doc.to_html(pretty=False)

    def test_editdocument_runs_once_on_root(self) -> None:
        seen: list[str] = []

        def cb(root: Node) -> None:
            seen.append(str(root.name))
            root.append_child(Node("p"))

        doc = JustHTML("<p>x</p>", fragment=True, transforms=[EditDocument(cb)])
        assert seen == ["#document-fragment"]
        assert doc.to_html(pretty=False) == "<p>x</p><p></p>"

    def test_walk_transforms_traverse_root_template_content(self) -> None:
        root = Template("template", attrs={}, namespace="html")
        assert root.template_content is not None
        root.template_content.append_child(Element("p", {}, "html"))

        apply_compiled_transforms(root, compile_transforms([SetAttrs("p", id="x")]))
        assert root.to_html(pretty=False) == '<template><p id="x"></p></template>'

    def test_transform_callbacks_can_emit_errors_without_parse_error_collection(self) -> None:
        def cb(node: Node) -> None:
            emit_error("transform-warning", node=node, message="bad <p>")

        doc = JustHTML(
            "<!--x--><p>Hello</p>",
            track_node_locations=True,
            transforms=[Edit("p", cb), SetAttrs("p", id="x")],
        )
        assert len(doc.errors) == 1
        err = doc.errors[0]
        assert err.category == "transform"
        assert err.code == "transform-warning"
        assert err.message == "bad <p>"
        assert err.line is not None
        assert err.column is not None
        assert 'id="x"' in doc.to_html(pretty=False)

    def test_transform_callback_errors_merge_with_parse_errors_when_collect_errors_true(self) -> None:
        doc = JustHTML(
            "<p>\x00</p>",
            collect_errors=True,
            track_node_locations=True,
            transforms=[Edit("p", lambda n: emit_error("transform-warning", node=n, message="bad <p>"))],
        )
        codes = {e.code for e in doc.errors}
        assert "transform-warning" in codes
        assert "unexpected-null-character" in codes

    def test_emit_error_noops_without_active_sink(self) -> None:
        root = JustHTML("<p>x</p>", fragment=True, track_node_locations=True).root
        compiled = compile_transforms([Edit("p", lambda n: emit_error("x", node=n, message="msg"))])
        apply_compiled_transforms(root, compiled)

        errs = []
        compiled2 = compile_transforms([Edit("p", lambda n: emit_error("x", line=1, column=2, message="msg"))])
        apply_compiled_transforms(root, compiled2, errors=errs)
        assert len(errs) == 1
        assert errs[0].code == "x"
        assert errs[0].line == 1
        assert errs[0].column == 2

    def test_transforms_run_in_order_and_drop_short_circuits(self) -> None:
        doc = JustHTML(
            "<p>Hello</p>",
            transforms=[SetAttrs("p", id="x"), Drop("p"), SetAttrs("p", class_="y")],
        )
        assert doc.to_html(pretty=False) == "<html><head></head><body></body></html>"

    def test_disabled_transforms_are_omitted_at_compile_time(self) -> None:
        doc = JustHTML(
            "<p>ok</p><script>alert(1)</script><div><b>x</b></div>",
            transforms=[
                Drop("script", enabled=False),
                Unwrap("b", enabled=False),
                Empty("div", enabled=False),
            ],
        )
        assert (
            doc.to_html(pretty=False)
            == "<html><head></head><body><p>ok</p><script>alert(1)</script><div><b>x</b></div></body></html>"
        )

    def test_drop_with_callback_uses_general_selector_path_when_not_simple_tag_list(self) -> None:
        dropped: list[str] = []

        def callback(node: Node) -> None:
            dropped.append(str(node.name))

        doc = JustHTML(
            '<div class="x"></div><div class="y"></div>',
            fragment=True,
            transforms=[Drop("div.x", callback=callback)],
        )
        assert doc.to_html(pretty=False) == '<div class="y"></div>'
        assert dropped == ["div"]

    def test_drop_with_callback_tag_list_fast_path_rejection_still_validates_selector(self) -> None:
        dropped: list[str] = []

        def callback(node: Node) -> None:
            dropped.append(str(node.name))

        doc = JustHTML(
            "<script>x</script><p>ok</p>",
            fragment=True,
            transforms=[Drop("script, ", callback=callback)],
        )
        assert doc.to_html(pretty=False) == "<p>ok</p>"
        assert dropped == ["script"]

    def test_hook_callback_property_exposes_configured_hook(self) -> None:
        def cb_node(n: Node) -> None:
            return None

        def cb_report(msg: str, *, node: object | None = None) -> None:
            return None

        assert Drop("p", callback=cb_node).callback is cb_node
        assert Unwrap("p", callback=cb_node).callback is cb_node
        assert DropForeignNamespaces(report=cb_report).report is cb_report
        assert DropAttrs("*", report=cb_report).report is cb_report
        assert AllowlistAttrs("*", allowed_attributes={"*": []}, report=cb_report).report is cb_report

        url_policy = UrlPolicy()
        assert DropUrlAttrs("*", url_policy=url_policy, report=cb_report).report is cb_report
        assert AllowStyleAttrs("[style]", allowed_css_properties=set(), report=cb_report).report is cb_report

    def test_callbacks_and_reports_run_for_structural_transforms(self) -> None:
        calls: list[tuple[str, str]] = []

        def on_node(n: Node) -> None:
            calls.append(("node", str(n.name)))

        def on_report(msg: str, *, node: object | None = None) -> None:
            calls.append(("report", msg))

        root = DocumentFragment()
        root.append_child(Comment(data="x"))
        root.append_child(Node("!doctype"))
        root.append_child(Element("a", {"rel": "nofollow"}, "html"))

        compiled = compile_transforms(
            [
                DropComments(callback=on_node, report=on_report),
                DropDoctype(callback=on_node, report=on_report),
                MergeAttrs("a", attr="rel", tokens={"noopener"}, callback=on_node, report=on_report),
            ]
        )
        apply_compiled_transforms(root, compiled)

        assert root.to_html(pretty=False) == '<a rel="nofollow noopener"></a>'
        assert ("node", "#comment") in calls
        assert ("node", "!doctype") in calls
        assert ("node", "a") in calls
        assert any(msg == "Dropped comment" for kind, msg in calls if kind == "report")
        assert any(msg == "Dropped doctype" for kind, msg in calls if kind == "report")
        assert any("Merged tokens" in msg for kind, msg in calls if kind == "report")

    def test_callback_and_report_run_for_text_transforms(self) -> None:
        calls: list[str] = []

        def on_node(n: Node) -> None:
            calls.append(str(n.name))

        def on_report(msg: str, *, node: object | None = None) -> None:
            calls.append(msg)

        root = DocumentFragment()
        p = Element("p", {}, "html")
        p.append_child(Text("visit https://example.com  now"))
        root.append_child(p)

        compiled = compile_transforms(
            [
                CollapseWhitespace(callback=on_node, report=on_report),
                Linkify(callback=on_node, report=on_report),
            ]
        )
        apply_compiled_transforms(root, compiled)

        assert "Collapsed whitespace in text node" in calls
        assert any(c.startswith("Linkified ") for c in calls)
        assert root.to_html(pretty=False) == '<p>visit <a href="https://example.com">https://example.com</a> now</p>'

    def test_setattrs_change_detection_controls_hooks(self) -> None:
        calls: list[str] = []

        def on_node(n: Node) -> None:
            calls.append("node")

        def on_report(msg: str, *, node: object | None = None) -> None:
            calls.append(msg)

        root = DocumentFragment()
        p = Element("p", {"id": "x"}, "html")
        root.append_child(p)

        # First: no change.
        apply_compiled_transforms(
            root, compile_transforms([SetAttrs("p", callback=on_node, report=on_report, id="x")])
        )
        assert calls == []

        # Then: change.
        apply_compiled_transforms(
            root, compile_transforms([SetAttrs("p", callback=on_node, report=on_report, id="y")])
        )
        assert calls and calls[0] == "node"
        assert any("Set attributes" in c for c in calls)

    def test_unwrap_hoists_template_content_and_runs_hooks(self) -> None:
        called: list[str] = []

        def on_node(n: Node) -> None:
            called.append(str(n.name))

        def on_report(msg: str, *, node: object | None = None) -> None:
            called.append(msg)

        root = DocumentFragment()
        tpl = Template("template", attrs={}, namespace="html")
        assert tpl.template_content is not None
        tpl.template_content.append_child(Element("b", {}, "html"))
        root.append_child(tpl)

        apply_compiled_transforms(root, compile_transforms([Unwrap("template", callback=on_node, report=on_report)]))
        assert root.to_html(pretty=False) == "<b></b>"
        assert "template" in called
        assert any("Unwrapped" in c for c in called)

    def test_decide_escape_covers_reconstruction_branches(self) -> None:
        def decide(n: Node) -> DecideAction:
            if n.name in {"#comment", "x", "y"}:
                return Decide.ESCAPE
            return Decide.KEEP

        root = DocumentFragment()
        root.append_child(Comment(data="x"))

        x = Element("x", {}, "html")
        x._start_tag_start = 0
        x._start_tag_end = 3
        x._end_tag_start = 5
        x._end_tag_end = 9
        x._end_tag_present = True
        x.append_child(Text("hi"))
        root.append_child(x)

        y = Element("y", {}, "html")
        y._start_tag_start = 0
        y._start_tag_end = 3
        y._self_closing = True
        root.append_child(y)

        # Force tag reconstruction rather than slicing from source HTML.
        root._source_html = None
        x._source_html = None
        y._source_html = None

        apply_compiled_transforms(root, compile_transforms([Decide("*", decide)]))

        assert root.to_html(pretty=False) == "&lt;x&gt;hi&lt;/x&gt;&lt;y/&gt;"

    def test_fused_sanitizer_callbacks_and_reporting(self) -> None:
        called = []
        reported = []

        def cb(n: Node) -> None:
            called.append(n.name)

        def rep(msg: str, node: Any | None = None) -> None:
            reported.append(msg)

        root = Document()
        root.append_child(Comment(data="foo"))
        root.append_child(Node("!doctype", data="html"))

        # Add foreign namespace node
        foreign = Node("svg", namespace="svg")
        root.append_child(foreign)

        # Add dropped content tag (e.g. script)
        script = Node("script")
        root.append_child(script)

        # Add unsafe URL attribute to verify reporting
        bad_link = Node("a", attrs={"href": "javascript:alert(1)"})
        root.append_child(bad_link)

        # Add unsafe inline style
        bad_style = Node("div", attrs={"style": "expression(alert(1))"})
        root.append_child(bad_style)

        # Policy that drops comments and doctypes
        policy = SanitizationPolicy(
            allowed_tags=frozenset({"a", "div"}),
            allowed_attributes={"a": {"href"}, "div": {"style"}},
            allowed_css_properties={"color"},  # Enabling style check
            drop_comments=True,
            drop_doctype=True,
            # drop_foreign_namespaces=True (default is usually True, but verify)
            # drop_content_tags defaults include script
        )
        # Use simple transform compilation to trigger fused path
        transforms = [Sanitize(policy, callback=cb, report=rep)]
        compiled = compile_transforms(transforms)

        apply_compiled_transforms(root, compiled)

        assert "#comment" in called
        assert "!doctype" in called
        assert "svg" in called
        assert "script" in called
        assert any("Dropped comment" in m for m in reported)
        assert any("Dropped doctype" in m for m in reported)
        assert any("foreign namespace" in m for m in reported)
        assert any("dropped content" in m for m in reported)
        assert any("Unsafe URL" in m for m in reported)
        assert any("Unsafe inline style" in m for m in reported)
        assert root.to_html(pretty=False) == "<a></a><div></div>"

    def test_reconstruct_end_tag_handles_void_elements(self) -> None:
        # We need a Decide.ESCAPE on a void element that has NO metadata source

        def decide_escape_br(n: Node) -> DecideAction:
            if n.name == "br":
                return Decide.ESCAPE
            return Decide.KEEP

        root_void = Node("div")
        br = Node("br")
        # Ensure it has NO source metadata and is NOT self-closing explicitly
        # so it hits the VOID_ELEMENTS check
        root_void.append_child(br)

        apply_compiled_transforms(root_void, compile_transforms([Decide("*", decide_escape_br)]))

        # Expect <br> (escaped as &lt;br&gt;) and no end tag
        assert root_void.to_html(pretty=False) == "<div>&lt;br&gt;</div>"

    def test_unwrap_moves_template_text_children(self) -> None:
        root = DocumentFragment()
        tpl = Template("template", attrs={}, namespace="html")
        assert tpl.template_content is not None
        tpl.template_content.append_child(Text("x"))
        root.append_child(tpl)

        apply_compiled_transforms(root, compile_transforms([Unwrap("template")]))
        assert root.to_html(pretty=False) == "x"

    def test_unwrap_moves_template_element_children_from_parsed_html(self) -> None:
        doc = JustHTML("<template><b>x</b></template>", fragment=True, transforms=[Unwrap("template")])
        assert doc.to_html(pretty=False) == "<b>x</b>"

    def test_edit_editdocument_decide_editattrs_hooks_and_reports(self) -> None:
        calls: list[str] = []

        def on_node(n: Node) -> None:
            calls.append(f"node:{n.name}")

        def on_report(msg: str, *, node: object | None = None) -> None:
            calls.append(msg)

        root = DocumentFragment()
        p = Element("p", {}, "html")
        p.append_child(Text("x"))
        root.append_child(p)

        def edit_p(n: Node) -> None:
            n.attrs["data-x"] = "1"

        def decide_drop(n: Node) -> DecideAction:
            return Decide.DROP

        def edit_attrs(n: Node) -> dict[str, str | None] | None:
            return {"id": "y"}

        compiled = compile_transforms(
            [
                Edit("p", edit_p, callback=on_node, report=on_report),
                EditAttrs("p", edit_attrs, callback=on_node, report=on_report),
                Decide("p", decide_drop, callback=on_node, report=on_report),
                EditDocument(lambda r: None, callback=on_node, report=on_report),
            ]
        )
        apply_compiled_transforms(root, compiled)

        # Decide drops <p>.
        assert root.children == []
        assert any(c.startswith("node:") for c in calls)
        assert any("Edited <p>" in c for c in calls)
        assert any("Edited attributes" in c for c in calls)
        assert any("Decide -> drop" in c for c in calls)
        assert "Edited document root" in calls

    def test_pruneempty_and_stage_hooks_can_report(self) -> None:
        calls: list[str] = []

        def on_node(n: Node) -> None:
            calls.append(f"node:{n.name}")

        def on_report(msg: str, *, node: object | None = None) -> None:
            calls.append(msg)

        root = DocumentFragment()
        root.append_child(Element("div", {}, "html"))
        root.append_child(Comment(data="x"))

        transforms = [
            Stage([DropComments()], callback=on_node, report=on_report),
            Stage([PruneEmpty("div", callback=on_node, report=on_report)]),
        ]
        apply_compiled_transforms(root, compile_transforms(transforms))

        assert root.children == []
        assert any(c.startswith("Stage ") for c in calls)
        assert any("Pruned empty" in c for c in calls)

    def test_drop_tag_list_fast_path_skips_comments_and_can_report(self) -> None:
        calls: list[str] = []

        def on_node(n: Node) -> None:
            calls.append(f"node:{n.name}")

        def on_report(msg: str, *, node: object | None = None) -> None:
            calls.append(msg)

        root = DocumentFragment()
        root.append_child(Comment(data="x"))
        root.append_child(Element("script", {}, "html"))

        apply_compiled_transforms(
            root, compile_transforms([Drop("script, style", callback=on_node, report=on_report)])
        )
        assert root.children is not None
        assert [c.name for c in root.children] == ["#comment"]
        assert "node:script" in calls
        assert any("Dropped tag 'script'" in c for c in calls)

    def test_drop_foreign_namespaces_skips_comment_and_doctype(self) -> None:
        calls: list[str] = []

        def on_node(n: Node) -> None:
            calls.append(str(n.name))

        def on_report(msg: str, *, node: object | None = None) -> None:
            calls.append(msg)

        root = DocumentFragment()
        root.append_child(Comment(data="x"))
        root.append_child(Node("!doctype"))
        root.append_child(Element("svg", {}, "svg"))

        apply_compiled_transforms(
            root, compile_transforms([DropForeignNamespaces(callback=on_node, report=on_report)])
        )
        assert root.children is not None
        assert [c.name for c in root.children] == ["#comment", "!doctype"]
        assert "svg" in calls
        assert any("foreign namespace" in c for c in calls)

    def test_policy_transforms_can_run_node_hook_without_reporting(self) -> None:
        seen: list[str] = []

        def on_node(n: Node) -> None:
            seen.append(str(n.name))

        root = DocumentFragment()
        div = Element("div", {"onclick": "x()", "bad": "y"}, "html")
        root.append_child(div)

        apply_compiled_transforms(
            root,
            compile_transforms(
                [
                    DropAttrs("*", patterns=("on*",), callback=on_node, report=None),
                    AllowlistAttrs("*", allowed_attributes={"*": set()}, callback=on_node, report=None),
                ]
            ),
        )
        assert div.attrs == {}
        assert seen == ["div", "div"]

    def test_dropurlattrs_and_allowstyleattrs_can_run_node_hook(self) -> None:
        seen: list[str] = []

        def on_node(n: Node) -> None:
            seen.append(str(n.name))

        url_policy = UrlPolicy(
            default_handling="allow",
            allow_rules={
                ("a", "href"): UrlRule(allowed_schemes={"http", "https"}),
            },
        )

        root = DocumentFragment()
        a = Element("a", {"href": "javascript:alert(1)"}, "html")
        a_ws = Element("a", {"href": " https://example.com "}, "html")
        s_none = Element("span", {"style": None}, "html")
        s_bad = Element("span", {"style": "position: fixed"}, "html")
        s_partial = Element("span", {"style": "color: red; position: fixed"}, "html")
        root.append_child(a)
        root.append_child(a_ws)
        root.append_child(s_none)
        root.append_child(s_bad)
        root.append_child(s_partial)

        apply_compiled_transforms(
            root,
            compile_transforms(
                [
                    DropUrlAttrs("*", url_policy=url_policy, callback=on_node, report=None),
                    AllowStyleAttrs("span", allowed_css_properties={"color"}, callback=on_node, report=None),
                ]
            ),
        )
        assert "href" not in a.attrs
        assert a_ws.attrs.get("href") == "https://example.com"
        assert "style" not in s_none.attrs
        assert "style" not in s_bad.attrs
        assert s_partial.attrs.get("style") == "color: red"
        assert seen == ["a", "a", "span", "span", "span"]

    def test_sanitize_can_forward_user_callback_and_report(self) -> None:
        events: list[str] = []

        def on_node(n: Node) -> None:
            events.append(f"node:{n.name}")

        def on_report(msg: str, *, node: object | None = None) -> None:
            events.append(msg)

        root = DocumentFragment()
        root.append_child(Element("script", {"onclick": "x()"}, "html"))
        root.append_child(Element("blink", {}, "html"))
        root.append_child(Element("p", {"onclick": "x()"}, "html"))

        apply_compiled_transforms(root, compile_transforms([Sanitize(callback=on_node, report=on_report)]))
        assert root.to_html(pretty=False) == "<p></p>"
        assert any(e.startswith("node:") for e in events)
        assert any("Unsafe tag" in e for e in events)
        assert any("Unsafe attribute" in e for e in events)

    def test_decide_unwrap_can_hoist_template_content(self) -> None:
        root = DocumentFragment()
        tpl = Template("template", attrs={}, namespace="html")
        assert tpl.template_content is not None
        tpl.template_content.append_child(Element("b", {}, "html"))
        root.append_child(tpl)

        apply_compiled_transforms(root, compile_transforms([Decide("template", lambda n: Decide.UNWRAP)]))
        assert root.to_html(pretty=False) == "<b></b>"

    def test_decide_escape_hoists_template_content(self) -> None:
        root = DocumentFragment()
        tpl = Template("template", attrs={}, namespace="html")
        assert tpl.template_content is not None
        tpl.template_content.append_child(Element("b", {}, "html"))
        root.append_child(tpl)

        apply_compiled_transforms(root, compile_transforms([Decide("template", lambda n: Decide.ESCAPE)]))
        assert root.to_html(pretty=False) == "&lt;template&gt;<b></b>"

    def test_empty_and_drop_selector_hooks(self) -> None:
        calls: list[str] = []

        def on_node(n: Node) -> None:
            calls.append(str(n.name))

        def on_report(msg: str, *, node: object | None = None) -> None:
            calls.append(msg)

        root = DocumentFragment()
        div = Element("div", {}, "html")
        div.append_child(Text("x"))
        root.append_child(div)
        root.append_child(Element("div", {}, "html"))
        root.append_child(Element("p", {"class": "x"}, "html"))
        root.append_child(Element("p", {"class": "y"}, "html"))

        apply_compiled_transforms(
            root,
            compile_transforms(
                [
                    Empty("div", callback=on_node, report=on_report),
                    Drop("p.x", callback=on_node, report=on_report),
                    Drop("p.y", report=on_report),
                ]
            ),
        )
        assert root.to_html(pretty=False) == "<div></div><div></div>"
        assert "div" in calls
        assert any("Emptied" in c for c in calls)
        assert any("Dropped" in c for c in calls)

    def test_drop_foreign_namespaces_can_report_to_policy(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["p"],
            allowed_attributes={"*": []},
            unsafe_handling="collect",
        )
        policy.reset_collected_security_errors()

        root = DocumentFragment()
        root.append_child(Element("svg", {}, "svg"))

        apply_compiled_transforms(root, compile_transforms([DropForeignNamespaces(report=policy.handle_unsafe)]))
        assert root.children == []
        assert policy.collected_security_errors()

    def test_drop_foreign_namespaces_drops_even_without_policy(self) -> None:
        root = DocumentFragment()
        root.append_child(Element("svg", {}, "svg"))

        apply_compiled_transforms(root, compile_transforms([DropForeignNamespaces(report=None)]))
        assert root.children == []

    def test_dropattrs_patterns_cover_event_namespaced_and_exact(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["div"],
            allowed_attributes={"*": []},
            unsafe_handling="collect",
        )
        policy.reset_collected_security_errors()

        root = DocumentFragment()
        node = Element(
            "div",
            {
                "onClick": "1",
                "xml:lang": "sv",
                "srcdoc": "<p>x</p>",
                "href": "https://example.com/",
                " ": "ignored",
            },
            "html",
        )
        root.append_child(node)

        apply_compiled_transforms(
            root,
            compile_transforms(
                [
                    DropAttrs(
                        "*",
                        patterns=("on*", "*:*", "srcdoc", "href"),
                        report=policy.handle_unsafe,
                    )
                ]
            ),
        )
        assert node.attrs == {}
        assert len(policy.collected_security_errors()) == 4

    def test_dropattrs_can_be_disabled(self) -> None:
        root = DocumentFragment()
        node = Element("div", {"onclick": "1"}, "html")
        root.append_child(node)

        apply_compiled_transforms(root, compile_transforms([DropAttrs("*", patterns=("on*",), enabled=False)]))
        assert node.attrs == {"onclick": "1"}

    def test_dropattrs_with_no_policy_still_drops(self) -> None:
        root = DocumentFragment()
        node = Element("div", {"onClick": "1", "xml:lang": "sv", "srcdoc": "x"}, "html")
        root.append_child(node)

        apply_compiled_transforms(
            root,
            compile_transforms([DropAttrs("*", patterns=("on*", "*:*", "srcdoc"), report=None)]),
        )
        assert node.attrs == {}

    def test_allowlistattrs_lowercases_keys_skips_blank_and_reports_disallowed(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["a"],
            allowed_attributes={"*": [], "a": ["href"]},
            force_link_rel={"noopener"},
            unsafe_handling="collect",
        )
        policy.reset_collected_security_errors()

        root = DocumentFragment()
        a = Element(
            "a",
            {
                "HREF": "https://example.com",
                "Rel": "noreferrer",
                "BAD": "x",
                " ": "ignored",
            },
            "html",
        )
        root.append_child(a)

        apply_compiled_transforms(
            root,
            compile_transforms(
                [
                    AllowlistAttrs(
                        "*",
                        allowed_attributes={"*": [], "a": ["href", "rel"]},
                        report=policy.handle_unsafe,
                    )
                ]
            ),
        )
        assert a.attrs.get("href") == "https://example.com"
        assert a.attrs.get("rel") == "noreferrer"
        assert "bad" not in a.attrs
        assert policy.collected_security_errors()

    def test_allowlistattrs_can_be_disabled(self) -> None:
        root = DocumentFragment()
        a = Element("a", {"href": "https://example.com", "bad": "x"}, "html")
        root.append_child(a)

        apply_compiled_transforms(
            root,
            compile_transforms([AllowlistAttrs("*", allowed_attributes={"*": [], "a": ["href"]}, enabled=False)]),
        )
        assert a.attrs == {"href": "https://example.com", "bad": "x"}

    def test_allowlistattrs_without_policy_drops_without_reporting(self) -> None:
        root = DocumentFragment()
        a = Element("a", {"href": "https://example.com", "bad": "x"}, "html")
        root.append_child(a)

        apply_compiled_transforms(
            root,
            compile_transforms(
                [
                    AllowlistAttrs(
                        "*",
                        allowed_attributes={"*": [], "a": ["href"]},
                        report=None,
                    )
                ],
            ),
        )
        assert a.attrs == {"href": "https://example.com"}

    def test_dropurlattrs_branches_raw_none_no_rule_and_invalid_url(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["a", "img"],
            allowed_attributes={"*": [], "a": ["href"], "img": ["src"]},
            url_policy=UrlPolicy(
                default_handling="allow",
                allow_rules={
                    ("a", "href"): UrlRule(allowed_schemes={"http", "https"}),
                },
            ),
            unsafe_handling="collect",
        )
        policy.reset_collected_security_errors()

        root = DocumentFragment()
        a_none = Element("a", {"href": None}, "html")
        img_no_rule = Element("img", {"src": "https://example.com/x.png"}, "html")
        a_bad = Element("a", {"href": "javascript:alert(1)"}, "html")
        root.append_child(a_none)
        root.append_child(img_no_rule)
        root.append_child(a_bad)

        apply_compiled_transforms(
            root,
            compile_transforms([DropUrlAttrs("*", url_policy=policy.url_policy, report=policy.handle_unsafe)]),
        )
        assert "href" not in a_none.attrs
        assert "src" not in img_no_rule.attrs
        assert "href" not in a_bad.attrs
        assert len(policy.collected_security_errors()) == 3

    def test_dropurlattrs_works_without_on_unsafe_callback(self) -> None:
        url_policy = UrlPolicy(
            default_handling="allow",
            allow_rules={
                ("a", "href"): UrlRule(allowed_schemes={"http", "https"}),
            },
        )

        root = DocumentFragment()
        a_none = Element("a", {"href": None}, "html")
        img_no_rule = Element("img", {"src": "https://example.com/x.png"}, "html")
        a_bad = Element("a", {"href": "javascript:alert(1)"}, "html")
        root.append_child(a_none)
        root.append_child(img_no_rule)
        root.append_child(a_bad)

        apply_compiled_transforms(root, compile_transforms([DropUrlAttrs("*", url_policy=url_policy)]))
        assert "href" not in a_none.attrs
        assert "src" not in img_no_rule.attrs
        assert "href" not in a_bad.attrs

    def test_dropurlattrs_allows_valid_srcset(self) -> None:
        url_policy = UrlPolicy(
            default_handling="allow",
            allow_rules={
                ("img", "srcset"): UrlRule(allowed_schemes={"https"}),
            },
        )

        root = DocumentFragment()
        img = Element("img", {"srcset": "https://example.com/a 1x"}, "html")
        root.append_child(img)

        apply_compiled_transforms(root, compile_transforms([DropUrlAttrs("*", url_policy=url_policy)]))
        assert img.attrs.get("srcset") == "https://example.com/a 1x"

    def test_dropurlattrs_can_be_disabled(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["a"],
            allowed_attributes={"*": [], "a": ["href"]},
            url_policy=UrlPolicy(allow_rules={("a", "href"): UrlRule(allowed_schemes={"http", "https"})}),
            unsafe_handling="collect",
        )
        policy.reset_collected_security_errors()

        doc = JustHTML(
            '<a href="javascript:alert(1)">x</a>',
            fragment=True,
            transforms=[DropUrlAttrs("*", url_policy=policy.url_policy, enabled=False, report=policy.handle_unsafe)],
        )
        assert doc.to_html(pretty=False) == '<a href="javascript:alert(1)">x</a>'
        assert policy.collected_security_errors() == []

    def test_allowstyleattrs_branches_raw_none_and_sanitized_none(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["span"],
            allowed_attributes={"*": ["style"]},
            allowed_css_properties={"color"},
            unsafe_handling="collect",
        )
        policy.reset_collected_security_errors()

        root = DocumentFragment()
        s_none = Element("span", {"style": None}, "html")
        s_bad = Element("span", {"style": "position: fixed"}, "html")
        s_ok = Element("span", {"style": "color: red; position: fixed"}, "html")
        s_no_style = Element("span", {}, "html")
        root.append_child(s_none)
        root.append_child(s_bad)
        root.append_child(s_ok)
        root.append_child(s_no_style)

        apply_compiled_transforms(
            root,
            compile_transforms(
                [
                    AllowStyleAttrs(
                        "span",
                        allowed_css_properties=policy.allowed_css_properties,
                        report=policy.handle_unsafe,
                    )
                ]
            ),
        )
        assert "style" not in s_none.attrs
        assert "style" not in s_bad.attrs
        assert s_ok.attrs.get("style") == "color: red"
        assert s_no_style.attrs == {}
        assert len(policy.collected_security_errors()) == 2

    def test_allowstyleattrs_works_without_on_unsafe_callback(self) -> None:
        root = DocumentFragment()
        s_none = Element("span", {"style": None}, "html")
        s_bad = Element("span", {"style": "position: fixed"}, "html")
        s_ok = Element("span", {"style": "color: red"}, "html")
        root.append_child(s_none)
        root.append_child(s_bad)
        root.append_child(s_ok)

        apply_compiled_transforms(
            root,
            compile_transforms([AllowStyleAttrs("span", allowed_css_properties={"color"})]),
        )
        assert "style" not in s_none.attrs
        assert "style" not in s_bad.attrs
        assert s_ok.attrs.get("style") == "color: red"

    def test_allowstyleattrs_can_be_disabled(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["span"],
            allowed_attributes={"*": ["style"]},
            allowed_css_properties={"color"},
            unsafe_handling="collect",
        )
        policy.reset_collected_security_errors()

        doc = JustHTML(
            '<span style="position: fixed">x</span>',
            fragment=True,
            transforms=[
                AllowStyleAttrs(
                    "[style]",
                    allowed_css_properties=policy.allowed_css_properties,
                    enabled=False,
                    report=policy.handle_unsafe,
                )
            ],
        )
        assert doc.to_html(pretty=False) == '<span style="position: fixed">x</span>'
        assert policy.collected_security_errors() == []

    def test_mergeattrs_rewrites_on_add_missing_and_normalization(self) -> None:
        doc = JustHTML(
            '<a></a><a rel="NoOpEnEr noopener"></a><a rel="noreferrer"></a><a rel="noopener"></a>',
            fragment=True,
            transforms=[MergeAttrs("a", attr="rel", tokens={"noopener"})],
        )
        assert (
            doc.to_html(pretty=False)
            == '<a rel="noopener"></a><a rel="noopener"></a><a rel="noreferrer noopener"></a><a rel="noopener"></a>'
        )

    def test_mergeattrs_skips_non_matching_elements(self) -> None:
        doc = JustHTML(
            "<div></div><a></a>",
            fragment=True,
            transforms=[MergeAttrs("a", attr="rel", tokens={"noopener"})],
        )
        assert doc.to_html(pretty=False) == '<div></div><a rel="noopener"></a>'

    def test_mergeattrs_is_skipped_if_no_tokens(self) -> None:
        compiled = compile_transforms([MergeAttrs("a", attr="rel", tokens=set())])
        assert compiled == []

    def test_dropattrs_noops_when_patterns_empty(self) -> None:
        root = DocumentFragment()
        node = Element("div", {"id": "x"}, "html")
        root.append_child(node)

        apply_compiled_transforms(root, compile_transforms([DropAttrs("*", patterns=())]))
        assert node.attrs == {"id": "x"}

    def test_disabled_top_level_stage_is_skipped(self) -> None:
        # Ensure disabled stages are skipped both when flattening and when
        # splitting into top-level stages.
        doc = JustHTML(
            "<p>Hello</p>",
            fragment=True,
            transforms=[
                Stage([SetAttrs("p", id="x")], enabled=False),
                Stage([SetAttrs("p", **{"class": "y"})]),
            ],
        )
        html = doc.to_html(pretty=False)
        assert 'id="x"' not in html
        assert 'class="y"' in html

    def test_apply_compiled_transforms_empty_list_noops(self) -> None:
        root = DocumentFragment()
        root.append_child(Element("p", {}, "html"))
        apply_compiled_transforms(root, [])

    def test_selector_transforms_skip_comment_nodes(self) -> None:
        doc = JustHTML("<!--x--><p>y</p>", transforms=[SetAttrs("p", id="x")])
        assert '<p id="x">y</p>' in doc.to_html(pretty=False)

    def test_decide_star_can_drop_comment_nodes(self) -> None:
        def decide(node: object) -> DecideAction:
            name = getattr(node, "name", "")
            if name == "#comment":
                return Decide.DROP
            return Decide.KEEP

        doc = JustHTML("<!--x--><p>y</p>", fragment=True, transforms=[Decide("*", decide)])
        assert doc.to_html(pretty=False) == "<p>y</p>"

    def test_decide_selector_only_runs_on_elements(self) -> None:
        seen: list[str] = []

        def decide(node: object) -> DecideAction:
            name = getattr(node, "name", "")
            # Decide("p", ...) should never be called for non-elements.
            assert not str(name).startswith("#")
            seen.append(str(name))
            return Decide.DROP

        doc = JustHTML("<!--x--><p>y</p>", fragment=True, transforms=[Decide("p", decide)])
        assert doc.to_html(pretty=False) == "<!--x-->"
        assert seen == ["p"]

    def test_decide_empty_clears_template_content(self) -> None:
        def decide(node: object) -> DecideAction:
            if getattr(node, "name", "") == "template":
                return Decide.EMPTY
            return Decide.KEEP

        doc = JustHTML("<template><b>x</b></template>", fragment=True, transforms=[Decide("*", decide)])
        assert doc.to_html(pretty=False) == "<template></template>"

    def test_decide_empty_clears_element_children(self) -> None:
        doc = JustHTML(
            "<div><span>x</span>y</div>",
            fragment=True,
            transforms=[Decide("div", lambda n: Decide.EMPTY)],
        )
        assert doc.to_html(pretty=False) == "<div></div>"

    def test_decide_unwrap_hoists_template_content(self) -> None:
        doc = JustHTML(
            "<div><template><b>x</b></template>y</div>",
            fragment=True,
            transforms=[Decide("template", lambda n: Decide.UNWRAP)],
        )
        assert doc.to_html(pretty=False) == "<div><b>x</b>y</div>"

    def test_decide_unwrap_hoists_element_children(self) -> None:
        doc = JustHTML(
            "<div><span><b>x</b></span>y</div>",
            fragment=True,
            transforms=[Decide("span", lambda n: Decide.UNWRAP)],
        )
        assert doc.to_html(pretty=False) == "<div><b>x</b>y</div>"

    def test_decide_unwrap_with_no_children_still_removes_node(self) -> None:
        doc = JustHTML(
            "<div><span></span>ok</div><div><template></template>y</div>",
            fragment=True,
            transforms=[Decide("span, template", lambda n: Decide.UNWRAP)],
        )
        assert doc.to_html(pretty=False) == "<div>ok</div><div>y</div>"

    def test_rewriteattrs_can_replace_attribute_dict(self) -> None:
        def rewrite(node: Node) -> dict[str, str | None] | None:
            assert node.name == "a"
            return {"href": node.attrs.get("href"), "data-ok": "1"}

        doc = JustHTML('<a href="x" onclick="y">t</a>', fragment=True, transforms=[EditAttrs("a", rewrite)])
        assert doc.to_html(pretty=False) == '<a href="x" data-ok="1">t</a>'

    def test_rewriteattrs_returning_none_noops(self) -> None:
        doc = JustHTML('<a href="x">t</a>', fragment=True, transforms=[EditAttrs("a", lambda n: None)])
        assert doc.to_html(pretty=False) == '<a href="x">t</a>'

    def test_rewriteattrs_skips_non_matching_elements(self) -> None:
        doc = JustHTML("<p>t</p>", fragment=True, transforms=[EditAttrs("a", lambda n: {"x": "1"})])
        assert doc.to_html(pretty=False) == "<p>t</p>"

    def test_walk_transforms_traverse_nested_document_containers(self) -> None:
        root = DocumentFragment()
        nested = DocumentFragment()
        nested.append_child(Node("p"))
        root.append_child(nested)

        apply_compiled_transforms(root, compile_transforms([SetAttrs("p", id="x")]))
        assert root.to_html(pretty=False) == '<p id="x"></p>'

    def test_apply_compiled_transforms_handles_empty_root(self) -> None:
        root = Node("div")
        apply_compiled_transforms(root, compile_transforms([SetAttrs("div", id="x")]))
        assert root.to_html(pretty=False) == "<div></div>"

    def test_apply_compiled_transforms_noops_with_no_transforms(self) -> None:
        root = Node("div")
        apply_compiled_transforms(root, [])
        assert root.to_html(pretty=False) == "<div></div>"

    def test_apply_compiled_transforms_supports_text_root(self) -> None:
        root = Text("example.com")
        apply_compiled_transforms(root, compile_transforms([Linkify()]))  # type: ignore[arg-type]
        assert root.data == "example.com"

    def test_apply_compiled_transforms_rejects_unknown_compiled_transform(self) -> None:
        root = Node("div")
        with self.assertRaises(TypeError):
            apply_compiled_transforms(root, [object()])  # type: ignore[list-item]

    def test_transforms_can_run_after_sanitize(self) -> None:
        doc = JustHTML(
            "<p>x</p>",
            fragment=True,
            transforms=[Sanitize(), SetAttrs("p", **{"class": "y"})],
        )
        assert doc.to_html(pretty=False) == '<p class="y">x</p>'

    def test_sanitize_root_comment_and_doctype_keep(self) -> None:
        policy_keep = SanitizationPolicy(
            allowed_tags=[],
            allowed_attributes={"*": []},
            url_policy=UrlPolicy(allow_rules={}),
            drop_comments=False,
            drop_doctype=False,
        )

        compiled = compile_transforms([Sanitize(policy_keep)])

        c = Comment(data="x")
        apply_compiled_transforms(c, compiled)
        assert c.to_html(pretty=False) == "<!--x-->"

        d = Node("!doctype", data="html")
        apply_compiled_transforms(d, compiled)
        assert d.to_html(pretty=False) == "<!DOCTYPE html>"

    def test_collapsewhitespace_collapses_text_nodes(self) -> None:
        doc = JustHTML(
            "<p>Hello \n\t world</p><p>a  b</p>",
            fragment=True,
            transforms=[CollapseWhitespace()],
        )
        assert doc.to_html(pretty=False) == "<p>Hello world</p><p>a b</p>"

    def test_collapsewhitespace_skips_pre_by_default(self) -> None:
        doc = JustHTML(
            "<pre>a  b</pre><p>a  b</p>",
            fragment=True,
            transforms=[CollapseWhitespace()],
        )
        assert doc.to_html(pretty=False) == "<pre>a  b</pre><p>a b</p>"

    def test_collapsewhitespace_noops_when_no_collapse_needed(self) -> None:
        doc = JustHTML(
            "<p>Hello world</p>",
            fragment=True,
            transforms=[CollapseWhitespace()],
        )
        assert doc.to_html(pretty=False) == "<p>Hello world</p>"

    def test_collapsewhitespace_can_skip_custom_tags(self) -> None:
        doc = JustHTML(
            "<p>a  b</p>",
            fragment=True,
            transforms=[CollapseWhitespace(skip_tags=("p",))],
        )
        assert doc.to_html(pretty=False) == "<p>a  b</p>"

    def test_collapsewhitespace_ignores_empty_text_nodes(self) -> None:
        root = Node("div")
        root.append_child(Text(""))
        apply_compiled_transforms(root, compile_transforms([CollapseWhitespace()]))
        assert root.to_html(pretty=False) == "<div></div>"

    def test_to_html_still_sanitizes_by_default_after_transforms_and_mutation(self) -> None:
        doc = _JustHTML("<p>ok</p>")
        # Mutate the tree after parse.
        doc.root.append_child(Node("script"))
        # Sanitization happens at construction time; later mutations are not re-sanitized.
        assert "<script" in doc.to_html(pretty=False)

    def test_pruneempty_drops_empty_elements(self) -> None:
        doc = JustHTML(
            "<p></p><p><img></p><p>   </p>",
            fragment=True,
            transforms=[PruneEmpty("p")],
        )
        assert doc.to_html(pretty=False) == "<p><img></p>"

    def test_pruneempty_is_recursive_post_order(self) -> None:
        doc = JustHTML(
            "<div><p></p></div>",
            fragment=True,
            transforms=[PruneEmpty("p, div")],
        )
        assert doc.to_html(pretty=False) == ""

    def test_pruneempty_drops_nested_empty_elements(self) -> None:
        doc = JustHTML(
            "<span><span></span></span>",
            fragment=True,
            transforms=[PruneEmpty("span")],
        )
        assert doc.to_html(pretty=False) == ""

    def test_pruneempty_supports_consecutive_prune_transforms(self) -> None:
        doc = JustHTML(
            "<div><p></p></div>",
            fragment=True,
            transforms=[PruneEmpty("p"), PruneEmpty("div")],
        )
        assert doc.to_html(pretty=False) == ""

    def test_pruneempty_can_run_before_other_transforms(self) -> None:
        # If pruning runs before later transforms, it only prunes emptiness at
        # that point in the pipeline.
        doc = JustHTML(
            "<p></p><p><img></p>",
            fragment=True,
            transforms=[PruneEmpty("p"), Drop("img")],
        )
        assert doc.to_html(pretty=False) == "<p></p>"

    def test_pruneempty_ignores_comments_when_determining_emptiness(self) -> None:
        doc = JustHTML(
            "<p><!--x--></p>",
            fragment=True,
            transforms=[PruneEmpty("p")],
        )
        assert doc.to_html(pretty=False) == ""

    def test_pruneempty_can_preserve_whitespace_only_text(self) -> None:
        doc = JustHTML(
            "<p>   </p>",
            fragment=True,
            transforms=[PruneEmpty("p", strip_whitespace=False)],
        )
        assert doc.to_html(pretty=False) == "<p>   </p>"

    def test_pruneempty_does_not_prune_void_elements(self) -> None:
        doc = JustHTML(
            '<img src="/static/images/icons/wikipedia.png" alt height="50" width="50">',
            fragment=True,
            transforms=[PruneEmpty("*")],
        )
        assert doc.to_html(pretty=False) == (
            '<img src="/static/images/icons/wikipedia.png" alt height="50" width="50">'
        )

    def test_pruneempty_strip_whitespace_false_still_drops_empty_text_nodes(self) -> None:
        root = Node("div")
        p = Node("p")
        p.append_child(Text(""))
        root.append_child(p)

        apply_compiled_transforms(root, compile_transforms([PruneEmpty("p", strip_whitespace=False)]))
        assert root.to_html(pretty=False) == "<div></div>"

    def test_pruneempty_considers_template_content(self) -> None:
        doc = JustHTML(
            "<template>ok</template><template><p></p></template>",
            fragment=True,
            transforms=[PruneEmpty("p, template")],
        )
        assert doc.to_html(pretty=False) == "<template>ok</template>"

    def test_transform_order_is_respected_for_linkify_and_drop(self) -> None:
        # Drop runs before Linkify: it should not remove links created later.
        doc_keep = JustHTML(
            "<p>example.com</p>",
            fragment=True,
            transforms=[Drop("a"), Linkify()],
        )
        assert doc_keep.to_html(pretty=False) == '<p><a href="http://example.com">example.com</a></p>'

        # Drop runs after Linkify: it should remove the linkified <a>.
        doc_drop = JustHTML(
            "<p>example.com</p>",
            fragment=True,
            transforms=[Linkify(), Drop("a")],
        )
        assert doc_drop.to_html(pretty=False) == "<p></p>"

    def test_stage_auto_grouping_does_not_change_ordering(self) -> None:
        # Stage boundaries split passes, but ordering semantics are preserved.
        doc_stage = JustHTML(
            "<p>example.com</p>",
            fragment=True,
            transforms=[Drop("a"), Stage([Linkify()])],
        )
        assert doc_stage.to_html(pretty=False) == '<p><a href="http://example.com">example.com</a></p>'

    def test_stage_can_be_nested_and_is_flattened(self) -> None:
        doc = JustHTML(
            "<p>example.com</p>",
            fragment=True,
            transforms=[Stage([Stage([Linkify()])])],
        )
        assert doc.to_html(pretty=False) == '<p><a href="http://example.com">example.com</a></p>'

    def test_stage_auto_grouping_includes_trailing_transforms(self) -> None:
        # When a Stage exists at the top level, transforms outside stages are
        # implicitly grouped into stages too (including a trailing segment).
        doc = JustHTML(
            "<p>Hello</p>",
            fragment=True,
            transforms=[Stage([SetAttrs("p", id="x")]), SetAttrs("p", **{"class": "y"})],
        )
        html = doc.to_html(pretty=False)
        assert "<p" in html
        assert 'id="x"' in html
        assert 'class="y"' in html

    def test_linkify_noops_when_no_links_found(self) -> None:
        # Covers the linkify path where we scan text but find no matches.
        doc = JustHTML(
            "<p>Hello world</p>",
            fragment=True,
            transforms=[Linkify()],
        )
        assert doc.to_html(pretty=False) == "<p>Hello world</p>"
