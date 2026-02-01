import unittest

from justhtml.node import Document, DocumentFragment, Element, Template, Text
from justhtml.sanitize import DEFAULT_POLICY
from justhtml.selector import parse_selector
from justhtml.transforms import (
    AllowlistAttrs,
    Decide,
    DecideAction,
    DropAttrs,
    DropUrlAttrs,
    _CompiledDecideChain,
    _CompiledDecideElementsChain,
    _CompiledRewriteAttrsChain,
    apply_compiled_transforms,
    compile_transforms,
)


class TestTransformsCompiler(unittest.TestCase):
    def test_compile_transforms_fuses_rewrite_attrs_chain(self) -> None:
        compiled = compile_transforms(
            [
                DropAttrs(selector="*", patterns=("id",)),
                DropAttrs(selector="*", patterns=("class",)),
                DropAttrs(selector="*", patterns=("title",)),
            ]
        )
        assert len(compiled) == 1
        assert isinstance(compiled[0], _CompiledRewriteAttrsChain)
        assert len(compiled[0].funcs) == 3

        compiled_mixed = compile_transforms(
            [
                DropAttrs(selector="*", patterns=("id",)),
                DropAttrs(selector="*", patterns=("class",)),
                DropAttrs(selector="a", patterns=("title",)),
            ]
        )
        assert len(compiled_mixed) == 2
        assert isinstance(compiled_mixed[0], _CompiledRewriteAttrsChain)

    def test_compile_transforms_fuses_decide_chain(self) -> None:
        def _keep(_node):
            return DecideAction.KEEP

        compiled = compile_transforms(
            [
                Decide("*", _keep),
                Decide("*", _keep),
                Decide("*", _keep),
            ]
        )
        assert len(compiled) == 1
        assert isinstance(compiled[0], _CompiledDecideChain)
        assert len(compiled[0].callbacks) == 3

        compiled_mixed = compile_transforms([Decide("*", _keep), Decide("*", _keep), Decide("div", _keep)])
        assert len(compiled_mixed) == 2
        assert isinstance(compiled_mixed[0], _CompiledDecideChain)

        compiled_no_fuse = compile_transforms([Decide("*", _keep), Decide("div", _keep)])
        assert len(compiled_no_fuse) == 2

    def test_compiled_decide_wrapper_skips_callbacks_for_keep_and_calls_for_drop(self) -> None:
        seen: list[str] = []

        def decide(node):
            if str(node.name).lower() == "keep":
                return DecideAction.KEEP
            return DecideAction.DROP

        def hook(_node):
            seen.append("hook")

        def report(_msg: str, *, node=None):
            _ = node
            seen.append("report")

        compiled = compile_transforms([Decide("*", decide, callback=hook, report=report)])
        assert len(compiled) == 1
        t = compiled[0]
        assert getattr(t, "kind", None) == "decide"

        keep_node = Element("keep", {}, "html")
        action = t.callback(keep_node)  # type: ignore[attr-defined]
        assert action is DecideAction.KEEP
        assert seen == []

        drop_node = Element("drop", {}, "html")
        action = t.callback(drop_node)  # type: ignore[attr-defined]
        assert action is DecideAction.DROP
        assert seen == ["hook", "report"]

        seen.clear()

        compiled2 = compile_transforms([Decide("*", lambda _n: DecideAction.DROP, report=report)])
        t2 = compiled2[0]
        _ = t2.callback(Element("drop", {}, "html"))  # type: ignore[attr-defined]
        assert seen == ["report"]

        seen.clear()

        compiled3 = compile_transforms([Decide("*", lambda _n: DecideAction.DROP, callback=hook)])
        t3 = compiled3[0]
        _ = t3.callback(Element("drop", {}, "html"))  # type: ignore[attr-defined]
        assert seen == ["hook"]

    def test_rewrite_attrs_drop_allowlist_and_drop_url(self) -> None:
        root = Document()
        node = Element(
            "a",
            {
                "onclick": "x",
                "srcdoc": "y",
                "data:bad": "1",
                "href": "//example.com",
                "id": "ok",
            },
            "html",
        )
        root.append_child(node)

        compiled = compile_transforms(
            [
                DropAttrs(selector="*", patterns=("on*", "srcdoc", "*:*")),
                AllowlistAttrs(selector="*", allowed_attributes={"*": {"id"}, "a": {"href"}}),
                DropUrlAttrs(selector="*", url_policy=DEFAULT_POLICY.url_policy),
            ]
        )
        apply_compiled_transforms(root, compiled)

        assert node.attrs == {"href": "https://example.com", "id": "ok"}

    def test_drop_attrs_generic_path_skips_empty_attribute_names(self) -> None:
        root = Document()
        node = Element("div", {"": "x", "id": "y"}, "html")
        root.append_child(node)

        compiled = compile_transforms([DropAttrs(selector="*", patterns=("id",))])
        apply_compiled_transforms(root, compiled)

        assert node.attrs == {}

    def test_allowlist_attrs_defensive_paths(self) -> None:
        class _MyStr(str):
            __slots__ = ()

        root = Document()
        node = Element("DIV", {_MyStr("id"): "ok"}, "html")
        root.append_child(node)

        compiled = compile_transforms([AllowlistAttrs(selector="*", allowed_attributes={"div": {"id"}})])
        apply_compiled_transforms(root, compiled)

        assert node.attrs == {_MyStr("id"): "ok"}

    def test_drop_url_attrs_initializes_tag_once_and_lowercases(self) -> None:
        root = Document()
        node = Element("A", {"href": "//example.com", "src": "//example.com/x"}, "html")
        root.append_child(node)

        compiled = compile_transforms([DropUrlAttrs(selector="*", url_policy=DEFAULT_POLICY.url_policy)])
        apply_compiled_transforms(root, compiled)

        assert node.attrs.get("href") == "https://example.com"

    def test_apply_compiled_transforms_decide_chain_actions(self) -> None:
        root = Document()
        parent = Element("div", {}, "html")
        root.append_child(parent)

        emptyme = Element("emptyme", {}, "html")
        emptyme.append_child(Element("child", {}, "html"))
        parent.append_child(emptyme)

        emptynochild = Element("emptynochild", {}, "html")
        parent.append_child(emptynochild)

        unwrapme = Element("unwrapme", {}, "html")
        unwrapme.append_child(Text("x"))
        parent.append_child(unwrapme)

        unwrapempty = Element("unwrapempty", {}, "html")
        parent.append_child(unwrapempty)

        tmpl_empty = Template("tmpl_empty", {}, None, "html")
        tmpl_empty.append_child(Element("span", {}, "html"))
        assert tmpl_empty.template_content is not None
        tmpl_empty.template_content.append_child(Element("i", {}, "html"))
        parent.append_child(tmpl_empty)

        tmpl_unwrap = Template("tmpl_unwrap", {}, None, "html")
        tmpl_unwrap.append_child(Element("b", {}, "html"))
        assert tmpl_unwrap.template_content is not None
        tmpl_unwrap.template_content.append_child(Element("u", {}, "html"))
        parent.append_child(tmpl_unwrap)

        tmpl_unwrap_empty_tc = Template("tmpl_unwrap_empty_tc", {}, None, "html")
        tmpl_unwrap_empty_tc.append_child(Element("b", {}, "html"))
        parent.append_child(tmpl_unwrap_empty_tc)

        escapeme = Element("escapeme", {"id": "1"}, "html")
        escapeme.append_child(Text("y"))
        parent.append_child(escapeme)

        dropme = Element("dropme", {}, "html")
        parent.append_child(dropme)

        selbreak = Element("selbreak", {}, "html")
        parent.append_child(selbreak)

        parent.append_child(Text("t"))

        def sel_keep(_node):
            return DecideAction.KEEP

        def sel_drop(_node):
            return DecideAction.DROP

        def decide_all_nodes(node):
            nm = str(node.name)
            if nm == "emptyme":
                return DecideAction.EMPTY
            if nm == "emptynochild":
                return DecideAction.EMPTY
            if nm == "unwrapme":
                return DecideAction.UNWRAP
            if nm == "unwrapempty":
                return DecideAction.UNWRAP
            if nm == "tmpl_empty":
                return DecideAction.EMPTY
            if nm == "tmpl_unwrap":
                return DecideAction.UNWRAP
            if nm == "tmpl_unwrap_empty_tc":
                return DecideAction.UNWRAP
            if nm == "escapeme":
                return DecideAction.ESCAPE
            if nm == "dropme":
                return DecideAction.DROP
            return DecideAction.KEEP

        compiled = compile_transforms(
            [
                Decide("selbreak", sel_keep),
                Decide("selbreak", sel_drop),
                Decide("*", decide_all_nodes),
                Decide("*", decide_all_nodes),
            ]
        )
        apply_compiled_transforms(root, compiled)

        assert emptyme.children == []
        assert not parent.query("unwrapme")
        assert not parent.query("dropme")
        assert not parent.query("escapeme")
        texts = [n for n in parent.children or [] if getattr(n, "name", None) == "#text"]
        assert any(isinstance(t.data, str) and "<escapeme" in t.data for t in texts)

    def test_apply_compiled_transforms_decide_elements_chain_empty(self) -> None:
        root = Document()
        parent = Element("div", {}, "html")
        root.append_child(parent)

        template = Template("template", {}, None, "html")
        template.append_child(Element("span", {}, "html"))
        assert isinstance(template.template_content, DocumentFragment)
        template.template_content.append_child(Element("i", {}, "html"))
        parent.append_child(template)

        empty_element = Element("span", {}, "html")
        parent.append_child(empty_element)

        def empty_templates(node):
            if type(node) is Template:
                return DecideAction.EMPTY
            if str(node.name) == "span":
                return DecideAction.EMPTY
            return DecideAction.KEEP

        compiled = [_CompiledDecideElementsChain(callbacks=[empty_templates])]
        apply_compiled_transforms(root, compiled)

        assert template.children == []
        assert template.template_content is not None
        assert template.template_content.children == []

    def test_apply_compiled_transforms_decide_chain_empty_callbacks(self) -> None:
        root = Document()
        root.append_child(Element("div", {}, "html"))

        compiled = [
            _CompiledDecideChain(
                selector_str="div",
                selector=parse_selector("div"),
                all_nodes=False,
                callbacks=[],
            )
        ]
        apply_compiled_transforms(root, compiled)
