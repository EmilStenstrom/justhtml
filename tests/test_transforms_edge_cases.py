import unittest

from justhtml import JustHTML, SanitizationPolicy, UrlRule, to_html
from justhtml.node import Element, Text
from justhtml.transforms import (
    AllowlistAttrs,
    AllowStyleAttrs,
    Decide,
    DecideAction,
    DropAttrs,
    DropForeignNamespaces,
    DropUrlAttrs,
    Edit,
    EditAttrs,
    EditDocument,
    Empty,
    Sanitize,
    Unwrap,
    UrlPolicy,
    apply_compiled_transforms,
    compile_transforms,
)


class TestTransformsEdgeCases(unittest.TestCase):
    def test_chained_edit_attrs_coverage(self):
        """
        Cover the chaining optimization branch in compile_transforms calling both
        callbacks where both return changes.
        """

        def cb1(node):
            return {"data-step": "1"}

        def cb2(node):
            attrs = node.attrs.copy()
            attrs["data-step"] += "-2"
            return attrs

        transforms = [EditAttrs("div", cb1), EditAttrs("div", cb2)]

        html = "<div></div>"
        processor = JustHTML(html, transforms=transforms, safe=False)
        result = to_html(processor.root)
        self.assertIn('data-step="1-2"', result)

    def test_chained_edit_attrs_second_returns_none(self):
        """Cover case where chained next_cb returns None."""

        def cb1(node):
            return {"a": "1"}

        def cb2(node):
            return None

        transforms = [EditAttrs("div", cb1), EditAttrs("div", cb2)]
        html = "<div></div>"
        processor = JustHTML(html, transforms=transforms, safe=False)
        result = to_html(processor.root)
        self.assertIn('a="1"', result)

    def test_wrappers_execution(self):
        """
        Cover wrapper functions in compile_transforms hooks via JustHTML/manual exec.
        """
        called_hook = False

        def hook(node):
            nonlocal called_hook
            called_hook = True

        reported = []

        def report(msg, node):
            reported.append(msg)

        # 1. Decide wrapper (Must ensure DROP to trigger hook, KEEP skips hook)
        def decide_drop(node):
            return DecideAction.DROP

        transforms = [Decide("div", decide_drop, callback=hook, report=report)]
        JustHTML("<div></div>", transforms=transforms, safe=False)
        self.assertTrue(called_hook, "Decide hook not called")
        called_hook = False

        # 2. Edit wrapper
        def edit_cb(node):
            pass

        transforms_edit = [Edit("div", edit_cb, callback=hook, report=report)]
        JustHTML("<div></div>", transforms=transforms_edit, safe=False)
        self.assertTrue(called_hook, "Edit hook not called")
        called_hook = False

        # 3. EditDocument wrapper
        def edit_doc_cb(node):
            pass

        transforms_doc = [EditDocument(edit_doc_cb, callback=hook, report=report)]
        # EditDocument runs on root. JustHTML processes a doc.
        JustHTML("", transforms=transforms_doc, safe=False)
        self.assertTrue(called_hook, "EditDocument hook not called")
        called_hook = False

        # 4. EditAttrs wrapper
        def edit_attrs_cb(node):
            return {"a": "1"}

        transforms_attrs = [EditAttrs("div", edit_attrs_cb, callback=hook, report=report)]
        JustHTML("<div></div>", transforms=transforms_attrs, safe=False)
        self.assertTrue(called_hook, "EditAttrs hook not called")
        called_hook = False

        # 5. DropForeignNamespaces wrapper
        # Must drop to trigger hook. HTML parser puts SVG in SVG namespace (safe).
        # We manually construct a node with Unsafe namespace.
        t_foreign = DropForeignNamespaces(callback=hook, report=report)
        foreign_node = Element("bad", {}, "unsafe:namespace")
        root = Element("root", {}, "html")
        root.append_child(foreign_node)
        # Manual apply to hit the wrapper
        apply_compiled_transforms(root, compile_transforms([t_foreign]))
        self.assertTrue(called_hook, "DropForeignNamespaces hook not called")

    def test_drop_foreign_namespaces_keep_path(self):
        """Cover DropForeignNamespaces KEEP return for HTML namespace nodes."""
        t_foreign = DropForeignNamespaces(report=lambda m, node: None)
        root = Element("root", {}, "html")
        root.append_child(Element("p", {}, "html"))
        apply_compiled_transforms(root, compile_transforms([t_foreign]))
        self.assertEqual(len(root.children), 1)

    def test_drop_attrs_no_attrs_returns_none(self):
        """Cover DropAttrs early return when attrs is empty."""
        t = DropAttrs("div", patterns=("data-*",), report=lambda m, node: None)
        root = Element("root", {}, "html")
        root.append_child(Element("div", {}, "html"))
        apply_compiled_transforms(root, compile_transforms([t]))
        self.assertEqual(len(root.children), 1)

    def test_drop_attrs_no_match_returns_none(self):
        """Cover DropAttrs early return when no attribute matched patterns."""
        t = DropAttrs("div", patterns=("data-*",), report=lambda m, node: None)
        root = Element("root", {}, "html")
        root.append_child(Element("div", {"class": "x"}, "html"))
        apply_compiled_transforms(root, compile_transforms([t]))
        self.assertEqual(len(root.children), 1)

    def test_allowlist_attrs_wrapper_coverage(self):
        """
        Cover AllowlistAttrs wrapper hooks + uppercase normalization logic.
        """
        called_hook = False

        def hook(node):
            nonlocal called_hook
            called_hook = True

        def set_upper(node):
            return {"DATA-TEST": "val", "valid": "ok"}

        t_setup = EditAttrs("div", set_upper)
        t_allow = AllowlistAttrs(
            "div", allowed_attributes={"div": ["data-test", "valid"]}, callback=hook, report=lambda m, n: None
        )

        processor = JustHTML("<div></div>", transforms=[t_setup, t_allow], safe=False)

        self.assertTrue(called_hook, "AllowlistAttrs hook not called")
        result = to_html(processor.root)
        self.assertIn('data-test="val"', result)

    def test_allowlist_attrs_no_attrs_returns_none(self):
        """Cover AllowlistAttrs early return when attrs is empty."""
        t = AllowlistAttrs("div", allowed_attributes={"div": ["id"]}, report=lambda m, node: None)
        root = Element("root", {}, "html")
        root.append_child(Element("div", {}, "html"))
        apply_compiled_transforms(root, compile_transforms([t]))
        self.assertEqual(len(root.children), 1)

    def test_allowlist_attrs_no_change_returns_none(self):
        """Cover AllowlistAttrs early return when nothing changes."""
        t = AllowlistAttrs("div", allowed_attributes={"div": ["id"]}, report=lambda m, node: None)
        root = Element("root", {}, "html")
        root.append_child(Element("div", {"id": "x"}, "html"))
        apply_compiled_transforms(root, compile_transforms([t]))
        self.assertEqual(len(root.children), 1)

    def test_drop_attrs_wrapper_coverage(self):
        """
        Cover DropAttrs wrapper reporting/callback with glob matching.
        """
        called_hook = False

        def hook(node):
            nonlocal called_hook
            called_hook = True

        reported = []

        def report(msg, node):
            reported.append(msg)

        t = DropAttrs("div", patterns=("data-*",), callback=hook, report=report)

        JustHTML('<div data-foo="1"></div>', transforms=[t], safe=False)

        self.assertTrue(called_hook, "DropAttrs hook not called")
        self.assertTrue(any("matched pattern 'data-*'" in m for m in reported))

    def test_drop_url_attrs_wrapper_coverage(self):
        """
        Cover DropUrlAttrs wrapper edge cases and callback.
        """
        called_hook = False

        def hook(node):
            nonlocal called_hook
            called_hook = True

        reported = []

        def report(msg, node):
            reported.append(msg)

        div = Element("div", {"href": None}, "html")
        rule = UrlRule()
        policy = UrlPolicy(allow_rules={("div", "href"): rule})
        t = DropUrlAttrs("div", url_policy=policy, callback=hook, report=report)
        compiled = compile_transforms([t])

        # Apply - need root
        root = Element("root", {}, "html")
        root.append_child(div)
        apply_compiled_transforms(root, compiled)

        self.assertTrue(called_hook, "DropUrlAttrs hook not called")
        self.assertTrue(any("Unsafe URL" in m for m in reported))

    def test_drop_url_attrs_no_attrs_branch(self):
        """
        Cover 'if not attrs: return None' in DropUrlAttrs wrapper.
        """
        div = Element("div", {}, "html")
        policy = UrlPolicy()
        t = DropUrlAttrs("div", url_policy=policy)
        compiled = compile_transforms([t])
        root = Element("root", {}, "html")
        root.append_child(div)
        apply_compiled_transforms(root, compiled)
        self.assertEqual(div.name, "div")

    def test_allow_style_attrs_wrapper_coverage(self):
        """
        Cover AllowStyleAttrs wrapper with unsafe style failing sanitization and hook.
        """
        called_hook = False

        def hook(node):
            nonlocal called_hook
            called_hook = True

        reported = []

        def report(msg, node):
            reported.append(msg)

        t = AllowStyleAttrs("div", allowed_css_properties=("color",), callback=hook, report=report)

        JustHTML('<div style="position: absolute;"></div>', transforms=[t], safe=False)

        self.assertTrue(called_hook, "AllowStyleAttrs hook not called")
        self.assertTrue(any("Unsafe inline style" in m for m in reported))

    def test_decide_escape_children_and_template(self):
        """
        Cover Decide.ESCAPE handling for nodes with children and template content.
        """

        def decide_escape(node):
            return DecideAction.ESCAPE

        t = Decide("div", decide_escape)
        t_tmpl = Decide("template", decide_escape)

        # 1. Div with children
        html_div = "<div><span>child</span></div>"
        res_div = to_html(JustHTML(html_div, transforms=[t], safe=False).root)
        self.assertIn("&lt;div&gt;", res_div)
        self.assertIn("<span>child</span>", res_div)

        # 2. Template with content
        html_tmpl = "<template><span>content</span></template>"
        res_tmpl = to_html(JustHTML(html_tmpl, transforms=[t_tmpl], safe=False).root)
        self.assertIn("&lt;template&gt;", res_tmpl)
        self.assertIn("<span>content</span>", res_tmpl)

    def test_sanitize_unsafe_style_fused(self):
        """
        Cover _apply_fused_sanitize branch for unsafe style.
        """
        reported = []

        def report(msg, node):
            reported.append(msg)

        policy = SanitizationPolicy(
            allowed_tags={"div"}, allowed_attributes={"div": {"style"}}, allowed_css_properties=("color",)
        )

        t = Sanitize(policy, report=report)
        html = '<div style="position: absolute"></div>'
        JustHTML(html, transforms=[t])

        self.assertTrue(any("Unsafe inline style" in m for m in reported))

    def test_sanitize_safe_style_unchanged_branch(self):
        """Cover fused sanitize branch where style sanitizes to same value."""
        policy = SanitizationPolicy(
            allowed_tags={"div"},
            allowed_attributes={"div": {"style"}},
            allowed_css_properties=("color",),
        )
        t = Sanitize(policy)
        html = '<div style="color: red"></div>'
        out = to_html(JustHTML(html, transforms=[t]).root)
        self.assertIn('style="color: red"', out)

    def test_decide_escape_uses_raw_tag_spans(self):
        """Cover ESCAPE path where raw start/end tag text is available."""
        t = Decide("div", lambda node: DecideAction.ESCAPE)
        compiled = compile_transforms([t])

        root = Element("root", {}, "html")
        div = Element("div", {}, "html")
        div.append_child(Text("hi"))

        src = "<div>hi</div>"
        div._source_html = src
        div._start_tag_start = 0
        div._start_tag_end = 5
        div._end_tag_start = 7
        div._end_tag_end = len(src)
        div._end_tag_present = True

        root.append_child(div)
        apply_compiled_transforms(root, compiled)
        out = to_html(root)
        self.assertIn("&lt;div&gt;", out)
        self.assertIn("hi", out)
        self.assertIn("&lt;/div&gt;", out)

    def test_sanitize_fused_comment_doctype(self):
        """Cover fused sanitize dropping/keeping comments/doctypes."""
        policy = SanitizationPolicy(allowed_tags=set(), allowed_attributes={}, drop_comments=True, drop_doctype=True)
        t = Sanitize(policy)
        html = "<!-- comment -->"
        res = to_html(JustHTML(html, transforms=[t]).root)
        self.assertNotIn("comment", res)

    def test_unwrap_simple(self):
        """Cover Unwrap transform compilation and execution."""
        t = Unwrap("div")
        html = "<div><span>text</span></div>"
        processor = JustHTML(html, transforms=[t], safe=False)
        result = to_html(processor.root)
        self.assertIn("<span>text</span>", result)
        self.assertNotIn("<div>", result)

    def test_empty_simple(self):
        """Cover Empty transform compilation and execution."""
        t = Empty("div")
        html = "<div><span>text</span></div>"
        processor = JustHTML(html, transforms=[t], safe=False)
        result = to_html(processor.root)
        self.assertIn("<div></div>", result)
