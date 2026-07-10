import unittest
from time import perf_counter

from justhtml import JustHTML, SanitizationPolicy, UrlRule, to_html
from justhtml.dom import Element, Node, Text
from justhtml.transforms import (
    AllowlistAttrs,
    AllowStyleAttrs,
    Decide,
    DecideAction,
    Drop,
    DropAttrs,
    DropForeignNamespaces,
    DropUrlAttrs,
    Edit,
    EditAttrs,
    EditDocument,
    Empty,
    Sanitize,
    SetAttrs,
    Unwrap,
    UrlPolicy,
    apply_compiled_transforms,
    compile_transforms,
)


class TestTransformCallbacksAndAttributeFilters(unittest.TestCase):
    def test_chained_edit_attrs_applies_both_callbacks(self):
        """Chained attribute editors apply changes in callback order."""

        def cb1(node):
            return {"data-step": "1"}

        def cb2(node):
            attrs = node.attrs.copy()
            attrs["data-step"] += "-2"
            return attrs

        transforms = [EditAttrs("div", cb1), EditAttrs("div", cb2)]

        html = "<div></div>"
        processor = JustHTML(html, transforms=transforms, sanitize=False)
        result = to_html(processor.root)
        self.assertIn('data-step="1-2"', result)

    def test_chained_edit_attrs_second_returns_none(self):
        """A no-op second editor preserves changes from the first editor."""

        def cb1(node):
            return {"a": "1"}

        def cb2(node):
            return None

        transforms = [EditAttrs("div", cb1), EditAttrs("div", cb2)]
        html = "<div></div>"
        processor = JustHTML(html, transforms=transforms, sanitize=False)
        result = to_html(processor.root)
        self.assertIn('a="1"', result)

    def test_wrappers_execution(self):
        """Compiled transform wrappers invoke their configured hooks."""
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
        JustHTML("<div></div>", transforms=transforms, sanitize=False)
        self.assertTrue(called_hook, "Decide hook not called")
        called_hook = False

        # 2. Edit wrapper
        def edit_cb(node):
            pass

        transforms_edit = [Edit("div", edit_cb, callback=hook, report=report)]
        JustHTML("<div></div>", transforms=transforms_edit, sanitize=False)
        self.assertTrue(called_hook, "Edit hook not called")
        called_hook = False

        # 3. EditDocument wrapper
        def edit_doc_cb(node):
            pass

        transforms_doc = [EditDocument(edit_doc_cb, callback=hook, report=report)]
        # EditDocument runs on root. JustHTML processes a doc.
        JustHTML("", transforms=transforms_doc, sanitize=False)
        self.assertTrue(called_hook, "EditDocument hook not called")
        called_hook = False

        # 4. EditAttrs wrapper
        def edit_attrs_cb(node):
            return {"a": "1"}

        transforms_attrs = [EditAttrs("div", edit_attrs_cb, callback=hook, report=report)]
        JustHTML("<div></div>", transforms=transforms_attrs, sanitize=False)
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
        """Foreign-namespace filtering leaves HTML nodes in place."""
        t_foreign = DropForeignNamespaces(report=lambda m, node: None)
        root = Element("root", {}, "html")
        root.append_child(Element("p", {}, "html"))
        apply_compiled_transforms(root, compile_transforms([t_foreign]))
        self.assertEqual(len(root.children), 1)

    def test_drop_attrs_no_attrs_returns_none(self):
        """Attribute dropping is a no-op when the element has no attributes."""
        t = DropAttrs("div", patterns=("data-*",), report=lambda m, node: None)
        root = Element("root", {}, "html")
        root.append_child(Element("div", {}, "html"))
        apply_compiled_transforms(root, compile_transforms([t]))
        self.assertEqual(len(root.children), 1)

    def test_drop_attrs_no_match_returns_none(self):
        """Attribute dropping is a no-op when no pattern matches."""
        t = DropAttrs("div", patterns=("data-*",), report=lambda m, node: None)
        root = Element("root", {}, "html")
        root.append_child(Element("div", {"class": "x"}, "html"))
        apply_compiled_transforms(root, compile_transforms([t]))
        self.assertEqual(len(root.children), 1)

    def test_allowlist_attrs_wrapper_normalizes_and_reports(self):
        """Attribute allowlisting normalizes names and invokes its hook."""
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

        processor = JustHTML("<div></div>", transforms=[t_setup, t_allow], sanitize=False)

        self.assertTrue(called_hook, "AllowlistAttrs hook not called")
        result = to_html(processor.root)
        self.assertIn('data-test="val"', result)

    def test_allowlist_attrs_no_attrs_returns_none(self):
        """Attribute allowlisting is a no-op when there are no attributes."""
        t = AllowlistAttrs("div", allowed_attributes={"div": ["id"]}, report=lambda m, node: None)
        root = Element("root", {}, "html")
        root.append_child(Element("div", {}, "html"))
        apply_compiled_transforms(root, compile_transforms([t]))
        self.assertEqual(len(root.children), 1)

    def test_allowlist_attrs_no_change_returns_none(self):
        """Attribute allowlisting preserves an already-valid mapping."""
        t = AllowlistAttrs("div", allowed_attributes={"div": ["id"]}, report=lambda m, node: None)
        root = Element("root", {}, "html")
        root.append_child(Element("div", {"id": "x"}, "html"))
        apply_compiled_transforms(root, compile_transforms([t]))
        self.assertEqual(len(root.children), 1)

    def test_drop_attrs_wrapper_reports_matching_pattern(self):
        """Glob-matched attributes are reported and invoke the callback."""
        called_hook = False

        def hook(node):
            nonlocal called_hook
            called_hook = True

        reported = []

        def report(msg, node):
            reported.append(msg)

        t = DropAttrs("div", patterns=("data-*",), callback=hook, report=report)

        JustHTML('<div data-foo="1"></div>', transforms=[t], sanitize=False)

        self.assertTrue(called_hook, "DropAttrs hook not called")
        self.assertTrue(any("matched forbidden pattern 'data-*'" in m for m in reported))

    def test_drop_url_attrs_wrapper_reports_and_calls_hook(self):
        """Invalid URL attributes are reported and invoke the callback."""
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
        """URL filtering is a no-op when the element has no attributes."""
        div = Element("div", {}, "html")
        policy = UrlPolicy()
        t = DropUrlAttrs("div", url_policy=policy)
        compiled = compile_transforms([t])
        root = Element("root", {}, "html")
        root.append_child(div)
        apply_compiled_transforms(root, compiled)
        self.assertEqual(div.name, "div")

    def test_allow_style_attrs_wrapper_reports_and_calls_hook(self):
        """Rejected inline styles are reported and invoke the callback."""
        called_hook = False

        def hook(node):
            nonlocal called_hook
            called_hook = True

        reported = []

        def report(msg, node):
            reported.append(msg)

        t = AllowStyleAttrs("div", allowed_css_properties=("color",), callback=hook, report=report)

        JustHTML('<div style="position: absolute;"></div>', transforms=[t], sanitize=False)

        self.assertTrue(called_hook, "AllowStyleAttrs hook not called")
        self.assertTrue(any("Unsafe inline style" in m for m in reported))


class TestTransformSanitizationAndEscaping(unittest.TestCase):
    def test_decide_escape_children_and_template(self):
        """Escape decisions preserve ordinary and template children as text."""

        def decide_escape(node):
            return DecideAction.ESCAPE

        t = Decide("div", decide_escape)
        t_tmpl = Decide("template", decide_escape)

        # 1. Div with children
        html_div = "<div><span>child</span></div>"
        res_div = to_html(JustHTML(html_div, transforms=[t], sanitize=False).root)
        self.assertIn("&lt;div&gt;", res_div)
        self.assertIn("<span>child</span>", res_div)

        # 2. Template with content
        html_tmpl = "<template><span>content</span></template>"
        res_tmpl = to_html(JustHTML(html_tmpl, transforms=[t_tmpl], sanitize=False).root)
        self.assertIn("&lt;template&gt;", res_tmpl)
        self.assertIn("<span>content</span>", res_tmpl)

    def test_sanitize_unsafe_style_fused(self):
        """Fused sanitization removes unsafe inline styles."""
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
        """Fused sanitization preserves an already-safe inline style."""
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
        """Escape decisions reuse tracked raw start and end tag text."""
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

    def test_decide_escape_inherits_source_from_ancestor(self):
        t = Decide("div", lambda node: DecideAction.ESCAPE)
        compiled = compile_transforms([t])

        src = "<div>hi</div>"
        root = Element("root", {}, "html")
        root._source_html = src
        div = Element("div", {}, "html")
        div._start_tag_start = 0
        div._start_tag_end = 5
        div._end_tag_start = 7
        div._end_tag_end = len(src)
        div._end_tag_present = True
        div.append_child(Text("hi"))
        root.append_child(div)

        apply_compiled_transforms(root, compiled)

        self.assertEqual(div._source_html, src)
        self.assertIn("&lt;div&gt;hi&lt;/div&gt;", to_html(root))

    def test_sanitize_fused_comment_doctype(self):
        """Fused sanitization applies comment and doctype policy settings."""
        policy = SanitizationPolicy(allowed_tags=set(), allowed_attributes={}, drop_comments=True, drop_doctype=True)
        t = Sanitize(policy)
        html = "<!-- comment -->"
        res = to_html(JustHTML(html, transforms=[t]).root)
        self.assertNotIn("comment", res)

    def test_sanitize_drops_comments_inside_unwrapped_disallowed_tags(self):
        policy = SanitizationPolicy(allowed_tags={"b"}, allowed_attributes={}, drop_comments=True)
        t = Sanitize(policy)
        html = "<foo><!-- comment --><b>hi</b></foo>"
        res = to_html(JustHTML(html, fragment=True, transforms=[t]).root)
        self.assertNotIn("comment", res)
        self.assertIn("<b>hi</b>", res)

    def test_unwrap_simple(self):
        """Unwrap removes the element while preserving its children."""
        t = Unwrap("div")
        html = "<div><span>text</span></div>"
        processor = JustHTML(html, transforms=[t], sanitize=False)
        result = to_html(processor.root)
        self.assertIn("<span>text</span>", result)
        self.assertNotIn("<div>", result)

    def test_empty_simple(self):
        """Empty removes all children from the selected element."""
        t = Empty("div")
        html = "<div><span>text</span></div>"
        processor = JustHTML(html, transforms=[t], sanitize=False)
        result = to_html(processor.root)
        self.assertIn("<div></div>", result)


class TestTransformStructuralMatching(unittest.TestCase):
    def test_edit_attrs_structural_selector_scales_linearly(self):
        # EditAttrs never mutates tree structure, so nth-child matching may
        # share cached sibling data across nodes instead of recomputing it
        # from scratch for every one. Previously this was quadratic.
        def mark(node):
            attrs = dict(node.attrs)
            attrs["marked"] = "1"
            return attrs

        html = "<p>x</p>" * 4000
        JustHTML(html, fragment=True, transforms=[EditAttrs("p:nth-child(2n)", mark)], sanitize=False)
        start = perf_counter()
        doc = JustHTML(html, fragment=True, transforms=[EditAttrs("p:nth-child(2n)", mark)], sanitize=False)
        elapsed = perf_counter() - start
        self.assertLess(elapsed, 2.0)
        self.assertEqual(doc.to_text(separator="").count("x"), 4000)

    def test_edit_attrs_nth_child_correct_with_mixed_structural_transform(self):
        # A structurally-mutating transform (Drop) earlier in the same pipeline
        # must still be reflected in EditAttrs's nth-child results: sharing the
        # selector-matching cache is only enabled when every transform in the
        # walk is provably attribute-only, so this falls back to the always-
        # correct per-node behavior instead of using a stale cache.
        marked: list[str] = []

        def mark(node):
            marked.append(node.to_text())
            attrs = dict(node.attrs)
            attrs["marked"] = "1"
            return attrs

        doc = JustHTML(
            "<p>1</p><div>skip</div><p>2</p><p>3</p><p>4</p>",
            fragment=True,
            transforms=[Drop("div"), EditAttrs("p:nth-child(2n)", mark)],
            sanitize=False,
        )
        self.assertEqual(marked, ["2", "4"])
        self.assertEqual(
            to_html(doc.root, pretty=False),
            '<p>1</p><p marked="1">2</p><p>3</p><p marked="1">4</p>',
        )

    def test_edit_with_structural_mutation_and_later_setattrs_does_not_hang(self):
        # Edit's callback can arbitrarily mutate tree structure (e.g. insert a
        # new sibling) with no visibility for the framework. Selector-matching
        # cache sharing must stay disabled whenever a pipeline contains a
        # kind like this, or a later structural-selector match could act on
        # stale cached data and (in the worst case) loop forever.
        def insert_marker(p):
            marker = Node("span")
            marker.append_child(Text("NEW "))
            p.parent.insert_before(marker, p)

        doc = JustHTML(
            "<p>one</p><p>two</p>",
            fragment=True,
            transforms=[
                Edit("p:first-child", insert_marker),
                SetAttrs("span", id="marker"),
            ],
        )
        # SetAttrs doesn't reach the inserted <span> since it lands before the
        # walker's current cursor (see docs/transforms.md); the point of this
        # test is that it terminates with the same result, not that it hangs.
        self.assertEqual(
            to_html(doc.root, pretty=False),
            "<span>NEW </span><p>one</p><p>two</p>",
        )
