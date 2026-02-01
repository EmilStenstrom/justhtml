from __future__ import annotations

import unittest

import justhtml
from justhtml import JustHTML
from justhtml.node import Comment, DocumentFragment, Element, Node, Template, Text
from justhtml.sanitize import (
    CSS_PRESET_TEXT,
    DEFAULT_POLICY,
    SanitizationPolicy,
    UnsafeHandler,
    UrlPolicy,
    UrlProxy,
    UrlRule,
    _css_value_contains_disallowed_functions,
    _css_value_has_disallowed_resource_functions,
    _css_value_may_load_external_resource,
    _effective_allow_relative,
    _effective_proxy,
    _effective_url_handling,
    _is_valid_css_property_name,
    _sanitize_css_url_functions,
    _sanitize_inline_style,
    _sanitize_url_value_with_rule,
    sanitize_dom,
)
from justhtml.sanitize import _sanitize as sanitize
from justhtml.serialize import to_html
from justhtml.tokens import ParseError


class _CoverageSentinel:
    def __getitem__(self, key: str):
        raise AssertionError("unreachable")


class TestSanitizePlumbing(unittest.TestCase):
    def test_public_api_exports_exist(self) -> None:
        assert isinstance(DEFAULT_POLICY, SanitizationPolicy)
        assert "sanitize" not in justhtml.__all__
        assert "Sanitize" in justhtml.__all__
        assert "HTMLContext" in justhtml.__all__
        assert callable(sanitize)

    def test_urlproxy_rejects_empty_url(self) -> None:
        with self.assertRaises(ValueError):
            UrlProxy(url="")

    def test_urlrule_and_policy_normalize_inputs(self) -> None:
        rule = UrlRule(allowed_schemes=["https"], allowed_hosts=["example.com"])
        assert isinstance(rule.allowed_schemes, set)
        assert isinstance(rule.allowed_hosts, set)

        policy = SanitizationPolicy(
            allowed_tags=["div"],
            allowed_attributes={"*": [], "div": []},
            url_policy=UrlPolicy(allow_rules={}),
            drop_content_tags=["script", "style"],
            force_link_rel=["noopener"],
            allowed_css_properties=["color"],
        )
        assert isinstance(policy.allowed_tags, set)
        assert isinstance(policy.allowed_attributes, dict)
        assert isinstance(policy.drop_content_tags, set)
        assert isinstance(policy.force_link_rel, set)
        assert isinstance(policy.allowed_css_properties, set)

    def test_urlrule_rejects_non_urlproxy_instance(self) -> None:
        with self.assertRaises(TypeError):
            UrlRule(proxy="/proxy")  # type: ignore[arg-type]

    def test_policy_rejects_invalid_unsafe_handling(self) -> None:
        with self.assertRaises(ValueError):
            SanitizationPolicy(
                allowed_tags=["div"],
                allowed_attributes={"*": [], "div": []},
                url_policy=UrlPolicy(allow_rules={}),
                allowed_css_properties=["color"],
                unsafe_handling="nope",  # type: ignore[arg-type]
            )

    def test_policy_rejects_invalid_disallowed_tag_handling(self) -> None:
        with self.assertRaises(ValueError):
            SanitizationPolicy(
                allowed_tags=["div"],
                allowed_attributes={"*": [], "div": []},
                url_policy=UrlPolicy(allow_rules={}),
                allowed_css_properties=["color"],
                disallowed_tag_handling="nope",  # type: ignore[arg-type]
            )

    def test_policy_rejects_string_allowed_tags(self) -> None:
        with self.assertRaises(TypeError):
            SanitizationPolicy(
                allowed_tags="div",  # type: ignore[arg-type]
                allowed_attributes={"*": []},
            )

    def test_policy_rejects_string_allowed_attributes_value(self) -> None:
        with self.assertRaises(TypeError):
            SanitizationPolicy(
                allowed_tags=["custom-element"],
                allowed_attributes={"custom-element": "attribute"},  # type: ignore[arg-type]
            )

    def test_policy_rejects_non_mapping_allowed_attributes(self) -> None:
        with self.assertRaises(TypeError):
            SanitizationPolicy(
                allowed_tags=["div"],
                allowed_attributes=[],  # type: ignore[arg-type]
            )

    def test_policy_rejects_empty_tag_key_in_allowed_attributes(self) -> None:
        with self.assertRaises(ValueError):
            SanitizationPolicy(
                allowed_tags=["div"],
                allowed_attributes={"": ["id"]},
            )

    def test_policy_normalizes_and_merges_allowed_attributes_keys(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["DIV"],
            allowed_attributes={"DIV": ["ID"], "div": ["class"]},
        )
        assert policy.allowed_tags == {"div"}
        assert policy.allowed_attributes["div"] == {"id", "class"}

    def test_url_policy_rejects_invalid_url_handling(self) -> None:
        with self.assertRaises(ValueError):
            UrlPolicy(default_handling="nope")  # type: ignore[arg-type]

    def test_url_policy_coerces_rules_to_dict(self) -> None:
        url_policy = UrlPolicy(
            allow_rules=[(("a", "href"), UrlRule(allowed_schemes={"https"}))],  # type: ignore[arg-type]
        )
        assert isinstance(url_policy.allow_rules, dict)

    def test_url_policy_rejects_non_urlproxy_instance(self) -> None:
        with self.assertRaises(TypeError):
            UrlPolicy(proxy="/proxy")  # type: ignore[arg-type]

    def test_url_policy_proxy_mode_requires_proxy_config(self) -> None:
        with self.assertRaises(ValueError):
            UrlPolicy(
                allow_rules={("a", "href"): UrlRule(handling="proxy", allowed_schemes={"https"})},
            )

    def test_urlrule_post_init_branches(self) -> None:
        # Cover coercion paths and the proxy type-check.
        rule = UrlRule(allowed_schemes=["https"], allowed_hosts=["example.com"], proxy=None)
        assert rule.allowed_schemes == {"https"}
        assert rule.allowed_hosts == {"example.com"}

        with self.assertRaises(TypeError):
            UrlRule(proxy=_CoverageSentinel())  # type: ignore[arg-type]


class TestSanitizeDom(unittest.TestCase):
    def test_sanitize_dom_document_fragment(self) -> None:
        root = DocumentFragment()
        root.append_child(Node("script"))
        root.append_child(Node("b"))
        policy = SanitizationPolicy(
            allowed_tags=["b"],
            allowed_attributes={"*": []},
            disallowed_tag_handling="drop",
        )

        out = sanitize_dom(root, policy=policy)
        assert out is root
        assert [child.name for child in (root.children or [])] == ["b"]

    def test_sanitize_dom_element_root(self) -> None:
        root = Node("div")
        root.append_child(Node("script"))
        policy = SanitizationPolicy(
            allowed_tags=["div"],
            allowed_attributes={"*": []},
            disallowed_tag_handling="drop",
        )

        out = sanitize_dom(root, policy=policy)
        assert out is root
        assert root.children == []

    def test_sanitize_dom_default_policy(self) -> None:
        root = DocumentFragment()
        root.append_child(Node("b"))
        out = sanitize_dom(root)
        assert out is root
        assert [child.name for child in (root.children or [])] == ["b"]

    def test_sanitize_dom_compiled_cache_reuse(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["b"],
            allowed_attributes={"*": []},
            disallowed_tag_handling="drop",
        )
        root = DocumentFragment()
        root.append_child(Node("b"))
        sanitize_dom(root, policy=policy)

        root2 = DocumentFragment()
        root2.append_child(Node("b"))
        out = sanitize_dom(root2, policy=policy)
        assert out is root2

    def test_sanitize_dom_returns_wrapper_on_drop(self) -> None:
        root = Node("script")
        policy = SanitizationPolicy(
            allowed_tags=[],
            allowed_attributes={"*": []},
            disallowed_tag_handling="drop",
        )

        out = sanitize_dom(root, policy=policy)
        assert out.name == "#document-fragment"
        assert out.children == []

    def test_is_valid_css_property_name(self) -> None:
        assert _is_valid_css_property_name("border-top") is True
        assert _is_valid_css_property_name("") is False
        assert _is_valid_css_property_name("co_lor") is False

    def test_sanitize_inline_style_edge_cases(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["div"],
            allowed_attributes={"*": [], "div": ["style"]},
            url_policy=UrlPolicy(allow_rules={}),
            allowed_css_properties={"color"},
        )

        assert (
            _sanitize_inline_style(
                allowed_css_properties=policy.allowed_css_properties,
                value="",
                tag="div",
                url_policy=policy.url_policy,
            )
            is None
        )

        assert (
            _sanitize_inline_style(
                allowed_css_properties=policy.allowed_css_properties,
                value="margin: 0",
                tag="div",
                url_policy=policy.url_policy,
            )
            is None
        )

        value = "color; co_lor: red; margin: 0; color: ; COLOR: red"
        assert (
            _sanitize_inline_style(
                allowed_css_properties=policy.allowed_css_properties,
                value=value,
                tag="div",
                url_policy=policy.url_policy,
            )
            == "color: red"
        )

    def test_sanitize_inline_style_returns_none_when_allowlist_empty(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["div"],
            allowed_attributes={"*": [], "div": []},
            url_policy=UrlPolicy(allow_rules={}),
            allowed_css_properties=set(),
        )

        assert (
            _sanitize_inline_style(
                allowed_css_properties=policy.allowed_css_properties,
                value="color: red",
                tag="div",
                url_policy=policy.url_policy,
            )
            is None
        )

    def test_sanitize_css_url_functions_allows_relative_url_when_rule_present(self) -> None:
        url_policy = UrlPolicy(
            default_handling="allow",
            allow_rules={
                ("*", "style:background-image"): UrlRule(
                    allowed_schemes=set(),
                    resolve_protocol_relative=None,
                    allow_relative=True,
                )
            },
        )

        out = _sanitize_css_url_functions(
            url_policy=url_policy,
            tag="div",
            prop="background-image",
            value="linear-gradient(rgba(0,0,0,.45), rgba(0,0,0,.45)), url('/site_media/covers/cover.jpg')",
        )
        assert out is not None
        assert "url('/site_media/covers/cover.jpg')" in out

    def test_sanitize_css_url_functions_rejects_comments(self) -> None:
        url_policy = UrlPolicy(
            default_handling="allow",
            allow_rules={
                ("*", "style:background-image"): UrlRule(
                    allowed_schemes=set(),
                    resolve_protocol_relative=None,
                    allow_relative=True,
                )
            },
        )
        assert (
            _sanitize_css_url_functions(
                url_policy=url_policy,
                tag="div",
                prop="background-image",
                value="url(/x)/*comment*/",
            )
            is None
        )

    def test_sanitize_css_url_functions_requires_matching_rule(self) -> None:
        url_policy = UrlPolicy(default_handling="allow", allow_rules={})
        assert (
            _sanitize_css_url_functions(
                url_policy=url_policy,
                tag="div",
                prop="background-image",
                value="url('/x')",
            )
            is None
        )

    def test_lookup_css_url_rule_prefers_tag_specific_then_wildcard(self) -> None:
        # Coverage for the (tag, key) branch.
        url_policy = UrlPolicy(
            default_handling="allow",
            allow_rules={
                ("div", "style:background-image"): UrlRule(allowed_schemes=set(), resolve_protocol_relative=None),
                ("*", "style:background-image"): UrlRule(allowed_schemes=set(), resolve_protocol_relative=None),
            },
        )
        out = _sanitize_css_url_functions(
            url_policy=url_policy,
            tag="div",
            prop="background-image",
            value="url('/x.png')",
        )
        assert out == "url('/x.png')"

    def test_sanitize_css_url_functions_requires_closing_paren(self) -> None:
        url_policy = UrlPolicy(
            default_handling="allow",
            allow_rules={
                ("*", "style:background-image"): UrlRule(
                    allowed_schemes=set(),
                    resolve_protocol_relative=None,
                    allow_relative=True,
                )
            },
        )
        assert (
            _sanitize_css_url_functions(
                url_policy=url_policy,
                tag="div",
                prop="background-image",
                value="url('/x'",
            )
            is None
        )

    def test_sanitize_css_url_functions_rejects_open_paren_only(self) -> None:
        url_policy = UrlPolicy(
            default_handling="allow",
            allow_rules={
                ("*", "style:background-image"): UrlRule(
                    allowed_schemes=set(),
                    resolve_protocol_relative=None,
                    allow_relative=True,
                )
            },
        )
        assert (
            _sanitize_css_url_functions(
                url_policy=url_policy,
                tag="div",
                prop="background-image",
                value="url(",
            )
            is None
        )

    def test_sanitize_css_url_functions_rejects_only_whitespace_after_open(self) -> None:
        url_policy = UrlPolicy(
            default_handling="allow",
            allow_rules={
                ("*", "style:background-image"): UrlRule(
                    allowed_schemes=set(),
                    resolve_protocol_relative=None,
                    allow_relative=True,
                )
            },
        )
        assert (
            _sanitize_css_url_functions(
                url_policy=url_policy,
                tag="div",
                prop="background-image",
                value="url(   ",
            )
            is None
        )

    def test_sanitize_css_url_functions_rejects_backslash_in_value(self) -> None:
        url_policy = UrlPolicy(
            default_handling="allow",
            allow_rules={
                ("*", "style:background-image"): UrlRule(
                    allowed_schemes=set(),
                    resolve_protocol_relative=None,
                    allow_relative=True,
                )
            },
        )
        assert (
            _sanitize_css_url_functions(
                url_policy=url_policy,
                tag="div",
                prop="background-image",
                value="url('/x\\y.png')",
            )
            is None
        )

    def test_sanitize_css_url_functions_allows_whitespace_before_closing_paren(self) -> None:
        url_policy = UrlPolicy(
            default_handling="allow",
            allow_rules={
                ("*", "style:background-image"): UrlRule(
                    allowed_schemes=set(),
                    resolve_protocol_relative=None,
                    allow_relative=True,
                )
            },
        )
        out = _sanitize_css_url_functions(
            url_policy=url_policy,
            tag="div",
            prop="background-image",
            value="url('/x.png'   )",
        )
        assert out == "url('/x.png')"

    def test_sanitize_css_url_functions_rejects_unquoted_missing_closing_paren(self) -> None:
        url_policy = UrlPolicy(
            default_handling="allow",
            allow_rules={
                ("*", "style:background-image"): UrlRule(
                    allowed_schemes=set(),
                    resolve_protocol_relative=None,
                    allow_relative=True,
                )
            },
        )
        assert (
            _sanitize_css_url_functions(
                url_policy=url_policy,
                tag="div",
                prop="background-image",
                value="url(/x",
            )
            is None
        )

    def test_sanitize_css_url_functions_rejects_unquoted_del_char(self) -> None:
        url_policy = UrlPolicy(
            default_handling="allow",
            allow_rules={
                ("*", "style:background-image"): UrlRule(
                    allowed_schemes=set(),
                    resolve_protocol_relative=None,
                    allow_relative=True,
                )
            },
        )
        assert (
            _sanitize_css_url_functions(
                url_policy=url_policy,
                tag="div",
                prop="background-image",
                value=f"url(/x{chr(0x7F)}y)",
            )
            is None
        )

    def test_sanitize_css_url_functions_allows_unquoted_simple_url(self) -> None:
        url_policy = UrlPolicy(
            default_handling="allow",
            allow_rules={
                ("*", "style:background-image"): UrlRule(
                    allowed_schemes=set(),
                    resolve_protocol_relative=None,
                    allow_relative=True,
                )
            },
        )
        out = _sanitize_css_url_functions(
            url_policy=url_policy,
            tag="div",
            prop="background-image",
            value="url(/x.png)",
        )
        assert out == "url('/x.png')"

    def test_sanitize_css_url_functions_allows_trailing_whitespace(self) -> None:
        url_policy = UrlPolicy(
            default_handling="allow",
            allow_rules={
                ("*", "style:background-image"): UrlRule(
                    allowed_schemes=set(),
                    resolve_protocol_relative=None,
                    allow_relative=True,
                )
            },
        )
        out = _sanitize_css_url_functions(
            url_policy=url_policy,
            tag="div",
            prop="background-image",
            value="url('/x.png') ",
        )
        assert out == "url('/x.png') "

    def test_sanitize_css_url_functions_allows_comma_between_urls(self) -> None:
        url_policy = UrlPolicy(
            default_handling="allow",
            allow_rules={
                ("*", "style:background-image"): UrlRule(
                    allowed_schemes=set(),
                    resolve_protocol_relative=None,
                    allow_relative=True,
                )
            },
        )
        out = _sanitize_css_url_functions(
            url_policy=url_policy,
            tag="div",
            prop="background-image",
            value="url('/x.png'), url('/y.png')",
        )
        assert out == "url('/x.png'), url('/y.png')"

    def test_sanitize_css_url_functions_rejects_missing_closing_quote(self) -> None:
        url_policy = UrlPolicy(
            default_handling="allow",
            allow_rules={
                ("*", "style:background-image"): UrlRule(
                    allowed_schemes=set(),
                    resolve_protocol_relative=None,
                    allow_relative=True,
                )
            },
        )
        assert (
            _sanitize_css_url_functions(
                url_policy=url_policy,
                tag="div",
                prop="background-image",
                value="url('/x)",
            )
            is None
        )

    def test_sanitize_css_url_functions_rejects_missing_closing_paren_after_quote(self) -> None:
        url_policy = UrlPolicy(
            default_handling="allow",
            allow_rules={
                ("*", "style:background-image"): UrlRule(
                    allowed_schemes=set(),
                    resolve_protocol_relative=None,
                    allow_relative=True,
                )
            },
        )
        assert (
            _sanitize_css_url_functions(
                url_policy=url_policy,
                tag="div",
                prop="background-image",
                value="url('/x')x",
            )
            is None
        )

    def test_sanitize_css_url_functions_rejects_suspicious_chars_in_rewritten_url(self) -> None:
        # Coverage for the per-character rejection path.
        def filt(tag: str, attr: str, value: str) -> str | None:
            if attr == "style:background-image":
                return "/x y"
            return value

        url_policy = UrlPolicy(
            default_handling="allow",
            allow_rules={
                ("*", "style:background-image"): UrlRule(
                    allowed_schemes=set(),
                    resolve_protocol_relative=None,
                    allow_relative=True,
                )
            },
            url_filter=filt,
        )
        assert (
            _sanitize_css_url_functions(
                url_policy=url_policy,
                tag="div",
                prop="background-image",
                value="url('/x.png')",
            )
            is None
        )

    def test_css_value_has_disallowed_resource_functions_rejects_unterminated_comment(self) -> None:
        assert _css_value_has_disallowed_resource_functions("/*x") is True

    def test_css_value_has_disallowed_resource_functions_allows_closed_comment_then_url(self) -> None:
        # Exercise the comment-skip "break" path.
        assert _css_value_has_disallowed_resource_functions("/*x*/ url('/x.png')") is False

    def test_sanitize_css_url_functions_rejects_url_filter_empty_string(self) -> None:
        # Coverage for `if not sanitized:`.
        def filt(tag: str, attr: str, value: str) -> str | None:
            if attr == "style:background-image":
                return ""
            return value

        url_policy = UrlPolicy(
            default_handling="allow",
            allow_rules={
                ("*", "style:background-image"): UrlRule(
                    allowed_schemes=set(),
                    resolve_protocol_relative=None,
                    allow_relative=True,
                )
            },
            url_filter=filt,
        )
        assert (
            _sanitize_css_url_functions(
                url_policy=url_policy,
                tag="div",
                prop="background-image",
                value="url('/x.png')",
            )
            is None
        )

    def test_sanitize_css_url_functions_rewrites_using_url_filter(self) -> None:
        def filt(tag: str, attr: str, value: str) -> str | None:
            if attr == "style:background-image":
                return f"/proxied{value}"
            return value

        url_policy = UrlPolicy(
            default_handling="allow",
            allow_rules={
                ("*", "style:background-image"): UrlRule(
                    allowed_schemes=set(),
                    resolve_protocol_relative=None,
                    allow_relative=True,
                )
            },
            url_filter=filt,
        )

        out = _sanitize_css_url_functions(
            url_policy=url_policy,
            tag="div",
            prop="background-image",
            value="url('/x.png')",
        )
        assert out == "url('/proxied/x.png')"

    def test_sanitize_css_url_functions_rejects_invalid_url(self) -> None:
        url_policy = UrlPolicy(
            default_handling="allow",
            allow_rules={
                ("*", "style:background-image"): UrlRule(
                    allowed_schemes=set(),
                    resolve_protocol_relative=None,
                    allow_relative=True,
                )
            },
        )
        assert (
            _sanitize_css_url_functions(
                url_policy=url_policy,
                tag="div",
                prop="background-image",
                value="url('javascript:alert(1)')",
            )
            is None
        )

    def test_sanitize_css_url_functions_rejects_unquoted_whitespace(self) -> None:
        url_policy = UrlPolicy(
            default_handling="allow",
            allow_rules={
                ("*", "style:background-image"): UrlRule(
                    allowed_schemes=set(),
                    resolve_protocol_relative=None,
                    allow_relative=True,
                )
            },
        )
        assert (
            _sanitize_css_url_functions(
                url_policy=url_policy,
                tag="div",
                prop="background-image",
                value="url(/x y)",
            )
            is None
        )

    def test_sanitize_css_url_functions_rejects_empty_url(self) -> None:
        url_policy = UrlPolicy(
            default_handling="allow",
            allow_rules={
                ("*", "style:background-image"): UrlRule(
                    allowed_schemes=set(),
                    resolve_protocol_relative=None,
                    allow_relative=True,
                )
            },
        )
        assert (
            _sanitize_css_url_functions(
                url_policy=url_policy,
                tag="div",
                prop="background-image",
                value="url()",
            )
            is None
        )

    def test_sanitize_inline_style_drops_url_declaration_when_url_policy_missing(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["div"],
            allowed_attributes={"*": [], "div": ["style"]},
            allowed_css_properties={"background-image"},
            url_policy=UrlPolicy(allow_rules={}),
        )

        value = "background-image: url('/x.png')"
        assert (
            _sanitize_inline_style(
                allowed_css_properties=policy.allowed_css_properties,
                value=value,
                tag="div",
                url_policy=None,
            )
            is None
        )

    def test_sanitize_inline_style_drops_url_declaration_without_matching_rule(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["div"],
            allowed_attributes={"*": [], "div": ["style"]},
            allowed_css_properties={"background-image"},
            url_policy=UrlPolicy(default_handling="allow", allow_rules={}),
        )

        value = "background-image: url('/x.png')"
        assert (
            _sanitize_inline_style(
                allowed_css_properties=policy.allowed_css_properties,
                value=value,
                tag="div",
                url_policy=policy.url_policy,
            )
            is None
        )

    def test_sanitize_inline_style_drops_disallowed_resource_function_even_with_rule(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["div"],
            allowed_attributes={"*": [], "div": ["style"]},
            allowed_css_properties={"background-image"},
            url_policy=UrlPolicy(
                default_handling="allow",
                allow_rules={
                    ("*", "style:background-image"): UrlRule(
                        allowed_schemes=set(),
                        resolve_protocol_relative=None,
                        allow_relative=True,
                    )
                },
            ),
        )

        value = "background-image: image-set(url('/x.png') 1x)"
        assert (
            _sanitize_inline_style(
                allowed_css_properties=policy.allowed_css_properties,
                value=value,
                tag="div",
                url_policy=policy.url_policy,
            )
            is None
        )

    def test_sanitize_inline_style_can_allow_background_image_relative_url_via_url_policy(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["div"],
            allowed_attributes={"*": [], "div": ["style"]},
            allowed_css_properties={"background-image"},
            url_policy=UrlPolicy(
                default_handling="allow",
                allow_rules={
                    ("*", "style:background-image"): UrlRule(
                        allowed_schemes=set(),
                        resolve_protocol_relative=None,
                        allow_relative=True,
                    )
                },
            ),
        )

        value = "background-image: url('/site_media/covers/cover.jpg')"
        assert (
            _sanitize_inline_style(
                allowed_css_properties=policy.allowed_css_properties,
                value=value,
                tag="div",
                url_policy=policy.url_policy,
            )
            == value
        )

    def test_sanitize_inline_style_still_blocks_background_image_with_absolute_url(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["div"],
            allowed_attributes={"*": [], "div": ["style"]},
            allowed_css_properties={"background-image"},
            url_policy=UrlPolicy(
                default_handling="allow",
                allow_rules={
                    ("*", "style:background-image"): UrlRule(
                        allowed_schemes=set(),
                        resolve_protocol_relative=None,
                        allow_relative=True,
                    )
                },
            ),
        )

        value = "background-image: url('https://evil.example/x.png')"
        assert (
            _sanitize_inline_style(
                allowed_css_properties=policy.allowed_css_properties,
                value=value,
                tag="div",
                url_policy=policy.url_policy,
            )
            is None
        )

    def test_css_preset_text_is_conservative(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["div"],
            allowed_attributes={"*": [], "div": ["style"]},
            url_policy=UrlPolicy(allow_rules={}),
            allowed_css_properties=CSS_PRESET_TEXT,
        )

        html = '<div style="color: red; position: fixed; top: 0">x</div>'
        out = JustHTML(html, fragment=True, policy=policy).to_html()
        assert out == '<div style="color: red">x</div>'

    def test_style_attribute_is_dropped_when_nothing_survives(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["div"],
            allowed_attributes={"*": [], "div": ["style"]},
            url_policy=UrlPolicy(allow_rules={}),
            allowed_css_properties=CSS_PRESET_TEXT,
        )

        html = '<div style="position: fixed">x</div>'
        out = JustHTML(html, fragment=True, policy=policy).to_html()
        assert out == "<div>x</div>"

    def test_css_value_may_load_external_resource(self) -> None:
        assert _css_value_may_load_external_resource("url(https://evil.example/x)") is True
        assert _css_value_may_load_external_resource("URL(https://evil.example/x)") is True
        assert _css_value_may_load_external_resource("u r l (https://evil.example/x)") is True
        assert _css_value_may_load_external_resource("u\\72l(https://evil.example/x)") is True
        assert _css_value_may_load_external_resource("u/**/rl(https://evil.example/x)") is True
        assert _css_value_may_load_external_resource("u/*x*/rl(https://evil.example/x)") is True
        assert _css_value_may_load_external_resource("IMAGE-SET(foo)") is True
        assert _css_value_may_load_external_resource("image/**/-set(foo)") is True
        assert _css_value_may_load_external_resource("expression(alert(1))") is True
        assert _css_value_may_load_external_resource("ex/**/pression(alert(1))") is True
        assert _css_value_may_load_external_resource("progid:DXImageTransform.Microsoft.AlphaImageLoader") is True
        assert _css_value_may_load_external_resource("AlphaImageLoader") is True
        assert _css_value_may_load_external_resource("behavior: url(x)") is True
        assert _css_value_may_load_external_resource("-moz-binding: url(x)") is True
        assert _css_value_may_load_external_resource("color: red /*") is True
        assert _css_value_may_load_external_resource("a" * 64) is False
        assert _css_value_may_load_external_resource("red") is False

    def test_css_value_contains_disallowed_functions_allow_url_flag(self) -> None:
        assert _css_value_contains_disallowed_functions("url(/x)", allow_url=False) is True
        assert _css_value_contains_disallowed_functions("url(/x)", allow_url=True) is False

    def test_css_value_has_disallowed_resource_functions(self) -> None:
        assert _css_value_has_disallowed_resource_functions("image-set(foo)") is True
        assert _css_value_has_disallowed_resource_functions("expression(alert(1))") is True
        assert _css_value_has_disallowed_resource_functions("behavior: url(x)") is True
        assert _css_value_has_disallowed_resource_functions("-moz-binding: url(x)") is True
        assert _css_value_has_disallowed_resource_functions("AlphaImageLoader") is True
        assert (
            _css_value_has_disallowed_resource_functions("progid:DXImageTransform.Microsoft.AlphaImageLoader") is True
        )
        assert _css_value_has_disallowed_resource_functions("url(/x)") is False
        assert _css_value_has_disallowed_resource_functions("color: red") is False

    def test_sanitize_url_value_keeps_non_empty_relative_url(self) -> None:
        policy = DEFAULT_POLICY
        rule = UrlRule(allowed_schemes=[])
        assert (
            _sanitize_url_value_with_rule(
                rule=rule,
                value="/x.png",
                tag="img",
                attr="src",
                handling=_effective_url_handling(url_policy=policy.url_policy, rule=rule),
                allow_relative=_effective_allow_relative(url_policy=policy.url_policy, rule=rule),
                proxy=_effective_proxy(url_policy=policy.url_policy, rule=rule),
                url_filter=policy.url_policy.url_filter,
                apply_filter=True,
            )
            == "/x.png"
        )
        assert (
            _sanitize_url_value_with_rule(
                rule=rule,
                value="\x00",
                tag="img",
                attr="src",
                handling=_effective_url_handling(url_policy=policy.url_policy, rule=rule),
                allow_relative=_effective_allow_relative(url_policy=policy.url_policy, rule=rule),
                proxy=_effective_proxy(url_policy=policy.url_policy, rule=rule),
                url_filter=policy.url_policy.url_filter,
                apply_filter=True,
            )
            is None
        )

    def test_url_like_attributes_require_explicit_rules(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["img"],
            allowed_attributes={"*": [], "img": ["src"]},
            url_policy=UrlPolicy(allow_rules={}),
        )
        out = JustHTML('<img src="/x">', fragment=True, policy=policy).to_html()
        assert out == "<img>"

    def test_url_rule_default_allows_even_when_policy_strip(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["img"],
            allowed_attributes={"*": [], "img": ["src"]},
            url_policy=UrlPolicy(
                default_handling="strip",
                default_allow_relative=True,
                allow_rules={("img", "src"): UrlRule(allowed_schemes={"https"})},
            ),
        )

        out = JustHTML('<img src="https://example.com/x">', fragment=True, policy=policy).to_html()
        assert out == '<img src="https://example.com/x">'

        out = JustHTML('<img src="/x">', fragment=True, policy=policy).to_html()
        assert out == '<img src="/x">'

    def test_url_rule_handling_strip_drops_absolute_url(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["img"],
            allowed_attributes={"*": [], "img": ["src"]},
            url_policy=UrlPolicy(
                allow_rules={
                    ("img", "src"): UrlRule(
                        handling="strip",
                        allowed_schemes={"https"},
                    )
                },
            ),
        )

        out = JustHTML('<img src="https://example.com/x">', fragment=True, policy=policy).to_html()
        assert out == "<img>"

    def test_url_rule_handling_strip_drops_relative_url(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["img"],
            allowed_attributes={"*": [], "img": ["src"]},
            url_policy=UrlPolicy(
                default_allow_relative=True,
                allow_rules={
                    ("img", "src"): UrlRule(
                        handling="strip",
                        allowed_schemes=set(),
                        resolve_protocol_relative=None,
                    )
                },
            ),
        )

        out = JustHTML('<img src="/x">', fragment=True, policy=policy).to_html()
        assert out == "<img>"

    def test_url_rule_relative_only_blocks_remote_but_keeps_relative(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["img"],
            allowed_attributes={"*": [], "img": ["src"]},
            url_policy=UrlPolicy(
                default_handling="allow",
                default_allow_relative=True,
                allow_rules={
                    ("img", "src"): UrlRule(
                        allowed_schemes=set(),
                        resolve_protocol_relative=None,
                    )
                },
            ),
        )

        out = JustHTML('<img src="https://example.com/x">', fragment=True, policy=policy).to_html()
        assert out == "<img>"

        out = JustHTML('<img src="/x">', fragment=True, policy=policy).to_html()
        assert out == '<img src="/x">'

    def test_url_rule_can_override_global_strip(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["img"],
            allowed_attributes={"*": [], "img": ["src"]},
            url_policy=UrlPolicy(
                default_allow_relative=False,
                allow_rules={
                    ("img", "src"): UrlRule(
                        allow_relative=True,
                        allowed_schemes=set(),
                        resolve_protocol_relative=None,
                    )
                },
            ),
        )

        out = JustHTML('<img src="/x">', fragment=True, policy=policy).to_html()
        assert out == '<img src="/x">'

        out = JustHTML('<img src="https://example.com/x">', fragment=True, policy=policy).to_html()
        assert out == "<img>"

    def test_url_policy_remote_strip_blocks_protocol_relative(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["img"],
            allowed_attributes={"*": [], "img": ["src"]},
            url_policy=UrlPolicy(
                default_allow_relative=True,
                allow_rules={
                    ("img", "src"): UrlRule(
                        handling="strip",
                        allowed_schemes={"https"},
                        resolve_protocol_relative="https",
                    )
                },
            ),
        )

        out = JustHTML('<img src="//example.com/x">', fragment=True, policy=policy).to_html()
        assert out == "<img>"

    def test_url_policy_remote_proxy_rewrites_protocol_relative(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["img"],
            allowed_attributes={"*": [], "img": ["src"]},
            url_policy=UrlPolicy(
                default_allow_relative=True,
                proxy=UrlProxy(url="/proxy"),
                allow_rules={
                    ("img", "src"): UrlRule(
                        handling="proxy",
                        allowed_schemes={"https"},
                        resolve_protocol_relative="https",
                    )
                },
            ),
        )

        out = JustHTML('<img src="//example.com/x">', fragment=True, policy=policy).to_html()
        assert out == '<img src="/proxy?url=https%3A%2F%2Fexample.com%2Fx">'

    def test_url_policy_remote_proxy_global_and_img_override(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["a", "img"],
            allowed_attributes={"*": [], "a": ["href"], "img": ["src"]},
            url_policy=UrlPolicy(
                default_allow_relative=True,
                proxy=UrlProxy(url="/proxy", param="url"),
                allow_rules={
                    ("a", "href"): UrlRule(handling="proxy", allowed_schemes={"https"}),
                    ("img", "src"): UrlRule(
                        handling="proxy",
                        allowed_schemes={"https"},
                        proxy=UrlProxy(url="/image-proxy", param="url"),
                    ),
                },
            ),
        )

        out = JustHTML('<a href="https://example.com/x">x</a>', fragment=True, policy=policy).to_html()
        assert out == '<a href="/proxy?url=https%3A%2F%2Fexample.com%2Fx">x</a>'

        out = JustHTML('<img src="https://example.com/x">', fragment=True, policy=policy).to_html()
        assert out == '<img src="/image-proxy?url=https%3A%2F%2Fexample.com%2Fx">'

    def test_url_policy_proxy_does_not_bypass_scheme_checks(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["a"],
            allowed_attributes={"*": [], "a": ["href"]},
            url_policy=UrlPolicy(
                default_allow_relative=True,
                proxy=UrlProxy(url="/proxy"),
                allow_rules={
                    ("a", "href"): UrlRule(
                        handling="proxy",
                        allowed_schemes=set(),
                    )
                },
            ),
        )

        out = JustHTML('<a href="https://example.com/x">x</a>', fragment=True, policy=policy).to_html()
        assert out == "<a>x</a>"

    def test_url_policy_proxy_rewrites_fragment_urls(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["a"],
            allowed_attributes={"*": [], "a": ["href"]},
            url_policy=UrlPolicy(
                proxy=UrlProxy(url="/proxy"),
                allow_rules={
                    ("a", "href"): UrlRule(
                        handling="proxy",
                        allowed_schemes={"https"},
                        allow_fragment=True,
                    )
                },
            ),
        )

        out = JustHTML('<a href="#x">x</a>', fragment=True, policy=policy).to_html()
        assert out == '<a href="/proxy?url=%23x">x</a>'

    def test_url_policy_strip_drops_fragment_urls(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["a"],
            allowed_attributes={"*": [], "a": ["href"]},
            url_policy=UrlPolicy(
                default_allow_relative=True,
                allow_rules={
                    ("a", "href"): UrlRule(
                        handling="strip",
                        allowed_schemes={"https"},
                        allow_fragment=True,
                    )
                },
            ),
        )

        out = JustHTML('<a href="#x">x</a>', fragment=True, policy=policy).to_html()
        assert out == "<a>x</a>"

    def test_url_policy_proxy_rewrites_remote_srcset_candidates(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["img"],
            allowed_attributes={"*": [], "img": ["srcset"]},
            url_policy=UrlPolicy(
                default_allow_relative=True,
                proxy=UrlProxy(url="/proxy"),
                allow_rules={("img", "srcset"): UrlRule(handling="proxy", allowed_schemes={"https"})},
            ),
        )

        out = JustHTML(
            '<img srcset="https://example.com/a 1x, /b 2x">',
            fragment=True,
            policy=policy,
        ).to_html()
        assert out == '<img srcset="/proxy?url=https%3A%2F%2Fexample.com%2Fa 1x, /proxy?url=%2Fb 2x">'

    def test_srcset_is_dropped_if_url_filter_drops_value(self) -> None:
        def url_filter(tag: str, attr: str, value: str) -> str | None:
            assert tag == "img"
            assert attr == "srcset"
            assert value
            return None

        policy = SanitizationPolicy(
            allowed_tags=["img"],
            allowed_attributes={"*": [], "img": ["srcset"]},
            url_policy=UrlPolicy(
                default_handling="allow",
                default_allow_relative=True,
                allow_rules={("img", "srcset"): UrlRule(allowed_schemes={"https"})},
                url_filter=url_filter,
            ),
        )

        out = JustHTML('<img srcset="https://example.com/a 1x">', fragment=True, policy=policy).to_html()
        assert out == "<img>"

    def test_srcset_is_dropped_if_empty(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["img"],
            allowed_attributes={"*": [], "img": ["srcset"]},
            url_policy=UrlPolicy(
                default_allow_relative=True,
                allow_rules={("img", "srcset"): UrlRule(allowed_schemes={"https"})},
            ),
        )

        out = JustHTML('<img srcset="  \t\n  ">', fragment=True, policy=policy).to_html()
        assert out == "<img>"

    def test_srcset_url_filter_can_rewrite_value(self) -> None:
        def url_filter(tag: str, attr: str, value: str) -> str | None:
            assert tag == "img"
            assert attr == "srcset"
            assert value == "ignored"
            return "https://example.com/a 1x"

        policy = SanitizationPolicy(
            allowed_tags=["img"],
            allowed_attributes={"*": [], "img": ["srcset"]},
            url_policy=UrlPolicy(
                default_handling="allow",
                default_allow_relative=True,
                allow_rules={("img", "srcset"): UrlRule(allowed_schemes={"https"})},
                url_filter=url_filter,
            ),
        )

        out = JustHTML('<img srcset="ignored">', fragment=True, policy=policy).to_html()
        assert out == '<img srcset="https://example.com/a 1x">'

    def test_srcset_skips_empty_candidates(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["img"],
            allowed_attributes={"*": [], "img": ["srcset"]},
            url_policy=UrlPolicy(
                default_allow_relative=True,
                proxy=UrlProxy(url="/proxy"),
                allow_rules={("img", "srcset"): UrlRule(handling="proxy", allowed_schemes={"https"})},
            ),
        )

        out = JustHTML('<img srcset=", https://example.com/a 1x">', fragment=True, policy=policy).to_html()
        assert out == '<img srcset="/proxy?url=https%3A%2F%2Fexample.com%2Fa 1x">'

    def test_srcset_is_dropped_if_any_candidate_is_invalid(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["img"],
            allowed_attributes={"*": [], "img": ["srcset"]},
            url_policy=UrlPolicy(
                default_allow_relative=True,
                proxy=UrlProxy(url="/proxy"),
                allow_rules={("img", "srcset"): UrlRule(handling="proxy", allowed_schemes={"https"})},
            ),
        )

        out = JustHTML(
            '<img srcset="http://example.com/a 1x, https://example.com/b 2x">',
            fragment=True,
            policy=policy,
        ).to_html()
        assert out == "<img>"

    def test_policy_accepts_pre_normalized_sets(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags={"div"},
            allowed_attributes={"*": set(), "div": {"id"}},
            url_policy=UrlPolicy(allow_rules={}),
            drop_content_tags={"script"},
            force_link_rel={"noopener"},
        )
        assert policy.allowed_tags == {"div"}
        assert policy.allowed_attributes["div"] == {"id"}

        rule = UrlRule(allowed_schemes={"https"}, allowed_hosts=None)
        assert rule.allowed_schemes == {"https"}

    def test_url_rule_rejects_invalid_url_handling_override(self) -> None:
        with self.assertRaises(ValueError):
            UrlRule(handling="nope")  # type: ignore[arg-type]

    def test_url_policy_rejects_non_urlrule_values(self) -> None:
        with self.assertRaises(TypeError):
            UrlPolicy(allow_rules={("a", "href"): "not-a-rule"})  # type: ignore[arg-type]

    def test_sanitize_handles_nested_document_containers(self) -> None:
        # This is intentionally a "plumbing" test: these container nodes are not
        # produced by the parser as nested children, but the sanitizer supports
        # them for manually constructed DOMs.
        policy = SanitizationPolicy(
            allowed_tags=[],
            allowed_attributes={"*": []},
            url_policy=UrlPolicy(allow_rules={}),
        )
        root = DocumentFragment()
        nested = DocumentFragment()
        nested.append_child(Text("t"))
        root.append_child(nested)

        out = sanitize(root, policy=policy)
        assert to_html(out, pretty=False) == "t"

    def test_sanitize_template_subtree_without_template_content_branch(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["template"],
            allowed_attributes={"*": [], "template": []},
            url_policy=UrlPolicy(allow_rules={}),
        )
        root = DocumentFragment()
        root.append_child(Template("template", namespace=None))
        out = sanitize(root, policy=policy)
        assert to_html(out, pretty=False) == "<template></template>"

    def test_sanitize_attribute_edge_cases_do_not_crash(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["div"],
            allowed_attributes={"*": ["id"], "div": ["disabled"]},
            url_policy=UrlPolicy(allow_rules={}),
        )
        n = Node("div", attrs={"": "x", "   ": "y", "id": None, "disabled": None})
        out = sanitize(n, policy=policy)
        html = to_html(out, pretty=False)
        assert html in {"<div disabled id></div>", "<div id disabled></div>"}

    def test_sanitize_drops_disallowed_attribute_and_reports(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["p"],
            allowed_attributes={"*": [], "p": []},
            url_policy=UrlPolicy(allow_rules={}),
            unsafe_handling="collect",
        )
        policy.reset_collected_security_errors()

        n = Node("p", attrs={"foo": "1"})
        out = sanitize(n, policy=policy)
        assert to_html(out, pretty=False) == "<p></p>"
        assert len(policy.collected_security_errors()) == 1

    def test_sanitize_drops_style_attribute_with_no_value(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["span"],
            allowed_attributes={"*": ["style"], "span": ["style"]},
            url_policy=UrlPolicy(allow_rules={}),
            allowed_css_properties={"color"},
            unsafe_handling="collect",
        )
        policy.reset_collected_security_errors()

        n = Node("span", attrs={"style": None})
        out = sanitize(n, policy=policy)
        assert to_html(out, pretty=False) == "<span></span>"
        assert len(policy.collected_security_errors()) == 1

    def test_sanitize_force_link_rel_inserts_rel_when_missing(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["a"],
            allowed_attributes={"*": [], "a": ["href"]},
            url_policy=UrlPolicy(
                default_handling="allow",
                allow_rules={
                    ("a", "href"): UrlRule(allowed_schemes={"https"}),
                },
            ),
            force_link_rel={"noopener"},
            unsafe_handling="collect",
        )
        policy.reset_collected_security_errors()

        n = Node("a", attrs={"href": "https://example.com"})
        out = sanitize(n, policy=policy)
        html = to_html(out, pretty=False)
        assert 'rel="noopener"' in html
        assert len(policy.collected_security_errors()) == 0

    def test_sanitize_reports_url_attr_with_none_value(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["a"],
            allowed_attributes={"*": [], "a": ["href"]},
            url_policy=UrlPolicy(
                default_handling="allow",
                allow_rules={
                    ("a", "href"): UrlRule(allowed_schemes={"https"}),
                },
            ),
            unsafe_handling="collect",
        )
        policy.reset_collected_security_errors()

        n = Node("a", attrs={"href": None})
        out = sanitize(n, policy=policy)
        assert to_html(out, pretty=False) == "<a></a>"
        assert len(policy.collected_security_errors()) == 1

    def test_sanitize_force_link_rel_does_not_rewrite_when_already_normalized(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["a"],
            allowed_attributes={"*": [], "a": ["href", "rel"]},
            url_policy=UrlPolicy(
                default_handling="allow",
                allow_rules={
                    ("a", "href"): UrlRule(allowed_schemes={"https"}),
                },
            ),
            force_link_rel={"noopener"},
            unsafe_handling="collect",
        )
        policy.reset_collected_security_errors()

        n = Node("a", attrs={"href": "https://example.com", "rel": "noopener"})
        out = sanitize(n, policy=policy)
        html = to_html(out, pretty=False)
        assert 'rel="noopener"' in html
        assert len(policy.collected_security_errors()) == 0

    def test_sanitize_lowercases_attribute_names(self) -> None:
        # The parser already lowercases attribute names; build a manual node to
        # ensure sanitize() is robust to unexpected input.
        n = Node("a", attrs={"HREF": "https://example.com"})
        out = sanitize(n)
        html = to_html(out, pretty=False)
        assert 'href="https://example.com"' in html

    def test_sanitize_text_root_is_cloned(self) -> None:
        out = sanitize(Text("x"))
        assert to_html(out, pretty=False) == "x"

    def test_sanitize_root_comment_and_doctype_nodes_do_not_crash(self) -> None:
        # Another plumbing-only test: root comment/doctype nodes aren't typical
        # parser outputs, but sanitize() accepts any node.
        policy_keep = SanitizationPolicy(
            allowed_tags=[],
            allowed_attributes={"*": []},
            url_policy=UrlPolicy(allow_rules={}),
            drop_comments=False,
            drop_doctype=False,
        )

        c = Comment(data="x")
        d = Node("!doctype", data="html")

        assert to_html(sanitize(c, policy=policy_keep), pretty=False) == "<!--x-->"
        assert to_html(sanitize(d, policy=policy_keep), pretty=False) == "<!DOCTYPE html>"

        # Default policy drops these root nodes (turned into empty fragments).
        assert to_html(sanitize(c), pretty=False) == ""
        assert to_html(sanitize(d), pretty=False) == ""

        def test_sanitize_default_policy_differs_for_document_vs_fragment(self) -> None:
            root = JustHTML("<p>Hi</p>").root
            out = sanitize(root)
            assert to_html(out, pretty=False) == "<html><head></head><body><p>Hi</p></body></html>"

    def test_sanitize_root_element_edge_cases(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=["div"],
            allowed_attributes={"*": [], "div": []},
            url_policy=UrlPolicy(allow_rules={}),
        )

        foreign = Node("div", namespace="svg")
        assert to_html(sanitize(foreign, policy=policy), pretty=False) == ""

        disallowed_subtree_drop = SanitizationPolicy(
            allowed_tags=["div"],
            allowed_attributes={"*": [], "div": []},
            url_policy=UrlPolicy(allow_rules={}),
        )
        span = Node("span")
        span.append_child(Text("x"))
        assert to_html(sanitize(span, policy=disallowed_subtree_drop), pretty=False) == "x"

        drop_content = SanitizationPolicy(
            allowed_tags=["div"],
            allowed_attributes={"*": [], "div": []},
            url_policy=UrlPolicy(allow_rules={}),
            drop_content_tags={"script"},
        )
        script = Node("script")
        script.append_child(Text("alert(1)"))
        assert to_html(sanitize(script, policy=drop_content), pretty=False) == ""

        template_policy = SanitizationPolicy(
            allowed_tags=["template"],
            allowed_attributes={"*": [], "template": []},
            url_policy=UrlPolicy(allow_rules={}),
        )
        tpl = Template("template", namespace="html")
        assert tpl.template_content is not None
        tpl.template_content.append_child(Text("T"))
        assert to_html(sanitize(tpl, policy=template_policy), pretty=False) == "<template>T</template>"

        tpl_no_content = Template("template", namespace=None)
        assert to_html(sanitize(tpl_no_content, policy=template_policy), pretty=False) == "<template></template>"


class TestSanitizeUnsafe(unittest.TestCase):
    def test_unsafe_handler_collect_initializes_on_first_use(self) -> None:
        # Cover the centralized handler's fast path when used standalone
        # (without an explicit reset call).
        h = UnsafeHandler("collect")
        h.handle("Unsafe tag 'x'", node=None)
        errs = h.collected()
        assert len(errs) == 1
        assert errs[0].category == "security"

    def test_unsafe_handler_collected_filters_shared_sink(self) -> None:
        h = UnsafeHandler("collect")
        sink = [
            ParseError("unexpected-null-character", category="tokenizer"),
            ParseError("unsafe-html", category="security"),
        ]
        h.sink = sink
        errs = h.collected()
        assert len(errs) == 1
        assert errs[0].category == "security"

    def test_collect_mode_can_run_with_no_security_findings(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=set(DEFAULT_POLICY.allowed_tags),
            allowed_attributes=DEFAULT_POLICY.allowed_attributes,
            url_policy=DEFAULT_POLICY.url_policy,
            allowed_css_properties=DEFAULT_POLICY.allowed_css_properties,
            force_link_rel=DEFAULT_POLICY.force_link_rel,
            unsafe_handling="collect",
        )

        doc = JustHTML("<p>ok</p>", fragment=True, policy=policy)
        _ = doc.to_html(pretty=False)
        assert doc.errors == []

    def test_policy_collect_helpers_cover_empty_paths(self) -> None:
        # Non-collect policy: collection helpers are no-ops.
        policy_strip = SanitizationPolicy(
            allowed_tags={"p"},
            allowed_attributes={},
            url_policy=UrlPolicy(allow_rules={}),
            unsafe_handling="strip",
        )
        policy_strip.reset_collected_security_errors()
        assert policy_strip.collected_security_errors() == []

        # Collect policy: calling handle_unsafe directly records a security finding.
        policy_collect = SanitizationPolicy(
            allowed_tags={"p"},
            allowed_attributes={},
            url_policy=UrlPolicy(allow_rules={}),
            unsafe_handling="collect",
        )
        policy_collect.reset_collected_security_errors()
        policy_collect.handle_unsafe("Unsafe tag 'x'", node=None)
        errs = policy_collect.collected_security_errors()
        assert len(errs) == 1

    def test_unsafe_handler_reset_removes_security_errors_from_shared_sink(self) -> None:
        sink: list[ParseError] = [
            ParseError("x", category="security", message="s"),
            ParseError("y", category="tokenizer", message="t"),
            ParseError("z", category="security", message="s2"),
        ]

        handler = UnsafeHandler("collect", sink=sink)
        handler.reset()

        assert [e.category for e in sink] == ["tokenizer"]

    def test_unsafe_handler_collected_filters_security_from_shared_sink(self) -> None:
        sink: list[ParseError] = [
            ParseError("x", category="security", message="a", line=2, column=2),
            ParseError("y", category="tokenizer", message="t", line=1, column=1),
            ParseError("z", category="security", message="b", line=1, column=2),
        ]

        handler = UnsafeHandler("collect", sink=sink)
        out = handler.collected()

        assert [e.category for e in out] == ["security", "security"]
        assert [e.message for e in out] == ["b", "a"]

    def test_unsafe_handler_handle_writes_to_shared_sink(self) -> None:
        sink: list[ParseError] = []

        handler = UnsafeHandler("collect", sink=sink)
        handler.handle("Unsafe tag 'x'", node=None)

        assert len(sink) == 1
        assert sink[0].category == "security"
        assert sink[0].line is None
        assert sink[0].column is None

    def test_sanitize_unsafe_collects_security_errors(self) -> None:
        html = "<script>alert(1)</script>"
        node = JustHTML(html, fragment=True, track_node_locations=True, sanitize=False).root

        policy = SanitizationPolicy(
            allowed_tags={"p"},
            allowed_attributes={},
            url_policy=UrlPolicy(allow_rules={}),
            unsafe_handling="collect",
        )

        out = sanitize(node, policy=policy)
        assert to_html(out) == ""

        errors = policy.collected_security_errors()
        assert len(errors) == 1
        assert errors[0].category == "security"
        assert errors[0].code == "unsafe-html"
        assert "Unsafe tag 'script'" in errors[0].message

    def test_collect_mode_merges_into_doc_errors(self) -> None:
        html = "<p>\x00</p><script>alert(1)</script>"
        policy = SanitizationPolicy(
            allowed_tags=set(DEFAULT_POLICY.allowed_tags),
            allowed_attributes=DEFAULT_POLICY.allowed_attributes,
            url_policy=DEFAULT_POLICY.url_policy,
            allowed_css_properties=DEFAULT_POLICY.allowed_css_properties,
            force_link_rel=DEFAULT_POLICY.force_link_rel,
            unsafe_handling="collect",
        )

        doc = JustHTML(
            html,
            fragment=True,
            collect_errors=True,
            track_node_locations=True,
            policy=policy,
        )
        assert any(e.category == "tokenizer" for e in doc.errors)
        assert any(e.category == "security" for e in doc.errors)

        # Repeated serialization should not accumulate duplicates.
        before = len([e for e in doc.errors if e.category == "security"])
        _ = doc.to_html(pretty=False)
        _ = doc.to_html(pretty=False)
        after = len([e for e in doc.errors if e.category == "security"])
        assert before == after

    def test_collect_mode_merges_into_doc_errors_text_and_markdown(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=set(DEFAULT_POLICY.allowed_tags),
            allowed_attributes=DEFAULT_POLICY.allowed_attributes,
            url_policy=DEFAULT_POLICY.url_policy,
            allowed_css_properties=DEFAULT_POLICY.allowed_css_properties,
            force_link_rel=DEFAULT_POLICY.force_link_rel,
            unsafe_handling="collect",
        )

        doc = JustHTML(
            "<p>ok</p><script>alert(1)</script>",
            fragment=True,
            track_node_locations=True,
            policy=policy,
        )

        _ = doc.to_text()
        assert any(e.category == "security" for e in doc.errors)

        _ = doc.to_markdown()
        assert any(e.category == "security" for e in doc.errors)

    def test_justhtml_serialization_clears_stale_security_errors_and_sanitize_false_paths(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags=set(DEFAULT_POLICY.allowed_tags),
            allowed_attributes=DEFAULT_POLICY.allowed_attributes,
            url_policy=DEFAULT_POLICY.url_policy,
            allowed_css_properties=DEFAULT_POLICY.allowed_css_properties,
            force_link_rel=DEFAULT_POLICY.force_link_rel,
            unsafe_handling="collect",
        )

        doc_collect = JustHTML(
            "<p>ok</p><script>alert(1)</script>",
            fragment=True,
            track_node_locations=True,
            policy=policy,
        )
        assert any(e.category == "security" for e in doc_collect.errors)

        doc_default = JustHTML(
            "<p>ok</p><script>alert(1)</script>",
            fragment=True,
            track_node_locations=True,
        )
        assert not any(e.category == "security" for e in doc_default.errors)

        # sanitize=False documents should still serialize without crashing.
        doc_unsafe = JustHTML(
            "<p>ok</p><script>alert(1)</script>",
            fragment=True,
            track_node_locations=True,
            sanitize=False,
        )
        _ = doc_unsafe.to_html(pretty=False)
        _ = doc_unsafe.to_text()
        _ = doc_unsafe.to_markdown()

    def test_sanitize_unsafe_raises(self) -> None:
        html = "<script>alert(1)</script>"
        node = JustHTML(html, fragment=True, sanitize=False).root

        # Default behavior: script is removed
        sanitized = sanitize(node)
        assert to_html(sanitized) == ""

        # New behavior: raise exception
        policy = SanitizationPolicy(
            allowed_tags={"p"},
            allowed_attributes={},
            url_policy=UrlPolicy(allow_rules={}),
            unsafe_handling="raise",
        )

        with self.assertRaisesRegex(ValueError, "Unsafe tag"):
            sanitize(node, policy=policy)

    def test_sanitize_unsafe_attribute_raises(self) -> None:
        html = '<p onclick="alert(1)">Hello</p>'
        node = JustHTML(html, fragment=True, sanitize=False).root

        policy = SanitizationPolicy(
            allowed_tags={"p"},
            allowed_attributes={"p": set()},
            url_policy=UrlPolicy(allow_rules={}),
            unsafe_handling="raise",
        )

        with self.assertRaisesRegex(ValueError, "Unsafe attribute.*matched forbidden pattern"):
            sanitize(node, policy=policy)

    def test_sanitize_unsafe_url_raises(self) -> None:
        html = '<a href="javascript:alert(1)">Link</a>'
        node = JustHTML(html, fragment=True, sanitize=False).root

        policy = SanitizationPolicy(
            allowed_tags={"a"},
            allowed_attributes={"a": {"href"}},
            url_policy=UrlPolicy(allow_rules={("a", "href"): UrlRule(allowed_schemes={"https"})}),
            unsafe_handling="raise",
        )

        with self.assertRaisesRegex(ValueError, "Unsafe URL"):
            sanitize(node, policy=policy)

    def test_sanitize_unsafe_namespaced_attribute_raises(self) -> None:
        html = '<p xlink:href="foo">Hello</p>'
        node = JustHTML(html, fragment=True, sanitize=False).root
        policy = SanitizationPolicy(
            allowed_tags={"p"},
            allowed_attributes={"p": set()},
            url_policy=UrlPolicy(allow_rules={}),
            unsafe_handling="raise",
        )
        with self.assertRaisesRegex(ValueError, "Unsafe attribute.*xlink:href.*matched forbidden pattern"):
            sanitize(node, policy=policy)

    def test_sanitize_unsafe_srcdoc_attribute_raises(self) -> None:
        html = '<iframe srcdoc="<script>"></iframe>'
        node = JustHTML(html, fragment=True, sanitize=False).root
        policy = SanitizationPolicy(
            allowed_tags={"iframe"},
            allowed_attributes={"iframe": {"srcdoc"}},  # Even if allowed, srcdoc is dangerous
            url_policy=UrlPolicy(allow_rules={}),
            unsafe_handling="raise",
        )
        with self.assertRaisesRegex(ValueError, "Unsafe attribute.*srcdoc.*matched forbidden pattern"):
            sanitize(node, policy=policy)

    def test_sanitize_unsafe_disallowed_attribute_raises(self) -> None:
        html = '<p foo="bar">Hello</p>'
        node = JustHTML(html, fragment=True, sanitize=False).root
        policy = SanitizationPolicy(
            allowed_tags={"p"},
            allowed_attributes={"p": set()},  # No attributes allowed
            url_policy=UrlPolicy(allow_rules={}),
            unsafe_handling="raise",
        )
        with self.assertRaisesRegex(ValueError, "Unsafe attribute.*not allowed"):
            sanitize(node, policy=policy)

    def test_sanitize_unsafe_inline_style_raises(self) -> None:
        html = '<p style="background: url(javascript:alert(1))">Hello</p>'
        node = JustHTML(html, fragment=True, sanitize=False).root
        policy = SanitizationPolicy(
            allowed_tags={"p"},
            allowed_attributes={"p": {"style"}},
            allowed_css_properties={"background"},
            url_policy=UrlPolicy(allow_rules={}),
            unsafe_handling="raise",
        )
        with self.assertRaisesRegex(ValueError, "Unsafe inline style"):
            sanitize(node, policy=policy)

    def test_sanitize_unsafe_root_tag_raises(self) -> None:
        # Test disallowed tag as root
        html = "<div>Content</div>"
        node = JustHTML(html, fragment=True, sanitize=False).root
        # node is a div (because fragment=True parses into a list of nodes, but JustHTML.root wraps them?
        # Wait, JustHTML(fragment=True).root is a DocumentFragment containing the nodes.
        # sanitize() on a DocumentFragment iterates children.
        # To test root handling, we need to pass the element directly.

        assert node.children is not None
        div = node.children[0]
        policy = SanitizationPolicy(
            allowed_tags={"p"},
            allowed_attributes={},
            url_policy=UrlPolicy(allow_rules={}),
            unsafe_handling="raise",
        )
        with self.assertRaisesRegex(ValueError, "Unsafe tag.*not allowed"):
            sanitize(div, policy=policy)

    def test_sanitize_unsafe_root_dropped_content_raises(self) -> None:
        html = "<script>alert(1)</script>"
        node = JustHTML(html, fragment=True, sanitize=False).root
        assert node.children is not None
        script = node.children[0]

        policy = SanitizationPolicy(
            allowed_tags={"p"},
            allowed_attributes={},
            url_policy=UrlPolicy(allow_rules={}),
            unsafe_handling="raise",
        )
        with self.assertRaisesRegex(ValueError, "Unsafe tag.*dropped content"):
            sanitize(script, policy=policy)

    def test_sanitize_unsafe_child_dropped_content_raises(self) -> None:
        html = "<div><script>alert(1)</script></div>"
        node = JustHTML(html, fragment=True, sanitize=False).root
        assert node.children is not None
        div = node.children[0]

        policy = SanitizationPolicy(
            allowed_tags={"div"},
            allowed_attributes={"div": set()},
            url_policy=UrlPolicy(allow_rules={}),
            unsafe_handling="raise",
        )
        with self.assertRaisesRegex(ValueError, "Unsafe tag.*dropped content"):
            sanitize(div, policy=policy)

    def test_sanitize_unsafe_child_disallowed_tag_raises(self) -> None:
        html = "<div><foo></foo></div>"
        node = JustHTML(html, fragment=True, sanitize=False).root
        assert node.children is not None
        div = node.children[0]

        policy = SanitizationPolicy(
            allowed_tags={"div"},
            allowed_attributes={"div": set()},
            url_policy=UrlPolicy(allow_rules={}),
            unsafe_handling="raise",
        )
        with self.assertRaisesRegex(ValueError, "Unsafe tag.*not allowed"):
            sanitize(div, policy=policy)

    def test_sanitize_unsafe_root_foreign_namespace_raises(self) -> None:
        # <svg> puts elements in SVG namespace
        html = "<svg><title>foo</title></svg>"
        node = JustHTML(html, fragment=True, sanitize=False).root
        assert node.children is not None
        svg = node.children[0]

        policy = SanitizationPolicy(
            allowed_tags={"svg"},  # Even if allowed, foreign namespaces might be dropped
            allowed_attributes={"svg": set()},
            url_policy=UrlPolicy(allow_rules={}),
            drop_foreign_namespaces=True,
            unsafe_handling="raise",
        )
        with self.assertRaisesRegex(ValueError, "Unsafe tag.*foreign namespace"):
            sanitize(svg, policy=policy)

    def test_sanitize_unsafe_child_foreign_namespace_raises(self) -> None:
        html = "<div><svg></svg></div>"
        node = JustHTML(html, fragment=True, sanitize=False).root
        assert node.children is not None
        div = node.children[0]

        policy = SanitizationPolicy(
            allowed_tags={"div"},
            allowed_attributes={"div": set()},
            url_policy=UrlPolicy(allow_rules={}),
            drop_foreign_namespaces=True,
            unsafe_handling="raise",
        )
        with self.assertRaisesRegex(ValueError, "Unsafe tag.*foreign namespace"):
            sanitize(div, policy=policy)

    def test_sanitize_unsafe_root_disallowed_raises(self) -> None:
        html = "<x-foo></x-foo>"
        node = JustHTML(html, fragment=True, sanitize=False).root
        assert node.children is not None
        xfoo = node.children[0]

        policy = SanitizationPolicy(
            allowed_tags={"p"},
            allowed_attributes={},
            url_policy=UrlPolicy(allow_rules={}),
            unsafe_handling="raise",
        )
        with self.assertRaisesRegex(ValueError, "Unsafe tag.*not allowed"):
            sanitize(xfoo, policy=policy)

    def test_sanitize_escape_disallowed_template_preserves_children(self) -> None:
        html = "<template><b>x</b></template>"
        policy = SanitizationPolicy(
            allowed_tags={"b"},
            allowed_attributes={"*": set(), "b": set()},
            url_policy=UrlPolicy(allow_rules={}),
            disallowed_tag_handling="escape",
        )
        out = JustHTML(html, fragment=True, policy=policy).to_html(pretty=False)
        assert out == "&lt;template&gt;<b>x</b>&lt;/template&gt;"

    def test_sanitize_escape_disallowed_without_source_html(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags={"p"},
            allowed_attributes={"*": set(), "p": set()},
            url_policy=UrlPolicy(allow_rules={}),
            disallowed_tag_handling="escape",
        )
        node = Element("x", {}, "html")
        node._start_tag_start = 0
        node._start_tag_end = 2
        node.append_child(Text("ok"))
        sanitized = sanitize(node, policy=policy)
        assert to_html(sanitized, pretty=False) == "&lt;x&gt;ok"

    def test_sanitize_escape_disallowed_reconstructs_end_tag_without_source_html(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags={"p"},
            allowed_attributes={"*": set(), "p": set()},
            url_policy=UrlPolicy(allow_rules={}),
            disallowed_tag_handling="escape",
        )
        node = Element("x", {}, "html")
        node._start_tag_start = 0
        node._start_tag_end = 3
        node._end_tag_start = 5
        node._end_tag_end = 9
        node._end_tag_present = True
        node.append_child(Text("ok"))
        sanitized = sanitize(node, policy=policy)
        assert to_html(sanitized, pretty=False) == "&lt;x&gt;ok&lt;/x&gt;"

    def test_sanitize_escape_disallowed_reconstructs_self_closing_without_source_html(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags={"p"},
            allowed_attributes={"*": set(), "p": set()},
            url_policy=UrlPolicy(allow_rules={}),
            disallowed_tag_handling="escape",
        )
        node = Element("x", {"a": "b"}, "html")
        node._start_tag_start = 0
        node._start_tag_end = 3
        node._self_closing = True
        sanitized = sanitize(node, policy=policy)
        out = to_html(sanitized, pretty=False)
        assert out.startswith("&lt;x")
        assert 'a="b"' in out
        assert out.endswith("/&gt;")

    def test_sanitize_escape_disallowed_can_inherit_source_html_from_parent(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags={"p"},
            allowed_attributes={"*": set(), "p": set()},
            url_policy=UrlPolicy(allow_rules={}),
            disallowed_tag_handling="escape",
        )

        root = DocumentFragment()
        root._source_html = "<x>hi</x>"

        node = Element("x", {}, "html")
        node._start_tag_start = 0
        node._start_tag_end = 3
        node._end_tag_start = 5
        node._end_tag_end = 9
        node._end_tag_present = True
        node.append_child(Text("hi"))

        # Ensure tag extraction has to walk up to the parent.
        node._source_html = None
        root.append_child(node)

        sanitized = sanitize(root, policy=policy)
        assert to_html(sanitized, pretty=False) == "&lt;x&gt;hi&lt;/x&gt;"
        assert node._source_html == root._source_html

    def test_sanitize_escape_disallowed_inherits_source_html_for_template_content(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags={"p"},
            allowed_attributes={"*": set(), "p": set()},
            url_policy=UrlPolicy(allow_rules={}),
            disallowed_tag_handling="escape",
        )

        root = DocumentFragment()
        root._source_html = "<template>x</template>"

        # Template has a template_content container in the HTML namespace.
        template = Template("template", namespace="html")
        assert template.template_content is not None

        # Ensure inheritance has to walk up (both template and template_content
        # start without source html).
        template._source_html = None
        template.template_content._source_html = None
        root.append_child(template)

        sanitize(root, policy=policy)
        assert template.template_content._source_html == root._source_html

    def test_sanitize_escape_disallowed_does_not_override_existing_template_content_source_html(self) -> None:
        policy = SanitizationPolicy(
            allowed_tags={"p"},
            allowed_attributes={"*": set(), "p": set()},
            url_policy=UrlPolicy(allow_rules={}),
            disallowed_tag_handling="escape",
        )

        root = DocumentFragment()
        root._source_html = "<template>x</template>"

        template = Template("template", namespace="html")
        assert template.template_content is not None

        # Cover the branch where the child already has source html (no overwrite).
        template._source_html = "<template>own</template>"
        template.template_content._source_html = "tc"
        root.append_child(template)

        sanitize(root, policy=policy)
        assert template._source_html == "<template>own</template>"
        assert template.template_content._source_html == "tc"


if __name__ == "__main__":
    unittest.main()
