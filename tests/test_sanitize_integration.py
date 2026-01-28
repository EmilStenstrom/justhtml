from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

from justhtml import DEFAULT_POLICY, JustHTML
from justhtml.context import FragmentContext
from justhtml.sanitize import SanitizationPolicy, UrlPolicy, UrlProxy, UrlRule

_CASES_DIR = Path(__file__).with_name("justhtml-sanitize-tests")


def _url_filter_by_name(name: str):
    if name == "drop_or_rewrite":

        def url_filter(tag: str, attr: str, value: str) -> str | None:
            if tag == "a" and attr == "href" and value == "https://drop.me":
                return None
            if tag == "a" and attr == "href" and value == "https://rewrite.me":
                return "https://example.com"
            return value

        return url_filter

    raise ValueError(f"Unknown url_filter name: {name}")


def _build_policy(spec: Any) -> SanitizationPolicy:
    if spec == "DEFAULT":
        return DEFAULT_POLICY

    if not isinstance(spec, dict):
        raise TypeError("policy must be 'DEFAULT' or an object")

    allowed_tags = spec["allowed_tags"]
    allowed_attributes = spec["allowed_attributes"]

    url_rules_list = spec.get("url_rules", [])
    url_rules: dict[tuple[str, str], UrlRule] = {}
    for rule_spec in url_rules_list:
        if not isinstance(rule_spec, dict):
            raise TypeError("url_rules entries must be objects")
        tag = rule_spec["tag"]
        attr = rule_spec["attr"]
        proxy_url = rule_spec.get("proxy_url")
        handling = rule_spec.get("handling")
        if handling is None and proxy_url is not None:
            handling = "proxy"
        if handling is None:
            handling = "allow"

        url_rules[(tag, attr)] = UrlRule(
            allow_fragment=rule_spec.get("allow_fragment", True),
            resolve_protocol_relative=rule_spec.get("resolve_protocol_relative", "https"),
            allowed_schemes=rule_spec.get("allowed_schemes", []),
            allowed_hosts=rule_spec.get("allowed_hosts", None),
            handling=handling,
            allow_relative=rule_spec.get("allow_relative", None),
            proxy=(
                UrlProxy(url=str(proxy_url), param=str(rule_spec.get("proxy_param", "url")))
                if proxy_url is not None
                else None
            ),
        )

    url_filter_name = spec.get("url_filter")
    url_filter = _url_filter_by_name(url_filter_name) if isinstance(url_filter_name, str) else None

    # Map the old test schema to the new API shape.
    url_policy = UrlPolicy(
        default_handling=str(spec.get("default_handling", "strip")),
        allow_rules=url_rules,
        url_filter=url_filter,
    )

    return SanitizationPolicy(
        allowed_tags=allowed_tags,
        allowed_attributes=allowed_attributes,
        url_policy=url_policy,
        drop_comments=spec.get("drop_comments", True),
        drop_doctype=spec.get("drop_doctype", True),
        drop_foreign_namespaces=spec.get("drop_foreign_namespaces", True),
        drop_content_tags=spec.get("drop_content_tags", ["script", "style"]),
        force_link_rel=spec.get("force_link_rel", []),
        allowed_css_properties=spec.get("allowed_css_properties", []),
        disallowed_tag_handling=spec.get("disallowed_tag_handling", "unwrap"),
    )


class TestSanitizeIntegration(unittest.TestCase):
    def test_sanitize_cases(self) -> None:
        cases_path = _CASES_DIR / "cases.json"
        cases = json.loads(cases_path.read_text(encoding="utf-8"))
        if not isinstance(cases, list):
            raise TypeError("cases.json must contain a list")

        for case in cases:
            name = case["name"]
            expected_error = case.get("expected_error")
            if expected_error is not None:
                with self.assertRaises(ValueError) as ctx:
                    _build_policy(case["policy"])
                msg = str(ctx.exception)
                if str(expected_error) not in msg:
                    self.fail(
                        "\n".join(
                            [
                                f"Case: {name}",
                                f"Expected error containing: {expected_error}",
                                f"Actual error: {msg}",
                            ]
                        )
                    )
                continue

            policy = _build_policy(case["policy"])
            input_html = case["input_html"]
            expected_html = case["expected_html"]

            parse_mode = case.get("parse", "fragment")
            if parse_mode == "fragment":
                ctx = case.get("fragment_context", "div")
                doc = JustHTML(
                    input_html,
                    fragment_context=FragmentContext(ctx),
                    policy=policy,
                )
            elif parse_mode == "document":
                doc = JustHTML(input_html, policy=policy)
            else:
                raise ValueError(f"Unknown parse mode in {name}: {parse_mode}")

            actual = doc.to_html(pretty=False)
            if actual != expected_html:
                self.fail(
                    "\n".join(
                        [
                            f"Case: {name}",
                            f"Input: {input_html}",
                            f"Expected: {expected_html}",
                            f"Actual:   {actual}",
                        ]
                    )
                )


if __name__ == "__main__":
    unittest.main()
