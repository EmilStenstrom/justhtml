"""HTML sanitization policy API.

This module defines the public API for JustHTML sanitization.

The sanitizer operates on the parsed JustHTML DOM and is intentionally
policy-driven.
"""

from __future__ import annotations

from .css import (
    _css_value_contains_disallowed_functions,
    _css_value_has_disallowed_resource_functions,
    _css_value_may_load_external_resource,
    _is_valid_css_property_name,
    _lookup_css_url_rule,
    _sanitize_css_url_functions,
    _sanitize_inline_style,
    _sanitize_url_function_value,
)
from .dom import _sanitize, sanitize_dom
from .policy import (
    CSS_PRESET_TEXT,
    DEFAULT_DOCUMENT_POLICY,
    DEFAULT_POLICY,
    ALLOW_ALL_TAGS,
    CompiledSanitizationPolicy,
    SanitizationPolicy,
    UnsafeHandler,
    _seal_url_policy,
)
from .rawtext import (
    _neutralize_rawtext_end_tag_sequences,
    _sanitize_rawtext_element_contents,
)
from .url import (
    _URL_BEARING_PARAM_NAMES,
    _URL_LIKE_ATTRS,
    DisallowedTagHandling,
    UnsafeHandling,
    UnsafeHtmlError,
    UrlFilter,
    UrlHandling,
    UrlPolicy,
    UrlProxy,
    UrlRule,
    _effective_allow_relative,
    _effective_proxy,
    _effective_url_handling,
    _is_legacy_ipv4_number,
    _is_noncanonical_numeric_ipv4_host,
    _prepare_standalone_url_value_for_checking,
    _raw_authority_host,
    _sanitize_comma_or_space_separated_url_list,
    _sanitize_space_separated_url_list,
    _sanitize_srcset_value,
    _sanitize_url_value_with_rule,
    _strip_invisible_unicode,
)

__all__ = [
    "ALLOW_ALL_TAGS",
    "CSS_PRESET_TEXT",
    "DEFAULT_DOCUMENT_POLICY",
    "DEFAULT_POLICY",
    "_URL_BEARING_PARAM_NAMES",
    "_URL_LIKE_ATTRS",
    "CompiledSanitizationPolicy",
    "DisallowedTagHandling",
    "SanitizationPolicy",
    "UnsafeHandler",
    "UnsafeHandling",
    "UnsafeHtmlError",
    "UrlFilter",
    "UrlHandling",
    "UrlPolicy",
    "UrlProxy",
    "UrlRule",
    "_css_value_contains_disallowed_functions",
    "_css_value_has_disallowed_resource_functions",
    "_css_value_may_load_external_resource",
    "_effective_allow_relative",
    "_effective_proxy",
    "_effective_url_handling",
    "_is_legacy_ipv4_number",
    "_is_noncanonical_numeric_ipv4_host",
    "_is_valid_css_property_name",
    "_lookup_css_url_rule",
    "_neutralize_rawtext_end_tag_sequences",
    "_prepare_standalone_url_value_for_checking",
    "_raw_authority_host",
    "_sanitize",
    "_sanitize_comma_or_space_separated_url_list",
    "_sanitize_css_url_functions",
    "_sanitize_inline_style",
    "_sanitize_rawtext_element_contents",
    "_sanitize_space_separated_url_list",
    "_sanitize_srcset_value",
    "_sanitize_url_function_value",
    "_sanitize_url_value_with_rule",
    "_seal_url_policy",
    "_strip_invisible_unicode",
    "sanitize_dom",
]
