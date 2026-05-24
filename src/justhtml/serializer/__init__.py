"""Serialization facade for JustHTML DOM nodes."""

from .html import (
    _FORMAT_SEP as _FORMAT_SEP,
)
from .html import (
    _LAYOUT_BLOCK_ELEMENTS as _LAYOUT_BLOCK_ELEMENTS,
)
from .html import (
    _LITERAL_TEXT_SERIALIZATION_ELEMENTS as _LITERAL_TEXT_SERIALIZATION_ELEMENTS,
)
from .html import (
    _SERIALIZABLE_ATTR_NAME_RE as _SERIALIZABLE_ATTR_NAME_RE,
)
from .html import (
    _SERIALIZABLE_TAG_NAME_RE as _SERIALIZABLE_TAG_NAME_RE,
)
from .html import (
    _UNQUOTED_ATTR_VALUE_INVALID as _UNQUOTED_ATTR_VALUE_INVALID,
)
from .html import (
    HTMLContext as HTMLContext,
)
from .html import (
    _attrs_to_test_format as _attrs_to_test_format,
)
from .html import (
    _can_unquote_attr_value as _can_unquote_attr_value,
)
from .html import (
    _choose_attr_quote as _choose_attr_quote,
)
from .html import (
    _collapse_html_whitespace as _collapse_html_whitespace,
)
from .html import (
    _doctype_to_test_format as _doctype_to_test_format,
)
from .html import (
    _escape_attr_value as _escape_attr_value,
)
from .html import (
    _escape_html_chars as _escape_html_chars,
)
from .html import (
    _escape_js_string as _escape_js_string,
)
from .html import (
    _escape_text as _escape_text,
)
from .html import (
    _escape_url_value as _escape_url_value,
)
from .html import (
    _is_blocky_element as _is_blocky_element,
)
from .html import (
    _is_formatting_whitespace_text as _is_formatting_whitespace_text,
)
from .html import (
    _is_layout_blocky_element as _is_layout_blocky_element,
)
from .html import (
    _is_whitespace_text_node as _is_whitespace_text_node,
)
from .html import (
    _neutralize_rawtext_end_tag_sequences as _neutralize_rawtext_end_tag_sequences,
)
from .html import (
    _node_to_html as _node_to_html,
)
from .html import (
    _node_to_html_compact as _node_to_html_compact,
)
from .html import (
    _node_to_test_format as _node_to_test_format,
)
from .html import (
    _normalize_formatting_whitespace as _normalize_formatting_whitespace,
)
from .html import (
    _pretty_renders_nonempty as _pretty_renders_nonempty,
)
from .html import (
    _qualified_name as _qualified_name,
)
from .html import (
    _serialize_comment_data as _serialize_comment_data,
)
from .html import (
    _serialize_doctype as _serialize_doctype,
)
from .html import (
    _serialize_text_for_parent as _serialize_text_for_parent,
)
from .html import (
    _should_pretty_indent_children as _should_pretty_indent_children,
)
from .html import (
    _validate_serializable_attr_name as _validate_serializable_attr_name,
)
from .html import (
    _validate_serializable_name as _validate_serializable_name,
)
from .html import (
    _validate_serializable_tag_name as _validate_serializable_tag_name,
)
from .html import (
    serialize_end_tag as serialize_end_tag,
)
from .html import (
    serialize_start_tag as serialize_start_tag,
)
from .html import (
    to_html as to_html,
)
from .html import (
    to_test_format as to_test_format,
)
