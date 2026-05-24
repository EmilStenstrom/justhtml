"""Tokenizer facade for JustHTML."""

from .html import (
    Tokenizer as Tokenizer,
)
from .html import (
    TokenizerOpts as TokenizerOpts,
)
from .html import (
    _coerce_comment_for_xml as _coerce_comment_for_xml,
)
from .html import (
    _coerce_text_for_xml as _coerce_text_for_xml,
)
from .html import (
    _is_noncharacter_codepoint as _is_noncharacter_codepoint,
)
from .html import (
    _xml_coercion_callback as _xml_coercion_callback,
)
