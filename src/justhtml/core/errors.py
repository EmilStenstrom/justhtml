"""Error-code definitions and human-readable messages."""

PARSER_ERROR_CODES = frozenset(
    {
        "eof-in-comment",
        "eof-in-tag",
        "expected-doctype-but-got-chars",
        "expected-doctype-but-got-start-tag",
        "unexpected-end-tag",
        "unexpected-null-character",
        "unknown-doctype",
    }
)

SECURITY_ERROR_CODES = frozenset(
    {
        "unsafe-html",
        "unsafe-rawtext-child",
        "unsafe-rawtext-end-tag",
        "unsafe-style-resource",
    }
)


def generate_error_message(code: str, tag_name: str | None = None) -> str:
    """Generate a human-readable message for a built-in error code."""
    messages = {
        "eof-in-comment": "Unexpected end of file in comment",
        "eof-in-tag": "Unexpected end of file in tag",
        "expected-doctype-but-got-chars": "Expected DOCTYPE but got text content",
        "expected-doctype-but-got-start-tag": f"Expected DOCTYPE but got <{tag_name}> tag",
        "unexpected-end-tag": f"Unexpected </{tag_name}> end tag",
        "unexpected-null-character": "Unexpected NULL character (U+0000)",
        "unknown-doctype": "Unknown DOCTYPE (expected <!DOCTYPE html>)",
        "unsafe-html": "Unsafe HTML detected by sanitization policy",
        "unsafe-rawtext-child": "Unsafe non-text child in raw text element",
        "unsafe-rawtext-end-tag": "Unsafe closing tag sequence in raw text element",
        "unsafe-style-resource": "Unsafe resource-loading construct in style element",
    }
    return messages.get(code, code)
