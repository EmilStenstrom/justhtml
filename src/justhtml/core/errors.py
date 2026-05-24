"""Centralized error message definitions and helpers for JustHTML errors.

This module provides human-readable error messages for parse error codes
emitted by the tokenizer and tree builder during HTML parsing, plus selected
security findings emitted by the sanitizer.
"""

from __future__ import annotations


def generate_error_message(code: str, tag_name: str | None = None) -> str:
    """Generate human-readable error message from error code.

    Args:
        code: The error code string (kebab-case format)
        tag_name: Optional tag name to include in the message for context

    Returns:
        Human-readable error message string
    """
    messages = {
        # ================================================================
        # TOKENIZER ERRORS
        # ================================================================
        # DOCTYPE errors
        "eof-in-doctype": "Unexpected end of file in DOCTYPE declaration",
        "eof-in-doctype-name": "Unexpected end of file while reading DOCTYPE name",
        "eof-in-doctype-public-identifier": "Unexpected end of file in DOCTYPE public identifier",
        "eof-in-doctype-system-identifier": "Unexpected end of file in DOCTYPE system identifier",
        "expected-doctype-name-but-got-right-bracket": "Expected DOCTYPE name but got >",
        "missing-whitespace-before-doctype-name": "Missing whitespace after <!DOCTYPE",
        "abrupt-doctype-public-identifier": "DOCTYPE public identifier ended abruptly",
        "abrupt-doctype-system-identifier": "DOCTYPE system identifier ended abruptly",
        "missing-quote-before-doctype-public-identifier": "Missing quote before DOCTYPE public identifier",
        "missing-quote-before-doctype-system-identifier": "Missing quote before DOCTYPE system identifier",
        "missing-doctype-public-identifier": "Missing DOCTYPE public identifier",
        "missing-doctype-system-identifier": "Missing DOCTYPE system identifier",
        "missing-whitespace-before-doctype-public-identifier": "Missing whitespace before DOCTYPE public identifier",
        "missing-whitespace-after-doctype-public-identifier": "Missing whitespace after DOCTYPE public identifier",
        "missing-whitespace-between-doctype-public-and-system-identifiers": "Missing whitespace between DOCTYPE identifiers",
        "missing-whitespace-after-doctype-name": "Missing whitespace after DOCTYPE name",
        "unexpected-character-after-doctype-public-keyword": "Unexpected character after PUBLIC keyword",
        "unexpected-character-after-doctype-system-keyword": "Unexpected character after SYSTEM keyword",
        "unexpected-character-after-doctype-public-identifier": "Unexpected character after public identifier",
        "unexpected-character-after-doctype-system-identifier": "Unexpected character after system identifier",
        # Comment errors
        "eof-in-comment": "Unexpected end of file in comment",
        "abrupt-closing-of-empty-comment": "Comment ended abruptly with -->",
        "incorrectly-closed-comment": "Comment ended with --!> instead of -->",
        # Tag errors
        "eof-in-tag": "Unexpected end of file in tag",
        "eof-before-tag-name": "Unexpected end of file before tag name",
        "empty-end-tag": "Empty end tag </> is not allowed",
        "invalid-first-character-of-tag-name": "Invalid first character of tag name",
        "unexpected-question-mark-instead-of-tag-name": "Unexpected ? instead of tag name",
        "unexpected-character-after-solidus-in-tag": "Unexpected character after / in tag",
        # Attribute errors
        "duplicate-attribute": "Duplicate attribute name",
        "missing-attribute-value": "Missing attribute value",
        "unexpected-character-in-attribute-name": "Unexpected character in attribute name",
        "unexpected-character-in-unquoted-attribute-value": "Unexpected character in unquoted attribute value",
        "missing-whitespace-between-attributes": "Missing whitespace between attributes",
        "unexpected-equals-sign-before-attribute-name": "Unexpected = before attribute name",
        # Script errors
        "eof-in-script-html-comment-like-text": "Unexpected end of file in script with HTML-like comment",
        "eof-in-script-in-script": "Unexpected end of file in nested script tag",
        # CDATA errors
        "eof-in-cdata": "Unexpected end of file in CDATA section",
        "cdata-in-html-content": "CDATA section only allowed in SVG/MathML content",
        # NULL character errors
        "unexpected-null-character": "Unexpected NULL character (U+0000)",
        # Markup declaration errors
        "incorrectly-opened-comment": "Incorrectly opened comment",
        # Character reference errors
        "control-character-reference": "Invalid control character in character reference",
        "illegal-codepoint-for-numeric-entity": "Invalid codepoint in numeric character reference",
        "missing-semicolon-after-character-reference": "Missing semicolon after character reference",
        "named-entity-without-semicolon": "Named entity used without semicolon",
        "noncharacter-character-reference": "Noncharacter in character reference",
        "noncharacter-in-input-stream": "Noncharacter in input stream",
        # ================================================================
        # TREE BUILDER ERRORS
        # ================================================================
        # DOCTYPE errors
        "unexpected-doctype": "Unexpected DOCTYPE declaration",
        "unknown-doctype": "Unknown DOCTYPE (expected <!DOCTYPE html>)",
        "expected-doctype-but-got-chars": "Expected DOCTYPE but got text content",
        "expected-doctype-but-got-eof": "Expected DOCTYPE but reached end of file",
        "expected-doctype-but-got-start-tag": f"Expected DOCTYPE but got <{tag_name}> tag",
        "expected-doctype-but-got-end-tag": f"Expected DOCTYPE but got </{tag_name}> tag",
        "unexpected-doctype-in-foreign-content": "Unexpected DOCTYPE in SVG/MathML content",
        # Unexpected tag errors
        "unexpected-start-tag": f"Unexpected <{tag_name}> start tag",
        "unexpected-end-tag": f"Unexpected </{tag_name}> end tag",
        "unexpected-end-tag-before-html": f"Unexpected </{tag_name}> end tag before <html>",
        "unexpected-end-tag-before-head": f"Unexpected </{tag_name}> end tag before <head>",
        "unexpected-end-tag-after-head": f"Unexpected </{tag_name}> end tag after <head>",
        "unexpected-start-tag-ignored": f"<{tag_name}> start tag ignored in current context",
        "unexpected-start-tag-implies-end-tag": f"<{tag_name}> start tag implicitly closes previous element",
        # EOF errors
        "expected-closing-tag-but-got-eof": f"Expected </{tag_name}> closing tag but reached end of file",
        "expected-named-closing-tag-but-got-eof": f"Expected </{tag_name}> closing tag but reached end of file",
        # Invalid character errors
        "invalid-codepoint": "Invalid character (U+0000 NULL or U+000C FORM FEED)",
        "invalid-codepoint-before-head": "Invalid character before <head>",
        "invalid-codepoint-in-body": "Invalid character in <body>",
        "invalid-codepoint-in-table-text": "Invalid character in table text",
        "invalid-codepoint-in-select": "Invalid character in <select>",
        "invalid-codepoint-in-foreign-content": "Invalid character in SVG/MathML content",
        # Foster parenting / table errors
        "foster-parenting-character": "Text content in table requires foster parenting",
        "foster-parenting-start-tag": "Start tag in table requires foster parenting",
        "unexpected-character-implies-table-voodoo": "Unexpected character in table triggers foster parenting",
        "unexpected-start-tag-implies-table-voodoo": f"<{tag_name}> start tag in table triggers foster parenting",
        "unexpected-end-tag-implies-table-voodoo": f"</{tag_name}> end tag in table triggers foster parenting",
        "unexpected-implied-end-tag-in-table-view": "Unexpected implied end tag while closing table",
        "eof-in-table": "Unexpected end of file in table",
        "unexpected-cell-in-table-body": "Unexpected table cell outside of table row",
        "unexpected-form-in-table": "Form element not allowed in table context",
        "unexpected-hidden-input-in-table": "Hidden input in table triggers foster parenting",
        # Context-specific errors
        "unexpected-hidden-input-after-head": "Unexpected hidden input after <head>",
        "unexpected-token-in-frameset": "Unexpected content in <frameset>",
        "unexpected-token-after-frameset": "Unexpected content after <frameset>",
        "unexpected-token-after-after-frameset": "Unexpected content after frameset closed",
        "unexpected-token-after-body": "Unexpected content after </body>",
        "unexpected-char-after-body": "Unexpected character after </body>",
        "unexpected-characters-in-column-group": "Text not allowed in <colgroup>",
        "unexpected-characters-in-template-column-group": "Text not allowed in template column group",
        "unexpected-start-tag-in-column-group": f"<{tag_name}> start tag not allowed in <colgroup>",
        "unexpected-start-tag-in-template-column-group": f"<{tag_name}> start tag not allowed in template column group",
        "unexpected-start-tag-in-template-table-context": f"<{tag_name}> start tag not allowed in template table context",
        "unexpected-start-tag-in-cell-fragment": f"<{tag_name}> start tag not allowed in cell fragment context",
        # Foreign content errors
        "unexpected-html-element-in-foreign-content": "HTML element breaks out of SVG/MathML content",
        "unexpected-end-tag-in-foreign-content": f"Mismatched </{tag_name}> end tag in SVG/MathML content",
        "unexpected-end-tag-in-fragment-context": f"</{tag_name}> end tag not allowed in fragment parsing context",
        # Miscellaneous errors
        "end-tag-too-early": f"</{tag_name}> end tag closed early (unclosed children)",
        "adoption-agency-1.3": "Misnested tags require adoption agency algorithm",
        "non-void-html-element-start-tag-with-trailing-solidus": f"<{tag_name}/> self-closing syntax on non-void element",
        "image-start-tag": f"Deprecated <{tag_name}> tag (use <img> instead)",
        # Select insertion mode (context-specific taxonomy)
        "unexpected-start-tag-in-select": f"Unexpected <{tag_name}> start tag in <select>",
        "unexpected-end-tag-in-select": f"Unexpected </{tag_name}> end tag in <select>",
        "unexpected-select-in-select": "Unexpected nested <select> in <select>",
        # ================================================================
        # SECURITY ERRORS
        # ================================================================
        "unsafe-html": "Unsafe HTML detected by sanitization policy",
    }

    # Return message or fall back to the code itself if not found
    return messages.get(code, code)
