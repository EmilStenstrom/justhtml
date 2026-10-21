"""Whitespace controls shared by behavior-specific HTML/CSS tests."""

import sys

ASCII_WHITESPACE = "\t\n\f\r "
NON_HTML_WHITESPACE = tuple(
    char
    for codepoint in range(sys.maxunicode + 1)
    if (char := chr(codepoint)).isspace() and char not in ASCII_WHITESPACE
)
