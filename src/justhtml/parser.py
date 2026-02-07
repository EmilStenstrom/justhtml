"""Minimal JustHTML parser entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .context import FragmentContext
from .encoding import decode_html
from .sanitize import UrlRule, _sanitize_url_value_with_rule
from .tokenizer import Tokenizer, TokenizerOpts
from .transforms import apply_compiled_transforms, compile_transforms
from .treebuilder import TreeBuilder

if TYPE_CHECKING:
    from .node import Node
    from .sanitize import SanitizationPolicy
    from .serialize import HTMLContext
    from .tokens import ParseError
    from .transforms import TransformSpec


class StrictModeError(SyntaxError):
    """Raised when strict mode encounters a parse error.

    Inherits from SyntaxError to provide Python 3.11+ enhanced error display
    with source location highlighting.
    """

    error: ParseError

    def __init__(self, error: ParseError) -> None:
        self.error = error
        # Use the ParseError's as_exception() to get enhanced display
        exc = error.as_exception()
        super().__init__(exc.msg)
        # Copy SyntaxError attributes for enhanced display
        self.filename = exc.filename
        self.lineno = exc.lineno
        self.offset = exc.offset
        self.text = exc.text
        self.end_lineno = getattr(exc, "end_lineno", None)
        self.end_offset = getattr(exc, "end_offset", None)


class JustHTML:
    __slots__ = ("debug", "encoding", "errors", "fragment_context", "root", "tokenizer", "tree_builder")

    debug: bool
    encoding: str | None
    errors: list[ParseError]
    fragment_context: FragmentContext | None
    root: Node
    tokenizer: Tokenizer
    tree_builder: TreeBuilder

    def __init__(
        self,
        html: str | bytes | bytearray | memoryview | None,
        *,
        sanitize: bool | None = None,
        safe: bool | None = None,
        policy: SanitizationPolicy | None = None,
        collect_errors: bool = False,
        track_node_locations: bool = False,
        debug: bool = False,
        encoding: str | None = None,
        fragment: bool = False,
        fragment_context: FragmentContext | None = None,
        iframe_srcdoc: bool = False,
        scripting_enabled: bool = True,
        strict: bool = False,
        tokenizer_opts: TokenizerOpts | None = None,
        tree_builder: TreeBuilder | None = None,
        transforms: list[TransformSpec] | None = None,
    ) -> None:
        # `sanitize` is the primary API (preferred). `safe` is kept as a
        # backwards-compatible alias.
        if sanitize is None and safe is None:
            sanitize_enabled = True
        elif sanitize is None and safe is not None:
            sanitize_enabled = bool(safe)
        elif sanitize is not None and safe is None:
            sanitize_enabled = bool(sanitize)
        else:
            sanitize_enabled = bool(sanitize)
            if sanitize_enabled != bool(safe):
                raise ValueError("Conflicting values for sanitize and safe; use only sanitize=")

        if fragment_context is not None:
            fragment = True

        if fragment and fragment_context is None:
            fragment_context = FragmentContext("div")

        track_tag_spans = False
        has_sanitize_transform = False
        needs_escape_incomplete_tags = False
        if transforms:
            from .sanitize import DEFAULT_POLICY  # noqa: PLC0415
            from .transforms import Sanitize  # noqa: PLC0415

            for t in transforms:
                if isinstance(t, Sanitize):
                    has_sanitize_transform = True
                    effective = t.policy or DEFAULT_POLICY
                    if effective.disallowed_tag_handling == "escape":
                        track_tag_spans = True
                        needs_escape_incomplete_tags = True
                        break

        # If we will auto-sanitize (sanitize=True and no Sanitize in transforms),
        # escape-mode tag reconstruction may require tracking tag spans.
        if sanitize_enabled and not has_sanitize_transform and policy is not None:
            if policy.disallowed_tag_handling == "escape":
                track_tag_spans = True
                needs_escape_incomplete_tags = True

        self.debug = bool(debug)
        self.fragment_context = fragment_context
        self.encoding = None

        html_str: str
        if isinstance(html, (bytes, bytearray, memoryview)):
            html_str, chosen = decode_html(bytes(html), transport_encoding=encoding)
            self.encoding = chosen
        elif html is not None:
            html_str = str(html)
        else:
            html_str = ""

        # Enable error collection if strict mode is on.
        # Node location tracking is opt-in to avoid slowing down the common case.
        should_collect = collect_errors or strict

        self.tree_builder = tree_builder or TreeBuilder(
            fragment_context=fragment_context,
            iframe_srcdoc=iframe_srcdoc,
            collect_errors=should_collect,
            scripting_enabled=scripting_enabled,
            track_tag_spans=track_tag_spans,
        )
        opts = tokenizer_opts.copy() if tokenizer_opts is not None else TokenizerOpts()
        opts.scripting_enabled = bool(scripting_enabled)
        if needs_escape_incomplete_tags:
            opts.emit_bogus_markup_as_text = True

        # For RAWTEXT fragment contexts, set initial tokenizer state and rawtext tag
        if fragment_context and not fragment_context.namespace:
            rawtext_elements = {"textarea", "title", "style"}
            tag_name = fragment_context.tag_name.lower()
            if tag_name in rawtext_elements:
                opts.initial_state = Tokenizer.RAWTEXT
                opts.initial_rawtext_tag = tag_name
            elif tag_name in ("plaintext", "script"):
                opts.initial_state = Tokenizer.PLAINTEXT

        self.tokenizer = Tokenizer(
            self.tree_builder,
            opts,
            collect_errors=should_collect,
            track_node_locations=bool(track_node_locations),
            track_tag_positions=bool(track_node_locations) or track_tag_spans,
        )
        # Link tokenizer to tree_builder for position info
        self.tree_builder.tokenizer = self.tokenizer

        self.tokenizer.run(html_str)
        self.root = self.tree_builder.finish()

        transform_errors: list[ParseError] = []

        # Apply transforms after parse.
        # Safety model: when sanitize=True, the in-memory tree is sanitized exactly once
        # during construction by ensuring a Sanitize transform runs.
        if transforms or sanitize_enabled:
            from .sanitize import DEFAULT_DOCUMENT_POLICY, DEFAULT_POLICY  # noqa: PLC0415
            from .transforms import Sanitize  # noqa: PLC0415

            final_transforms: list[TransformSpec] = list(transforms or [])

            # Normalize explicit Sanitize() transforms to use the same default policy
            # choice as the old safe-output sanitizer (document vs fragment).
            if final_transforms:
                default_mode_policy = DEFAULT_DOCUMENT_POLICY if self.root.name == "#document" else DEFAULT_POLICY
                for i, t in enumerate(final_transforms):
                    if isinstance(t, Sanitize) and t.policy is None:
                        final_transforms[i] = Sanitize(
                            policy=default_mode_policy, enabled=t.enabled, callback=t.callback, report=t.report
                        )

            # Auto-append a final Sanitize step only if the user didn't include
            # Sanitize anywhere in their transform list.
            if sanitize_enabled and not any(isinstance(t, Sanitize) for t in final_transforms):
                effective_policy = (
                    policy
                    if policy is not None
                    else (DEFAULT_DOCUMENT_POLICY if self.root.name == "#document" else DEFAULT_POLICY)
                )
                # Avoid stale collected errors on reused policy objects.
                if effective_policy.unsafe_handling == "collect":
                    effective_policy.reset_collected_security_errors()
                final_transforms.append(Sanitize(policy=effective_policy))

            if final_transforms:
                compiled_transforms = compile_transforms(tuple(final_transforms))
                apply_compiled_transforms(self.root, compiled_transforms, errors=transform_errors)

                # Merge collected security errors into the document error list.
                # This mirrors the old behavior where safe output could feed
                # security findings into doc.errors.
                for t in final_transforms:
                    if isinstance(t, Sanitize):
                        t_policy = t.policy
                        if t_policy is not None and t_policy.unsafe_handling == "collect":
                            if t_policy.collects_security_errors_into(transform_errors):
                                continue
                            transform_errors.extend(t_policy.collected_security_errors())

        if should_collect:
            # Merge errors from both tokenizer and tree builder.
            # Public API: users expect errors to be ordered by input position.
            merged_errors = self.tokenizer.errors + self.tree_builder.errors + transform_errors
            self.errors = self._sorted_errors(merged_errors)
        else:
            self.errors = transform_errors

        # In strict mode, raise on first error
        if strict and self.errors:
            raise StrictModeError(self.errors[0])

    @staticmethod
    def escape_js_string(value: str, *, quote: str = '"') -> str:
        """Escape a value for safe inclusion in a JavaScript string literal."""
        from .serialize import _escape_js_string  # noqa: PLC0415

        return _escape_js_string(value, quote=quote)

    @staticmethod
    def escape_attr_value(value: str, *, quote: str = '"') -> str:
        """Escape a value for safe inclusion in a quoted HTML attribute value."""
        if quote not in {'"', "'"}:
            raise ValueError("quote must be ' or \"")

        from .serialize import _escape_attr_value  # noqa: PLC0415

        return _escape_attr_value(value, quote)

    @staticmethod
    def escape_url_value(value: str) -> str:
        """Percent-encode a URL value (useful before embedding into non-URL contexts)."""
        from .serialize import _escape_url_value  # noqa: PLC0415

        return _escape_url_value(value)

    @staticmethod
    def clean_url_value(*, value: str, url_rule: UrlRule) -> str | None:
        """Validate and rewrite a URL value according to an explicit UrlRule.

        This is URL *cleaning* (allowlisting, scheme/host checks, optional proxying),
        not URL escaping. It returns `None` if the URL is disallowed.
        """
        if not isinstance(url_rule, UrlRule):
            raise TypeError("url_rule must be a UrlRule")

        # Keep consistent validation with UrlPolicy.__post_init__ for proxy rules.
        if url_rule.handling == "proxy" and url_rule.proxy is None:
            raise ValueError("UrlRule.handling='proxy' requires a per-rule UrlRule.proxy")

        cleaned = _sanitize_url_value_with_rule(
            rule=url_rule,
            value=value,
            tag="*",
            attr="*",
            handling=url_rule.handling if url_rule.handling is not None else "allow",
            allow_relative=url_rule.allow_relative if url_rule.allow_relative is not None else True,
            proxy=url_rule.proxy,
            url_filter=None,
            apply_filter=False,
        )
        if cleaned is None:
            return None
        return JustHTML.escape_url_value(cleaned)

    @staticmethod
    def escape_url_in_js_string(value: str, *, quote: str = '"') -> str:
        """Escape a URL value for inclusion in a JavaScript string literal."""
        return JustHTML.escape_js_string(JustHTML.escape_url_value(value), quote=quote)

    @staticmethod
    def clean_url_in_js_string(
        *,
        value: str,
        url_rule: UrlRule,
        quote: str = '"',
    ) -> str | None:
        """Clean a URL using an explicit UrlRule, then make it JS-string-safe.

        Returns `None` if the URL is disallowed by the policy.
        """
        cleaned = JustHTML.clean_url_value(value=value, url_rule=url_rule)
        if cleaned is None:
            return None
        # cleaned is already percent-encoded by clean_url_value
        return JustHTML.escape_js_string(cleaned, quote=quote)

    @staticmethod
    def escape_html_text_in_js_string(value: str, *, quote: str = '"') -> str:
        """Escape plain text for assigning to innerHTML from a JavaScript string.

        This produces a JS-string-safe value that, when assigned to innerHTML,
        will be interpreted as text (not markup).
        """
        from .serialize import _escape_html_text  # noqa: PLC0415

        return JustHTML.escape_js_string(_escape_html_text(value), quote=quote)

    def query(self, selector: str) -> list[Any]:
        """Query the document using a CSS selector. Delegates to root.query()."""
        return self.root.query(selector)

    def query_one(self, selector: str) -> Any | None:
        """Return the first matching descendant for a CSS selector, or None."""
        return self.root.query_one(selector)

    @staticmethod
    def _sorted_errors(errors: list[ParseError]) -> list[ParseError]:
        indexed_errors = enumerate(errors)
        return [
            e
            for _, e in sorted(
                indexed_errors,
                key=lambda t: (
                    t[1].line if t[1].line is not None else 1_000_000_000,
                    t[1].column if t[1].column is not None else 1_000_000_000,
                    t[0],
                ),
            )
        ]

    def to_html(
        self,
        pretty: bool = True,
        indent_size: int = 2,
        *,
        context: HTMLContext | None = None,
        quote: str = '"',
    ) -> str:
        """Serialize the document to HTML.

        Sanitization (when enabled) happens during construction.
        """
        return self.root.to_html(
            indent=0,
            indent_size=indent_size,
            pretty=pretty,
            context=context,
            quote=quote,
        )

    def to_text(
        self,
        separator: str = " ",
        strip: bool = True,
        *,
        separator_blocks_only: bool = False,
    ) -> str:
        """Return the document's concatenated text."""
        return self.root.to_text(
            separator=separator,
            strip=strip,
            separator_blocks_only=separator_blocks_only,
        )

    def to_markdown(self, html_passthrough: bool = False) -> str:
        """Return a GitHub Flavored Markdown representation."""
        return self.root.to_markdown(html_passthrough=html_passthrough)
