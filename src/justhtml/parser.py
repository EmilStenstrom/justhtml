"""Minimal JustHTML parser entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .context import FragmentContext
from .encoding import decode_html
from .tokenizer import Tokenizer, TokenizerOpts
from .transforms import apply_compiled_transforms, compile_transforms
from .treebuilder import TreeBuilder

if TYPE_CHECKING:
    from .node import SimpleDomNode
    from .sanitize import SanitizationPolicy
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
    root: SimpleDomNode
    tokenizer: Tokenizer
    tree_builder: TreeBuilder

    def __init__(
        self,
        html: str | bytes | bytearray | memoryview | None,
        *,
        safe: bool = True,
        policy: SanitizationPolicy | None = None,
        collect_errors: bool = False,
        track_node_locations: bool = False,
        debug: bool = False,
        encoding: str | None = None,
        fragment: bool = False,
        fragment_context: FragmentContext | None = None,
        iframe_srcdoc: bool = False,
        strict: bool = False,
        tokenizer_opts: TokenizerOpts | None = None,
        tree_builder: TreeBuilder | None = None,
        transforms: list[TransformSpec] | None = None,
    ) -> None:
        if fragment_context is not None:
            fragment = True

        if fragment and fragment_context is None:
            fragment_context = FragmentContext("div")

        track_tag_spans = False
        has_sanitize_transform = False
        if transforms:
            from .sanitize import DEFAULT_POLICY  # noqa: PLC0415
            from .transforms import Sanitize  # noqa: PLC0415

            for t in transforms:
                if isinstance(t, Sanitize):
                    has_sanitize_transform = True
                    effective = t.policy or DEFAULT_POLICY
                    if effective.disallowed_tag_handling == "escape":
                        track_tag_spans = True
                        break

        # If we will auto-sanitize (safe=True and no Sanitize in transforms),
        # escape-mode tag reconstruction may require tracking tag spans.
        if safe and not has_sanitize_transform and policy is not None:
            if policy.disallowed_tag_handling == "escape":
                track_tag_spans = True

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
            track_tag_spans=track_tag_spans,
        )
        opts = tokenizer_opts or TokenizerOpts()

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
        # Safety model: when safe=True, the in-memory tree is sanitized exactly once
        # during construction by ensuring a Sanitize transform runs.
        if transforms or safe:
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
            if safe and not any(isinstance(t, Sanitize) for t in final_transforms):
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

    def query(self, selector: str) -> list[Any]:
        """Query the document using a CSS selector. Delegates to root.query()."""
        return self.root.query(selector)

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
    ) -> str:
        """Serialize the document to HTML.

        Sanitization (when enabled) happens during construction.
        """
        return self.root.to_html(
            indent=0,
            indent_size=indent_size,
            pretty=pretty,
        )

    def to_text(
        self,
        separator: str = " ",
        strip: bool = True,
    ) -> str:
        """Return the document's concatenated text."""
        return self.root.to_text(separator=separator, strip=strip)

    def to_markdown(self) -> str:
        """Return a GitHub Flavored Markdown representation."""
        return self.root.to_markdown()
