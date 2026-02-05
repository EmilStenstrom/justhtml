#!/usr/bin/env python3
"""Command-line interface for JustHTML."""

from __future__ import annotations

import argparse
import io
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, TextIO, cast

from . import JustHTML
from .context import FragmentContext
from .selector import SelectorError

if TYPE_CHECKING:
    from .transforms import TransformSpec


def _get_version() -> str:
    try:
        return version("justhtml")
    except PackageNotFoundError:  # pragma: no cover
        return "dev"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="justhtml",
        description="Parse HTML5 and output text, pretty-printed HTML, or Markdown.",
        epilog=(
            "Examples:\n"
            "  justhtml page.html\n"
            "  curl -s https://example.com | justhtml -\n"
            "  justhtml page.html --selector 'main p' --format text\n"
            "  justhtml page.html --selector 'a' --format html\n"
            "  justhtml page.html --selector 'article' --allow-tags article --format markdown\n"
            "  curl -s https://example.com | justhtml - --format html --cleanup\n"
            "\n"
            "If you don't have the 'justhtml' command available, use:\n"
            "  python -m justhtml ...\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "path",
        nargs="?",
        help="HTML file to parse, or '-' to read from stdin",
    )
    parser.add_argument("--output", help="File to write output to")
    parser.add_argument(
        "--selector",
        help="CSS selector for choosing nodes (defaults to the document root)",
    )
    parser.add_argument(
        "--format",
        choices=["html", "text", "markdown"],
        default="html",
        help="Output format (default: html)",
    )

    parser.add_argument(
        "--unsafe",
        action="store_true",
        help="Disable sanitization (trusted input only)",
    )

    parser.add_argument(
        "--allow-tags",
        help=(
            "Safe mode: allow these additional tags during sanitization (comma-separated). "
            "Example: --allow-tags article,section"
        ),
    )

    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove common useless output (unwrap <a> without href, drop <img> without src, prune empty tags)",
    )
    parser.add_argument(
        "--first",
        action="store_true",
        help="Only output the first matching node",
    )

    parser.add_argument(
        "--fragment",
        action="store_true",
        help="Parse input as an HTML fragment (context: <div>)",
    )

    parser.add_argument(
        "--separator",
        default=" ",
        help="Text-only: join string between text nodes (default: a single space)",
    )
    parser.add_argument(
        "--separator-blocks-only",
        action="store_true",
        help="Text-only: only apply --separator between block-level elements (avoid separators inside inline tags)",
    )
    strip_group = parser.add_mutually_exclusive_group()
    strip_group.add_argument(
        "--strip",
        action="store_true",
        default=True,
        help="Text-only: strip each text node and drop empty segments (default)",
    )
    strip_group.add_argument(
        "--no-strip",
        action="store_false",
        dest="strip",
        help="Text-only: preserve text node whitespace",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"justhtml {_get_version()}",
    )

    args = parser.parse_args(argv)

    if not args.path:
        parser.print_help(sys.stderr)
        raise SystemExit(1)

    return args


def _read_html(path: str) -> str | bytes:
    if path == "-":
        stdin = sys.stdin
        if isinstance(stdin, io.TextIOWrapper):
            data: bytes = stdin.buffer.read()
            return data
        return cast("str", stdin.read())

    return Path(path).read_bytes()


def main() -> None:
    args = _parse_args(sys.argv[1:])
    html = _read_html(args.path)
    fragment_context = FragmentContext("div") if args.fragment else None
    safe = not args.unsafe

    policy = None
    if safe and args.allow_tags:
        from .sanitize import DEFAULT_DOCUMENT_POLICY, DEFAULT_POLICY, SanitizationPolicy  # noqa: PLC0415

        extra_tags: set[str] = set()
        for part in str(args.allow_tags).replace(" ", ",").split(","):
            tag = part.strip().lower()
            if tag:
                extra_tags.add(tag)

        base = DEFAULT_POLICY if fragment_context is not None else DEFAULT_DOCUMENT_POLICY
        allowed = set(base.allowed_tags)
        allowed.update(extra_tags)
        policy = SanitizationPolicy(
            allowed_tags=allowed,
            allowed_attributes=base.allowed_attributes,
            url_policy=base.url_policy,
            drop_comments=base.drop_comments,
            drop_doctype=base.drop_doctype,
            drop_foreign_namespaces=base.drop_foreign_namespaces,
            drop_content_tags=base.drop_content_tags,
            allowed_css_properties=base.allowed_css_properties,
            force_link_rel=base.force_link_rel,
            unsafe_handling=base.unsafe_handling,
            disallowed_tag_handling=base.disallowed_tag_handling,
        )

    transforms: list[TransformSpec] | None = None
    if args.cleanup:
        from .sanitize import DEFAULT_DOCUMENT_POLICY, DEFAULT_POLICY  # noqa: PLC0415
        from .transforms import Drop, PruneEmpty, Sanitize, Unwrap  # noqa: PLC0415

        default_policy = DEFAULT_POLICY if fragment_context is not None else DEFAULT_DOCUMENT_POLICY
        effective_policy = policy if policy is not None else default_policy

        transforms_list: list[TransformSpec] = []
        if safe:
            # Ensure cleanup happens after sanitization so it can react to
            # stripped attributes (e.g. <a> whose unsafe href was removed).
            transforms_list.append(Sanitize(policy=effective_policy))

        transforms_list.append(Unwrap("a:not([href])"))
        transforms_list.append(Drop('img:not([src]), img[src=""]'))
        transforms_list.append(PruneEmpty("*"))
        transforms = transforms_list

    doc = JustHTML(
        html,
        fragment_context=fragment_context,
        sanitize=safe,
        policy=policy,
        transforms=transforms,
    )

    try:
        nodes = doc.query(args.selector) if args.selector else [doc.root]
    except SelectorError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(2) from e

    if not nodes:
        raise SystemExit(1)

    if args.first:
        nodes = [nodes[0]]

    def write_output(out: TextIO) -> None:
        if args.format == "html":
            outputs = [node.to_html() for node in nodes]
            out.write("\n".join(outputs))
            out.write("\n")
            return

        if args.format == "text":
            # Keep these branches explicit so coverage will highlight untested CLI options.
            if args.separator == " ":
                if args.strip:
                    outputs = [
                        node.to_text(strip=True, separator_blocks_only=args.separator_blocks_only) for node in nodes
                    ]
                else:
                    outputs = [
                        node.to_text(strip=False, separator_blocks_only=args.separator_blocks_only) for node in nodes
                    ]
            else:
                if args.strip:
                    outputs = [
                        node.to_text(
                            separator=args.separator,
                            strip=True,
                            separator_blocks_only=args.separator_blocks_only,
                        )
                        for node in nodes
                    ]
                else:
                    outputs = [
                        node.to_text(
                            separator=args.separator,
                            strip=False,
                            separator_blocks_only=args.separator_blocks_only,
                        )
                        for node in nodes
                    ]
            out.write("\n".join(outputs))
            out.write("\n")
            return

        outputs = [node.to_markdown() for node in nodes]
        out.write("\n\n".join(outputs))
        out.write("\n")

    if args.output:
        with Path(args.output).open(mode="w", encoding="utf-8") as outfile:
            write_output(outfile)
        return

    write_output(sys.stdout)


if __name__ == "__main__":
    main()
