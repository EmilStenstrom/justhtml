#!/usr/bin/env python3
"""Profile JustHTML on real-world HTML."""

import argparse
import importlib
import pathlib
import pstats
import sys
import tarfile

import zstandard as zstd

from justhtml import JustHTML


def load_dict(dict_path: pathlib.Path) -> bytes:
    """Load the zstd dictionary."""
    return dict_path.read_bytes()


def load_html_files(batch_path: pathlib.Path, dict_bytes: bytes, limit: int = 100):
    """Load HTML files from batch."""
    results = []
    tar_dctx = zstd.ZstdDecompressor()
    with batch_path.open("rb") as batch_file:
        with tar_dctx.stream_reader(batch_file) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                html_dctx = zstd.ZstdDecompressor(
                    dict_data=zstd.ZstdCompressionDict(dict_bytes),
                )

                count = 0
                for member in tar:
                    if not member.isfile() or not member.name.endswith(".html.zst"):
                        continue

                    if count >= limit:
                        break

                    compressed_html = tar.extractfile(member).read()
                    html_content = html_dctx.decompress(compressed_html).decode(
                        "utf-8",
                        errors="replace",
                    )
                    results.append((member.name, html_content))
                    count += 1

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--to-html",
        action="store_true",
        help="Profile parse + serialization via to_html() (pretty=True by default)",
    )
    return parser.parse_args()


def remove_script_dir_from_import_path() -> None:
    script_dir = str(pathlib.Path(__file__).resolve().parent)
    if sys.path and sys.path[0] == script_dir:
        sys.path.pop(0)


def main() -> None:
    remove_script_dir_from_import_path()
    cprofile = importlib.import_module("cProfile")

    args = parse_args()

    dict_bytes = load_dict(pathlib.Path("/home/emilstenstrom/Projects/web100k/html.dict"))
    html_files = load_html_files(
        pathlib.Path("/home/emilstenstrom/Projects/web100k/batches/web100k-batch-001.tar.zst"),
        dict_bytes,
        limit=100,
    )

    print(f"Loaded {len(html_files)} files")

    profiler = cprofile.Profile()
    profiler.enable()

    for _filename, html in html_files:
        parser = JustHTML(html)
        _ = parser.to_html() if args.to_html else parser.root

    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats("tottime")
    stats.print_stats(80)


if __name__ == "__main__":
    main()
