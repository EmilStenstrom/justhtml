#!/usr/bin/env python
"""Interactive release helper for JustHTML.

What it does (in order):
1) Verifies the repo-wide pre-commit checks CI runs for releases
2) Bumps version in pyproject.toml ([project].version)
3) Commits the change
4) Creates an annotated git tag
5) Pushes commit + tag
6) Creates a GitHub release (marked as latest) via `gh`
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PYPROJECT_PATH = Path(__file__).resolve().parents[1] / "pyproject.toml"
CHANGELOG_PATH = PYPROJECT_PATH.parent / "CHANGELOG.md"


@dataclass(frozen=True)
class CmdResult:
    stdout: str
    returncode: int


def _fail(msg: str) -> None:
    raise RuntimeError(msg)


def _run(cmd: list[str], *, check: bool = True) -> CmdResult:
    p = subprocess.run(  # noqa: S603
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    out = CmdResult(stdout=p.stdout, returncode=p.returncode)
    if check and p.returncode != 0:
        raise RuntimeError(
            "Command failed (exit {code}): {cmd}\n{out}".format(
                code=p.returncode,
                cmd=_quote_cmd(cmd),
                out=(p.stdout or "").rstrip(),
            )
        )
    return out


def _run_with_env(cmd: list[str], *, env: dict[str, str], check: bool = True) -> CmdResult:
    merged_env = os.environ.copy()
    merged_env.update(env)
    p = subprocess.run(  # noqa: S603
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=merged_env,
    )
    out = CmdResult(stdout=p.stdout, returncode=p.returncode)
    if check and p.returncode != 0:
        raise RuntimeError(
            "Command failed (exit {code}): {cmd}\n{out}".format(
                code=p.returncode,
                cmd=_quote_cmd(cmd),
                out=(p.stdout or "").rstrip(),
            )
        )
    return out


def _run_quiet_ok(cmd: list[str]) -> bool:
    p = subprocess.run(  # noqa: S603
        cmd,
        check=False,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return p.returncode == 0


def _quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def _prompt(msg: str, *, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        v = input(f"{msg}{suffix}: ").strip()
        if v:
            return v
        if default is not None:
            return default


def _confirm(msg: str, *, default_no: bool = True) -> bool:
    prompt = "[y/N]" if default_no else "[Y/n]"
    v = input(f"{msg} {prompt}: ").strip().lower()
    if not v:
        return not default_no
    return v in {"y", "yes"}


_VERSION_RE = re.compile(r'(?m)^(\s*version\s*=\s*")([^"]+)("\s*)$')


def _read_current_version(pyproject_text: str) -> str:
    # Find version only inside [project] table.
    in_project = False
    for line in pyproject_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project = stripped == "[project]"
            continue
        if not in_project:
            continue
        m = _VERSION_RE.match(line)
        if m:
            return m.group(2)
    raise RuntimeError("Could not find [project].version in pyproject.toml")


def _bump_version_file(path: Path, new_version: str) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    old_version = _read_current_version(text)

    in_project = False
    out_lines: list[str] = []
    replaced = False
    for line in text.splitlines(keepends=False):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project = stripped == "[project]"
            out_lines.append(line)
            continue

        if in_project:
            m = _VERSION_RE.match(line)
            if m and not replaced:
                out_lines.append(f"{m.group(1)}{new_version}{m.group(3)}")
                replaced = True
                continue

        out_lines.append(line)

    if not replaced:
        raise RuntimeError("Did not update version (no match inside [project] table)")

    new_text = "\n".join(out_lines) + ("\n" if text.endswith("\n") else "")
    if new_text == text:
        raise RuntimeError("Version update produced no file changes")

    path.write_text(new_text, encoding="utf-8")
    return old_version, new_version


def _require_clean_git() -> None:
    if not _run_quiet_ok(["git", "diff", "--quiet"]):
        raise RuntimeError("Working tree has unstaged changes")
    if not _run_quiet_ok(["git", "diff", "--cached", "--quiet"]):
        raise RuntimeError("Index has staged changes")


def _git_current_branch() -> str:
    return _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()


def _git_last_tag() -> str | None:
    p = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0"],  # noqa: S607
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if p.returncode != 0:
        return None
    return (p.stdout or "").strip() or None


def _notes_from_git(last_tag: str | None) -> str:
    if last_tag:
        rng = f"{last_tag}..HEAD"
        out = _run(["git", "log", "--oneline", rng]).stdout.strip()
    else:
        out = _run(["git", "log", "--oneline"]).stdout.strip()

    if not out:
        return ""

    # Keep it short: max 20 lines.
    lines = out.splitlines()[:20]
    return "\n".join(f"- {line}" for line in lines)


def _tag_exists(tag: str) -> bool:
    return _run_quiet_ok(["git", "rev-parse", "--verify", "--quiet", tag])


def _check_changelog(version: str) -> None:
    if not CHANGELOG_PATH.exists():
        print(f"WARNING: Changelog file not found at {CHANGELOG_PATH}")
        return

    content = CHANGELOG_PATH.read_text(encoding="utf-8")
    # Look for "## [version]" or "## version"
    # Escape version just in case
    escaped_version = re.escape(version)
    pattern = re.compile(rf"^##\s+\[?{escaped_version}\]?", re.MULTILINE)

    if not pattern.search(content):
        _fail(f"Changelog entry for version {version} not found in {CHANGELOG_PATH}")


def _extract_changelog_notes(version: str) -> str | None:
    if not CHANGELOG_PATH.exists():
        return None

    content = CHANGELOG_PATH.read_text(encoding="utf-8")
    escaped_version = re.escape(version)
    # Match start of section: ## [version] or ## version
    start_pattern = re.compile(rf"^##\s+\[?{escaped_version}\]?(?:.*)$", re.MULTILINE)

    m_start = start_pattern.search(content)
    if not m_start:
        return None

    start_idx = m_start.end()

    # Find next section start (## ...) or end of file
    next_section_pattern = re.compile(r"^##\s+\[?.*\]?", re.MULTILINE)
    m_end = next_section_pattern.search(content, start_idx)

    if m_end:
        section_text = content[start_idx : m_end.start()]
    else:
        section_text = content[start_idx:]

    return section_text.strip()


_GITHUB_RE_RE = re.compile(
    r"""
    \A(?:
        https://github\.com/ |
        git@github\.com: |
        ssh://git@github\.com/
    )
    (?P<owner>[^/]+)/(?P<repo>[^/]+?)
    (?:\.git)?\Z
    """,
    re.VERBOSE,
)


def _default_repo_from_remote(remote: str) -> str:
    url = _run(["git", "remote", "get-url", remote]).stdout.strip()
    m = _GITHUB_RE_RE.match(url)
    if not m:
        raise RuntimeError(f"Could not parse GitHub repo from {remote} URL: {url}")
    return f"{m.group('owner')}/{m.group('repo')}"


def _run_release_checks() -> None:
    print("Running release checks: SKIP=mypy pre-commit run --all-files")
    out = _run_with_env(["pre-commit", "run", "--all-files"], env={"SKIP": "mypy"}).stdout
    if out.strip():
        print(out.rstrip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bump version, tag, and create a GitHub release.")
    parser.add_argument("--version", help="New version, e.g. 0.21.0 (will be tagged as v0.21.0 unless --tag is set)")
    parser.add_argument("--tag", help="Tag name, defaults to v{version}")
    parser.add_argument("--title", help="Release title, defaults to 'Release {tag}'")
    parser.add_argument(
        "--notes",
        help="Release notes (short description). If omitted, you will be prompted.\n"
        "Use --notes-from-commits to append a short commit list.",
    )
    parser.add_argument(
        "--notes-from-commits",
        action="store_true",
        help="Append recent commit subjects since last tag.",
    )
    parser.add_argument(
        "--remote",
        default="origin",
        help="Git remote to push to (default: origin)",
    )
    parser.add_argument(
        "--repo",
        help=(
            "GitHub repository in OWNER/REPO form. If omitted, inferred from the 'origin' remote. "
            "Useful when multiple remotes exist and `gh` can't pick a default."
        ),
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Do not push commit/tag (still creates local commit/tag).",
    )
    parser.add_argument(
        "--no-release",
        action="store_true",
        help="Do not create GitHub release (still bumps/commits/tags).",
    )
    parser.add_argument(
        "--target",
        help="Target branch/commitish for the release (default: current branch).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Do not prompt for confirmation before push/release.",
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip repo-wide pre-commit validation. Use only when you have already validated the exact release commit.",
    )

    args = parser.parse_args(argv)

    try:
        _require_clean_git()

        if not args.skip_checks:
            _run_release_checks()

        py_text = PYPROJECT_PATH.read_text(encoding="utf-8")
        current_version = _read_current_version(py_text)

        new_version = args.version or _prompt("New version", default=current_version)

        _check_changelog(new_version)

        resume_only = False
        if new_version == current_version:
            resume_only = True

        tag = args.tag or f"v{new_version}"
        title = args.title or f"Release {tag}"

        branch = _git_current_branch()

        if resume_only:
            if not _tag_exists(tag):
                _fail(
                    f"Version is already {current_version} but tag does not exist: {tag}. "
                    "Provide a new version to bump and create a tag, or pass --tag to match an existing tag."
                )
            print(f"Using existing version/tag: {new_version} ({tag})")
        else:
            if _tag_exists(tag):
                _fail(f"Tag already exists: {tag}")

            old_v, _ = _bump_version_file(PYPROJECT_PATH, new_version)
            print(f"Updated pyproject.toml version: {old_v} -> {new_version}")

            commit_msg = f"Release {tag}"
            print(f"Committing: {commit_msg}")
            _run(["git", "add", str(PYPROJECT_PATH)])
            _run(["git", "commit", "-m", commit_msg])

            print(f"Creating annotated tag: {tag}")
            _run(["git", "tag", "-a", tag, "-m", tag])
        print(f"On branch: {branch}")

        if not args.no_push:
            if args.yes or _confirm(f"Push commit and tag to {args.remote}?"):
                _run(["git", "push", args.remote, branch])
                _run(["git", "push", args.remote, tag])
            else:
                print("Skipping push.")

        if not args.no_release:
            repo = args.repo or _default_repo_from_remote(args.remote)
            notes = args.notes

            if notes is None:
                # Try to extract from changelog
                changelog_notes = _extract_changelog_notes(new_version)
                if changelog_notes:
                    print(f"Extracted release notes from CHANGELOG.md for {new_version}")
                    notes = changelog_notes
                else:
                    notes = _prompt("Release notes (one line)")

            target = args.target or branch

            # If the release already exists, we don't need to create it again.
            existing = _run(["gh", "release", "view", tag, "--repo", repo], check=False)
            if existing.returncode == 0:
                print(f"Release already exists on GitHub: {tag}")
                return 0

            if args.notes_from_commits:
                last_tag = _git_last_tag()
                extra = _notes_from_git(last_tag)
                if extra:
                    notes = f"{notes}\n\nChanges:\n{extra}"

            cmd = [
                "gh",
                "release",
                "create",
                tag,
                "--repo",
                repo,
                "--title",
                title,
                "--notes",
                notes,
                "--latest",
                "--target",
                target,
            ]

            if args.yes or _confirm(f"Create GitHub release {tag} (latest)?"):
                print(_quote_cmd(cmd))
                out = _run(cmd).stdout
                if out.strip():
                    print(out.rstrip())
            else:
                print("Skipping GitHub release creation.")

    except KeyboardInterrupt:
        print("\nAborted.")
        return 130
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    else:
        print("Done.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
