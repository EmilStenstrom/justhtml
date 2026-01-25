from __future__ import annotations

import ast
import codeop
import difflib
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class _DocExample:
    doc_path: Path
    code_start_line: int
    code: str
    output_lang: str
    expected_output: str
    expects_raise: bool


@dataclass(frozen=True, slots=True)
class _ReadmeExample:
    readme_path: Path
    code_start_line: int
    original_code: str
    runnable_code: str
    expected_output: str


def _strip_expected_prefix(line: str) -> str | None:
    stripped = line.lstrip()
    if not stripped.startswith("#"):
        return None
    stripped = stripped[1:].lstrip()
    if not stripped.startswith("=>"):
        return None
    return stripped[2:].lstrip()


_INLINE_EXPECT_RE = re.compile(r"^(?P<code>.*?)(?P<ws>\s*)#\s*=>\s*(?P<expected>.*)$")


def _is_print_expr(node: ast.AST) -> bool:
    if not isinstance(node, ast.Expr):
        return False
    value = node.value
    if not isinstance(value, ast.Call):
        return False
    func = value.func
    return isinstance(func, ast.Name) and func.id == "print"


def _rewrite_readme_block_to_runnable(code: str) -> tuple[str, str]:
    lines = code.splitlines()

    runnable_lines: list[str] = [
        "from __future__ import annotations",
        "",
        "from justhtml import JustHTML",
        "",
    ]
    expected_lines: list[str] = []

    compiler = codeop.CommandCompiler()
    i = 0
    while i < len(lines):
        line = lines[i]

        expected = _strip_expected_prefix(line)
        if expected is not None:
            raise AssertionError("Found '# => ...' without a preceding statement")

        # Preserve blank lines as-is.
        if line.strip() == "":
            runnable_lines.append("")
            i += 1
            continue

        # Accumulate a complete Python statement.
        stmt_lines: list[str] = []
        while i < len(lines):
            stmt_lines.append(lines[i])
            src = "\n".join(stmt_lines)
            try:
                code = compiler(src, symbol="exec")
            except (SyntaxError, OverflowError, ValueError):
                # Keep as-is; execution will surface the error.
                code = object()
            i += 1
            if code is not None:
                break

        stmt_src = "\n".join(stmt_lines).rstrip("\n")

        expected_for_stmt: list[str] = []
        if stmt_lines:
            m = _INLINE_EXPECT_RE.match(stmt_lines[-1])
            if m is not None:
                expected_inline = m.group("expected")
                stmt_lines[-1] = m.group("code").rstrip()
                stmt_src = "\n".join(stmt_lines).rstrip("\n")
                expected_for_stmt.append(expected_inline)

        # Collect consecutive expectation lines immediately following the statement.
        while i < len(lines):
            exp = _strip_expected_prefix(lines[i])
            if exp is None:
                break
            expected_for_stmt.append(exp)
            i += 1

        if expected_for_stmt:
            try:
                parsed = ast.parse(stmt_src, mode="exec")
            except SyntaxError:
                raise AssertionError("Unable to parse statement preceding '# =>'") from None

            if len(parsed.body) != 1:
                raise AssertionError("Expected a single statement before '# =>'")

            only = parsed.body[0]
            if _is_print_expr(only):
                runnable_lines.append(stmt_src)
            elif isinstance(only, ast.Expr):
                expr_src = stmt_src.strip()
                runnable_lines.append(f"print(\n{expr_src}\n)")
            else:
                raise AssertionError("'# =>' must follow an expression or print(...) statement")

            expected_lines.extend(expected_for_stmt)
        else:
            runnable_lines.append(stmt_src)

    runnable_code = "\n".join(runnable_lines).rstrip("\n")
    expected_output = "\n".join(expected_lines).rstrip("\n")
    return runnable_code, expected_output


def _iter_readme_examples(project_root: Path) -> list[_ReadmeExample]:
    return _iter_markdown_hash_arrow_examples(project_root, project_root / "README.md")


def _iter_markdown_hash_arrow_examples(project_root: Path, md_path: Path) -> list[_ReadmeExample]:
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    examples: list[_ReadmeExample] = []

    last_html: str | None = None

    i = 0
    while i < len(lines):
        parsed = _parse_fence_line(lines[i])
        if parsed is None:
            i += 1
            continue

        code_fence, fence_lang = parsed
        if fence_lang == "html":
            i += 1
            html_lines: list[str] = []
            while i < len(lines) and not lines[i].lstrip().startswith(code_fence):
                html_lines.append(lines[i])
                i += 1
            if i < len(lines) and lines[i].lstrip().startswith(code_fence):
                i += 1
            last_html = "\n".join(html_lines).rstrip("\n")
            last_html = textwrap.dedent(last_html).rstrip("\n")
            continue

        if fence_lang not in {"python", "py"}:
            i += 1
            continue

        code_start_line = i + 1  # 1-based line number of opening fence
        i += 1
        code_lines: list[str] = []
        while i < len(lines) and not lines[i].lstrip().startswith(code_fence):
            code_lines.append(lines[i])
            i += 1
        if i < len(lines) and lines[i].lstrip().startswith(code_fence):
            i += 1

        original_code = "\n".join(code_lines).rstrip("\n")
        original_code = textwrap.dedent(original_code).rstrip("\n")
        if "# =>" not in original_code:
            continue

        try:
            runnable_code, expected_output = _rewrite_readme_block_to_runnable(original_code)
        except AssertionError as e:
            raise AssertionError(
                f"Markdown doctest parse error at {md_path.relative_to(project_root)}:{code_start_line}: {e}"
            ) from None

        if expected_output.strip() == "":
            continue

        # Special-case docs pages: code blocks often assume `html` comes from the last preceding
        # HTML code fence. If we have one, inject it as a Python string literal.
        if last_html is not None:
            uses_html = re.search(r"\bhtml\b", original_code) is not None
            defines_html = re.search(r"^\s*html\s*=", original_code, flags=re.MULTILINE) is not None
            if uses_html and not defines_html:
                inject = f"html = {last_html!r}\n\n"
                marker = "from justhtml import JustHTML\n\n"
                if marker in runnable_code:
                    runnable_code = runnable_code.replace(marker, marker + inject, 1)
                else:
                    runnable_code = inject + runnable_code

        examples.append(
            _ReadmeExample(
                readme_path=md_path,
                code_start_line=code_start_line,
                original_code=original_code,
                runnable_code=runnable_code,
                expected_output=expected_output,
            )
        )

    return examples


def _parse_fence_line(line: str) -> tuple[str, str] | None:
    stripped = line.lstrip()
    if not stripped.startswith("`"):
        return None
    i = 0
    while i < len(stripped) and stripped[i] == "`":
        i += 1
    if i < 3:
        return None
    fence = stripped[:i]
    lang = stripped[i:].strip().lower()
    return fence, lang


def _iter_doc_examples(docs_dir: Path) -> list[_DocExample]:
    examples: list[_DocExample] = []

    for doc_path in sorted(docs_dir.rglob("*.md")):
        text = doc_path.read_text(encoding="utf-8")
        lines = text.splitlines()

        i = 0
        while i < len(lines):
            line = lines[i]
            parsed = _parse_fence_line(line)
            if parsed is not None:
                code_fence, fence_lang = parsed
                if fence_lang in {"python", "py"}:
                    code_start_line = i + 1  # 1-based line number of fence
                    i += 1
                    code_lines: list[str] = []
                    while i < len(lines) and not lines[i].lstrip().startswith(code_fence):
                        code_lines.append(lines[i])
                        i += 1
                    # Skip closing fence if present.
                    if i < len(lines) and lines[i].lstrip().startswith(code_fence):
                        i += 1

                    code = "\n".join(code_lines).rstrip("\n")
                    code = textwrap.dedent(code).rstrip("\n")

                    expects_raise = "doctest: raises" in code

                    # Optional skip marker inside the code block.
                    if "doctest: skip" in code:
                        continue

                    j = i
                    while j < len(lines) and lines[j].strip() == "":
                        j += 1

                    if j < len(lines) and lines[j].strip() == "Output:":
                        k = j + 1
                        while k < len(lines) and lines[k].strip() == "":
                            k += 1
                        out_parsed = _parse_fence_line(lines[k]) if k < len(lines) else None
                        if out_parsed is not None:
                            out_fence, output_lang = out_parsed
                            k += 1
                            out_lines: list[str] = []
                            while k < len(lines) and not lines[k].lstrip().startswith(out_fence):
                                out_lines.append(lines[k])
                                k += 1
                            if k < len(lines) and lines[k].lstrip().startswith(out_fence):
                                k += 1

                            expected_output = "\n".join(out_lines).rstrip("\n")
                            examples.append(
                                _DocExample(
                                    doc_path=doc_path,
                                    code_start_line=code_start_line,
                                    code=code,
                                    output_lang=output_lang,
                                    expected_output=expected_output,
                                    expects_raise=expects_raise,
                                )
                            )
                            i = k
                            continue

                    continue

            i += 1

    return examples


def _run_python_snippet(project_root: Path, code: str) -> tuple[int, str, str]:
    env = os.environ.copy()
    src_dir = str(project_root / "src")
    env["PYTHONPATH"] = src_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTHONUTF8", "1")

    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "snippet.py"
        script_path.write_text(code + "\n", encoding="utf-8")

        try:
            proc = subprocess.run(  # noqa: S603
                [sys.executable, str(script_path)],
                check=False,
                cwd=str(project_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
        except subprocess.TimeoutExpired as e:
            stdout = (e.stdout or "").replace("\r\n", "\n").rstrip("\n")
            stderr = (e.stderr or "").replace("\r\n", "\n").rstrip("\n")
            msg = "Doc snippet timed out after 5s"
            if stderr:
                msg = msg + "\n" + stderr
            return 124, stdout, msg

    stdout = proc.stdout.replace("\r\n", "\n").rstrip("\n")
    stderr = proc.stderr.replace("\r\n", "\n").rstrip("\n")
    return proc.returncode, stdout, stderr


def _line_matches(expected_line: str, actual_line: str) -> bool:
    if expected_line == actual_line:
        return True

    # Allow single-line wildcards using "..." inside a line.
    if "..." not in expected_line:
        return False

    parts = expected_line.split("...")
    pos = 0
    for part in parts:
        if part == "":
            continue
        idx = actual_line.find(part, pos)
        if idx == -1:
            return False
        pos = idx + len(part)
    return True


def _matches_with_ellipsis(expected: str, actual: str) -> bool:
    expected_lines = expected.splitlines()
    actual_lines = actual.splitlines()

    i = 0
    j = 0
    while i < len(expected_lines):
        el = expected_lines[i]

        # A line that is only "..." (ignoring surrounding whitespace) matches
        # any number of actual lines until the next expected line matches.
        if el.strip() == "...":
            i += 1
            if i >= len(expected_lines):
                return True
            next_el = expected_lines[i]
            while j < len(actual_lines) and not _line_matches(next_el, actual_lines[j]):
                j += 1
            if j >= len(actual_lines):
                return False
            continue

        if j >= len(actual_lines):
            return False
        if not _line_matches(el, actual_lines[j]):
            return False
        i += 1
        j += 1

    return j == len(actual_lines)


def _diff(expected: str, actual: str) -> str:
    expected_lines = expected.splitlines(keepends=True)
    actual_lines = actual.splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            expected_lines,
            actual_lines,
            fromfile="expected",
            tofile="actual",
        )
    )


class TestDocsExamples(unittest.TestCase):
    def test_docs_snippets_match_output_blocks(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        docs_dir = project_root / "docs"

        examples = _iter_doc_examples(docs_dir)
        if not examples:
            self.fail("No doc examples with Output blocks found")

        failures: list[str] = []

        for ex in examples:
            returncode, stdout, stderr = _run_python_snippet(project_root, ex.code)
            expected = ex.expected_output.replace("\r\n", "\n").rstrip("\n")

            if ex.expects_raise:
                actual = stderr
                if returncode == 0:
                    failures.append(
                        "\n".join(
                            [
                                f"Doc: {ex.doc_path.relative_to(project_root)}:{ex.code_start_line}",
                                f"Output fence language: {ex.output_lang}",
                                "Expected snippet to raise but it exited 0",
                            ]
                        )
                    )
                    continue
            else:
                actual = stdout
                if returncode != 0:
                    if returncode == 124:
                        failures.append(
                            "\n".join(
                                [
                                    f"Doc: {ex.doc_path.relative_to(project_root)}:{ex.code_start_line}",
                                    f"Output fence language: {ex.output_lang}",
                                    "Snippet execution timed out:",
                                    stderr,
                                    "Snippet:",
                                    ex.code,
                                ]
                            )
                        )
                        continue
                    failures.append(
                        "\n".join(
                            [
                                f"Doc: {ex.doc_path.relative_to(project_root)}:{ex.code_start_line}",
                                f"Output fence language: {ex.output_lang}",
                                "Snippet execution failed:",
                                stderr,
                            ]
                        )
                    )
                    continue

            if not _matches_with_ellipsis(expected, actual):
                failures.append(
                    "\n".join(
                        [
                            f"Doc: {ex.doc_path.relative_to(project_root)}:{ex.code_start_line}",
                            f"Output fence language: {ex.output_lang}",
                            _diff(expected, actual),
                        ]
                    )
                )

        if failures:
            self.fail("\n\n".join(failures))

    def test_readme_python_blocks_match_hash_arrow_output(self) -> None:
        project_root = Path(__file__).resolve().parents[1]

        examples = _iter_readme_examples(project_root)
        if not examples:
            self.fail("No README doctest-style examples (# => ...) found")

        failures: list[str] = []
        for ex in examples:
            returncode, stdout, stderr = _run_python_snippet(project_root, ex.runnable_code)
            if returncode != 0:
                if returncode == 124:
                    failures.append(
                        "\n".join(
                            [
                                f"Doc: {ex.readme_path.relative_to(project_root)}:{ex.code_start_line}",
                                "Snippet execution timed out:",
                                stderr,
                                "Snippet:",
                                ex.original_code,
                            ]
                        )
                    )
                    continue
                failures.append(
                    "\n".join(
                        [
                            f"Doc: {ex.readme_path.relative_to(project_root)}:{ex.code_start_line}",
                            "Snippet execution failed:",
                            stderr,
                        ]
                    )
                )
                continue

            expected = ex.expected_output.replace("\r\n", "\n").rstrip("\n")
            actual = stdout
            if not _matches_with_ellipsis(expected, actual):
                failures.append(
                    "\n".join(
                        [
                            f"Doc: {ex.readme_path.relative_to(project_root)}:{ex.code_start_line}",
                            _diff(expected, actual),
                        ]
                    )
                )

        if failures:
            self.fail("\n\n".join(failures))

    def test_docs_python_blocks_match_hash_arrow_output(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        docs_dir = project_root / "docs"

        examples: list[_ReadmeExample] = []
        for md_path in sorted(docs_dir.rglob("*.md")):
            examples.extend(_iter_markdown_hash_arrow_examples(project_root, md_path))

        if not examples:
            self.fail("No docs doctest-style examples (# => ...) found")

        failures: list[str] = []
        for ex in examples:
            returncode, stdout, stderr = _run_python_snippet(project_root, ex.runnable_code)
            if returncode != 0:
                if returncode == 124:
                    failures.append(
                        "\n".join(
                            [
                                f"Doc: {ex.readme_path.relative_to(project_root)}:{ex.code_start_line}",
                                "Snippet execution timed out:",
                                stderr,
                                "Snippet:",
                                ex.original_code,
                            ]
                        )
                    )
                    continue
                failures.append(
                    "\n".join(
                        [
                            f"Doc: {ex.readme_path.relative_to(project_root)}:{ex.code_start_line}",
                            "Snippet execution failed:",
                            stderr,
                        ]
                    )
                )
                continue

            expected = ex.expected_output.replace("\r\n", "\n").rstrip("\n")
            actual = stdout
            if not _matches_with_ellipsis(expected, actual):
                failures.append(
                    "\n".join(
                        [
                            f"Doc: {ex.readme_path.relative_to(project_root)}:{ex.code_start_line}",
                            _diff(expected, actual),
                        ]
                    )
                )

        if failures:
            self.fail("\n\n".join(failures))
