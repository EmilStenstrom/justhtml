import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO, TextIOWrapper
from tempfile import NamedTemporaryFile

import justhtml.__main__ as cli


class TestCLI(unittest.TestCase):
    def _run_cli(self, argv, stdin_text=""):
        stdout = StringIO()
        stderr = StringIO()

        old_argv = sys.argv
        old_stdin = sys.stdin
        try:
            sys.argv = ["justhtml", *argv]
            sys.stdin = StringIO(stdin_text)
            with redirect_stdout(stdout), redirect_stderr(stderr):
                try:
                    cli.main()
                except SystemExit as e:
                    return e.code, stdout.getvalue(), stderr.getvalue()
            return 0, stdout.getvalue(), stderr.getvalue()
        finally:
            sys.argv = old_argv
            sys.stdin = old_stdin

    def test_help(self):
        code, out, err = self._run_cli(["--help"])
        self.assertEqual(code, 0)
        self.assertIn("usage: justhtml", out)
        self.assertIn("--selector", out)
        self.assertIn("--format", out)
        self.assertEqual(err, "")

    def test_version(self):
        code, out, err = self._run_cli(["--version"])
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("justhtml "))
        self.assertEqual(err, "")

    def test_no_args_prints_help_and_exits_1(self):
        code, out, err = self._run_cli([])
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertIn("usage: justhtml", err)

    def test_stdin_html_default_format_html(self):
        html = "<p>Hello <b>world</b></p>"
        code, out, err = self._run_cli(["-"], stdin_text=html)
        self.assertEqual(code, 0)
        self.assertIn("<p>", out)
        self.assertIn("Hello", out)
        self.assertIn("world", out)
        self.assertEqual(err, "")

    def test_format_html_preserves_preformatted_text(self):
        html = "<pre><code>a</code>-&gt;<code>b</code></pre>"
        code, out, err = self._run_cli(["-", "--format", "html"], stdin_text=html)
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        # Pretty-printing should not inject whitespace/newlines inside <pre>.
        self.assertIn("</code>-&gt;<code>", out)

    def test_fragment_parsing_does_not_insert_document_wrappers(self):
        html = "<li>Hi</li>"
        code, out, err = self._run_cli(["-", "--fragment"], stdin_text=html)
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        self.assertEqual(out, "<li>Hi</li>\n")

    def test_stdin_non_utf8_bytes_does_not_crash(self):
        stdout = StringIO()
        stderr = StringIO()

        old_argv = sys.argv
        old_stdin = sys.stdin
        try:
            sys.argv = ["justhtml", "-", "--format", "text"]
            sys.stdin = TextIOWrapper(BytesIO(b"<p>Hello</p>\xfc"), encoding="utf-8", errors="strict")
            with redirect_stdout(stdout), redirect_stderr(stderr):
                try:
                    cli.main()
                except SystemExit as e:
                    self.assertEqual(e.code, 0)
                    self.assertIn("Hello", stdout.getvalue())
                    return
            self.assertIn("Hello", stdout.getvalue())
        finally:
            sys.argv = old_argv
            sys.stdin = old_stdin

    def test_selector_text_multiple_matches(self):
        html = "<article><p>Hi <b>there</b></p><p>Bye</p></article>"
        code, out, err = self._run_cli(["-", "--selector", "p", "--format", "text"], stdin_text=html)
        self.assertEqual(code, 0)
        self.assertEqual(out, "Hi there\nBye\n")
        self.assertEqual(err, "")

    def test_format_text_sanitizes_by_default(self):
        html = "<p>Hello<script>alert(1)</script>World</p>"
        code, out, err = self._run_cli(["-", "--format", "text"], stdin_text=html)
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        self.assertEqual(out, "Hello World\n")

    def test_format_text_unsafe_includes_script_text(self):
        html = "<p>Hello<script>alert(1)</script>World</p>"
        code, out, err = self._run_cli(["-", "--format", "text", "--unsafe"], stdin_text=html)
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        self.assertEqual(out, "Hello alert(1) World\n")

    def test_selector_text_first(self):
        html = "<article><p>Hi <b>there</b></p><p>Bye</p></article>"
        code, out, err = self._run_cli(
            ["-", "--selector", "p", "--format", "text", "--first"],
            stdin_text=html,
        )
        self.assertEqual(code, 0)
        self.assertEqual(out, "Hi there\n")
        self.assertEqual(err, "")

    def test_selector_markdown(self):
        html = "<article><p>Hello <b>world</b></p></article>"
        code, out, err = self._run_cli(
            [
                "-",
                "--selector",
                "article",
                "--allow-tags",
                ",, article,,",
                "--format",
                "markdown",
            ],
            stdin_text=html,
        )
        self.assertEqual(code, 0)
        self.assertEqual(out, "Hello **world**\n")
        self.assertEqual(err, "")

    def test_selector_no_matches_exits_1(self):
        html = "<p>Hello</p>"
        code, out, err = self._run_cli(["-", "--selector", ".does-not-exist"], stdin_text=html)
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertEqual(err, "")

    def test_invalid_selector_exits_2_and_writes_stderr(self):
        html = "<p>Hello</p>"
        code, out, err = self._run_cli(["-", "--selector", "["], stdin_text=html)
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertNotEqual(err, "")

    def test_file_input_path(self):
        html = "<p>Hello</p>"
        with NamedTemporaryFile("w+", suffix=".html") as f:
            f.write(html)
            f.flush()
            code, out, err = self._run_cli([f.name, "--format", "text"])
        self.assertEqual(code, 0)
        self.assertEqual(out, "Hello\n")
        self.assertEqual(err, "")

    def test_output_writes_to_file_and_not_stdout(self):
        html = "<p>Hello</p>"
        with NamedTemporaryFile("r+", suffix=".txt") as out_file:
            code, out, err = self._run_cli(["-", "--format", "text", "--output", out_file.name], stdin_text=html)
            self.assertEqual(code, 0)
            self.assertEqual(out, "")
            self.assertEqual(err, "")
            out_file.seek(0)
            self.assertEqual(out_file.read(), "Hello\n")

    def test_separator_changes_text_joining(self):
        html = "<p>Hello <b>world</b></p>"
        code, out, err = self._run_cli(["-", "--format", "text", "--separator", "|"], stdin_text=html)
        self.assertEqual(code, 0)
        self.assertEqual(out, "Hello|world\n")
        self.assertEqual(err, "")

    def test_no_strip_preserves_whitespace(self):
        html = "<p>  Hello  <b>world</b>  </p>"
        code, out, err = self._run_cli(["-", "--format", "text", "--separator", "|", "--no-strip"], stdin_text=html)
        self.assertEqual(code, 0)
        self.assertEqual(out, "  Hello  |world|  \n")
        self.assertEqual(err, "")

    def test_no_strip_with_default_separator(self):
        html = "<p>Hello<b>world</b></p>"
        code, out, err = self._run_cli(["-", "--format", "text", "--no-strip"], stdin_text=html)
        self.assertEqual(code, 0)
        self.assertEqual(out, "Hello world\n")
        self.assertEqual(err, "")
