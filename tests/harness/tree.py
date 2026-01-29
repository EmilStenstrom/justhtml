from __future__ import annotations

import os
import re
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from justhtml import JustHTML, to_test_format
from justhtml.context import FragmentContext
from justhtml.tokenizer import TokenizerOpts

from .reporter import TestReporter


class TestCase:
    __slots__ = [
        "data",
        "document",
        "errors",
        "fragment_context",
        "iframe_srcdoc",
        "script_directive",
        "xml_coercion",
    ]

    def __init__(
        self,
        data,
        errors,
        document,
        fragment_context=None,
        script_directive=None,
        xml_coercion=False,
        iframe_srcdoc=False,
    ):
        self.data = data
        self.errors = errors
        self.document = document
        self.fragment_context = fragment_context
        self.script_directive = script_directive
        self.xml_coercion = xml_coercion
        self.iframe_srcdoc = iframe_srcdoc


class TestResult:
    __slots__ = [
        "actual_error_count",
        "actual_errors",
        "actual_output",
        "debug_output",
        "error_check_mode",
        "errors_matched",
        "expected_error_count",
        "expected_errors",
        "expected_output",
        "input_html",
        "passed",
        "tree_matched",
    ]

    def __init__(
        self,
        passed,
        input_html,
        expected_errors,
        expected_output,
        actual_output,
        actual_errors=None,
        errors_matched=False,
        error_check_mode="codes",
        expected_error_count=None,
        actual_error_count=None,
        tree_matched=False,
        debug_output="",
    ):
        self.passed = passed
        self.input_html = input_html
        self.expected_errors = expected_errors
        self.expected_output = expected_output
        self.actual_output = actual_output
        self.actual_errors = actual_errors or []
        self.errors_matched = errors_matched
        self.error_check_mode = error_check_mode
        self.expected_error_count = expected_error_count
        self.actual_error_count = actual_error_count
        self.tree_matched = tree_matched
        self.debug_output = debug_output


def compare_outputs(expected, actual):
    def normalize(text: str) -> str:
        return "\n".join(line.rstrip() for line in text.strip().splitlines())

    return normalize(expected) == normalize(actual)


class TestRunner:
    def __init__(self, test_dir, config):
        self.test_dir = test_dir
        self.config = config
        self.results = []
        self.file_results = {}

    def _natural_sort_key(self, path):
        def convert(text):
            return int(text) if text.isdigit() else text.lower()

        return [convert(c) for c in re.split("([0-9]+)", str(path))]

    def _parse_dat_file(self, path):
        with path.open("r", encoding="utf-8", newline="") as f:
            content = f.read()
        tests = []
        lines = content.split("\n")

        current_test_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            current_test_lines.append(line)

            if i + 1 >= len(lines) or (i + 1 < len(lines) and lines[i + 1] == "#data"):
                if current_test_lines and any(line.strip() for line in current_test_lines):
                    test = self._parse_single_test(current_test_lines)
                    if test:
                        tests.append(test)

                current_test_lines = []

            i += 1

        return tests

    def _decode_escapes(self, text):
        if "\\x" not in text and "\\u" not in text:
            return text
        result = []
        i = 0
        while i < len(text):
            if text[i : i + 2] == "\\x" and i + 3 < len(text):
                try:
                    byte_val = int(text[i + 2 : i + 4], 16)
                    result.append(chr(byte_val))
                    i += 4
                    continue
                except ValueError:
                    pass
            elif text[i : i + 2] == "\\u" and i + 5 < len(text):
                try:
                    code_point = int(text[i + 2 : i + 6], 16)
                    result.append(chr(code_point))
                    i += 6
                    continue
                except ValueError:
                    pass
            result.append(text[i])
            i += 1
        return "".join(result)

    def _parse_single_test(self, lines):
        data = []
        errors = []
        document = []
        fragment_context = None
        script_directive = None
        xml_coercion = False
        iframe_srcdoc = False
        mode = None

        for line in lines:
            if line.startswith("#"):
                directive = line[1:]
                if directive in ("script-on", "script-off"):
                    script_directive = directive
                elif directive == "xml-coercion":
                    xml_coercion = True
                elif directive == "iframe-srcdoc":
                    iframe_srcdoc = True
                elif directive == "new-errors":
                    mode = "errors"
                else:
                    mode = directive
            elif mode == "data":
                data.append(line)
            elif mode == "errors":
                errors.append(line)
            elif mode == "document":
                document.append(line)
            elif mode == "document-fragment":
                fragment_str = line.strip()
                if " " in fragment_str:
                    namespace, tag_name = fragment_str.split(" ", 1)
                    fragment_context = FragmentContext(tag_name, namespace)
                else:
                    fragment_context = FragmentContext(fragment_str)

        if data or document:
            raw_data = "\n".join(data)
            return TestCase(
                data=self._decode_escapes(raw_data),
                errors=errors,
                document="\n".join(document),
                fragment_context=fragment_context,
                script_directive=script_directive,
                xml_coercion=xml_coercion,
                iframe_srcdoc=iframe_srcdoc,
            )

        return None

    def _should_run_test(self, filename, index, test):
        if test.script_directive == "script-on" and "<script" in test.data.lower():
            return False

        if self.config["test_specs"]:
            spec_match = False
            for spec in self.config["test_specs"]:
                if ":" in spec:
                    spec_file, indices = spec.split(":")
                    if filename == spec_file and str(index) in indices.split(","):
                        spec_match = True
                        break
                else:
                    if spec in filename:
                        spec_match = True
                        break
            if not spec_match:
                return False

        if self.config["exclude_html"]:
            if any(exclude in test.data for exclude in self.config["exclude_html"]):
                return False

        if self.config["filter_html"]:
            if not any(include in test.data for include in self.config["filter_html"]):
                return False

        if self.config["exclude_errors"] and any(
            exclude in error for exclude in self.config["exclude_errors"] for error in test.errors
        ):
            return False

        return not (
            self.config["filter_errors"]
            and not any(include in error for include in self.config["filter_errors"] for error in test.errors)
        )

    def load_tests(self):
        test_files = self._collect_test_files()
        return [(path, self._parse_dat_file(path)) for path in test_files]

    def _collect_test_files(self):
        files = []
        for root, _, filenames in os.walk(self.test_dir, followlinks=True):
            files.extend(Path(root) / filename for filename in filenames if filename.endswith(".dat"))

        if self.config["exclude_files"]:
            files = [f for f in files if not any(exclude in f.name for exclude in self.config["exclude_files"])]

        return sorted(files, key=self._natural_sort_key)

    def run(self):
        passed = failed = skipped = 0

        for file_path, tests in self.load_tests():
            file_passed = file_failed = file_skipped = 0
            file_test_indices = []

            for i, test in enumerate(tests):
                if not self._should_run_test(file_path.name, i, test):
                    if test.script_directive in ("script-on", "script-off"):
                        skipped += 1
                        file_skipped += 1
                        file_test_indices.append(("skip", i))
                    continue

                result = self._run_single_test(test, xml_coercion=test.xml_coercion)
                self.results.append(result)

                if result.passed:
                    passed += 1
                    file_passed += 1
                    file_test_indices.append(("pass", i))
                else:
                    failed += 1
                    file_failed += 1
                    file_test_indices.append(("fail", i))
                    self._handle_failure(file_path, i, result)

                if failed and self.config["fail_fast"]:
                    relative_path = file_path.relative_to(self.test_dir)
                    key = str(relative_path)
                    if self.test_dir.name != "tests":
                        key = f"{self.test_dir.name}/{key}"
                    self.file_results[key] = {
                        "passed": file_passed,
                        "failed": file_failed,
                        "skipped": file_skipped,
                        "total": file_passed + file_failed + file_skipped,
                        "test_indices": file_test_indices,
                    }
                    return passed, failed, skipped

            if file_test_indices:
                if self.config.get("test_specs") and file_passed == 0 and file_failed == 0:
                    pass
                else:
                    relative_path = file_path.relative_to(self.test_dir)
                    key = str(relative_path)
                    if self.test_dir.name != "tests":
                        key = f"{self.test_dir.name}/{key}"

                    self.file_results[key] = {
                        "passed": file_passed,
                        "failed": file_failed,
                        "skipped": file_skipped,
                        "total": file_passed + file_failed + file_skipped,
                        "test_indices": file_test_indices,
                    }

        return passed, failed, skipped

    def _run_single_test(self, test, xml_coercion=False):
        verbosity = self.config["verbosity"]
        capture_debug = verbosity >= 2
        debug_output = ""
        opts = TokenizerOpts(xml_coercion=xml_coercion)
        if test.script_directive in {"script-on", "script-off"}:
            opts.scripting_enabled = test.script_directive == "script-on"
        if capture_debug:
            f = StringIO()
            with redirect_stdout(f):
                parser = JustHTML(
                    test.data,
                    debug=True,
                    fragment_context=test.fragment_context,
                    tokenizer_opts=opts,
                    iframe_srcdoc=test.iframe_srcdoc,
                    scripting_enabled=opts.scripting_enabled,
                    collect_errors=True,
                    sanitize=False,
                )
                actual_tree = to_test_format(parser.root)
            debug_output = f.getvalue()
        else:
            parser = JustHTML(
                test.data,
                fragment_context=test.fragment_context,
                tokenizer_opts=opts,
                iframe_srcdoc=test.iframe_srcdoc,
                scripting_enabled=opts.scripting_enabled,
                collect_errors=True,
                sanitize=False,
            )
            actual_tree = to_test_format(parser.root)

        tree_passed = compare_outputs(test.document, actual_tree)
        error_check_mode = "count" if self.test_dir.name == "html5lib-tests-tree" else "codes"

        if error_check_mode == "count":
            expected_count = len([line for line in test.errors if line.strip()])
            actual_count = len(parser.errors)
            errors_matched = actual_count == expected_count
            expected_errors = test.errors
        else:
            actual_codes = [e.code for e in parser.errors]
            expected_codes = self._extract_error_codes(test.errors)
            errors_matched = actual_codes == expected_codes
            expected_count = None
            actual_count = None
            expected_errors = test.errors

        actual_error_strs = [f"({e.line},{e.column}): {e.code}" for e in parser.errors]

        if self.config.get("check_errors"):
            passed = tree_passed and errors_matched
        else:
            passed = tree_passed

        return TestResult(
            passed=passed,
            input_html=test.data,
            expected_errors=expected_errors,
            expected_output=test.document,
            actual_output=actual_tree,
            actual_errors=actual_error_strs,
            errors_matched=errors_matched,
            error_check_mode=error_check_mode,
            expected_error_count=expected_count,
            actual_error_count=actual_count,
            tree_matched=tree_passed,
            debug_output=debug_output,
        )

    def _extract_error_codes(self, error_lines):
        codes = []
        for raw_line in error_lines:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(("#", "|")):
                continue
            if ": " in line:
                code = line.split(": ", 1)[1]
            elif ") " in line:
                code = line.split(") ", 1)[1]
            else:
                code = line
            codes.append(code)
        return codes

    def _handle_failure(self, file_path, test_index, result):
        if self.config["verbosity"] >= 1 and not self.config["quiet"]:
            TestReporter(self.config).print_test_result(result)
