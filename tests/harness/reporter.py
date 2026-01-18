from __future__ import annotations

import math
import re
from pathlib import Path


class TestReporter:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def _escape_control_chars_for_display(text: str) -> str:
        if not text:
            return text
        out = []
        for ch in text:
            code = ord(ch)
            if ch == "\n":
                out.append(ch)
            elif ch == "\t":
                out.append("\\t")
            elif ch == "\r":
                out.append("\\r")
            elif ch == "\f":
                out.append("\\x0c")
            elif code < 0x20 or code == 0x7F:
                out.append(f"\\x{code:02x}")
            else:
                out.append(ch)
        return "".join(out)

    def is_full_run(self):
        return (self.config.get("suite", "all") == "all") and not (
            self.config.get("test_specs")
            or self.config.get("exclude_files")
            or self.config.get("exclude_errors")
            or self.config.get("filter_errors")
            or self.config.get("exclude_html")
            or self.config.get("filter_html")
            or self.config.get("check_errors")
        )

    def print_test_result(self, result):
        verbosity = self.config["verbosity"]
        if result.passed:
            return
        if verbosity >= 1:
            show_error_diff = self.config.get("check_errors") and not result.errors_matched
            show_tree_diff = not result.tree_matched
            lines = [
                "FAILED:",
                f"=== INCOMING HTML ===\n{self._escape_control_chars_for_display(result.input_html)}\n",
            ]
            if show_error_diff:
                expected_str = "\n".join(result.expected_errors) if result.expected_errors else "(none)"
                actual_str = "\n".join(result.actual_errors) if result.actual_errors else "(none)"
                lines.append(f"=== EXPECTED ERRORS ===\n{expected_str}\n")
                lines.append(f"=== ACTUAL ERRORS ===\n{actual_str}\n")

            if show_tree_diff:
                lines.append(f"=== WHATWG HTML5 SPEC COMPLIANT TREE ===\n{result.expected_output}\n")
                lines.append(f"=== CURRENT PARSER OUTPUT TREE ===\n{result.actual_output}")
            if verbosity >= 2 and result.debug_output:
                lines.insert(3, f"=== DEBUG PRINTS WHEN PARSING ===\n{result.debug_output.rstrip()}\n")
            print("\n".join(lines))

    def print_summary(self, passed, failed, skipped=0, file_results=None):
        total = passed + failed
        percentage = math.floor(passed * 1000 / total) / 10 if total else 0
        result = "FAILED" if failed else "PASSED"
        header = f"{result}: {passed}/{total} passed ({percentage}%)"
        if skipped:
            header += f", {skipped} skipped"

        full_run = self.is_full_run()
        summary_file = "test-summary.txt"
        write_summary = bool(self.config.get("write_summary", True))

        if not file_results:
            if full_run and write_summary:
                Path(summary_file).write_text(header + "\n")
            return

        detailed = self._generate_detailed_summary(header, file_results)
        if full_run and write_summary:
            Path(summary_file).write_text(detailed + "\n")
        if self.config.get("quiet"):
            print(header)
        else:
            print(detailed)

    def _generate_detailed_summary(self, overall_summary, file_results):
        lines = []

        def natural_sort_key(filename):
            return [int(text) if text.isdigit() else text.lower() for text in re.split("([0-9]+)", filename)]

        sorted_files = sorted(file_results.keys(), key=natural_sort_key)

        for filename in sorted_files:
            result = file_results[filename]
            runnable_tests = result["passed"] + result["failed"]
            skipped_tests = result.get("skipped", 0)

            if runnable_tests > 0:
                percentage = round(result["passed"] * 100 / runnable_tests)
                status_line = f"{filename}: {result['passed']}/{runnable_tests} ({percentage}%)"
            else:
                status_line = f"{filename}: 0/0 (N/A)"

            pattern = self.generate_test_pattern(result["test_indices"])
            if pattern:
                status_line += f" [{pattern}]"

            if skipped_tests > 0:
                status_line += f" ({skipped_tests} skipped)"

            lines.append(status_line)

        lines.extend(["", overall_summary])
        return "\n".join(lines)

    def generate_test_pattern(self, test_indices):
        if not test_indices:
            return ""
        sorted_tests = sorted(test_indices, key=lambda x: x[1])
        pattern = ""
        for status, _idx in sorted_tests:
            if status == "pass":
                pattern += "."
            elif status == "fail":
                pattern += "x"
            elif status == "skip":
                pattern += "s"
        return pattern
