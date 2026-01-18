"""Project test harness entrypoint.

This script orchestrates:
- html5lib fixture suites (tree/tokenizer/serializer/encoding)
- JustHTML custom fixtures
- Python unit tests in tests/test_*.py

Detailed harness logic lives in tests/harness/*.
"""

import argparse
import signal
import sys
import unittest
from io import StringIO
from pathlib import Path

from tests.harness.encoding import _run_encoding_tests as harness_run_encoding_tests
from tests.harness.regressions import run_regression_check as harness_run_regression_check
from tests.harness.reporter import TestReporter
from tests.harness.serializer import _run_serializer_tests as harness_run_serializer_tests
from tests.harness.tokenizer import _run_tokenizer_tests as harness_run_tokenizer_tests
from tests.harness.tree import TestRunner

# Minimal Unix-friendly fix: if stdout is a pipe and the reader (e.g. `head`) closes early,
# writes would raise BrokenPipeError at interpreter shutdown.
# Reset SIGPIPE so the process exits quietly instead of emitting a traceback.
try:  # pragma: no cover - platform dependent
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
except (
    AttributeError,
    OSError,
    RuntimeError,
):  # AttributeError on non-Unix, others just in case
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-x",
        "--fail-fast",
        action="store_true",
        help="Break on first test failure",
    )
    parser.add_argument(
        "--test-specs",
        type=str,
        nargs="+",
        default=None,
        help="Space-separated list of test specs in format: file:indices (e.g., test1.dat:0,1,2 test2.dat:5,6)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity: -v show failing test diffs; -vv add parser debug for failures; -vvv capture debug for all tests (currently printed only on failures)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Quiet mode: only print the header line (no per-file breakdown). For a full unfiltered run the detailed summary is still written to test-summary.txt",
    )

    parser.add_argument(
        "--no-write-summary",
        action="store_true",
        help=(
            "Do not write test-summary.txt, even for a full (unfiltered) run. "
            "Useful for pre-commit/CI runs where hooks must not modify files."
        ),
    )
    parser.add_argument(
        "--exclude-errors",
        type=str,
        help="Skip tests containing any of these strings in their errors (comma-separated)",
    )
    parser.add_argument(
        "--exclude-files",
        type=str,
        help="Skip files containing any of these strings in their names (comma-separated)",
    )
    parser.add_argument(
        "--exclude-html",
        type=str,
        help="Skip tests containing any of these strings in their HTML input (comma-separated)",
    )
    parser.add_argument(
        "--filter-html",
        type=str,
        help="Only run tests containing any of these strings in their HTML input (comma-separated)",
    )
    parser.add_argument(
        "--filter-errors",
        type=str,
        help="Only run tests containing any of these strings in their errors (comma-separated)",
    )
    parser.add_argument(
        "--suite",
        choices=["all", "tree", "justhtml", "tokenizer", "serializer", "encoding", "unit"],
        default="all",
        help=(
            "Run a single suite instead of the full test run. "
            "Choices: all, tree, justhtml, tokenizer, serializer, encoding, unit (default: all)."
        ),
    )
    parser.add_argument(
        "--regressions",
        action="store_true",
        help="After a full (unfiltered) run, compare results to committed HEAD test-summary.txt and report new failures (exits 1 if regressions).",
    )
    parser.add_argument(
        "--check-errors",
        action="store_true",
        help=(
            "Enable additional error validation. For html5lib tree-construction .dat tests, "
            "validates the number of parse errors; for tests/justhtml-tests, validates the exact "
            "ordered list of error codes; for html5lib tokenizer .test tests, validates tokenizer "
            "parse errors (code+line+col) when provided by the fixture."
        ),
    )
    args = parser.parse_args()

    if args.regressions and args.suite != "all":
        parser.error("--regressions requires --suite all")

    # Preserve each provided spec exactly so patterns like 'tests1.dat:1,2,3' remain intact.
    # Keeping the raw spec strings allows _should_run_test to parse the comma-separated index
    # list correctly.
    test_specs = list(args.test_specs or [])

    exclude_errors = args.exclude_errors.split(",") if args.exclude_errors else None
    exclude_files = args.exclude_files.split(",") if args.exclude_files else None
    exclude_html = args.exclude_html.split(",") if args.exclude_html else None
    filter_html = args.filter_html.split(",") if args.filter_html else None
    filter_errors = args.filter_errors.split(",") if args.filter_errors else None

    return {
        "fail_fast": args.fail_fast,
        "test_specs": test_specs,
        "quiet": args.quiet,
        "write_summary": not args.no_write_summary,
        "exclude_errors": exclude_errors,
        "exclude_files": exclude_files,
        "exclude_html": exclude_html,
        "filter_html": filter_html,
        "filter_errors": filter_errors,
        "verbosity": args.verbose,
        "regressions": args.regressions,
        "check_errors": args.check_errors,
        "suite": args.suite,
    }


# ---------------- Python unittest runner ----------------


def _run_unit_tests(config):
    """Discover and run Python unittest files in tests/ directory."""
    test_dir = Path("tests")
    test_specs = config.get("test_specs", [])
    quiet = config.get("quiet", False)
    verbosity = config.get("verbosity", 0)

    # Find all test_*.py files
    test_files = sorted(test_dir.glob("test_*.py"))

    if not test_files:
        return 0, 0, {}

    # Filter by test_specs if provided
    if test_specs:
        filtered_files = []
        for tf in test_files:
            for spec in test_specs:
                spec_file = spec.split(":")[0] if ":" in spec else spec
                if spec_file in tf.name or tf.name in spec_file:
                    filtered_files.append(tf)
                    break
        test_files = filtered_files

    if not test_files:
        return 0, 0, {}

    class _CollectingResult(unittest.TextTestResult):
        def __init__(self, *args, file_key: str, **kwargs):
            super().__init__(*args, **kwargs)
            self._file_key = file_key
            self._test_counter = 0
            self._test_to_index: dict[str, int] = {}
            self.test_indices: list[tuple[str, int]] = []

        def startTest(self, test):  # noqa: N802 - unittest API
            super().startTest(test)
            tid = test.id()
            if tid not in self._test_to_index:
                self._test_to_index[tid] = self._test_counter
                self._test_counter += 1

        def _idx(self, test) -> int:
            return self._test_to_index.get(test.id(), -1)

        def addSuccess(self, test):  # noqa: N802 - unittest API
            super().addSuccess(test)
            self.test_indices.append(("pass", self._idx(test)))

        def addFailure(self, test, err):  # noqa: N802 - unittest API
            super().addFailure(test, err)
            self.test_indices.append(("fail", self._idx(test)))

        def addError(self, test, err):  # noqa: N802 - unittest API
            super().addError(test, err)
            self.test_indices.append(("fail", self._idx(test)))

        def addSkip(self, test, reason):  # noqa: N802 - unittest API
            super().addSkip(test, reason)
            self.test_indices.append(("skip", self._idx(test)))

    total_passed = 0
    total_failed = 0
    file_results = {}

    for test_file in test_files:
        # Load tests from file (module-only), so patterns and counts match that filename.
        module_name = f"tests.{test_file.stem}"
        __import__(module_name)
        module = sys.modules[module_name]
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(module)

        # Run tests
        stream = StringIO() if quiet or verbosity < 1 else sys.stdout
        runner = unittest.TextTestRunner(
            stream=stream,
            verbosity=0 if quiet else verbosity,
            resultclass=(
                lambda *args, file_key=test_file.name, **kwargs: _CollectingResult(*args, file_key=file_key, **kwargs)
            ),
        )
        result: _CollectingResult = runner.run(suite)

        file_skipped = len(getattr(result, "skipped", []))
        file_failed = len(result.failures) + len(result.errors)
        file_passed = result.testsRun - file_failed - file_skipped

        total_passed += file_passed
        total_failed += file_failed

        file_results[test_file.name] = {
            "passed": file_passed,
            "failed": file_failed,
            "skipped": file_skipped,
            "total": result.testsRun,
            "test_indices": result.test_indices,
        }

        # Print failures if verbose
        if verbosity >= 1 and not quiet and (result.failures or result.errors):
            for test, traceback in result.failures + result.errors:
                print(f"\nFAILED: {test}")
                print(traceback)

    return total_passed, total_failed, file_results


def main():
    config = parse_args()
    test_dir = Path("tests")

    suite = config.get("suite", "all")
    run_tree = suite in {"all", "tree"}
    run_justhtml_tree = suite in {"all", "justhtml"}
    run_tokenizer = suite in {"all", "tokenizer"}
    run_serializer = suite in {"all", "serializer"}
    run_encoding = suite in {"all", "encoding"}
    run_unit = suite in {"all", "unit"}

    # Check that html5lib-tests symlinks exist (only for the selected suites)
    tree_tests = test_dir / "html5lib-tests-tree"
    tokenizer_tests = test_dir / "html5lib-tests-tokenizer"
    serializer_tests = test_dir / "html5lib-tests-serializer"
    encoding_tests = test_dir / "html5lib-tests-encoding"
    missing = []
    if run_tree and not tree_tests.exists():
        missing.append(str(tree_tests))
    if run_tokenizer and not tokenizer_tests.exists():
        missing.append(str(tokenizer_tests))
    if run_serializer and not serializer_tests.exists():
        missing.append(str(serializer_tests))
    if run_encoding and not encoding_tests.exists():
        missing.append(str(encoding_tests))
    if len(missing) > 0:
        print("ERROR: html5lib-tests not found. Please create symlinks:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        print("\nTo set up, clone html5lib-tests and create symlinks:", file=sys.stderr)
        print("  git clone https://github.com/html5lib/html5lib-tests.git ../html5lib-tests", file=sys.stderr)
        if run_tree:
            print("  ln -s ../../html5lib-tests/tree-construction tests/html5lib-tests-tree", file=sys.stderr)
        if run_tokenizer:
            print("  ln -s ../../html5lib-tests/tokenizer tests/html5lib-tests-tokenizer", file=sys.stderr)
        if run_serializer:
            print("  ln -s ../../html5lib-tests/serializer tests/html5lib-tests-serializer", file=sys.stderr)
        if run_encoding:
            print("  ln -s ../../html5lib-tests/encoding tests/html5lib-tests-encoding", file=sys.stderr)
        sys.exit(1)
    reporter = TestReporter(config)

    total_passed = 0
    total_failed = 0
    total_skipped = 0
    combined_results = {}

    runner = None

    if run_tree:
        runner = TestRunner(tree_tests, config)
        tree_passed, tree_failed, skipped = runner.run()
        total_passed += tree_passed
        total_failed += tree_failed
        total_skipped += skipped

        # With fail-fast enabled, stop after the first failing suite to avoid
        # printing large summaries of unrelated passing tests.
        if config.get("fail_fast") and tree_failed:
            sys.exit(1)

        combined_results.update(runner.file_results)

    if run_justhtml_tree:
        # Run JustHTML-specific tree-construction tests (custom .dat fixtures).
        # These live outside the upstream html5lib-tests checkout.
        justhtml_tree_tests = test_dir / "justhtml-tests"
        justhtml_runner = TestRunner(justhtml_tree_tests, config)
        justhtml_tree_passed, justhtml_tree_failed, justhtml_tree_skipped = justhtml_runner.run()
        total_passed += justhtml_tree_passed
        total_failed += justhtml_tree_failed
        total_skipped += justhtml_tree_skipped

        if config.get("fail_fast") and justhtml_tree_failed:
            sys.exit(1)

        combined_results.update(justhtml_runner.file_results)

    if run_tokenizer:
        tok_passed, tok_total, tok_file_results = harness_run_tokenizer_tests(config)
        total_passed += tok_passed
        total_failed += tok_total - tok_passed
        combined_results.update(tok_file_results)

        if config.get("fail_fast") and (tok_total - tok_passed):
            sys.exit(1)

    if run_serializer:
        ser_passed, ser_total, ser_skipped, ser_file_results = harness_run_serializer_tests(config)
        total_passed += ser_passed
        total_failed += ser_total - ser_passed - ser_skipped
        total_skipped += ser_skipped
        combined_results.update(ser_file_results)

        if config.get("fail_fast") and (ser_total - ser_passed - ser_skipped):
            sys.exit(1)

    if run_encoding:
        enc_passed, enc_total, enc_skipped, enc_file_results = harness_run_encoding_tests(config)
        total_passed += enc_passed
        total_failed += enc_total - enc_passed - enc_skipped
        total_skipped += enc_skipped
        combined_results.update(enc_file_results)

        if config.get("fail_fast") and (enc_total - enc_passed - enc_skipped):
            sys.exit(1)

    if run_unit:
        unit_passed, unit_failed, unit_file_results = _run_unit_tests(config)
        total_passed += unit_passed
        total_failed += unit_failed
        combined_results.update(unit_file_results)

        if config.get("fail_fast") and unit_failed:
            sys.exit(1)

    reporter.print_summary(
        total_passed,
        total_failed,
        total_skipped,
        combined_results,
    )

    if total_failed:
        sys.exit(1)

    if config.get("regressions"):
        # Only meaningful for full unfiltered run
        if not reporter.is_full_run():
            return
        if runner is None:
            return
        harness_run_regression_check(runner, reporter)


if __name__ == "__main__":
    main()
