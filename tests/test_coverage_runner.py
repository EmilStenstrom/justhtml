import subprocess
import sys
import unittest
from unittest.mock import patch

from tests.check_coverage import ROOT, _commands, _test_ids, main


class TestCoverageRunner(unittest.TestCase):
    def test_shards_include_every_unit_test_once_and_all_fixture_suites(self) -> None:
        loader = unittest.TestLoader()
        expected = list(_test_ids(loader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT))))
        commands = _commands(3)
        actual = []
        for command in commands:
            if command[:2] == ["-m", "unittest"]:
                actual.extend(_test_ids(loader.loadTestsFromNames(command[3:])))
        assert sorted(actual) == sorted(expected)
        assert [command[2] for command in commands if command[0] == "run_tests.py"] == ["fixtures"]

    def test_failed_worker_cannot_pass_by_reporting_existing_coverage(self) -> None:
        failure = subprocess.CompletedProcess([], 1, stdout="", stderr="")
        with (
            patch.object(sys, "argv", ["check_coverage.py", "--jobs", "1"]),
            patch("tests.check_coverage._commands", return_value=[["run_tests.py"]]),
            patch("tests.check_coverage._run", return_value=failure),
            patch("tests.check_coverage.Coverage") as report,
        ):
            assert main() == 1
            report.assert_not_called()
