"""Run every fixture suite and unit test in parallel, then enforce coverage."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

from coverage import Coverage

if TYPE_CHECKING:
    from collections.abc import Iterator

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "coverage-timings.json"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def _test_ids(suite: unittest.TestSuite) -> Iterator[str]:
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from _test_ids(test)
        else:
            yield test.id()


def _commands(jobs: int) -> list[list[str]]:
    commands = [["run_tests.py", "--suite", "fixtures", "--no-write-summary", "--quiet"]]
    test_ids: list[str] = []
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        suite = unittest.defaultTestLoader.loadTestsFromName(f"tests.{path.stem}")
        if path.stem == "test_docs_examples":
            commands.insert(0, ["-m", "unittest", "-q", f"tests.{path.stem}"])
        else:
            test_ids.extend(_test_ids(suite))
    shard_count = max(1, jobs - 2)
    shards: list[list[str]] = [[] for _ in range(shard_count)]
    try:
        timings = json.loads(CACHE.read_text())
    except (OSError, ValueError):
        timings = {}
    if not isinstance(timings, dict):
        timings = {}
    loads = [0.0] * shard_count
    for test_id in sorted(test_ids, key=lambda name: -timings.get(name, 0.01)):
        index = min(range(shard_count), key=loads.__getitem__)
        shards[index].append(test_id)
        loads[index] += timings.get(test_id, 0.01)
    commands.extend(["-m", "unittest", "-q", *shard] for shard in shards if shard)
    return commands


def _run(command: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    test_input = None
    if command[:2] == ["-m", "unittest"]:
        test_input = "\n".join(command[3:])
        command = ["tests/check_coverage.py", "--unit-worker"]
    return subprocess.run(  # noqa: S603 - commands come from local test discovery
        [sys.executable, "-m", "coverage", "run", "--parallel-mode", *command],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        input=test_input,
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=min(10, os.cpu_count() or 1))
    parser.add_argument("--unit-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.unit_worker:
        timings = {}

        class TimingResult(unittest.TextTestResult):
            def startTest(self, test):  # noqa: N802 - unittest API
                self.started = perf_counter()
                super().startTest(test)

            def stopTest(self, test):  # noqa: N802 - unittest API
                timings[test.id()] = perf_counter() - self.started
                super().stopTest(test)

        suite = unittest.defaultTestLoader.loadTestsFromNames(sys.stdin.read().splitlines())
        result = unittest.TextTestRunner(verbosity=0, resultclass=TimingResult).run(suite)
        directory = Path(os.environ["COVERAGE_FILE"]).parent
        (directory / f"timings-{os.getpid()}.json").write_text(json.dumps(timings))
        return int(not result.wasSuccessful())
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    with tempfile.TemporaryDirectory(prefix="justhtml-coverage-") as directory:
        env = os.environ.copy()
        env["COVERAGE_FILE"] = str(Path(directory) / ".coverage")
        env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")))
        commands = _commands(args.jobs)
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(_run, command, env) for command in commands]
            failed = False
            for future in futures:
                result = future.result()
                print(result.stdout, end="")
                print(result.stderr, end="", file=sys.stderr)
                failed |= result.returncode != 0
        if failed:
            return 1
        coverage = Coverage()
        coverage.combine(data_paths=[directory], strict=True)
        coverage.save()
        if coverage.report() < 100:
            return 1
        timings = {}
        for path in Path(directory).glob("timings-*.json"):
            timings.update(json.loads(path.read_text()))
        CACHE.parent.mkdir(exist_ok=True)
        cache_tmp = CACHE.with_suffix(f".{os.getpid()}.tmp")
        cache_tmp.write_text(json.dumps(timings))
        cache_tmp.replace(CACHE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
