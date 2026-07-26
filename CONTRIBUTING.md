# Contributing to JustHTML

Thanks for considering contributing to JustHTML! This document explains how to set up your development environment and the standards we follow.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/emilstenstrom/justhtml.git
   cd justhtml
   ```

2. Create a virtual environment and install dev dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Running Tests

The test suite uses the web platform html5 treebuilder tests plus additional tests for selector functionality.

If you want to run the full external fixture suite locally, clone the web platform html5 treebuilder tests and `html5lib-tests` serializer/encoding fixtures next to this repository, then create the symlinks described in [tests/README.md](tests/README.md).

```bash
# Run all tests
python run_tests.py

# Run one suite (faster iteration)
python run_tests.py --suite tree
python run_tests.py --suite justhtml
python run_tests.py --suite tokenizer

# Run with coverage report
coverage run run_tests.py && coverage report

# Run specific test file
python run_tests.py --test-specs test2.test:5,10 -v

# Quick iteration - test a snippet
python -c 'from justhtml import JustHTML, to_test_format; print(to_test_format(JustHTML("<html>").root))'
```

**Coverage is required to be 100%.** All new code must be fully tested.

### Test placement

- Put input/output parser recovery cases in the matching
  `tests/justhtml-tests/*.dat` behavior file. Keep Python tests for public API
  behavior, internal state invariants, and cases that need programmatic setup.
- Add regressions to the behavior-specific test module or data file; do not
  create generic "edge case", "regression", or "coverage" buckets.
- When changing shared scanner behavior, test both `JustHTML` and `stream()`.
- When changing plan-selected parsing or sanitization, test the fused default
  path and the raw-plus-compiled-transform path.
- When changing HTML serialization, exercise compact and pretty output when
  both traverse the affected construct.
- When changing an adaptive index or cache, cover behavior below and above its
  activation threshold and every stack or tree mutation that can invalidate it.

### Documentation examples

Runnable documentation examples use an explicit output marker. Place it between
a self-contained Python fence and the expected-output fence:

````text
```python
from justhtml import JustHTML
print(JustHTML("<p>x</p>", fragment=True).to_html())
```

<!-- justhtml: output -->

```html
<p>x</p>
```
````

The docs test executes each marked Python block independently and compares its
stdout with the following fence. Keep ordinary HTML fences unmarked so they
remain explanatory examples rather than tests.

## Pre-commit Hooks

Pre-commit runs automatically on every commit and checks:

- **Trailing whitespace** and **end-of-file** formatting
- **YAML** and **TOML** validity
- **Ruff check** - linting with auto-fix
- **Ruff format** - code formatting
- **Tests & Coverage** - full test suite with 100% coverage requirement
- **Parser Differential** - exact agreement with the reference parser path across scored web platform html5 treebuilder cases

Run manually:
```bash
pre-commit run --all-files
```

## Coding Standards

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

- **Line length**: 119 characters
- **Target**: Python 3.10+
- **Rules**: Nearly all Ruff rules enabled (see `pyproject.toml` for exceptions)

Code must also pass mypy with the configuration in `pyproject.toml`. Keep
function signatures and shared data structures accurately typed; do not silence
type errors with broad `Any` annotations or unnecessary ignores.

- Follow the WHATWG HTML specification for parsing behavior. Do not replace
  specified recovery rules with heuristics.
- Keep parser state and other known structures explicit. Do not use
  `hasattr`, `getattr`, or `delattr` to probe deterministic internal objects.
- Do not use exceptions for ordinary branching in parser, sanitizer,
  transform, selector, or serializer hot paths.
- Do not add production branches, constants, or comments that exist only to
  satisfy a test.
- Comments explain non-obvious reasons, constraints, or specification rules,
  not what the code already says. Cite the relevant specification section when
  useful (for example, `Per §13.2.5.72`).
- Do not leave historical narration such as "previously", "fixed", or
  "changed" in code comments. Describe the current constraint directly.
- Use plain `assert` in tests rather than `self.assertEqual` and related
  methods.
- Treat additions to `src/justhtml/__init__.py` as public API changes. Keep
  internal helpers private unless they are intentionally supported.

## Keep Changes Small

Make the smallest coherent change that solves the problem.

- Keep each pull request and commit focused on one concern.
- Prefer a narrow fix over a broad rewrite. Do not refactor adjacent code,
  rename unrelated symbols, or reformat untouched areas unless the requested
  change requires it.
- Reuse existing abstractions before adding new layers, helpers, or state.
- Add only the tests and documentation needed to establish the changed
  behavior.
- Split independent changes into separate commits or pull requests.
- Remove superseded code instead of leaving parallel implementations.

There is no fixed line limit, but diff size must follow from the problem rather
than convenience. If a change reaches hundreds of lines, first look for a
smaller design and separate mechanical work from behavioral work. Explain in
the pull request when a large change is inherently indivisible.

## Commit Messages

Use:

```text
type: imperative summary
```

Use a lowercase type:

- `feat`: user-visible functionality
- `fix`: correctness bug
- `security`: security or hostile-input hardening
- `perf`: measured performance improvement
- `refactor`: internal restructuring without behavior changes
- `test`: test-only change
- `docs`: documentation-only change
- `ci`, `build`, or `deps`: tooling, packaging, or dependency changes

Keep the summary concise, specific, and written as an imperative action, for
example `fix: preserve text in legacy rawtext elements`. Do not end it with a
period.

Use a body when the reason, tradeoff, security boundary, or performance result
is not obvious from the diff. Explain why the change is needed; do not narrate
the implementation line by line. Reference issues or specifications there when
useful.

## Benchmarking

After making changes, verify performance impact:

```bash
# Quick benchmark
python benchmarks/performance.py --parsers justhtml --iterations 1

# Profile hotspots
python benchmarks/profile.py
```

For a parser-speed improvement, measure the relevant pipeline on the web100k
corpus before and after the change; see [PERFORMANCE.md](PERFORMANCE.md). For
an availability fix, add a regression that exercises the affected hostile-input
shape and verify that growth is linear or otherwise bounded.

## Releases

Use `python scripts/release.py --version X.Y.Z --yes` for releases. The script
runs release checks, bumps `pyproject.toml`, commits, tags, pushes, and creates
the GitHub release. Before invoking it, ensure the worktree is clean and move
the relevant Unreleased changelog entries into a dated `## [X.Y.Z]` section;
the helper requires that section to exist.

## Submitting Changes

1. Fork the repository
2. Create a feature branch
3. Add an entry under the relevant section of `CHANGELOG.md` when the change
   affects users
4. Make your changes with tests
5. Ensure pre-commit passes
6. Submit a pull request

Changelog entries are for user-visible features, API changes, correctness
fixes, security hardening, and library performance changes. Do not add entries
for internal tooling or infrastructure such as benchmark harness changes, test
organization, CI configuration, dependency maintenance, or behavior-preserving
refactors.

Questions? Open an issue on GitHub. For security vulnerabilities, please see our [Security Policy](SECURITY.md).
