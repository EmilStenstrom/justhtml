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

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

- **Line length**: 119 characters
- **Target**: Python 3.10+
- **Rules**: Nearly all Ruff rules enabled (see `pyproject.toml` for exceptions)

Key style points:
- Use plain `assert` for tests, not `self.assertEqual` etc.
- Comments explain **why**, not **what**
- No typing annotations
- Cite spec sections when relevant (e.g., "Per §13.2.5.72")

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

## Architecture Notes

- **Parser engine** (`src/justhtml/parser/engine.py`): Plan-driven tokenization, tree construction, and sanitizer projection
- **Parser scanner** (`src/justhtml/parser/scanner.py`): Shared low-level tag and rawtext scanning helpers
- **DOM** (`src/justhtml/dom/`): DOM-like node tree; use `append_child()` / `insert_before()` for public tree operations
- **Selectors** (`src/justhtml/selector/`): CSS selector parsing and matching
- **Transforms** (`src/justhtml/transforms/`): Compiled post-parse tree transformations

Golden rules:
1. Follow WHATWG HTML5 spec exactly
2. No exceptions in hot paths
3. Minimal allocations in parser hot paths
4. No `hasattr`/`getattr`/`delattr` - all structures are deterministic

## Submitting Changes

1. Fork the repository
2. Create a feature branch
3. Add an entry under the relevant section of `CHANGELOG.md`
4. Make your changes with tests
5. Ensure pre-commit passes
6. Submit a pull request

Questions? Open an issue on GitHub. For security vulnerabilities, please see our [Security Policy](SECURITY.md).
