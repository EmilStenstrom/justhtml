# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.6.0] - 2026-02-06

### Added
- Text extraction: Add `separator_blocks_only` to `to_text()` (and CLI `--separator-blocks-only`) to only apply `separator` between block-level elements.

### Changed
- Transforms: Improve performance of URL attribute handling and comment sanitization when applying DOM transforms.

## [1.5.0] - 2026-02-02

### Added
- Serialization & Sanitization: Introduce additional serialization contexts, and update docs to talk about the importance of putting your sanitized content in the right context (see [docs/sanitization.md](docs/sanitization.md)).

### Changed
- Sanitization: Switch the sanitizer pipeline to be built up entirely of basic transform blocks (see [docs/transforms.md](docs/transforms.md)).

### Changed
- Tokenizer: Add fast-path handling for tag names and attribute parsing to reduce overhead in common cases.
- Sanitization: Speed up URL normalization and scheme validation while preserving policy semantics (see [docs/url-cleaning.md](docs/url-cleaning.md)).
- Transforms: Optimize sanitizer transform dispatch and attribute rewrite hot paths for lower per-node overhead (see [docs/transforms.md](docs/transforms.md)).

## [1.4.0] - 2026-01-29

### Changed
- Serializer: Always escape `<` and `>` in attribute values (quoted values) and escape `<` in unquoted values for spec-compliant output. This follows a [whatwg html specification and browser change](https://github.com/whatwg/html/issues/6235) not yet in the html5lib test suite.

## [1.3.0] - 2026-01-28

### Added
- Parser: Add `scripting_enabled` option to `JustHTML(...)` for HTML5 scripting flag control (affects `<noscript>` handling).

### Changed
- Sanitization: Default URL handling now strips URL-like attributes unless explicitly allowed by `UrlPolicy` (see [URL Cleaning](docs/url-cleaning.md)).

### Security
- (Severity: Low) JustHTML's parsing used "scripting disabled" mode which opened the door for [differential parsing (mXSS)](https://www.sonarsource.com/blog/mxss-the-vulnerability-hiding-in-your-code/) attacks. In "scripting disabled" mode `<noscript>` tags could be handled differently in the sanitizer compared to when being parsed by browsers with scripting enabled. This could be used to bypass the allowed_tags sanitizer. **Fortunately, the serializer escaped `<` and `>` in style tags, with contained the attack vector completely**.

  Example from justhtml==1.2.0:

  ```python
  from justhtml import JustHTML, SanitizationPolicy, UrlPolicy, UrlRule
  xss = '<noscript><style></noscript><img src=x onerror="alert(1)">'
  JustHTML(xss, fragment=True, policy=SanitizationPolicy(
    allowed_tags=["noscript", "style"],
    allowed_attributes={},
  )).to_html()
  # => <noscript>\n  <style>&lt;/noscript&gt;&lt;img src=x onerror="alert(1)"&gt;</style>\n</noscript>
  ```

  Example from justhtml==1.3.0. Note how the img tag is removed by the sanitizer.

  ```python
  from justhtml import JustHTML, SanitizationPolicy, UrlPolicy, UrlRule
  xss = '<noscript><style></noscript><img src=x onerror="alert(1)">'
  JustHTML(xss, fragment=True, policy=SanitizationPolicy(
    allowed_tags=["noscript", "style"],
    allowed_attributes={},
  )).to_html()
  # => <noscript>&lt;style&gt;</noscript>
  ```

## [1.2.0] - 2026-01-26

### Added
- Selectors: Add `:comment` pseudo-class for selecting HTML comment nodes.
- Transforms: Add `Escape(selector)` (escape an element’s tags as text while hoisting its children).
- CLI: Add `--cleanup` option to remove unhelpful output artifacts (empty links, images, and empty tags).
- Docs: Add “Learn by examples” migration page and JustHTML agent usage notes (`llms.txt`).

### Fixed
- CSS sanitization: Make it possible to allow url:s in inline styles.

### Changed
- Public API: Export all transforms from `justhtml` so they’re available via `from justhtml import ...`.

## [1.1.0] - 2026-01-24

### Added
- Docs: Add search to the documentation site.

### Fixed
- `SanitizationPolicy` now validates and normalizes `allowed_tags` / `allowed_attributes` to prevent silent misconfiguration (for example accidentally passing a string).

### Changed
- Prefer `sanitize=` over `safe=` on `JustHTML(...)` (`safe` remains as a backwards-compatible alias).

## [1.0.0] - 2026-01-21

### Changed
- Declare JustHTML stable: the public API is now considered 1.0, and breaking changes will follow SemVer.

## [0.40.0] - 2026-01-19

### Added
- Add `html_passthrough` option to `to_markdown()` to preserve raw HTML (for example `<script>`, `<style>`, and `<textarea>`) instead of dropping it by default.

### Fixed
- Playground cleanup now runs against the sanitized tree when `safe=True`, so cleanup rules also apply after unsafe URLs are stripped.

### Changed
- Playground: rename “Prune empty” to “Cleanup” and clarify behavior via tooltip.

### Docs
- Clarify transform ordering around `safe=True` and when `Sanitize(...)` runs relative to custom transforms.

## [0.39.0] - 2026-01-18
### Added
- Expand sanitize escape-mode fixtures to cover malformed markup edge cases (EOF tag fragments, bogus end tags, markup declarations).
- Add `sanitize_dom(...)` helper to re-sanitize a mutated DOM tree.

### Changed
- Rename the TokenizerOpts flag for emitting malformed markup as text to `emit_bogus_markup_as_text` (was `emit_eof_tag_as_text`).
- BREAKING: Rename DOM node classes to DOM-style names (`Node`, `Element`, `Text`, `Template`, `Comment`, `Document`, `DocumentFragment`).

## [0.38.0] - 2026-01-18
### Fixed
- Escape-mode sanitization now preserves malformed tag-like text across more tokenizer states (end tags, markup declarations, and EOF-in-tag paths) instead of dropping tail content.

## [0.37.0] - 2026-01-18
### Added
- Speed up sanitization with a fused transform and optimized regex matching. Despite these improvements, the switch from imperative style sanitization to one based on transforms is 20% slower. We believe it's worth it because of the improved reviewability of the code.

### Changed
- BREAKING: Sanitization now happens during parsing/construction instead of at serialization time. The the `safe` and `policy` keywords move from to_html to the JustHTML constructor. Before: `JustHTML(...).to_html(safe=..., policy=...)`, After: `JustHTML(safe=..., policy=...).to_html()`.

### Docs
- Update documentation to reflect sanitize-at-construction behavior.
- Add CLI documentation for `--allow-tags`.
- Add a transforms example and refresh performance benchmark snippet in README.
- Clarify lxml sanitization guidance in README.

## [0.36.0] - 2026-01-17
### Added
- Sanitization is now fully constructed from a set of transforms instead of imperative code. This makes the code reviewable in a way not seen in other libraries. See [Sanitization](docs/sanitization.md) for details.
- Add `Decide(...)`, `EditDocument(...)`, and `RewriteAttrs(...)` transforms for policy-driven editing.
- Add `SanitizationPolicy.disallowed_tag_handling` with modes: `"unwrap"` (default), `"escape"`, and `"drop"`. Escape mode mirrors bleach's strip=False behaviour which was the last missing incompatibility.
- Add `justhtml.transforms.emit_error(...)` to emit `ParseError`s from inside transform callbacks.

### Changed
- BREAKING: Unify transform hook parameters: all transforms now support both `callback=` (node hook) and `report=` (message hook). Transforms that take a primary callable now use `func` for that callable (e.g. `Edit`, `EditAttrs`, `EditDocument`, `Decide`).
- BREAKING: Removed `SanitizationPolicy.strip_disallowed_tags`, which was undocumented. Use `disallowed_tag_handling="unwrap"` to drop tags but keep/sanitize children, or `disallowed_tag_handling="escape"` to escape disallowed tags.

### Docs
- Clarify sanitization and disallowed-tag handling, including Bleach `strip=` migration guidance.

## [0.35.0] - 2026-01-11
### Added
- Add `Stage([...])` to make transform pass boundaries explicit. Stages can be nested and are flattened; if any Stage exists at the top level, surrounding top-level transforms are automatically grouped into implicit stages.

### Changed
- Transform pipelines now preserve strict left-to-right ordering semantics within a stage (no transform-type “magic ordering”).

### Docs
- Refine transform documentation around stages and multi-pass semantics (see [Transforms](docs/transforms.md)).

## [0.34.0] - 2026-01-10
### Changed
- `Sanitize(...)` can now be used inline anywhere in a transform pipeline (it is no longer required to be last).
- Pretty-printing is more readable for Wikipedia-like markup:
  - mixed inline text + block children (e.g. `ul`) no longer loses indentation
  - “inline runs” are split into separate lines when the input contains formatting whitespace between siblings

## [0.33.0] - 2026-01-10
### Added
- Add `CollapseWhitespace(...)` transform (html5lib-style whitespace collapsing) (see [Transforms](docs/transforms.md)).

### Changed
- Unify the default “whitespace-preserving elements” across pretty-printing and text transforms; whitespace is now consistently preserved inside `pre`, `code`, `textarea`, `script`, and `style`.
- `Linkify(...)` now skips `textarea` by default (in addition to `a`, `pre`, `code`, `script`, and `style`).
- `Sanitize(...)` must still be last, except it may be followed by cleanup transforms like `PruneEmpty(...)` and `CollapseWhitespace(...)`.

### Docs
- Expand `Drop(...)` examples (see [Transforms](docs/transforms.md)).
- Document Bleach/html5lib whitespace filter migration to `CollapseWhitespace(...)` (see [Migrating from Bleach](docs/bleach-migration.md)).

## [0.32.0] - 2026-01-10
### Added
- Add constructor-time DOM transforms via `JustHTML(..., transforms=[...])` (see [Transforms](docs/transforms.md)).
- Add `Linkify(...)` transform for wrapping detected URLs/emails in `<a>` tags (see [Linkify](docs/linkify.md)).
- Add `Sanitize(...)` transform to sanitize the **in-memory DOM tree** (must be last) (see [HTML Cleaning](docs/html-cleaning.md) and [Transforms](docs/transforms.md)). Note that this does **not** replace the sanitization happening on serilization; that's still there.

### Changed
- BREAKING: Remove the public `sanitize(...)` function. If you need a sanitized DOM tree, use `Sanitize(...)` as the last transform; safe-by-default output sanitization remains available via `safe=True` serialization (see [HTML Cleaning](docs/html-cleaning.md)).
- Improve playground layout responsiveness and parse error display (see [Playground](https://emilstenstrom.github.io/justhtml/playground/)).

### Docs
- Add a migration guide for users coming from Bleach (see [Migrating from Bleach](docs/bleach-migration.md)).

## [0.31.0] - 2026-01-09
### Changed
- Add more type hints across tokenizer and tree builder internals (thanks @collinanderson).

## [0.30.0] - 2026-01-03
### Changed
- BREAKING: Rename URL sanitization API (see [URL Cleaning](docs/url-cleaning.md)):
  - `UrlPolicy.rules` -> `UrlPolicy.allow_rules`
  - `UrlPolicy.url_handling` -> `UrlPolicy.default_handling`
  - `UrlPolicy.allow_relative` -> `UrlPolicy.default_allow_relative`
  - `UrlRule.url_handling` -> `UrlRule.handling`
- BREAKING: URL allow rules now behave like an allowlist: if an attribute matches `UrlPolicy.allow_rules` and the URL validates, it is kept by default. To strip or proxy a specific attribute, set `UrlRule.handling="strip"` / `"proxy"` (see [URL Cleaning](docs/url-cleaning.md)).
- BREAKING: Proxying is still supported, but is now configured per attribute rule (`UrlRule.handling="proxy"`) instead of via a policy-wide default. Proxy mode requires a proxy to be configured either globally (`UrlPolicy.proxy`) or per rule (`UrlRule.proxy`) (see [URL Cleaning](docs/url-cleaning.md)).
- BREAKING: `UrlPolicy.default_handling` now defaults to `"strip"` (see [URL Cleaning](docs/url-cleaning.md)).

## [0.29.0] - 2026-01-03
### Changed
- Default policy change: `DEFAULT_POLICY` now blocks remote image loads by default (`img[src]` only allows relative URLs). Use a custom policy to allow `http(s)` images if you want them (see [URL Cleaning](docs/url-cleaning.md)).

## [0.28.0] - 2026-01-03
### Changed
- BREAKING: URL sanitization is now explicitly controlled by `UrlPolicy`/`UrlRule`. URL-like attributes (for example `href`, `src`, `srcset`) are dropped by default unless you provide an explicit `(tag, attr)` rule in `UrlPolicy.rules` (see [URL Cleaning](docs/url-cleaning.md)).
- BREAKING: Replace legacy “remote URL handling” configuration with `UrlPolicy(url_handling="allow"|"strip"|"proxy", allow_relative=...)` (see [URL Cleaning](docs/url-cleaning.md)).

### Added
- Add `UrlProxy` and URL rewriting via `UrlPolicy(url_handling="proxy")`.
- Add `srcset` parsing + sanitization using the same URL policy rules.
- Split sanitization docs into an overview plus deeper guides for HTML cleaning, URL cleaning, and unsafe handling (see [Sanitization](docs/sanitization.md)).

## [0.27.0] - 2026-01-03
### Added
- Add `unsafe_handling` mode to `SanitizationPolicy`, including an option to raise on all security findings.

### Changed
- Enhance sanitization policy behavior and error collection to support reporting security findings.
- Improve ordering of collected security errors by input position.
- Improve Playground parse error UI, including sanitizer security findings.

### Security
- (Severity: Low) Set explicit GitHub Actions workflow token permissions (`contents: read`) to address a CodeQL code scanning alert.

## [0.26.0] - 2026-01-02
### Added
- Add security policy (`SECURITY.md`) and update documentation to reference it.
### Changed
- Optimize whitespace collapsing and enhance attribute unquoting logic in serialization.
- Enhance `clone_node` method to support attribute overriding in `Node`.
- Normalize `rel` tokens in `SanitizationPolicy` for performance improvement.

## [0.25.0] - 2026-01-02
### Added
- Improve serialization speed by 5%.
- Introduce `CSS_PRESET_TEXT` for conservative inline styling and enhance sanitization policy validation.
### Changed
- Add benchmark for `justhtml parse` and `serialize --to-html` flag.

## [0.24.0] - 2026-01-01
### Security
- (Severity: Low) Fix inline CSS sanitization bypass where comments (`/**/`) inside `url()` could evade blacklisting of constructs that made external network calls. No XSS was possible. This required `style` attributes and URL-accepting properties to be allowlisted. No known exploits in the wild.
### Added
- Add [JustHTML Playground](https://emilstenstrom.github.io/justhtml/playground/) with HTML and Markdown support.
- Add optional node location tracking and usage examples in documentation.
### Fixed
- Update Pyodide installation code to use latest `justhtml` package version.
- Update Playground link to use correct file extension in documentation.
- Remove redundant label from Playground link in documentation.
- Add a migration guide for users coming from Bleach (see [Migrating from Bleach](docs/bleach-migration.md)).
### Changed
- Enhance README with code examples, documentation links, and improved clarity in usage and comparison sections.

## [0.23.0] - 2025-12-30
### Added
- Add support for running specific test suites with `--suite` option.
### Changed
- Update compliance scores and add browser engine agreement section in README.md.

## [0.22.0] - 2025-12-28
### Added
- Add CLI sanitization option and corresponding tests.
- Add sanitization options to text extraction methods and update documentation.
- Enhance Markdown link destination handling for safety and formatting.
- Add interactive release helper for version bumping and GitHub releases.

## [0.21.0] - 2025-12-28
### Changed
- Refactor sanitization policy and enhance Markdown conversion tests.

## [0.20.0] - 2025-12-28
### Added
- Enhance HTML serialization by collapsing whitespace and normalizing formatting in text nodes.
- Update README to clarify HTML5 compliance and security features.
### Changed
- Add line breaks for improved readability in README sections.
- Streamline README sections for clarity and consistency.
- Enhance README with additional context on correctness, sanitization, and CSS selector API.
- Add test harness for `justhtml` with tokenizer, serializer, and tree validation.

## [0.19.0] - 2025-12-28
### Added
- Enhance fragment context handling and improve template content checks.
- Enhance documentation examples with output formatting and add tests for code snippets.
- Document built-in HTML sanitizer with default policies and fragment support.
- Enhance URL sanitization to drop empty or control-only values and add corresponding tests.
- Add inline style sanitization with allowlist and enhance test coverage.
- Add proxy URL handling in URL rules and enhance test cases.
- Enhance serialization with new attribute handling and tests.
- Add HTML sanitization policy API and integrate sanitization in `to_html` function.
### Changed
- Remove unused attribute quoting function and simplify test assertions.
- Refactor serialization and sanitization logic; enhance test coverage.

## [0.18.0] - 2025-12-21
### Added
- Enhance selector parsing and add tests for new functionality.
- Enhance error handling for numeric and noncharacter references in tokenizer and entities.
- Make `--check-errors` also test errors in tokenizer tests.
- Enhance error handling in test results and reporting.
### Changed
- Update compliance scores and details in README and correctness documentation.
- Update copyright information in license file.

## [0.17.0] - 2025-12-21
### Added
- Enhance error handling for control characters in tokenizer and treebuilder.
### Changed
- Add detailed explanation of error locations in documentation.
- Enhance error handling and parsing logic in `justhtml`.
- Add copyright notice for html5ever project.

## [0.16.0] - 2025-12-18
### Added
- Enhance output handling with file writing and separator options.
- Add `--output` option for specifying file to write to.
### Fixed
- Update test summary to reflect correct number of passed tests.
### Changed
- Update dataset usage documentation with URL reference.
- Update dataset path handling and improve documentation.

## [0.15.0] - 2025-12-18
### Added
- Enhance pretty printing by skipping whitespace text nodes and comments.
### Changed
- Optimize position handling in tag and attribute parsing.
- Improve tokenizer and treebuilder handling of null characters and whitespace.

## [0.14.0] - 2025-12-17
### Added
- Add `--fragment` option for parsing HTML fragments without wrappers.

## [0.13.1] - 2025-12-17
### Fixed
- Preserve `<pre>` content in `--format html`.

## [0.13.0] - 2025-12-16
### Added
- Add support for `:contains()` pseudo-class and related tests.
- Enable manual triggering of the publish workflow.

## [0.12.0] - 2025-12-15
### Added
- CLI: preserve non-UTF-8 input by reading stdin as bytes when available.
- CLI: read file inputs as bytes to avoid decode failures on non-UTF-8.
- Add tests for CLI stdin handling with non-UTF-8 bytes.

### Fixed
- CI: skip mypy check in pre-commit step.

### Changed
- Update test summary for CLI/CI changes.

## [0.11.0] - 2025-12-15
### Added
- Add mypy hook for type checking in pre-commit configuration.
### Changed
- Refactor code for improved clarity and consistency; add tests for `Node` behavior.
- Add additional usage examples to documentation for parsing and text extraction.
- Update documentation to reflect test suite changes and improve clarity.
- Add command line interface documentation and update index.

## [0.10.0] - 2025-12-14
### Changed
- Refactor file reading to use `pathlib` for improved readability and consistency across documentation and CLI.
- Enhance CLI functionality and add comprehensive tests for `justhtml`.

## [0.9.0] - 2025-12-14
### Changed
- Add text extraction methods and documentation for `justhtml`.

## [0.8.0] - 2025-12-13
### Changed
- Add encoding support and tests for `justhtml`.
- Add symlink for html5lib-tests serializer in CI setup and documentation.

## [0.7.0] - 2025-12-13
### Changed
- Update documentation and tests for serializer improvements and HTML5 compliance.
- Revise fuzz testing section title.
- Update fuzz testing statistic in README.
- Add design proposal for optional HTML sanitizer in `justhtml`.
- Update test case counts in documentation to reflect current compliance status.
- Add `FragmentContext` support to `justhtml` and update documentation.

## [0.6.0] - 2025-12-08
### Fixed
- Parse `<noscript>` content in `<head>` as HTML when scripting is disabled.

### Changed
- Adjust test runner to skip script-on tests and add new `<noscript>` fixtures.
- Update correctness docs and test summary to reflect the new results.

### Docs
- Add acknowledgments section to README.md, crediting html5ever as the foundation for JustHTML.

## [0.5.2] - 2025-12-07
### Changed
- Add comprehensive documentation for `justhtml`, including API reference, correctness testing, error codes, CSS selectors, and streaming API usage.
- Remove `watch_tests.sh` script; eliminate unused test monitoring functionality.
- Remove unused `CharacterTokens` class and related references; clean up constants in tokenizer and constants files.
- Refactor attribute terminators in tokenizer; remove redundant patterns and simplify regex definitions.
- Optimize line tracking in tokenizer by pre-computing newline positions; replace manual line counting with binary search for improved performance.

## [0.5.1] - 2025-12-07
### Changed
- Enhance line counting in tokenizer for whitespace and attribute values; add tests for error collection.

## [0.5.0] - 2025-12-07
### Changed
- Enhance error handling in parser and tokenizer; implement strict mode with detailed error reporting and source highlighting.
- Refactor error handling in treebuilder and related classes.
- Add error checking option to `TestReporter` configuration.
- Enhance error handling in tokenizer and treebuilder; track token positions for improved error reporting.
- Implement error collection and strict mode in `justhtml` parser; add tests for error handling.
- Add node manipulation methods and text property to `Node`.

## [0.4.0] - 2025-12-06
### Changed
- Implement streaming API for efficient HTML parsing and add corresponding tests.
- Format `_rawtext_switch_tags` for improved readability.
- Add treebuilder utilities and update test for HTML conversion.
- Add `CONTRIBUTING.md` to outline development setup and contribution guidelines.
- Add Contributor Covenant Code of Conduct.

## [0.3.0] - 2025-12-02
### Changed
- Add `query` and `to_html` methods to `JustHTML` class; enhance README examples.

## [0.2.0] - 2025-12-02
### Changed
- Fix typos and improve clarity in README.md.
- Refactor code structure for improved readability and maintainability.
- Update HTML5 compliance status for html5lib in parser comparison table.
- Fix HTML5 compliance scores in parser comparison table in README.md.
- Update parser comparison table in README.md with compliance scores and additional parsers.
- Remove empty test summary file.
- Add checks for html5lib-tests symlinks in test runner.
- Rearrange performance benchmark tests and add correctness tests.
- Improve Pyodide testing: refactor wheel installation and enhance test structure.
- Fix Python version in CI configuration for PyPy.
- Refactor Pyodide testing: update installation method and improve test structure.
- Remove conditional execution for pre-commit hook in CI configuration.
- Add Pyodide testing and add PyPy to testing matrix in CI configuration.
- Fix typos and improve clarity in README.md.
- Rename ruff hook to ruff-check for consistency in pre-commit configuration.
- Refactor serialization and testing: streamline test format conversion, update test coverage, and remove redundant test file.
- Update Python version requirements to 3.10 in CI, README, and pyproject.toml for compatibility.
- Add coverage to CI and pre-commit.
- Update CI Python version matrix to include 3.13, 3.14, and 3.15-dev for broader compatibility.
- Update Python version to >=3.9 requirements in CI and pyproject.toml for compatibility.
- Add "exe001" to ruff ignore list for improved linting flexibility.
- Specify ruff version in pyproject.toml for consistent dependency management.
- Remove unnecessary blank lines in `profile_real.py` and `run_tests.py` for improved readability.
- Refactor whitespace for consistency in benchmark and fuzz scripts; remove unnecessary blank lines in profile and test scripts.
- Fix ruff errors.
- Add CI workflow and pre-commit configuration for automated testing.
- Update installation instructions and add development dependencies.
- Add README.md for test setup and execution instructions.
- Update README.md for clarity and consistency in messaging.
