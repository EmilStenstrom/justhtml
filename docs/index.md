# JustHTML Documentation

A pure Python HTML5 parser that just works.

<div id="jh-search" class="jh-search" hidden>
  <label class="jh-search__label" for="jh-search-input">Search</label>
  <input id="jh-search-input" class="jh-search__input" type="search" placeholder="e.g. sanitize, selector, fragment" autocomplete="off" spellcheck="false" />
  <div id="jh-search-status" class="jh-search__status" aria-live="polite"></div>
  <ul id="jh-search-results" class="jh-search__results"></ul>
</div>

<noscript>
  <p>
    Search requires JavaScript. You can still search the docs via GitHub:
    <a href="https://github.com/search?q=repo%3AEmilStenstrom%2Fjusthtml+path%3Adocs+">Search in docs/</a>
  </p>
</noscript>

<script src="assets/search.js"></script>

## Contents

- **[Quickstart](quickstart.md)** - Get up and running in 2 minutes
- **[API Reference](api.md)** - Complete public API documentation
- **[Command Line](cli.md)** - Use `justhtml` to extract HTML, text, or Markdown
- **[Extracting Text](text.md)** - `to_text()` and `to_markdown()`
- **[CSS Selectors](selectors.md)** - Query elements with familiar CSS syntax
- **[Transforms](transforms.md)** - Apply declarative DOM transforms after parsing
    - **[Linkify](linkify.md)** - Convert URLs/emails in text nodes into links
- **[Fragment Parsing](fragments.md)** - Parse HTML fragments in context
- **[Sanitization & Security](sanitization.md)** - Overview of safe-by-default sanitization and policy configuration
    - **[HTML Cleaning](html-cleaning.md)** - Tags/attributes allowlists and inline styles
    - **[URL Cleaning](url-cleaning.md)** - URL validation, URL handling, and `srcset`
    - **[Unsafe Handling](unsafe-handling.md)** - What happens when unsafe input is encountered (strip/collect/raise)
    - **[Migrating from Bleach](bleach-migration.md)** - Guide for replacing Bleach cleaner/filter pipelines
- **[Streaming](streaming.md)** - Memory-efficient parsing for large files
- **[Encoding & Byte Input](encoding.md)** - How byte streams are decoded (including `windows-1252` fallback)
- **[Error Codes](errors.md)** - Parse error codes and their meanings
- **[Correctness Testing](correctness.md)** - How we verify 100% HTML5 compliance
- **[Playground](playground)** - Run JustHTML in your browser
