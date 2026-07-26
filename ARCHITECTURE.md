# Architecture

JustHTML is a pure-Python HTML5 parser. The main pipeline is:

```text
str/bytes/DOM input
  -> encoding detection (bytes only)
  -> plan-driven scan + HTML5 tree construction
  -> compiled transforms/sanitization
  -> DOM
  -> HTML, text, or Markdown serialization
```

`JustHTML` in `src/justhtml/parser/__init__.py` is the public orchestrator.
Parsing sanitizes by default. The common default-policy path projects allowed
content while parsing; other policies and transform lists parse a raw tree and
then run the compiled transform pipeline. An explicit `Sanitize` transform
defines the sanitization point; transforms after it are not sanitized again.

## Execution paths

- Document parsing produces a `Document`; fragment parsing produces a
  `DocumentFragment` and uses `FragmentContext` to initialize parser state.
- String input enters the parser directly. Byte input first passes through
  `parser/encoding.py`. DOM input is cloned/serialized before being reparsed.
- `ParseEngine` owns scanning, insertion modes, open-element state, active
  formatting state, error collection, and DOM construction. `EnginePlan`
  supplies the behavior that differs between raw and fused safe parsing.
- Constructor transforms are normalized, compiled once, and applied in order
  after raw parsing. `Stage` makes multiple tree walks explicit.
- `stream()` is a lower-level event API with no DOM or full tree-construction
  recovery. Do not use it as the implementation of `JustHTML`.
- `Node.to_html()` and `Node.to_markdown()` delegate to the serializer package.
  Plain-text extraction lives with DOM traversal in `dom/`.

## Package map

- `parser/__init__.py`: `JustHTML`, constructor orchestration, strict mode, and
  integration of decoding, parsing, transforms, sanitization, and errors.
- `parser/engine.py`: Single-pass scanner and HTML5 tree builder. `EnginePlan`
  selects raw or fused safe behavior.
- `parser/scanner.py`: Shared low-level tag, attribute, and rawtext scanning.
- `parser/encoding.py`: HTML byte-encoding sniffing and decoding.
- `parser/context.py`: Fragment context model.
- `parser/stream.py`: Event-stream scanner. It does not build a DOM and is
  separate from the tree-construction path.
- `dom/`: DOM node types, traversal/query conveniences, cloning, and tree
  mutation. `dom/builder.py` contains programmatic construction helpers.
- `sanitizer/policy.py` and `policy_defaults.py`: Policy model, compilation,
  and built-in policies. `sanitizer/dom.py` is the standalone DOM entry point;
  `css.py`, `rawtext.py`, and `url/` own their respective value domains.
- `transforms/spec.py`: Public declarative transform types.
  `transforms/compile.py` lowers them to internal operations, and
  `transforms/runtime.py` applies those operations to the DOM.
- `selector/`: Bounded CSS selector parsing and matching, shared by DOM queries
  and selector-targeted transforms.
- `serializer/html.py` and `markdown.py`: HTML and Markdown output.
  Serialization is iterative where tree depth matters.
- `core/`: Shared HTML constants, entities, doctypes, rawtext rules, errors,
  and lightweight types.
- `__main__.py`: CLI adapter over the public API.

`src/justhtml/__init__.py` defines the supported public surface. Treat other
imports as internal unless documentation says otherwise.

## Architectural goals

- Keep one authoritative HTML5 parsing path. Specialized APIs may expose
  different outputs, but must not grow competing parser implementations.
- Make equivalent behavior converge on shared machinery: sanitization compiles
  to transforms, DOM queries and targeted transforms share selectors, and all
  HTML output uses the serializer.
- Keep orchestration at the package boundaries and domain logic in its owning
  package. The parser builds trees; transforms mutate them; serializers render
  them.
- Keep the public API small and stable while allowing internal representations
  and compiled plans to evolve.
- Represent policy and transform configuration declaratively, then compile it
  before walking input or trees.

## Cross-package contracts

- Raw and fused-safe `EnginePlan` variants share tree-construction semantics.
  Plans may change projection and fast paths, not HTML5 recovery behavior.
- Parser stacks and DOM child lists are canonical state. Counts, position
  indexes, cached classifications, and other accelerators are derived state and
  must be updated with every append, removal, truncation, insertion,
  replacement, and reordering.
- `parser/engine.py` and `parser/stream.py` share lexical helpers from
  `parser/scanner.py`, but have different structural guarantees. Changes to
  names, attributes, comments, doctypes, entities, rawtext, script data, or
  foreign-content boundaries must account for both consumers.
- DOM mutations go through the node mutation API. It owns parent links, cycle
  prevention, node adoption, `DocumentFragment` splicing, and template-content
  ownership.
- Serialization decisions use element name, namespace, and content model.
  Attribute shape alone is not enough to identify boolean attributes, and
  text escaping differs between normal, escapable-rawtext, and rawtext
  elements.
- HTML syntax uses ASCII-defined whitespace and case folding. Do not substitute
  Unicode-wide operations where source positions or HTML token boundaries must
  remain stable.

## Change routing

- HTML syntax, recovery, namespaces, fragment behavior, or parse errors:
  `parser/engine.py`, using shared primitives from `parser/scanner.py` and
  `core/`.
- Input bytes or encoding labels: `parser/encoding.py`.
- Node shape, traversal, cloning, ownership, or mutation: `dom/`.
- Allowed markup, unsafe handling, CSS, URLs, or rawtext hardening:
  `sanitizer/`; expose the behavior through policy compilation rather than a
  parallel cleanup pass.
- A new DOM rewrite: define its public form in `transforms/spec.py`, compile it
  in `transforms/compile.py`, and execute it in `transforms/runtime.py`.
- Selector syntax or matching: `selector/core.py`; both queries and transforms
  must consume the same matcher.
- Output escaping or formatting: `serializer/`; keep output-context decisions
  out of the parser and DOM.
- Public exports: package facade plus `src/justhtml/__init__.py`. CLI behavior
  belongs in `__main__.py`.

Changes that cross these boundaries should preserve the direction of
dependencies: the public parser orchestrates; parser, sanitizer, transforms,
selectors, and serializers operate on the shared DOM; `core/` does not depend
on higher-level packages.
