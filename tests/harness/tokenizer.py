from __future__ import annotations

import json
import re
from pathlib import Path

from justhtml.tokenizer import Tokenizer, TokenizerOpts
from justhtml.tokens import CharacterTokens, CommentToken, Doctype, DoctypeToken, EOFToken, Tag
from justhtml.treebuilder import InsertionMode, TreeBuilder


class RecordingTreeBuilder(TreeBuilder):
    __slots__ = ("tokens",)

    def __init__(self):
        super().__init__()
        self.tokens = []

    def process_token(self, token):
        if isinstance(token, Tag):
            token_copy = Tag(token.kind, token.name, token.attrs, token.self_closing)
            self.tokens.append(token_copy)
        else:
            if isinstance(token, CharacterTokens):
                self.tokens.append(CharacterTokens(token.data))
            elif isinstance(token, CommentToken):
                self.tokens.append(CommentToken(token.data))
            elif isinstance(token, DoctypeToken):
                d = token.doctype
                self.tokens.append(DoctypeToken(Doctype(d.name, d.public_id, d.system_id, d.force_quirks)))
            elif isinstance(token, EOFToken):
                self.tokens.append(EOFToken())
            else:
                self.tokens.append(token)

        return super().process_token(token)

    def process_characters(self, data):
        if self.mode == InsertionMode.IN_BODY:
            self.tokens.append(CharacterTokens(data))
        return super().process_characters(data)


def _unescape_unicode(text: str) -> str:
    return re.sub(r"\\u([0-9A-Fa-f]{4})", lambda m: chr(int(m.group(1), 16)), text)


def _map_initial_state(name):
    mapping = {
        "Data state": (Tokenizer.DATA, None),
        "PLAINTEXT state": (Tokenizer.PLAINTEXT, None),
        "RCDATA state": (Tokenizer.RCDATA, None),
        "RAWTEXT state": (Tokenizer.RAWTEXT, None),
        "Script data state": (Tokenizer.RAWTEXT, "script"),
        "CDATA section state": (Tokenizer.CDATA_SECTION, None),
    }
    return mapping.get(name)


def _token_to_list(token):
    if isinstance(token, DoctypeToken):
        d = token.doctype
        return ["DOCTYPE", d.name, d.public_id, d.system_id, not d.force_quirks]
    if isinstance(token, CommentToken):
        return ["Comment", token.data]
    if isinstance(token, CharacterTokens):
        return ["Character", token.data]
    if isinstance(token, Tag):
        if token.kind == Tag.START:
            attrs = token.attrs or {}
            arr = ["StartTag", token.name, attrs]
            if token.self_closing:
                arr.append(True)
            return arr
        return ["EndTag", token.name]
    if isinstance(token, EOFToken):
        return None
    return ["Unknown"]


def _collapse_characters(tokens):
    collapsed = []
    for t in tokens:
        if t and t[0] == "Character" and collapsed and collapsed[-1][0] == "Character":
            collapsed[-1][1] += t[1]
        else:
            collapsed.append(t)
    return collapsed


def _run_tokenizer_tests(config):
    root = Path("tests")
    test_files = [p for p in root.glob("*/*.test") if p.parent.name != "html5lib-tests-serializer"]

    if not test_files:
        print("No tokenizer tests found.")
        return 0, 0, {}

    total = 0
    passed = 0
    file_results = {}
    verbosity = config.get("verbosity", 0)
    quiet = config.get("quiet", False)
    test_specs = config.get("test_specs", [])
    fail_fast = bool(config.get("fail_fast"))

    for path in sorted(test_files, key=lambda p: p.name):
        filename = path.name
        rel_path = str(path.relative_to(Path("tests")))

        should_run_file = False
        specific_indices = None

        if test_specs:
            for spec in test_specs:
                if ":" in spec:
                    spec_file, indices_str = spec.split(":", 1)
                    if spec_file in rel_path or spec_file in filename:
                        should_run_file = True
                        specific_indices = set(int(i) for i in indices_str.split(","))
                        break
                else:
                    if spec in rel_path or spec in filename:
                        should_run_file = True
                        break

            if not should_run_file:
                continue
        else:
            should_run_file = True

        data = json.loads(path.read_text())
        key = "tests" if "tests" in data else "xmlViolationTests"
        is_xml_violation = key == "xmlViolationTests"
        tests = data.get(key, [])
        file_passed = 0
        file_failed = 0
        test_indices = []

        for idx, test in enumerate(tests):
            if specific_indices is not None and idx not in specific_indices:
                continue

            total += 1
            ok = _run_single_tokenizer_test(
                test, xml_coercion=is_xml_violation, check_errors=config.get("check_errors")
            )
            status = "pass" if ok else "fail"
            test_indices.append((status, idx))
            if ok:
                passed += 1
                file_passed += 1
            else:
                file_failed += 1
                if verbosity >= 1 and not quiet:
                    _print_tokenizer_failure(
                        test,
                        path.name,
                        idx,
                        xml_coercion=is_xml_violation,
                        check_errors=config.get("check_errors"),
                    )

                if fail_fast:
                    rel_name = str(path.relative_to(Path("tests")))
                    file_results[rel_name] = {
                        "passed": file_passed,
                        "failed": file_failed,
                        "skipped": 0,
                        "total": file_passed + file_failed,
                        "test_indices": test_indices,
                    }
                    return passed, total, file_results
        rel_name = str(path.relative_to(Path("tests")))
        file_results[rel_name] = {
            "passed": file_passed,
            "failed": file_failed,
            "skipped": 0,
            "total": file_passed + file_failed,
            "test_indices": test_indices,
        }
    return passed, total, file_results


def _print_tokenizer_failure(test, filename, test_index, xml_coercion=False, check_errors=False):
    input_text = test["input"]
    expected_tokens = test["output"]
    expected_errors = test.get("errors") or []

    if test.get("doubleEscaped"):
        input_text = _unescape_unicode(input_text)

        def recurse(val):
            if isinstance(val, str):
                return _unescape_unicode(val)
            if isinstance(val, list):
                return [recurse(v) for v in val]
            if isinstance(val, dict):
                return {k: recurse(v) for k, v in val.items()}
            return val

        expected_tokens = recurse(expected_tokens)

    initial_states = test.get("initialStates") or ["Data state"]
    last_start_tag = test.get("lastStartTag")

    state_results = []

    for state_name in initial_states:
        mapped = _map_initial_state(state_name)
        if not mapped:
            print(f"\n!!! State {state_name} not mapped !!!")
            continue
        initial_state, raw_tag = mapped
        if last_start_tag:
            raw_tag = last_start_tag
        sink = RecordingTreeBuilder()
        discard_bom = test.get("discardBom", False)
        opts = TokenizerOpts(
            initial_state=initial_state,
            initial_rawtext_tag=raw_tag,
            discard_bom=discard_bom,
            xml_coercion=xml_coercion,
        )
        collect_errors = bool(check_errors) or bool(test.get("collectErrors"))
        tok = Tokenizer(sink, opts, collect_errors=collect_errors)
        tok.last_start_tag_name = last_start_tag
        tok.run(input_text)
        actual = [r for t in sink.tokens if (r := _token_to_list(t)) is not None]
        actual = _collapse_characters(actual)

        actual_errors = []
        if check_errors:
            actual_errors = [{"code": e.code, "line": e.line, "col": e.column} for e in getattr(tok, "errors", [])]

        token_mismatch = actual != expected_tokens
        error_mismatch = False
        if check_errors:
            expected_err_codes = [e.get("code") for e in expected_errors]
            actual_err_codes = [e.get("code") for e in actual_errors]
            if test.get("ignoreErrorOrder"):
                expected_err_codes = sorted(expected_err_codes)
                actual_err_codes = sorted(actual_err_codes)
            error_mismatch = actual_err_codes != expected_err_codes

        state_results.append(
            {
                "state": state_name,
                "actual_tokens": actual,
                "actual_errors": actual_errors,
                "token_mismatch": token_mismatch,
                "error_mismatch": error_mismatch,
            }
        )

    show_tokens = any(r["token_mismatch"] for r in state_results)
    show_errors = check_errors and any(r["error_mismatch"] for r in state_results)

    print(f"\nFAILED: {filename} test #{test_index}")
    print(f"Description: {test.get('description', 'N/A')}")
    print(f"Input: {input_text!r}")
    print(f"Initial states: {initial_states}")
    if last_start_tag:
        print(f"Last start tag: {last_start_tag}")

    if show_tokens:
        print("\n=== EXPECTED TOKENS ===")
        for tok in expected_tokens:
            print(f"  {tok}")

    if show_errors:
        print("\n=== EXPECTED ERRORS ===")
        if expected_errors:
            for e in expected_errors:
                print(f"  ({e.get('line')},{e.get('col')}): {e.get('code')}")
        else:
            print("  (none)")

    for r in state_results:
        state_name = r["state"]
        actual = r["actual_tokens"]
        actual_errors = r["actual_errors"]

        if show_tokens:
            print(f"\n=== ACTUAL TOKENS (state: {state_name}) ===")
            for t in actual:
                print(f"  {t}")

        if show_errors:
            print(f"\n=== ACTUAL ERRORS (state: {state_name}) ===")
            if actual_errors:
                for e in actual_errors:
                    print(f"  ({e.get('line')},{e.get('col')}): {e.get('code')}")
            else:
                print("  (none)")

        if show_tokens and actual != expected_tokens:
            print("\n=== DIFFERENCES ===")
            max_len = max(len(expected_tokens), len(actual))
            for i in range(max_len):
                exp = expected_tokens[i] if i < len(expected_tokens) else "<missing>"
                act = actual[i] if i < len(actual) else "<missing>"
                if exp != act:
                    print(f"  Token {i}: expected {exp}, got {act}")


def _run_single_tokenizer_test(test, xml_coercion=False, check_errors=False):
    input_text = test["input"]
    expected_tokens = test["output"]
    expected_errors = test.get("errors") or []
    ignore_error_order = bool(test.get("ignoreErrorOrder"))
    if test.get("doubleEscaped"):
        input_text = _unescape_unicode(input_text)

        def recurse(val):
            if isinstance(val, str):
                return _unescape_unicode(val)
            if isinstance(val, list):
                return [recurse(v) for v in val]
            if isinstance(val, dict):
                return {k: recurse(v) for k, v in val.items()}
            return val

        expected_tokens = recurse(expected_tokens)

    initial_states = test.get("initialStates") or ["Data state"]
    last_start_tag = test.get("lastStartTag")

    for state_name in initial_states:
        mapped = _map_initial_state(state_name)
        if not mapped:
            return False
        initial_state, raw_tag = mapped
        if last_start_tag:
            raw_tag = last_start_tag
        sink = RecordingTreeBuilder()
        discard_bom = test.get("discardBom", False)
        opts = TokenizerOpts(
            initial_state=initial_state,
            initial_rawtext_tag=raw_tag,
            discard_bom=discard_bom,
            xml_coercion=xml_coercion,
        )
        collect_errors = bool(check_errors) or bool(test.get("collectErrors"))
        tok = Tokenizer(sink, opts, collect_errors=collect_errors)
        tok.last_start_tag_name = last_start_tag
        tok.run(input_text)
        actual = [r for t in sink.tokens if (r := _token_to_list(t)) is not None]
        actual = _collapse_characters(actual)
        if actual != expected_tokens:
            return False

        if check_errors:
            actual_codes = [e.code for e in getattr(tok, "errors", [])]
            expected_codes = [e.get("code") for e in expected_errors]

            if ignore_error_order:
                actual_codes = sorted(actual_codes)
                expected_codes = sorted(expected_codes)

            if actual_codes != expected_codes:
                return False
    return True
