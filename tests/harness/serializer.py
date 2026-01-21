from __future__ import annotations

import json
from pathlib import Path

from justhtml.constants import VOID_ELEMENTS
from justhtml.serialize import serialize_end_tag, serialize_start_tag


def _serializer_attr_list_to_dict(attrs):
    if isinstance(attrs, dict):
        return attrs
    if not attrs:
        return {}
    out = {}
    for a in attrs:
        name = a.get("name")
        value = a.get("value")
        out[name] = value
    return out


def _escape_text_for_serializer_tests(text):
    if not text:
        return ""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _strip_whitespace_for_serializer_tests(text):
    if not text:
        return ""
    out = []
    last_space = False
    for ch in text:
        mapped = " " if ch in {"\t", "\r", "\n", "\f"} else ch
        if mapped == " ":
            if last_space:
                continue
            last_space = True
            out.append(" ")
        else:
            last_space = False
            out.append(mapped)
    return "".join(out)


def _update_meta_content_type_charset(content, encoding):
    if content is None:
        return None
    if not encoding:
        return content
    s = str(content)
    lower = s.lower()
    idx = lower.find("charset=")
    if idx == -1:
        return s
    start = idx + len("charset=")
    end = start
    while end < len(s) and s[end] not in {";", " ", "\t", "\r", "\n", "\f"}:
        end += 1
    return s[:start] + str(encoding) + s[end:]


def _apply_inject_meta_charset(tokens, encoding):
    if not encoding:
        return []

    saw_head = False
    in_head = False
    content_tokens = []
    for t in tokens:
        kind = t[0]
        if not in_head:
            if kind == "StartTag" and t[2] == "head":
                saw_head = True
                in_head = True
            continue
        if kind == "EndTag" and t[2] == "head":
            break
        content_tokens.append(t)

    if not saw_head:
        content_tokens = list(tokens)

    processed = []
    found_charset = False
    for t in content_tokens:
        if t[0] == "EmptyTag" and t[1] == "meta":
            attrs = _serializer_attr_list_to_dict(t[2] if len(t) > 2 else {})
            if "charset" in attrs:
                attrs["charset"] = encoding
                found_charset = True
            elif str(attrs.get("http-equiv", "")).lower() == "content-type" and "content" in attrs:
                attrs["content"] = _update_meta_content_type_charset(attrs.get("content"), encoding)
                found_charset = True
            processed.append(["EmptyTag", "meta", attrs])
        else:
            processed.append(t)

    if not found_charset:
        processed.insert(0, ["EmptyTag", "meta", {"charset": encoding}])

    return processed


def _serializer_tok_name(tok):
    if tok is None:
        return None
    kind = tok[0]
    if kind == "StartTag":
        return tok[2]
    if kind == "EndTag":
        return tok[2]
    if kind == "EmptyTag":
        return tok[1]
    return None


def _serializer_tok_is_space_chars(tok):
    return tok is not None and tok[0] == "Characters" and tok[1].startswith(" ")


def _serializer_should_omit_start_tag(name, attrs, prev_tok, next_tok):
    if attrs:
        return False

    if name == "html":
        if next_tok is None:
            return True
        if next_tok[0] == "Comment" or _serializer_tok_is_space_chars(next_tok):
            return False
        if next_tok[0] == "Characters" and next_tok[1] == "":
            return False
        return True

    if name == "head":
        if next_tok is None:
            return True
        if next_tok[0] in {"Comment", "Characters"}:
            return False
        if next_tok[0] == "EndTag" and _serializer_tok_name(next_tok) == "head":
            return True
        if next_tok[0] in {"StartTag", "EmptyTag", "EndTag"}:
            return True
        return False

    if name == "body":
        if next_tok is None:
            return True
        if next_tok[0] == "Comment" or _serializer_tok_is_space_chars(next_tok):
            return False
        return True

    if name == "colgroup":
        if prev_tok is not None and prev_tok[0] == "StartTag" and _serializer_tok_name(prev_tok) == "table":
            if (
                next_tok is not None
                and next_tok[0] in {"StartTag", "EmptyTag"}
                and _serializer_tok_name(next_tok) == "col"
            ):
                return True
        return False

    if name == "tbody":
        if prev_tok is not None and prev_tok[0] == "StartTag" and _serializer_tok_name(prev_tok) == "table":
            if next_tok is not None and next_tok[0] == "StartTag" and _serializer_tok_name(next_tok) == "tr":
                return True
        return False

    return False


def _serializer_should_omit_end_tag(name, next_tok):
    if name in {"html", "head", "body", "colgroup"}:
        if next_tok is None:
            return True
        if next_tok[0] == "Comment" or _serializer_tok_is_space_chars(next_tok):
            return False
        if next_tok[0] in {"StartTag", "EmptyTag", "EndTag"}:
            return True
        if next_tok[0] == "Characters":
            return not next_tok[1].startswith(" ")
        return True

    if name == "li":
        if next_tok is None:
            return True
        if next_tok[0] == "StartTag" and _serializer_tok_name(next_tok) == "li":
            return True
        if next_tok[0] == "EndTag":
            return True
        return False

    if name == "dt":
        if next_tok is None:
            return False
        if next_tok[0] == "StartTag" and _serializer_tok_name(next_tok) in {"dt", "dd"}:
            return True
        return False

    if name == "dd":
        if next_tok is None:
            return True
        if next_tok[0] == "StartTag" and _serializer_tok_name(next_tok) in {"dd", "dt"}:
            return True
        if next_tok[0] == "EndTag":
            return True
        return False

    if name == "p":
        if next_tok is None:
            return True
        if next_tok[0] == "EndTag":
            return True
        if next_tok[0] in {"StartTag", "EmptyTag"}:
            next_name = _serializer_tok_name(next_tok)
            if next_name in {
                "address",
                "article",
                "aside",
                "blockquote",
                "datagrid",
                "dialog",
                "dir",
                "div",
                "dl",
                "fieldset",
                "footer",
                "form",
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
                "header",
                "hr",
                "menu",
                "nav",
                "ol",
                "p",
                "pre",
                "section",
                "table",
                "ul",
            }:
                return True
        return False

    if name == "optgroup":
        if next_tok is None:
            return True
        if next_tok[0] == "StartTag" and _serializer_tok_name(next_tok) == "optgroup":
            return True
        if next_tok[0] == "EndTag":
            return True
        return False

    if name == "option":
        if next_tok is None:
            return True
        if next_tok[0] == "StartTag" and _serializer_tok_name(next_tok) in {"option", "optgroup"}:
            return True
        if next_tok[0] == "EndTag":
            return True
        return False

    if name == "tbody":
        if next_tok is None:
            return True
        if next_tok[0] == "StartTag" and _serializer_tok_name(next_tok) in {"tbody", "tfoot"}:
            return True
        if next_tok[0] == "EndTag":
            return True
        return False

    if name == "tfoot":
        if next_tok is None:
            return True
        if next_tok[0] == "StartTag" and _serializer_tok_name(next_tok) == "tbody":
            return True
        if next_tok[0] == "EndTag":
            return True
        return False

    if name == "thead":
        if next_tok is not None and next_tok[0] == "StartTag" and _serializer_tok_name(next_tok) in {"tbody", "tfoot"}:
            return True
        return False

    if name == "tr":
        if next_tok is None:
            return True
        if next_tok[0] == "StartTag" and _serializer_tok_name(next_tok) == "tr":
            return True
        if next_tok[0] == "EndTag":
            return True
        return False

    if name in {"td", "th"}:
        if next_tok is None:
            return True
        if next_tok[0] == "StartTag" and _serializer_tok_name(next_tok) in {"td", "th"}:
            return True
        if next_tok[0] == "EndTag":
            return True
        return False

    return False


def _serialize_serializer_token_stream(tokens, options=None):
    parts = []
    rawtext = None
    options = options or {}

    if options.get("inject_meta_charset"):
        encoding = options.get("encoding")
        if not encoding:
            return ""
        tokens = _apply_inject_meta_charset(tokens, encoding)

    open_elements = []
    strip_ws = bool(options.get("strip_whitespace"))
    escape_rcdata = bool(options.get("escape_rcdata"))
    ws_preserve = {"pre", "textarea", "script", "style"}

    for i, t in enumerate(tokens):
        prev_tok = tokens[i - 1] if i else None
        next_tok = tokens[i + 1] if i + 1 < len(tokens) else None

        kind = t[0]
        if kind == "StartTag":
            name = t[2]
            attrs = _serializer_attr_list_to_dict(t[3] if len(t) > 3 else {})

            if attrs:
                attrs = {k: attrs[k] for k in sorted(attrs.keys())}

            open_elements.append(name)

            if _serializer_should_omit_start_tag(name, attrs, prev_tok, next_tok):
                continue

            parts.append(
                serialize_start_tag(
                    name,
                    attrs,
                    quote_attr_values=bool(options.get("quote_attr_values")),
                    minimize_boolean_attributes=options.get("minimize_boolean_attributes", True),
                    quote_char=options.get("quote_char"),
                    escape_lt_in_attrs=bool(options.get("escape_lt_in_attrs")),
                    use_trailing_solidus=bool(options.get("use_trailing_solidus")),
                    is_void=name in VOID_ELEMENTS,
                )
            )
            if name in {"script", "style"} and not escape_rcdata:
                rawtext = name
        elif kind == "EndTag":
            name = t[2]

            if open_elements:
                if open_elements[-1] == name:
                    open_elements.pop()
                else:
                    for j in range(len(open_elements) - 1, -1, -1):
                        if open_elements[j] == name:
                            del open_elements[j:]
                            break

            if _serializer_should_omit_end_tag(name, next_tok):
                continue

            parts.append(serialize_end_tag(name))
            if rawtext == name:
                rawtext = None
        elif kind == "EmptyTag":
            name = t[1]
            attrs = t[2] if len(t) > 2 else {}

            if attrs:
                attrs = {k: attrs[k] for k in sorted(attrs.keys())}

            parts.append(
                serialize_start_tag(
                    name,
                    attrs,
                    quote_attr_values=bool(options.get("quote_attr_values")),
                    minimize_boolean_attributes=options.get("minimize_boolean_attributes", True),
                    quote_char=options.get("quote_char"),
                    escape_lt_in_attrs=bool(options.get("escape_lt_in_attrs")),
                    use_trailing_solidus=bool(options.get("use_trailing_solidus")),
                    is_void=True,
                )
            )
        elif kind == "Characters":
            if rawtext is not None:
                parts.append(t[1])
            else:
                text = t[1]
                if strip_ws and not (set(open_elements) & ws_preserve):
                    text = _strip_whitespace_for_serializer_tests(text)
                parts.append(_escape_text_for_serializer_tests(text))
        elif kind == "Comment":
            parts.append(f"<!--{t[1]}-->")
        elif kind == "Doctype":
            name = t[1] if len(t) > 1 else ""
            public_id = t[2] if len(t) > 2 else None
            system_id = t[3] if len(t) > 3 else None

            if public_id is None and system_id is None:
                parts.append(f"<!DOCTYPE {name}>")
            else:
                has_public = public_id not in {None, ""}
                has_system = system_id not in {None, ""}
                if has_public:
                    if has_system:
                        parts.append(f'<!DOCTYPE {name} PUBLIC "{public_id}" "{system_id}">')
                    else:
                        parts.append(f'<!DOCTYPE {name} PUBLIC "{public_id}">')
                elif has_system:
                    parts.append(f'<!DOCTYPE {name} SYSTEM "{system_id}">')
                else:
                    parts.append(f"<!DOCTYPE {name}>")
        else:
            return None
    return "".join(parts)


def _run_serializer_tests(config):
    root = Path("tests")
    fixture_dir = root / "html5lib-tests-serializer"
    if not fixture_dir.exists():
        return 0, 0, 0, {}
    test_files = sorted(fixture_dir.glob("*.test"))
    if not test_files:
        print("No serializer tests found.")
        return 0, 0, 0, {}

    verbosity = config.get("verbosity", 0)
    quiet = config.get("quiet", False)
    test_specs = config.get("test_specs", [])
    fail_fast = bool(config.get("fail_fast"))

    total = 0
    passed = 0
    skipped = 0
    file_results = {}

    for path in test_files:
        filename = path.name
        rel_name = str(path.relative_to(Path("tests")))

        if test_specs:
            should_run_file = False
            specific_indices = None
            for spec in test_specs:
                if ":" in spec:
                    spec_file, indices_str = spec.split(":", 1)
                    if spec_file in rel_name or spec_file in filename:
                        should_run_file = True
                        specific_indices = set(int(i) for i in indices_str.split(","))
                        break
                else:
                    if spec in rel_name or spec in filename:
                        should_run_file = True
                        break
            if not should_run_file:
                continue
        else:
            specific_indices = None

        data = json.loads(path.read_text())
        tests = data.get("tests", [])
        file_passed = 0
        file_failed = 0
        file_skipped = 0
        test_indices = []

        supported_option_keys = {
            "encoding",
            "inject_meta_charset",
            "strip_whitespace",
            "quote_attr_values",
            "use_trailing_solidus",
            "minimize_boolean_attributes",
            "quote_char",
            "escape_lt_in_attrs",
            "escape_rcdata",
        }

        for idx, test in enumerate(tests):
            if specific_indices is not None and idx not in specific_indices:
                continue

            total += 1

            options = test.get("options") or {}
            if not isinstance(options, dict):
                skipped += 1
                file_skipped += 1
                test_indices.append(("skip", idx))
                continue

            if any(k not in supported_option_keys for k in options.keys()):
                skipped += 1
                file_skipped += 1
                test_indices.append(("skip", idx))
                continue

            actual = _serialize_serializer_token_stream(test.get("input", []), options)
            if actual is None:
                skipped += 1
                file_skipped += 1
                test_indices.append(("skip", idx))
                continue
            expected_list = test.get("expected", [])
            ok = actual in expected_list

            if ok:
                passed += 1
                file_passed += 1
                test_indices.append(("pass", idx))
            else:
                file_failed += 1
                test_indices.append(("fail", idx))
                if verbosity >= 1 and not quiet:
                    desc = test.get("description", "")
                    print(f"\nSERIALIZER FAIL: {filename}:{idx} {desc}")
                    print("EXPECTED one of:")
                    for e in expected_list:
                        print(repr(e))
                    print("ACTUAL:")
                    print(repr(actual))

                if fail_fast:
                    file_results[rel_name] = {
                        "passed": file_passed,
                        "failed": file_failed,
                        "skipped": file_skipped,
                        "total": file_passed + file_failed + file_skipped,
                        "test_indices": test_indices,
                    }
                    return passed, total, skipped, file_results

        file_results[rel_name] = {
            "passed": file_passed,
            "failed": file_failed,
            "skipped": file_skipped,
            "total": file_passed + file_failed + file_skipped,
            "test_indices": test_indices,
        }

    return passed, total, skipped, file_results
