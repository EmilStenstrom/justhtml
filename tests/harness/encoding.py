from __future__ import annotations

from pathlib import Path

from justhtml.encoding import normalize_encoding_label, sniff_html_encoding


def _parse_encoding_dat_file(path):
    data = path.read_bytes()
    tests = []
    mode = None
    current_data = []
    current_encoding = None

    def flush():
        nonlocal current_data, current_encoding
        if current_data is None or current_encoding is None:
            return
        tests.append((b"".join(current_data), current_encoding))
        current_data = []
        current_encoding = None

    for line in data.splitlines(keepends=True):
        stripped = line.rstrip(b"\r\n")
        if stripped == b"#data":
            flush()
            mode = "data"
            continue
        if stripped == b"#encoding":
            mode = "encoding"
            continue

        if mode == "data":
            current_data.append(line)
        elif mode == "encoding":
            if current_encoding is None and stripped:
                current_encoding = stripped.decode("ascii", "ignore")
        else:
            continue

    flush()
    return tests


def _run_encoding_tests(config):
    root = Path("tests")
    fixture_dir = root / "html5lib-tests-encoding"
    if not fixture_dir.exists():
        return 0, 0, 0, {}

    test_files = sorted([p for p in fixture_dir.rglob("*.dat") if p.is_file()])
    if not test_files:
        print("No encoding tests found.")
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
        rel_name = str(path.relative_to(root))

        if test_specs:
            should_run_file = False
            specific_indices = None
            for spec in test_specs:
                if ":" in spec:
                    spec_file, indices_str = spec.split(":", 1)
                    if spec_file in rel_name or spec_file in filename:
                        should_run_file = True
                        specific_indices = set(int(i) for i in indices_str.split(",") if i)
                        break
                else:
                    if spec in rel_name or spec in filename:
                        should_run_file = True
                        break
            if not should_run_file:
                continue
        else:
            specific_indices = None

        tests = _parse_encoding_dat_file(path)
        file_passed = 0
        file_failed = 0
        file_skipped = 0
        test_indices = []

        is_scripted = "scripted" in path.parts

        for idx, (data, expected_label) in enumerate(tests):
            if specific_indices is not None and idx not in specific_indices:
                continue

            total += 1

            expected = normalize_encoding_label(expected_label)
            if expected is None:
                skipped += 1
                file_skipped += 1
                test_indices.append(("skip", idx))
                continue

            if is_scripted:
                skipped += 1
                file_skipped += 1
                test_indices.append(("skip", idx))
                continue

            sniffed = sniff_html_encoding(data)
            actual = sniffed[0] if isinstance(sniffed, tuple) else sniffed

            if actual == expected:
                passed += 1
                file_passed += 1
                test_indices.append(("pass", idx))
            else:
                file_failed += 1
                test_indices.append(("fail", idx))
                if verbosity >= 1 and not quiet:
                    print(f"\nENCODING FAIL: {rel_name}:{idx}")
                    print(f"EXPECTED: {expected!r} (raw: {expected_label!r})")
                    print(f"ACTUAL:   {actual!r}")

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
