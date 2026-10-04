# Running the tests

JustHTML uses the web platform html5 treebuilder tests to ensure parser compliance. These tests live in [web-platform-tests/wpt](https://github.com/web-platform-tests/wpt/tree/master/html/syntax/parsing/resources); serializer and encoding tests remain in [html5lib-tests](https://github.com/html5lib/html5lib-tests). These tests are not included in the repository to keep it lightweight and to make updates easy.

## Setup

1.  Clone the test repositories next to your `justhtml` directory:

    ```bash
    cd ..
    git clone --filter=blob:none --sparse https://github.com/web-platform-tests/wpt.git
    cd wpt
    git sparse-checkout set html/syntax/parsing/resources
    cd ..
    git clone https://github.com/html5lib/html5lib-tests.git
    cd justhtml
    ```

2.  Create symlinks in the `tests/` directory:

    ```bash
    cd tests
    ln -s ../../wpt/html/syntax/parsing/resources html5lib-tests-tree
    ln -s ../../html5lib-tests/serializer html5lib-tests-serializer
    ln -s ../../html5lib-tests/encoding html5lib-tests-encoding
    ```

## Running tests

Once the symlinks are set up, you can run the tests using:

```bash
python run_tests.py
```

To run only one suite:

```bash
python run_tests.py --suite tree
python run_tests.py --suite justhtml
python run_tests.py --suite serializer
python run_tests.py --suite encoding
python run_tests.py --suite unit
```

Documentation examples run in separate Python processes, with up to four
examples running concurrently. Each example has a five-second timeout, and
failures are reported in documentation order.

Complexity tests compare inputs of 500 and 1,000 elements using five timing
samples per size. Short operations are batched to keep samples measurable.

The pre-commit coverage hook runs all fixture suites and shards unit tests
across up to 10 Python processes. It combines fresh coverage data only
after every worker passes, then enforces 100% coverage. Run it directly with:

```bash
python tests/check_coverage.py
python tests/check_coverage.py --jobs 4
```

Successful runs save test durations in `.cache/coverage-timings.json` to balance
worker loads. Test results and coverage are recomputed on every run.
