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
