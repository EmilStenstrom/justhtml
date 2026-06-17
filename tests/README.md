# Running the tests

JustHTML uses the official [html5lib-tests](https://github.com/html5lib/html5lib-tests) suite to ensure spec compliance. These tests are not included in the repository to keep it lightweight and to make sure we can easily update them to the latest version with a git pull.

## Setup

1.  Clone the `html5lib-tests` repository next to your `justhtml` directory:

    ```bash
    cd ..
    git clone https://github.com/html5lib/html5lib-tests.git
    cd justhtml
    ```

2.  Create symlinks in the `tests/` directory:

    ```bash
    cd tests
    ln -s ../../html5lib-tests/tree-construction html5lib-tests-tree
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
