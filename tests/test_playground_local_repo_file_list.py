import re
import unittest
from pathlib import Path


class TestPlaygroundLocalRepoFileList(unittest.TestCase):
    def test_local_repo_file_list_includes_all_justhtml_modules(self) -> None:
        """Keep the Pyodide playground's local-repo loader in sync with src/justhtml.

        The playground copies the working tree sources into Pyodide's virtual FS.
        If a new module is added (or renamed) and the JS list isn't updated, the
        local playground mode breaks at import time.
        """

        app_js = Path("docs/playground/app.js").read_text(encoding="utf-8")
        m = re.search(r"const files = \x5b(?P<body>.*?)\x5d;", app_js, flags=re.DOTALL)
        self.assertIsNotNone(m, "Couldn't find `const files = [...]` in docs/playground/app.js")
        body = m.group("body")

        listed = re.findall(r'"(?P<name>[^"]+\.py)"', body)
        self.assertTrue(listed, "No .py entries found in the playground file list")

        src_files = sorted(p.name for p in Path("src/justhtml").glob("*.py"))

        self.assertEqual(len(listed), len(set(listed)), "Playground file list contains duplicates")
        self.assertEqual(set(listed), set(src_files), "Playground file list is out of sync with src/justhtml")
