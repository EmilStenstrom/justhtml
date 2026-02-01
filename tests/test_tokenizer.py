import unittest

from justhtml import JustHTML
from justhtml.tokenizer import Tokenizer, TokenizerOpts
from justhtml.tokens import Tag


class _TokenizerSink:
    __slots__ = ("open_elements",)

    def __init__(self) -> None:
        self.open_elements = []

    def process_token(self, token):
        return 0

    def process_characters(self, data):
        return 0


class TestTokenizerStates(unittest.TestCase):
    def test_after_attribute_name_lowercases_uppercase(self) -> None:
        sink = _TokenizerSink()
        tokenizer = Tokenizer(sink, TokenizerOpts(), collect_errors=True)
        tokenizer.initialize("A")
        tokenizer.state = Tokenizer.AFTER_ATTRIBUTE_NAME
        tokenizer.current_tag_attrs = {}
        tokenizer.current_attr_name[:] = ["x"]
        tokenizer.current_attr_value.clear()
        tokenizer.current_attr_value_has_amp = False

        tokenizer._state_after_attribute_name()
        self.assertEqual(tokenizer.current_attr_name, ["a"])

    def test_after_attribute_name_handles_null(self) -> None:
        sink = _TokenizerSink()
        tokenizer = Tokenizer(sink, TokenizerOpts(), collect_errors=True)
        tokenizer.initialize("\x00")
        tokenizer.state = Tokenizer.AFTER_ATTRIBUTE_NAME
        tokenizer.current_tag_attrs = {}
        tokenizer.current_attr_name[:] = ["x"]
        tokenizer.current_attr_value.clear()
        tokenizer.current_attr_value_has_amp = False

        tokenizer._state_after_attribute_name()
        self.assertEqual(tokenizer.current_attr_name, ["\ufffd"])

    def test_attribute_name_state_handles_null(self) -> None:
        sink = _TokenizerSink()
        tokenizer = Tokenizer(sink, TokenizerOpts(), collect_errors=True)
        tokenizer.initialize("\x00")
        tokenizer.state = Tokenizer.ATTRIBUTE_NAME
        tokenizer.current_tag_attrs = {}

        tokenizer._state_attribute_name()
        self.assertEqual(tokenizer.current_attr_name, ["\ufffd"])

    def test_attribute_name_state_appends_non_ascii(self) -> None:
        sink = _TokenizerSink()
        tokenizer = Tokenizer(sink, TokenizerOpts(), collect_errors=True)
        tokenizer.initialize("é")
        tokenizer.state = Tokenizer.ATTRIBUTE_NAME
        tokenizer.current_tag_attrs = {}

        tokenizer._state_attribute_name()
        self.assertEqual(tokenizer.current_attr_name, ["é"])

    def test_after_attribute_name_skips_whitespace_run(self) -> None:
        sink = _TokenizerSink()
        tokenizer = Tokenizer(sink, TokenizerOpts(), collect_errors=True)
        tokenizer.initialize("   =")
        tokenizer.state = Tokenizer.AFTER_ATTRIBUTE_NAME
        tokenizer.reconsume = False

        done = tokenizer._state_after_attribute_name()
        self.assertFalse(done)
        self.assertEqual(tokenizer.state, Tokenizer.BEFORE_ATTRIBUTE_VALUE)

    def test_after_attribute_name_no_whitespace_run(self) -> None:
        sink = _TokenizerSink()
        tokenizer = Tokenizer(sink, TokenizerOpts(), collect_errors=True)
        tokenizer.initialize("=")
        tokenizer.state = Tokenizer.AFTER_ATTRIBUTE_NAME
        tokenizer.reconsume = False

        done = tokenizer._state_after_attribute_name()
        self.assertFalse(done)
        self.assertEqual(tokenizer.state, Tokenizer.BEFORE_ATTRIBUTE_VALUE)

    def test_after_attribute_name_whitespace_continue(self) -> None:
        sink = _TokenizerSink()
        tokenizer = Tokenizer(sink, TokenizerOpts(), collect_errors=True)
        tokenizer.initialize(" =")
        tokenizer.state = Tokenizer.AFTER_ATTRIBUTE_NAME
        tokenizer.reconsume = True

        done = tokenizer._state_after_attribute_name()
        self.assertFalse(done)
        self.assertEqual(tokenizer.state, Tokenizer.BEFORE_ATTRIBUTE_VALUE)

    def test_data_end_tag_name_run_falls_back_on_null_terminator(self) -> None:
        sink = _TokenizerSink()
        tokenizer = Tokenizer(sink, TokenizerOpts(), collect_errors=True)
        tokenizer.run("</div\0>")

    def test_data_end_tag_name_run_falls_back_at_eof(self) -> None:
        sink = _TokenizerSink()
        tokenizer = Tokenizer(sink, TokenizerOpts(), collect_errors=True)
        tokenizer.run("</div")

    def test_tag_name_fast_path_emits_on_gt(self) -> None:
        sink = _TokenizerSink()
        tokenizer = Tokenizer(sink, TokenizerOpts(), collect_errors=False)
        tokenizer.initialize("abc>")
        tokenizer.state = Tokenizer.TAG_NAME
        tokenizer.current_tag_kind = Tag.START
        tokenizer.current_tag_name.clear()
        tokenizer.current_tag_attrs = {}
        tokenizer.current_token_start_pos = 0

        done = tokenizer._state_tag_name()
        self.assertFalse(done)
        self.assertEqual(tokenizer.current_tag_name, ["abc"])

    def test_tag_name_fast_path_uppercase_chunk_is_lowered(self) -> None:
        sink = _TokenizerSink()
        tokenizer = Tokenizer(sink, TokenizerOpts(), collect_errors=False)
        tokenizer.initialize("ABC>")
        tokenizer.state = Tokenizer.TAG_NAME
        tokenizer.current_tag_kind = Tag.START
        tokenizer.current_tag_name.clear()
        tokenizer.current_tag_attrs = {}
        tokenizer.current_token_start_pos = 0

        done = tokenizer._state_tag_name()
        self.assertFalse(done)
        self.assertEqual(tokenizer.current_tag_name, ["abc"])

    def test_tag_name_fast_path_eof_after_name_run(self) -> None:
        sink = _TokenizerSink()
        tokenizer = Tokenizer(sink, TokenizerOpts(), collect_errors=True)
        tokenizer.initialize("abc")
        tokenizer.state = Tokenizer.TAG_NAME
        tokenizer.current_tag_kind = Tag.START
        tokenizer.current_tag_name.clear()
        tokenizer.current_tag_attrs = {}
        tokenizer.current_token_start_pos = 0

        done = tokenizer._state_tag_name()
        self.assertTrue(done)

    def test_tag_name_fast_path_null_terminator_after_name_run(self) -> None:
        sink = _TokenizerSink()
        tokenizer = Tokenizer(sink, TokenizerOpts(), collect_errors=True)
        tokenizer.initialize("abc\0>")
        tokenizer.state = Tokenizer.TAG_NAME
        tokenizer.current_tag_kind = Tag.START
        tokenizer.current_tag_name.clear()
        tokenizer.current_tag_attrs = {}
        tokenizer.current_token_start_pos = 0

        tokenizer._state_tag_name()

    def test_tag_name_emits_rawtext_switch_does_not_reset_to_data(self) -> None:
        sink = _TokenizerSink()
        tokenizer = Tokenizer(sink, TokenizerOpts(), collect_errors=False)
        tokenizer.initialize("textarea>")
        tokenizer.state = Tokenizer.TAG_NAME
        tokenizer.current_tag_kind = Tag.START
        tokenizer.current_tag_name.clear()
        tokenizer.current_tag_attrs = {}
        tokenizer.current_token_start_pos = 0

        done = tokenizer._state_tag_name()
        self.assertFalse(done)
        self.assertEqual(tokenizer.state, Tokenizer.RCDATA)

    def test_tag_name_fast_path_self_closing(self) -> None:
        sink = _TokenizerSink()
        tokenizer = Tokenizer(sink, TokenizerOpts(), collect_errors=False)
        tokenizer.initialize("abc/>")
        tokenizer.state = Tokenizer.TAG_NAME
        tokenizer.current_tag_kind = Tag.START
        tokenizer.current_tag_name.clear()
        tokenizer.current_tag_attrs = {}
        tokenizer.current_token_start_pos = 0

        done = tokenizer._state_tag_name()
        self.assertFalse(done)

    def test_tag_name_fast_path_whitespace_enters_before_attribute_name(self) -> None:
        sink = _TokenizerSink()
        tokenizer = Tokenizer(sink, TokenizerOpts(), collect_errors=False)
        tokenizer.initialize("abc >")
        tokenizer.state = Tokenizer.TAG_NAME
        tokenizer.current_tag_kind = Tag.START
        tokenizer.current_tag_name.clear()
        tokenizer.current_tag_attrs = {}
        tokenizer.current_token_start_pos = 0

        done = tokenizer._state_tag_name()
        self.assertFalse(done)

    def test_tag_name_end_tag_bogus_markup_emits_raw_as_text(self) -> None:
        sink = _TokenizerSink()
        opts = TokenizerOpts(emit_bogus_markup_as_text=True)
        tokenizer = Tokenizer(sink, opts, collect_errors=False)
        tokenizer.initialize("abc >")
        tokenizer.state = Tokenizer.TAG_NAME
        tokenizer.current_tag_kind = Tag.END
        tokenizer.current_tag_name.clear()
        tokenizer.current_tag_attrs = {}
        tokenizer.current_token_start_pos = 0

        done = tokenizer._state_tag_name()
        self.assertFalse(done)
        self.assertEqual(tokenizer.pos, len("abc >"))

    def test_tag_name_slow_path_whitespace_and_solidus(self) -> None:
        sink = _TokenizerSink()

        tokenizer_ws = Tokenizer(sink, TokenizerOpts(), collect_errors=True)
        tokenizer_ws.initialize(" ")
        tokenizer_ws.state = Tokenizer.TAG_NAME
        tokenizer_ws.current_tag_kind = Tag.START
        tokenizer_ws.current_tag_attrs = {}
        tokenizer_ws.current_token_start_pos = 0
        done_ws = tokenizer_ws._state_tag_name()
        self.assertTrue(done_ws)

        tokenizer_slash = Tokenizer(sink, TokenizerOpts(), collect_errors=False)
        tokenizer_slash.initialize("/>")
        tokenizer_slash.state = Tokenizer.TAG_NAME
        tokenizer_slash.current_tag_kind = Tag.START
        tokenizer_slash.current_tag_attrs = {}
        tokenizer_slash.current_token_start_pos = 0
        done_slash = tokenizer_slash._state_tag_name()
        self.assertFalse(done_slash)

    def test_tag_name_end_tag_bogus_markup_solidus_branch(self) -> None:
        sink = _TokenizerSink()
        tokenizer = Tokenizer(sink, TokenizerOpts(emit_bogus_markup_as_text=True), collect_errors=False)
        tokenizer.initialize("abc/>")
        tokenizer.state = Tokenizer.TAG_NAME
        tokenizer.current_tag_kind = Tag.END
        tokenizer.current_tag_name.clear()
        tokenizer.current_tag_attrs = {}
        tokenizer.current_token_start_pos = 0

        done = tokenizer._state_tag_name()
        self.assertFalse(done)

    def test_tag_name_end_tag_bogus_markup_slow_path_ws_and_slash(self) -> None:
        sink = _TokenizerSink()

        tok_ws = Tokenizer(sink, TokenizerOpts(emit_bogus_markup_as_text=True), collect_errors=True)
        tok_ws.initialize(" >")
        tok_ws.state = Tokenizer.TAG_NAME
        tok_ws.current_tag_kind = Tag.END
        tok_ws.current_tag_attrs = {}
        tok_ws.current_token_start_pos = 0
        done_ws = tok_ws._state_tag_name()
        self.assertFalse(done_ws)

        tok_slash = Tokenizer(sink, TokenizerOpts(emit_bogus_markup_as_text=True), collect_errors=True)
        tok_slash.initialize("/>")
        tok_slash.state = Tokenizer.TAG_NAME
        tok_slash.current_tag_kind = Tag.END
        tok_slash.current_tag_attrs = {}
        tok_slash.current_token_start_pos = 0
        done_slash = tok_slash._state_tag_name()
        self.assertFalse(done_slash)

    def test_before_attribute_name_reconsume_eof_and_gt(self) -> None:
        sink = _TokenizerSink()

        tokenizer_eof = Tokenizer(sink, TokenizerOpts(), collect_errors=True)
        tokenizer_eof.initialize("")
        tokenizer_eof.state = Tokenizer.BEFORE_ATTRIBUTE_NAME
        tokenizer_eof.reconsume = True
        tokenizer_eof.current_char = None
        tokenizer_eof.current_tag_kind = Tag.START
        tokenizer_eof.current_tag_attrs = {}
        tokenizer_eof.current_token_start_pos = 0
        done_eof = tokenizer_eof._state_before_attribute_name()
        self.assertTrue(done_eof)

        tokenizer_gt = Tokenizer(sink, TokenizerOpts(), collect_errors=False)
        tokenizer_gt.initialize("")
        tokenizer_gt.state = Tokenizer.BEFORE_ATTRIBUTE_NAME
        tokenizer_gt.reconsume = True
        tokenizer_gt.current_char = ">"
        tokenizer_gt.current_tag_kind = Tag.START
        tokenizer_gt.current_tag_name[:] = ["div"]
        tokenizer_gt.current_tag_attrs = {}
        tokenizer_gt.current_token_start_pos = 0
        done_gt = tokenizer_gt._state_before_attribute_name()
        self.assertFalse(done_gt)

    def test_before_attribute_name_reconsume_gt_rawtext_switch(self) -> None:
        sink = _TokenizerSink()
        tokenizer = Tokenizer(sink, TokenizerOpts(), collect_errors=False)
        tokenizer.initialize("")
        tokenizer.state = Tokenizer.BEFORE_ATTRIBUTE_NAME
        tokenizer.reconsume = True
        tokenizer.current_char = ">"
        tokenizer.current_tag_kind = Tag.START
        tokenizer.current_tag_name[:] = ["textarea"]
        tokenizer.current_tag_attrs = {}
        tokenizer.current_token_start_pos = 0

        done = tokenizer._state_before_attribute_name()
        self.assertFalse(done)
        self.assertEqual(tokenizer.state, Tokenizer.RCDATA)

    def test_before_attribute_name_fast_path_value_fallback_and_eof(self) -> None:
        sink = _TokenizerSink()
        tokenizer = Tokenizer(sink, TokenizerOpts(), collect_errors=False)
        tokenizer.initialize("x=")
        tokenizer.state = Tokenizer.BEFORE_ATTRIBUTE_NAME
        tokenizer.current_tag_kind = Tag.START
        tokenizer.current_tag_name[:] = ["div"]
        tokenizer.current_tag_attrs = {}
        tokenizer.current_token_start_pos = 0

        done = tokenizer._state_before_attribute_name()
        self.assertTrue(done)

    def test_before_attribute_name_fast_path_quoted_value_eof_after_quote(self) -> None:
        sink = _TokenizerSink()
        tokenizer = Tokenizer(sink, TokenizerOpts(), collect_errors=False)
        tokenizer.initialize('x="y"')
        tokenizer.state = Tokenizer.BEFORE_ATTRIBUTE_NAME
        tokenizer.current_tag_kind = Tag.START
        tokenizer.current_tag_name[:] = ["div"]
        tokenizer.current_tag_attrs = {}
        tokenizer.current_token_start_pos = 0

        done = tokenizer._state_before_attribute_name()
        self.assertTrue(done)
        self.assertEqual(tokenizer.current_tag_attrs.get("x"), "y")

    def test_before_attribute_name_fast_path_unquoted_value_eof(self) -> None:
        sink = _TokenizerSink()
        tokenizer = Tokenizer(sink, TokenizerOpts(), collect_errors=False)
        tokenizer.initialize("x=y")
        tokenizer.state = Tokenizer.BEFORE_ATTRIBUTE_NAME
        tokenizer.current_tag_kind = Tag.START
        tokenizer.current_tag_name[:] = ["div"]
        tokenizer.current_tag_attrs = {}
        tokenizer.current_token_start_pos = 0

        done = tokenizer._state_before_attribute_name()
        self.assertTrue(done)
        self.assertEqual(tokenizer.current_tag_attrs.get("x"), "y")

    def test_before_attribute_name_fast_path_boolean_attr_terminators(self) -> None:
        sink = _TokenizerSink()

        tok_ws = Tokenizer(sink, TokenizerOpts(), collect_errors=False)
        tok_ws.initialize("x ")
        tok_ws.state = Tokenizer.BEFORE_ATTRIBUTE_NAME
        tok_ws.current_tag_kind = Tag.START
        tok_ws.current_tag_name[:] = ["div"]
        tok_ws.current_tag_attrs = {}
        tok_ws.current_token_start_pos = 0
        done_ws = tok_ws._state_before_attribute_name()
        self.assertFalse(done_ws)
        self.assertEqual(tok_ws.current_tag_attrs.get("x"), "")

        tok_gt = Tokenizer(sink, TokenizerOpts(), collect_errors=False)
        tok_gt.initialize("x>")
        tok_gt.state = Tokenizer.BEFORE_ATTRIBUTE_NAME
        tok_gt.current_tag_kind = Tag.START
        tok_gt.current_tag_name[:] = ["div"]
        tok_gt.current_tag_attrs = {}
        tok_gt.current_token_start_pos = 0
        done_gt = tok_gt._state_before_attribute_name()
        self.assertFalse(done_gt)

        tok_sc = Tokenizer(sink, TokenizerOpts(), collect_errors=False)
        tok_sc.initialize("x/>")
        tok_sc.state = Tokenizer.BEFORE_ATTRIBUTE_NAME
        tok_sc.current_tag_kind = Tag.START
        tok_sc.current_tag_name[:] = ["div"]
        tok_sc.current_tag_attrs = {}
        tok_sc.current_token_start_pos = 0
        done_sc = tok_sc._state_before_attribute_name()
        self.assertFalse(done_sc)

        dup_ws = Tokenizer(sink, TokenizerOpts(), collect_errors=False)
        dup_ws.initialize("x ")
        dup_ws.state = Tokenizer.BEFORE_ATTRIBUTE_NAME
        dup_ws.current_tag_kind = Tag.START
        dup_ws.current_tag_name[:] = ["div"]
        dup_ws.current_tag_attrs = {"x": "1"}
        dup_ws.current_token_start_pos = 0
        done_dup_ws = dup_ws._state_before_attribute_name()
        self.assertFalse(done_dup_ws)
        self.assertEqual(dup_ws.current_tag_attrs, {"x": "1"})

        dup_gt = Tokenizer(sink, TokenizerOpts(), collect_errors=False)
        dup_gt.initialize("x>")
        dup_gt.state = Tokenizer.BEFORE_ATTRIBUTE_NAME
        dup_gt.current_tag_kind = Tag.START
        dup_gt.current_tag_name[:] = ["div"]
        dup_gt.current_tag_attrs = {"x": "1"}
        dup_gt.current_token_start_pos = 0
        done_dup_gt = dup_gt._state_before_attribute_name()
        self.assertFalse(done_dup_gt)

        dup_sc = Tokenizer(sink, TokenizerOpts(), collect_errors=False)
        dup_sc.initialize("x/>")
        dup_sc.state = Tokenizer.BEFORE_ATTRIBUTE_NAME
        dup_sc.current_tag_kind = Tag.START
        dup_sc.current_tag_name[:] = ["div"]
        dup_sc.current_tag_attrs = {"x": "1"}
        dup_sc.current_token_start_pos = 0
        done_dup_sc = dup_sc._state_before_attribute_name()
        self.assertFalse(done_dup_sc)

    def test_attribute_name_fast_path_gt_slash_and_eof(self) -> None:
        sink = _TokenizerSink()

        tok_gt = Tokenizer(sink, TokenizerOpts(), collect_errors=False)
        tok_gt.initialize("x>")
        tok_gt.state = Tokenizer.ATTRIBUTE_NAME
        tok_gt.current_tag_kind = Tag.START
        tok_gt.current_tag_name[:] = ["div"]
        tok_gt.current_tag_attrs = {}
        tok_gt.current_token_start_pos = 0
        done_gt = tok_gt._state_attribute_name()
        self.assertFalse(done_gt)

        tok_slash = Tokenizer(sink, TokenizerOpts(), collect_errors=False)
        tok_slash.initialize("x/>")
        tok_slash.state = Tokenizer.ATTRIBUTE_NAME
        tok_slash.current_tag_kind = Tag.START
        tok_slash.current_tag_name[:] = ["div"]
        tok_slash.current_tag_attrs = {}
        tok_slash.current_token_start_pos = 0
        done_slash = tok_slash._state_attribute_name()
        self.assertFalse(done_slash)

        tok_eof = Tokenizer(sink, TokenizerOpts(), collect_errors=False)
        tok_eof.initialize("x")
        tok_eof.state = Tokenizer.ATTRIBUTE_NAME
        tok_eof.current_tag_kind = Tag.START
        tok_eof.current_tag_name[:] = ["div"]
        tok_eof.current_tag_attrs = {}
        tok_eof.current_token_start_pos = 0
        done_eof = tok_eof._state_attribute_name()
        self.assertTrue(done_eof)

    def test_attribute_name_fast_path_rawtext_switch(self) -> None:
        sink = _TokenizerSink()
        tok = Tokenizer(sink, TokenizerOpts(), collect_errors=False)
        tok.initialize("x>")
        tok.state = Tokenizer.ATTRIBUTE_NAME
        tok.current_tag_kind = Tag.START
        tok.current_tag_name[:] = ["textarea"]
        tok.current_tag_attrs = {}
        tok.current_token_start_pos = 0
        done = tok._state_attribute_name()
        self.assertFalse(done)
        self.assertEqual(tok.state, Tokenizer.RCDATA)

    def test_attribute_name_fast_path_unhandled_terminator_falls_back(self) -> None:
        sink = _TokenizerSink()
        tok = Tokenizer(sink, TokenizerOpts(), collect_errors=True)
        tok.initialize("x'")
        tok.state = Tokenizer.ATTRIBUTE_NAME
        tok.current_tag_kind = Tag.START
        tok.current_tag_name[:] = ["div"]
        tok.current_tag_attrs = {}
        tok.current_token_start_pos = 0
        tok._state_attribute_name()

    def test_before_attribute_value_skips_ws_then_gt(self) -> None:
        sink = _TokenizerSink()
        tokenizer = Tokenizer(sink, TokenizerOpts(), collect_errors=True)
        tokenizer.initialize(" >")
        tokenizer.state = Tokenizer.BEFORE_ATTRIBUTE_VALUE
        tokenizer.current_tag_kind = Tag.START
        tokenizer.current_tag_name[:] = ["div"]
        tokenizer.current_tag_attrs = {}
        tokenizer.current_attr_name[:] = ["x"]
        tokenizer.current_attr_value.clear()
        tokenizer.current_attr_value_has_amp = False
        tokenizer.current_token_start_pos = 0

        done = tokenizer._state_before_attribute_value()
        self.assertFalse(done)
        self.assertEqual(tokenizer.state, Tokenizer.DATA)

    def test_before_attribute_value_gt_rawtext_switch(self) -> None:
        sink = _TokenizerSink()
        tokenizer = Tokenizer(sink, TokenizerOpts(), collect_errors=True)
        tokenizer.initialize(">")
        tokenizer.state = Tokenizer.BEFORE_ATTRIBUTE_VALUE
        tokenizer.current_tag_kind = Tag.START
        tokenizer.current_tag_name[:] = ["textarea"]
        tokenizer.current_tag_attrs = {}
        tokenizer.current_attr_name[:] = ["x"]
        tokenizer.current_attr_value.clear()
        tokenizer.current_attr_value_has_amp = False
        tokenizer.current_token_start_pos = 0

        done = tokenizer._state_before_attribute_value()
        self.assertFalse(done)
        self.assertEqual(tokenizer.state, Tokenizer.RCDATA)

    def test_single_quoted_value_null_replacement(self) -> None:
        sink = _TokenizerSink()
        tokenizer = Tokenizer(sink, TokenizerOpts(), collect_errors=True)
        tokenizer.initialize("\0'")
        tokenizer.state = Tokenizer.ATTRIBUTE_VALUE_SINGLE
        tokenizer.current_tag_kind = Tag.START
        tokenizer.current_tag_name[:] = ["div"]
        tokenizer.current_tag_attrs = {}
        tokenizer.current_attr_name[:] = ["x"]
        tokenizer.current_attr_value.clear()
        tokenizer.current_attr_value_has_amp = False
        tokenizer.current_token_start_pos = 0

        tokenizer._state_attribute_value_single()
        self.assertEqual(tokenizer.current_attr_value, ["\ufffd"])

    def test_after_attribute_value_quoted_missing_whitespace_reconsumes(self) -> None:
        sink = _TokenizerSink()
        tokenizer = Tokenizer(sink, TokenizerOpts(), collect_errors=True)
        tokenizer.initialize("x")
        tokenizer.state = Tokenizer.AFTER_ATTRIBUTE_VALUE_QUOTED
        tokenizer.current_tag_kind = Tag.START
        tokenizer.current_tag_name[:] = ["div"]
        tokenizer.current_tag_attrs = {}
        tokenizer.current_attr_name[:] = ["a"]
        tokenizer.current_attr_value[:] = ["1"]
        tokenizer.current_attr_value_has_amp = False
        tokenizer.current_token_start_pos = 0

        done = tokenizer._state_after_attribute_value_quoted()
        self.assertFalse(done)
        self.assertTrue(tokenizer.reconsume)
        self.assertEqual(tokenizer.state, Tokenizer.BEFORE_ATTRIBUTE_NAME)

    def test_location_at_pos_lazy_newline_index(self) -> None:
        tokenizer = Tokenizer(None, None, collect_errors=False)
        tokenizer.initialize("a\nb\nc")
        self.assertIsNone(tokenizer._newline_positions)

        self.assertEqual(tokenizer.location_at_pos(0), (1, 1))
        self.assertIsNotNone(tokenizer._newline_positions)
        self.assertEqual(tokenizer.location_at_pos(2), (2, 1))


class TestTokenizerFastPaths(unittest.TestCase):
    def test_fast_path_tag_name_and_attrs(self) -> None:
        html = '<DIV class="a&amp;b" id=foo data-x=bar disabled></DIV>'
        doc = JustHTML(html, sanitize=False)
        div = doc.root.query("div")[0]
        assert div.attrs["class"] == "a&b"
        assert div.attrs["id"] == "foo"
        assert div.attrs["data-x"] == "bar"
        assert div.attrs["disabled"] == ""

    def test_fast_path_missing_whitespace_between_attrs(self) -> None:
        html = '<div a="1"b=2 c="3"d="4"></div>'
        doc = JustHTML(html, sanitize=False)
        div = doc.root.query("div")[0]
        assert div.attrs == {"a": "1", "b": "2", "c": "3", "d": "4"}

    def test_fast_path_duplicate_attribute(self) -> None:
        html = "<div id=1 id=2></div>"
        doc = JustHTML(html, sanitize=False)
        div = doc.root.query("div")[0]
        assert div.attrs["id"] == "1"
