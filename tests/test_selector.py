"""Comprehensive tests for CSS selector functionality."""

import unittest

from justhtml import JustHTML as _JustHTML
from justhtml import SelectorError, matches, query
from justhtml.selector import (
    ComplexSelector,
    CompoundSelector,
    SelectorList,
    SelectorMatcher,
    SelectorParser,
    SelectorTokenizer,
    SimpleSelector,
    Token,
    TokenType,
    _is_simple_tag_selector,
    _query_descendants,
    _query_descendants_tag,
    parse_selector,
)


def JustHTML(*args, **kwargs):  # noqa: N802
    if "safe" not in kwargs:
        kwargs["safe"] = False
    return _JustHTML(*args, **kwargs)


class SelectorTestCase(unittest.TestCase):
    """Base test case with common fixtures."""

    def get_simple_doc(self):
        """A simple HTML document for testing."""
        html = """
        <html>
        <head><title>Test</title></head>
        <body>
            <div id="main" class="container">
                <h1>Title</h1>
                <p class="intro first">First paragraph</p>
                <p class="content">Second paragraph</p>
                <ul>
                    <li>Item 1</li>
                    <li class="special">Item 2</li>
                    <li>Item 3</li>
                </ul>
            </div>
            <div id="sidebar" class="container secondary">
                <a href="http://example.com" data-id="123">Link</a>
            </div>
        </body>
        </html>
        """
        return JustHTML(html).root

    def get_nested_doc(self):
        """A document with deeply nested elements."""
        html = """
        <html><body>
            <div class="a">
                <div class="b">
                    <div class="c">
                        <span id="deep">Deep</span>
                    </div>
                </div>
            </div>
        </body></html>
        """
        return JustHTML(html).root

    def get_sibling_doc(self):
        """A document for testing sibling selectors."""
        html = """
        <html><body>
            <div>
                <h1>Heading</h1>
                <p class="first">First</p>
                <p class="second">Second</p>
                <p class="third">Third</p>
                <span>Not a p</span>
                <p class="fourth">Fourth</p>
            </div>
        </body></html>
        """
        return JustHTML(html).root

    def get_empty_and_root_doc(self):
        """Document for testing :empty and :root."""
        html = """
        <html><body>
            <div class="empty"></div>
            <div class="whitespace">   </div>
            <div class="text">content</div>
            <div class="nested"><span></span></div>
        </body></html>
        """
        return JustHTML(html).root


class TestTagSelector(SelectorTestCase):
    """Test tag name selectors."""

    def test_tag_selector(self):
        result = query(self.get_simple_doc(), "p")
        assert len(result) == 2
        assert all(n.name == "p" for n in result)

    def test_tag_selector_case_insensitive(self):
        result = query(self.get_simple_doc(), "P")
        assert len(result) == 2

    def test_tag_selector_div(self):
        result = query(self.get_simple_doc(), "div")
        assert len(result) == 2

    def test_tag_selector_no_match(self):
        result = query(self.get_simple_doc(), "article")
        assert len(result) == 0

    def test_tag_selector_head_elements(self):
        result = query(self.get_simple_doc(), "title")
        assert len(result) == 1
        assert result[0].name == "title"


class TestUniversalSelector(SelectorTestCase):
    """Test universal selector (*)."""

    def test_universal_selector(self):
        result = query(self.get_simple_doc(), "*")
        # Should match all elements (excluding text nodes)
        assert len(result) > 10

    def test_universal_in_compound(self):
        result = query(self.get_simple_doc(), "*.container")
        assert len(result) == 2
        assert all("container" in (n.attrs.get("class") or "") for n in result)


class TestIdSelector(SelectorTestCase):
    """Test ID selectors."""

    def test_id_selector(self):
        result = query(self.get_simple_doc(), "#main")
        assert len(result) == 1
        assert result[0].attrs["id"] == "main"

    def test_id_selector_no_match(self):
        result = query(self.get_simple_doc(), "#nonexistent")
        assert len(result) == 0

    def test_id_selector_case_sensitive(self):
        result = query(self.get_simple_doc(), "#MAIN")
        assert len(result) == 0  # IDs are case-sensitive

    def test_id_with_tag(self):
        result = query(self.get_simple_doc(), "div#main")
        assert len(result) == 1

    def test_id_with_wrong_tag(self):
        result = query(self.get_simple_doc(), "span#main")
        assert len(result) == 0


class TestClassSelector(SelectorTestCase):
    """Test class selectors."""

    def test_class_selector(self):
        result = query(self.get_simple_doc(), ".container")
        assert len(result) == 2

    def test_class_selector_single(self):
        result = query(self.get_simple_doc(), ".intro")
        assert len(result) == 1

    def test_class_selector_no_match(self):
        result = query(self.get_simple_doc(), ".nonexistent")
        assert len(result) == 0

    def test_class_selector_case_sensitive(self):
        result = query(self.get_simple_doc(), ".Container")
        assert len(result) == 0  # Class names are case-sensitive

    def test_multiple_classes(self):
        result = query(self.get_simple_doc(), ".intro.first")
        assert len(result) == 1

    def test_class_with_tag(self):
        result = query(self.get_simple_doc(), "p.intro")
        assert len(result) == 1

    def test_class_with_wrong_tag(self):
        result = query(self.get_simple_doc(), "div.intro")
        assert len(result) == 0


class TestAttributePresence(SelectorTestCase):
    """Test [attr] presence selector."""

    def test_attribute_presence(self):
        result = query(self.get_simple_doc(), "[href]")
        assert len(result) == 1
        assert result[0].name == "a"

    def test_attribute_presence_id(self):
        result = query(self.get_simple_doc(), "[id]")
        assert len(result) == 2

    def test_attribute_presence_data(self):
        result = query(self.get_simple_doc(), "[data-id]")
        assert len(result) == 1


class TestAttributeExact(SelectorTestCase):
    """Test [attr=value] exact match selector."""

    def test_attribute_exact(self):
        result = query(self.get_simple_doc(), '[id="main"]')
        assert len(result) == 1
        assert result[0].attrs["id"] == "main"

    def test_attribute_exact_no_match(self):
        result = query(self.get_simple_doc(), '[id="wrong"]')
        assert len(result) == 0

    def test_attribute_exact_unquoted(self):
        result = query(self.get_simple_doc(), "[id=main]")
        assert len(result) == 1

    def test_attribute_exact_single_quotes(self):
        result = query(self.get_simple_doc(), "[id='main']")
        assert len(result) == 1


class TestAttributeContainsWord(SelectorTestCase):
    """Test [attr~=value] space-separated word selector."""

    def test_attribute_contains_word(self):
        result = query(self.get_simple_doc(), '[class~="container"]')
        assert len(result) == 2

    def test_attribute_contains_word_single(self):
        result = query(self.get_simple_doc(), '[class~="secondary"]')
        assert len(result) == 1

    def test_attribute_contains_word_no_partial(self):
        result = query(self.get_simple_doc(), '[class~="contain"]')
        assert len(result) == 0  # Must be exact word


class TestAttributeHyphenPrefix(SelectorTestCase):
    """Test [attr|=value] hyphen-prefix selector."""

    def test_attribute_hyphen_exact(self):
        doc = JustHTML('<html><body><p lang="en">Text</p></body></html>').root
        result = query(doc, '[lang|="en"]')
        assert len(result) == 1

    def test_attribute_hyphen_prefix(self):
        doc = JustHTML('<html><body><p lang="en-US">Text</p></body></html>').root
        result = query(doc, '[lang|="en"]')
        assert len(result) == 1

    def test_attribute_hyphen_no_match(self):
        doc = JustHTML('<html><body><p lang="eng">Text</p></body></html>').root
        result = query(doc, '[lang|="en"]')
        assert len(result) == 0


class TestAttributeStartsWith(SelectorTestCase):
    """Test [attr^=value] starts-with selector."""

    def test_attribute_starts_with(self):
        result = query(self.get_simple_doc(), '[href^="http"]')
        assert len(result) == 1

    def test_attribute_starts_with_full(self):
        result = query(self.get_simple_doc(), '[href^="http://example"]')
        assert len(result) == 1

    def test_attribute_starts_with_no_match(self):
        result = query(self.get_simple_doc(), '[href^="https"]')
        assert len(result) == 0


class TestAttributeEndsWith(SelectorTestCase):
    """Test [attr$=value] ends-with selector."""

    def test_attribute_ends_with(self):
        result = query(self.get_simple_doc(), '[href$=".com"]')
        assert len(result) == 1

    def test_attribute_ends_with_no_match(self):
        result = query(self.get_simple_doc(), '[href$=".org"]')
        assert len(result) == 0


class TestAttributeContains(SelectorTestCase):
    """Test [attr*=value] contains selector."""

    def test_attribute_contains(self):
        result = query(self.get_simple_doc(), '[href*="example"]')
        assert len(result) == 1

    def test_attribute_contains_no_match(self):
        result = query(self.get_simple_doc(), '[href*="google"]')
        assert len(result) == 0


class TestDescendantCombinator(SelectorTestCase):
    """Test descendant combinator (space)."""

    def test_descendant(self):
        result = query(self.get_simple_doc(), "div p")
        assert len(result) == 2

    def test_descendant_deep(self):
        result = query(self.get_nested_doc(), "div span")
        assert len(result) == 1

    def test_descendant_multiple_levels(self):
        result = query(self.get_nested_doc(), ".a span")
        assert len(result) == 1
        assert result[0].attrs["id"] == "deep"

    def test_descendant_no_match(self):
        result = query(self.get_simple_doc(), "span div")
        assert len(result) == 0


class TestChildCombinator(SelectorTestCase):
    """Test child combinator (>)."""

    def test_child(self):
        result = query(self.get_simple_doc(), "div > h1")
        assert len(result) == 1

    def test_child_direct_only(self):
        result = query(self.get_nested_doc(), "body > span")
        assert len(result) == 0  # span is not direct child of body

    def test_child_with_class(self):
        result = query(self.get_nested_doc(), ".a > .b")
        assert len(result) == 1


class TestAdjacentSiblingCombinator(SelectorTestCase):
    """Test adjacent sibling combinator (+)."""

    def test_adjacent_sibling(self):
        result = query(self.get_sibling_doc(), "h1 + p")
        assert len(result) == 1
        assert "first" in result[0].attrs.get("class", "")

    def test_adjacent_sibling_chain(self):
        result = query(self.get_sibling_doc(), ".first + p")
        assert len(result) == 1
        assert "second" in result[0].attrs.get("class", "")

    def test_adjacent_sibling_no_match(self):
        result = query(self.get_sibling_doc(), ".first + span")
        assert len(result) == 0


class TestGeneralSiblingCombinator(SelectorTestCase):
    """Test general sibling combinator (~)."""

    def test_general_sibling(self):
        result = query(self.get_sibling_doc(), "h1 ~ p")
        assert len(result) == 4  # All p elements after h1

    def test_general_sibling_with_class(self):
        result = query(self.get_sibling_doc(), ".first ~ p")
        assert len(result) == 3  # second, third, fourth


class TestFirstChild(SelectorTestCase):
    """Test :first-child pseudo-class."""

    def test_first_child(self):
        result = query(self.get_simple_doc(), "li:first-child")
        assert len(result) == 1

    def test_first_child_with_tag(self):
        result = query(self.get_sibling_doc(), "div > :first-child")
        assert len(result) == 1
        assert result[0].name == "h1"


class TestLastChild(SelectorTestCase):
    """Test :last-child pseudo-class."""

    def test_last_child(self):
        result = query(self.get_simple_doc(), "li:last-child")
        assert len(result) == 1

    def test_last_child_not_last(self):
        result = query(self.get_sibling_doc(), "h1:last-child")
        assert len(result) == 0


class TestNthChild(SelectorTestCase):
    """Test :nth-child() pseudo-class."""

    def test_nth_child_number(self):
        result = query(self.get_simple_doc(), "li:nth-child(2)")
        assert len(result) == 1
        assert "special" in result[0].attrs.get("class", "")

    def test_nth_child_odd(self):
        result = query(self.get_simple_doc(), "li:nth-child(odd)")
        assert len(result) == 2  # 1st and 3rd

    def test_nth_child_even(self):
        result = query(self.get_simple_doc(), "li:nth-child(even)")
        assert len(result) == 1  # 2nd

    def test_nth_child_formula(self):
        result = query(self.get_simple_doc(), "li:nth-child(2n+1)")
        assert len(result) == 2  # Same as odd

    def test_nth_child_n(self):
        doc = JustHTML("<html><body><ul><li>1</li><li>2</li><li>3</li><li>4</li><li>5</li></ul></body></html>").root
        result = query(doc, "li:nth-child(n)")
        assert len(result) == 5  # All

    def test_nth_child_2n(self):
        doc = JustHTML("<html><body><ul><li>1</li><li>2</li><li>3</li><li>4</li><li>5</li></ul></body></html>").root
        result = query(doc, "li:nth-child(2n)")
        assert len(result) == 2  # 2nd and 4th

    def test_nth_child_negative_offset(self):
        doc = JustHTML("<html><body><ul><li>1</li><li>2</li><li>3</li><li>4</li><li>5</li></ul></body></html>").root
        result = query(doc, "li:nth-child(-n+3)")
        assert len(result) == 3  # First 3


class TestNot(SelectorTestCase):
    """Test :not() pseudo-class."""

    def test_not_tag(self):
        result = query(self.get_simple_doc(), "div:not(#sidebar)")
        assert len(result) == 1
        assert result[0].attrs["id"] == "main"

    def test_not_class(self):
        result = query(self.get_simple_doc(), "p:not(.intro)")
        assert len(result) == 1
        assert "content" in result[0].attrs.get("class", "")

    def test_not_with_combinator(self):
        result = query(self.get_simple_doc(), "div > p:not(.content)")
        assert len(result) == 1


class TestOnlyChild(SelectorTestCase):
    """Test :only-child pseudo-class."""

    def test_only_child(self):
        doc = JustHTML("<html><body><div><span>Only</span></div></body></html>").root
        result = query(doc, "span:only-child")
        assert len(result) == 1

    def test_only_child_no_match(self):
        result = query(self.get_simple_doc(), "li:only-child")
        assert len(result) == 0


class TestEmpty(SelectorTestCase):
    """Test :empty pseudo-class."""

    def test_empty(self):
        result = query(self.get_empty_and_root_doc(), ".empty:empty")
        assert len(result) == 1

    def test_empty_whitespace(self):
        # Whitespace-only text nodes still count as content per spec
        result = query(self.get_empty_and_root_doc(), ".whitespace:empty")
        assert len(result) == 1  # Whitespace-only counts as empty per CSS spec

    def test_empty_with_text(self):
        result = query(self.get_empty_and_root_doc(), ".text:empty")
        assert len(result) == 0


class TestRoot(SelectorTestCase):
    """Test :root pseudo-class."""

    def test_root(self):
        result = query(self.get_simple_doc(), ":root")
        assert len(result) == 1
        assert result[0].name == "html"

    def test_root_with_tag(self):
        result = query(self.get_simple_doc(), "html:root")
        assert len(result) == 1


class TestFirstOfType(SelectorTestCase):
    """Test :first-of-type pseudo-class."""

    def test_first_of_type(self):
        result = query(self.get_sibling_doc(), "p:first-of-type")
        assert len(result) == 1
        assert "first" in result[0].attrs.get("class", "")


class TestLastOfType(SelectorTestCase):
    """Test :last-of-type pseudo-class."""

    def test_last_of_type(self):
        result = query(self.get_sibling_doc(), "p:last-of-type")
        assert len(result) == 1
        assert "fourth" in result[0].attrs.get("class", "")


class TestNthOfType(SelectorTestCase):
    """Test :nth-of-type() pseudo-class."""

    def test_nth_of_type(self):
        result = query(self.get_sibling_doc(), "p:nth-of-type(2)")
        assert len(result) == 1
        assert "second" in result[0].attrs.get("class", "")

    def test_nth_of_type_odd(self):
        result = query(self.get_sibling_doc(), "p:nth-of-type(odd)")
        assert len(result) == 2


class TestOnlyOfType(SelectorTestCase):
    """Test :only-of-type pseudo-class."""

    def test_only_of_type(self):
        result = query(self.get_sibling_doc(), "h1:only-of-type")
        assert len(result) == 1

    def test_only_of_type_no_match(self):
        result = query(self.get_sibling_doc(), "p:only-of-type")
        assert len(result) == 0


class TestSelectorGroups(SelectorTestCase):
    """Test comma-separated selector lists."""

    def test_two_selectors(self):
        result = query(self.get_simple_doc(), "h1, h2")
        assert len(result) == 1
        assert result[0].name == "h1"

    def test_multiple_selectors(self):
        result = query(self.get_simple_doc(), "h1, p, li")
        assert len(result) == 6  # 1 h1 + 2 p + 3 li

    def test_complex_selectors(self):
        result = query(self.get_simple_doc(), "#main p, #sidebar a")
        assert len(result) == 3  # 2 p + 1 a


class TestComplexSelectors(SelectorTestCase):
    """Test complex selector combinations."""

    def test_compound_selector(self):
        result = query(self.get_simple_doc(), "div.container#main")
        assert len(result) == 1

    def test_compound_with_attribute(self):
        result = query(self.get_simple_doc(), "a[href][data-id]")
        assert len(result) == 1

    def test_multiple_combinators(self):
        result = query(self.get_nested_doc(), ".a > .b > .c span")
        assert len(result) == 1

    def test_pseudo_with_combinator(self):
        result = query(self.get_sibling_doc(), "div > p:first-child")
        assert len(result) == 0  # First child is h1, not p

    def test_complex_real_world(self):
        result = query(self.get_simple_doc(), "div.container > ul li.special")
        assert len(result) == 1


class TestNodeQuery(SelectorTestCase):
    """Test the query() method on nodes."""

    def test_query_from_document(self):
        result = self.get_simple_doc().query("p")
        assert len(result) == 2

    def test_query_from_subtree(self):
        main_div = query(self.get_simple_doc(), "#main")[0]
        result = main_div.query("p")
        assert len(result) == 2

    def test_query_from_subtree_limited(self):
        sidebar = query(self.get_simple_doc(), "#sidebar")[0]
        result = sidebar.query("p")
        assert len(result) == 0

    def test_query_from_subtree_includes_self(self):
        main_div = query(self.get_simple_doc(), "#main")[0]
        result = main_div.query("div")
        # Should NOT include self, just descendants
        assert len(result) == 0


class TestMatchesFunction(SelectorTestCase):
    """Test the matches() function."""

    def test_matches_true(self):
        div = query(self.get_simple_doc(), "#main")[0]
        assert matches(div, "div")
        assert matches(div, "#main")
        assert matches(div, ".container")
        assert matches(div, "div.container")

    def test_matches_false(self):
        div = query(self.get_simple_doc(), "#main")[0]
        assert not matches(div, "span")
        assert not matches(div, "#sidebar")
        assert not matches(div, ".other")

    def test_matches_with_combinator(self):
        p = query(self.get_simple_doc(), "p.intro")[0]
        assert matches(p, "div p")
        assert matches(p, "#main p")
        assert not matches(p, "#sidebar p")


class TestErrorHandling(SelectorTestCase):
    """Test error handling for invalid selectors."""

    def test_empty_selector(self):
        try:
            query(self.get_simple_doc(), "")
            raise AssertionError("Expected SelectorError")
        except SelectorError:
            pass

    def test_whitespace_only(self):
        try:
            query(self.get_simple_doc(), "   ")
            raise AssertionError("Expected SelectorError")
        except SelectorError:
            pass

    def test_invalid_character(self):
        try:
            query(self.get_simple_doc(), "div@foo")
            raise AssertionError("Expected SelectorError")
        except SelectorError:
            pass

    def test_unclosed_attribute(self):
        try:
            query(self.get_simple_doc(), "[attr")
            raise AssertionError("Expected SelectorError")
        except SelectorError:
            pass

    def test_missing_attribute_name(self):
        try:
            query(self.get_simple_doc(), "[]")
            raise AssertionError("Expected SelectorError")
        except SelectorError:
            pass

    def test_invalid_attribute_operator(self):
        try:
            query(self.get_simple_doc(), "[attr!=value]")
            raise AssertionError("Expected SelectorError")
        except SelectorError:
            pass

    def test_unclosed_string(self):
        try:
            query(self.get_simple_doc(), '[attr="unclosed]')
            raise AssertionError("Expected SelectorError")
        except SelectorError:
            pass

    def test_missing_pseudo_name(self):
        try:
            query(self.get_simple_doc(), "div:")
            raise AssertionError("Expected SelectorError")
        except SelectorError:
            pass

    def test_unsupported_pseudo(self):
        try:
            query(self.get_simple_doc(), "div:hover")
            raise AssertionError("Expected SelectorError")
        except SelectorError:
            pass

    def test_dangling_combinator(self):
        try:
            query(self.get_simple_doc(), "div >")
            raise AssertionError("Expected SelectorError")
        except SelectorError:
            pass

    def test_double_combinator(self):
        try:
            query(self.get_simple_doc(), "div > > p")
            raise AssertionError("Expected SelectorError")
        except SelectorError:
            pass

    def test_missing_id_name(self):
        try:
            query(self.get_simple_doc(), "#")
            raise AssertionError("Expected SelectorError")
        except SelectorError:
            pass

    def test_missing_class_name(self):
        try:
            query(self.get_simple_doc(), ".")
            raise AssertionError("Expected SelectorError")
        except SelectorError:
            pass


class TestEdgeCases(SelectorTestCase):
    """Test edge cases and special scenarios."""

    def test_escaped_characters(self):
        doc = JustHTML('<html><body><div id="foo:bar">Test</div></body></html>').root
        result = query(doc, "[id='foo:bar']")
        assert len(result) == 1

    def test_deeply_nested(self):
        html = "<html><body>" + "<div>" * 100 + "<span>Deep</span>" + "</div>" * 100 + "</body></html>"
        doc = JustHTML(html).root
        result = query(doc, "span")
        assert len(result) == 1

    def test_many_siblings(self):
        html = "<html><body><ul>" + "".join(f"<li>{i}</li>" for i in range(100)) + "</ul></body></html>"
        doc = JustHTML(html).root
        result = query(doc, "li:nth-child(50)")
        assert len(result) == 1

    def test_empty_document(self):
        doc = JustHTML("").root
        result = query(doc, "div")
        assert len(result) == 0

    def test_text_only_document(self):
        doc = JustHTML("Just text").root
        result = query(doc, "*")
        # Should only match html, head, body (created by parser)
        assert len(result) == 3

    def test_special_attribute_values(self):
        doc = JustHTML('<html><body><a href="has spaces">Link</a></body></html>').root
        result = query(doc, '[href="has spaces"]')
        assert len(result) == 1

    def test_unicode_content(self):
        doc = JustHTML('<html><body><div class="日本語">テスト</div></body></html>').root
        result = query(doc, ".日本語")
        assert len(result) == 1

    def test_query_on_text_node(self):
        body = query(self.get_simple_doc(), "body")[0]
        # Text nodes don't match element selectors
        result = []
        for child in body.children:
            if hasattr(child, "name") and child.name == "#text":
                if matches(child, "div"):
                    result.append(child)
        assert len(result) == 0

    def test_fragment_query(self):
        doc = JustHTML("<html><body><div><p>Test</p></div></body></html>").root
        result = query(doc, "p")
        assert len(result) == 1

    def test_nth_child_zero(self):
        doc = JustHTML("<html><body><ul><li>1</li><li>2</li></ul></body></html>").root
        result = query(doc, "li:nth-child(0)")
        assert len(result) == 0  # No 0th child

    def test_nth_child_negative(self):
        doc = JustHTML("<html><body><ul><li>1</li><li>2</li></ul></body></html>").root
        result = query(doc, "li:nth-child(-1)")
        assert len(result) == 0  # Negative indices don't match

    def test_nth_child_large_number(self):
        doc = JustHTML("<html><body><ul><li>1</li><li>2</li></ul></body></html>").root
        result = query(doc, "li:nth-child(100)")
        assert len(result) == 0

    def test_attribute_empty_value(self):
        doc = JustHTML('<html><body><input type="" /></body></html>').root
        result = query(doc, '[type=""]')
        assert len(result) == 1

    def test_class_with_hyphen(self):
        doc = JustHTML('<html><body><div class="my-class">Test</div></body></html>').root
        result = query(doc, ".my-class")
        assert len(result) == 1

    def test_id_with_hyphen(self):
        doc = JustHTML('<html><body><div id="my-id">Test</div></body></html>').root
        result = query(doc, "#my-id")
        assert len(result) == 1

    def test_tag_with_hyphen(self):
        doc = JustHTML("<html><body><my-element>Test</my-element></body></html>").root
        result = query(doc, "my-element")
        assert len(result) == 1


class TestTemplateContent(SelectorTestCase):
    """Test querying into template content."""

    def test_template_query(self):
        doc = JustHTML("<html><body><template><div class='inside'>Content</div></template></body></html>").root
        result = query(doc, ".inside")
        assert len(result) == 1


class TestFastPathCoverage(unittest.TestCase):
    def test_parse_selector_empty_raises(self):
        with self.assertRaises(SelectorError):
            parse_selector("   ")

    def test_is_simple_tag_selector_empty(self):
        assert _is_simple_tag_selector("") is False

    def test_template_content_none_branches(self):
        class DummyNode:
            __slots__ = ("attrs", "children", "name", "namespace", "parent", "template_content")

            def __init__(
                self, *, name: str, namespace: str, children: list[object] | None = None, template_content=None
            ):
                self.name = name
                self.namespace = namespace
                self.children = children or []
                self.template_content = template_content
                self.attrs = {}
                self.parent = None

        # Cover branches where template_content is None.
        root_template_no_content = DummyNode(name="template", namespace="html", children=[], template_content=None)
        root_with_template_child = DummyNode(
            name="div",
            namespace="html",
            children=[DummyNode(name="template", namespace="html", children=[], template_content=None)],
            template_content=None,
        )

        _query_descendants_tag(root_template_no_content, "p", [])
        _query_descendants_tag(root_with_template_child, "p", [])

        selector = parse_selector("p")
        _query_descendants(root_template_no_content, selector, [])
        _query_descendants(root_with_template_child, selector, [])

    def test_template_root_tag_query_fast_path(self):
        jt = JustHTML("<template><div class='inside'>Content</div></template>")
        template = jt.query("template")[0]
        result = template.query("div")
        assert len(result) == 1
        assert result[0].name == "div"

    def test_template_root_non_tag_query_slow_path(self):
        jt = JustHTML("<template><div class='inside'>Content</div></template>")
        template = jt.query("template")[0]
        result = template.query(".inside")
        assert len(result) == 1

    def test_simple_tag_selector_rejects_complex(self):
        doc = JustHTML("<html><body><p id='x'>Hi</p></body></html>").root
        result = query(doc, "p#x")
        assert len(result) == 1


class TestTokenizerCoverage(SelectorTestCase):
    """Tests for additional tokenizer coverage."""

    def test_escaped_character_in_string(self):
        doc = JustHTML('<html><body><div data-value="hello world">Test</div></body></html>').root
        result = query(doc, "[data-value='hello world']")
        assert len(result) == 1

    def test_non_ascii_identifier(self):
        doc = JustHTML('<html><body><div class="über">Test</div></body></html>').root
        result = query(doc, ".über")
        assert len(result) == 1

    def test_underscore_in_identifier(self):
        doc = JustHTML('<html><body><div class="my_class">Test</div></body></html>').root
        result = query(doc, ".my_class")
        assert len(result) == 1

    def test_digit_in_identifier(self):
        doc = JustHTML('<html><body><div class="class1">Test</div></body></html>').root
        result = query(doc, ".class1")
        assert len(result) == 1


class TestParserCoverage(SelectorTestCase):
    """Tests for additional parser coverage."""

    def test_empty_selector_parts(self):
        doc = JustHTML("<html><body><div>Test</div></body></html>").root
        result = query(doc, "div")
        assert len(result) == 1


class TestMatcherCoverage(SelectorTestCase):
    """Tests for additional matcher coverage."""

    def test_empty_parts_complex_selector(self):
        doc = JustHTML("<html><body><div><p>Test</p></div></body></html>").root
        result = query(doc, "div p")
        assert len(result) == 1

    def test_node_without_parent(self):
        doc = JustHTML("<html><body><div>Test</div></body></html>").root
        html_elem = query(doc, "html")[0]
        assert matches(html_elem, ":first-child")

    def test_attribute_case_insensitive(self):
        doc = JustHTML('<html><body><div DATA-ID="123">Test</div></body></html>').root
        result = query(doc, "[data-id]")
        assert len(result) == 1

    def test_attribute_operator_no_value(self):
        doc = JustHTML('<html><body><div data-x="">Test</div></body></html>').root
        # Empty value in ^= doesn't match anything
        assert len(query(doc, '[data-x^=""]')) == 0
        assert len(query(doc, '[data-x$=""]')) == 0
        assert len(query(doc, '[data-x*=""]')) == 0


class TestNthExpressionCoverage(SelectorTestCase):
    """Tests for nth-expression parsing coverage."""

    def test_nth_just_n(self):
        doc = JustHTML("<html><body><ul><li>1</li><li>2</li><li>3</li></ul></body></html>").root
        result = query(doc, "li:nth-child(n)")
        assert len(result) == 3

    def test_nth_negative_a(self):
        doc = JustHTML("<html><body><ul><li>1</li><li>2</li><li>3</li><li>4</li><li>5</li></ul></body></html>").root
        result = query(doc, "li:nth-child(-n+3)")
        assert len(result) == 3

    def test_nth_positive_n_only(self):
        doc = JustHTML("<html><body><ul><li>1</li><li>2</li><li>3</li></ul></body></html>").root
        result = query(doc, "li:nth-child(+n)")
        assert len(result) == 3

    def test_nth_invalid_expression(self):
        doc = JustHTML("<html><body><ul><li>1</li><li>2</li></ul></body></html>").root
        result = query(doc, "li:nth-child(invalid)")
        assert len(result) == 0

    def test_nth_invalid_a_part(self):
        doc = JustHTML("<html><body><ul><li>1</li><li>2</li></ul></body></html>").root
        result = query(doc, "li:nth-child(xn+1)")
        assert len(result) == 0

    def test_nth_invalid_b_part(self):
        doc = JustHTML("<html><body><ul><li>1</li><li>2</li></ul></body></html>").root
        result = query(doc, "li:nth-child(2n+x)")
        assert len(result) == 0

    def test_nth_of_type_no_parent(self):
        doc = JustHTML("<html><body><div>Test</div></body></html>").root
        result = query(doc, "html:nth-of-type(1)")
        assert len(result) == 1

    def test_nth_child_no_parent(self):
        doc = JustHTML("<html><body><div>Test</div></body></html>").root
        assert not matches(doc, ":nth-child(1)")


class TestPseudoClassCoverage(SelectorTestCase):
    """Tests for pseudo-class coverage."""

    def test_empty_with_element_child(self):
        result = query(self.get_empty_and_root_doc(), ".nested:empty")
        assert len(result) == 0  # Has element child

    def test_root_with_no_parent(self):
        doc = JustHTML("<html><body><div>Test</div></body></html>").root
        result = query(doc, "html:root")
        assert len(result) == 1

    def test_first_of_type_multiple_types(self):
        doc = JustHTML("<html><body><div>1</div><span>2</span><div>3</div></body></html>").root
        result = query(doc, "div:first-of-type")
        assert len(result) == 1

    def test_last_of_type_multiple_types(self):
        doc = JustHTML("<html><body><div>1</div><span>2</span><div>3</div></body></html>").root
        result = query(doc, "div:last-of-type")
        assert len(result) == 1

    def test_not_with_empty_arg(self):
        doc = JustHTML("<html><body><div>Test</div></body></html>").root
        result = query(doc, "div:not()")
        assert len(result) == 1  # Empty :not() matches everything


class TestComplexCombinatorCoverage(SelectorTestCase):
    """Tests for complex combinator edge cases."""

    def test_multiple_descendants(self):
        doc = JustHTML("<html><body><div><section><p>Deep</p></section></div></body></html>").root
        result = query(doc, "body div section p")
        assert len(result) == 1

    def test_child_no_parent(self):
        doc = JustHTML("<html><body><div>Test</div></body></html>").root
        result = query(doc, "nonexistent > div")
        assert len(result) == 0

    def test_sibling_no_previous(self):
        doc = JustHTML("<html><body><p>First</p></body></html>").root
        result = query(doc, "div + p")
        assert len(result) == 0

    def test_general_sibling_no_previous(self):
        doc = JustHTML("<html><body><p>First</p></body></html>").root
        result = query(doc, "div ~ p")
        assert len(result) == 0

    def test_general_sibling_multi_combinator(self):
        # This selector has ~ followed by another combinator, triggering loop continuation
        doc = JustHTML("<html><body><div><h1>Heading</h1><p>Para</p></div></body></html>").root
        # div > h1 ~ p: first match p, then check ~ (h1 is sibling), then check > (div is parent of h1)
        result = query(doc, "div > h1 ~ p")
        assert len(result) == 1

    def test_general_sibling_with_descendant_before(self):
        # Selector with ~ followed by space combinator
        doc = JustHTML("<html><body><div><h1>H</h1><p><span>S</span></p></div></body></html>").root
        # h1 ~ p span: match span, check space (p is ancestor), check ~ (h1 is sibling of p)
        result = query(doc, "h1 ~ p span")
        assert len(result) == 1

    def test_double_general_sibling(self):
        # Two ~ combinators in a row - covers branch 546->518 (loop back after ~)
        doc = JustHTML("<html><body><div><h1>H</h1><p>P</p><span>S</span></div></body></html>").root
        # h1 ~ p ~ span: match span, check ~ for p, check ~ for h1
        result = query(doc, "h1 ~ p ~ span")
        assert len(result) == 1


class TestAttributeSelectorEdgeCases(SelectorTestCase):
    """Tests for attribute selector edge cases."""

    def test_hyphen_prefix_no_match_without_hyphen(self):
        doc = JustHTML('<html><body><p lang="english">Text</p></body></html>').root
        result = query(doc, '[lang|="en"]')
        assert len(result) == 0

    def test_contains_word_empty_class(self):
        doc = JustHTML('<html><body><div class="">Text</div></body></html>').root
        result = query(doc, '[class~="foo"]')
        assert len(result) == 0

    def test_attribute_on_node_without_attrs(self):
        doc = JustHTML("<html><body><div>Test</div></body></html>").root
        result = query(doc, "[id]")
        assert len(result) == 0


class TestReprMethods(SelectorTestCase):
    """Test __repr__ methods for debugging."""

    def test_token_repr(self):
        tok = Token(TokenType.TAG, "div")
        assert "TAG" in repr(tok)
        assert "div" in repr(tok)

    def test_simple_selector_repr(self):
        sel = SimpleSelector(SimpleSelector.TYPE_TAG, name="div")
        assert "tag" in repr(sel)
        assert "div" in repr(sel)

    def test_simple_selector_repr_with_operator(self):
        sel = SimpleSelector(SimpleSelector.TYPE_ATTR, name="href", operator="^=", value="http")
        assert "attr" in repr(sel)
        assert "href" in repr(sel)
        assert "^=" in repr(sel)

    def test_compound_selector_repr(self):
        sel = CompoundSelector([SimpleSelector(SimpleSelector.TYPE_TAG, name="div")])
        assert "CompoundSelector" in repr(sel)

    def test_complex_selector_repr(self):
        sel = ComplexSelector()
        assert "ComplexSelector" in repr(sel)

    def test_selector_list_repr(self):
        sel = SelectorList([])
        assert "SelectorList" in repr(sel)


class TestTokenizerEdgeCases(SelectorTestCase):
    """Test tokenizer edge cases."""

    def test_attribute_with_spaces_around_operator(self):
        doc = JustHTML('<html><body><div id="test">Test</div></body></html>').root
        result = query(doc, "[ id = test ]")
        assert len(result) == 1

    def test_combinator_with_extra_spaces(self):
        doc = JustHTML("<html><body><div><p>Test</p></div></body></html>").root
        result = query(doc, "div   >   p")
        assert len(result) == 1

    def test_multiple_pseudo_classes(self):
        doc = JustHTML("<html><body><ul><li>Only</li></ul></body></html>").root
        result = query(doc, "li:first-child:last-child")
        assert len(result) == 1

    def test_pseudo_with_arg_containing_spaces(self):
        doc = JustHTML("<html><body><ul><li>1</li><li>2</li><li>3</li></ul></body></html>").root
        result = query(doc, "li:nth-child( 2n + 1 )")
        assert len(result) == 2

    def test_escaped_quote_in_string(self):
        tokenizer = SelectorTokenizer('[attr="hello\\"world"]')
        tokens = tokenizer.tokenize()
        string_tokens = [t for t in tokens if t.type == TokenType.STRING]
        assert len(string_tokens) == 1
        assert string_tokens[0].value == 'hello"world'

    def test_peek_past_end(self):
        tokenizer = SelectorTokenizer("a")
        assert tokenizer._peek(10) == ""

    def test_advance_method(self):
        tokenizer = SelectorTokenizer("ab")
        assert tokenizer._advance() == "a"
        assert tokenizer._advance() == "b"
        assert tokenizer._advance() == ""

    def test_escape_at_end_of_string(self):
        # String ends with backslash - edge case
        tokenizer = SelectorTokenizer('[attr="test\\"]')
        try:
            tokenizer.tokenize()
            raise AssertionError("Expected SelectorError")
        except SelectorError:
            pass

    def test_nested_parens_in_functional_pseudo(self):
        # Test nested parentheses in pseudo-class argument
        doc = JustHTML("<html><body><div><p>Test</p></div></body></html>").root
        result = query(doc, "p:not(div)")
        assert len(result) == 1


class TestMatcherEdgeCases(SelectorTestCase):
    """Test matcher edge cases."""

    def test_matches_with_simple_selector_direct(self):
        matcher = SelectorMatcher()
        doc = JustHTML("<html><body><div>Test</div></body></html>").root
        div = query(doc, "div")[0]

        simple = SimpleSelector(SimpleSelector.TYPE_TAG, name="div")
        assert matcher.matches(div, simple)

    def test_matches_unknown_type(self):
        matcher = SelectorMatcher()
        doc = JustHTML("<html><body><div>Test</div></body></html>").root
        div = query(doc, "div")[0]

        # Should return False for unknown types
        assert not matcher.matches(div, "not a selector")

    def test_sibling_with_no_parent(self):
        matcher = SelectorMatcher()
        doc = JustHTML("<html><body><div>Test</div></body></html>").root
        assert matcher._get_previous_sibling(doc) is None

    def test_element_children_no_children(self):
        matcher = SelectorMatcher()
        result = matcher._get_element_children(None)
        assert result == []

    def test_is_first_child_no_parent(self):
        matcher = SelectorMatcher()
        doc = JustHTML("<html><body><div>Test</div></body></html>").root
        assert not matcher._is_first_child(doc)

    def test_is_last_child_no_parent(self):
        matcher = SelectorMatcher()
        doc = JustHTML("<html><body><div>Test</div></body></html>").root
        assert not matcher._is_last_child(doc)

    def test_is_first_of_type_no_parent(self):
        matcher = SelectorMatcher()
        doc = JustHTML("<html><body><div>Test</div></body></html>").root
        assert not matcher._is_first_of_type(doc)

    def test_is_last_of_type_no_parent(self):
        matcher = SelectorMatcher()
        doc = JustHTML("<html><body><div>Test</div></body></html>").root
        assert not matcher._is_last_of_type(doc)

    def test_nth_child_node_not_in_children(self):
        matcher = SelectorMatcher()
        doc = JustHTML("<html><body><ul><li>1</li></ul></body></html>").root
        li = query(doc, "li")[0]
        selector = SimpleSelector(SimpleSelector.TYPE_PSEUDO, name="nth-child", arg="1")
        assert matcher._matches_pseudo(li, selector)

    def test_nth_of_type_node_not_found(self):
        matcher = SelectorMatcher()
        doc = JustHTML("<html><body><ul><li>1</li></ul></body></html>").root
        li = query(doc, "li")[0]
        selector = SimpleSelector(SimpleSelector.TYPE_PSEUDO, name="nth-of-type", arg="1")
        assert matcher._matches_pseudo(li, selector)

    def test_matches_compound_selector_direct(self):
        matcher = SelectorMatcher()
        doc = JustHTML('<html><body><div class="foo">Test</div></body></html>').root
        div = query(doc, "div")[0]

        compound = CompoundSelector(
            [
                SimpleSelector(SimpleSelector.TYPE_TAG, name="div"),
                SimpleSelector(SimpleSelector.TYPE_CLASS, name="foo"),
            ]
        )
        assert matcher.matches(div, compound)

    def test_complex_selector_empty_parts(self):
        matcher = SelectorMatcher()
        doc = JustHTML("<html><body><div>Test</div></body></html>").root
        div = query(doc, "div")[0]

        complex_sel = ComplexSelector()
        # Empty parts should not match
        assert not matcher._matches_complex(div, complex_sel)

    def test_unknown_selector_type_in_simple(self):
        matcher = SelectorMatcher()
        doc = JustHTML("<html><body><div>Test</div></body></html>").root
        div = query(doc, "div")[0]

        # Create a SimpleSelector with unknown type
        selector = SimpleSelector("unknown_type", name="test")
        assert not matcher._matches_simple(div, selector)

    def test_unknown_attribute_operator(self):
        matcher = SelectorMatcher()
        doc = JustHTML('<html><body><div data-x="abc">Test</div></body></html>').root
        div = query(doc, "div")[0]

        # Create an attribute selector with unknown operator
        selector = SimpleSelector(SimpleSelector.TYPE_ATTR, name="data-x", operator="??", value="abc")
        assert not matcher._matches_attribute(div, selector)


class TestParserEdgeCases(SelectorTestCase):
    """Test parser edge cases."""

    def test_parser_peek_past_end(self):
        tokenizer = SelectorTokenizer("div")
        tokens = tokenizer.tokenize()
        parser = SelectorParser(tokens)
        # Consume all tokens
        while parser._peek().type != TokenType.EOF:
            parser._advance()
        parser._advance()  # Consume EOF
        # Peek past end should return EOF
        assert parser._peek().type == TokenType.EOF

    def test_parser_expect_wrong_type(self):
        tokenizer = SelectorTokenizer("div")
        tokens = tokenizer.tokenize()
        parser = SelectorParser(tokens)
        try:
            parser._expect(TokenType.ID)
            raise AssertionError("Expected SelectorError")
        except SelectorError:
            pass

    def test_parser_unexpected_token(self):
        # Create a token list directly that will cause the parser to error
        tokens = [
            Token(TokenType.TAG, "div"),
            Token(TokenType.ATTR_END),  # Unexpected ] without [
            Token(TokenType.EOF),
        ]
        parser = SelectorParser(tokens)
        try:
            parser.parse()
            raise AssertionError("Expected SelectorError")
        except SelectorError:
            pass

    def test_complex_selector_returns_none(self):
        # Empty input after comma should cause issues
        tokenizer = SelectorTokenizer("div,")
        tokens = tokenizer.tokenize()
        parser = SelectorParser(tokens)
        # Should handle the trailing comma gracefully or error
        try:
            result = parser.parse()
            # If it parsed, check the result
            assert result is not None
        except SelectorError:
            pass  # This is acceptable too


class TestAdditionalCoverage(SelectorTestCase):
    """Additional tests to cover remaining uncovered lines."""

    def test_escape_at_very_end_of_input(self):
        # Line 102: Backslash at the very end of input (no character after it)
        # The string is unterminated so we get an error, but line 102 is executed first
        try:
            tokenizer = SelectorTokenizer('[attr="test\\')
            tokenizer.tokenize()
            raise AssertionError("Expected SelectorError")
        except SelectorError:
            pass

    def test_unquoted_attr_empty_value(self):
        # Line 111->116: _read_unquoted_attr_value when at ] immediately
        # When parsing [attr=] with no value, the unquoted reader returns empty
        tokenizer = SelectorTokenizer("[attr=]")
        tokens = tokenizer.tokenize()
        # Should have an empty string token
        string_tokens = [t for t in tokens if t.type == TokenType.STRING]
        assert len(string_tokens) == 1
        assert string_tokens[0].value == ""

    def test_unquoted_attr_value_at_end_of_input(self):
        # Line 111->116: _read_unquoted_attr_value called at end of input
        try:
            tokenizer = SelectorTokenizer("[attr=")
            tokenizer.tokenize()
            raise AssertionError("Expected SelectorError")
        except SelectorError:
            pass

    def test_nested_parens_in_pseudo_arg(self):
        # Line 253: Nested parentheses in functional pseudo-class
        # For example: :nth-child((2n+1)) - extra parens
        tokenizer = SelectorTokenizer(":nth-child((2n+1))")
        tokens = tokenizer.tokenize()
        # The arg should be "(2n+1)" including the inner parens
        string_tokens = [t for t in tokens if t.type == TokenType.STRING]
        assert len(string_tokens) == 1
        assert string_tokens[0].value == "(2n+1)"

    def test_pending_whitespace_at_start(self):
        # Line 144->146: pending whitespace when tokens list is empty
        # This happens when selector starts with whitespace before anything meaningful
        doc = JustHTML("<html><body><div>Test</div></body></html>").root
        result = query(doc, "  div")  # leading spaces
        assert len(result) == 1

    def test_whitespace_after_combinator(self):
        # Line 144->146: pending whitespace after a combinator (should not add extra combinator)
        doc = JustHTML("<html><body><div><p>Test</p></div></body></html>").root
        result = query(doc, "div >  p")  # space after >
        assert len(result) == 1

    def test_whitespace_after_comma(self):
        # Line 144->146: pending whitespace after comma
        doc = JustHTML("<html><body><div><p>Test</p></div></body></html>").root
        result = query(doc, "div,  p")  # space after comma
        assert len(result) == 2

    def test_tokenizer_missing_closing_bracket(self):
        # Line 218: Expected ] error
        try:
            tokenizer = SelectorTokenizer('[attr="value"')
            tokenizer.tokenize()
            raise AssertionError("Expected SelectorError")
        except SelectorError:
            pass

    def test_tokenizer_nested_parens(self):
        # Line 253: Nested parentheses
        tokenizer = SelectorTokenizer(":not(div.foo)")
        tokens = tokenizer.tokenize()
        # Should tokenize successfully
        assert any(t.type == TokenType.COLON for t in tokens)

    def test_tokenizer_unclosed_paren(self):
        # Line 264: Expected ) error
        try:
            tokenizer = SelectorTokenizer(":nth-child(2n+1")
            tokenizer.tokenize()
            raise AssertionError("Expected SelectorError")
        except SelectorError:
            pass

    def test_last_of_type_no_match(self):
        # Line 750: _is_last_of_type returns False when not found
        matcher = SelectorMatcher()
        doc2 = JustHTML("<html><body><div>1</div><span>2</span><div>3</div></body></html>").root
        first_div = query(doc2, "div")[0]
        assert not matcher._is_last_of_type(first_div)

    def test_root_no_parent(self):
        # Line 645: :root with no parent returns False
        matcher = SelectorMatcher()
        doc = JustHTML("<html><body></body></html>").root
        # The document itself has no parent, so :root check returns False for it
        assert not matcher._matches_pseudo(doc, SimpleSelector(SimpleSelector.TYPE_PSEUDO, name="root"))

    def test_first_of_type_no_match(self):
        # Line 723: _is_first_of_type returns False when not the first
        matcher = SelectorMatcher()
        doc = JustHTML("<html><body><div>1</div><span>2</span><div>3</div></body></html>").root
        divs = query(doc, "div")
        second_div = divs[1]  # The second div
        assert not matcher._is_first_of_type(second_div)

    def test_string_with_escape_no_content_before(self):
        # Line 93->95: Escape at start of string (no content before backslash)
        tokenizer = SelectorTokenizer('[attr="\\"test"]')
        tokens = tokenizer.tokenize()
        string_tokens = [t for t in tokens if t.type == TokenType.STRING]
        assert len(string_tokens) == 1
        assert string_tokens[0].value == '"test'

    def test_string_with_only_escape(self):
        # Line 102: Escape character handling when nothing before backslash
        tokenizer = SelectorTokenizer('[attr="\\x"]')
        tokens = tokenizer.tokenize()
        string_tokens = [t for t in tokens if t.type == TokenType.STRING]
        assert len(string_tokens) == 1
        assert string_tokens[0].value == "x"

    def test_nested_parens_in_not(self):
        # Line 253: Nested parentheses in :not()
        doc = JustHTML("<html><body><div class='foo'><p>Test</p></div></body></html>").root
        # :not with nested selector that could have parens
        result = query(doc, "div:not(.bar)")
        assert len(result) == 1

    def test_nth_child_node_not_in_elements_list(self):
        # Lines 840, 861: Node not found in elements list
        # This is hard to trigger since we're iterating through children
        # But we test the fallthrough case
        matcher = SelectorMatcher()
        doc = JustHTML("<html><body><ul><li>1</li><li>2</li></ul></body></html>").root
        li = query(doc, "li")[0]
        # Test that it returns correct value
        result = matcher._matches_nth_child(li, "1")
        assert result
        result = matcher._matches_nth_child(li, "2")
        assert not result

    def test_nth_of_type_with_multiple_types(self):
        # Test nth-of-type with mixed element types
        doc = JustHTML("<html><body><div>1</div><span>2</span><div>3</div><span>4</span></body></html>").root
        spans = query(doc, "span")
        # First span should be nth-of-type(1)
        matcher = SelectorMatcher()
        assert matcher._matches_nth_of_type(spans[0], "1")
        assert not matcher._matches_nth_of_type(spans[0], "2")
        # Second span should be nth-of-type(2)
        assert matcher._matches_nth_of_type(spans[1], "2")

    def test_get_previous_sibling_not_found(self):
        # Test when node is first child (no previous sibling found)
        matcher = SelectorMatcher()
        doc = JustHTML("<html><body><ul><li>First</li><li>Second</li></ul></body></html>").root
        first_li = query(doc, "li")[0]
        result = matcher._get_previous_sibling(first_li)
        assert result is None

    def test_get_previous_sibling_detached_node(self):
        # Test with a node that's been detached from its parent's children list
        # This tests the defensive return None at the end
        matcher = SelectorMatcher()
        doc = JustHTML("<html><body><div><p>Para</p></div></body></html>").root
        p = query(doc, "p")[0]
        div = p.parent
        # Manually break the DOM invariant by clearing children but keeping parent ref
        original_children = div.children
        div.children = []
        result = matcher._get_previous_sibling(p)
        assert result is None
        # Restore
        div.children = original_children

    def test_nth_child_invalid_just_b(self):
        # Lines 808-809: Invalid b part (just a number but invalid)
        doc = JustHTML("<html><body><ul><li>1</li><li>2</li></ul></body></html>").root
        result = query(doc, "li:nth-child(abc)")  # Not a valid number
        assert len(result) == 0

    def test_nth_child_zero_index(self):
        # Test :nth-child(0) - matches nothing (1-indexed)
        doc = JustHTML("<html><body><ul><li>1</li><li>2</li></ul></body></html>").root
        result = query(doc, "li:nth-child(0)")
        assert len(result) == 0

    def test_nth_with_spaces_in_formula(self):
        # Test various nth-child formulas
        doc = JustHTML("<html><body><ul><li>1</li><li>2</li><li>3</li><li>4</li></ul></body></html>").root
        # -2n+4 should match 4, 2
        result = query(doc, "li:nth-child(-2n+4)")
        assert len(result) == 2

    def test_is_first_of_type_returns_false(self):
        # Line 723: Test when node type is not found (which means we exit loop without return)
        # This is actually impossible with valid DOM, but let's try
        matcher = SelectorMatcher()
        doc = JustHTML("<html><body><div>1</div><span>2</span><div>3</div></body></html>").root
        # Get the second div - it's not first of type
        divs = query(doc, "div")
        assert len(divs) == 2
        second_div = divs[1]
        assert not matcher._is_first_of_type(second_div)

    def test_general_sibling_with_no_match(self):
        # Line 549->521: General sibling doesn't find a match
        doc = JustHTML("<html><body><div><span>1</span><p>2</p></div></body></html>").root
        result = query(doc, "div ~ p")  # No div before p
        assert len(result) == 0

    def test_nth_expression_empty(self):
        # Line 767: Empty expression
        matcher = SelectorMatcher()
        result = matcher._parse_nth_expression("")
        assert result is None

    def test_nth_expression_none(self):
        # Line 767: None expression
        matcher = SelectorMatcher()
        result = matcher._parse_nth_expression(None)
        assert result is None

    def test_empty_pseudo_no_children_attr(self):
        # Test :empty when node doesn't have children attribute
        matcher = SelectorMatcher()

        # Create a minimal node-like object
        class FakeNode:
            name = "div"

            def __init__(self):
                self.attrs = {}

            def has_child_nodes(self):
                return False

        fake = FakeNode()
        selector = SimpleSelector(SimpleSelector.TYPE_PSEUDO, name="empty")
        result = matcher._matches_pseudo(fake, selector)
        assert result

    def test_empty_pseudo_with_comment(self):
        # :empty with comment child - should still be empty per CSS spec
        doc = JustHTML("<html><body><div><!-- comment --></div></body></html>").root
        result = query(doc, "div:empty")
        # Comments are #comment nodes which start with #, so they're ignored
        assert len(result) == 1

    def test_nth_child_on_document_root(self):
        # Line 826: :nth-child on node with no parent
        matcher = SelectorMatcher()
        doc = JustHTML("<html><body></body></html>").root
        selector = SimpleSelector(SimpleSelector.TYPE_PSEUDO, name="nth-child", arg="1")
        result = matcher._matches_pseudo(doc, selector)
        assert not result

    def test_nth_of_type_on_document_root(self):
        # Line 843: :nth-of-type on node with no parent
        matcher = SelectorMatcher()
        doc = JustHTML("<html><body></body></html>").root
        selector = SimpleSelector(SimpleSelector.TYPE_PSEUDO, name="nth-of-type", arg="1")
        result = matcher._matches_pseudo(doc, selector)
        assert not result

    def test_nth_of_type_invalid_expression(self):
        # Line 847: :nth-of-type with invalid expression
        doc = JustHTML("<html><body><div>1</div><div>2</div></body></html>").root
        result = query(doc, "div:nth-of-type(invalid)")
        assert len(result) == 0

    def test_is_first_of_type_detached_node(self):
        # Line 747: Test _is_first_of_type with detached node (unreachable in normal use)
        matcher = SelectorMatcher()
        doc = JustHTML("<html><body><div><p>Para</p></div></body></html>").root
        p = query(doc, "p")[0]
        div = p.parent
        # Detach node from parent's children
        original_children = div.children
        div.children = []
        result = matcher._is_first_of_type(p)
        assert not result
        div.children = original_children

    def test_nth_child_detached_node(self):
        # Line 837: Test _matches_nth_child with detached node (unreachable in normal use)
        matcher = SelectorMatcher()
        doc = JustHTML("<html><body><div><p>Para</p></div></body></html>").root
        p = query(doc, "p")[0]
        div = p.parent
        # Detach node from parent's children
        original_children = div.children
        div.children = []
        result = matcher._matches_nth_child(p, "1")
        assert not result
        div.children = original_children

    def test_nth_of_type_detached_node(self):
        # Line 858: Test _matches_nth_of_type with detached node (unreachable in normal use)
        matcher = SelectorMatcher()
        doc = JustHTML("<html><body><div><p>Para</p></div></body></html>").root
        p = query(doc, "p")[0]
        div = p.parent
        # Detach node from parent's children
        original_children = div.children
        div.children = []
        result = matcher._matches_nth_of_type(p, "1")
        assert not result
        div.children = original_children

    def test_empty_child_without_name(self):
        # Line 672->671: Test :empty when child has no name attribute
        matcher = SelectorMatcher()
        doc = JustHTML("<html><body><div id='test'>text</div></body></html>").root
        div = query(doc, "div")[0]

        # Insert a fake child without name attribute
        class FakeChild:
            pass

        original_children = div.children
        div.children = [FakeChild()]
        selector = SimpleSelector(SimpleSelector.TYPE_PSEUDO, name="empty")
        result = matcher._matches_pseudo(div, selector)
        assert result  # Unknown child is ignored
        div.children = original_children


class TestPseudoContains(SelectorTestCase):
    """Test non-standard :contains() pseudo-class."""

    def get_contains_doc(self):
        html = """
        <html><body>
            <div id="a"><button>click me</button></div>
            <div id="b"><button>do not click</button></div>
            <div id="c"><span>click</span> me</div>
        </body></html>
        """
        return JustHTML(html).root

    def test_contains_basic(self):
        result = query(self.get_contains_doc(), 'button:contains("click me")')
        assert len(result) == 1
        assert result[0].name == "button"
        assert result[0].to_text() == "click me"

    def test_contains_unquoted_arg(self):
        result = query(self.get_contains_doc(), "button:contains(click)")
        assert len(result) == 2

    def test_contains_descendant_text(self):
        result = query(self.get_contains_doc(), 'div:contains("click me")')
        ids = {n.attrs.get("id") for n in result}
        assert ids == {"a", "c"}

    def test_contains_case_sensitive(self):
        result = query(self.get_contains_doc(), 'button:contains("Click")')
        assert len(result) == 0

    def test_contains_empty_string_matches_all(self):
        result = query(self.get_contains_doc(), 'button:contains("")')
        assert len(result) == 2

    def test_contains_requires_arg(self):
        with self.assertRaises(SelectorError):
            query(self.get_contains_doc(), "button:contains()")


class TestJustHTMLMethods(unittest.TestCase):
    """Test JustHTML convenience methods that delegate to root."""

    def test_doc_query(self):
        doc = JustHTML("<html><body><div id='main'><p>Hello</p></div></body></html>")
        result = doc.query("#main")
        assert len(result) == 1
        assert result[0].attrs["id"] == "main"

    def test_doc_to_html(self):
        doc = JustHTML("<html><body><p>Test</p></body></html>")
        html = doc.to_html(pretty=False)
        assert "<p>Test</p>" in html


if __name__ == "__main__":
    unittest.main()
