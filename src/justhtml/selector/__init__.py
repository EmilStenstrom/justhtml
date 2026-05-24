"""Selector facade for JustHTML."""

from .core import (
    DEFAULT_SELECTOR_LIMITS as DEFAULT_SELECTOR_LIMITS,
)
from .core import (
    ComplexSelector as ComplexSelector,
)
from .core import (
    CompoundSelector as CompoundSelector,
)
from .core import (
    NodeAttributeData as NodeAttributeData,
)
from .core import (
    ParentSelectorData as ParentSelectorData,
)
from .core import (
    ParsedSelector as ParsedSelector,
)
from .core import (
    SelectorError as SelectorError,
)
from .core import (
    SelectorLimits as SelectorLimits,
)
from .core import (
    SelectorList as SelectorList,
)
from .core import (
    SelectorMatcher as SelectorMatcher,
)
from .core import (
    SelectorParser as SelectorParser,
)
from .core import (
    SelectorQueryContext as SelectorQueryContext,
)
from .core import (
    SelectorTokenizer as SelectorTokenizer,
)
from .core import (
    SimpleSelector as SimpleSelector,
)
from .core import (
    Token as Token,
)
from .core import (
    TokenType as TokenType,
)
from .core import (
    _complex_selector_signature as _complex_selector_signature,
)
from .core import (
    _compound_selector_signature as _compound_selector_signature,
)
from .core import (
    _is_simple_tag_selector as _is_simple_tag_selector,
)
from .core import (
    _parse_selector as _parse_selector,
)
from .core import (
    _parse_selector_cached as _parse_selector_cached,
)
from .core import (
    _query_descendants as _query_descendants,
)
from .core import (
    _query_descendants_tag as _query_descendants_tag,
)
from .core import (
    _selector_allows_non_elements as _selector_allows_non_elements,
)
from .core import (
    _selector_signature as _selector_signature,
)
from .core import (
    _simple_selector_signature as _simple_selector_signature,
)
from .core import (
    matches as matches,
)
from .core import (
    parse_selector as parse_selector,
)
from .core import (
    query as query,
)
