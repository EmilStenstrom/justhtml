"""Default sanitizer policies and preset helpers."""

from __future__ import annotations

from types import MappingProxyType

from justhtml.sanitizer.url import UrlPolicy, UrlRule


def _seal_url_policy(url_policy: UrlPolicy) -> None:
    if not isinstance(url_policy, UrlPolicy):
        raise TypeError("url_policy must be a UrlPolicy")

    sealed_rules: dict[tuple[str, str], UrlRule] = {}
    for (tag, attr), rule in url_policy.allow_rules.items():
        object.__setattr__(rule, "allowed_schemes", frozenset(str(s) for s in rule.allowed_schemes))
        if rule.allowed_hosts is not None:
            object.__setattr__(rule, "allowed_hosts", frozenset(str(h).lower() for h in rule.allowed_hosts))
        sealed_rules[(str(tag).lower(), str(attr).lower())] = rule

    object.__setattr__(url_policy, "allow_rules", MappingProxyType(sealed_rules))


def _seal_default_policy(policy: object) -> None:
    from justhtml.sanitizer.policy import SanitizationPolicy  # noqa: PLC0415

    if not isinstance(policy, SanitizationPolicy):
        raise TypeError("policy must be a SanitizationPolicy")

    object.__setattr__(
        policy,
        "allowed_attributes",
        MappingProxyType({str(tag).lower(): frozenset(attrs) for tag, attrs in policy.allowed_attributes.items()}),
    )
    object.__setattr__(policy, "drop_content_tags", frozenset(policy.drop_content_tags))
    object.__setattr__(policy, "allowed_css_properties", frozenset(policy.allowed_css_properties))
    object.__setattr__(policy, "force_link_rel", frozenset(policy.force_link_rel))
    _seal_url_policy(policy.url_policy)


def _build_default_policies() -> tuple[object, frozenset[str], object]:
    from justhtml.sanitizer.policy import SanitizationPolicy  # noqa: PLC0415

    default_policy = SanitizationPolicy(
        allowed_tags=[
            "p",
            "br",
            "div",
            "span",
            "blockquote",
            "pre",
            "code",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "ul",
            "ol",
            "li",
            "table",
            "caption",
            "thead",
            "tbody",
            "tfoot",
            "tr",
            "th",
            "td",
            "b",
            "strong",
            "i",
            "em",
            "u",
            "s",
            "sub",
            "sup",
            "small",
            "mark",
            "hr",
            "a",
            "img",
        ],
        allowed_attributes={
            "*": ["class", "id", "title", "lang", "dir"],
            "a": ["href", "title"],
            "img": ["src", "alt", "title", "width", "height", "loading", "decoding"],
            "th": ["colspan", "rowspan"],
            "td": ["colspan", "rowspan"],
        },
        url_policy=UrlPolicy(
            default_handling="strip",
            allow_rules={
                ("a", "href"): UrlRule(
                    allowed_schemes=["http", "https", "mailto", "tel"],
                    handling="allow",
                    resolve_protocol_relative="https",
                ),
                ("img", "src"): UrlRule(
                    allowed_schemes=[],
                    handling="allow",
                    resolve_protocol_relative=None,
                ),
            },
        ),
        allowed_css_properties=set(),
    )

    css_preset_text: frozenset[str] = frozenset(
        {
            "background-color",
            "color",
            "font-size",
            "font-style",
            "font-weight",
            "letter-spacing",
            "line-height",
            "text-align",
            "text-decoration",
            "text-transform",
            "white-space",
            "word-break",
            "word-spacing",
            "word-wrap",
        }
    )

    default_document_policy = SanitizationPolicy(
        allowed_tags=sorted(set(default_policy.allowed_tags) | {"html", "head", "body", "title"}),
        allowed_attributes=default_policy.allowed_attributes,
        url_policy=default_policy.url_policy,
        drop_comments=default_policy.drop_comments,
        drop_doctype=False,
        drop_content_tags=default_policy.drop_content_tags,
        allowed_css_properties=default_policy.allowed_css_properties,
        force_link_rel=default_policy.force_link_rel,
        strip_invisible_unicode=default_policy.strip_invisible_unicode,
    )

    _seal_default_policy(default_policy)
    _seal_default_policy(default_document_policy)
    return default_policy, css_preset_text, default_document_policy


DEFAULT_POLICY, CSS_PRESET_TEXT, DEFAULT_DOCUMENT_POLICY = _build_default_policies()


def _build_allow_all_tags() -> "SanitizationPolicy":
    """Build a policy that preserves all standard HTML5 elements.

    Drops only ``<script>`` and ``<style>`` *content* (same as the
    default policy), but keeps structural and form elements
    (``<input>``, ``<meta>``, ``<link>``, ``<form>``, etc.) that the
    default content-oriented policy strips.

    Use for **web scraping** where you want XSS/URL protection but
    need access to the full DOM::

        from justhtml import JustHTML, ALLOW_ALL_TAGS
        doc = JustHTML(html, policy=ALLOW_ALL_TAGS)
        doc.query("meta[property='og:title']")  # works!
    """
    from justhtml.sanitizer.policy import SanitizationPolicy  # noqa: PLC0415

    all_tags = {
        # Document structure
        "html", "head", "body", "title",
        # Metadata
        "meta", "link", "base",
        # Sections
        "article", "aside", "footer", "header", "hgroup", "main", "nav", "section",
        # Headings
        "h1", "h2", "h3", "h4", "h5", "h6",
        # Block
        "address", "blockquote", "dd", "div", "dl", "dt", "figcaption", "figure",
        "hr", "li", "ol", "p", "pre", "ul",
        # Inline
        "a", "abbr", "b", "bdi", "bdo", "br", "cite", "code", "data", "dfn", "em",
        "i", "kbd", "mark", "q", "rp", "rt", "rtc", "ruby", "s", "samp", "small",
        "span", "strong", "sub", "sup", "time", "u", "var", "wbr",
        # Media
        "area", "audio", "img", "map", "track", "video",
        # Embedded
        "embed", "iframe", "object", "param", "picture", "source",
        # Table
        "caption", "col", "colgroup", "table", "tbody", "td", "tfoot", "th",
        "thead", "tr",
        # Form
        "button", "datalist", "fieldset", "form", "input", "label", "legend",
        "meter", "optgroup", "option", "output", "progress", "select", "textarea",
        # Interactive
        "details", "dialog", "menu", "summary",
        # Scripting (content dropped by default, element preserved)
        "noscript", "canvas", "template",
        # Deprecated but still in the wild
        "center", "font", "big", "strike", "tt",
    }

    policy = SanitizationPolicy(
        allowed_tags=sorted(all_tags),
        allowed_attributes={
            "*": ["class", "id", "title", "lang", "dir", "role", "tabindex"],
            "a": ["href", "title", "target", "rel"],
            "img": ["src", "alt", "title", "width", "height", "loading", "decoding",
                    "srcset", "sizes"],
            "input": ["type", "name", "value", "placeholder", "checked", "disabled",
                      "readonly", "required", "min", "max", "step", "pattern",
                      "accept", "autocomplete", "autofocus"],
            "form": ["action", "method", "enctype", "novalidate"],
            "textarea": ["name", "rows", "cols", "placeholder", "disabled",
                         "readonly", "required"],
            "select": ["name", "multiple", "disabled", "required"],
            "option": ["value", "selected", "disabled"],
            "button": ["type", "name", "value", "disabled"],
            "meta": ["charset", "name", "content", "http-equiv", "property"],
            "link": ["rel", "href", "type", "sizes", "media"],
            "th": ["colspan", "rowspan", "scope"],
            "td": ["colspan", "rowspan"],
            "iframe": ["src", "width", "height", "frameborder", "allowfullscreen"],
            "video": ["src", "controls", "autoplay", "loop", "muted", "poster",
                      "width", "height"],
            "audio": ["src", "controls", "autoplay", "loop", "muted"],
            "source": ["src", "type", "srcset", "sizes"],
            "time": ["datetime"],
            "data": ["value"],
            "meter": ["value", "min", "max", "low", "high", "optimum"],
            "progress": ["value", "max"],
            "output": ["for", "form", "name"],
            "details": ["open"],
            "dialog": ["open"],
        },
        url_policy=UrlPolicy(
            default_handling="strip",
            allow_rules={
                ("a", "href"): UrlRule(
                    allowed_schemes=["http", "https", "mailto", "tel"],
                    handling="allow",
                    resolve_protocol_relative="https",
                ),
                ("img", "src"): UrlRule(
                    allowed_schemes=[],
                    handling="allow",
                    resolve_protocol_relative=None,
                ),
                ("form", "action"): UrlRule(
                    allowed_schemes=["http", "https"],
                    handling="allow",
                    resolve_protocol_relative="https",
                ),
            },
        ),
        allowed_css_properties=set(),
    )
    _seal_default_policy(policy)
    return policy


ALLOW_ALL_TAGS = _build_allow_all_tags()
