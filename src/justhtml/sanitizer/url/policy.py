"""URL policy models and signatures."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

UrlFilter = Callable[[str, str, str], str | None]


class UnsafeHtmlError(ValueError):
    """Raised when unsafe HTML is encountered and unsafe_handling='raise'."""


UnsafeHandling = Literal["strip", "raise", "collect"]

DisallowedTagHandling = Literal["unwrap", "escape", "drop"]

UrlHandling = Literal["allow", "strip", "proxy"]


@dataclass(frozen=True, slots=True)
class UrlProxy:
    url: str
    param: str = "url"

    def __post_init__(self) -> None:
        proxy_url = str(self.url).strip()
        if not proxy_url:
            raise ValueError("UrlProxy.url must be a non-empty string")
        proxy_param = str(self.param)
        if not proxy_param:
            raise ValueError("UrlProxy.param must be a non-empty string")
        from justhtml.sanitizer.url import _validate_proxy_url  # noqa: PLC0415

        _validate_proxy_url(proxy_url, proxy_param)
        object.__setattr__(self, "url", proxy_url)
        object.__setattr__(self, "param", proxy_param)


@dataclass(frozen=True, slots=True)
class UrlRule:
    """Rule for a single URL-valued attribute (e.g. a[href], img[src]).

    This is intentionally rendering-oriented.

    - Returning/keeping a URL can still cause network requests when the output
        is rendered (notably for <img src>). Applications like email viewers often
        want to block remote loads by default.
    """

    allow_fragment: bool = True
    resolve_protocol_relative: str | None = "https"
    allowed_schemes: Collection[str] = field(default_factory=set)
    allowed_hosts: Collection[str] | None = None
    handling: UrlHandling | None = None
    allow_relative: bool | None = None
    proxy: UrlProxy | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.allowed_schemes, set):
            object.__setattr__(self, "allowed_schemes", set(self.allowed_schemes))
        if self.allowed_hosts is not None and not isinstance(self.allowed_hosts, set):
            object.__setattr__(self, "allowed_hosts", set(self.allowed_hosts))

        if self.proxy is not None and not isinstance(self.proxy, UrlProxy):
            raise TypeError("UrlRule.proxy must be a UrlProxy or None")

        if self.handling is not None:
            mode = str(self.handling)
            if mode not in {"allow", "strip", "proxy"}:
                raise ValueError("Invalid UrlRule.handling. Expected one of: 'allow', 'strip', 'proxy'")
            object.__setattr__(self, "handling", mode)

        if self.allow_relative is not None:
            object.__setattr__(self, "allow_relative", bool(self.allow_relative))


@dataclass(frozen=True, slots=True)
class UrlPolicy:
    default_handling: UrlHandling = "allow"
    default_allow_relative: bool = True
    allow_rules: Mapping[tuple[str, str], UrlRule] = field(default_factory=dict)
    url_filter: UrlFilter | None = None
    proxy: UrlProxy | None = None

    def __post_init__(self) -> None:
        mode = str(self.default_handling)
        if mode not in {"allow", "strip", "proxy"}:
            raise ValueError("Invalid default_handling. Expected one of: 'allow', 'strip', 'proxy'")
        object.__setattr__(self, "default_handling", mode)

        object.__setattr__(self, "default_allow_relative", bool(self.default_allow_relative))

        if not isinstance(self.allow_rules, dict):
            object.__setattr__(self, "allow_rules", dict(self.allow_rules))

        if self.proxy is not None and not isinstance(self.proxy, UrlProxy):
            raise TypeError("UrlPolicy.proxy must be a UrlProxy or None")

        for rule in self.allow_rules.values():
            if not isinstance(rule, UrlRule):
                raise TypeError("UrlPolicy.allow_rules values must be UrlRule")
            effective_handling = rule.handling if rule.handling is not None else mode
            if effective_handling == "proxy" and self.proxy is None and rule.proxy is None:
                raise ValueError("URL handling 'proxy' requires a UrlPolicy.proxy or a per-rule UrlRule.proxy")


def _url_rule_signature(rule: UrlRule) -> tuple[Any, ...]:
    allowed_schemes = tuple(sorted(str(s) for s in rule.allowed_schemes))
    allowed_hosts = None
    if rule.allowed_hosts is not None:
        allowed_hosts = tuple(sorted(str(h).lower() for h in rule.allowed_hosts))

    proxy_sig = None
    if rule.proxy is not None:
        proxy_sig = (rule.proxy.url, rule.proxy.param)

    return (
        rule.allow_fragment,
        rule.resolve_protocol_relative,
        allowed_schemes,
        allowed_hosts,
        rule.handling,
        rule.allow_relative,
        proxy_sig,
    )


def _url_policy_signature(url_policy: UrlPolicy) -> tuple[Any, ...]:
    allow_rules_sig = tuple(
        sorted(
            (
                (str(tag).lower(), str(attr).lower()),
                _url_rule_signature(rule),
            )
            for (tag, attr), rule in url_policy.allow_rules.items()
        )
    )

    proxy_sig = None
    if url_policy.proxy is not None:
        proxy_sig = (url_policy.proxy.url, url_policy.proxy.param)

    return (
        url_policy.default_handling,
        url_policy.default_allow_relative,
        allow_rules_sig,
        url_policy.url_filter,
        proxy_sig,
    )
