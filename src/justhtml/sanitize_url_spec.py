"""Declarative URL sink definitions and matching helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

UrlSinkKind = Literal["url", "srcset", "comma_or_space_list", "space_list", "meta_refresh"]


@dataclass(frozen=True, slots=True)
class UrlSink:
    kind: UrlSinkKind
    tag: str
    attr: str
    guard_attr: str | None = None
    guard_values: Collection[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "tag", str(self.tag).lower())
        object.__setattr__(self, "attr", str(self.attr).lower())
        if self.guard_attr is not None:
            object.__setattr__(self, "guard_attr", str(self.guard_attr).lower())
        object.__setattr__(self, "guard_values", frozenset(str(value).lower() for value in self.guard_values))


_URL_BEARING_PARAM_NAMES: frozenset[str] = frozenset(
    {
        "code",
        "codebase",
        "data",
        "filename",
        "href",
        "movie",
        "src",
        "url",
    }
)

_URL_SINKS: tuple[UrlSink, ...] = (
    UrlSink(kind="url", tag="*", attr="href"),
    UrlSink(kind="url", tag="*", attr="icon"),
    UrlSink(kind="url", tag="*", attr="dynsrc"),
    UrlSink(kind="url", tag="*", attr="lowsrc"),
    UrlSink(kind="url", tag="*", attr="src"),
    UrlSink(kind="srcset", tag="*", attr="srcset"),
    UrlSink(kind="srcset", tag="*", attr="imagesrcset"),
    UrlSink(kind="url", tag="*", attr="poster"),
    UrlSink(kind="url", tag="*", attr="action"),
    UrlSink(kind="url", tag="*", attr="formaction"),
    UrlSink(kind="url", tag="*", attr="data"),
    UrlSink(kind="url", tag="*", attr="cite"),
    UrlSink(kind="url", tag="*", attr="background"),
    UrlSink(kind="url", tag="*", attr="classid"),
    UrlSink(kind="url", tag="*", attr="code"),
    UrlSink(kind="url", tag="*", attr="codebase"),
    UrlSink(kind="url", tag="*", attr="longdesc"),
    UrlSink(kind="url", tag="*", attr="manifest"),
    UrlSink(kind="url", tag="*", attr="object"),
    UrlSink(kind="comma_or_space_list", tag="*", attr="profile"),
    UrlSink(kind="url", tag="*", attr="usemap"),
    UrlSink(kind="comma_or_space_list", tag="*", attr="archive"),
    UrlSink(kind="space_list", tag="*", attr="ping"),
    UrlSink(kind="space_list", tag="*", attr="attributionsrc"),
    UrlSink(
        kind="meta_refresh",
        tag="meta",
        attr="content",
        guard_attr="http-equiv",
        guard_values=("refresh",),
    ),
    UrlSink(
        kind="url",
        tag="param",
        attr="value",
        guard_attr="name",
        guard_values=_URL_BEARING_PARAM_NAMES,
    ),
)

_URL_SINKS_BY_ATTR: Mapping[str, tuple[UrlSink, ...]] = {
    attr: tuple(sink for sink in _URL_SINKS if sink.attr == attr) for attr in {sink.attr for sink in _URL_SINKS}
}


def _url_sink_kind_for_attr(*, tag: str, attr: str, attrs: Mapping[str, str | None]) -> UrlSinkKind | None:
    for sink in _URL_SINKS_BY_ATTR.get(attr, ()):
        if sink.tag != "*" and sink.tag != tag:
            continue
        if sink.guard_attr is None:
            return sink.kind
        for key, raw_value in attrs.items():
            lower_key = key if key.islower() else key.lower()
            if lower_key == sink.guard_attr and raw_value is not None:
                if str(raw_value).strip().lower() in sink.guard_values:
                    return sink.kind
                break
    return None


_URL_LIKE_ATTRS: frozenset[str] = frozenset(sink.attr for sink in _URL_SINKS if sink.guard_attr is None)
