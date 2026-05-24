from __future__ import annotations


class FragmentContext:
    __slots__ = ("namespace", "tag_name")

    tag_name: str
    namespace: str | None

    def __init__(self, tag_name: str, namespace: str | None = None) -> None:
        self.tag_name = tag_name
        self.namespace = namespace
