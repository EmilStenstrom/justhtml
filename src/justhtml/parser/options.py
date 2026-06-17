from __future__ import annotations


class ParserOptions:
    """Internal parser knobs used by conformance harnesses and compatibility tests."""

    __slots__ = ("discard_bom", "emit_bogus_markup_as_text", "scripting_enabled", "xml_coercion")

    discard_bom: bool
    emit_bogus_markup_as_text: bool
    scripting_enabled: bool
    xml_coercion: bool

    def __init__(
        self,
        *,
        discard_bom: bool = True,
        emit_bogus_markup_as_text: bool = False,
        scripting_enabled: bool = True,
        xml_coercion: bool = False,
    ) -> None:
        self.discard_bom = bool(discard_bom)
        self.emit_bogus_markup_as_text = bool(emit_bogus_markup_as_text)
        self.scripting_enabled = bool(scripting_enabled)
        self.xml_coercion = bool(xml_coercion)

    def copy(self) -> ParserOptions:
        return ParserOptions(
            discard_bom=self.discard_bom,
            emit_bogus_markup_as_text=self.emit_bogus_markup_as_text,
            scripting_enabled=self.scripting_enabled,
            xml_coercion=self.xml_coercion,
        )
