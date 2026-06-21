"""HTML5 spec constants for tree building."""

from __future__ import annotations

from typing import Final

HTML_SPACE_CHARACTERS: Final = "\t\n\f\r "
HTML_FORMATTING_SPACE_CHARACTERS: Final = "\t\n\f\r"
HTML_SPACE_OR_TAG_END_CHARACTERS: Final = HTML_SPACE_CHARACTERS + "/>"

# HTML5 spec: Foreign attribute adjustments for SVG/MathML
# Maps lowercase attribute names to (prefix, local_name, namespace_url)
FOREIGN_ATTRIBUTE_ADJUSTMENTS = {
    "xlink:actuate": ("xlink", "actuate", "http://www.w3.org/1999/xlink"),
    "xlink:arcrole": ("xlink", "arcrole", "http://www.w3.org/1999/xlink"),
    "xlink:href": ("xlink", "href", "http://www.w3.org/1999/xlink"),
    "xlink:role": ("xlink", "role", "http://www.w3.org/1999/xlink"),
    "xlink:show": ("xlink", "show", "http://www.w3.org/1999/xlink"),
    "xlink:title": ("xlink", "title", "http://www.w3.org/1999/xlink"),
    "xlink:type": ("xlink", "type", "http://www.w3.org/1999/xlink"),
    "xml:lang": ("xml", "lang", "http://www.w3.org/XML/1998/namespace"),
    "xml:space": ("xml", "space", "http://www.w3.org/XML/1998/namespace"),
    "xmlns": (None, "xmlns", "http://www.w3.org/2000/xmlns/"),
    "xmlns:xlink": ("xmlns", "xlink", "http://www.w3.org/2000/xmlns/"),
}

# MathML attribute case adjustments
MATHML_ATTRIBUTE_ADJUSTMENTS = {
    "definitionurl": "definitionURL",
}

# SVG attribute case adjustments
SVG_ATTRIBUTE_ADJUSTMENTS = {
    "attributename": "attributeName",
    "attributetype": "attributeType",
    "basefrequency": "baseFrequency",
    "baseprofile": "baseProfile",
    "calcmode": "calcMode",
    "clippathunits": "clipPathUnits",
    "diffuseconstant": "diffuseConstant",
    "edgemode": "edgeMode",
    "filterunits": "filterUnits",
    "glyphref": "glyphRef",
    "gradienttransform": "gradientTransform",
    "gradientunits": "gradientUnits",
    "kernelmatrix": "kernelMatrix",
    "kernelunitlength": "kernelUnitLength",
    "keypoints": "keyPoints",
    "keysplines": "keySplines",
    "keytimes": "keyTimes",
    "lengthadjust": "lengthAdjust",
    "limitingconeangle": "limitingConeAngle",
    "markerheight": "markerHeight",
    "markerunits": "markerUnits",
    "markerwidth": "markerWidth",
    "maskcontentunits": "maskContentUnits",
    "maskunits": "maskUnits",
    "numoctaves": "numOctaves",
    "pathlength": "pathLength",
    "patterncontentunits": "patternContentUnits",
    "patterntransform": "patternTransform",
    "patternunits": "patternUnits",
    "pointsatx": "pointsAtX",
    "pointsaty": "pointsAtY",
    "pointsatz": "pointsAtZ",
    "preservealpha": "preserveAlpha",
    "preserveaspectratio": "preserveAspectRatio",
    "primitiveunits": "primitiveUnits",
    "refx": "refX",
    "refy": "refY",
    "repeatcount": "repeatCount",
    "repeatdur": "repeatDur",
    "requiredextensions": "requiredExtensions",
    "requiredfeatures": "requiredFeatures",
    "specularconstant": "specularConstant",
    "specularexponent": "specularExponent",
    "spreadmethod": "spreadMethod",
    "startoffset": "startOffset",
    "stddeviation": "stdDeviation",
    "stitchtiles": "stitchTiles",
    "surfacescale": "surfaceScale",
    "systemlanguage": "systemLanguage",
    "tablevalues": "tableValues",
    "targetx": "targetX",
    "targety": "targetY",
    "textlength": "textLength",
    "viewbox": "viewBox",
    "viewtarget": "viewTarget",
    "xchannelselector": "xChannelSelector",
    "ychannelselector": "yChannelSelector",
    "zoomandpan": "zoomAndPan",
}

# HTML integration points (SVG/MathML elements that allow HTML content)
# Structure: (namespace_url, element_name)
HTML_INTEGRATION_POINT_ELEMENTS = {
    ("http://www.w3.org/1998/Math/MathML", "annotation-xml"),
    ("http://www.w3.org/2000/svg", "foreignObject"),
    ("http://www.w3.org/2000/svg", "desc"),
    ("http://www.w3.org/2000/svg", "title"),
}

# MathML text integration points
# Structure: (namespace_url, element_name)
MATHML_TEXT_INTEGRATION_POINT_ELEMENTS = {
    ("http://www.w3.org/1998/Math/MathML", "mi"),
    ("http://www.w3.org/1998/Math/MathML", "mo"),
    ("http://www.w3.org/1998/Math/MathML", "mn"),
    ("http://www.w3.org/1998/Math/MathML", "ms"),
    ("http://www.w3.org/1998/Math/MathML", "mtext"),
}

QUIRKY_PUBLIC_PREFIXES = (
    "-//advasoft ltd//dtd html 3.0 aswedit + extensions//",
    "-//as//dtd html 3.0 aswedit + extensions//",
    "-//ietf//dtd html 2.0 level 1//",
    "-//ietf//dtd html 2.0 level 2//",
    "-//ietf//dtd html 2.0 strict level 1//",
    "-//ietf//dtd html 2.0 strict level 2//",
    "-//ietf//dtd html 2.0 strict//",
    "-//ietf//dtd html 2.0//",
    "-//ietf//dtd html 2.1e//",
    "-//ietf//dtd html 3.0//",
    "-//ietf//dtd html 3.2 final//",
    "-//ietf//dtd html 3.2//",
    "-//ietf//dtd html 3//",
    "-//ietf//dtd html level 0//",
    "-//ietf//dtd html level 1//",
    "-//ietf//dtd html level 2//",
    "-//ietf//dtd html level 3//",
    "-//ietf//dtd html strict level 0//",
    "-//ietf//dtd html strict level 1//",
    "-//ietf//dtd html strict level 2//",
    "-//ietf//dtd html strict level 3//",
    "-//ietf//dtd html strict//",
    "-//ietf//dtd html//",
    "-//metrius//dtd metrius presentational//",
    "-//microsoft//dtd internet explorer 2.0 html strict//",
    "-//microsoft//dtd internet explorer 2.0 html//",
    "-//microsoft//dtd internet explorer 2.0 tables//",
    "-//microsoft//dtd internet explorer 3.0 html strict//",
    "-//microsoft//dtd internet explorer 3.0 html//",
    "-//microsoft//dtd internet explorer 3.0 tables//",
    "-//netscape comm. corp.//dtd html//",
    "-//netscape comm. corp.//dtd strict html//",
    "-//o'reilly and associates//dtd html 2.0//",
    "-//o'reilly and associates//dtd html extended 1.0//",
    "-//o'reilly and associates//dtd html extended relaxed 1.0//",
    "-//softquad software//dtd hotmetal pro 6.0::19990601::extensions to html 4.0//",
    "-//softquad//dtd hotmetal pro 4.0::19971010::extensions to html 4.0//",
    "-//spyglass//dtd html 2.0 extended//",
    "-//sq//dtd html 2.0 hotmetal + extensions//",
    "-//sun microsystems corp.//dtd hotjava html//",
    "-//sun microsystems corp.//dtd hotjava strict html//",
    "-//w3c//dtd html 3 1995-03-24//",
    "-//w3c//dtd html 3.2 draft//",
    "-//w3c//dtd html 3.2 final//",
    "-//w3c//dtd html 3.2//",
    "-//w3c//dtd html 3.2s draft//",
    "-//w3c//dtd html 4.0 frameset//",
    "-//w3c//dtd html 4.0 transitional//",
    "-//w3c//dtd html experimental 19960712//",
    "-//w3c//dtd html experimental 970421//",
    "-//w3c//dtd html experimental 970421//",
    "-//w3c//dtd w3 html//",
    "-//w3o//dtd w3 html 3.0//",
    "-//webtechs//dtd mozilla html 2.0//",
    "-//webtechs//dtd mozilla html//",
)

QUIRKY_PUBLIC_MATCHES = (
    "-//w3o//dtd w3 html strict 3.0//en//",
    "-/w3c/dtd html 4.0 transitional/en",
    "html",
)

QUIRKY_SYSTEM_MATCHES = ("http://www.ibm.com/data/dtd/v11/ibmxhtml1-transitional.dtd",)

LIMITED_QUIRKY_PUBLIC_PREFIXES = (
    "-//w3c//dtd xhtml 1.0 frameset//",
    "-//w3c//dtd xhtml 1.0 transitional//",
)

HTML4_PUBLIC_PREFIXES = (
    "-//w3c//dtd html 4.01 frameset//",
    "-//w3c//dtd html 4.01 transitional//",
)

HEADING_ELEMENTS = {"h1", "h2", "h3", "h4", "h5", "h6"}

# Elements where pretty-printing and whitespace-collapsing transforms should
# preserve text node whitespace.
WHITESPACE_PRESERVING_ELEMENTS: Final[frozenset[str]] = frozenset(
    {
        "code",
        "pre",
        "script",
        "style",
        "textarea",
    }
)

FORMATTING_ELEMENTS = {
    "a",
    "b",
    "big",
    "code",
    "em",
    "font",
    "i",
    "nobr",
    "s",
    "small",
    "strike",
    "strong",
    "tt",
    "u",
}

SPECIAL_ELEMENTS = {
    "address",
    "applet",
    "area",
    "article",
    "aside",
    "base",
    "basefont",
    "bgsound",
    "blockquote",
    "body",
    "br",
    "button",
    "caption",
    "center",
    "col",
    "colgroup",
    "dd",
    "details",
    "dialog",
    "dir",
    "div",
    "dl",
    "dt",
    "embed",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "frame",
    "frameset",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "head",
    "header",
    "hgroup",
    "hr",
    "html",
    "iframe",
    "img",
    "input",
    "keygen",
    "li",
    "link",
    "listing",
    "main",
    "marquee",
    "menu",
    "meta",
    "nav",
    "noembed",
    "noframes",
    "noscript",
    "object",
    "ol",
    "p",
    "param",
    "plaintext",
    "pre",
    "script",
    "search",
    "section",
    "select",
    "source",
    "style",
    "summary",
    "table",
    "tbody",
    "td",
    "template",
    "textarea",
    "tfoot",
    "th",
    "thead",
    "title",
    "tr",
    "track",
    "ul",
    "wbr",
}

FORMAT_MARKER: Final[object] = object()

DEFAULT_SCOPE_TERMINATORS = {
    "applet",
    "caption",
    "html",
    "table",
    "td",
    "th",
    "marquee",
    "object",
    "template",
}

BUTTON_SCOPE_TERMINATORS = DEFAULT_SCOPE_TERMINATORS | {"button"}
LIST_ITEM_SCOPE_TERMINATORS = DEFAULT_SCOPE_TERMINATORS | {"ol", "ul"}
DEFINITION_SCOPE_TERMINATORS = DEFAULT_SCOPE_TERMINATORS | {"dl"}

TABLE_FOSTER_TARGETS = {"table", "tbody", "tfoot", "thead", "tr"}

SVG_TAG_NAME_ADJUSTMENTS = {
    "altglyph": "altGlyph",
    "altglyphdef": "altGlyphDef",
    "altglyphitem": "altGlyphItem",
    "animatecolor": "animateColor",
    "animatemotion": "animateMotion",
    "animatetransform": "animateTransform",
    "clippath": "clipPath",
    "feblend": "feBlend",
    "fecolormatrix": "feColorMatrix",
    "fecomponenttransfer": "feComponentTransfer",
    "fecomposite": "feComposite",
    "feconvolvematrix": "feConvolveMatrix",
    "fediffuselighting": "feDiffuseLighting",
    "fedisplacementmap": "feDisplacementMap",
    "fedistantlight": "feDistantLight",
    "feflood": "feFlood",
    "fefunca": "feFuncA",
    "fefuncb": "feFuncB",
    "fefuncg": "feFuncG",
    "fefuncr": "feFuncR",
    "fegaussianblur": "feGaussianBlur",
    "feimage": "feImage",
    "femerge": "feMerge",
    "femergenode": "feMergeNode",
    "femorphology": "feMorphology",
    "feoffset": "feOffset",
    "fepointlight": "fePointLight",
    "fespecularlighting": "feSpecularLighting",
    "fespotlight": "feSpotLight",
    "fetile": "feTile",
    "feturbulence": "feTurbulence",
    "foreignobject": "foreignObject",
    "glyphref": "glyphRef",
    "lineargradient": "linearGradient",
    "radialgradient": "radialGradient",
    "textpath": "textPath",
}

FOREIGN_BREAKOUT_ELEMENTS = {
    "b",
    "big",
    "blockquote",
    "body",
    "br",
    "center",
    "code",
    "dd",
    "div",
    "dl",
    "dt",
    "em",
    "embed",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "head",
    "hr",
    "i",
    "img",
    "li",
    "listing",
    "menu",
    "meta",
    "nobr",
    "ol",
    "p",
    "pre",
    "ruby",
    "s",
    "small",
    "span",
    "strong",
    "strike",
    "sub",
    "sup",
    "table",
    "tt",
    "u",
    "ul",
    "var",
}

NAMESPACE_URL_TO_PREFIX = {
    "http://www.w3.org/1999/xhtml": "html",
    "http://www.w3.org/1998/Math/MathML": "math",
    "http://www.w3.org/2000/svg": "svg",
}

HTML_INTEGRATION_POINT_SET = {
    (NAMESPACE_URL_TO_PREFIX.get(ns, ns), name) for ns, name in HTML_INTEGRATION_POINT_ELEMENTS
}

MATHML_TEXT_INTEGRATION_POINT_SET = {
    (NAMESPACE_URL_TO_PREFIX.get(ns, ns), name) for ns, name in MATHML_TEXT_INTEGRATION_POINT_ELEMENTS
}

TABLE_ALLOWED_CHILDREN = {
    "caption",
    "colgroup",
    "tbody",
    "tfoot",
    "thead",
    "tr",
    "td",
    "th",
    "script",
    "template",
    "style",
}

TABLE_SCOPE_TERMINATORS = {"html", "table", "template"}
IMPLIED_END_TAGS = {"dd", "dt", "li", "option", "optgroup", "p", "rb", "rp", "rt", "rtc"}

VOID_ELEMENTS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    },
)
