"""Design tokens for the results page.

The palette encodes the report's actual argument rather than decorating it: one
hue for what Cadence knows from *playlist history*, one for what Timbre recovers
from *audio content*. Baselines are deliberately not a third hue -- a floor is a
reference mark, not an identity, so it wears muted ink.

Every categorical pair below was run through the dataviz validator
(`validate_palette.js`) against its own surface and passes all six checks:
light  #CE7A0E / #0D8AA0 on #F6F7F9 -- CVD dE 17.0 protan, normal 25.0
dark   #C57F22 / #1E9CB2 on #12171C -- CVD dE 17.9 protan, normal 22.8
"""

TOKENS = {
    "light": {
        "surface": "#F6F7F9",
        "raised": "#FFFFFF",
        "ink": "#161A1F",
        "ink_soft": "#3C4650",
        "muted": "#5C6570",
        "hairline": "#DDE1E6",
        "grid": "#E8EBEF",
        "timbre": "#CE7A0E",
        "cadence": "#0D8AA0",
        "floor": "#8C959F",
        "good": "#2E7D4F",
        "critical": "#B3261E",
    },
    "dark": {
        "surface": "#12171C",
        "raised": "#181E25",
        "ink": "#E6EAEE",
        "ink_soft": "#C2CAD2",
        "muted": "#93A0AC",
        "hairline": "#242C34",
        "grid": "#1E262E",
        "timbre": "#C57F22",
        "cadence": "#1E9CB2",
        "floor": "#7A8794",
        "good": "#4CAF7D",
        "critical": "#E5766D",
    },
}


def css_vars() -> str:
    """Light palette on bare :root, dark redefined in both of its two states.

    The un-stamped document (viewer on "system") only has prefers-color-scheme,
    so the media query must exist; an explicit toggle stamps data-theme, so the
    attribute selector must exist too. Components read tokens only.
    """
    light = "\n".join(f"    --{k}: {v};" for k, v in TOKENS["light"].items())
    dark = "\n".join(f"    --{k}: {v};" for k, v in TOKENS["dark"].items())
    return f""":root {{
{light}
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
{dark}
    }}
  }}
  :root[data-theme="dark"] {{
{dark}
  }}"""
