"""Design tokens for the Concerto report.

Three roles, because the report only ever asks one question: of the money a fan
hands over, who ends up with it. Blue is the fan, orange the broker, violet the
artist. Blue against orange is the pairing chosen deliberately: it is the one
that stays separable under protanopia and deuteranopia, which the rose/blue used
in Gamut does not manage nearly as well. Violet is a third hue rather than a
tint of either, so the artist's share never reads as a shade of the broker's.

Reference marks -- a face-price line, a break-even rule -- wear muted ink. A
reference is not an identity, and giving it a colour makes the reader look for
a fourth actor that is not there.
"""

TOKENS = {
    "light": {
        "surface": "#F7F7F8",
        "raised": "#FFFFFF",
        "ink": "#191A1D",
        "ink_soft": "#3E4046",
        "muted": "#63666E",
        "hairline": "#E1E2E5",
        "grid": "#ECEDEF",
        "fan": "#1F6F8B",
        "broker": "#C0562A",
        "artist": "#6A4C93",
        "floor": "#8E9199",
        "good": "#2E7D4F",
        "critical": "#B3261E",
    },
    "dark": {
        "surface": "#131418",
        "raised": "#1C1E23",
        "ink": "#E9EAEE",
        "ink_soft": "#C4C6CD",
        "muted": "#94979F",
        "hairline": "#2A2C33",
        "grid": "#22242A",
        "fan": "#4FA3BF",
        "broker": "#E08050",
        "artist": "#A88BD8",
        "floor": "#7E818A",
        "good": "#4CAF7D",
        "critical": "#E5766D",
    },
}


def css_vars() -> str:
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
