"""Design tokens for the Gamut audit page.

Rose for the popularity-penalty intervention, blue for the artist cap -- the two
knobs the report exists to distinguish. Baselines and catalog reference marks
wear muted ink, because a reference is not an identity.

Both pairs pass all six dataviz-validator checks against their own surface:
light  #C2456B / #3A78B5 on #F8F7F7 -- CVD dE 12.2 protan, 28.9 tritan, 23.5 normal
dark   #D4658A / #4F94D4 on #15141A -- CVD dE 11.9 protan, 27.9 tritan, 21.9 normal
"""

TOKENS = {
    "light": {
        "surface": "#F8F7F7",
        "raised": "#FFFFFF",
        "ink": "#1B181A",
        "ink_soft": "#443E42",
        "muted": "#665F63",
        "hairline": "#E2DEE0",
        "grid": "#EDE9EB",
        "penalty": "#C2456B",
        "cap": "#3A78B5",
        "floor": "#948C90",
        "good": "#2E7D4F",
        "critical": "#B3261E",
    },
    "dark": {
        "surface": "#15141A",
        "raised": "#1D1B23",
        "ink": "#ECE8EC",
        "ink_soft": "#C9C2C8",
        "muted": "#9A9199",
        "hairline": "#2A2730",
        "grid": "#232029",
        "penalty": "#D4658A",
        "cap": "#4F94D4",
        "floor": "#867E86",
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
