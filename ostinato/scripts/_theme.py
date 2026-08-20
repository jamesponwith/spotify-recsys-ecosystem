"""Design tokens for the Ostinato simulation page.

Vermilion for the closed loop, indigo for the exposure-aware arm, muted ink for
the organic control -- a control is a reference line, not an identity.

Both pairs pass all six dataviz-validator checks against their own surface:
light  #C74A38 / #4557A8 on #F7F6F4 -- CVD dE 19.2 protan, 30.8 tritan, 27.0 normal
dark   #DD6E5E / #7C8BD8 on #16151A -- CVD dE 18.0 protan, 24.9 tritan, 21.9 normal
"""

TOKENS = {
    "light": {
        "surface": "#F7F6F4",
        "raised": "#FFFFFF",
        "ink": "#1A1815",
        "ink_soft": "#443F39",
        "muted": "#65605A",
        "hairline": "#E2DED8",
        "grid": "#EDE9E4",
        "loop": "#C74A38",
        "aware": "#4557A8",
        "floor": "#948E87",
        "good": "#2E7D4F",
        "critical": "#B3261E",
    },
    "dark": {
        "surface": "#16151A",
        "raised": "#1E1C22",
        "ink": "#ECE9E5",
        "ink_soft": "#C9C4BD",
        "muted": "#9A948C",
        "hairline": "#2B2830",
        "grid": "#232028",
        "loop": "#DD6E5E",
        "aware": "#7C8BD8",
        "floor": "#867F78",
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
