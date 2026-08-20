"""Design tokens for the Segue results page.

The palette encodes the argument: violet for the systems that read prefix
*order*, green for the control that has order destroyed, muted ink for the
order-free baselines -- a floor is a reference mark, not an identity.

Both categorical pairs were run through the dataviz validator against their own
surface and pass all six checks:
light  #8259E0 / #2E8F5E on #F7F7FA -- CVD dE 22.3 deutan, 10.5 tritan, 29.3 normal
dark   #8F6DE6 / #35A06E on #131318 -- CVD dE 20.0 deutan, 11.2 tritan, 27.7 normal
"""

TOKENS = {
    "light": {
        "surface": "#F7F7FA",
        "raised": "#FFFFFF",
        "ink": "#17171D",
        "ink_soft": "#3D3D4A",
        "muted": "#5E5E6E",
        "hairline": "#DEDEE6",
        "grid": "#E9E9F0",
        "segue": "#8259E0",
        "control": "#2E8F5E",
        "floor": "#8E8EA0",
        "good": "#2E7D4F",
        "critical": "#B3261E",
    },
    "dark": {
        "surface": "#131318",
        "raised": "#1A1A22",
        "ink": "#E8E8EF",
        "ink_soft": "#C5C5D2",
        "muted": "#9595A8",
        "hairline": "#26262F",
        "grid": "#20202A",
        "segue": "#8F6DE6",
        "control": "#35A06E",
        "floor": "#7C7C90",
        "good": "#4CAF7D",
        "critical": "#E5766D",
    },
}


def css_vars() -> str:
    """Light on bare :root; dark redefined in both of its two states.

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
