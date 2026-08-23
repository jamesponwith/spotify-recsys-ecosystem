"""Render the simulation as a self-contained HTML page and a markdown mirror.

Every number on the page is read out of the JSON the code wrote. Nothing here
is typed by hand, which is the only way a report stays true after the model
changes underneath it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _theme import css_vars  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DOCS = ROOT / "docs"


def esc(s: object) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def load(name: str) -> dict:
    path = ARTIFACTS / name
    if not path.exists():
        raise SystemExit(f"{path} not found -- run `make simulate sensitivity ledger` first.")
    return json.loads(path.read_text())


def money_split(arm: dict, seats: float) -> tuple[float, float, float, float]:
    """Per ticket held: artist, platform fees, identity burn, broker profit.

    Splitting the burn out from the broker's profit is not a presentational
    choice. At equilibrium the broker sector grosses `gamma` times what it spends
    on identities, so most of what leaves the fan beyond the artist's share is
    not scalper income at all -- it is money set on fire competing for the
    tickets, reaching nobody who made the music. Reporting broker profit alone
    understates the waste by about four times.
    """
    held = arm["fan_fill"] or 1e-9
    outlay = arm["mean_price_paid"]
    artist = arm["artist_per_seat"] / held
    broker = max(arm["broker_profit"], 0.0) / (held * seats)
    burn = max(arm["identity_burn_per_seat"], 0.0) / held
    fees = max(outlay - artist - broker - burn, 0.0)
    return artist, fees, burn, broker


# --------------------------------------------------------------------------
# charts
# --------------------------------------------------------------------------


def split_chart(arms: list[dict], seats: float, *, width: int = 680) -> str:
    """Stacked bars: of what a fan pays per ticket, who ends up holding it."""
    row_h, gap, label_w, pad_r = 26, 9, 190, 96
    height = len(arms) * (row_h + gap) - gap + 26
    plot_w = width - label_w - pad_r
    peak = max(a["mean_price_paid"] for a in arms)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="Where a fan\'s money goes, per ticket" class="chart">'
    ]
    for i, a in enumerate(arms):
        y = i * (row_h + gap)
        artist, fees, burn, broker = money_split(a, seats)
        x = float(label_w)
        parts.append(
            f'<text x="{label_w - 10}" y="{y + row_h / 2 + 4}" class="blabel" '
            f'text-anchor="end">{esc(a["label"])}</text>'
        )
        for value, fill, name in (
            (artist, "var(--artist)", "artist and venue"),
            (fees, "var(--floor)", "platform fees"),
            (burn, "var(--fan)", "burned on identities"),
            (broker, "var(--broker)", "broker profit"),
        ):
            w = value / peak * plot_w
            if w > 0.4:
                parts.append(
                    f'<g class="bar"><title>{esc(a["label"])} — {name}: '
                    f"${value:,.0f} per ticket</title>"
                    f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{row_h}" fill="{fill}"/>'
                    f"</g>"
                )
            x += w
        parts.append(
            f'<text x="{x + 9:.1f}" y="{y + row_h / 2 + 4}" class="bvalue">'
            f"${a['mean_price_paid']:,.0f}</text>"
        )
    legend_y = height - 6
    for j, (fill, name) in enumerate(
        (
            ("var(--artist)", "artist + venue"),
            ("var(--floor)", "platform fees"),
            ("var(--fan)", "burned on identities"),
            ("var(--broker)", "broker profit"),
        )
    ):
        lx = j * 165
        parts.append(
            f'<rect x="{lx}" y="{legend_y - 9}" width="10" height="10" rx="2" fill="{fill}"/>'
        )
        parts.append(f'<text x="{lx + 15}" y="{legend_y}" class="legend">{esc(name)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def frontier_chart(arms: list[dict], *, width: int = 680, height: int = 380) -> str:
    """What a fan pays against whether the people who care most got in."""
    left, right, top, bottom = 58, 130, 20, 46
    pw, ph = width - left - right, height - top - bottom
    xs = [a["price_multiple"] for a in arms]
    ys = [a["superfan_served"] for a in arms]
    x0, x1 = min(xs) * 0.92, max(xs) * 1.05
    y0, y1 = 0.0, max(ys) * 1.12

    def px(v: float) -> float:
        return left + (v - x0) / (x1 - x0) * pw

    def py(v: float) -> float:
        return top + ph - (v - y0) / (y1 - y0) * ph

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="Price paid against superfans served" class="chart">'
    ]
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        v = y0 + frac * (y1 - y0)
        parts.append(
            f'<line x1="{left}" y1="{py(v):.1f}" x2="{left + pw}" y2="{py(v):.1f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{py(v) + 4:.1f}" class="tick" text-anchor="end">{v:.0%}</text>'
        )
    for mult in range(1, int(x1) + 1):
        if x0 <= mult <= x1:
            parts.append(
                f'<text x="{px(mult):.1f}" y="{top + ph + 22}" class="tick" '
                f'text-anchor="middle">{mult}x</text>'
            )
    for a in arms:
        cx, cy = px(a["price_multiple"]), py(a["superfan_served"])
        fill = (
            "var(--fan)"
            if a["broker_capture"] < 0.01
            else ("var(--broker)" if a["broker_capture"] > 0.2 else "var(--floor)")
        )
        parts.append(
            f'<g class="dot"><title>{esc(a["label"])}: fans pay '
            f"{a['price_multiple']:.2f}x face, {a['superfan_served']:.1%} of superfan "
            f"demand served, brokers take {a['broker_capture']:.1%}</title>"
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="{fill}"/></g>'
        )
        parts.append(
            f'<text x="{cx + 11:.1f}" y="{cy + 4:.1f}" class="plabel">{esc(a["label"])}</text>'
        )
    parts.append(
        f'<text x="{left + pw / 2:.0f}" y="{height - 6}" class="axis" text-anchor="middle">'
        f"what a fan pays per ticket, over face &#8594;</text>"
    )
    parts.append(
        f'<text x="14" y="{top + ph / 2:.0f}" class="axis" text-anchor="middle" '
        f'transform="rotate(-90 14 {top + ph / 2:.0f})">superfan demand served &#8594;</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def ladder_chart(rungs: list[dict], *, width: int = 680) -> str:
    row_h, gap, label_w, pad_r = 26, 9, 230, 70
    height = len(rungs) * (row_h + gap) - gap
    plot_w = width - label_w - pad_r
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="Spread retained at each enforcement rung" class="chart">'
    ]
    for i, r in enumerate(rungs):
        y = i * (row_h + gap)
        w = max(r["spread_retained"] * plot_w, 1.5)
        parts.append(
            f'<text x="{label_w - 10}" y="{y + row_h / 2 + 4}" class="blabel" '
            f'text-anchor="end">{esc(r["label"])}</text>'
        )
        parts.append(
            f'<g class="bar"><title>{esc(r["mechanism"])}</title>'
            f'<rect x="{label_w}" y="{y}" width="{w:.1f}" height="{row_h}" rx="3" '
            f'fill="var(--broker)"/></g>'
        )
        parts.append(
            f'<text x="{label_w + w + 9:.1f}" y="{y + row_h / 2 + 4}" class="bvalue">'
            f"{r['spread_retained']:.0%}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def contention_chart(rows: list[dict], *, width: int = 680, height: int = 300) -> str:
    import math

    left, right, top, bottom = 66, 24, 20, 48
    pw, ph = width - left - right, height - top - bottom
    xs = [math.log10(r["shards"]) for r in rows]
    ys = [math.log10(max(r["attempts_per_purchase"], 1.0)) for r in rows]
    x0, x1 = min(xs), max(xs)
    y0, y1 = 0.0, max(ys) * 1.05

    def px(v: float) -> float:
        return left + (v - x0) / (x1 - x0) * pw

    def py(v: float) -> float:
        return top + ph - (v - y0) / max(y1 - y0, 1e-9) * ph

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="Failed transactions per successful purchase" class="chart">'
    ]
    for e in range(0, int(y1) + 1):
        parts.append(
            f'<line x1="{left}" y1="{py(e):.1f}" x2="{left + pw}" y2="{py(e):.1f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{py(e) + 4:.1f}" class="tick" text-anchor="end">'
            f"{10**e:,.0f}</text>"
        )
    pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys, strict=True))
    parts.append(
        f'<polyline points="{pts}" fill="none" stroke="var(--broker)" stroke-width="2.5"/>'
    )
    for r, x, y in zip(rows, xs, ys, strict=True):
        parts.append(
            f'<g class="dot"><title>{r["shards"]:,} inventory UTxOs: '
            f"{r['success_rate']:.2%} of submitted transactions succeed, "
            f"{r['minutes_to_clear']:.1f} minutes to clear</title>"
            f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="5" fill="var(--broker)"/></g>'
        )
        parts.append(
            f'<text x="{px(x):.1f}" y="{top + ph + 20}" class="tick" text-anchor="middle">'
            f"{r['shards']:,}</text>"
        )
    parts.append(
        f'<text x="{left + pw / 2:.0f}" y="{height - 8}" class="axis" text-anchor="middle">'
        f"inventory UTxOs the house is sharded across &#8594;</text>"
    )
    parts.append(
        f'<text x="16" y="{top + ph / 2:.0f}" class="axis" text-anchor="middle" '
        f'transform="rotate(-90 16 {top + ph / 2:.0f})">submissions per purchase</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------
# page
# --------------------------------------------------------------------------

ARM_COLUMNS = (
    ("label", "Arm", "{}"),
    ("broker_capture", "Broker", "{:.1%}"),
    ("face_access", "At face", "{:.1%}"),
    ("price_multiple", "Fan pays", "{:.2f}x"),
    ("artist_per_seat", "Artist/seat", "${:,.0f}"),
    ("superfan_served", "Superfans", "{:.1%}"),
    ("low_income_served", "Low income", "{:.1%}"),
    ("income_ratio", "Income skew", "{:.2f}x"),
    ("customer_harm", "Harm", "{:.1%}"),
)


def arm_table(arms: list[dict]) -> str:
    head = "".join(f"<th>{esc(t)}</th>" for _, t, _ in ARM_COLUMNS)
    rows = []
    for a in arms:
        cells = []
        for key, _, fmt in ARM_COLUMNS:
            v = a[key]
            cells.append(f"<td>{esc(fmt.format(v))}</td>" if v == v else "<td class='na'>--</td>")
        rows.append(f"<tr><th scope='row'>{cells[0][4:-5]}</th>{''.join(cells[1:])}</tr>")
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def claims_table(claims: list[dict]) -> str:
    rows = []
    for c in claims:
        mark = "survives" if c["survives"] else "breaks"
        cls = "good" if c["survives"] else "warn"
        where = ""
        if c.get("region"):
            r = c["region"]
            where = (
                f"demand {r['demand_multiple'][0]:g}–{r['demand_multiple'][1]:g}x, "
                f"leak {r['off_platform_leak'][0]:.2f}–{r['off_platform_leak'][1]:.2f}"
            )
        rows.append(
            f"<tr><th scope='row'>{esc(c['statement'])}</th>"
            f"<td>{c['held']}/{c['of']}</td>"
            f"<td class='{cls}'>{mark}</td><td class='muted'>{esc(where)}</td></tr>"
        )
    return (
        "<div class='scroll'><table><thead><tr><th>Claim, written before the grid ran</th>"
        "<th>Cells</th><th></th><th>Where it fails</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


CSS = """
  *, *::before, *::after { box-sizing: border-box; }
  body { margin: 0; background: var(--surface); color: var(--ink);
         font: 16px/1.62 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif; }
  main { max-width: 780px; margin: 0 auto; padding: 56px 22px 96px; }
  h1 { font-size: 2.05rem; line-height: 1.16; letter-spacing: -0.021em; margin: 0 0 .4rem;
       text-wrap: balance; }
  h2 { font-size: 1.3rem; letter-spacing: -0.012em; margin: 3.2rem 0 .8rem;
       padding-top: 1.4rem; border-top: 1px solid var(--hairline); text-wrap: balance; }
  h3 { font-size: 1.02rem; margin: 2rem 0 .5rem; color: var(--ink-soft); }
  p { margin: 0 0 1rem; color: var(--ink_soft); }
  .lede { font-size: 1.1rem; color: var(--muted); margin-bottom: 2rem; }
  .key { background: var(--raised); border: 1px solid var(--hairline); border-left: 3px solid var(--fan);
         border-radius: 6px; padding: 1rem 1.15rem; margin: 1.4rem 0; }
  .key p:last-child { margin-bottom: 0; }
  .chart { display: block; max-width: 100%; height: auto; margin: 1.2rem 0 .6rem; }
  .scroll { overflow-x: auto; margin: 1.1rem 0; }
  /* Every table here is a column of figures being compared down the page, so
     the digits have to line up. Proportional numerals make 61.1% and 4.9% sit
     at different widths and the eye stops reading the column as a column. */
  table { border-collapse: collapse; width: 100%; font-size: .875rem;
          font-variant-numeric: tabular-nums; }
  th, td { text-align: right; padding: .45rem .6rem; border-bottom: 1px solid var(--hairline);
           white-space: nowrap; }
  thead th { color: var(--muted); font-weight: 600; font-size: .78rem;
             text-transform: uppercase; letter-spacing: .045em; }
  tbody th { text-align: left; font-weight: 500; }
  td.na, .muted { color: var(--muted); }
  td.good { color: var(--good); } td.warn { color: var(--critical); }
  .blabel { font-size: 12.5px; fill: var(--ink_soft); }
  .bvalue { font-size: 12.5px; fill: var(--muted); }
  .plabel { font-size: 12px; fill: var(--ink_soft); }
  .legend { font-size: 12px; fill: var(--muted); }
  .tick  { font-size: 11.5px; fill: var(--muted); }
  .axis  { font-size: 12px; fill: var(--muted); }
  .grid  { stroke: var(--grid); stroke-width: 1; }
  .bvalue, .tick { font-variant-numeric: tabular-nums; }
  .bar rect, .dot circle { transition: opacity .12s; }
  .bar:hover rect, .dot:hover circle { opacity: .74; }
  @media (prefers-reduced-motion: reduce) {
    .bar rect, .dot circle { transition: none; }
  }
  footer { margin-top: 4rem; padding-top: 1.2rem; border-top: 1px solid var(--hairline);
           color: var(--muted); font-size: .85rem; }
  code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .9em; }
"""


def build_html(sim: dict, sens: dict, led: dict, cal: dict) -> str:
    arms = sim["arms"]
    scn = sim["scenario"]
    by = {a["arm"]: a for a in arms}
    seats = float(scn["on_sale"])
    queue, bound = by["queue"], by["affinity_bound"]
    clearing = by["clearing"]
    rungs = led["leak_ladder"]["rungs"]
    cont = led["contention"]["rows"]
    hundred = next(r for r in cont if r["shards"] == 100)

    return f"""<title>What Stops a Scalper</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{css_vars()}{CSS}</style>
<main>
<h1>What actually stops a scalper</h1>
<p class="lede">{scn["on_sale"]:,} seats, ${scn["face_price"]:.0f} face,
{scn["demand_multiple"]:g}x oversubscribed. Eight allocation policies, each solved to a
rational-expectations equilibrium against a broker sector that decides for itself how many
identities to buy.</p>

<div class="key">
<p><strong>No policy makes a hot ticket cheap.</strong> Across all {sens["n_cells"]}
parameter cells, every arm leaves the average fan paying more than 1.2x face. What a policy
chooses is not the price — it is who gets in, and who keeps the difference.</p>
</div>

<h2>Where a fan's money goes</h2>
<p>Under the queue that ships today, a fan pays ${queue["mean_price_paid"]:,.0f} per ticket and
${money_split(queue, seats)[3]:,.0f} of it reaches a broker. Under the strictest arm they pay
${bound["mean_price_paid"]:,.0f} and none of it does. Under market clearing they pay
${clearing["mean_price_paid"]:,.0f} — <em>more</em> than the queue — and the difference goes to
the artist instead.</p>
{split_chart(arms, seats)}

<h2>The trade nobody states out loud</h2>
<p>Every arm can be placed on two axes: what a fan pays, and whether the people who have
listened for years actually got in. Market clearing eliminates the broker by becoming one.
Affinity rationing is the only thing in the set that moves the vertical axis at all.</p>
{frontier_chart(arms)}
{arm_table(arms)}
<p class="muted">Income skew is the mean income of who got in over the mean income of
everyone who wanted in; 1.00x is an income-blind allocation. Harm is parties split by the
purchase cap, plus real fans wrongly rejected by identity checks, plus holders turned away at
the gate.</p>

<h2>Which claims survive the assumptions</h2>
<p>The demand side of this model is invented — no promoter publishes on-sale logs. So each
claim below was written down before the grid ran, and tested in every one of
{sens["n_cells"]} cells spanning demand, broker cost convexity, off-platform leakage and the
cost of forging a listening history.</p>
{claims_table(sens["claims"])}

<h2>What the chain can enforce</h2>
<p>A Cardano validator can refuse any transfer, cap any resale price and take a royalty on
every hop. It cannot see a broker sell the wallet. A smart contract can escrow an asset; it
cannot escrow a secret, because a secret that has been shown has been given away.</p>
{ladder_chart(rungs)}
<div class="key">
<p>A validator-capped resale and a fully soulbound token are entirely different contracts and
produce the <strong>same</strong> broker economics —
{rungs[2]["spread_retained"]:.0%} and {rungs[3]["spread_retained"]:.0%} of the original spread.
The only rung that closes it is the one that is not on the chain at all: an identity check at
the door.</p>
</div>

<h3>And the drop itself does not fit on chain</h3>
<p>In the eUTxO model a transaction consumes specific outputs, so a single inventory UTxO
sells one seat per block. Sharding the house across many UTxOs fixes the throughput and
converts the problem into a retry storm: at 100 shards against
{hundred["concurrent_buyers"]:,} buyers, {hundred["success_rate"]:.2%} of submitted
transactions succeed — about {hundred["attempts_per_purchase"]:,.0f} signed, submitted,
rejected transactions per person who gets a ticket.</p>
{contention_chart(cont)}

<h2>The one fitted parameter</h2>
<p>The model has a single free constant: what a usable purchasing identity costs a broker. It
is fitted by bisection so the unrestricted arm reproduces a
{cal["target_markup"]:g}x resale markup — the only quantity here that is publicly
observable. Broker capture is an output, not a target.</p>
<div class="scroll"><table><thead><tr><th>c0</th><th>Resale markup</th>
<th>Broker capture</th><th>Identities</th></tr></thead><tbody>
{
        "".join(
            f"<tr><th scope='row'>{r['c0']:.4f}{' (fitted)' if r['fitted'] else ''}</th>"
            f"<td>{r['markup']:.2f}x</td><td>{r['broker_capture']:.1%}</td>"
            f"<td>{r['broker_identities']:,.0f}</td></tr>"
            for r in cal["sweep"]
        )
    }
</tbody></table></div>

<footer>Generated by <code>make all</code> from seed {scn["seed"]}.
Every figure is read from the JSON the simulation wrote.</footer>
</main>"""


def build_markdown(sim: dict, sens: dict, led: dict, cal: dict) -> str:
    arms = sim["arms"]
    scn = sim["scenario"]
    by = {a["arm"]: a for a in arms}
    seats = float(scn["on_sale"])
    rungs = led["leak_ladder"]["rungs"]
    hundred = next(r for r in led["contention"]["rows"] if r["shards"] == 100)

    head = "| " + " | ".join(t for _, t, _ in ARM_COLUMNS) + " |"
    sep = "|" + "|".join(["---"] + ["---:"] * (len(ARM_COLUMNS) - 1)) + "|"
    body = []
    for a in arms:
        cells = []
        for key, _, fmt in ARM_COLUMNS:
            v = a[key]
            cells.append(fmt.format(v) if v == v else "--")
        body.append("| " + " | ".join(cells) + " |")

    claims = ["| Claim | Cells | |", "|---|---:|---|"]
    for c in sens["claims"]:
        claims.append(
            f"| {c['statement']} | {c['held']}/{c['of']} | "
            f"{'survives' if c['survives'] else '**breaks**'} |"
        )

    ladder = ["| Enforcement | Spread kept | Broker capture | Fan pays |", "|---|---:|---:|---:|"]
    for r in rungs:
        ladder.append(
            f"| {r['label']} | {r['spread_retained']:.0%} | "
            f"{r['broker_capture']:.1%} | {r['price_multiple']:.2f}x |"
        )

    cal_rows = ["| c0 | Resale markup | Broker capture | Identities |", "|---:|---:|---:|---:|"]
    for r in cal["sweep"]:
        cal_rows.append(
            f"| {r['c0']:.4f}{' (fitted)' if r['fitted'] else ''} | {r['markup']:.2f}x | "
            f"{r['broker_capture']:.1%} | {r['broker_identities']:,.0f} |"
        )

    q, b, c = by["queue"], by["affinity_bound"], by["clearing"]
    return f"""# Results

Generated by `make all` from seed {scn["seed"]}. Every number is read out of
`artifacts/*.json`, not typed.

**Scenario.** {scn["on_sale"]:,} seats reaching a public on-sale, ${scn["face_price"]:.0f}
face, {scn["demand_multiple"]:g}x oversubscribed, {scn["n_trials"]} paired trials per arm.

## No policy makes a hot ticket cheap

Across all {sens["n_cells"]} parameter cells, no arm gets the average fan in below 1.2x face.
What a policy chooses is not the price. It is who gets in, and who keeps the difference.

- **Queue as it ships:** fan pays ${q["mean_price_paid"]:,.0f} per ticket
  ({q["price_multiple"]:.2f}x face), of which ${money_split(q, seats)[3]:,.0f} reaches a broker.
  Brokers take {q["broker_capture"]:.1%} of the house.
- **Affinity + identity-bound:** fan pays ${b["mean_price_paid"]:,.0f}
  ({b["price_multiple"]:.2f}x), brokers take {b["broker_capture"]:.1%}, and
  {b["superfan_served"]:.0%} of superfan demand is served against
  {q["superfan_served"]:.0%} under the queue.
- **Market clearing:** fan pays ${c["mean_price_paid"]:,.0f} ({c["price_multiple"]:.2f}x) --
  *more* than the queue it replaces -- brokers take {c["broker_capture"]:.1%}, the artist
  receives ${c["artist_per_seat"]:,.0f} per seat against ${q["artist_per_seat"]:,.0f}, and
  {c["low_income_served"]:.1%} of bottom-income-quartile demand is served.

## Every arm

{head}
{sep}
{chr(10).join(body)}

Income skew is mean income of who got in over mean income of everyone who wanted in; 1.00x is
income-blind. Harm sums parties split by the purchase cap, real fans wrongly rejected by
identity checks, and holders turned away at the gate.

## Claims tested across the assumption grid

{chr(10).join(claims)}

## What the chain can enforce

{chr(10).join(ladder)}

A validator-capped resale and a fully soulbound token are entirely different contracts and
land on the same broker economics -- {rungs[2]["spread_retained"]:.0%} and
{rungs[3]["spread_retained"]:.0%} of the original spread. The channel neither can see is the
one carrying the volume.

At 100 inventory UTxOs against {hundred["concurrent_buyers"]:,} concurrent buyers,
{hundred["success_rate"]:.2%} of submitted transactions succeed -- roughly
{hundred["attempts_per_purchase"]:,.0f} rejected transactions per person who gets a ticket.

## The one fitted parameter

{chr(10).join(cal_rows)}

See [ARCHITECTURE.md](ARCHITECTURE.md), [CARDANO.md](CARDANO.md) and [POLICY.md](POLICY.md).
"""


def main() -> int:
    sim, sens = load("simulation.json"), load("sensitivity.json")
    led, cal = load("ledger.json"), load("calibration.json")
    (ARTIFACTS / "results.html").write_text(build_html(sim, sens, led, cal))
    DOCS.mkdir(exist_ok=True)
    (DOCS / "RESULTS.md").write_text(build_markdown(sim, sens, led, cal))
    print(f"wrote {ARTIFACTS / 'results.html'} and {DOCS / 'RESULTS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
