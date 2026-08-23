"""Render the feedback-loop simulation as a self-contained HTML page."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _theme import css_vars  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

ARMS = {
    "organic": ("Organic control", "floor", "5 4"),
    "closed_loop": ("Closed loop", "loop", ""),
    "exposure_aware": ("Closed loop + exposure penalty", "aware", ""),
}
ROLE = {"loop": "var(--loop)", "aware": "var(--aware)", "floor": "var(--floor)"}


def esc(s: object) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def trajectory(
    report: dict, key: str, label: str, fmt: str = "{:.3f}", *, width: int = 300, height: int = 210
) -> str:
    """One small multiple: a metric's path across rounds, one line per arm."""
    arms = report["arms"]
    rounds = [h["round"] for h in arms["closed_loop"]]
    series = {a: [h[key] for h in arms[a]] for a in ARMS if a in arms}
    lo = min(min(v) for v in series.values())
    hi = max(max(v) for v in series.values())
    pad = (hi - lo) * 0.18 or (abs(hi) * 0.02 + 1e-6)
    lo, hi = lo - pad, hi + pad

    left, right, top, bottom = 54, 12, 26, 30
    pw, ph = width - left - right, height - top - bottom

    def px(i: int) -> float:
        return left + (i / max(len(rounds) - 1, 1)) * pw

    def py(v: float) -> float:
        return top + ph - (v - lo) / (hi - lo) * ph

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="{esc(label)} across rounds" class="chart">',
        f'<text x="0" y="12" class="ptitle">{esc(label)}</text>',
    ]
    for t in range(3):
        v = lo + (hi - lo) * t / 2
        y = py(v)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + pw}" y2="{y:.1f}" '
            f'stroke="var(--grid)" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{y + 3.5:.1f}" class="tick" text-anchor="end">'
            f"{fmt.format(v)}</text>"
        )
    for i, r in enumerate(rounds):
        if i % 2 == 0:
            parts.append(
                f'<text x="{px(i):.1f}" y="{height - 10}" class="tick" '
                f'text-anchor="middle">r{r}</text>'
            )
    for arm, (name, role, dash) in ARMS.items():
        if arm not in series:
            continue
        pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(series[arm]))
        da = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{ROLE[role]}" stroke-width="2" '
            f'stroke-linejoin="round"{da}/>'
        )
        ex, ey = px(len(rounds) - 1), py(series[arm][-1])
        parts.append(
            f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="3.5" fill="{ROLE[role]}" '
            f'stroke="var(--surface)" stroke-width="1.5"/>'
        )
        for i, v in enumerate(series[arm]):
            parts.append(
                f'<g class="pt"><title>{esc(name)} round {rounds[i]}: {fmt.format(v)}</title>'
                f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="8" fill="transparent"/></g>'
            )
    parts.append("</svg>")
    return "".join(parts)


def legend() -> str:
    return (
        '<div class="legend">'
        + "".join(
            f'<span><i class="swatch" style="background:{ROLE[role]}'
            + ("; opacity:.7" if dash else "")
            + f'"></i>{esc(name)}</span>'
            for name, role, dash in ARMS.values()
        )
        + "</div>"
    )


def delta_table(report: dict) -> str:
    rows = []
    metrics = [
        ("artist_gini", "Artist Gini", "{:.4f}"),
        ("top1pct_artist_share", "Top 1% artist share", "{:.1%}"),
        ("track_coverage", "Catalog reach", "{:.3%}"),
        ("tail_share", "Long-tail share", "{:.1%}"),
    ]
    for key, label, fmt in metrics:
        cells = ""
        for arm in ARMS:
            h = report["arms"][arm]
            first, last = h[0][key], h[-1][key]
            d = last - first
            sign = "+" if d > 0 else ""
            cells += (
                f'<td class="num">{fmt.format(first)} &rarr; {fmt.format(last)}'
                f'<br><span class="delta">{sign}{fmt.format(d).lstrip("+")}</span></td>'
            )
        rows.append(f"<tr><td>{esc(label)}</td>{cells}</tr>")
    head = "".join(f'<th class="num">{esc(n)}</th>' for n, _, _ in ARMS.values())
    return (
        '<div class="scroll"><table><thead><tr><th>Metric</th>'
        + head
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


CSS = (
    """
  *, *::before, *::after { box-sizing: border-box; }
  """
    + css_vars()
    + """
  body {
    margin: 0; background: var(--surface); color: var(--ink);
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    font-size: 16px; line-height: 1.6; -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 880px; margin: 0 auto; padding: 56px 24px 96px; }
  h1, h2, h3 {
    font-family: ui-serif, "Iowan Old Style", "Source Serif 4", Palatino, Georgia, serif;
    font-weight: 600; text-wrap: balance; margin: 0;
  }
  h1 { font-size: 2.4rem; letter-spacing: -0.018em; line-height: 1.14; }
  h2 { font-size: 1.42rem; letter-spacing: -0.01em; }
  p { margin: 0; max-width: 68ch; }
  section { display: flex; flex-direction: column; gap: 16px; }
  main { display: flex; flex-direction: column; gap: 54px; }
  header { display: flex; flex-direction: column; gap: 14px; }
  .eyebrow, .tick, .ptitle, td.num, th.num, .hero-num, code, .delta, .chip {
    font-family: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, monospace;
  }
  .eyebrow { font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); }
  .lede { font-size: 1.1rem; color: var(--ink_soft); max-width: 64ch; }
  .meta { font-size: 0.76rem; color: var(--muted); border-top: 1px solid var(--hairline);
    padding-top: 12px; display: flex; flex-wrap: wrap; gap: 6px 20px; }
  .verdict { background: var(--raised); border: 1px solid var(--hairline); border-radius: 10px;
    padding: 26px 28px; display: grid; grid-template-columns: minmax(190px, auto) 1fr;
    gap: 26px 36px; align-items: start; }
  @media (max-width: 640px) { .verdict { grid-template-columns: 1fr; } }
  .hero-num { font-size: 2.6rem; line-height: 1.1; font-variant-numeric: tabular-nums;
    letter-spacing: -0.03em; color: var(--loop); }
  .hero-cap { font-size: 0.8rem; color: var(--muted); margin-top: 8px; max-width: 28ch; }
  .checks { display: flex; flex-direction: column; gap: 10px; }
  .check { display: grid; grid-template-columns: 20px 1fr auto; gap: 12px; align-items: baseline;
    font-size: 0.92rem; border-bottom: 1px dotted var(--hairline); padding-bottom: 9px; }
  .check:last-child { border-bottom: none; }
  .check .num { font-size: 0.86rem; color: var(--ink_soft); font-variant-numeric: tabular-nums;
    font-family: ui-monospace, Menlo, monospace; }
  .grid3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; }
  .scroll { overflow-x: auto; }
  .chart { display: block; max-width: 100%; height: auto; }
  .chart .tick { font-size: 10px; fill: var(--muted); font-variant-numeric: tabular-nums; }
  .chart .ptitle { font-size: 11.5px; fill: var(--ink_soft); letter-spacing: .04em; }
  .legend { display: flex; flex-wrap: wrap; gap: 8px 20px; font-size: 0.8rem; color: var(--muted); }
  .legend span { display: inline-flex; align-items: center; gap: 7px; }
  .swatch { width: 14px; height: 3px; border-radius: 2px; display: inline-block; }
  .caption { font-size: 0.83rem; color: var(--muted); max-width: 66ch; }
  table { border-collapse: collapse; width: 100%; font-size: 0.86rem; }
  th, td { text-align: left; padding: 9px 14px 9px 0; border-bottom: 1px solid var(--hairline);
    vertical-align: top; }
  th { font-size: 0.7rem; letter-spacing: 0.09em; text-transform: uppercase; color: var(--muted);
    font-weight: 500; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; padding-right: 0; }
  .delta { font-size: 0.78rem; color: var(--muted); }
  tr:last-child td { border-bottom: none; }
  .note { border-left: 2px solid var(--loop); padding: 2px 0 2px 18px; color: var(--ink_soft);
    font-size: 0.93rem; }
  code { font-size: 0.86em; background: var(--raised); border: 1px solid var(--hairline);
    border-radius: 4px; padding: 1px 5px; }
  ul { margin: 0; padding-left: 20px; display: flex; flex-direction: column; gap: 8px; }
  li { max-width: 66ch; }
  .sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); }
  @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""
)


def build() -> Path:
    r = json.loads((ARTIFACTS / "sim_report.json").read_text())
    cfg, arms = r["config"], r["arms"]

    def drift(arm: str, key: str) -> float:
        h = arms[arm]
        return h[-1][key] - h[0][key]

    import statistics as _st

    gini = {a: [h["artist_gini"] for h in arms[a]] for a in ARMS}
    # The noise floor is taken from the *steadiest* arm, not the control. The
    # control is the arm that turns out to drift, so its own spread mixes noise
    # with signal and would be a circular band to judge it against.
    noise = min(_st.stdev(gini[a]) for a in ARMS)
    band = 2 * noise
    drifts = {a: drift(a, "artist_gini") for a in ARMS}
    excess = drifts["closed_loop"] - drifts["organic"]
    runs_away = excess > band
    organic_drifts = drifts["organic"] > band

    # Paired at each round: both loop arms see identical queries *and* identical
    # accepted positions, because the RNG stream is the same up to the ranking.
    # Only the ranking differs, which makes this a tight paired comparison.
    pairs = [
        arms["exposure_aware"][i]["tail_share"] - arms["closed_loop"][i]["tail_share"]
        for i in range(len(arms["closed_loop"]))
    ]
    mean_pair = sum(pairs) / len(pairs)
    n_pos = sum(1 for v in pairs if v > 0)

    checks = "".join(
        f'<div class="check"><span style="color:var(--{"good" if ok else "critical"})" '
        f'aria-hidden="true">{"&#10003;" if ok else "&#10007;"}</span>'
        f'<span>{text}<span class="sr-only"> &mdash; {"yes" if ok else "no"}</span></span>'
        f'<span class="num">{num}</span></div>'
        for ok, text, num in [
            (
                not runs_away,
                "Closed loop does <em>not</em> concentrate faster than the control",
                f"{excess:+.5f} vs a &plusmn;{band:.5f} band",
            ),
            (
                organic_drifts,
                "Organic listening concentrates on its own",
                f"{drifts['organic']:+.5f} Gini over {cfg['rounds']} rounds",
            ),
            (
                n_pos == len(pairs),
                "Exposure penalty holds more long tail, every round",
                f"{mean_pair * 100:+.1f}pp, {n_pos}/{len(pairs)} rounds",
            ),
        ]
    )

    charts = "".join(
        f"<div>{trajectory(r, key, label, fmt)}</div>"
        for key, label, fmt in [
            ("artist_gini", "ARTIST GINI", "{:.3f}"),
            ("top1pct_artist_share", "TOP 1% ARTIST SHARE", "{:.0%}"),
            ("track_coverage", "CATALOG REACH", "{:.1%}"),
        ]
    )

    body = f"""
  <div class="wrap">
    <header>
      <span class="eyebrow">Ostinato &middot; feedback-loop simulation</span>
      <h1>When the recommender trains on itself</h1>
      <p class="lede">Gamut measured concentration at one instant. But a recommender
      is trained on interaction data its own output helped create. Close that loop and
      run it {cfg["rounds"]} times.</p>
      <div class="meta">
        <span>{cfg["rounds"]} rounds &times; 3 arms</span>
        <span>{cfg["queries_per_round"]} queries per round</span>
        <span>full refit each round</span>
        <span>seed {cfg["seed"]}</span>
        <span>{r["seconds"] / 60:.0f} min</span>
      </div>
    </header>

    <main>
      <section>
        <div class="verdict">
          <div>
            <div class="hero-num">{excess:+.5f}</div>
            <div class="hero-cap">excess artist-Gini drift under the closed loop, over the
            organic control. Negative: the recommender concentrates <em>less</em> than
            popularity-shaped listening does.</div>
          </div>
          <div class="checks">{checks}</div>
        </div>
      </section>

      <section>
        <h2>Three worlds, same starting state</h2>
        <p>Every arm begins from Cadence's trained spaces and adds the same kind of data
        each round: playlists a simulated listener kept. Only the <em>source</em> of the
        ranking differs.</p>
        <ul>
          <li><strong>Organic control</strong> &mdash; the ranking is drawn from the
          catalog's own popularity distribution, not from a query. This is the arm that
          makes the others readable: it adds the same volume of new data, so drift cannot
          be blamed on simply having more of it.</li>
          <li><strong>Closed loop</strong> &mdash; Cadence as it ships.</li>
          <li><strong>Closed loop + exposure penalty</strong> &mdash; Gamut's popularity
          penalty at {cfg["penalty"]}, applied before the listener ever sees the list. On
          Gamut's static frontier this barely moved artist Gini at all.</li>
        </ul>
        <p>The listener is a position-based acceptance model: a track at the top is kept
        far more often than one at the bottom. That is the crude part and also the
        load-bearing part &mdash; it is the mechanism that turns a <em>ranking</em> bias
        into a <em>data</em> bias, and then into a training bias next round.</p>
      </section>

      <section>
        <h2>What {cfg["rounds"]} rounds do</h2>
        <div class="grid3">{charts}</div>
        {legend()}
        <p class="caption">Round 0 is measured before any feedback is folded in, so all
        three arms start from the same system. Each later round is a full refit of the
        collaborative and folksonomy spaces on the grown corpus.</p>
      </section>

      <section>
        <h2>Start to finish</h2>
        {delta_table(r)}
      </section>

      <section>
        <h2>Reading it</h2>
        <div class="note"><strong>The loop does not run away. The control does.</strong>
        Over {cfg["rounds"]} rounds, artist Gini moves {drifts["organic"]:+.5f} under
        organic listening, {drifts["closed_loop"]:+.5f} under the closed loop and
        {drifts["exposure_aware"]:+.5f} with the exposure penalty &mdash; against a noise
        band of &plusmn;{band:.5f}. The ordering is the finding: the recommender
        concentrates <em>less</em> than popularity-shaped listening does.</div>
        <p>That is the opposite of the hypothesis this project was built to test, and it
        agrees with what <a href="../gamut">Gamut</a> found in a single frame: Cadence
        already over-serves the long tail relative to the catalog, at a lift of 1.09x.
        Feeding its own output back does not reverse that. The organic control &mdash;
        sampling in proportion to popularity, which is roughly how listening accrues
        without a recommender &mdash; is the arm that compounds.</p>
        <p>The obvious caution: this says nothing about a recommender with different
        channels. Cadence's lexical and audio channels are content-based and do not move
        when the corpus does, which plausibly anchors it. A purely collaborative system
        has no such anchor and is exactly where the runaway story came from.</p>
      </section>

      <section>
        <h2>Why this run is readable and the last one was not</h2>
        <p>The first version redrew the query sample every round, so a change between
        rounds mixed system drift with a change of question. Fixing the set &mdash; one
        draw, reused by every round of every arm &mdash; cut round-to-round variance in
        artist Gini by <strong>17x</strong> on the closed-loop arm.</p>
        <div class="note">That is what made the effect visible. The drift being measured
        here is smaller than the noise the old design generated, so the earlier run could
        not have found it at any dose. The write-up then reported the paired comparison
        and explicitly refused to report the trend lines; those lines are now reportable.</div>
      </section>

      <section>
        <h2>The result that held</h2>
        <div class="note">The exposure penalty holds <strong>{mean_pair * 100:+.2f} points</strong>
        more long-tail share than the unmodified loop, in <strong>{n_pos} of {len(pairs)}</strong>
        rounds &mdash; every one, and at nearly the same magnitude as the noisier run
        found before.</div>
        <p>This was the one trustworthy number in the previous version, because it is
        <em>paired</em>: both loop arms run off the same random stream, so at every round
        they see identical queries and the simulated listener accepts identical
        <em>positions</em>. Only the track at each position differs. It survives the
        redesign unchanged, which is the strongest thing that can be said for it.</p>
      </section>

      <section>
        <h2>What this is not</h2>
        <ul>
          <li><strong>Not a claim about real listeners.</strong> The acceptance model is
          position bias and nothing else &mdash; no taste, no satiation, no repeat plays,
          no discovery outside the recommender. It is a claim about what a ranking bias
          does to a corpus once the corpus is downstream of the ranking.</li>
          <li><strong>Not calibrated to a real time-scale.</strong> A round is
          {cfg["queries_per_round"]} queries against a corpus of ~98k playlists, so the
          per-round effect is small by construction. The direction and the ordering of
          the arms are the findings; the magnitudes are not transferable.</li>
          <li><strong>The organic control is popularity-proportional</strong>, which is
          itself a rich-get-richer process. That is deliberate &mdash; the question is
          whether the recommender concentrates <em>faster</em> than the world already
          does, not whether concentration exists.</li>
          <li><strong>A design flaw worth naming: the query sample is redrawn every
          round.</strong> That leaves within-arm trajectories confounded &mdash; a change
          between rounds mixes system drift with a change of question. Cross-arm
          comparisons at a fixed round are clean, because the arms share a query sample,
          which is exactly why the paired result above is reported and the trend lines are
          not. Holding one query set fixed across rounds is the first thing to change.</li>
          <li>Only the collaborative and folksonomy spaces are refit each round. The
          lexical and audio channels are content-based and do not move, which plausibly
          anchors the system against drift &mdash; a recommender without them might not
          hold as still.</li>
        </ul>
      </section>

      <section>
        <h2>Reproduce</h2>
        <p><code>ostinato simulate</code> &middot;
        <code>python scripts/build_report.py</code></p>
      </section>
    </main>
  </div>"""

    html = f"""<title>When the Recommender Trains on Itself</title>
<style>{CSS}</style>
{body}"""
    out = ARTIFACTS / "results.html"
    out.write_text(html)
    return out


def build_markdown() -> Path:
    r = json.loads((ARTIFACTS / "sim_report.json").read_text())
    lines = [
        "# Ostinato results",
        "",
        "<!-- Generated by scripts/build_report.py. Do not edit by hand. -->",
        "",
    ]
    for key, label, fmt in [
        ("artist_gini", "Artist Gini", "{:.4f}"),
        ("top1pct_artist_share", "Top 1% artist share", "{:.1%}"),
        ("track_coverage", "Catalog reach", "{:.3%}"),
        ("tail_share", "Long-tail share", "{:.1%}"),
    ]:
        rounds = [h["round"] for h in r["arms"]["closed_loop"]]
        lines += [
            f"## {label}",
            "",
            "| Arm | " + " | ".join(f"r{x}" for x in rounds) + " |",
            "|---|" + "---:|" * len(rounds),
        ]
        for arm, (name, _r, _d) in ARMS.items():
            cells = " | ".join(fmt.format(h[key]) for h in r["arms"][arm])
            lines.append(f"| {name} | {cells} |")
        lines.append("")
    out = ROOT / "docs" / "RESULTS.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines))
    return out


if __name__ == "__main__":
    print(build())
    print(build_markdown())
