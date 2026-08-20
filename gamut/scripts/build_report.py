"""Render the exposure audit as a self-contained HTML page."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _theme import css_vars  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
ROLE = {"penalty": "var(--penalty)", "cap": "var(--cap)", "floor": "var(--floor)"}


def esc(s: object) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def funnel(report: dict, *, width: int = 660) -> str:
    """Three nested bars: the catalog, what retrieval reaches, what is shown."""
    cat = report["catalog"]
    b = report["baseline"]
    rows = [
        ("Catalog", 1.0, f"{cat['n_tracks']:,} tracks"),
        (
            "Reached by retrieval",
            b["pool"]["track_coverage"],
            f"{b['pool']['track_coverage']:.1%}",
        ),
        ("Shown to the listener", b["track_coverage"], f"{b['track_coverage']:.1%}"),
    ]
    row_h, gap, label_w, pad_r = 30, 10, 190, 130
    height = len(rows) * (row_h + gap) - gap
    plot_w = width - label_w - pad_r
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="Catalog reach funnel" class="chart">'
    ]
    for i, (label, frac, note) in enumerate(rows):
        y = i * (row_h + gap)
        w = max(frac * plot_w, 2.0)
        fill = "var(--floor)" if i == 0 else ("var(--cap)" if i == 1 else "var(--penalty)")
        parts.append(
            f'<text x="{label_w - 12}" y="{y + row_h / 2 + 4}" class="blabel" '
            f'text-anchor="end">{esc(label)}</text>'
        )
        parts.append(
            f'<g class="bar"><title>{esc(label)}: {esc(note)}</title>'
            f'<rect x="{label_w}" y="{y}" width="{w:.1f}" height="{row_h}" rx="4" fill="{fill}"/>'
            f"</g>"
        )
        parts.append(
            f'<text x="{label_w + w + 10:.1f}" y="{y + row_h / 2 + 4}" class="bvalue">{esc(note)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def frontier_chart(report: dict, *, width: int = 660, height: int = 330) -> str:
    """Accuracy against long-tail share, one point per intervention strength."""
    pen = report["frontier"]
    caps = report["artist_caps"]
    left, right, top, bottom = 62, 26, 18, 44
    pw, ph = width - left - right, height - top - bottom

    xs = [p["tail_share"] for p in pen] + [c["tail_share"] for c in caps]
    ys = [p["r_precision"] for p in pen] + [c["r_precision"] for c in caps]
    x0, x1 = min(xs) * 0.98, max(xs) * 1.02
    y0, y1 = min(ys) * 0.9, max(ys) * 1.05

    def px(v: float) -> float:
        return left + (v - x0) / (x1 - x0) * pw

    def py(v: float) -> float:
        return top + ph - (v - y0) / (y1 - y0) * ph

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="Accuracy against long-tail exposure" class="chart">'
    ]
    for t in range(5):
        yv = y0 + (y1 - y0) * t / 4
        y = py(yv)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + pw}" y2="{y:.1f}" stroke="var(--grid)" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" class="tick" text-anchor="end">{yv:.4f}</text>'
        )
    for t in range(5):
        xv = x0 + (x1 - x0) * t / 4
        x = px(xv)
        parts.append(
            f'<text x="{x:.1f}" y="{height - 24}" class="tick" text-anchor="middle">{xv:.0%}</text>'
        )
    parts.append(
        f'<text x="{left + pw / 2:.1f}" y="{height - 6}" class="tick" text-anchor="middle">'
        f"share of recommendations going to the long tail &#8594;</text>"
    )

    for series, role, key, label in (
        (pen, "penalty", "penalty", "Popularity penalty"),
        (caps, "cap", "cap", "Artist cap"),
    ):
        pts = " ".join(f"{px(d['tail_share']):.1f},{py(d['r_precision']):.1f}" for d in series)
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{ROLE[role]}" stroke-width="2" '
            f'stroke-linejoin="round"/>'
        )
        for d in series:
            cx, cy = px(d["tail_share"]), py(d["r_precision"])
            parts.append(
                f'<g class="pt"><title>{label} {d[key]}: {d["tail_share"]:.1%} tail, '
                f"R-prec {d['r_precision']:.4f}, artist gini {d['artist_gini']:.3f}</title>"
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" fill="{ROLE[role]}" '
                f'stroke="var(--surface)" stroke-width="1.5"/>'
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="10" fill="transparent"/></g>'
            )
        last = series[-1]
        parts.append(
            f'<text x="{px(last["tail_share"]) - 6:.1f}" y="{py(last["r_precision"]) + 18:.1f}" '
            f'class="dlabel" text-anchor="end">{label}</text>'
        )
    # the do-nothing point
    base = pen[0]
    parts.append(
        f'<circle cx="{px(base["tail_share"]):.1f}" cy="{py(base["r_precision"]):.1f}" r="7" '
        f'fill="none" stroke="var(--ink)" stroke-width="1.5"/>'
        f'<text x="{px(base["tail_share"]) + 12:.1f}" y="{py(base["r_precision"]) - 8:.1f}" '
        f'class="dlabel">today</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


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
  h3 { font-size: 1.02rem; }
  p { margin: 0; max-width: 68ch; }
  section { display: flex; flex-direction: column; gap: 16px; }
  main { display: flex; flex-direction: column; gap: 54px; }
  header { display: flex; flex-direction: column; gap: 14px; }
  .eyebrow, .tick, .dlabel, .blabel, .bvalue, td.num, th.num, .hero-num, .chip, code, .mono {
    font-family: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, monospace;
  }
  .eyebrow { font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); }
  .lede { font-size: 1.1rem; color: var(--ink_soft); max-width: 64ch; }
  .meta {
    font-size: 0.76rem; color: var(--muted); border-top: 1px solid var(--hairline);
    padding-top: 12px; display: flex; flex-wrap: wrap; gap: 6px 20px;
  }
  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 14px; }
  .stat {
    background: var(--raised); border: 1px solid var(--hairline); border-radius: 10px;
    padding: 18px 20px; display: flex; flex-direction: column; gap: 4px;
  }
  .hero-num { font-size: 2.1rem; line-height: 1.1; font-variant-numeric: tabular-nums;
    letter-spacing: -0.03em; color: var(--penalty); }
  .stat .cap { font-size: 0.78rem; color: var(--muted); }
  .figure { display: flex; flex-direction: column; gap: 12px; }
  .scroll { overflow-x: auto; }
  .chart { display: block; max-width: 100%; height: auto; }
  .chart .tick { font-size: 10.5px; fill: var(--muted); font-variant-numeric: tabular-nums; }
  .chart .dlabel { font-size: 11.5px; fill: var(--ink_soft); }
  .chart .blabel { font-size: 12px; fill: var(--ink_soft); }
  .chart .bvalue { font-size: 12px; fill: var(--ink); font-variant-numeric: tabular-nums; }
  .chart .bar rect { transition: opacity .12s ease; }
  .chart .bar:hover rect { opacity: .8; }
  .legend { display: flex; flex-wrap: wrap; gap: 8px 20px; font-size: 0.8rem; color: var(--muted); }
  .legend span { display: inline-flex; align-items: center; gap: 7px; }
  .swatch { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
  .caption { font-size: 0.83rem; color: var(--muted); max-width: 66ch; }
  table { border-collapse: collapse; width: 100%; font-size: 0.86rem; }
  th, td { text-align: left; padding: 8px 14px 8px 0; border-bottom: 1px solid var(--hairline); }
  th { font-size: 0.7rem; letter-spacing: 0.09em; text-transform: uppercase; color: var(--muted);
    font-weight: 500; white-space: nowrap; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; padding-right: 0; }
  tr:last-child td { border-bottom: none; }
  .two { display: grid; grid-template-columns: 1fr 1fr; gap: 26px; }
  @media (max-width: 720px) { .two { grid-template-columns: 1fr; } }
  .case { display: flex; flex-direction: column; gap: 14px; }
  .qtitle { font-size: 1.1rem; }
  .note { border-left: 2px solid var(--penalty); padding: 2px 0 2px 18px; color: var(--ink_soft); font-size: 0.93rem; }
  code { font-size: 0.86em; background: var(--raised); border: 1px solid var(--hairline);
    border-radius: 4px; padding: 1px 5px; }
  ul { margin: 0; padding-left: 20px; display: flex; flex-direction: column; gap: 8px; }
  li { max-width: 66ch; }
  @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""
)


def channel_table(report: dict) -> str:
    rows = sorted(report["channels"].items(), key=lambda kv: -kv[1]["r_precision"])
    body = "".join(
        f"<tr><td>{esc(name)}</td>"
        f'<td class="num">{c["r_precision"]:.4f}</td>'
        f'<td class="num">{c["track_coverage"]:.2%}</td>'
        f'<td class="num">{c["tail_share"]:.1%}</td>'
        f'<td class="num">{c["artist_gini"]:.3f}</td></tr>'
        for name, c in rows
    )
    return (
        '<div class="scroll"><table><thead><tr><th>Channel</th>'
        '<th class="num">R-precision</th><th class="num">Catalog reach</th>'
        '<th class="num">Long-tail share</th><th class="num">Artist gini</th>'
        "</tr></thead><tbody>" + body + "</tbody></table></div>"
    )


def demo_section(demo: dict) -> str:
    blocks = []
    for case in demo["cases"]:
        cols = []
        for label, side in (
            ("As it ships today", case["before"]),
            (f"Penalty {case['penalty']}", case["after"]),
        ):
            rows = "".join(
                f'<tr><td class="num">{r["rank"]}</td>'
                f"<td>{'<strong>&#43;</strong> ' if r.get('new') else ''}{esc(r['name'])}</td>"
                f"<td>{esc(r['artist'])}</td>"
                f'<td class="num">{r["playlists"]:,}{" &middot; tail" if r["tail"] else ""}</td></tr>'
                for r in side["rows"]
            )
            cols.append(
                f'<div class="figure"><h3>{esc(label)}</h3>'
                f'<p class="caption">{side["tail_share"]:.0%} long tail &middot; '
                f"{side['distinct_artists']} distinct artists &middot; "
                f"median {side['median_playlists']:.0f} playlists</p>"
                f'<div class="scroll"><table><thead><tr><th class="num">#</th><th>Track</th>'
                f'<th>Artist</th><th class="num">Playlists</th></tr></thead>'
                f"<tbody>{rows}</tbody></table></div></div>"
            )
        blocks.append(
            f'<div class="case"><h3 class="qtitle">&ldquo;{esc(case["query"])}&rdquo;</h3>'
            f'<div class="two">{"".join(cols)}</div></div>'
        )
    return "".join(blocks)


def build() -> Path:
    r = json.loads((ARTIFACTS / "audit_report.json").read_text())
    b, cat, cfg = r["baseline"], r["catalog"], r["config"]
    pen, caps = r["frontier"], r["artist_caps"]

    ginis = [x["artist_gini"] for x in pen + caps] + [b["artist_gini"]]
    best = min(
        (p for p in pen if p["r_precision"] >= b["r_precision"] * 0.9),
        key=lambda p: -p["tail_share"],
    )
    tail_gain = best["tail_share"] - b["tail_share"]
    acc_cost = best["r_precision"] / b["r_precision"] - 1

    demo_path = ARTIFACTS / "demo.json"
    demo = json.loads(demo_path.read_text()) if demo_path.exists() else None
    demo_html = demo_section(demo) if demo else ""
    qd = r.get("query_diversity")

    body = f"""
  <div class="wrap">
    <header>
      <span class="eyebrow">Gamut · exposure audit</span>
      <h1>Who actually gets heard</h1>
      <p class="lede">Cadence, Timbre and Segue all measure whether the <em>listener</em>
      was served. None of them asks which artists were served at all. This is that
      measurement, run over Cadence's own held-out queries.</p>
      <div class="meta">
        <span>{cfg["n_queries"]} title-only queries</span>
        <span>{cat["n_tracks"]:,} tracks · {cat["n_artists"]:,} artists</span>
        <span>top {cfg["cut"]} counted as shown</span>
        <span>seed {cfg["seed"]}</span>
      </div>
    </header>

    <main>
      <section>
        <div class="stats">
          <div class="stat">
            <div class="hero-num">{b["top1pct_artist_share"]:.0%}</div>
            <div class="cap">of all exposure goes to the top 1% of artists
            ({round(cat["n_artists"] * 0.01):,} of {cat["n_artists"]:,})</div>
          </div>
          <div class="stat">
            <div class="hero-num">{b["track_coverage"]:.1%}</div>
            <div class="cap">of the catalog is ever shown across all
            {cfg["n_queries"]} queries</div>
          </div>
          <div class="stat">
            <div class="hero-num">{b["artist_gini"]:.3f}</div>
            <div class="cap">artist exposure Gini &mdash; 0 is perfectly even, 1 is
            one artist taking everything</div>
          </div>
          <div class="stat">
            <div class="hero-num">{b["tail_lift"]:.2f}&times;</div>
            <div class="cap">long-tail share relative to the tail's share of the
            catalog &mdash; above 1.0 means the tail is <em>over</em>-served</div>
          </div>
        </div>
      </section>

      <section>
        <h2>The funnel is the story</h2>
        <p>Retrieval reaches {b["pool"]["track_coverage"]:.1%} of the catalog. By the time
        the list is cut to what a listener sees, {b["track_coverage"]:.1%} remains. Most of
        the catalog is not being ranked badly — it is never a candidate at all.</p>
        <div class="figure">
          <div class="scroll">{funnel(r)}</div>
          <p class="caption">Distinct tracks appearing at least once across all
          {cfg["n_queries"]} queries. The catalog is already the filtered one: Cadence
          discards tracks on fewer than four playlists before any of this runs.</p>
        </div>
      </section>

      <section>
        <h2>One surprise, in Cadence's favour</h2>
        <div class="note">{b["tail_share"]:.1%} of what Cadence shows comes from the long
        tail — the bottom half of the catalog by playlist count — against
        {cat["tail_share_of_catalog"]:.0%} of the catalog being tail. A tail lift of
        <strong>{b["tail_lift"]:.2f}&times;</strong>. The system is <em>not</em>
        popularity-biased in the usual sense. Its concentration problem is about
        <em>artists</em>, not about hits.</div>
      </section>

      <section>
        <h2>Any one playlist looks fine</h2>
        <div class="note">A median of <strong>{qd["distinct_artists_per_query"]["median"]:.0f}
        distinct artists</strong> in the {cfg["cut"]} tracks shown per query, and only
        <strong>{qd["single_artist_queries"]} of {qd["n_queries_measured"]}</strong>
        queries return a single artist. Per-playlist diversity is not the problem.</div>
        <p>The concentration is <em>across</em> queries: the same artists reappear for
        different requests. A diversity metric computed per playlist &mdash; which is
        the usual way this gets measured &mdash; cannot see that, which is why the audit
        is run over the whole battery and reported as one distribution.</p>
      </section>

      <section>
        <h2>Which channel concentrates exposure</h2>
        <p>Cadence fuses seven candidate sources. Auditing them separately names the
        trade rather than blaming the system as a whole.</p>
        {channel_table(r)}
        <p class="caption"><code>collaborative</code> and <code>cooccurrence</code> are
        absent: these are title-only cold queries with no seed tracks, so two of the
        seven channels contribute essentially nothing to the case the system exists
        for. The most accurate channel, <code>tag_exact</code>, is also the most
        head-concentrated — its long-tail share is
        {r["channels"]["tag_exact"]["tail_share"]:.1%} against
        {b["tail_share"]:.1%} overall. Accuracy here is bought with concentration.</p>
      </section>

      <section>
        <h2>What it costs to intervene</h2>
        <p>Two knobs, swept over the same cached candidates so the only thing differing
        between conditions is the intervention. A popularity penalty asks <em>is this
        track popular?</em>; an artist cap asks <em>has this artist been heard
        already?</em></p>
        <div class="figure">
          <div class="scroll">{frontier_chart(r)}</div>
          <div class="legend">
            <span><i class="swatch" style="background:var(--penalty)"></i>Popularity penalty</span>
            <span><i class="swatch" style="background:var(--cap)"></i>Artist cap</span>
          </div>
          <p class="caption">Up and to the right is better: more long-tail exposure at
          less accuracy cost. The circled point is the system as it ships.</p>
        </div>
        <div class="note">The best cheap trade on offer: a penalty of
        <strong>{best["penalty"]}</strong> moves long-tail share
        <strong>{b["tail_share"]:.1%} &rarr; {best["tail_share"]:.1%}</strong>
        (+{tail_gain * 100:.1f} points) for <strong>{acc_cost:.1%}</strong> R-precision.
        Whether that is worth paying is a product decision, which is the point of
        measuring it instead of guessing.</div>
      </section>

      <section>
        <h2>Seeing it on one query</h2>
        <p>Two cases, deliberately paired. The first is a theme, where
        concentration is a defect the penalty can fix. The second is an artist's
        name, where concentration is the <em>correct</em> answer and the penalty
        makes the result worse without improving artist diversity at all &mdash;
        it just digs for more obscure tracks by the same artist.</p>
        {demo_html}
        <p class="caption">&#43; marks a track the intervention pulled into the top
        {demo["n"] if demo else 0}. An audit that reported only the first case would
        be recommending a change that breaks the second.</p>
      </section>

      <section>
        <h2>The finding that matters</h2>
        <div class="note">Across <em>every</em> intervention tested — nine penalty
        strengths and four artist caps — artist Gini moves only from
        <strong>{max(ginis):.3f}</strong> to <strong>{min(ginis):.3f}</strong>.
        You cannot re-rank your way out of a retrieval problem.</div>
        <p>Concentration is not being created by the ranking function. It is already
        present in the candidate pool, and re-ordering a hundred candidates cannot
        introduce the {100 - b["pool"]["track_coverage"] * 100:.0f}% of the catalog that
        never became a candidate. The popularity penalty does move long-tail share
        substantially — it works on the metric it targets — and barely touches artist
        equality, because a few artists owning many catalog entries is not a fact about
        any single track's popularity.</p>
        <p>That lines up with Timbre's result from the other direction: 76.6% of
        distinct tracks are filtered out before Cadence's index is even built. The
        exposure ceiling is set upstream of everything measured here.</p>
      </section>

      <section>
        <h2>Honest limits</h2>
        <ul>
          <li>This audits the <strong>retrieval and ranking</strong> stage. Cadence's
          final assembly applies MMR and a two-per-artist cap of its own, so the shipped
          playlist is more diverse than the top-{cfg["cut"]} measured here. The funnel
          numbers are about candidate reach, which is the binding constraint either way.</li>
          <li>Exposure is counted over {cfg["n_queries"]} queries. Coverage grows with
          query volume — the concentration ratios are the stable quantities, the absolute
          coverage percentages are not.</li>
          <li>Popularity is proxied by playlist count in the MPD sample. It is a
          reasonable stand-in for exposure, not a measurement of streams or revenue.</li>
          <li>The interventions are deliberately simple. A calibrated or provider-fair
          re-ranker would likely find better points on this frontier; the contribution
          here is the frontier itself, and a measurement harness to place any future
          method on it.</li>
        </ul>
      </section>

      <section>
        <h2>Reproduce</h2>
        <p><code>gamut collect</code> · <code>gamut audit</code> ·
        <code>python scripts/build_report.py</code></p>
      </section>
    </main>
  </div>"""

    html = f"""<title>Who Gets Heard</title>
<style>{CSS}</style>
{body}"""
    out = ARTIFACTS / "results.html"
    out.write_text(html)
    return out


def build_markdown() -> Path:
    r = json.loads((ARTIFACTS / "audit_report.json").read_text())
    b = r["baseline"]
    lines = [
        "# Gamut audit results",
        "",
        "<!-- Generated by scripts/build_report.py. Do not edit by hand. -->",
        "",
        f"- top 1% of artists take **{b['top1pct_artist_share']:.1%}** of exposure",
        f"- **{b['track_coverage']:.2%}** of the catalog is ever shown "
        f"({b['pool']['track_coverage']:.2%} reaches the candidate pool)",
        f"- artist Gini **{b['artist_gini']:.3f}**, track Gini {b['track_gini']:.3f}",
        f"- long-tail share **{b['tail_share']:.1%}** (lift {b['tail_lift']:.2f}x)",
        "",
        "## Per-channel",
        "",
        "| Channel | R-precision | Catalog reach | Long-tail share | Artist gini |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, c in sorted(r["channels"].items(), key=lambda kv: -kv[1]["r_precision"]):
        lines.append(
            f"| {name} | {c['r_precision']:.4f} | {c['track_coverage']:.2%} | "
            f"{c['tail_share']:.1%} | {c['artist_gini']:.3f} |"
        )
    lines += [
        "",
        "## Popularity-penalty frontier",
        "",
        "| Penalty | Long-tail share | Artist gini | R-precision |",
        "|---:|---:|---:|---:|",
    ]
    for p in r["frontier"]:
        lines.append(
            f"| {p['penalty']} | {p['tail_share']:.1%} | {p['artist_gini']:.3f} | {p['r_precision']:.4f} |"
        )
    lines += [
        "",
        "## Artist-cap frontier",
        "",
        "| Cap | Long-tail share | Artist gini | Artist reach | R-precision |",
        "|---:|---:|---:|---:|---:|",
    ]
    for c in r["artist_caps"]:
        lines.append(
            f"| {c['cap']} | {c['tail_share']:.1%} | {c['artist_gini']:.3f} | "
            f"{c['artist_coverage']:.2%} | {c['r_precision']:.4f} |"
        )
    lines.append("")
    out = ROOT / "docs" / "RESULTS.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines))
    return out


if __name__ == "__main__":
    print(build())
    print(build_markdown())
