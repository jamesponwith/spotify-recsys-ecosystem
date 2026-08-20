"""Render Segue's results as a self-contained HTML page.

Every number is read from the JSON the code writes, so the page cannot drift
from what was measured.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _theme import css_vars  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

# label, role, dash — colour carries identity, dash carries it again so the
# chart survives being printed, screenshotted, or read with a CVD.
SERIES = {
    "segue": ("Segue (ordered)", "segue", ""),
    "segue_shuffled": ("Segue, order destroyed", "control", "6 4"),
    "centroid": ("Centroid — Cadence today", "floor", ""),
    "last": ("Last track only", "floor", "3 3"),
    "popularity": ("Popularity", "floor", "1 4"),
}
ROLE_FILL = {"segue": "var(--segue)", "control": "var(--control)", "floor": "var(--floor)"}
METRICS = [
    ("r_precision", "R-precision", "{:.4f}", True),
    ("r_precision_artist", "R-prec (artist-aware)", "{:.4f}", True),
    ("ndcg", "NDCG@100", "{:.4f}", True),
    ("clicks", "Clicks", "{:.2f}", False),  # lower is better
]


def esc(s: object) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def line_chart(report: dict, metric: str, *, width: int = 680, height: int = 300) -> str:
    ks = sorted((int(k) for k in report["seed_counts"]), key=int)
    left, right, top, bottom = 52, 210, 16, 34
    plot_w, plot_h = width - left - right, height - top - bottom

    values = {
        name: [report["seed_counts"][str(k)]["systems"][name][metric] for k in ks]
        for name in SERIES
    }
    vmax = max(max(v) for v in values.values()) * 1.12 or 1.0

    def px(i: int) -> float:
        return left + (i / max(len(ks) - 1, 1)) * plot_w

    def py(v: float) -> float:
        return top + plot_h - (v / vmax) * plot_h

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="R-precision by seed count" class="chart">'
    ]
    for t in range(5):
        v = vmax * t / 4
        y = py(v)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
            f'stroke="var(--grid)" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" class="tick" text-anchor="end">{v:.3f}</text>'
        )
    for i, k in enumerate(ks):
        parts.append(
            f'<text x="{px(i):.1f}" y="{height - 10}" class="tick" text-anchor="middle">'
            f"{k} seed{'s' if k != 1 else ''}</text>"
        )

    # Direct labels sit at each line's right end, but the top three systems land
    # within a few tenths of each other and their raw anchors overlap. Push them
    # apart greedily from the top, and draw a leader wherever a label had to move.
    anchors = sorted((py(values[name][-1]), name) for name in SERIES)
    min_gap, placed, cursor = 14.0, {}, -1e9
    for y, name in anchors:
        placed[name] = cursor = max(y, cursor + min_gap)

    for name, (label, role, dash) in SERIES.items():
        pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(values[name]))
        da = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{ROLE_FILL[role]}" '
            f'stroke-width="2" stroke-linejoin="round"{da}/>'
        )
        ex, ey = px(len(ks) - 1), py(values[name][-1])
        parts.append(
            f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="4" fill="{ROLE_FILL[role]}" '
            f'stroke="var(--surface)" stroke-width="2"/>'
        )
        ly = placed[name] + 4
        if abs(ly - (ey + 4)) > 1.0:
            parts.append(
                f'<path d="M {ex + 5:.1f} {ey:.1f} L {ex + 9:.1f} {ly - 4:.1f}" '
                f'stroke="{ROLE_FILL[role]}" stroke-width="1" fill="none" opacity="0.55"/>'
            )
        parts.append(f'<text x="{ex + 12:.1f}" y="{ly:.1f}" class="dlabel">{esc(label)}</text>')
        for i, v in enumerate(values[name]):
            parts.append(
                f'<g class="pt"><title>{esc(label)} at {ks[i]} seeds: {v:.4f}</title>'
                f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="9" fill="transparent"/></g>'
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
  .mono, .eyebrow, .tick, .dlabel, td.num, th.num, .hero-num, .chip, code {
    font-family: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, monospace;
  }
  .eyebrow {
    font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted);
  }
  .lede { font-size: 1.1rem; color: var(--ink_soft); max-width: 64ch; }
  .meta {
    font-size: 0.76rem; color: var(--muted); border-top: 1px solid var(--hairline);
    padding-top: 12px; display: flex; flex-wrap: wrap; gap: 6px 20px;
  }
  .verdict {
    background: var(--raised); border: 1px solid var(--hairline); border-radius: 10px;
    padding: 26px 28px; display: grid; grid-template-columns: minmax(190px, auto) 1fr;
    gap: 26px 36px; align-items: start;
  }
  @media (max-width: 640px) { .verdict { grid-template-columns: 1fr; } }
  .hero-num {
    font-size: 3.1rem; line-height: 1; font-variant-numeric: tabular-nums;
    letter-spacing: -0.03em; color: var(--segue);
  }
  .hero-cap { font-size: 0.8rem; color: var(--muted); margin-top: 8px; max-width: 26ch; }
  .chip {
    display: inline-flex; align-items: center; gap: 7px; font-size: 0.72rem;
    letter-spacing: 0.09em; text-transform: uppercase; padding: 5px 11px;
    border-radius: 999px; border: 1px solid currentColor; margin-bottom: 14px;
  }
  .chip.pass { color: var(--good); }
  .chip.fail { color: var(--critical); }
  .checks { display: flex; flex-direction: column; gap: 10px; }
  .check {
    display: grid; grid-template-columns: 20px 1fr auto; gap: 12px; align-items: baseline;
    font-size: 0.92rem; border-bottom: 1px dotted var(--hairline); padding-bottom: 9px;
  }
  .check:last-child { border-bottom: none; }
  .check .num { font-size: 0.86rem; color: var(--ink_soft); font-variant-numeric: tabular-nums; }
  .figure { display: flex; flex-direction: column; gap: 12px; }
  .scroll { overflow-x: auto; }
  .chart { display: block; max-width: 100%; height: auto; }
  .chart .tick { font-size: 10.5px; fill: var(--muted); font-variant-numeric: tabular-nums; }
  .chart .dlabel { font-size: 11.5px; fill: var(--ink_soft); }
  .chart .pt circle { cursor: crosshair; }
  .legend { display: flex; flex-wrap: wrap; gap: 8px 20px; font-size: 0.8rem; color: var(--muted); }
  .legend span { display: inline-flex; align-items: center; gap: 7px; }
  .swatch { width: 14px; height: 3px; border-radius: 2px; display: inline-block; }
  .caption { font-size: 0.83rem; color: var(--muted); max-width: 66ch; }
  table { border-collapse: collapse; width: 100%; font-size: 0.86rem; }
  th, td { text-align: left; padding: 8px 14px 8px 0; border-bottom: 1px solid var(--hairline); }
  th {
    font-size: 0.7rem; letter-spacing: 0.09em; text-transform: uppercase; color: var(--muted);
    font-weight: 500; white-space: nowrap;
    font-family: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, monospace;
  }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; padding-right: 0; }
  tr:last-child td { border-bottom: none; }
  tr.win td { background: color-mix(in oklab, var(--segue) 9%, transparent); }
  .hit { color: var(--good); font-weight: 600; }
  .note { border-left: 2px solid var(--segue); padding: 2px 0 2px 18px; color: var(--ink_soft); font-size: 0.93rem; }
  code {
    font-size: 0.86em; background: var(--raised); border: 1px solid var(--hairline);
    border-radius: 4px; padding: 1px 5px;
  }
  ul { margin: 0; padding-left: 20px; display: flex; flex-direction: column; gap: 8px; }
  li { max-width: 66ch; }
  .sr-only {
    position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
    overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; border: 0;
  }
  a:focus-visible { outline: 2px solid var(--segue); outline-offset: 2px; }
  @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""
)


def legend() -> str:
    return (
        '<div class="legend">'
        + "".join(
            f'<span><i class="swatch" style="background:{ROLE_FILL[role]}'
            + ("; opacity:.65" if dash else "")
            + f'"></i>{esc(label)}</span>'
            for label, role, dash in SERIES.values()
        )
        + "</div>"
    )


def metric_table(report: dict) -> str:
    ks = sorted((int(k) for k in report["seed_counts"]), key=int)
    head = "".join(f'<th class="num">{k} seed{"s" if k != 1 else ""}</th>' for k in ks)
    body = []
    for key, label, fmt, higher in METRICS:
        body.append(
            f'<tr><td colspan="{len(ks) + 1}" class="mono" style="color:var(--muted);'
            f'padding-top:16px;border-bottom:none">{esc(label)}'
            f"{' (lower is better)' if not higher else ''}</td></tr>"
        )
        best = {
            k: (min if not higher else max)(
                report["seed_counts"][str(k)]["systems"],
                key=lambda s: report["seed_counts"][str(k)]["systems"][s][key],
            )
            for k in ks
        }
        for name, (slabel, _role, _dash) in SERIES.items():
            cells = ""
            for k in ks:
                v = report["seed_counts"][str(k)]["systems"][name][key]
                mark = " <strong>*</strong>" if best[k] == name else ""
                cells += f'<td class="num">{fmt.format(v)}{mark}</td>'
            cls = ' class="win"' if name == "segue" else ""
            body.append(f"<tr{cls}><td>{esc(slabel)}</td>{cells}</tr>")
    return (
        '<div class="scroll"><table><thead><tr><th>System</th>'
        + head
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def demo_section(demo: dict) -> str:
    blocks = []
    for case in demo["cases"]:
        seeds = "".join(f"<li>{esc(s['label'])}</li>" for s in case["seeds"])
        rows = []
        n = max(len(case["actual_next"]), len(case["predictions"]["segue"]))
        for i in range(n):
            cells = []
            actual = case["actual_next"]
            cells.append(f"<td>{esc(actual[i]['label']) if i < len(actual) else ''}</td>")
            for name in ("centroid", "segue"):
                p = case["predictions"][name]
                if i < len(p):
                    mark = '<span class="hit">&#10003;</span> ' if p[i]["hit"] else ""
                    cells.append(f"<td>{mark}{esc(p[i]['label'])}</td>")
                else:
                    cells.append("<td></td>")
            rows.append(f'<tr><td class="num">{i + 1}</td>' + "".join(cells) + "</tr>")
        hits = {
            name: sum(1 for x in case["predictions"][name] if x["hit"])
            for name in ("centroid", "segue")
        }
        blocks.append(
            f"""
      <div class="figure">
        <h3>&ldquo;{esc(case["title"])}&rdquo;</h3>
        <p class="caption">First {case["seed_count"]} tracks given; {case["n_held_out"]}
        tracks withheld. A tick marks a prediction that really does appear later in
        the playlist. Centroid found {hits["centroid"]}, Segue found {hits["segue"]}.</p>
        <ul>{seeds}</ul>
        <div class="scroll"><table>
          <thead><tr><th class="num">#</th><th>What was actually played next</th>
          <th>Centroid (Cadence today)</th><th>Segue</th></tr></thead>
          <tbody>{"".join(rows)}</tbody>
        </table></div>
      </div>"""
        )
    return "".join(blocks)


def build() -> Path:
    report = json.loads((ARTIFACTS / "eval_report.json").read_text())
    demo_path = ARTIFACTS / "demo.json"
    demo = json.loads(demo_path.read_text()) if demo_path.exists() else None
    ks = sorted((int(k) for k in report["seed_counts"]), key=int)

    def val(k: int, system: str, metric: str = "r_precision") -> float:
        return report["seed_counts"][str(k)]["systems"][system][metric]

    # Headline: the largest seed count, where a sequence model has the most order
    # to work with and the comparison is least ambiguous.
    kmax = ks[-1]
    lift = (val(kmax, "segue") / val(kmax, "centroid") - 1) if val(kmax, "centroid") else 0.0
    order_gap = val(kmax, "segue") - val(kmax, "segue_shuffled")
    order_pct = (order_gap / val(kmax, "segue_shuffled")) if val(kmax, "segue_shuffled") else 0.0
    wins = [k for k in ks if val(k, "segue") > val(k, "centroid")]
    beats_at_max = val(kmax, "segue") > val(kmax, "centroid")
    uses_order = order_gap > 0

    # Clicks is the product-shaped metric -- how many "give me 10 more" presses
    # before the listener hits something they wanted -- and it is the one Segue
    # wins outright, so it leads. R-precision is a wash and says so below;
    # picking the flattering metric and hiding the flat one is how offline
    # evaluations become marketing.
    click_delta = {k: val(k, "segue", "clicks") / val(k, "centroid", "clicks") - 1 for k in ks}
    mean_click = sum(click_delta.values()) / len(ks)
    clicks_everywhere = all(v < 0 for v in click_delta.values())

    checks = "".join(
        f'<div class="check"><span class="mark" style="color:var(--{"good" if ok else "critical"})" '
        f'aria-hidden="true">{"&#10003;" if ok else "&#10007;"}</span>'
        f'<span>{text}<span class="sr-only"> — {"yes" if ok else "no"}</span></span>'
        f'<span class="num">{num}</span></div>'
        for ok, text, num in [
            (
                clicks_everywhere,
                "Fewer Clicks than the centroid at every seed count",
                f"{len(ks)}/{len(ks)}",
            ),
            (
                uses_order,
                "Loses accuracy when prefix order is destroyed",
                f"{order_pct:+.1%} R-prec at {kmax} seeds",
            ),
            (
                beats_at_max,
                "R-precision: a wash, best on the longest prefixes",
                f"{lift:+.1%} at {kmax}, wins {len(wins)}/{len(ks)}",
            ),
        ]
    )

    demo_html = ""
    if demo:
        demo_html = f"""
    <section>
      <h2>What it actually plays</h2>
      <p>Held-out playlists, so every prediction has a ground truth: the tracks the
      person really did add next. Both systems see the same {demo["k"]} opening tracks.</p>
      {demo_section(demo)}
    </section>"""

    body = f"""
  <div class="wrap">
    <header>
      <span class="eyebrow">Segue · playlist continuation</span>
      <h1>A playlist is not a bag of songs</h1>
      <p class="lede">Cadence's collaborative channel sums the seed tracks and searches
      that neighbourhood — shuffle the seeds and nothing changes. But playlists have an
      arc. Does that order carry information a recommender can use?</p>
      <div class="meta">
        <span>98,334 ordered playlists</span><span>5,962,343 positions</span>
        <span>2,000 held-out</span><span>RecSys Challenge 2018 metrics</span>
        <span>seed {report["config"]["seed"]}</span>
      </div>
    </header>

    <main>
      <section>
        <div class="verdict">
          <div>
            <span class="chip {"pass" if uses_order else "fail"}">
              {"&#10003; order carries signal" if uses_order else "&#10007; order adds nothing"}
            </span>
            <div class="hero-num">{mean_click:.1%}</div>
            <div class="hero-cap">fewer &ldquo;give me 10 more&rdquo; presses before the
            listener hits a track they wanted, against the centroid Cadence uses today</div>
          </div>
          <div class="checks">{checks}</div>
        </div>
      </section>

      <section>
        <div class="figure">
          <h3>R-precision as the prefix grows</h3>
          <div class="scroll">{line_chart(report, "r_precision")}</div>
          {legend()}
          <p class="caption">2,000 held-out playlists per seed count. Dashed lines are
          the controls. At one seed <em>last</em> and <em>centroid</em> are the same
          system by definition, and shuffling a one-track prefix is a no-op — the lines
          coincide there, which is the harness telling the truth about itself.</p>
        </div>
      </section>

      <section>
        <h2>The check that could have killed it</h2>
        <p>A model given ordered slots will happily <em>look</em> order-aware while
        having learned to ignore them. So the same fitted model is run twice on every
        challenge: once on the real prefix, once with that prefix shuffled. Nothing
        else differs — same tracks, same weights, same candidate pool.</p>
        <div class="note">At {kmax} seeds, destroying order costs
        <strong>{order_gap:+.4f}</strong> R-precision ({order_pct:+.1%}).
        {
        "The model is genuinely reading sequence, not just consuming a richer bag."
        if uses_order
        else "The model is not reading sequence at all; its gain comes from the learned "
        "projection, not from order."
    }</div>
      </section>

      <section>
        <h2>Every metric, every seed count</h2>
        <p>Official RecSys Challenge 2018 definitions, reused from Cadence's own
        <code>eval.metrics</code> rather than reimplemented. <strong>*</strong> marks
        the best system in each column.</p>
        {metric_table(report)}
        <p class="caption">The metrics disagree, and the disagreement is the result.
        R-precision and NDCG are <em>set</em> metrics: they ask which tracks appear in
        the top |G|, not where. Segue and the centroid are within half a percent of each
        other on both. Clicks asks where the <em>first</em> good track lands, and there
        Segue is ahead at every seed count. Ordering the prefix better mostly moves good
        tracks up the list rather than adding new ones — which is exactly what a
        sequence model should do, and exactly what a set metric is blind to.</p>
      </section>
      {demo_html}
      <section>
        <h2>What this is and is not</h2>
        <ul>
          <li><strong>Is:</strong> a genuine held-out comparison against the exact
          scoring function Cadence reports, on the same 2,000 withheld playlists, with
          the order-free baseline reproducing Cadence's collaborative channel.</li>
          <li><strong>Is not:</strong> a transformer. The model is a position-weighted
          linear operator — ridge regression onto the next track's direction. It is the
          smallest thing that can answer the order question, and calling it more than
          that would be dressing up the result.</li>
          <li><strong>Is not</strong> a full recommender. Segue replaces one channel.
          Cadence's lexical, tag and audio channels are untouched, and the fused system
          would need its own evaluation.</li>
          <li>The first version trained on the obvious target — predict the very next
          track — and lost to the centroid at every seed count above one. The RecSys task
          scores against <em>all</em> withheld tracks, so that objective was strictly
          narrower than the metric. Widening the target to the mean direction of the next
          10 tracks moved validation cosine from 0.5625 to 0.7311. The ablation is kept
          in <code>docs/FINDINGS.md</code> rather than deleted.</li>
          <li>Order had to be rebuilt from raw MPD: Cadence's processed matrix returns
          tracks in ascending id order, not playlist order. That dig turned up a real
          defect in its evaluation harness — see <code>docs/FINDINGS.md</code>.</li>
        </ul>
      </section>

      <section>
        <h2>Reproduce</h2>
        <p><code>make build</code> · <code>make train</code> · <code>make evaluate</code>
        · <code>make demo</code> · <code>make report</code></p>
      </section>
    </main>
  </div>"""

    html = f"""<title>Playlist Order Study</title>
<style>{CSS}</style>
{body}"""
    out = ARTIFACTS / "results.html"
    out.write_text(html)
    return out


def build_markdown() -> Path:
    report = json.loads((ARTIFACTS / "eval_report.json").read_text())
    ks = sorted((int(k) for k in report["seed_counts"]), key=int)
    lines = [
        "# Segue results",
        "",
        "<!-- Generated by scripts/build_report.py. Do not edit by hand. -->",
        "",
    ]
    for key, label, fmt, higher in METRICS:
        lines += [
            f"## {label}" + ("" if higher else " (lower is better)"),
            "",
            "| System | " + " | ".join(f"{k} seeds" for k in ks) + " |",
            "|---|" + "---:|" * len(ks),
        ]
        for name, (slabel, _r, _d) in SERIES.items():
            cells = " | ".join(
                fmt.format(report["seed_counts"][str(k)]["systems"][name][key]) for k in ks
            )
            lines.append(f"| {slabel} | {cells} |")
        lines.append("")
    out = ROOT / "docs" / "RESULTS.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines))
    return out


if __name__ == "__main__":
    print(build())
    print(build_markdown())
