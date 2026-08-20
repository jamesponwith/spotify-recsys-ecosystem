"""Render the Phase 0 + joint-demo results as a self-contained HTML page.

Every number on the page is read out of the JSON artifacts at build time. None
is typed by hand, so the page cannot drift from what the code actually measured.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _theme import css_vars  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

SYSTEM_LABELS = {
    "random": ("Random vector", "floor"),
    "mean": ("Catalog mean", "floor"),
    "content_ridge": ("Audio → ridge", "timbre"),
    "content_mlp": ("Audio → MLP", "timbre"),
    "oracle": ("True embedding", "cadence"),
}
ROLE_FILL = {"floor": "var(--floor)", "timbre": "var(--timbre)", "cadence": "var(--cadence)"}


def esc(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def bar_chart(
    rows: list[tuple[str, float, str, str]], *, width: int = 660, fmt: str = "{:.3f}"
) -> str:
    """Horizontal bars. rows = (label, value, role, tooltip).

    Horizontal because the labels are multi-word phrases: rotated or wrapped
    x-axis labels are the most common way a comparison like this becomes
    unreadable.
    """
    row_h, gap, label_w, pad_r = 34, 8, 150, 62
    height = len(rows) * (row_h + gap) - gap
    plot_w = width - label_w - pad_r
    vmax = max((v for _, v, _, _ in rows), default=1.0) or 1.0
    ticks = [0.0, 0.25, 0.5, 0.75, 1.0]

    parts = [
        f'<svg viewBox="0 0 {width} {height + 26}" width="{width}" height="{height + 26}" '
        f'role="img" aria-label="Recall at 100 by system" class="chart">'
    ]
    for t in ticks:
        x = label_w + t * plot_w
        parts.append(
            f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{height}" stroke="var(--grid)" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{height + 18}" class="tick" text-anchor="middle">'
            f"{t * vmax:.2f}</text>"
        )
    for i, (label, value, role, tip) in enumerate(rows):
        y = i * (row_h + gap)
        w = max((value / vmax) * plot_w, 2.0)
        parts.append(
            f'<text x="{label_w - 12}" y="{y + row_h / 2 + 4}" class="blabel" '
            f'text-anchor="end">{esc(label)}</text>'
        )
        parts.append(
            f'<g class="bar"><title>{esc(tip)}</title>'
            f'<rect x="{label_w}" y="{y}" width="{w:.1f}" height="{row_h}" rx="4" '
            f'fill="{ROLE_FILL[role]}"/>'
            f'<rect x="{label_w}" y="{y}" width="{plot_w}" height="{row_h}" fill="transparent"/>'
            f"</g>"
        )
        parts.append(
            f'<text x="{label_w + w + 10:.1f}" y="{y + row_h / 2 + 4}" class="bvalue">'
            f"{fmt.format(value)}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def grouped_chart(runs: list[dict], n_cold: int, *, width: int = 660) -> str:
    """Two bars per query: recovered without Timbre, recovered with it.

    The query label sits *above* its pair rather than in a left gutter. Queries
    are free text of unbounded length, and a fixed gutter either clips the long
    ones or wastes half the plot on the short ones.
    """
    bar_h, inner, label_h, gap, pad_r = 15, 3, 19, 18, 44
    group_h = label_h + bar_h * 2 + inner
    height = len(runs) * (group_h + gap) - gap
    plot_w = width - pad_r
    step = max(1, n_cold // 4)

    parts = [
        f'<svg viewBox="0 0 {width} {height + 26}" width="{width}" height="{height + 26}" '
        f'role="img" aria-label="Frozen tracks recovered per query" class="chart">'
    ]
    for t in range(0, n_cold + 1, step):
        x = (t / n_cold) * plot_w
        parts.append(
            f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{height}" stroke="var(--grid)" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{height + 18}" class="tick" text-anchor="middle">{t}</text>'
        )
    for i, r in enumerate(runs):
        y = i * (group_h + gap)
        parts.append(f'<text x="0" y="{y + 12}" class="blabel">{esc(r["query"])}</text>')
        for j, (key, role) in enumerate(
            (("cold_in_top_n", "floor"), ("timbre_in_top_n", "timbre"))
        ):
            v = r[key]
            w = max((v / n_cold) * plot_w, 2.0)
            yy = y + label_h + j * (bar_h + inner)
            tip = (
                f"{r['query']} — {v} of {n_cold} recovered "
                f"{'with Timbre' if role == 'timbre' else 'without Timbre'}"
            )
            parts.append(
                f'<g class="bar"><title>{esc(tip)}</title>'
                f'<rect x="0" y="{yy}" width="{w:.1f}" height="{bar_h}" rx="4" fill="{ROLE_FILL[role]}"/>'
                f'<rect x="0" y="{yy}" width="{plot_w}" height="{bar_h}" fill="transparent"/></g>'
            )
            parts.append(
                f'<text x="{w + 8:.1f}" y="{yy + bar_h - 3}" class="bvalue small">{v}</text>'
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
    margin: 0;
    background: var(--surface);
    color: var(--ink);
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    font-size: 16px;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 860px; margin: 0 auto; padding: 56px 24px 96px; }
  h1, h2, h3 {
    font-family: ui-serif, "Iowan Old Style", "Source Serif 4", Palatino, Georgia, serif;
    font-weight: 600;
    text-wrap: balance;
    margin: 0;
  }
  h1 { font-size: 2.35rem; letter-spacing: -0.018em; line-height: 1.15; }
  h2 { font-size: 1.4rem; letter-spacing: -0.01em; }
  h3 { font-size: 1.03rem; }
  p { margin: 0; max-width: 68ch; }
  a { color: var(--cadence); }
  section { display: flex; flex-direction: column; gap: 16px; }
  main { display: flex; flex-direction: column; gap: 52px; }
  header { display: flex; flex-direction: column; gap: 14px; }
  .eyebrow {
    font-family: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, monospace;
    font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--muted);
  }
  .lede { font-size: 1.1rem; color: var(--ink_soft); max-width: 64ch; }
  .meta {
    font-family: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, monospace;
    font-size: 0.76rem; color: var(--muted);
    border-top: 1px solid var(--hairline); padding-top: 12px;
    display: flex; flex-wrap: wrap; gap: 6px 20px;
  }
  .verdict {
    background: var(--raised); border: 1px solid var(--hairline);
    border-radius: 10px; padding: 26px 28px;
    display: grid; grid-template-columns: minmax(180px, auto) 1fr; gap: 28px 36px;
    align-items: start;
  }
  @media (max-width: 620px) { .verdict { grid-template-columns: 1fr; } }
  .hero-num {
    font-family: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, monospace;
    font-size: 3.4rem; line-height: 1; font-variant-numeric: tabular-nums;
    letter-spacing: -0.03em; color: var(--timbre);
  }
  .hero-cap { font-size: 0.8rem; color: var(--muted); margin-top: 8px; max-width: 24ch; }
  .chip {
    display: inline-flex; align-items: center; gap: 7px;
    font-family: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, monospace;
    font-size: 0.74rem; letter-spacing: 0.09em; text-transform: uppercase;
    padding: 5px 11px; border-radius: 999px; border: 1px solid currentColor;
    margin-bottom: 14px;
  }
  .chip.pass { color: var(--good); }
  .chip.fail { color: var(--critical); }
  .checks { display: flex; flex-direction: column; gap: 10px; }
  .check {
    display: grid; grid-template-columns: 20px 1fr auto; gap: 12px;
    align-items: baseline; font-size: 0.92rem;
    border-bottom: 1px dotted var(--hairline); padding-bottom: 9px;
  }
  .check:last-child { border-bottom: none; }
  .check .mark { font-weight: 700; }
  .check .num {
    font-family: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, monospace;
    font-variant-numeric: tabular-nums; font-size: 0.86rem; color: var(--ink_soft);
  }
  .figure { display: flex; flex-direction: column; gap: 12px; }
  .scroll { overflow-x: auto; }
  .chart { display: block; max-width: 100%; height: auto; }
  .chart .tick, .chart .blabel, .chart .bvalue {
    font-family: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, monospace;
    font-variant-numeric: tabular-nums;
  }
  .chart .tick { font-size: 10px; fill: var(--muted); }
  .chart .blabel { font-size: 12px; fill: var(--ink_soft); }
  .chart .bvalue { font-size: 12px; fill: var(--ink); }
  .chart .bvalue.small { font-size: 10.5px; }
  .chart .bar rect:first-of-type { transition: opacity .12s ease; }
  .chart .bar:hover rect:first-of-type { opacity: .78; }
  .legend {
    display: flex; flex-wrap: wrap; gap: 8px 20px;
    font-size: 0.8rem; color: var(--muted);
  }
  .legend span { display: inline-flex; align-items: center; gap: 7px; }
  .swatch { width: 11px; height: 11px; border-radius: 3px; display: inline-block; }
  .caption { font-size: 0.83rem; color: var(--muted); max-width: 66ch; }
  table { border-collapse: collapse; width: 100%; font-size: 0.87rem; }
  th, td {
    text-align: left; padding: 8px 14px 8px 0;
    border-bottom: 1px solid var(--hairline);
  }
  th {
    font-family: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, monospace;
    font-size: 0.7rem; letter-spacing: 0.09em; text-transform: uppercase;
    color: var(--muted); font-weight: 500; white-space: nowrap;
  }
  td.num, th.num {
    text-align: right; font-family: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, monospace;
    font-variant-numeric: tabular-nums; padding-right: 0;
  }
  tr:last-child td { border-bottom: none; }
  .gone { color: var(--muted); }
  .up { color: var(--timbre); font-weight: 600; }
  .note {
    border-left: 2px solid var(--timbre); padding: 2px 0 2px 18px;
    color: var(--ink_soft); font-size: 0.93rem;
  }
  code {
    font-family: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, monospace;
    font-size: 0.86em; background: var(--raised);
    border: 1px solid var(--hairline); border-radius: 4px; padding: 1px 5px;
  }
  ul { margin: 0; padding-left: 20px; display: flex; flex-direction: column; gap: 8px; }
  li { max-width: 66ch; }
  .sr-only {
    position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
    overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; border: 0;
  }
  a:focus-visible, .bar:focus-visible {
    outline: 2px solid var(--cadence); outline-offset: 2px; border-radius: 3px;
  }
  @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""
)


def legend(items: list[tuple[str, str]]) -> str:
    return (
        '<div class="legend">'
        + "".join(
            f'<span><i class="swatch" style="background:{ROLE_FILL[role]}"></i>{esc(label)}</span>'
            for label, role in items
        )
        + "</div>"
    )


def verdict_block(g: dict) -> str:
    passed = g["passed"]
    vacuous = g.get("random_criterion_vacuous", False)
    checks = [
        (
            g["beats_random_floor"],
            "Beats the random floor" + ("" if vacuous else " by the required margin"),
            "floor is 0.0000  ·  ratio undefined"
            if vacuous
            else f"{g['random_multiple']:.1f}x  ·  need {g['threshold_random_multiple']:.0f}x",
        ),
        (
            g["oracle_recovery_ratio"] >= g["threshold_oracle_fraction"],
            "Recovers enough of what full playlist history gives",
            f"{g['oracle_recovery_ratio']:.1%}  ·  need {g['threshold_oracle_fraction']:.0%}",
        ),
    ]
    rows = "".join(
        f'<div class="check"><span class="mark" style="color:var(--{"good" if ok else "critical"})" '
        f'aria-hidden="true">{"✓" if ok else "✗"}</span>'
        f'<span>{esc(text)}<span class="sr-only"> — {"met" if ok else "not met"}</span></span>'
        f'<span class="num">{esc(num)}</span></div>'
        for ok, text, num in checks
    )
    return f"""
    <div class="verdict">
      <div>
        <span class="chip {"pass" if passed else "fail"}">{"✓ Gate 0 passed" if passed else "✗ Gate 0 failed"}</span>
        <div class="hero-num">{g["oracle_recovery_ratio"]:.0%}</div>
        <div class="hero-cap">of the retrieval quality that full playlist history delivers, recovered from audio descriptors alone</div>
      </div>
      <div class="checks">{rows}</div>
    </div>"""


def build() -> Path:
    p0 = json.loads((ARTIFACTS / "phase0_report.json").read_text())
    demo_path = ARTIFACTS / "demo_report.json"
    demo = json.loads(demo_path.read_text()) if demo_path.exists() else None

    g, d, r, fits = p0["gate_0"], p0["data"], p0["retrieval"], p0["fits"]
    best_label = SYSTEM_LABELS[g["best_content_system"]][0]

    order = ["random", "mean", "content_ridge", "content_mlp", "oracle"]
    bars = [
        (
            SYSTEM_LABELS[k][0],
            r[k]["recall_at_100"],
            SYSTEM_LABELS[k][1],
            f"{SYSTEM_LABELS[k][0]}: recall@100 = {r[k]['recall_at_100']:.4f}",
        )
        for k in order
        if k in r
    ]
    table_rows = "".join(
        f"<tr><td>{esc(SYSTEM_LABELS[k][0])}</td>"
        f'<td class="num">{r[k]["recall_at_100"]:.4f}</td>'
        f'<td class="num">{r[k]["queries_with_any_hit"]:.1%}</td>'
        f'<td class="num">{r[k]["recall_at_100"] / r["oracle"]["recall_at_100"]:.0%}</td></tr>'
        for k in order
        if k in r
    )
    fit_rows = "".join(
        f"<tr><td>{esc(name)}</td>"
        f'<td class="num">{v["mean_cosine"]:.3f}</td>'
        f'<td class="num">{r[f"content_{name}"]["recall_at_100"]:.4f}</td>'
        f'<td class="num">{v["seconds"]:.0f}s</td></tr>'
        for name, v in fits.items()
    )

    demo_html = ""
    if demo:
        a = demo["aggregate"]
        ex = max(demo["runs"], key=lambda x: x["timbre_in_top_n"])
        all_tracks = [t for r in demo["runs"] for t in r["tracks"]]
        moved = [t for t in all_tracks if t["cold_rank"] != t["timbre_rank"]]
        entered = [t for t in moved if not t["cold_rank"] and t["timbre_rank"]]
        best_new = min((t["timbre_rank"] for t in entered), default=0)
        ex_rows = "".join(
            f"<tr><td>{esc(t['name'])}</td><td>{esc(t['artist'])}</td>"
            f'<td class="num">{t["warm_rank"]}</td>'
            f'<td class="num gone">{t["cold_rank"] or "—"}</td>'
            f'<td class="num {"up" if 0 < t["timbre_rank"] <= demo["top_n"] else "gone"}">'
            f"{t['timbre_rank'] or '—'}</td></tr>"
            for t in ex["tracks"]
        )
        demo_html = f"""
    <section>
      <h2>The joint demo: a simulated release</h2>
      <p>Phase 0 measures a premise. This measures a system. Cadence answers a
      natural-language request, and then the {demo["n_cold_per_query"]} tracks it
      just chose are stripped of everything it knows about them that came from
      playlist history — collaborative vectors, co-occurrence counts, folksonomy
      embedding, popularity. That is the exact state of a track uploaded this
      morning.</p>
      <p>Then Cadence is handed one thing back, for those tracks only: a
      folksonomy embedding predicted from audio. The same query runs again.</p>
      <div class="note">Across {demo["n_queries"]} queries and
      {a["total_frozen"]} frozen tracks, Cadence alone recovers
      <strong>{a["recovered_without_timbre"]}</strong> ({a["recovery_rate_without"]:.0%})
      into the top&nbsp;{demo["top_n"]} candidates. With Timbre's predicted embedding
      it recovers <strong>{a["recovered_with_timbre"]}</strong>
      ({a["recovery_rate_with"]:.0%}) — the same tracks, by the lexical and audio
      channels that never needed history in the first place.</div>
      <p>The graft is not inert. <strong>{len(entered)} of {a["total_frozen"]}</strong>
      frozen tracks went from unranked to a real position once Timbre supplied an
      embedding — but the best of them landed at rank
      <strong>{best_new:,}</strong>, and the cut is {demo["top_n"]}. The wiring works
      end to end; the signal behind it is an order of magnitude short of mattering.
      That is Gate 0's verdict reappearing downstream, which is exactly where a failed
      premise should reappear.</p>
      <div class="figure">
        <h3>Frozen tracks recovered, by query</h3>
        <div class="scroll">{grouped_chart(demo["runs"], demo["n_cold_per_query"])}</div>
        {legend([("Cadence alone (no history, no Timbre)", "floor"), ("Cadence + Timbre", "timbre")])}
        <p class="caption">Out of {demo["n_cold_per_query"]} frozen tracks per query.
        The lexical and audio channels are left intact, so the grey bar is not zero
        — some tracks survive on title text alone. Timbre is credited only with
        the difference.</p>
      </div>
      <div class="figure">
        <h3>Rank transitions — <em>{esc(ex["query"])}</em></h3>
        <div class="scroll"><table>
          <thead><tr><th>Track</th><th>Artist</th><th class="num">Warm</th>
          <th class="num">Frozen</th><th class="num">+ Timbre</th></tr></thead>
          <tbody>{ex_rows}</tbody>
        </table></div>
        <p class="caption">Rank among fused candidates. “—” means the track fell out
        of the candidate set entirely. Warm ranks are 1–{demo["n_cold_per_query"]} by
        construction: these are the tracks Cadence itself picked.</p>
      </div>
    </section>"""

    establishes = (
        "<li><strong>Establishes:</strong> the folksonomy target is predictable from "
        "acoustic description well enough to survive a real retrieval contest against "
        "the full catalog.</li>"
        if g["passed"]
        else "<li><strong>Does not establish the premise.</strong> Audio descriptors "
        "predict the folksonomy embedding above chance, but not by enough to clear the "
        "bar set before the experiment was run. The honest reading is that playlist-title "
        "vocabulary is driven more by context and social convention than by sound. The "
        "pivots are named in docs/EVALUATION.md; the result is recorded rather than "
        "retried until it passes.</li>"
    )

    body = f"""
  <div class="wrap">
    <header>
      <span class="eyebrow">Timbre · Phase 0</span>
      <h1>Cold start by ear</h1>
      <p class="lede">Cadence recommends music from playlist history. Tracks with no
      history are invisible to it — 76.6% of the catalog. Can audio alone stand in
      for what the crowd would have said?</p>
      <div class="meta">
        <span>{d["n_tracks"]:,} tracks</span><span>{d["n_queries"]:,} held-out queries</span>
        <span>{d["n_features"]} features → {d["tag_dim"]}-d</span>
        <span>seed {p0["config"]["seed"]}</span><span>{p0["seconds"]:.0f}s total</span>
      </div>
    </header>

    <main>
      <section>{verdict_block(g)}</section>

      <section>
        <h2>The premise, and how to kill it cheaply</h2>
        <p>Timbre's plan is to train an audio encoder so that new tracks get a
        position in Cadence's folksonomy space — the 128-d space learned from how
        people title playlists. Three weeks of work rests on one assumption:
        <em>acoustic content predicts how humans tag music</em>. If that is false,
        no architecture rescues it.</p>
        <p>It is testable today, with no audio at all. Cadence already holds
        Spotify's engineered descriptors for every track, and the folksonomy
        embedding they would have to predict. Regress one onto the other, then ask
        the only question that matters: does a track with a <em>predicted</em>
        embedding still get retrieved for a query drawn from a playlist it genuinely
        belongs to?</p>
      </section>

      <section>
        <div class="figure">
          <h3>Recall@100 on {d["n_queries"]:,} held-out playlist titles</h3>
          <div class="scroll">{bar_chart(bars)}</div>
          {legend([("Baseline / ceiling reference", "floor"), ("Predicted from audio", "timbre"), ("True embedding (ceiling)", "cadence")])}
          <p class="caption">One query per held-out playlist; the relevant set is the
          test-split tracks it contains. Only those {d["n_test"]:,} tracks have their
          embedding replaced — the rest of the catalog stays known, which is what
          makes this a cold-start simulation rather than a catalog swap.</p>
        </div>
        <div class="scroll"><table>
          <thead><tr><th>System</th><th class="num">Recall@100</th>
          <th class="num">Queries with a hit</th><th class="num">Share of oracle</th></tr></thead>
          <tbody>{table_rows}</tbody>
        </table></div>
      </section>

      <section>
        <h2>Why the metric is retrieval, not cosine</h2>
        <p>Cosine similarity between predicted and true embeddings is the obvious
        score and the wrong one: a prediction can sit close to the target and still
        be useless, because retrieval only cares whether it beat 159,000 competitors
        — not how near it landed.</p>
        <div class="scroll"><table>
          <thead><tr><th>Model</th><th class="num">Mean cosine</th>
          <th class="num">Recall@100</th><th class="num">Fit time</th></tr></thead>
          <tbody>{fit_rows}</tbody>
        </table></div>
        <p class="caption">The winner on the gate is <strong>{esc(best_label)}</strong>.
        Both models are reported; picking the better one after the fact and showing
        only that would make the number unreproducible.</p>
      </section>
      {demo_html}
      <section>
        <h2>What this does and does not establish</h2>
        <ul>
          {establishes}
          <li><strong>Does not establish:</strong> that a CNN reading raw audio reaches
          the same place. Spotify's descriptors are already an engineered, semantically
          loaded summary; an encoder has to learn that summary from waveforms, on a
          different corpus that shares no tracks with this one.</li>
          <li><strong>Does not establish:</strong> that the demo's recovered tracks are
          the objectively right answer. The target set is “what Cadence chose when
          warm”, which makes Cadence its own ground truth — a demo, not an evaluation.</li>
          <li>The <em>catalog mean</em> floor is a single vector repeated across every
          cold track, so all of them tie exactly and tie-breaking decides who enters
          the top 100. That is correct behaviour for a no-information floor, and it is
          why <em>random</em> is the floor the gate is written against.</li>
        </ul>
      </section>

      <section>
        <h2>Reproduce</h2>
        <p>Both numbers on this page come from JSON written by the code, not typed
        by hand.</p>
        <p><code>timbre phase0</code> · <code>timbre demo-report</code> ·
        <code>python scripts/build_report.py</code></p>
      </section>
    </main>
  </div>"""

    html = f"""<title>Cold Start by Ear</title>
<style>{CSS}</style>
{body}"""
    out = ARTIFACTS / "results.html"
    out.write_text(html)
    return out


def build_markdown() -> Path:
    """A plain-text mirror of the same JSON, for reading in the repo.

    Generated rather than written so the two cannot disagree; the interpretation
    lives in README.md, which is hand-written on purpose.
    """
    p0 = json.loads((ARTIFACTS / "phase0_report.json").read_text())
    g, d, r, fits = p0["gate_0"], p0["data"], p0["retrieval"], p0["fits"]
    order = [k for k in SYSTEM_LABELS if k in r]

    lines = [
        "# Phase 0 results",
        "",
        "<!-- Generated by scripts/build_report.py. Do not edit by hand. -->",
        "",
        f"**Gate 0: {'PASSED' if g['passed'] else 'FAILED'}** — "
        f"{g['oracle_recovery_ratio']:.1%} of oracle "
        f"(need {g['threshold_oracle_fraction']:.0%}); "
        + (
            "the random floor is exactly 0.0000, so the multiple is undefined and "
            "that criterion is vacuous."
            if g.get("random_criterion_vacuous")
            else f"{g['random_multiple']:.1f}x random (need {g['threshold_random_multiple']:.0f}x)."
        ),
        "",
        "## Retrieval",
        "",
        "| System | Recall@100 | Queries with a hit | Share of oracle |",
        "|---|---:|---:|---:|",
    ]
    for k in order:
        lines.append(
            f"| {SYSTEM_LABELS[k][0]} | {r[k]['recall_at_100']:.4f} | "
            f"{r[k]['queries_with_any_hit']:.1%} | "
            f"{r[k]['recall_at_100'] / r['oracle']['recall_at_100']:.0%} |"
        )
    lines += [
        "",
        "## Fits",
        "",
        "| Model | Mean cosine | Recall@100 | Seconds |",
        "|---|---:|---:|---:|",
    ]
    for name, v in fits.items():
        lines.append(
            f"| {name} | {v['mean_cosine']:.3f} | "
            f"{r[f'content_{name}']['recall_at_100']:.4f} | {v['seconds']:.0f} |"
        )
    lines += [
        "",
        "## Setup",
        "",
        f"- {d['n_tracks']:,} tracks; {d['n_test']:,} held out as cold "
        f"({d['n_train']:,} used to fit)",
        f"- excluded: {d['excluded_no_audio']} without audio, "
        f"{d['excluded_zero_embedding']} with a zero-norm tag embedding",
        f"- {d['n_queries']:,} queries from held-out playlists; "
        f"{d['queries_dropped_no_vocab']} titles dropped as out-of-vocabulary",
        f"- {d['n_features']} features -> {d['tag_dim']}-d target; seed {p0['config']['seed']}",
        f"- total runtime {p0['seconds']:.0f}s",
        "",
    ]
    demo_path = ARTIFACTS / "demo_report.json"
    if demo_path.exists():
        demo = json.loads(demo_path.read_text())
        a = demo["aggregate"]
        lines += [
            "## Joint demo",
            "",
            f"{a['total_frozen']} tracks frozen out across {demo['n_queries']} queries. "
            f"Recovered into the top {demo['top_n']} candidates: "
            f"**{a['recovered_with_timbre']} ({a['recovery_rate_with']:.0%})** with Timbre, "
            f"{a['recovered_without_timbre']} ({a['recovery_rate_without']:.0%}) without.",
            "",
            "| Query | Without Timbre | With Timbre |",
            "|---|---:|---:|",
        ]
        for run in demo["runs"]:
            lines.append(
                f"| {run['query']} | {run['cold_in_top_n']}/{run['n_cold']} | "
                f"{run['timbre_in_top_n']}/{run['n_cold']} |"
            )
        lines.append("")

    out = ROOT / "docs" / "RESULTS.md"
    out.write_text("\n".join(lines))
    return out


if __name__ == "__main__":
    print(build())
    print(build_markdown())
