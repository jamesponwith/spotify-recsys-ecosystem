"""Insert the generated-playlist section into the report page.

The page was hand-written; this splices one generated section into it between
markers, so re-running after `build_demo.py` refreshes the playlists without
touching the prose around them. Idempotent: the markers are replaced, not
appended.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "artifacts" / "results.html"
DEMO = ROOT / "artifacts" / "demo.json"
START, END = "<!-- DEMO:START -->", "<!-- DEMO:END -->"
ANCHOR = "  <!-- ============================ CONSTRAINTS ======================= -->"

EXTRA_CSS = """
  .demo-grid { display: flex; flex-direction: column; gap: 26px; }
  .demo-q { font-family: var(--mono); font-size: 13px; color: var(--accent-ink);
    letter-spacing: .01em; }
  .demo-title { font-size: 19px; font-weight: 650; margin-top: 4px; }
  .demo-desc { font-size: 14.5px; color: var(--ink-2); margin-top: 2px; max-width: 62ch; }
  .demo-facts { display: flex; flex-wrap: wrap; gap: 6px 16px; margin-top: 10px;
    font-family: var(--mono); font-size: 12px; color: var(--ink-3); }
  .demo-facts b { color: var(--ink-2); font-weight: 600; }
  .why { color: var(--ink-3); font-size: 12.5px; }
  .seg { font-family: var(--mono); font-size: 11.5px; color: var(--accent-ink);
    white-space: nowrap; }
  td.pos { font-family: var(--mono); color: var(--ink-3); width: 26px; text-align: right; }
"""


def esc(s: object) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def case_html(c: dict, limit: int, note: str = "") -> str:
    rows = []
    for t in c["tracks"][:limit]:
        why = esc(t["reasons"][0]) if t["reasons"] else ""
        seg = (
            f'<span class="seg">{esc(t["transition_note"])}</span>' if t["transition_note"] else ""
        )
        rows.append(
            f'<tr><td class="pos">{t["position"]}</td>'
            f"<td><strong>{esc(t['name'])}</strong><br><span class='why'>{esc(t['artist'])}"
            f"{' &middot; ' + why if why else ''}</span></td>"
            f"<td>{seg}</td></tr>"
        )
    ok = sum(1 for v in c["constraint_report"].values() if v)
    total = len(c["constraint_report"])
    facts = (
        f"<span><b>{c['n_tracks']}</b> tracks</span>"
        f"<span><b>{c['duration_min']:.0f}</b> min</span>"
        f"<span><b>{ok}/{total}</b> constraints met</span>"
        f"<span><b>{c['latency_ms']:.0f}</b> ms</span>"
    )
    return f"""
      <div class="panel">
        <div class="demo-q">&ldquo;{esc(c["query"])}&rdquo;</div>
        <div class="demo-title">{esc(c["title"])}</div>
        <div class="demo-desc">{esc(c["description"])}</div>
        <div class="demo-facts">{facts}</div>
        <div class="chart-scroll"><table>
          <thead><tr><th></th><th>Track &middot; why it is here</th><th>Segue</th></tr></thead>
          <tbody>{"".join(rows)}</tbody>
        </table></div>
        {note}
      </div>"""


def main() -> int:
    page = PAGE.read_text()
    demo = json.loads(DEMO.read_text())
    cases = demo["cases"]

    miss_note = (
        '<p style="font-size:13.5px;color:var(--ink-3);margin-top:12px;max-width:60ch">'
        "<strong>Still the weakest of the three, and the reasons say why.</strong> "
        "Retrieval gets this right &mdash; it returns Pixies, Third Eye Blind and No Doubt. "
        "Selection then trades some of that away for a valence target inferred from the "
        "words <em>road trip</em>. Two defects found here have been fixed (see "
        "<code>docs/FINDINGS.md</code>); what remains is that the audio weight is global "
        "when it should depend on whether the request names a genre or a mood. That is "
        "diagnosable in one line precisely because every track carries a grounded reason "
        "rather than a similarity score.</p>"
    )

    section = f"""{START}
  <!-- ============================ WHAT IT PLAYS ===================== -->
  <section>
    <div class="col">
      <p class="eyebrow">The output</p>
      <h2>What it actually plays</h2>
      <p class="lede">
        Three requests, run end to end. Every track carries the reason it was chosen
        and the transition into it.
      </p>
      <p>
        The two things worth looking at are the ones a ranked list does not have. Each
        row's reason is <em>grounded</em> &mdash; it names the tag and the count, or the
        feature and its distance from target, never a bare score. Each segue is the
        sequencer's actual output: a tempo move and a Camelot step.
      </p>
    </div>
    <div class="demo-grid">
      {case_html(cases[0], 8)}
      {case_html(cases[1], 6)}
      {case_html(cases[2], 8, miss_note)}
    </div>
  </section>

{END}
"""

    if START in page and END in page:
        pre, rest = page.split(START, 1)
        _, post = rest.split(END, 1)
        page = pre + section.strip() + post
    else:
        if ANCHOR not in page:
            print("anchor comment not found; page structure changed", file=sys.stderr)
            return 1
        page = page.replace(ANCHOR, section + ANCHOR, 1)

    if ".demo-grid" not in page:
        page = page.replace("  table { border-collapse", EXTRA_CSS + "  table { border-collapse", 1)

    PAGE.write_text(page)
    print(f"wrote {PAGE} ({len(page):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
