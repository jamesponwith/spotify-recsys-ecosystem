"""Close a bead only when its pull request's merge actually reached main.

spot-2ig was closed on 2026-08-30 with the reason "PR #4 merged", and its work
was not on main for a day. PR #4's base was fleet/builder-permissions, an
integration branch that had already reached main through the earlier PR #3, so
nothing merged it forward again: five commits sat off main while the board
reported the bead shipped. Nothing was lost -- the accounting was what broke.
gh said MERGED and nobody asked WHERE.

So merged-ness is not the test. Reachability is: the PR's merge commit must be
an ancestor of the released ref. A PR merged anywhere else leaves its bead open
with a note naming the branch it landed on, because the work still has a hop to
make and the board should say so rather than claim it shipped.

Every unknown holds. An unresolvable PR, a merge with no merge commit recorded,
a commit this clone has never seen -- none of those are evidence that the work
is on main, and this closes only on evidence.

`--audit` asks the same question of beads that are already closed, and only
reports: whether a closed bead's work is reachable from main is a fact worth
knowing, but reopening it is a judgement, and merging the branch forward is the
human's call (ADR 0003).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Fully qualified on purpose. gitrevisions resolves a bare "origin/main" as a
# tag or a local branch of that name BEFORE the remote-tracking ref, and the
# fetch below auto-follows tags -- so anyone able to push a tag called
# origin/main would get to choose which commits count as shipped, which is the
# spot-2ig failure again with an extra step.
RELEASED_REF = "refs/remotes/origin/main"

# The one line step 7 of flywheel-next writes, and nothing else. Notes only,
# never the description: spot-e11's own description discusses "PR #4" and
# "PR #6" as evidence and spot-ln4's names three more. Anchored and requiring
# "opened", because beads cite other beads' pull requests as preconditions --
# spot-24j's note reads "Do this only on a base that already contains PR #9
# (spot-7lj)", and a loose `PR #\d+` would close spot-24j the day spot-7lj's
# work shipped, on a merge that contains nothing of spot-24j's.
PR_REF = re.compile(r"^PR #(\d+) opened\b", re.M)


@dataclass(frozen=True)
class PullRequest:
    number: int
    state: str  # OPEN | MERGED | CLOSED, as gh reports it
    base: str  # baseRefName -- the branch it merged INTO
    head: str  # headRefName -- the branch the work was written on
    merge_commit: str | None
    title: str


@dataclass(frozen=True)
class Verdict:
    action: str  # "close", "strand", "hold" or "skip"
    reason: str
    note: str | None = None  # appended to the bead when the action is "strand"


@dataclass
class Report:
    """One row per bead, under the heading that says what happened to it."""

    closed: list[str] = field(default_factory=list)
    stranded: list[str] = field(default_factory=list)
    held: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


# resolve(number) -> PullRequest | None; reachable(sha) -> True, False, or None
# when the commit is not in this clone and nothing can be said about it.
Resolve = Callable[[int], "PullRequest | None"]
Reachable = Callable[[str], "bool | None"]


def _flatten(text: str, limit: int = 160) -> str:
    """One line of printable text, capped.

    PR titles and branch names are author-controlled and editable after the
    merge, and this writes them into the bead board and into the log every
    other agent reads. A title carrying newlines or an escape sequence could
    forge a line under the CLOSED heading, or read as an instruction to the
    next agent along; neither survives this.
    """
    clean = " ".join("".join(c if c.isprintable() else " " for c in text).split())
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


def _short_ref(ref: str) -> str:
    """refs/remotes/origin/main -> origin/main, for reading rather than resolving."""
    return ref.removeprefix("refs/remotes/").removeprefix("refs/heads/")


def pr_numbers(bead: dict) -> list[int]:
    """Pull request numbers this bead's notes claim, in the order written."""
    return list(dict.fromkeys(int(n) for n in PR_REF.findall(bead.get("notes") or "")))


def owns(bead_id: str, pr: PullRequest) -> bool:
    """Is this pull request plausibly this bead's own work?

    The note naming a PR is free text that any agent may append -- ADR 0003
    lets them comment on beads -- so without this check "may comment" is "may
    close anything": one line reading `PR #6 opened: x` on an untouched bead
    retires it the moment PR #6 lands. The fleet's own conventions carry the
    binding: branches are bead/<id> and titles start with "<id>: ". Requiring
    either one costs nothing on a real bead and refuses a borrowed number.
    """
    return pr.head == f"bead/{bead_id}" or pr.title.startswith(f"{bead_id}:")


def verdict_for(
    bead: dict,
    resolve: Resolve,
    reachable: Reachable,
    released_ref: str = RELEASED_REF,
) -> Verdict:
    bead_id = bead["id"]
    numbers = pr_numbers(bead)
    if not numbers:
        return Verdict("skip", "no pull request referenced in the bead's notes")

    released = _short_ref(released_ref)
    landed: list[PullRequest] = []
    stranded: list[PullRequest] = []
    waiting: list[str] = []
    for n in numbers:
        pr = resolve(n)
        if pr is None:
            waiting.append(f"PR #{n} could not be read from GitHub")
        elif not owns(bead_id, pr):
            waiting.append(f"PR #{n} is {_flatten(pr.head, 60)}, not this bead's work")
        elif pr.state != "MERGED":
            waiting.append(f"PR #{n} is {pr.state.lower()}, not merged")
        elif not pr.merge_commit:
            waiting.append(f"PR #{n} is merged but GitHub reports no merge commit")
        else:
            on_released = reachable(pr.merge_commit)
            if on_released is None:
                waiting.append(f"PR #{n}'s merge commit {pr.merge_commit[:7]} is not in this clone")
            elif on_released:
                landed.append(pr)
            else:
                stranded.append(pr)

    # Every PR is judged before anything is returned, because a merge that
    # missed the released branch is a defect someone has to resolve and an
    # earlier unmerged PR on the same bead must not hide it.
    if stranded:
        names = ", ".join(f"#{p.number}" for p in stranded)
        branches = ", ".join(sorted({_flatten(p.base, 60) for p in stranded}))
        return Verdict(
            "strand",
            f"PR {names} merged into {branches}, which {released} does not contain",
            note=(
                f"reconcile-board: staying open — PR {names} merged into {branches}, not into "
                f"{released}. The work is not on main until that branch reaches it, and how it "
                f"gets there is the human's call (ADR 0003: the fleet does not merge)."
            ),
        )
    if waiting:
        return Verdict("hold", "; ".join(waiting))

    # A bead that names several PRs ships when the last of them does.
    shipped = "; ".join(
        f"PR #{p.number} merged into {_flatten(p.base, 60)}: {_flatten(p.title)}" for p in landed
    )
    return Verdict("close", f"{shipped} — reachable from {released}")


def reconcile(
    beads: Iterable[dict],
    resolve: Resolve,
    reachable: Reachable,
    released_ref: str = RELEASED_REF,
) -> list[tuple[dict, Verdict]]:
    return [(b, verdict_for(b, resolve, reachable, released_ref)) for b in beads]


def apply_verdicts(
    results: Iterable[tuple[dict, Verdict]],
    close: Callable[[str, str], None],
    note: Callable[[str, str], None],
    dry_run: bool = False,
) -> Report:
    """Act on each verdict and file every bead under exactly one heading."""
    report = Report()
    for bead, verdict in results:
        bead_id = bead.get("id", "<no id>")
        line = f"{bead_id}: {verdict.reason}"
        if verdict.action == "strand":
            # The strand is recorded before the note is attempted: it was
            # detected whether or not the board accepts the annotation.
            report.stranded.append(line)
            if verdict.note and verdict.note not in (bead.get("notes") or "") and not dry_run:
                # Repeated runs must not append the same sentence until the
                # notes field is all heartbeat and no signal.
                try:
                    note(bead_id, verdict.note)
                except Exception as exc:
                    report.failed.append(f"{bead_id}: strand found but the note failed: {exc}")
        elif verdict.action == "close":
            try:
                if not dry_run:
                    close(bead_id, verdict.reason)
                report.closed.append(line)
            except Exception as exc:
                # One bad bead must not hide the rest. Letting this propagate
                # would abandon the remaining beads AND throw away the report
                # of what had already happened -- the same class of bad
                # accounting this tool exists to end.
                report.failed.append(f"{bead_id}: {exc}")
        elif verdict.action == "hold":
            report.held.append(line)
        else:
            report.skipped.append(line)
    return report


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """Run cmd at the repo root; a missing binary is a failed call, not a crash.

    cwd is pinned because all three tools resolve their target from it: bd
    discovers .beads by walking up, and gh reads the repo from the working
    directory's remote. A runner started in the wrong directory would otherwise
    reconcile this board against another repository's pull requests.
    """
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=ROOT)
    except FileNotFoundError:
        return subprocess.CompletedProcess(cmd, 127, "", f"{cmd[0]}: not found")


def gh_pull_request(number: int) -> PullRequest | None:
    proc = _run(
        [
            "gh",
            "pr",
            "view",
            str(number),
            "--json",
            "number,state,baseRefName,headRefName,mergeCommit,title",
        ]
    )
    if proc.returncode != 0:
        return None
    try:
        d = json.loads(proc.stdout)
        merge = d.get("mergeCommit") or {}
        return PullRequest(
            number=d["number"],
            state=d["state"],
            base=d["baseRefName"],
            head=d["headRefName"],
            merge_commit=merge.get("oid"),
            title=d["title"],
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        # A gh that answers 0 with a shape this does not recognise is not
        # evidence of anything, so it holds like every other unknown.
        return None


def git_reachable(sha: str, ref: str) -> bool | None:
    if _run(["git", "cat-file", "-e", f"{sha}^{{commit}}"]).returncode != 0:
        return None
    return _run(["git", "merge-base", "--is-ancestor", sha, ref]).returncode == 0


def bd_close(bead_id: str, reason: str) -> None:
    # `--` first: bd resolves a bare positional as an issue id, and bd close
    # with no id closes the LAST TOUCHED issue, so a bead whose id began with a
    # dash would otherwise be read as a flag against a bead nobody named.
    proc = _run(["bd", "close", "--reason", reason, "--", bead_id])
    if proc.returncode != 0:
        raise RuntimeError(f"bd close {bead_id} failed: {_flatten(proc.stderr, 200)}")


def bd_note(bead_id: str, text: str) -> None:
    proc = _run(["bd", "note", "--", bead_id, text])
    if proc.returncode != 0:
        raise RuntimeError(f"bd note {bead_id} failed: {_flatten(proc.stderr, 200)}")


def _bd_list(*extra: str) -> list[dict]:
    proc = _run(["bd", "list", "--json", "--limit", "0", *extra])
    if proc.returncode != 0:
        raise RuntimeError(f"bd list failed: {_flatten(proc.stderr, 200)}")
    try:
        return json.loads(proc.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"bd list returned JSON this cannot read: {exc}") from exc


def active_beads() -> list[dict]:
    """Every bead that is not closed yet -- the ones a merge could still close.

    Not `--status=open`: a bead with a PR in review is in_progress, because the
    builder claimed it before writing a line. Filtering on open skipped exactly
    the beads this tool exists to reconcile (2 of 13 on the board today, both
    mid-flight). `--limit 0` because the default is 50, and a board that grows
    past that would go quietly unreconciled from the 51st bead on.
    """
    return [b for b in _bd_list() if b.get("status") != "closed"]


def closed_beads() -> list[dict]:
    return [b for b in _bd_list("--all") if b.get("status") == "closed"]


def _print_rows(label: str, rows: list[str]) -> None:
    if rows:
        print(f"\n{label}:")
        for row in rows:
            print(f"  {row}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--released-ref",
        default=RELEASED_REF,
        help=f"the ref a merge must be reachable from to count as shipped (default {RELEASED_REF})",
    )
    ap.add_argument("--dry-run", action="store_true", help="print the verdicts, change nothing")
    ap.add_argument("--no-fetch", action="store_true", help="skip the git fetch before deciding")
    ap.add_argument(
        "--audit",
        action="store_true",
        help="report CLOSED beads whose merge never reached the released ref; change nothing",
    )
    args = ap.parse_args(argv)
    released = _short_ref(args.released_ref)

    # Without the fetch, a merge commit on a branch this clone has never seen
    # is simply absent -- and absence is indistinguishable from "not on main"
    # unless the refs are current first. A failed fetch only ever costs holds,
    # never a wrong close, but it is worth saying out loud. --no-fetch exists
    # because an unreachable remote blocks rather than failing fast.
    if not args.no_fetch and _run(["git", "fetch", "--quiet", "origin"]).returncode != 0:
        print("reconcile-board: git fetch failed; deciding on stale refs", file=sys.stderr)

    if _run(["git", "rev-parse", "--verify", f"{args.released_ref}^{{commit}}"]).returncode != 0:
        print(f"reconcile-board: {args.released_ref} does not resolve here", file=sys.stderr)
        return 2

    def reachable(sha: str) -> bool | None:
        return git_reachable(sha, args.released_ref)

    try:
        beads = closed_beads() if args.audit else active_beads()
    except RuntimeError as exc:
        print(f"reconcile-board: {exc}", file=sys.stderr)
        return 2

    results = reconcile(beads, gh_pull_request, reachable, args.released_ref)

    if args.audit:
        # Report only. Reopening a closed bead is a judgement about work, and
        # merging the branch forward is reserved to the human.
        strands = [f"{b['id']}: {v.reason}" for b, v in results if v.action == "strand"]
        _print_rows(f"STRANDED — closed, but the merge never reached {released}", strands)
        print(f"\nreconcile-board: audited {len(beads)} closed bead(s), {len(strands)} stranded")
        if strands:
            print(
                f"reconcile-board: exiting 1 — {len(strands)} closed bead(s) are not on {released}"
            )
            return 1
        return 0

    report = apply_verdicts(results, bd_close, bd_note, args.dry_run)

    _print_rows(f"CLOSED — merged and reachable from {released}", report.closed)
    _print_rows(f"STRANDED — merged, but not onto {released}", report.stranded)
    _print_rows("HELD — nothing here is evidence the work shipped", report.held)
    _print_rows("FAILED — the board rejected the update", report.failed)

    print(
        f"\nreconcile-board: {len(beads)} bead(s) not yet closed — "
        f"{len(report.closed)} closed, {len(report.stranded)} stranded, "
        f"{len(report.held)} in flight, {len(report.skipped)} with no PR to check"
    )
    if report.failed:
        print(f"reconcile-board: exiting 2 — {len(report.failed)} update(s) rejected")
        return 2
    # A stranded bead is a defect someone has to resolve by merging that branch
    # forward, so the run says so by failing rather than printing and passing.
    if report.stranded:
        print(f"reconcile-board: exiting 1 — {len(report.stranded)} bead(s) merged off {released}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
