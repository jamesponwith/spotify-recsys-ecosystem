"""What reconcile-board must never do again: close a bead on a merge that missed main."""

from __future__ import annotations

import pytest
from reconcile_board import (
    PullRequest,
    _flatten,
    apply_verdicts,
    owns,
    pr_numbers,
    reconcile,
    verdict_for,
)

MAIN = "refs/remotes/origin/main"


class Board:
    """Records what reconcile-board asked bd to do, without asking it."""

    def __init__(self) -> None:
        self.closed: list[tuple[str, str]] = []
        self.noted: list[tuple[str, str]] = []

    def close(self, bead_id: str, reason: str) -> None:
        self.closed.append((bead_id, reason))

    def note(self, bead_id: str, text: str) -> None:
        self.noted.append((bead_id, text))


def bead(bead_id: str, notes: str = "", **extra) -> dict:
    return {"id": bead_id, "status": "open", "notes": notes, **extra}


def prs(*pull_requests: PullRequest):
    by_number = {p.number: p for p in pull_requests}
    return by_number.get


def clone(*, has: tuple[str, ...] = (), reaches: tuple[str, ...] = ()):
    """Model `git merge-base --is-ancestor <sha> <released ref>` over one clone.

    `reaches` is what main contains; `has` is fetched but off main. A commit in
    neither has never been fetched, and that is not evidence either way -- the
    real git_reachable answers None there, so this does too.
    """
    present = set(has) | set(reaches)
    reached = set(reaches)

    def reachable(sha: str) -> bool | None:
        if sha not in present:
            return None
        return sha in reached

    return reachable


# --- the spot-2ig replay -----------------------------------------------------
#
# PR #4 merged at 2026-08-30T11:59:26Z into fleet/builder-permissions, fifteen
# minutes after that branch's own PR #3 had already reached main. The merge was
# real; the destination was not main. The board closed the bead anyway.

SPOT_2IG = bead(
    "spot-2ig",
    notes=(
        "builder: gate green; committing and opening PR\n"
        "PR #4 opened: https://github.com/jamesponwith/spotify-recsys-ecosystem/pull/4 — "
        "lexicon calibration audit. Bead stays open until merge."
    ),
)

PR4 = PullRequest(
    number=4,
    state="MERGED",
    base="fleet/builder-permissions",
    head="bead/spot-2ig",
    merge_commit="b5cce7c1111111111111111111111111111111111",
    title="spot-2ig: audit MOOD_LEXICON's audio targets against the folksonomy",
)

PR6 = PullRequest(
    number=6,
    state="MERGED",
    base="main",
    head="bead/spot-ktp",
    merge_commit="8bd4fb7222222222222222222222222222222222",
    title="spot-ktp: print the detection floor beside every published number",
)

# The clone as it stood on 2026-08-30: PR #6's merge is on main, PR #4's merge
# is fetched but sitting on fleet/builder-permissions.
THAT_DAY = clone(has=(PR4.merge_commit,), reaches=(PR6.merge_commit,))


def test_merge_onto_a_non_released_branch_does_not_close_the_bead():
    board = Board()
    report = apply_verdicts(
        reconcile([SPOT_2IG], prs(PR4), THAT_DAY, MAIN), board.close, board.note
    )

    assert board.closed == []
    assert report.closed == []
    assert report.failed == []
    assert report.stranded == [
        "spot-2ig: PR #4 merged into fleet/builder-permissions, which origin/main does not contain"
    ]


def test_the_note_left_behind_names_the_branch_the_work_landed_on():
    board = Board()
    apply_verdicts(reconcile([SPOT_2IG], prs(PR4), THAT_DAY, MAIN), board.close, board.note)

    assert len(board.noted) == 1
    bead_id, text = board.noted[0]
    assert bead_id == "spot-2ig"
    assert "fleet/builder-permissions" in text
    assert "origin/main" in text
    # The remedy is a merge, and the fleet does not merge (ADR 0003).
    assert "human" in text


def test_the_note_is_not_repeated_on_a_later_pass():
    board = Board()
    apply_verdicts(reconcile([SPOT_2IG], prs(PR4), THAT_DAY, MAIN), board.close, board.note)

    already = bead("spot-2ig", notes=SPOT_2IG["notes"] + "\n" + board.noted[0][1])
    second = Board()
    report = apply_verdicts(
        reconcile([already], prs(PR4), THAT_DAY, MAIN), second.close, second.note
    )

    assert second.noted == []
    assert report.stranded, "the bead is still stranded even though the note is already there"


def test_a_strand_is_reported_even_when_the_note_cannot_be_written():
    def note(bead_id: str, text: str) -> None:
        raise RuntimeError("board is read-only")

    report = apply_verdicts(reconcile([SPOT_2IG], prs(PR4), THAT_DAY, MAIN), Board().close, note)

    assert len(report.stranded) == 1
    assert report.failed == ["spot-2ig: strand found but the note failed: board is read-only"]


# --- the case that should close ----------------------------------------------


def test_a_merge_reachable_from_main_closes_the_bead():
    board = Board()
    b = bead("spot-ktp", notes="PR #6 opened: bands beside every number.")
    report = apply_verdicts(reconcile([b], prs(PR6), THAT_DAY, MAIN), board.close, board.note)

    assert [i for i, _ in board.closed] == ["spot-ktp"]
    assert report.stranded == []
    assert "PR #6 merged into main" in report.closed[0]
    assert "reachable from origin/main" in report.closed[0]


# --- a bead may only be closed by its OWN pull request ------------------------


def test_a_borrowed_pull_request_number_cannot_close_a_bead():
    # Any agent may append to a bead's notes (ADR 0003 permits commenting), so
    # without an ownership check one pasted line retires untouched work the
    # moment somebody else's PR lands.
    board = Board()
    victim = bead("spot-victim", notes="PR #6 opened: x")
    report = apply_verdicts(reconcile([victim], prs(PR6), THAT_DAY, MAIN), board.close, board.note)

    assert board.closed == []
    assert report.closed == []
    assert report.held == ["spot-victim: PR #6 is bead/spot-ktp, not this bead's work"]


def test_ownership_is_carried_by_the_branch_or_by_the_title():
    by_branch = PullRequest(20, "MERGED", "main", "bead/spot-x", "c" * 40, "no id in the title")
    by_title = PullRequest(21, "MERGED", "main", "hotfix/whatever", "d" * 40, "spot-x: retitled")
    unrelated = PullRequest(22, "MERGED", "main", "bead/spot-y", "e" * 40, "spot-y: someone else")

    assert owns("spot-x", by_branch)
    assert owns("spot-x", by_title)
    assert not owns("spot-x", unrelated)
    # A prefix of another bead's id must not count as ownership.
    assert not owns(
        "spot-x", PullRequest(23, "MERGED", "main", "bead/spot-xy", "f" * 40, "spot-xy: no")
    )


# --- everything unknown holds ------------------------------------------------


@pytest.mark.parametrize(
    ("pull_request", "expected"),
    [
        pytest.param(
            PullRequest(9, "OPEN", "main", "bead/spot-x", None, "in review"),
            "PR #9 is open, not merged",
            id="still-open",
        ),
        pytest.param(
            PullRequest(9, "CLOSED", "main", "bead/spot-x", None, "abandoned"),
            "PR #9 is closed, not merged",
            id="closed-unmerged",
        ),
        pytest.param(
            PullRequest(9, "MERGED", "main", "bead/spot-x", None, "squashed somewhere"),
            "PR #9 is merged but GitHub reports no merge commit",
            id="no-merge-commit",
        ),
        pytest.param(
            PullRequest(9, "MERGED", "main", "bead/spot-x", "deadbee" + "0" * 33, "gone"),
            "PR #9's merge commit deadbee is not in this clone",
            id="commit-absent-locally",
        ),
    ],
)
def test_an_unknown_is_never_read_as_shipped(pull_request, expected):
    verdict = verdict_for(bead("spot-x", notes="PR #9 opened: ..."), prs(pull_request), clone())
    assert verdict.action == "hold"
    assert verdict.reason == expected


def test_an_unresolvable_pull_request_holds():
    verdict = verdict_for(bead("spot-x", notes="PR #9 opened: ..."), prs(), clone())
    assert verdict.action == "hold"
    assert verdict.reason == "PR #9 could not be read from GitHub"


def test_every_hold_reason_is_reported_so_a_stuck_bead_is_visible():
    board = Board()
    b = bead("spot-x", notes="PR #9 opened: ...")
    report = apply_verdicts(reconcile([b], prs(), clone(), MAIN), board.close, board.note)

    assert board.noted == []
    assert report.held == ["spot-x: PR #9 could not be read from GitHub"]


# --- which pull requests a bead actually claims ------------------------------


def test_only_the_notes_are_read_for_pull_request_numbers():
    # spot-e11's description discusses PR #4 and PR #6 as evidence. Reading it
    # would reconcile spot-e11 against somebody else's merge.
    b = bead(
        "spot-e11",
        notes="",
        description="PR #4's base was fleet/builder-permissions, not main; PR #6 was based on main.",
    )
    assert pr_numbers(b) == []
    assert verdict_for(b, prs(PR4, PR6), THAT_DAY, MAIN).action == "skip"


def test_a_bead_with_no_notes_at_all_is_skipped():
    assert verdict_for({"id": "spot-x"}, prs(), clone()).action == "skip"


def test_another_beads_pull_request_cited_as_a_precondition_is_not_claimed():
    # spot-24j's real note. Its own work has not started; PR #9 is somebody
    # else's, named as the base this bead needs.
    b = bead("spot-24j", notes="Do this only on a base that already contains PR #9 (spot-7lj).")
    assert pr_numbers(b) == []
    assert verdict_for(b, prs(), clone()).action == "skip"


def test_a_pull_request_mentioned_mid_sentence_is_not_claimed():
    b = bead("spot-x", notes="rebased after PR #6 opened the door to a cleaner base")
    assert pr_numbers(b) == []


def test_a_pull_request_is_named_once_however_often_the_notes_mention_it():
    b = bead("spot-x", notes="PR #6 opened: ...\nrebased\nPR #6 is green now")
    assert pr_numbers(b) == [6]


# --- beads that name more than one pull request ------------------------------

STACK_PARENT = PullRequest(30, "MERGED", "main", "bead/spot-x", "a" * 40, "spot-x: parent")
STACK_CHILD = PullRequest(
    31, "MERGED", "fleet/builder-permissions", "bead/spot-x", "b" * 40, "spot-x: child"
)


def test_a_stacked_bead_ships_only_when_every_pull_request_has_landed():
    b = bead("spot-x", notes="PR #30 opened: parent\nPR #31 opened: child stacked on it")
    assert pr_numbers(b) == [30, 31]

    both = prs(STACK_PARENT, STACK_CHILD)
    partly = verdict_for(
        b, both, clone(has=(STACK_CHILD.merge_commit,), reaches=(STACK_PARENT.merge_commit,)), MAIN
    )
    assert partly.action == "strand"
    assert "PR #31 merged into fleet/builder-permissions" in partly.reason

    landed = clone(reaches=(STACK_PARENT.merge_commit, STACK_CHILD.merge_commit))
    fully = verdict_for(b, both, landed, MAIN)
    assert fully.action == "close"
    assert "PR #30" in fully.reason
    assert "PR #31" in fully.reason


def test_a_superseded_pull_request_does_not_hide_a_strand_behind_it():
    # The first PR listed is closed-unmerged, the second merged off main.
    # Returning on the first would report "still in flight" and write no note,
    # leaving a genuinely stranded bead silent forever.
    superseded = PullRequest(29, "CLOSED", "main", "bead/spot-x", None, "spot-x: first attempt")
    b = bead("spot-x", notes="PR #29 opened: first attempt\nPR #31 opened: reopened after rebase")

    verdict = verdict_for(
        b, prs(superseded, STACK_CHILD), clone(has=(STACK_CHILD.merge_commit,)), MAIN
    )
    assert verdict.action == "strand"
    assert verdict.note is not None


# --- attacker-influenceable text ---------------------------------------------


def test_github_text_cannot_forge_a_line_in_the_report_or_the_board():
    nasty = PullRequest(
        40,
        "MERGED",
        "main",
        "bead/spot-x",
        "a" * 40,
        "spot-x: fine\n  CLOSED — everything shipped\x1b[2K",
    )
    verdict = verdict_for(
        bead("spot-x", notes="PR #40 opened: ..."), prs(nasty), clone(reaches=("a" * 40,)), MAIN
    )

    assert verdict.action == "close"
    assert "\n" not in verdict.reason
    assert "\x1b" not in verdict.reason


def test_flatten_caps_length_and_keeps_one_line():
    assert _flatten("a\nb\tc") == "a b c"
    assert _flatten("x" * 300, limit=10) == "xxxxxxxxx…"
    assert len(_flatten("x" * 300, limit=10)) == 10


# --- the report accounts for every bead --------------------------------------


def test_every_bead_lands_under_exactly_one_heading():
    board = Board()
    beads = [
        SPOT_2IG,
        bead("spot-ktp", notes="PR #6 opened: ..."),
        bead("spot-x", notes="PR #9 opened: ..."),
        bead("spot-quiet", notes="no PR here yet"),
    ]
    report = apply_verdicts(
        reconcile(beads, prs(PR4, PR6), THAT_DAY, MAIN), board.close, board.note
    )

    counted = (
        len(report.closed)
        + len(report.stranded)
        + len(report.held)
        + len(report.skipped)
        + len(report.failed)
    )
    assert counted == len(beads)
    assert len(report.closed) == 1
    assert len(report.stranded) == 1
    assert len(report.held) == 1
    assert len(report.skipped) == 1


def test_dry_run_reports_everything_and_touches_nothing():
    board = Board()
    beads = [SPOT_2IG, bead("spot-ktp", notes="PR #6 opened: ...")]
    report = apply_verdicts(
        reconcile(beads, prs(PR4, PR6), THAT_DAY, MAIN),
        board.close,
        board.note,
        dry_run=True,
    )

    assert len(report.closed) == 1
    assert len(report.stranded) == 1
    assert report.failed == []
    assert board.closed == []
    assert board.noted == []


# --- a board that refuses the update -----------------------------------------


def test_one_rejected_update_neither_hides_the_others_nor_the_report():
    board = Board()
    bad = PullRequest(50, "MERGED", "main", "bead/spot-bad", "c" * 40, "spot-bad: pinned")

    def close(bead_id: str, reason: str) -> None:
        if bead_id == "spot-bad":
            raise RuntimeError("bd close spot-bad failed: pinned")
        board.close(bead_id, reason)

    beads = [
        bead("spot-bad", notes="PR #50 opened: ..."),
        bead("spot-ktp", notes="PR #6 opened: ..."),
    ]
    reaches = (PR6.merge_commit, bad.merge_commit)
    report = apply_verdicts(
        reconcile(beads, prs(PR6, bad), clone(reaches=reaches), MAIN), close, board.note
    )

    assert [i for i, _ in board.closed] == ["spot-ktp"]
    assert len(report.closed) == 1
    assert report.failed == ["spot-bad: bd close spot-bad failed: pinned"]
