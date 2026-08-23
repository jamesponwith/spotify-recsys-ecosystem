"""Verify every relative markdown link and heading anchor resolves.

Anchors are the ones that rot silently: GitHub slugifies a heading by lowercasing,
stripping punctuation and joining on hyphens, so an em dash inside a heading
collapses to a *double* hyphen and quietly breaks a hand-written single-hyphen
link. Checking by hand does not scale past a few documents.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def slugify(heading: str) -> str:
    return re.sub(r"[^a-z0-9 -]", "", heading.lower()).strip().replace(" ", "-")


def main() -> int:
    problems: list[str] = []
    for md in sorted(ROOT.glob("**/*.md")):
        if ".venv" in md.parts:
            continue
        for text, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", md.read_text()):
            if target.startswith(("http://", "https://", "#")):
                continue
            path, _, frag = target.partition("#")
            if not path:
                continue
            resolved = (md.parent / path).resolve()
            rel = md.relative_to(ROOT)
            if not resolved.exists():
                problems.append(f"{rel}: [{text}]({target}) -> file does not exist")
            elif frag and resolved.suffix == ".md":
                slugs = {slugify(h) for h in re.findall(r"^#+\s+(.*)$", resolved.read_text(), re.M)}
                if frag not in slugs:
                    problems.append(f"{rel}: [{text}]({target}) -> no such anchor")
    if problems:
        print("\n".join(problems))
        return 1
    print("all markdown links and anchors resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
