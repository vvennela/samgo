"""Offline supervision inventory; counts edits, not successful learned behaviors."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .assembly import save
from .running_gear_task import edits_from


def beams(text):
    return [e["part"] for e in edits_from(text) if e.get("part", {}).get("type") == "technic_brick"]


def centers(parts):
    return {(p["x"] + p["w"] / 2, p["y"], p["z"], p["l"]): p["w"] for p in parts}


def inventory(path):
    counts = Counter(
        rows=0,
        unparsed=0,
        feedback_rows=0,
        feedback_rows_changing_beam_width_multiset=0,
        feedback_rows_resizing_same_center_beams=0,
        feedback_rows_changing_width_with_same_beam_count=0,
        feedback_mentions_longer=0,
    )
    widths, feedback_widths = Counter(), Counter()
    for line in path.read_text().splitlines():
        counts["rows"] += 1
        row = json.loads(line)
        chat = row["messages"]
        answers = [i for i, m in enumerate(chat) if m["role"] == "assistant"]
        try:
            after = beams(chat[answers[-1]]["content"])
            widths.update(p["w"] for p in after)
            if len(answers) < 2:
                continue
            counts["feedback_rows"] += 1
            before = beams(chat[answers[-2]]["content"])
        except (ValueError, KeyError, IndexError, TypeError):
            counts["unparsed"] += 1
            continue
        feedback_widths.update(p["w"] for p in after)
        changed = Counter(p["w"] for p in before) != Counter(p["w"] for p in after)
        counts["feedback_rows_changing_beam_width_multiset"] += changed
        counts["feedback_rows_changing_width_with_same_beam_count"] += changed and len(before) == len(after)
        a, b = centers(before), centers(after)
        counts["feedback_rows_resizing_same_center_beams"] += any(a[k] != b[k] for k in a.keys() & b.keys())
        feedback = " ".join(m["content"] for m in chat[answers[-2] + 1 : answers[-1]] if m["role"] == "user")
        counts["feedback_mentions_longer"] += "longer" in feedback.lower()
    return dict(counts) | dict(
        beam_part_counts=dict(widths),
        feedback_target_beams=dict(feedback_widths),
        source=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    save(args.out, {str(p): inventory(p) for p in args.paths})


if __name__ == "__main__":
    main()
