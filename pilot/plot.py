"""Per-run dashboard PNG: what the pilot is doing and why.

Panels (one measure each, shared surface, no dual axes):
  A  pass rate by level - baseline vs round 1, trained band shaded, held-out level marked
  B  running pass rate while sampling, per phase
  C  training loss per optimizer step
  D  GPU utilization over time (mean of all GPUs), phase boundaries marked

Works mid-run (from live/ files + the event-stream state) and after (from runs/ archive).
Usage: uv run --with matplotlib python -m pilot.plot <run>  -> runs/pilot/<run>/dashboard.png
"""

import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LEVELS = ["1", "2", "5", "10", "20", "all"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # validated categorical slots 1-3 (light)
BAND = "#f0efec"
PHASE_COLOR = {"base_test": BLUE, "base_train": ORANGE, "r1_test": AQUA}
PHASE_NAME = {"base_test": "baseline (test)", "base_train": "baseline (train tasks)", "r1_test": "round 1 (test)"}
PRICE = {"g6e.12xlarge": 10.49, "g5.12xlarge": 5.67, "g6.12xlarge": 4.60}


def s3_text(uri: str) -> str:
    return subprocess.run(["aws", "s3", "cp", uri, "-"], capture_output=True, text=True).stdout


def jsonl(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def level_rates(rows: list[dict]) -> dict[str, float]:
    by = defaultdict(list)
    for r in rows:
        by[r["level"]].append(r["passed"])
    return {lv: sum(v) / len(v) for lv, v in by.items()}


def main(run: str) -> Path:
    live, archive = f"s3://samgo-runs-811269446285/live/{run}", f"s3://samgo-runs-811269446285/runs/{run}"
    out = Path("runs/pilot") / run
    out.mkdir(parents=True, exist_ok=True)
    state_path = Path(f"runs/pilot/watch_state_{run}.json")
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    archived = bool(s3_text(f"{archive}/report.md").strip())
    src = archive if archived else live
    plan = json.loads(s3_text(f"{src}/plan.json") or "{}")
    metrics = json.loads(s3_text(f"{archive}/metrics.json") or "{}") if archived else {}

    rates = {}
    for part in ("base_test", "r1_test"):
        if archived:
            rows = jsonl(s3_text(f"{archive}/{part}/samples.jsonl"))
            if rows:
                rates[part] = level_rates(rows)
        else:
            prog = json.loads(s3_text(f"{live}/{part}/progress.json") or "{}")
            if prog:
                rates[part] = prog.get("pass_rate_by_level", {})
    train_log = jsonl(s3_text(f"{src}/round1/train_log.jsonl"))
    gpu_lines = s3_text(f"{src}/gpu.log").splitlines()
    run_log = s3_text(f"{src}/run.log").splitlines()
    phase = json.loads(s3_text(f"{src}/phase.json") or "{}").get("phase", "?")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2), facecolor=SURFACE)
    (a, b), (c, d) = axes

    # A: pass rate by level
    style(a, "A  Pass rate by level")
    a.set_ylabel("fraction of samples passing", color=INK2, fontsize=9)
    x = list(range(len(LEVELS)))
    trained = set(plan.get("train_levels", []))
    for i, lv in enumerate(LEVELS):
        if lv in trained:
            a.axvspan(i - 0.5, i + 0.5, color=BAND, zorder=0)
    width = 0.36
    for j, (part, color) in enumerate((("base_test", BLUE), ("r1_test", ORANGE))):
        if part not in rates:
            continue
        xs = [i + (j - 0.5) * width for i in x]
        ys = [rates[part].get(lv, 0) for lv in LEVELS]
        a.bar(xs, ys, width=width - 0.04, color=color, label=PHASE_NAME[part], zorder=2)
        for xi, yi, lv in zip(xs, ys, LEVELS):
            if lv in rates[part]:
                a.text(xi, yi + 0.015, f"{yi:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    if plan.get("transfer_level") in LEVELS:
        i = LEVELS.index(plan["transfer_level"])
        a.text(i, 1.04, "held out", ha="center", fontsize=8, color=INK2)
    if trained:
        first = min(LEVELS.index(lv) for lv in trained)
        a.text(first - 0.45, 1.04, "trained band", ha="left", fontsize=8, color=INK2)
    a.set_xticks(x, [f"last {lv}" if lv != "all" else "full build" for lv in LEVELS])
    a.set_ylim(0, 1.1)
    a.set_xlabel("pieces missing", color=INK2, fontsize=9)
    if rates:
        a.legend(frameon=False, fontsize=8.5, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2)

    # B: running pass rate per sampling phase
    style(b, "B  Running pass rate while sampling")
    hist = state.get("history", [])
    ends = []
    for part in ("base_test", "base_train", "r1_test"):
        pts = [h for h in hist if h["part"] == part]
        if pts:
            xs = [h["tasks"] / h["of"] for h in pts]
            ys = [h["pass"] for h in pts]
            b.plot(xs, ys, color=PHASE_COLOR[part], linewidth=2, marker="o", markersize=4)
            ends.append([ys[-1], xs[-1], PHASE_NAME[part]])
    ends.sort()
    for i in range(1, len(ends)):  # keep direct labels at least 0.045 apart
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.045)
    for y, x_end, name in ends:
        b.text(x_end + 0.015, y, name, color=INK2, fontsize=8, va="center")
    b.set_xlim(0, 1.3)
    b.set_ylim(0, 1)
    b.set_xlabel("fraction of phase's tasks done", color=INK2, fontsize=9)
    if not hist:
        b.text(0.5, 0.5, "no sampling yet", ha="center", color=INK2, transform=b.transAxes)

    # C: training loss
    style(c, "C  Training loss per optimizer step (completion tokens only)")
    if train_log:
        steps = [r["step"] for r in train_log]
        c.plot(steps, [r["loss"] for r in train_log], color=BLUE, linewidth=2)
        c.set_xlim(0, train_log[-1]["of"])
        c.set_xlabel("step", color=INK2, fontsize=9)
    else:
        c.text(0.5, 0.5, "training not started", ha="center", color=INK2, transform=c.transAxes)

    # D: GPU utilization with phase boundaries
    style(d, "D  GPU utilization, mean of all GPUs (%)")
    util = defaultdict(list)
    for line in gpu_lines:
        parts = line.split(", ")
        if len(parts) >= 4:
            t = line.split(" ")[0]
            util[t].append(float(parts[1].rstrip(" %")))
    if util:
        times = sorted(util)
        t0 = datetime.strptime(times[0], "%H:%M:%S")
        mins = [((datetime.strptime(t, "%H:%M:%S") - t0).total_seconds() % 86400) / 60 for t in times]
        d.plot(mins, [sum(util[t]) / len(util[t]) for t in times], color=BLUE, linewidth=2)
        marks = []
        for line in run_log:
            if "phase: " in line:
                ts = line[1:20].split("T")[1][:8]
                m = ((datetime.strptime(ts, "%H:%M:%S") - t0).total_seconds() % 86400) / 60
                d.axvline(m, color=GRID, linewidth=1, zorder=0)
                marks.append((m, line.split("phase: ")[1][:14]))
        gap = max(mins[-1], 1) * 0.07
        for i, (m, name) in enumerate(marks):  # label only the last phase of each crowded cluster
            if i + 1 < len(marks) and marks[i + 1][0] - m < gap:
                continue
            d.text(m, 103, name, rotation=30, fontsize=7, color=INK2, ha="left")
        d.set_ylim(0, 115)
        d.set_xlabel("minutes since first heartbeat", color=INK2, fontsize=9)
        hours = mins[-1] / 60
    else:
        hours = 0.0
        d.text(0.5, 0.5, "no heartbeat yet", ha="center", color=INK2, transform=d.transAxes)

    itype = state.get("instance_type") or ""
    cost = f" | ~${hours * PRICE.get(itype, 5.67):.2f} so far" if hours else ""
    verdict = f" | verdict {metrics.get('verdict')}" if metrics.get("verdict") else ""
    fig.suptitle(f"SAMgo {run}: phase {phase}{verdict}{cost}", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = out / "dashboard.png"
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    return path


if __name__ == "__main__":
    print(main(sys.argv[1] if len(sys.argv) > 1 else "pilot-002"))
