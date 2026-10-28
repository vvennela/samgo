"""Event stream for a pilot run: prints one line per new development, for Claude's Monitor.

Polls EC2 and S3 every 30 s. Emits: instance launch/state changes, phase changes,
each sampling chunk (pass rate by level), each training step (loss, grad norm),
new errors in logs, stale heartbeat, idle GPUs, and the final report. Remembers what it
already emitted in a state file, so restarting it does not repeat events.
Exits after emitting the final report or a terminal instance state.

Usage: python3 pilot/events.py [run]   (run defaults to pilot-002)
"""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

RUN = sys.argv[1] if len(sys.argv) > 1 else "pilot-002"
BUCKET = f"s3://samgo-runs-811269446285/live/{RUN}"
ARCHIVE = f"s3://samgo-runs-811269446285/runs/{RUN}"
STATE = Path(f"runs/pilot/watch_state_{RUN}.json")
TERMINAL_STATES = {"stopping", "stopped", "shutting-down", "terminated"}
WORK_PHASES = ("baseline_", "train", "round1_test")


def sh(*args: str) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=60).stdout
    except subprocess.TimeoutExpired:
        return ""


def s3(name: str) -> str:
    return sh("aws", "s3", "cp", f"{BUCKET}/{name}", "-")


def emit(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


def instance() -> tuple[str, str, str] | None:
    out = sh("aws", "ec2", "describe-instances", "--region", "us-west-2",
             "--filters", f"Name=tag:Name,Values=samgo-{RUN}",
             "--query", "Reservations[].Instances[].[InstanceId,State.Name,LaunchTime,InstanceType]", "--output", "text")
    rows = [line.split("\t") for line in out.strip().splitlines() if line.strip()]
    if not rows:
        return None
    rows.sort(key=lambda r: r[2])
    return rows[-1][0], rows[-1][1], rows[-1][3]


def main() -> None:
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    st.setdefault("chunks", {})
    st.setdefault("errors", [])
    idle_since = None
    while True:
        inst = instance()
        if inst:
            iid, state, itype = inst
            if st.get("instance") != iid:
                st["instance"], st["instance_type"] = iid, itype
                emit(f"instance launched: {iid} ({itype})")
            if st.get("state") != state:
                st["state"] = state
                emit(f"instance {iid} is {state}")

        phase_raw = s3("phase.json")
        phase = json.loads(phase_raw)["phase"] if phase_raw.strip() else None
        if phase and phase != st.get("phase"):
            st["phase"] = phase
            emit(f"phase -> {phase}")

        for part in ("base_test", "base_train", "r1_test"):
            raw = s3(f"{part}/progress.json")
            if not raw.strip():
                continue
            p = json.loads(raw)
            key = f"{part}:{p['tasks_done']}"
            if key not in st["chunks"]:
                st["chunks"][key] = 1
                st.setdefault("history", []).append({"part": part, "tasks": p["tasks_done"], "of": p["tasks_total"],
                                                     "pass": p["pass_rate"], "by_level": p.get("pass_rate_by_level", {}),
                                                     "tok_s": p["tokens_per_s"], "t": time.time()})
                levels = " ".join(f"{k}={v:.2f}" for k, v in p.get("pass_rate_by_level", {}).items())
                emit(f"{part} {p['tasks_done']}/{p['tasks_total']} tasks | pass {p['pass_rate']:.3f} | {levels} | "
                     f"{p['mean_completion_tokens']:.0f} tok/sample, {p['tokens_per_s']:.0f} tok/s")

        log = s3("round1/train_log.jsonl").strip().splitlines()
        if len(log) > st.get("train_steps", 0):
            for line in log[st.get("train_steps", 0):]:
                r = json.loads(line)
                emit(f"train step {r['step']}/{r['of']} | loss {r['loss']:.4f} | grad {r['grad_norm']:.3f} | "
                     f"lr {r['lr']:.2e} | {r['elapsed_s'] / 60:.1f} min")
            st["train_steps"] = len(log)
        meta = s3("round1/train_meta.json")
        if meta.strip() and not st.get("train_meta"):
            st["train_meta"] = 1
            emit(f"training data: {meta.strip()[:300]}")
        plan = s3("plan.json")
        if plan.strip() and not st.get("plan"):
            st["plan"] = 1
            emit(f"plan: {plan.strip()}")

        for line in s3("errors.txt").splitlines():  # error lines collected on the box from all logs
            sig = line.strip()[:160]
            if sig and sig not in st["errors"]:
                st["errors"].append(sig)
                emit(f"ERROR: {line.strip()[:200]}")

        gpu = s3("gpu.log").strip().splitlines()
        if gpu and st.get("state") == "running":
            last = gpu[-1].split(" ")[0]
            now = datetime.now(timezone.utc)
            t = datetime.strptime(f"{now:%Y-%m-%d} {last}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            age = (now - t).total_seconds() % 86400
            if age > 300 and not st.get("stale_alerted"):
                st["stale_alerted"] = 1
                emit(f"ALERT: no heartbeat for {age:.0f}s (phase {phase})")
            if phase and phase.startswith(WORK_PHASES):
                busy = any(float(l.split(", ")[1].rstrip(" %")) > 5 for l in gpu[-4:] if l.count(", ") >= 3)
                idle_since = None if busy else (idle_since or time.time())
                if idle_since and time.time() - idle_since > 600 and not st.get(f"idle:{phase}"):
                    st[f"idle:{phase}"] = 1
                    emit(f"ALERT: GPUs idle 10+ min in {phase}")

        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(st))

        finished = (phase or "").startswith(("done", "failed"))
        if finished or st.get("state") in TERMINAL_STATES and st.get("phase"):
            emit(f"RUN ENDED: phase {phase}, instance {st.get('state')}")
            report = sh("aws", "s3", "cp", f"{ARCHIVE}/report.md", "-").strip()
            failure = s3("failure.txt").strip()
            print(report or failure or "(no report or failure note in S3)", flush=True)
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
