"""Cut a voice-over recording into its sections and tighten it: every pause inside a section longer than MAX is
shortened to KEEP (with short fades so no click), leading and trailing silence trimmed. Pauses are found from the audio
(frames quieter than THRESH dB).

    python scripts/film_vo.py recording.m4a out_dir "name:t0-t1" ...
writes out_dir/<name>.wav (48 kHz mono) and out_dir/sections.json (durations)
"""
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

SR, HOP, THRESH, MAX, KEEP, FADE = 48000, 0.02, -42.0, 0.45, 0.40, 0.012


def load(path: str) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768


def quiet_runs(a: np.ndarray) -> list[tuple[int, int]]:
    win = int(HOP * SR)
    n = len(a) // win
    e = np.sqrt((a[: n * win].reshape(n, win) ** 2).mean(axis=1))
    q = 20 * np.log10(e + 1e-9) < THRESH
    runs, s = [], None
    for i, v in enumerate(q):
        if v and s is None:
            s = i
        if not v and s is not None:
            runs.append((s * win, i * win)); s = None
    if s is not None:
        runs.append((s * win, n * win))
    return runs


def tighten(a: np.ndarray) -> np.ndarray:
    runs = quiet_runs(a)
    if runs and runs[0][0] == 0:                       # leading silence: keep 0.1 s
        a = a[max(0, runs[0][1] - int(0.1 * SR)):]
        runs = quiet_runs(a)
    if runs and runs[-1][1] >= len(a) - int(HOP * SR):  # trailing silence: keep 0.25 s
        a = a[: runs[-1][0] + int(0.25 * SR)]
        runs = quiet_runs(a)
    out, pos = [], 0
    f = int(FADE * SR)
    ramp = np.linspace(1, 0, f, dtype=np.float32)
    for s, e in runs:
        if (e - s) / SR <= MAX:
            continue
        keep = int(KEEP * SR)
        head = a[pos: s + keep // 2].copy()
        head[-f:] *= ramp
        out.append(head)
        pos = e - keep // 2
        nxt = a[pos: pos + f].copy()
        a = a.copy(); a[pos: pos + f] = nxt * ramp[::-1]
    out.append(a[pos:])
    return np.concatenate(out)


def main():
    src, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    a = load(src)
    info = {}
    for spec in sys.argv[3:]:
        name, rng = spec.split(":")
        t0, t1 = (float(v) for v in rng.split("-"))
        seg = tighten(a[int(t0 * SR): int(t1 * SR)])
        with wave.open(str(out / f"{name}.wav"), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
            w.writeframes((np.clip(seg, -1, 1) * 32767).astype("<i2").tobytes())
        info[name] = round(len(seg) / SR, 2)
        print(f"{name}: {t1 - t0:.1f}s -> {len(seg) / SR:.1f}s")
    (out / "sections.json").write_text(json.dumps(info))


if __name__ == "__main__":
    main()
