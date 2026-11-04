"""The final cut to the narration: each section's footage is fitted to its voice-over (held on its last frame, or a
moving shot slowed up to MAX_SLOW, so it lasts the line plus a breath), clips are joined with stud dissolves, and the
voice sections start just after each dissolve. Music under everything, lower while the voice speaks.

    python scripts/film_final.py plan.json out.mp4
plan.json: {"sections": [{"clips": [...], "vo": "x.wav" | null, "slow": true|false}, ...], "intro_audio": path|null,
            "music": path}
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DISSOLVE = 22 / 30          # film_join's dissolve length
LEAD, TAIL, MAX_SLOW = 0.35, 0.6, 1.25


def dur(p) -> float:
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                                capture_output=True, text=True, check=True).stdout)


def fit(clip: Path, extra: float, slow: bool, out: Path) -> Path:
    """clip made `extra` seconds longer: slowed (up to MAX_SLOW) if slow, then held on its last frame."""
    if extra <= 0.02:
        return clip
    d = dur(clip)
    k = min(MAX_SLOW, (d + extra) / d) if slow else 1.0
    hold = max(0.0, d + extra - d * k)
    vf = (f"setpts={k}*PTS," if k > 1.0 else "") + f"tpad=stop_mode=clone:stop_duration={hold:.3f},fps=30"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(clip), "-vf", vf, "-an", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "18", str(out)], check=True)
    return out


def main():
    plan = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    tmp = Path(tempfile.mkdtemp())
    clips, starts, t = [], [], 0.0
    for si, s in enumerate(plan["sections"]):
        cs = [Path(c) for c in s["clips"]]
        if s.get("speed"):                   # footage longer than its line: play it faster first
            sp = []
            for ci, c in enumerate(cs):
                q = tmp / f"sp{si}_{ci}.mp4"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(c), "-vf", f"setpts=PTS/{s['speed']},fps=30",
                                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(q)], check=True)
                sp.append(q)
            cs = sp
        vo = s.get("vo")
        need = (dur(vo) + LEAD + TAIL) if vo else 0.0
        have = sum(dur(c) for c in cs) + DISSOLVE * (len(cs) - 1)
        extra = max(0.0, need - have)
        total = sum(dur(c) for c in cs)
        fitted = []
        for ci, c in enumerate(cs):           # spread the extra time over the section's clips by their length
            share = extra * dur(c) / total
            fitted.append(fit(c, share, s.get("slow", False), tmp / f"s{si}_{ci}.mp4"))
        if clips:
            t += DISSOLVE
        starts.append(t)
        for ci, c in enumerate(fitted):
            if ci:
                t += DISSOLVE
            t += dur(c)
        clips += fitted
    video = tmp / "video.mp4"
    subprocess.run([sys.executable, str(HERE / "film_join.py"), str(video), *map(str, clips)], check=True)
    total = dur(video)
    # audio: voice sections at their section starts; the intro's own sound; music under, ducked under the voice
    inputs, filters, labels = ["-i", str(video)], [], []
    n = 1
    if plan.get("intro_audio"):
        inputs += ["-i", plan["intro_audio"]]
        filters.append(f"[{n}:a]atrim=0:{dur(plan['sections'][0]['clips'][0]):.3f},aresample=48000[ia]")
        labels.append("[ia]"); n += 1
    voices = []
    for s, st in zip(plan["sections"], starts):
        if s.get("vo"):
            inputs += ["-i", s["vo"]]
            ms = int((st + LEAD) * 1000)
            filters.append(f"[{n}:a]aresample=48000,volume=1.6,adelay={ms}|{ms}[v{n}]")
            voices.append(f"[v{n}]"); n += 1
    for path, at, gain in plan.get("extra_audio", []):      # e.g. the title's boom
        inputs += ["-i", path]
        ms = int(at * 1000)
        filters.append(f"[{n}:a]aresample=48000,volume={gain},adelay={ms}|{ms}[x{n}]")
        labels.append(f"[x{n}]"); n += 1
    inputs += ["-i", plan["music"]]
    filters.append(f"[{n}:a]aresample=48000,volume=0.55[mu]")
    vmix = "".join(voices)
    filters.append(f"{vmix}amix=inputs={len(voices)}:normalize=0[vo]")
    filters.append("[vo]asplit[vo1][vo2]")
    filters.append("[mu][vo2]sidechaincompress=threshold=0.02:ratio=6:attack=20:release=400[mud]")
    filters.append(f"{''.join(labels)}[vo1][mud]amix=inputs={len(labels) + 2}:duration=longest:normalize=0,"
                   f"atrim=0:{total:.3f},alimiter=limit=0.89:level=false[a]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(filters), "-map", "0:v",
                    "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(out)], check=True)
    print(f"final -> {out}  {total:.1f}s; section starts {[round(s, 1) for s in starts]}")


if __name__ == "__main__":
    main()
