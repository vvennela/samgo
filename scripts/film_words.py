"""Word timings for the film: any recording (video or audio) -> words.json [{"w": "SAMgo", "t0": 7.0, "t1": 8.04}, ...]
via whisper.cpp (one word per segment). The film's captions and beat times are read from these.

    python scripts/film_words.py recording.mov words.json [offset_seconds]
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

MODEL = Path.home() / ".cache/whisper/ggml-base.en.bin"


def words(media: str, offset: float = 0.0) -> list[dict]:
    tmp = Path(tempfile.mkdtemp())
    wav = tmp / "a.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", media, "-ar", "16000", "-ac", "1", str(wav)], check=True)
    subprocess.run(["whisper-cli", "-m", str(MODEL), "-f", str(wav), "-ml", "1", "-sow", "-oj", "-of", str(tmp / "w"),
                    "-np"], check=True, capture_output=True)
    data = json.loads((tmp / "w.json").read_text())
    out = []
    for seg in data.get("transcription", []):
        w = seg["text"].strip()
        if not w:
            continue
        t0 = seg["offsets"]["from"] / 1000 + offset
        t1 = seg["offsets"]["to"] / 1000 + offset
        out.append({"w": w, "t0": round(t0, 3), "t1": round(t1, 3)})
    return out


def main():
    media, out = sys.argv[1], Path(sys.argv[2])
    off = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    ws = words(media, off)
    out.write_text(json.dumps(ws, indent=0))
    print(len(ws), "words:", " ".join(w["w"] for w in ws))


if __name__ == "__main__":
    main()
