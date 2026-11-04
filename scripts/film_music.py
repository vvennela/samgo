"""A soft ambient music bed, synthesized here (no samples, no licences): slow major-seventh chords as warm pads with a
gentle swell, a quiet low root, and a soft high shimmer. Meant to sit about 22 dB under the voice.

    python scripts/film_music.py out.wav seconds [bpm=72]
"""
import sys
import wave

import numpy as np

SR = 48000


def note(f: float, t: np.ndarray) -> np.ndarray:
    """A warm pad voice: a few detuned partials with a slow vibrato."""
    v = 0.0015 * np.sin(2 * np.pi * 0.21 * t)
    out = np.zeros_like(t)
    for det, amp in ((-0.0035, 1.0), (0.0, 1.0), (0.0042, 0.9)):
        ph = 2 * np.pi * f * (1 + det + v) * t
        out += amp * (np.sin(ph) + 0.28 * np.sin(2 * ph) + 0.09 * np.sin(3 * ph))
    return out / 3


def chord(freqs, dur, fade=1.6):
    t = np.arange(int(dur * SR)) / SR
    env = np.minimum(1, t / fade) * np.minimum(1, (dur - t) / fade)
    env = env ** 1.5
    return sum(note(f, t) for f in freqs) * env / len(freqs)


def main():
    out, secs = sys.argv[1], float(sys.argv[2])
    bpm = float(sys.argv[3]) if len(sys.argv) > 3 else 72
    bar = 4 * 60 / bpm * 2                               # two bars per chord: slow
    A = lambda n: 440 * 2 ** ((n - 69) / 12)
    prog = [  # Cmaj9, Am9, Fmaj7(#11), Gsus2 (MIDI notes)
        [48, 55, 59, 62, 64], [45, 52, 55, 59, 60], [41, 48, 52, 57, 59], [43, 50, 55, 57, 62]]
    n = int(secs * SR) + SR * 4
    mix = np.zeros(n)
    t0, i = 0.0, 0
    while t0 < secs + bar:
        notes = prog[i % len(prog)]
        c = chord([A(m) for m in notes[1:]], bar + 1.6)
        bass = chord([A(notes[0] - 12)], bar + 1.6) * 0.8
        a = int(t0 * SR)
        seg = c + bass
        mix[a:a + len(seg)] += seg[: max(0, min(len(seg), n - a))]
        t0 += bar; i += 1
    # a soft shimmer: high sine taps on the chord tones, very quiet
    rng = np.random.default_rng(3)
    tap_t = np.arange(int(0.9 * SR)) / SR
    for k in range(int(secs / (60 / bpm))):
        if rng.random() < 0.45:
            notes = prog[int(k * 60 / bpm / bar) % len(prog)]
            f = A(rng.choice(notes[2:]) + 24)
            tap = np.sin(2 * np.pi * f * tap_t) * np.exp(-tap_t * 5) * 0.06
            a = int(k * 60 / bpm * SR)
            mix[a:a + len(tap)] += tap[: max(0, min(len(tap), n - a))]
    # gentle low-pass and fade in/out, normalised to -18 dBFS peak
    k = np.exp(-np.arange(64) / 10.0); k /= k.sum()
    mix = np.convolve(mix, k, mode="same")
    mix = mix[: int(secs * SR)]
    fade = np.minimum(1, np.arange(len(mix)) / (SR * 2.5)) * np.minimum(1, (len(mix) - np.arange(len(mix))) / (SR * 3))
    mix *= fade
    mix /= np.max(np.abs(mix)) + 1e-9
    mix *= 10 ** (-18 / 20)
    st = np.stack([mix, np.roll(mix, int(0.012 * SR))], axis=1)       # a touch of width
    with wave.open(out, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((st * 32767).astype("<i2").tobytes())
    print("music", secs, "s ->", out)


if __name__ == "__main__":
    main()
