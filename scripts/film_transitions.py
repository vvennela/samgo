"""Make the stud dissolves at the film's section cuts: for each cut time t in the spec's CUTS, render the film's frame
just before (t - 0.4 s) and just after (t + 0.4 s) without transitions, and write the dissolve between them to
FILM/trans/<i>/ (24 frames, 0.8 s). The spec places each one over its cut.

    PYTHONPATH=. python scripts/film_transitions.py scripts/film_spec.py
"""
import importlib.util
import os
import runpy
import sys
from pathlib import Path

os.environ["FILM_NO_TRANSITIONS"] = "1"
HERE = Path(__file__).resolve().parent
spec_path = sys.argv[1]

film_spec = importlib.util.spec_from_file_location("film", HERE / "film.py")
film = importlib.util.module_from_spec(film_spec)
film_spec.loader.exec_module(film)
sys.path.insert(0, str(HERE))
from film_dissolve import frames  # noqa: E402
from PIL import Image  # noqa: E402

S = runpy.run_path(spec_path)
film.SPEC.update(S)
film.SPEC.update({"T0": 0.0, "BLUR": 1})
out = Path(S["FILM"]) / "trans"
for i, t in enumerate(S["CUTS"]):
    a = film.render_at(t - 0.4)
    b = film.render_at(t + 0.4)
    d = out / str(i)
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.png"):
        f.unlink()
    for k, fr in enumerate(frames(a, b, 24, 30)):
        Image.fromarray(fr.astype("uint8")).save(d / f"{k:05d}.png")
    print("cut", i, "at", round(t, 2), "->", d, flush=True)
