"""The manual's finale (scripts/film_finale.py frames) in the film, placed exactly like the manual flip in
scripts/film_spec.py (same element, same header), so it continues the flip with no cut."""
import os
import runpy
from pathlib import Path

_S = runpy.run_path(str(Path(__file__).with_name("film_spec.py")))
FILM = _S["FILM"]
W, H, FPS, BG, BG_IMAGE, T0 = _S["W"], _S["H"], _S["FPS"], _S["BG"], _S["BG_IMAGE"], 0.0
CAMERA = _S["CAMERA"]
FINALE = os.environ.get("FINALE", f"{FILM}/finale")
DURATION = len(list(Path(FINALE).glob("0*.png"))) / FPS
flip = next(e for e in _S["ELEMENTS"] if str(e.get("src", "")).endswith("/flip"))
head = next(e for e in _S["ELEMENTS"] if str(e.get("src", "")).endswith("cap_manual.png"))
ELEMENTS = [dict(head, show=(0, 999), fade=0.0),
            dict({k: v for k, v in flip.items() if k not in ("schedule",)}, src=FINALE, start=0, show=(0, 999), fade=0.0)]
