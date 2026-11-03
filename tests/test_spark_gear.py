import asyncio

import numpy as np

from samgo import spark_gear
from samgo.builder import build
from samgo.parts import COLOR_INDEX


def test_model_is_not_left_raised_when_no_leg_is_built(monkeypatch, tmp_path):
    g = np.zeros((8, 8, 6), int)
    g[:] = COLOR_INDEX["white"]
    body = build(g, attempts=1, pull=lambda p: 0.0).pieces

    async def ask(*a, **k):
        return {"legs": [{"name": "foot", "x": 2, "y": 2}]}

    monkeypatch.setattr(spark_gear, "ask", ask)
    monkeypatch.setattr(spark_gear, "site", lambda *a, **k: None)  # no buildable spot: nothing gets built
    loop = asyncio.new_event_loop()
    try:
        parts, report = spark_gear.build(tmp_path / "p.jpg", body, "thing", None, tmp_path, loop)
    finally:
        loop.close()
    assert parts == body and "lift" not in report
