"""S2.5: what must the object do? Muse Spark Open reads the photo and returns a function spec.

    {"object": "sports car", "kind": "vehicle", "wheels": 4, "axles": 2, "rolls_along": "length"}

`kind` decides the build: "vehicle" gets the rolling-car builder (chassis, axles, wheels), anything
else gets the structural builder. Out of scope for now: steering, doors, other mechanisms.

    uv run python -m samgo.function_spec photos/car.jpg [more photos...]
"""

import asyncio
import base64
import io
import json
import os
import re
import sys
from pathlib import Path

from PIL import Image

MODEL = "meta-models/muse-spark-open-1026"
BASE_URL = "https://inference.baseten.co/v1"
PROMPT = """Look at the photo and say what the main object is and how it moves, so it can be rebuilt from
LEGO-compatible bricks that work like the real thing.

Answer with one JSON object and nothing else:
{"object": "<short name>", "kind": "vehicle" or "static", "wheels": <number of road wheels, 0 if none>,
 "axles": <number of wheel axles, 0 if none>, "rolls_along": "length" or null}

"vehicle" means it rolls on wheels (cars, trucks, toy vehicles). Count wheels the real object has,
including ones hidden from view (a car has 4 even if the photo shows 2)."""


def _data_url(photo: Path, max_side: int = 1024) -> str:
    image = Image.open(photo).convert("RGB")
    image.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def parse(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None
    try:
        spec = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    if spec.get("kind") not in ("vehicle", "static"):
        return None
    return spec


async def infer(photo: Path, client=None) -> dict:
    from dotenv import load_dotenv
    from openai import AsyncOpenAI

    load_dotenv()
    client = client or AsyncOpenAI(api_key=os.environ["BASETEN_API_KEY"], base_url=BASE_URL, max_retries=6)
    r = await client.chat.completions.create(
        model=MODEL, max_tokens=4000, extra_body={"reasoning_effort": "low"},
        messages=[{"role": "user", "content": [{"type": "text", "text": PROMPT},
                                               {"type": "image_url", "image_url": {"url": _data_url(photo)}}]}])
    text = r.choices[0].message.content or ""
    return parse(text) or {"kind": "static", "error": f"unparseable answer: {text[:200]}"}


async def _main(photos: list[Path]) -> None:
    for photo, spec in zip(photos, await asyncio.gather(*(infer(p) for p in photos))):
        print(photo.name, json.dumps(spec))


if __name__ == "__main__":
    asyncio.run(_main([Path(p) for p in sys.argv[1:]]))
