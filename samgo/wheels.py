"""Wheel positions for the car builder, from SAM 3.1 "wheel" masks on the photo (S1).

Masks are 2D and the target is 3D, so only ratios are used: each axle's position along the car's
length, as a fraction of the object mask's horizontal extent, mapped onto the target's length (x).
Masks that overlap in x are one axle (a partly hidden far wheel shows up as its own mask). A side
view gives round wheel masks; a 3/4 view squashes them, so the fallback (wheels at 15% / 85%)
is used, and flagged, unless there are exactly two axles with roundish masks.

    uv run python -m samgo.wheels runs/s1/car       # reads photo.png + mask.png, writes wheels.json
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np

FALLBACK = (0.15, 0.85)
ROUND = (0.6, 1.4)        # wheel mask width/height for a side-ish view
MIN_AREA = 0.002          # of the object mask, to drop specks


def _box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def wheel_fractions(obj: np.ndarray, wheels: list[np.ndarray]) -> dict:
    """{'fractions': (front, rear), 'source': 'masks' | 'fallback', 'reason': ..., 'axles': [...]}."""
    ox0, _, ox1, _ = _box(obj)
    length = ox1 - ox0
    boxes = [_box(m) for m in wheels if m.sum() > MIN_AREA * obj.sum()]
    axles: list[list[int]] = []  # merged x-ranges with their tallest mask height and width
    for x0, y0, x1, y1 in sorted(boxes):
        if axles and min(x1, axles[-1][1]) - max(x0, axles[-1][0]) > 0.3 * min(x1 - x0, axles[-1][1] - axles[-1][0]):
            a = axles[-1]
            a[0], a[1] = min(a[0], x0), max(a[1], x1)
            if y1 - y0 > a[3]:
                a[2], a[3] = x1 - x0, y1 - y0
        else:
            axles.append([x0, x1, x1 - x0, y1 - y0])
    info = {"axles": [{"center": round(((a[0] + a[1]) / 2 - ox0) / length, 3), "roundness": round(a[2] / a[3], 2),
                       "diameter": round(a[3] / length, 3)} for a in axles]}
    if len(axles) != 2:
        return {"fractions": FALLBACK, "source": "fallback", "reason": f"{len(axles)} axle(s) found", **info}
    if not all(ROUND[0] <= a["roundness"] <= ROUND[1] for a in info["axles"]):
        return {"fractions": FALLBACK, "source": "fallback", "reason": "wheels not round: not a side view", **info}
    centers = tuple(sorted(a["center"] for a in info["axles"]))
    return {"fractions": centers, "source": "masks", "reason": "", **info}


async def detect(folder: Path) -> dict:
    """SAM 3.1 'wheel' on folder/photo.png, ratios against folder/mask.png; writes folder/wheels.json."""
    import fal_client
    from dotenv import load_dotenv
    from meta_sam_parser import image_segmentation_format, parse_responses_stream
    from openai import AsyncOpenAI
    from PIL import Image

    from .perceive import META_BASE_URL, SAM_MODEL, decode_masks

    load_dotenv()
    photo = folder / "photo.png"
    obj = np.array(Image.open(folder / "mask.png").convert("L")) > 127
    url = await fal_client.upload_file_async(photo)
    client = AsyncOpenAI(api_key=os.environ["MODEL_API_KEY"], base_url=META_BASE_URL)
    stream = await client.responses.create(
        model=SAM_MODEL, metadata={"include_confidence": "true"}, stream=True,
        input=[{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "wheel"}, {"type": "input_image", "image_url": url}]}])
    result = await parse_responses_stream(stream, image_segmentation_format()).final_result()
    masks = [m for m, _ in decode_masks(list(result.records), obj.shape[::-1])]
    out = wheel_fractions(obj, masks) | {"masks": len(masks)}
    (folder / "wheels.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(asyncio.run(detect(Path(sys.argv[1]))), indent=2))
