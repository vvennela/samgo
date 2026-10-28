"""Kill check: does SAM 3.1 return one mask per visible wheel?

Usage: uv run python scripts/wheel_check.py photos/car.jpg [more photos...]
Writes runs/wheel_check/<name>.png overlays and prints mask boxes.
"""

import asyncio
import os
import sys
from pathlib import Path

import fal_client
import numpy as np
from dotenv import load_dotenv
from meta_sam_parser import SegmentationMaskRecord, image_segmentation_format, parse_responses_stream
from openai import AsyncOpenAI
from PIL import Image, ImageDraw

from samgo.perceive import META_BASE_URL, SAM_MODEL, decode_masks, prepare_photo

COLORS = [(255, 60, 60), (60, 160, 255), (60, 220, 90), (255, 200, 0), (200, 80, 255), (0, 220, 220)]


async def check(photo: Path, client: AsyncOpenAI, out: Path) -> None:
    image = prepare_photo(photo, out / f"{photo.stem}-input.png")
    url = await fal_client.upload_file_async(out / f"{photo.stem}-input.png")
    stream = await client.responses.create(
        model=SAM_MODEL,
        input=[{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": "wheel"}, {"type": "input_image", "image_url": url}]}],
        metadata={"include_confidence": "true"},
        stream=True,
    )
    result = await parse_responses_stream(stream, image_segmentation_format()).final_result()
    masks = decode_masks(list(result.records), image.size)
    overlay = image.convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    print(f"{photo.name}: {len(masks)} wheel mask(s), image {image.size}")
    for i, (raster, confidence) in enumerate(masks):
        full = raster
        ys, xs = np.nonzero(full)
        color = COLORS[i % len(COLORS)]
        tint = np.zeros((*full.shape, 4), dtype=np.uint8)
        tint[full] = (*color, 110)
        layer = Image.alpha_composite(layer, Image.fromarray(tint))
        draw = ImageDraw.Draw(layer)
        if len(xs):
            box = (xs.min(), ys.min(), xs.max(), ys.max())
            draw.rectangle(box, outline=(*color, 255), width=4)
            print(f"  mask {i}: conf={confidence} box x {box[0]}-{box[2]} y {box[1]}-{box[3]} "
                  f"(w {box[2] - box[0]}, h {box[3] - box[1]}), area {int(full.sum())}")
    Image.alpha_composite(overlay, layer).convert("RGB").save(out / f"{photo.stem}.png")


async def main() -> None:
    load_dotenv()
    out = Path("runs/wheel_check")
    out.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(api_key=os.environ["MODEL_API_KEY"], base_url=META_BASE_URL)
    for photo in sys.argv[1:]:
        await check(Path(photo), client, out)


if __name__ == "__main__":
    asyncio.run(main())
