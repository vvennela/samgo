"""S1: photo -> 3D mesh.

SAM 3.1 (Meta Model API) finds the named object and returns its mask; SAM 3D
Objects (fal.ai) turns the photo plus that mask into a GLB mesh.

Needs MODEL_API_KEY and FAL_KEY in the environment or .env.
"""

import argparse
import asyncio
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import fal_client
import httpx
import numpy as np
from dotenv import load_dotenv
from meta_sam_parser import SegmentationMaskRecord, decode_mask_to_raster, image_segmentation_format, parse_responses_stream
from openai import AsyncOpenAI
from PIL import Image, ImageOps

META_BASE_URL = "https://api.meta.ai/v1"
SAM_MODEL = "sam-3.1"
SAM_3D_ENDPOINT = "fal-ai/sam-3/3d-objects"
MAX_SIDE = 2048  # downscale large phone photos before uploading


@dataclass
class Perception:
    prompt: str
    image: str
    mask: str
    mesh: str
    confidence: float | None
    masks_found: int


def prepare_photo(src: Path, dest: Path) -> Image.Image:
    """Apply EXIF rotation and downscale, so every model sees the same pixels."""
    image = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    image.save(dest)
    return image


def decode_masks(records: list) -> list[tuple[np.ndarray, float | None]]:
    """SAM 3.1 mask records -> (0/1 raster at SAM's resolution, confidence)."""
    out = []
    for r in records:
        if isinstance(r, SegmentationMaskRecord):
            bits = np.frombuffer(decode_mask_to_raster(r.mask), dtype=np.uint8)
            out.append((bits.reshape(r.mask.height, r.mask.width), r.confidence))
    return out


def best_mask(candidates: list[tuple[np.ndarray, float | None]], size: tuple[int, int]) -> tuple[np.ndarray, float | None]:
    """Pick the most confident mask (largest if no confidence) and scale it to the photo's (width, height)."""
    if not candidates:
        raise RuntimeError("SAM 3.1 found no matching object")
    raster, confidence = max(candidates, key=lambda c: (c[1] if c[1] is not None else -1, int(c[0].sum())))
    small = Image.fromarray(raster.astype(np.uint8) * 255)
    return np.asarray(small.resize(size, Image.NEAREST)) > 127, confidence


async def segment(image_url: str, prompt: str, size: tuple[int, int], client: AsyncOpenAI) -> tuple[np.ndarray, float | None, int]:
    stream = await client.responses.create(
        model=SAM_MODEL,
        input=[{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": image_url},
        ]}],
        metadata={"include_confidence": "true"},
        stream=True,
    )
    result = await parse_responses_stream(stream, image_segmentation_format()).final_result()
    candidates = decode_masks(list(result.records))
    mask, confidence = best_mask(candidates, size)
    return mask, confidence, len(candidates)


async def reconstruct(image_url: str, mask_url: str) -> str:
    result = await fal_client.subscribe_async(SAM_3D_ENDPOINT, arguments={"image_url": image_url, "mask_urls": [mask_url]})
    if "model_glb" not in result:
        raise RuntimeError(f"SAM 3D returned no mesh: {sorted(result)}")
    return result["model_glb"]["url"]


async def perceive(photo: Path, prompt: str, out: Path) -> Perception:
    load_dotenv()
    out.mkdir(parents=True, exist_ok=True)
    image_path, mask_path, mesh_path = out / "photo.png", out / "mask.png", out / "object.glb"

    image = prepare_photo(photo, image_path)
    image_url = await fal_client.upload_file_async(image_path)

    client = AsyncOpenAI(api_key=os.environ["MODEL_API_KEY"], base_url=META_BASE_URL)
    mask, confidence, count = await segment(image_url, prompt, image.size, client)
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)  # white object on black
    mask_url = await fal_client.upload_file_async(mask_path)

    glb_url = await reconstruct(image_url, mask_url)
    async with httpx.AsyncClient(timeout=120) as http:
        response = await http.get(glb_url)
        response.raise_for_status()
        mesh_path.write_bytes(response.content)

    result = Perception(prompt, str(image_path), str(mask_path), str(mesh_path), confidence, count)
    (out / "perception.json").write_text(json.dumps(asdict(result), indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Photo -> masked object -> GLB mesh")
    parser.add_argument("photo", type=Path)
    parser.add_argument("prompt", help='short noun phrase naming the object, e.g. "wooden chair"')
    parser.add_argument("-o", "--out", type=Path, default=None, help="output folder (default: runs/<photo name>)")
    args = parser.parse_args()
    result = asyncio.run(perceive(args.photo, args.prompt, args.out or Path("runs") / args.photo.stem))
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
