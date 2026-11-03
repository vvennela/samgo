"""Spark judges whether the real object is mirror-symmetric left to right (its world knowledge, from the photo).
If it is, the body is voxelized exactly mirrored (samgo.voxelize exact_mirror), so one-sided noise from the mesh (a
bump on one side, a window on one side) does not make the LEGO model lopsided. Cached per photo and object."""

import asyncio
import hashlib
import json
import re
from pathlib import Path

PROMPT = """This is a photo of {obj}. From what you know about this kind of object in the real world: is it
mirror-symmetric left to right (like a car, an airplane or a desk fan seen from the front), apart from small
one-sided details? Answer with one JSON object and nothing else: {{"symmetric": true | false}}"""


async def _ask(photo: Path, obj: str) -> dict:
    import os

    from dotenv import load_dotenv
    from openai import AsyncOpenAI

    from .function_spec import BASE_URL, MODEL, _data_url

    key = hashlib.sha256(Path(photo).read_bytes() + (PROMPT + obj).encode()).hexdigest()[:16]
    cache = Path("runs/research_cache") / f"symmetry_{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    client = AsyncOpenAI(api_key=os.environ["BASETEN_API_KEY"], base_url=BASE_URL, max_retries=6)
    text = PROMPT.format(obj=f"a{'n' if obj[:1] in 'aeiou' else ''} {obj}")
    for _ in range(3):
        r = await client.chat.completions.create(
            model=MODEL, max_tokens=2000, extra_body={"reasoning_effort": "low"},
            messages=[{"role": "user", "content": [{"type": "text", "text": text},
                                                   {"type": "image_url", "image_url": {"url": _data_url(photo)}}]}])
        m = re.search(r"\{.*\}", r.choices[0].message.content or "", re.S)
        try:
            out = {"symmetric": bool(json.loads(m.group())["symmetric"])}
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out))
        return out
    return {"error": "no answer"}


def symmetric(photo: Path, obj: str) -> bool | None:
    try:
        return asyncio.run(_ask(photo, obj)).get("symmetric")
    except Exception:  # noqa: BLE001 - no Spark: the voxelizer's own per-section rule applies
        return None
