"""A BrickGPT-style caption of a photo from Muse Spark Open: shape and structure only, no color.

StableText2Brick captions read like "Square basket with layered construction and a recessed interior." The
fine-tuned model learned that register, so the photo is described the same way (color comes from the photo
itself, per part, after building).
"""

import asyncio
import os
from pathlib import Path

from samgo.function_spec import BASE_URL, MODEL, _data_url

PROMPT = """Describe the main object in this photo as a short caption for building it out of LEGO bricks.
Write one or two sentences about its shape and structure only: its overall form, main parts, proportions and
how they connect (for example "A tall chair with four thin legs, a flat square seat and a slatted backrest
that leans slightly back."). Do not mention color, material, brand, background or the photo itself.
Answer with the caption only."""


async def caption(photo: Path) -> str:
    from dotenv import load_dotenv
    from openai import AsyncOpenAI

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    client = AsyncOpenAI(api_key=os.environ["BASETEN_API_KEY"], base_url=BASE_URL, max_retries=6)
    r = await client.chat.completions.create(
        model=MODEL, max_tokens=3000, extra_body={"reasoning_effort": "low"},
        messages=[{"role": "user", "content": [{"type": "text", "text": PROMPT},
                                               {"type": "image_url", "image_url": {"url": _data_url(photo)}}]}])
    return (r.choices[0].message.content or "").strip().strip('"')


if __name__ == "__main__":
    import sys

    print(asyncio.run(caption(Path(sys.argv[1]))))
