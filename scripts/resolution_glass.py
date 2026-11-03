"""Spark's glass on each resolution-sweep body (same step as the pipeline): runs/resolution/<name>_<studs>_glass.json."""
import asyncio
import json
import sys
from pathlib import Path

from samgo import spark_glass

photo, name, obj = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
for studs in [int(s) for s in sys.argv[4:]]:
    parts = json.loads(Path(f"runs/resolution/{name}_{studs}.json").read_text())
    loop = asyncio.new_event_loop()
    try:
        out, rep = spark_glass.build(photo, parts, obj, Path(f"runs/resolution/glass_{name}_{studs}"), loop)
    finally:
        loop.close()
    Path(f"runs/resolution/{name}_{studs}_glass.json").write_text(json.dumps(out))
    print(name, studs, {k: v for k, v in rep.items() if k in ("cells", "glass_parts", "error")}, flush=True)
