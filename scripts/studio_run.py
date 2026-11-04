"""Run the real SAMgo pipeline with every live model call logged, for SAMgo Studio (scripts/studio.py).

Nothing in the pipeline changes: the client libraries are wrapped here, so each call to SAM 3.1 (Meta Model API),
Muse Spark (Baseten), SAM 3D (fal) and Muse Glimmer (Fireworks) appends one line to <out>/calls.jsonl when it starts
and when it returns: model, what was asked, seconds, and a short excerpt of the answer.

    PYTHONPATH=. python scripts/studio_run.py <pipeline args...>        (same arguments as python -m samgo.pipeline)
"""
import functools
import json
import sys
import threading
import time
from pathlib import Path

LOG = None
LOCK = threading.Lock()


def log(**rec):
    rec["t"] = round(time.time(), 3)
    with LOCK:
        with open(LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")


def excerpt(text, n=160):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n - 1] + "…"


def ask_of(kwargs):
    """A short label of what a chat call asks: the last user text."""
    for m in reversed(kwargs.get("messages") or []):
        c = m.get("content")
        if isinstance(c, list):
            c = " ".join(x.get("text", "") for x in c if isinstance(x, dict) and x.get("type") == "text")
        if c:
            return excerpt(c, 120)
    return ""


def patch():
    import openai.resources.chat.completions as chat
    import openai.resources.responses as resp

    orig_chat = chat.AsyncCompletions.create

    @functools.wraps(orig_chat)
    async def chat_create(self, *a, **kw):
        base = str(getattr(self._client, "base_url", ""))
        name = "Muse Spark" if "baseten" in base else kw.get("model", "chat")
        log(ev="start", model=name, id=kw.get("model"), ask=ask_of(kw))
        t0 = time.time()
        r = await orig_chat(self, *a, **kw)
        try:
            out = r.choices[0].message.content
        except Exception:  # noqa: BLE001
            out = ""
        log(ev="done", model=name, id=kw.get("model"), secs=round(time.time() - t0, 1), out=excerpt(out))
        return r
    chat.AsyncCompletions.create = chat_create

    orig_resp = resp.AsyncResponses.create

    @functools.wraps(orig_resp)
    async def resp_create(self, *a, **kw):
        model = kw.get("model", "")
        name = "SAM 3.1" if "sam" in str(model).lower() else str(model)
        text = ""
        for item in kw.get("input") or []:
            for c in (item.get("content") if isinstance(item, dict) else []) or []:
                if isinstance(c, dict) and c.get("type") in ("input_text", "text"):
                    text = c.get("text", "")
        log(ev="start", model=name, id=model, ask=excerpt(text, 80))
        t0 = time.time()
        r = await orig_resp(self, *a, **kw)
        log(ev="done", model=name, id=model, secs=round(time.time() - t0, 1), out="masks returned")
        return r
    resp.AsyncResponses.create = resp_create

    import fal_client
    orig_fal = fal_client.subscribe_async

    @functools.wraps(orig_fal)
    async def fal_subscribe(endpoint, *a, **kw):
        log(ev="start", model="SAM 3D", id=endpoint, ask="one photo + SAM 3.1's mask -> 3D mesh")
        t0 = time.time()
        r = await orig_fal(endpoint, *a, **kw)
        log(ev="done", model="SAM 3D", id=endpoint, secs=round(time.time() - t0, 1), out="textured mesh (.glb)")
        return r
    fal_client.subscribe_async = fal_subscribe

    from samgo import assembly
    orig_sample = assembly.FireworksSampler._sample

    @functools.wraps(orig_sample)
    def glimmer(self, messages, **kw):
        ask = ask_of({"messages": messages})
        log(ev="start", model="Muse Glimmer", id=str(self.model or "base"), ask=excerpt(ask, 120))
        t0 = time.time()
        r = orig_sample(self, messages, **kw)
        log(ev="done", model="Muse Glimmer", id=str(self.model or "base"), secs=round(time.time() - t0, 1),
            out=excerpt(r.text, 220), tokens=r.tokens)
        return r
    assembly.FireworksSampler._sample = glimmer


def main():
    global LOG
    args = sys.argv[1:]
    out = Path(args[args.index("-o") + 1] if "-o" in args else args[args.index("--out") + 1])
    out.mkdir(parents=True, exist_ok=True)
    LOG = out / "calls.jsonl"
    LOG.write_text("")
    patch()
    from samgo import pipeline
    sys.argv = ["samgo.pipeline"] + args
    log(ev="run", model="SAMgo", ask=" ".join(args))
    try:
        pipeline.main()
    finally:                       # pipeline.main exits with the run's status: log the end either way
        log(ev="end", model="SAMgo")


if __name__ == "__main__":
    main()
