"""Glimmer places a model's moving-part connections (hinges, turntables, axles, wheels, bushes, crank arms).

The task is run 4's placement task (gen_objects.place_case, audit_objects prompts and native checks): the
model without those parts, its requirements, and up to `rounds` answers, each checked; failures go back to
Glimmer with the checker's messages. The reference answer is never shown to the model. Every proposal, verdict
and the accepted state are recorded.
"""

import json
from pathlib import Path

from .assembly import save
from .audit_objects import inspect_case, messages, moving_groups
from .gen_objects import place_case
from .running_gear_task import FEEDBACK, edits_from


# What the research step calls an object -> the mechanism family that builds and checks it.
FAMILY_WORDS = {"chair": ("office chair", "swivel chair", "desk chair", "rolling chair", "chair"),
                "bicycle": ("bicycle", "bike", "cycle"), "skateboard": ("skateboard", "longboard"),
                "microwave": ("microwave",), "tv": ("television", "tv", "monitor"), "laptop": ("laptop", "notebook")}


def family_of(name: str, moving_parts: list[dict] | None = None) -> str | None:
    """The mechanism family for an object the research step identified (None: no mechanism builder for it)."""
    if moving_parts is not None and not moving_parts:
        return None  # e.g. a plain chair: the research found nothing that moves, so it stays a static model
    text = (name or "").lower()
    for family, words in FAMILY_WORDS.items():
        if any(w in text for w in words):
            return family
    return None


def model_for(family: str, target) -> tuple[list[dict], dict]:
    """The verified mechanism model of `family`, sized from the photo's target, and its checks context."""
    import random

    from .gen_objects import _model

    return _model(family, random.Random(0), None, target)


def car_door_context(parts: list[dict]) -> dict:
    from .doors import CAR_REQUIREMENTS

    groups = moving_groups(parts, {"hinge_top"})
    return {"requirements": CAR_REQUIREMENTS, "vehicle": True,
            "moving": {f"door_{i}": (g, 90) for i, g in enumerate(groups)}}


def place(parts: list[dict], family: str, ctx: dict, sampler, out: Path, *, rounds: int = 3,
          max_tokens: int = 8000, timeout: float = 180, only: set | None = None,
          k: int = 1) -> tuple[list[dict] | None, dict]:
    """(the accepted model or None, the trace). `only`: the part types left for Glimmer to place. `k`: answers asked
    for at once each round; the checker keeps one that passes, else the one with the fewest problems."""
    from concurrent.futures import ThreadPoolExecutor
    out.mkdir(parents=True, exist_ok=True)
    case = place_case(parts, family, ctx, f"pipeline_{family}", only)
    chat = messages(case)
    trace = {"family": family, "sampler": sampler.metadata, "removed": len(case["oracle"]),
             "input_parts": len(case["input"]), "rounds": [], "passed": False}
    accepted = None
    for rnd in range(1, rounds + 1):
        entry = {"round": rnd}

        def one(n):
            try:
                s = sampler(chat, max_tokens=max_tokens, temperature=0.0 if rnd == 1 and n == 0 else 0.7,
                            timeout=timeout)
                return s, inspect_case(case, edits_from(s.text))
            except Exception as e:  # noqa: BLE001 - a service or parse failure consumes the try
                return None, {"passed": False, "issues": [f"{type(e).__name__}: {e}"[:300]], "candidate": None,
                              "codes": []}

        if k > 1:
            with ThreadPoolExecutor(max_workers=k) as pool:
                tries = list(pool.map(one, range(k)))
        else:
            tries = [one(0)]
        sample, verdict = min(tries, key=lambda r: (not r[1]["passed"], r[0] is None, len(r[1]["issues"])))
        if sample is not None:
            entry.update(answer=sample.text, tokens=sample.tokens)
        entry.update(passed=verdict["passed"], codes=verdict.get("codes", []), issues=verdict["issues"][:12],
                     tries=k, passing_tries=sum(1 for _, v in tries if v["passed"]))
        trace["rounds"].append(entry)
        save(out / "trace.json", trace)
        if verdict["passed"]:
            accepted = verdict["candidate"]
            trace["passed"] = True
            break
        if sample is not None:
            chat = chat + [{"role": "assistant", "content": sample.text},
                           {"role": "user", "content": FEEDBACK.format(issues="\n".join(f"- {i}" for i in verdict["issues"][:12]))}]
    save(out / "trace.json", trace)
    if accepted is not None:
        save(out / "placed.json", accepted)
    return accepted, trace


def main() -> None:
    """Offline check with the reference answer standing in for the model (no inference)."""
    import sys

    from .assembly import Sample

    parts = json.loads(Path(sys.argv[1]).read_text())
    ctx = car_door_context(parts)

    class Oracle:
        metadata = {"source": "reference_answer_offline_check"}

        def __call__(self, chat, **_):
            return Sample(json.dumps(place_case(parts, "door", ctx, "x")["oracle"]), 0)

    placed, trace = place(parts, "door", ctx, Oracle(), Path("runs/learned_mechanisms_check"))
    print(json.dumps({"passed": trace["passed"], "removed": trace["removed"], "rounds": len(trace["rounds"])}))


if __name__ == "__main__":
    main()
