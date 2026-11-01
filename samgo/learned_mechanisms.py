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


def car_door_context(parts: list[dict]) -> dict:
    from .doors import CAR_REQUIREMENTS

    groups = moving_groups(parts, {"hinge_top"})
    return {"requirements": CAR_REQUIREMENTS, "vehicle": True,
            "moving": {f"door_{i}": (g, 90) for i, g in enumerate(groups)}}


def place(parts: list[dict], family: str, ctx: dict, sampler, out: Path, *, rounds: int = 3,
          max_tokens: int = 8000, timeout: float = 180, only: set | None = None) -> tuple[list[dict] | None, dict]:
    """(the accepted model or None, the trace). `only`: the part types left for Glimmer to place."""
    out.mkdir(parents=True, exist_ok=True)
    case = place_case(parts, family, ctx, f"pipeline_{family}", only)
    chat = messages(case)
    trace = {"family": family, "sampler": sampler.metadata, "removed": len(case["oracle"]),
             "input_parts": len(case["input"]), "rounds": [], "passed": False}
    accepted = None
    for rnd in range(1, rounds + 1):
        entry = {"round": rnd}
        try:
            sample = sampler(chat, max_tokens=max_tokens, temperature=0.0 if rnd == 1 else 0.7, timeout=timeout)
            entry.update(answer=sample.text, tokens=sample.tokens)
            edits = edits_from(sample.text)
            verdict = inspect_case(case, edits)
        except Exception as e:  # noqa: BLE001 - a service or parse failure consumes the round
            verdict = {"passed": False, "issues": [f"{type(e).__name__}: {e}"[:300]], "candidate": None, "codes": []}
            sample = None
        entry.update(passed=verdict["passed"], codes=verdict.get("codes", []), issues=verdict["issues"][:12])
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
