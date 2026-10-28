"""Pick training levels from the baseline: the model's 10-90% success band.

Writes <out>/plan.json with train_levels, the held-out transfer level (the next
harder level above the band, never trained on), and go/stop.
Usage: python -m pilot.band <out_dir>
"""

import json
import sys
from pathlib import Path

LEVELS = ["1", "2", "5", "10", "20", "all"]


def plan(rates: dict[str, float]) -> dict:
    rates = {lv: rates[lv] for lv in LEVELS if lv in rates}
    if rates.get("1", 0) < 0.02:
        return {"decision": "stop", "reason": "level 1 below 2%: the model can't read the task", "rates": rates}
    if all(p > 0.90 for p in rates.values()):
        return {"decision": "stop", "reason": "every level above 90%: nothing to learn here", "rates": rates}
    band = [lv for lv, p in rates.items() if 0.10 <= p <= 0.90]
    if not band:  # success jumps straight from >90% to <10%: train on the two levels around the jump
        easy = [lv for lv, p in rates.items() if p > 0.90]
        band = [easy[-1], LEVELS[LEVELS.index(easy[-1]) + 1]]
    harder = [lv for lv in LEVELS[LEVELS.index(band[-1]) + 1:] if lv in rates]
    if harder:
        transfer = harder[0]
    elif len(band) >= 2:  # band reaches the hardest level: hold that one out instead
        band, transfer = band[:-1], band[-1]
    else:
        transfer = None
    return {"decision": "go", "train_levels": band, "transfer_level": transfer, "rates": rates}


if __name__ == "__main__":
    out = Path(sys.argv[1])
    rates = json.loads((out / "base_test/progress.json").read_text())["pass_rate_by_level"]
    p = plan(rates)
    (out / "plan.json").write_text(json.dumps(p, indent=1))
    print(json.dumps(p))
