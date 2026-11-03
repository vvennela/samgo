"""A build kit for any verified model: what a person needs to build it in real LEGO.

From a run's artifacts (build.json, model.ldr, and manual.json when the build order was certified) writes:
- parts.csv: every LEGO element with its LDraw/BrickLink part number, colour and count;
- bricklink_wanted.xml: the same list as a BrickLink wanted list (Want -> Upload) to buy the parts;
- model.ldr: the model for BrickLink Studio / LDView (Studio can also export instructions from it);
- steps/step_NN.png + BUILD.md: the certified build order (samgo.manual), one picture per step, with the parts
  added in that step; when no certified order exists, the height-grouped schematic order of samgo.instructions.

    python -m samgo.build_kit <artifacts dir> <out dir> [--title "Camry"]
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

# LDraw colour code -> (BrickLink colour id, BrickLink colour name).
BRICKLINK_COLORS = {
    0: (11, "Black"), 1: (7, "Blue"), 2: (6, "Green"), 4: (5, "Red"), 14: (3, "Yellow"), 15: (1, "White"),
    25: (4, "Orange"), 19: (2, "Tan"), 70: (88, "Reddish Brown"), 71: (86, "Light Bluish Gray"),
    72: (85, "Dark Bluish Gray"), 272: (63, "Dark Blue"), 47: (12, "Trans-Clear"), 40: (13, "Trans-Black"),
}
# LDraw file -> BrickLink item number where they differ (most basic elements share the number).
BRICKLINK_IDS: dict[str, str] = {}


def census(ldr: str) -> Counter:
    """(part number, LDraw colour) -> count, from the type-1 lines of an LDraw model."""
    out = Counter()
    for line in ldr.splitlines():
        f = line.split()
        if len(f) >= 15 and f[0] == "1":
            out[(f[-1].removesuffix(".dat"), int(f[1]))] += 1
    return out


def wanted_xml(bom: Counter) -> str:
    rows = []
    for (part, color), n in sorted(bom.items()):
        bl_color = BRICKLINK_COLORS.get(color, (0, "?"))[0]
        rows.append(f"<ITEM><ITEMTYPE>P</ITEMTYPE><ITEMID>{BRICKLINK_IDS.get(part, part)}</ITEMID>"
                    f"<COLOR>{bl_color}</COLOR><MINQTY>{n}</MINQTY></ITEM>")
    return "<INVENTORY>\n" + "\n".join(rows) + "\n</INVENTORY>\n"


def plan(parts: list[dict], manual: dict | None) -> tuple[list[tuple[list[int], bool]], str]:
    """Steps as (part indices, built upside down?)."""
    if manual and manual.get("passed") and manual.get("steps"):
        groups = [(list(s["parts"]), bool((s.get("frame") or {}).get("flipped"))) for s in manual["steps"]]
        note = (f"Certified order (samgo.manual): every step is placed and held by stud forces within clutch, "
                f"checked exactly. Orientation: {manual.get('orientation') or 'upright'}.")
        return groups, note
    from .instructions import steps

    index = {id(p): i for i, p in enumerate(parts)}
    return ([([index[id(p)] for p in g], False) for g in steps(parts)],
            "Schematic order (layers bottom-up, then axles and wheels); not certified step by step.")


def write(art: Path, out: Path, title: str = "model") -> Path:
    from .instructions import _bounds, name, render_step

    parts = json.loads((art / "build.json").read_text())
    manual = json.loads((art / "manual.json").read_text()) if (art / "manual.json").exists() else None
    ldr = (art / "model.ldr").read_text()
    out.mkdir(parents=True, exist_ok=True)
    (out / "steps").mkdir(exist_ok=True)
    shutil.copy(art / "model.ldr", out / "model.ldr")

    bom = census(ldr)
    lines = ["part,bricklink_part,ldraw_color,bricklink_color,color_name,count"]
    for (part, color), n in sorted(bom.items(), key=lambda kv: (-kv[1], kv[0])):
        bl, cname = BRICKLINK_COLORS.get(color, (0, "?"))
        lines.append(f"{part},{BRICKLINK_IDS.get(part, part)},{color},{bl},{cname},{n}")
    (out / "parts.csv").write_text("\n".join(lines) + "\n")
    (out / "bricklink_wanted.xml").write_text(wanted_xml(bom))

    groups, note = plan(parts, manual)
    bounds = _bounds(parts)
    render_step([], parts, bounds, out / "steps" / "finished.png", f"{title}: finished")
    doc = [f"# Build {title} in LEGO", "", "![finished](steps/finished.png)", "",
           f"{len(parts)} model parts ({sum(bom.values())} LEGO elements: wheels are a tyre and a rim), "
           f"{len(groups)} steps.", "", note, "",
           "## Get the parts", "",
           "- `parts.csv`: part numbers, colours, counts.",
           "- `bricklink_wanted.xml`: on bricklink.com, Want -> Upload, paste it, then buy from the listed stores.",
           "- `model.ldr`: open in BrickLink Studio to inspect in 3D (Studio can also price the parts).", "",
           "| count | part | colour |", "|---|---|---|"]
    doc += [f"| {n} | {p} | {BRICKLINK_COLORS.get(c, (0, str(c)))[1]} |"
            for (p, c), n in sorted(bom.items(), key=lambda kv: (-kv[1], kv[0]))]
    def h(p):
        return 1 if "plate" in p["type"] else 3

    top = max(p["z"] + h(p) for p in parts)

    def turned(ps):  # the model as it sits on the table while that step is built upside down
        return [p | {"z": top - p["z"] - h(p)} for p in ps]

    done: list[dict] = []
    for k, (g, flipped) in enumerate(groups, 1):
        new = [parts[i] for i in g]
        img = out / "steps" / f"step_{k:02d}.png"
        label = f"Step {k} of {len(groups)}" + (" (upside down)" if flipped else "")
        render_step(turned(done) if flipped else done, turned(new) if flipped else new, bounds, img, label)
        doc += ["", f"## Step {k}" + (" (model upside down)" if flipped else ""), "", f"![step {k}](steps/{img.name})", ""]
        doc += [f"- {n} x {nm}" for nm, n in sorted(Counter(name(p) for p in new).items())]
        done += new
    (out / "BUILD.md").write_text("\n".join(doc) + "\n")
    return out / "BUILD.md"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifacts", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--title", default="model")
    a = ap.parse_args()
    print(write(a.artifacts, a.out, a.title))


if __name__ == "__main__":
    main()
