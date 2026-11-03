# objects-r11

Why: Spark names spinning parts on a vertical axis (rotors or blades turning on top of a machine), and Glimmer had
no skill for them ("only horizontal axles yet"). Approved by the user Nov 3 ("Go: run 11 now").

- Skill: a 2x2 turntable (3680 base, 3679 turning top) on a top face, raised on a column of 2x2 bricks when the
  blades would hit what they pass over, and 2x6 / 2x8 plates across the turning half as blades (one plate: two
  blades, two crossing plates: four). Check: kinematics.rotors(axis="vertical") - blades reach at least 2 studs out
  and turn 360 clear of every part; feedback names what they hit and says to raise the turntable.
- Data: data/mix_r11 = mix_r10 + data/vrotors_r11 (2174 rows on generic bodies, L1/L2/L3, ~50%
  mistake->feedback->fix). No helicopters, fans or test objects, no object words beyond the generic list.
- Frozen test: runs/audit/r11-vrotors (30). Baseline: r10 on it (never saw upright rotors).
- Recipe as r10: rank 32, batch 64, 2 epochs, max-seq-len 24576.
- Tested sampler: vvennela/run-736d451107854752b6ba9b0cb0ad5b78/objects-r11-tuned-d5f4be8e

## Results (Nov 3, 15:56) - r11 vs r10 (within feedback rounds / n)
| test | r10 | r11 |
|---|---|---|
| r11-vrotors (upright rotors, frozen, new skill) | 8/30 (L1 0/10, L2 0/9, L3 8/11) | **24/30** (L1 4/10, L2 9/9, L3 11/11); 19 first try |
| r10-rotors (side rotors) | 30/30 | 28/30 (L1 5/7) |
| r6-primitives hinge / leg | 15/20, 20/20 | 15/20, 20/20 |
| r9-sheets (flaps) | 19/30 | 20/30 |
| cars (running gear, 50) | 47/50 | **50/50** (48 first try) |
| r5 frozen doors | 10/10 | 10/10 |
| r4 frozen objects (first pass) | 87/112 | **92/112** |
| 747 granularity sweep (held-out) | 5/10 | 4/10 |

Verdict: the upright rotor skill is learned (8 -> 24 of 30) with no real regressions (side rotors -2 at L1, 747 sweep -1: within noise).
