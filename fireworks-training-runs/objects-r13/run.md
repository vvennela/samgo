# objects-r13 (mechanisms on large bodies)

Why: run 11's training bodies were small; at 40 studs the next skills to scale are flaps and car doors, with rotors
on much bigger faces. User, Nov 4: "we can do a training run provided we keep everything and save
everything here".

- Data: the same four generic skills (hinged flap `flap_data`, side rotor `rotor_data`, upright rotor
  `vrotor_data`, blade cross `blade_data`) on bodies 1.5-2.5x larger across (heights x sqrt), via
  `SAMGO_BODY_SCALE=1.5,2.5` in `flap_data.sheet_body` (unset = the original bodies and random sequence).
  1,500 seeds per skill, ~50% mistake -> feedback -> fix rows, no object words: `data/{flaps,rotors,vrotors,blades}_large_r13`.
  `runs/r13_data.sh`.
- Not covered: car doors (trained from car fixtures, not these generators) and legs (`primitives.sheet_body`); and
  nothing here teaches more hinges/legs as load grows beyond what the checker already demands of the references.
- Frozen tests (new): `runs/audit/r13-{flaps,rotors,vrotors,blades}-large` (30 each, held-out seeds).
- Training: from the base model on run 12's mix + the large data (22,980 rows), 2 epochs. Recipe otherwise as before: rank 32, batch 64, max-seq-len 24576.
- Kept: all data, logs (`runs/train/objects-r13*.log`, `runs/train/objects-r13/metrics.jsonl`), every checkpoint
  name and promoted model listed here; nothing deleted except the training session after it ends.
- Tests: the four large tests (r11 baseline first), regressions (r6 primitives, r9 sheets, r10 rotors, r11 vrotors,
  r12 blades, doors, objects, cars, 747 sweep), then the 40-stud showcase builds again.

## Log
- Nov 4 00:32: run id objects-r13s on data/mix_r13 (run 12's mix + 5,244 large rows = 22,980 rows, 2 epochs).
- Baseline, run 11 on the large tests (first try / within 3 rounds / n):
  flaps 15/20/30, side rotors 27/27/30, upright rotors 23/25/30, blade crosses 22/23/30 (any rotor counts).
- Training run id: objects-r13s (started Nov 04 00:32)
- 01:04 checkpoint promoted: accounts/vvennela/models/objects-r13s-s4096 (session ts-bfca046feccf4c7b88a8ce03a34af7d1)
- 01:20 checkpoint promoted: accounts/vvennela/models/objects-r13s-s6144
- 01:34 checkpoint promoted: accounts/vvennela/models/objects-r13s-s8192
