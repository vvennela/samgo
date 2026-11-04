# objects-r13 (mechanisms on large bodies)

Why: at 40 studs run 11 lost the mechanisms (helicopter main + tail rotor, wind-turbine rotor, Cessna flap; Camry
doors) - its training bodies were small. User, Nov 4: "we can do a training run provided we keep everything and save
everything here".

- Data: the same four generic skills (hinged flap `flap_data`, side rotor `rotor_data`, upright rotor
  `vrotor_data`, blade cross `blade_data`) on bodies 1.5-2.5x larger across (heights x sqrt), via
  `SAMGO_BODY_SCALE=1.5,2.5` in `flap_data.sheet_body` (unset = the original bodies and random sequence).
  1,500 seeds per skill, ~50% mistake -> feedback -> fix rows, no object words: `data/{flaps,rotors,vrotors,blades}_large_r13`.
  `runs/r13_data.sh`.
- Not covered: car doors (trained from car fixtures, not these generators) and legs (`primitives.sheet_body`); and
  nothing here teaches more hinges/legs as load grows beyond what the checker already demands of the references.
- Frozen tests (new): `runs/audit/r13-{flaps,rotors,vrotors,blades}-large` (30 each, held-out seeds).
- Training: warm start from run 11's last adapter (`objects-r11-s30720`) on large data + run 12's blade data +
  4,000 rows replayed from run 11's mix, 2 epochs. If the adapter fails to load (it did on runs 10 and 12), from
  scratch on run 12's mix + the large data, 2 epochs. Recipe otherwise as before: rank 32, batch 64, max-seq-len 24576.
- Kept: all data, logs (`runs/train/objects-r13*.log`, `runs/train/objects-r13/metrics.jsonl`), every checkpoint
  name and promoted model listed here; nothing deleted except the training session after it ends.
- Tests: the four large tests (r11 baseline first), regressions (r6 primitives, r9 sheets, r10 rotors, r11 vrotors,
  r12 blades, doors, objects, cars, 747 sweep), then the 40-stud showcase builds again.
