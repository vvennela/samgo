# objects-r12: the blade cross (a rotor built as real LEGO)

**Why:** a rotor out of a face used to end in liftarms; the hand-finished models showed a "propeller" whose axle ran
through its blade plates, which no real LEGO can do (plates have no hole). The part is now a blade cross: a round 2x2
plate with an axle hole (4032) as the hub on the axle's END, two crossed 2 x blade plates on its studs, a round 2x2
cap; the checker rejects an axle that runs on through the blades. Glimmer has never seen it.

**Data:** `samgo/blade_data.py` - run 10's generic bodies and faces (any of four), bearing in the face (slot cleared),
3-long axle, bush, the cross on the axle's end facing out, blades 4/6/8 (the largest that turns clear). Levels L1/L2/L3.
Half the rows: a habitual mistake (axle through the blades x2, slot not cleared, cross off the axle's end, bearing
outside the face) -> checker feedback -> fix. No object words (the part is "blade_cross").
Frozen test: `runs/audit/r12-blades` (30 held-out bodies). Mix: run 11's mix + the blade rows.

**Recipe:** rank 32, batch 64, 2 epochs, max-seq-len 24576 (as runs 10-11).

**Also (code, no training):** gear raises the model only as far as the wheels need (CLEAR 13 -> 10) and prefers leg
spots on the low underside, so legs are short.

## Results
