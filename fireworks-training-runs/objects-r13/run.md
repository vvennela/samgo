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
- 01:49 checkpoint promoted: accounts/vvennela/models/objects-r13s-s10240
- 02:05 checkpoint promoted: accounts/vvennela/models/objects-r13s-s12288
- 02:21 checkpoint promoted: accounts/vvennela/models/objects-r13s-s14336
- {"t": "00:49:32", "event": "promoted", "model": "accounts/vvennela/models/objects-r13s-s2048", "checkpoint": "accounts/vvennela/trainingSessions/ts-bfca046feccf4c7b88a8ce03a34af7d1/checkpoints/run-07f58e0991ec42e0968d0ccc9b5dd1f8-objects-r13s-s2048-d30863f8"}
- {"t": "01:04:50", "event": "promoted", "model": "accounts/vvennela/models/objects-r13s-s4096", "checkpoint": "accounts/vvennela/trainingSessions/ts-bfca046feccf4c7b88a8ce03a34af7d1/checkpoints/run-07f58e0991ec42e0968d0ccc9b5dd1f8-objects-r13s-s4096-d30863f8"}
- {"t": "01:20:37", "event": "promoted", "model": "accounts/vvennela/models/objects-r13s-s6144", "checkpoint": "accounts/vvennela/trainingSessions/ts-bfca046feccf4c7b88a8ce03a34af7d1/checkpoints/run-07f58e0991ec42e0968d0ccc9b5dd1f8-objects-r13s-s6144-d30863f8"}
- {"t": "01:34:23", "event": "promoted", "model": "accounts/vvennela/models/objects-r13s-s8192", "checkpoint": "accounts/vvennela/trainingSessions/ts-bfca046feccf4c7b88a8ce03a34af7d1/checkpoints/run-07f58e0991ec42e0968d0ccc9b5dd1f8-objects-r13s-s8192-d30863f8"}
- {"t": "01:49:55", "event": "promoted", "model": "accounts/vvennela/models/objects-r13s-s10240", "checkpoint": "accounts/vvennela/trainingSessions/ts-bfca046feccf4c7b88a8ce03a34af7d1/checkpoints/run-07f58e0991ec42e0968d0ccc9b5dd1f8-objects-r13s-s10240-d30863f8"}
- {"t": "02:05:37", "event": "promoted", "model": "accounts/vvennela/models/objects-r13s-s12288", "checkpoint": "accounts/vvennela/trainingSessions/ts-bfca046feccf4c7b88a8ce03a34af7d1/checkpoints/run-07f58e0991ec42e0968d0ccc9b5dd1f8-objects-r13s-s12288-d30863f8"}
- {"t": "02:21:18", "event": "promoted", "model": "accounts/vvennela/models/objects-r13s-s14336", "checkpoint": "accounts/vvennela/trainingSessions/ts-bfca046feccf4c7b88a8ce03a34af7d1/checkpoints/run-07f58e0991ec42e0968d0ccc9b5dd1f8-objects-r13s-s14336-d30863f8"}
- {"t": "02:36:58", "event": "promoted", "model": "accounts/vvennela/models/objects-r13s-s16384", "checkpoint": "accounts/vvennela/trainingSessions/ts-bfca046feccf4c7b88a8ce03a34af7d1/checkpoints/run-07f58e0991ec42e0968d0ccc9b5dd1f8-objects-r13s-s16384-d30863f8"}
- {"t": "02:52:40", "event": "promoted", "model": "accounts/vvennela/models/objects-r13s-s18432", "checkpoint": "accounts/vvennela/trainingSessions/ts-bfca046feccf4c7b88a8ce03a34af7d1/checkpoints/run-07f58e0991ec42e0968d0ccc9b5dd1f8-objects-r13s-s18432-d30863f8"}
- {"t": "03:08:31", "event": "promoted", "model": "accounts/vvennela/models/objects-r13s-s20480", "checkpoint": "accounts/vvennela/trainingSessions/ts-bfca046feccf4c7b88a8ce03a34af7d1/checkpoints/run-07f58e0991ec42e0968d0ccc9b5dd1f8-objects-r13s-s20480-d30863f8"}
- {"t": "03:24:35", "event": "promoted", "model": "accounts/vvennela/models/objects-r13s-s22528", "checkpoint": "accounts/vvennela/trainingSessions/ts-bfca046feccf4c7b88a8ce03a34af7d1/checkpoints/run-07f58e0991ec42e0968d0ccc9b5dd1f8-objects-r13s-s22528-d30863f8"}
- {"t": "03:39:51", "event": "promoted", "model": "accounts/vvennela/models/objects-r13s-s24576", "checkpoint": "accounts/vvennela/trainingSessions/ts-bfca046feccf4c7b88a8ce03a34af7d1/checkpoints/run-07f58e0991ec42e0968d0ccc9b5dd1f8-objects-r13s-s24576-d30863f8"}
- {"t": "03:54:19", "event": "promoted", "model": "accounts/vvennela/models/objects-r13s-s26624", "checkpoint": "accounts/vvennela/trainingSessions/ts-bfca046feccf4c7b88a8ce03a34af7d1/checkpoints/run-07f58e0991ec42e0968d0ccc9b5dd1f8-objects-r13s-s26624-d30863f8"}
- {"t": "04:09:22", "event": "promoted", "model": "accounts/vvennela/models/objects-r13s-s28672", "checkpoint": "accounts/vvennela/trainingSessions/ts-bfca046feccf4c7b88a8ce03a34af7d1/checkpoints/run-07f58e0991ec42e0968d0ccc9b5dd1f8-objects-r13s-s28672-d30863f8"}
- {"t": "04:24:45", "event": "promoted", "model": "accounts/vvennela/models/objects-r13s-s30720", "checkpoint": "accounts/vvennela/trainingSessions/ts-bfca046feccf4c7b88a8ce03a34af7d1/checkpoints/run-07f58e0991ec42e0968d0ccc9b5dd1f8-objects-r13s-s30720-d30863f8"}
- {"t": "04:40:22", "event": "promoted", "model": "accounts/vvennela/models/objects-r13s-s32768", "checkpoint": "accounts/vvennela/trainingSessions/ts-bfca046feccf4c7b88a8ce03a34af7d1/checkpoints/run-07f58e0991ec42e0968d0ccc9b5dd1f8-objects-r13s-s32768-d30863f8"}
- {"t": "04:56:06", "event": "promoted", "model": "accounts/vvennela/models/objects-r13s-s34816", "checkpoint": "accounts/vvennela/trainingSessions/ts-bfca046feccf4c7b88a8ce03a34af7d1/checkpoints/run-07f58e0991ec42e0968d0ccc9b5dd1f8-objects-r13s-s34816-d30863f8"}
- {"t": "05:11:51", "event": "promoted", "model": "accounts/vvennela/models/objects-r13s-s36864", "checkpoint": "accounts/vvennela/trainingSessions/ts-bfca046feccf4c7b88a8ce03a34af7d1/checkpoints/run-07f58e0991ec42e0968d0ccc9b5dd1f8-objects-r13s-s36864-d30863f8"}
- {"t": "05:27:17", "event": "promoted", "model": "accounts/vvennela/models/objects-r13s-s38912", "checkpoint": "accounts/vvennela/trainingSessions/ts-bfca046feccf4c7b88a8ce03a34af7d1/checkpoints/run-07f58e0991ec42e0968d0ccc9b5dd1f8-objects-r13s-s38912-d30863f8"}
- {"t": "05:43:17", "event": "promoted", "model": "accounts/vvennela/models/objects-r13s-s40960", "checkpoint": "accounts/vvennela/trainingSessions/ts-bfca046feccf4c7b88a8ce03a34af7d1/checkpoints/run-07f58e0991ec42e0968d0ccc9b5dd1f8-objects-r13s-s40960-d30863f8"}
- {"t": "05:59:19", "event": "promoted", "model": "accounts/vvennela/models/objects-r13s-s43008", "checkpoint": "accounts/vvennela/trainingSessions/ts-bfca046feccf4c7b88a8ce03a34af7d1/checkpoints/run-07f58e0991ec42e0968d0ccc9b5dd1f8-objects-r13s-s43008-d30863f8"}
- {"t": "06:15:14", "event": "promoted", "model": "accounts/vvennela/models/objects-r13s-s45056", "checkpoint": "accounts/vvennela/trainingSessions/ts-bfca046feccf4c7b88a8ce03a34af7d1/checkpoints/run-07f58e0991ec42e0968d0ccc9b5dd1f8-objects-r13s-s45056-d30863f8"}
- {"t": "06:15:22", "event": "sampler_saved", "label": "tuned", "path": "vvennela/run-07f58e0991ec42e0968d0ccc9b5dd1f8/objects-r13s-tuned-d30863f8"}
- {"t": "06:15:55", "event": "promoted", "model": "accounts/vvennela/models/objects-r13s-tuned", "checkpoint": "accounts/vvennela/trainingSessions/ts-bfca046feccf4c7b88a8ce03a34af7d1/checkpoints/run-07f58e0991ec42e0968d0ccc9b5dd1f8-objects-r13s-tuned-d30863f8"}
- {"t": "06:15:55", "event": "done", "promoted_model": "accounts/vvennela/models/objects-r13s-tuned", "base": {"label": "base", "n": 0, "skipped": true}, "tuned": {"label": "tuned", "n": 0, "skipped": true}}
- Tested sampler: vvennela/run-07f58e0991ec42e0968d0ccc9b5dd1f8/objects-r13s-tuned-d30863f8

## Results (Nov 4 06:24; first try / within 3 rounds / n)

Large tests (bodies 1.5-2.5x larger), run 11 -> run 13:
- flaps 15/20/30 -> 16/19/30 (no real change)
- side rotors 27/27/30 -> 26/27/30 (same)
- upright rotors 23/25/30 -> 25/28/30
- blade crosses 22/23/30 -> 29/30/30

Regressions, run 11 (or the last run tested) -> run 13:
- r10 rotors 28/28 -> 30/30/30; r11 upright rotors 19/24 -> 22/28/30; r12 blades 21/23 -> 29/29/30
- r9 sheets (flaps, normal size) r10 15/19 -> 12/20/30: first try down 3, within rounds up 1
- r6 primitives: hinges 15/17/20, legs 20/20/20
- doors 10/10 first try (same); cars 48 first try, 50/50 within rounds (same)
- objects (frozen, 112): 92 -> 88 first try. All four losses are car_1000{01,02,04,05}, "floating" (two also
  off_target); none gained
- 747 sweep: flaps pass at L3/S3 as before; gear fails at every level (run 11 passed L1 and L3)

Verdict: the large-body data fixed rotors and blades at larger sizes and did not fix flaps. It cost 4 frozen car-body
cases and the 747 gear at the levels where it passed. Run 11 stays the demo model; run 13 is kept (all checkpoints
above) for rotors and blades.
