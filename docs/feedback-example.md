# Feedback in action (for the video)

A real case from the frozen flap test `runs/audit/r9-sheets` (case `flap8_val_100006`, run 10, unseen body).

**Glimmer is told** (generic, no object named):
> These are the parts along one edge of a flat structure (the edge toward -x, rows y = 29..32). Turn the outer part of
> it into a flap: free it and hinge it about a horizontal axis along the edge.

**Round 1.** Glimmer clears a channel at x = 4 and puts the hinge there. But the cut is too close to the edge: the
clips land past the end of the structure, over empty space.

**The checker's feedback** (code, word for word):
> connected: these hinge parts hold nothing - their studs grip no other part, so they only hang on their partner:
> your added (clip_plate 2x1 at (x=2, y=29, z=11)); your added (clip_plate 2x1 at (x=2, y=32, z=11)) - put each on
> (or under) a part of the side it carries
>
> horizontal_surfaces: the hinge at (x=4.5, y=29.0) studs turns only parts you added; the surface must be parts of
> the input model, freed from it and hung on the hinge

**Round 2.** Glimmer moves the whole cut two studs inward (x = 6): the clips now sit on the freed panel, the bars on
the structure, and a real part of the model swings on the hinge. **Passes** every check: valid parts, every part
held by studs, the flap swings clear.

Why it matters: the clip over empty space is exactly the bug a physical check must catch (it used to pass, and data
made under the old check was regenerated). Glimmer learned to read this sentence and fix it from training rows built
the same way: its own mistakes, the checker's feedback, the right answer.
