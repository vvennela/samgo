# SAMgo

*LEGO comes from the Danish* leg godt, *"play well." SAMgo comes from* saml godt, *"assemble well," built on Meta's SAM.*

Photo → buildable LEGO-compatible model.

Meta Global AI Developer Hackathon · Indie track · Theme: Push the Frontier
Submission: **Wed Nov 4, 6:00 PM PT**

---

## 1. Problem

Turning a real object into something you can physically build takes a designer: someone who can judge shape, pick parts, and make sure the result holds together. AI models can describe objects and generate pictures and meshes of them, but they are bad at producing **physically valid assemblies**: parts that exist, fit, connect, and stand up. The output looks right and falls apart.

LEGO is a clean testbed for this: a fixed part catalog, exact geometry, and validity that can be checked by a program. If a model can learn to go from a photo to a buildable LEGO design, the same recipe (perception → structured target → generation checked by a verifier) points at real manufacturing problems.

---

## 2. Solution (bird's eye)

1. **See it.** Meta's SAM models turn a photo of an object into a 3D shape.
2. **Define the target.** The shape becomes a LEGO-scale target: a grid of what should be filled, and in which color.
3. **Build it.** A builder proposes a brick-by-brick design for that target.
4. **Check it.** A verifier, written as plain code, checks every design: real parts, no overlaps, everything connected and grounded, close enough to the target. Only designs that pass are shown or kept.
5. **Explain it.** Passing designs become step-by-step instructions and a parts list.
6. **Learn it (research).** An open-weight Meta model (size TBD) is trained to be the builder, using only the verifier as its training signal: generate designs, keep the ones that pass, train on them, repeat.

The product works without training (an algorithmic builder plus a Muse Spark agent). Training is the research claim: *a model can learn to build physically valid structures from a verifier alone.*

---

## 3. How we generate valid LEGO (bird's eye)

```
                         ┌──────────────────────────────────────────┐
  photo ──▶ SAM 3.1 ──▶  │  SAM 3D  ──▶  3D mesh  ──▶  voxelize     │
           (find/mask)   │  (shape)                    (LEGO scale, │
                         │                              palette)    │
                         └───────────────────┬──────────────────────┘
                                             ▼
                                       TARGET (grid)
                                             │
                                             ▼
                         ┌─────────────── BUILDER ───────────────┐
                         │  algorithmic (v0)  │  trained model   │
                         └───────────────────┬───────────────────┘
                                             ▼
                                       candidate build
                                             │
                                             ▼
                                ┌────────── VERIFIER ──────────┐
                                │ real parts? no overlaps?     │
                                │ connected? grounded?         │
                                │ matches target?              │
                                └───────┬──────────────┬───────┘
                                   fail │              │ pass
                                        ▼              ▼
                         repair / retry (agent)   VALID BUILD ──▶ instructions
                                        │                         + parts list
                                        └──────▶ back to BUILDER
```

**Training loop (research):**

```
   targets ──▶ model proposes K builds each ──▶ VERIFIER
      ▲                                            │
      │                                  keep only passing builds
      │                                            │
   new, slightly harder targets  ◀── fine-tune model on them
   (model proposes variants it can barely solve)
```

---

## 4. User flow (bird's eye)

```
  ┌────────────┐   ┌─────────────────┐   ┌──────────────┐   ┌──────────────┐
  │ Take/upload│──▶│ Confirm object  │──▶│ Choose size  │──▶│ 3D preview   │
  │ a photo    │   │ ("wooden chair" │   │ (small / med │   │ of the LEGO  │
  │            │   │  + mask shown)  │   │  / large)    │   │ model        │
  └────────────┘   └─────────────────┘   └──────────────┘   └──────┬───────┘
                                                                   │
                                                                   ▼
  ┌────────────┐   ┌─────────────────┐   ┌──────────────────────────────────┐
  │ Build it,  │◀──│ Download parts  │◀──│ Step-by-step instructions        │
  │ share it   │   │ list + LDraw    │   │ (one layer at a time, new bricks │
  │            │   │ file            │   │  highlighted)                    │
  └────────────┘   └─────────────────┘   └──────────────────────────────────┘
```

---

## 5. Schedule

| Day | Focus | Stages |
|---|---|---|
| **Wed 10/28** | Photo → target, and the verifier | S1, S2, S3 |
| **Thu 10/29** | Algorithmic builder, instructions, end-to-end app | S4, S5, S6 |
| **Fri 10/30** | Training harness + baseline numbers | S7 |
| **Sat 10/31** | First training run → **go/no-go** | S8 |
| **Sun 11/1** | More rounds; trained model as alternate builder | S8 → S9 |
| **Mon 11/2** | Ablation, final evals, physical build | S9, S10 |
| **Tue 11/3** | Demo video, write-up, repo cleanup | — |
| **Wed 11/4** | Buffer · submit by 6 PM PT · post on X | — |

If S8 fails, the product (S1–S6) still ships and training is reported as a partial result.

---

## 6. Stages

Each stage proves exactly one thing. Each one depends only on stages before it.

| # | Proves | Done when |
|---|---|---|
| **S1** | A photo can become a usable 3D shape. | 5 of 5 test photos produce a mesh you'd recognize as the object. |
| **S2** | A 3D shape can become a LEGO-scale target that still looks like the object. | Voxel renders of the S1 meshes are recognizable at the chosen size. |
| **S3** | We can reliably tell a valid build from an invalid one. | Verifier agrees with every hand-made good/bad test build (overlap, floating, disconnected, off-target). |
| **S4** | An algorithm can turn a target into a valid build. | ≥ 80% of targets produce a build that passes the verifier. |
| **S5** | A valid build can become instructions a person can follow. | Someone builds a small model from the generated steps without help. |
| **S6** | A user can go from photo to instructions with no manual steps. | A new photo goes through the app end to end, agent included. |
| **S7** | An open model can attempt builds, and we can measure how well it does. | Baseline pass rate recorded on a fixed held-out set of targets. |
| **S8** | Training against the verifier makes the model better at building. | **Trained model ≥ +5 points absolute pass rate over baseline.** |
| **S9** | Self-generated harder targets beat plain training. | Self-play run beats plain expert iteration at equal compute. *(stretch)* |
| **S10** | It's real. | A physical LEGO build from the instructions, next to the original object. |

S1–S6 = the product. S7–S9 = the research. S10 = the demo.

---

## 7. Dependencies

### API keys & accounts

| Service | Env var | Used for | Needed by | Status |
|---|---|---|---|---|
| Meta Model API | `MODEL_API_KEY` | Muse Spark (agent), SAM 3.1 (masks). $150 credits; card must be on file; claim by Nov 4 | S1 | ☐ |
| fal.ai | `FAL_KEY` | SAM 3D Objects (photo → mesh), ~$0.02/call | S1 | ☐ |
| Hugging Face | `HF_TOKEN` | Download open-weight model (e.g. Muse Glimmer 30B) | S7 | ☐ |
| Baseten | `BASETEN_API_KEY` | Muse Spark Open baseline (`meta-models/muse-spark-open-1026`) | S7 | ☐ |
Keep keys in `.env` (gitignored). The repo is public.

### Software

| Area | Packages / tools |
|---|---|
| Core | Python 3.11+, numpy, Pillow |
| Mesh & voxels | trimesh (GLB loading, voxelization) |
| Model APIs | openai SDK (Meta Model API is OpenAI-compatible), fal-client, anthropic |
| Training | vLLM (sampling), transformers, peft (LoRA/QLoRA), bitsandbytes, accelerate |
| Rendering | LDraw parts library (check its license before bundling), renderer (LDView or a custom voxel renderer) |
| App | Gradio |

### Data

| Item | Source | Needed by |
|---|---|---|
| Test photos | Own photos of chunky, matte objects | S1 |
| Synthetic targets | Procedural shapes + licensed meshes only (hackathon bans assets without rights) | S4 / S7 |
| Held-out eval set | Generated once, frozen, never trained on | S7 |

### Physical

| Item | Needed by |
|---|---|
| LEGO bricks for the demo build (order early or use what's on hand) | S10 |
