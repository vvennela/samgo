# 3D mesh datasets (vehicles first) and LEGO model collections as base objects for LEGO training data (as of 2026-10-29)

## ShapeNet / ShapeNetCore v2: vehicle counts, access, formats, size

### Takeaway
ShapeNetCore v2 has about 3,514 cars and about 939 buses, but the Hugging Face repo is gated with **manual approval**: you submit your name, your advisor/PI and your affiliation, and access is "considered for approval". Users report waiting, and the main repo and the archive repo are approved separately. You can't count on it inside one day. It is also non-commercial research only.

### Cited Findings
- HF `ShapeNet/ShapeNetCore` is gated. You must log in and accept conditions, giving your full name, advisor/PI name and school/company affiliation. Access is then "considered for approval", so it is not instant. — [HF ShapeNetCore](https://huggingface.co/datasets/ShapeNet/ShapeNetCore)
- License "other": non-commercial research and educational use only. You may share it only with colleagues who accept the same terms, Princeton/Stanford can terminate access, and a for-profit employer is bound too. — [HF ShapeNetCore](https://huggingface.co/datasets/ShapeNet/ShapeNetCore)
- The HF repo is 24 GB, covering 55 categories and about 51,300 models. Formats are described in the repo's DATA.md, not on the card. — [HF ShapeNetCore](https://huggingface.co/datasets/ShapeNet/ShapeNetCore)
- Users report pending access requests, and one was approved for ShapeNetCore but not for ShapeNetCore-archive (approval is per repo). Requests can be denied if the form is not filled out as the README asks. — [HF discussion #2](https://huggingface.co/datasets/ShapeNet/ShapeNetCore/discussions/2), [archive discussion #1](https://huggingface.co/datasets/ShapeNet/ShapeNetCore-archive/discussions/1)
- One aggregator says the v2 zip is 54.8 GB. Another source says about 25 GB without GLTF/GLB. — [hyper.ai](https://hyper.ai/datasets/16769)
- Car = 3,514 models in ShapeNet v2 (split 2465/359/690). "Nearly half of the CAD models in the car category were removed in ShapeNet v2." — [SDF-SRN, arXiv 2010.10505](https://arxiv.org/pdf/2010.10505)
- Bus = 939: this is my own sum of a survey's re-split of ShapeNetCore (bus 657/188/94, car 2460/703/351, which also sums to 3,514). — [Mirbauer et al. survey supplementary](https://cgg.mff.cuni.cz/~martinm/papers/2021-survey-eval/data/supplementary.pdf)
- StableText2Brick, the BrickGPT training set, was built from ShapeNetCore objects voxelized at 20x20x20. — [BrickGPT paper, arXiv 2505.05469](https://arxiv.org/pdf/2505.05469)

### Inferences
- In ShapeNetCore, trucks are not a separate top-level class. They seem to sit inside the car (02958343) synset and its sub-synsets, so "truck" counts need the sub-synset taxonomy (taxonomy.json).
- For a one-day turnaround, treat ShapeNet as "if approval arrives". Request access today anyway.
- Cars are the best-represented vehicle class anywhere at this quality: about 3.5k curated CAD cars with consistent orientation.

### Gaps
- No official per-category table for v2 was found. The 939 bus figure is derived, and I found no truck count.
- I found no typical approval time. Anecdotes only say "pending".
- The exact file formats in the HF repo (OBJ+MTL, plus GLB in some versions) were not confirmed from DATA.md.
- From memory and not verified here: ShapeNetCore also has airplane, motorbike, train, vessel and rocket synsets.

## Objaverse 1.0 and Objaverse-XL: vehicle categories, access, formats, licensing

### Takeaway
Objaverse is the fastest route. It is not gated, `pip install objaverse` downloads GLBs one at a time in parallel, and most objects are CC-BY. The LVIS subset has **1,436 road-vehicle objects** (exact counts below). Many more vehicles exist in the full 800K set, but you have to find them through name and tag metadata, and quality varies a lot.

### Cited Findings
- API: `objaverse.load_uids()` returns 798,759 UIDs. `load_annotations(uids)` returns name, license, tags and so on. `load_lvis_annotations()` returns a category-to-UIDs dict. `load_objects(uids, download_processes=N)` returns local .glb paths cached under `~/.objaverse/hf-objaverse-v1/glbs/`. The docs mention no login, token or approval. Objects are .glb (example about 1.5 MB). — [Objaverse 1.0 docs](https://objaverse.allenai.org/docs/objaverse-1.0)
- HF `allenai/objaverse`: no gating notice. The collection is ODC-By v1.0, with per-object licenses: CC-BY 4.0 721K, CC-BY-NC 4.0 25K, CC-BY-NC-SA 4.0 52K, CC-BY-SA 4.0 16K, CC0 3.5K. Total 8.9 TB, all GLB. — [HF allenai/objaverse](https://huggingface.co/datasets/allenai/objaverse)
- I downloaded `lvis-annotations.json.gz` (0.9 MB, ungated) myself and counted it: 1,156 categories, 46,207 objects. Road-vehicle categories: car_(automobile) 102, police_cruiser 113, race_car 122, pickup_truck 109, motor_vehicle 106, jeep 80, tractor_(farm_equipment) 75, bus_(vehicle) 71, truck 61, school_bus 60, trailer_truck 53, convertible_(automobile) 50, forklift 50, bulldozer 45, camper_(vehicle) 45, army_tank 90, minivan 39, cab_(taxi) 38, ambulance 37, garbage_truck 33, fire_engine 32, golfcart 32, tow_truck 29, limousine 9. The 23 categories without camper add up to **1,436 unique UIDs**. — [allenai/objaverse lvis-annotations.json.gz](https://huggingface.co/datasets/allenai/objaverse/resolve/main/lvis-annotations.json.gz)
- Other LVIS vehicles from the same file: motorcycle 81, motor_scooter 71, bicycle 122, tricycle 28, airplane 112, fighter_jet 126, helicopter 124, jet_plane 47, space_shuttle 58, train_(railroad_vehicle) 45, bullet_train 54, freight_car 52, passenger_car_(part_of_a_train) 45, boat 63, yacht 26, cargo_ship 27, cruise_ship 19, wagon 37, cart 27. — same file
- Furniture in LVIS: chair 453, table 101, sofa 81, bench 91, cabinet 60, dining_table 70, bed 53, bunk_bed 57, coffee_table 51, sofa_bed 49. — same file
- Objaverse-XL on HF (`allenai/objaverse-xl`): no gating notice except for the Polycam subset ("available ... to academic researchers for non-commercial use upon request and approval"). It is ODC-By v1.0 with mixed per-object licenses (MIT, GPL-3.0, none...). There are about 10.9M rows, and the file types include gltf, fbx, obj, glb, dae and blend. Downloads go through the Colab tutorial and the `objaverse.xl` API. — [HF objaverse-xl](https://huggingface.co/datasets/allenai/objaverse-xl)
- Objaverse-XL paper: 10M+ objects from GitHub, Thingiverse, Sketchfab, Polycam and Smithsonian. — [Objaverse-XL NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/70364304877b5e767de4e9a2a511be0c-Paper-Datasets_and_Benchmarks.pdf)

### Inferences
- Fastest way to a few thousand vehicles in one day:
  1. Take the 1,436 LVIS road-vehicle UIDs, which are human-verified.
  2. Add more by keyword search over `load_annotations()` names and tags ("car", "truck", "bus", "van", "sedan", "lowpoly car"...), filtered to license in {by, by-sa, cc0}.
  3. Run `load_objects(..., download_processes=16-32)`. At about 1-10 MB per GLB, 3-5k objects is roughly 5-30 GB, which is a few hours or less on a normal link.
- Objaverse GLBs are often not watertight: scenes, multiple loose parts, interior faces, flat ground planes, single-sided surfaces. Use solid voxelization (winding-number or flood-fill from outside) rather than ray-parity, and filter out scenes and bad scales by bounding-box aspect and part count.
- Objaverse-XL GitHub and Thingiverse objects need per-repo downloads and mixed format conversion (fbx/blend/dae), which is slower. Thingiverse STLs are often closed, printable solids that voxelize well, but they seldom carry vehicle labels. For a one-day job, 1.0 is the better choice.

### Gaps
- I did not count vehicles in the full Objaverse 1.0 beyond LVIS, because the full metadata is large.
- I found no per-source counts for XL vehicles.
- Typical GLB size is only one example (about 1.5 MB), not a distribution.

## Other mesh datasets (ModelNet40, 3D-FUTURE, ABO, Toys4K, OmniObject3D, GSO, Thingi10K, DrivAerNet/++, CADillac)

### Takeaway
None of these beats Objaverse plus ShapeNet for vehicles. DrivAerNet/DrivAerNet++ has thousands of parametric car bodies (clean STL), but they are all variations of one DrivAer sedan, CC BY-NC, and very large (Globus). 3D-FUTURE, ABO and GSO are furniture and household items. ModelNet40 has a small car class.

### Cited Findings
- ModelNet40: 12,311 CAD models in 40 classes (9,843 train / 2,468 test), and "car" is one of the classes. — [FusionNet, arXiv 1607.05695](https://arxiv.org/pdf/1607.05695), [gammagl docs](https://gammagl.readthedocs.io/en/stable/generated/gammagl.datasets.ModelNet40.html)
- A ModelNet40-FS split lists car = 297, but that is a few-shot split, not the standard split. — [arXiv 2304.00022](https://arxiv.org/pdf/2304.00022)
- DrivAerNet++: 8,000 car designs with high-fidelity CFD. Geometries are STL, the full dataset is about 39 TB, it is hosted on Harvard Dataverse and needs Globus, and the license is CC BY-NC 4.0. — [DrivAerNet GitHub](https://github.com/Mohamedelrefaie/DrivAerNet), [DrivAerNet++ arXiv 2406.09624](https://arxiv.org/html/2406.09624v2)
- The original DrivAerNet has 4,000 STL geometries totaling 84 GB. — [DrivAerNet arXiv 2403.08055](https://arxiv.org/pdf/2403.08055)
- 3D-FUTURE: 16K objects in 34 categories (other papers say about 10K CAD furniture shapes). You must apply through Alibaba Tianchi/TC Lab and agree to a "3D-FUTURE Data Sets Use License Agreement". The archive is about 19 GB. — [OmniObject3D paper table](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_OmniObject3D_Large-Vocabulary_3D_Object_Dataset_for_Realistic_Perception_Reconstruction_and_CVPR_2023_paper.pdf), [3DCoMPaT++ arXiv 2310.18511](https://arxiv.org/pdf/2310.18511), [Tianchi 3D-FUTURE](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future), [baseline README](https://github.com/3D-FRONT-FUTURE/3D-FUTURE-AI-Challenge-Baseline/blob/master/README.md)
- ABO: about 8K models in 63 categories, mostly furniture and home decor. It is a public S3 bucket (no sign-up) at `amazon-berkeley-objects` (us-east-1), but the license is disputed: the index page says CC BY 4.0, while the AWS registry and the bucket license file say CC BY-NC 4.0. — [TRELLIS arXiv 2412.01506](https://arxiv.org/pdf/2412.01506), [ABO index](https://amazon-berkeley-objects.s3.amazonaws.com/index.html), [AWS open-data registry](https://github.com/awslabs/open-data-registry/blob/main/datasets/amazon-berkeley-objects.yaml), [HF mirror](https://huggingface.co/datasets/suvadityamuk/amazon-berkeley-objects)
- Toys4K: about 4K objects in 105 categories. — [TRELLIS arXiv 2412.01506](https://arxiv.org/pdf/2412.01506)
- OmniObject3D: about 6,000 real-scanned objects in 190 categories, with textured meshes. The authors say they will "regulate use" of the data. — [OmniObject3D arXiv 2301.07525](https://arxiv.org/pdf/2301.07525)
- Google Scanned Objects: 1,030 scanned household items, 13 GB, CC-BY 4.0, hosted on Open Robotics/Gazebo Fuel (OBJ + textures). The mix is 17 action figures, 28 bags, 254 shoes and so on, with no vehicles of note. — [Google Research blog](https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/), [Jeff Dean post](https://x.com/JeffDean/status/1519165047917883392)

### Inferences
- DrivAerNet/++ STLs are CFD-ready, so they are probably closed, watertight surfaces that voxelize cleanly, and the 3D-mesh subset can likely be downloaded without the CFD fields. But they cover only about 3 body types (fastback, notchback, estateback) as parametric variations, which gives little shape diversity for LEGO training.
- GSO scans and OmniObject3D scans are watertight-ish single-surface meshes (scan reconstructions) and voxelize well. They contain few or no vehicles, apart from possible toy cars in OmniObject3D and Toys4K.
- 3D-FUTURE and ABO are good if you want furniture. Of the two, ABO is the faster download (open S3, no sign-up).
- Watertight ranking for 20-60 stud voxelization: DrivAerNet STL ≈ scans (GSO/OmniObject3D) > Thingi10K/Thingiverse STLs > ShapeNet/ModelNet CAD (non-manifold, but solid-fill works with winding numbers or tools like ManifoldPlus) > Objaverse GLB (most heterogeneous).

### Gaps
- Not researched within the call budget: Thingi10K counts and license (from memory, about 10K Thingiverse STLs with per-file licenses, unverified), Toys4K vehicle category counts and access, OmniObject3D access process (form?), and the standard ModelNet40 car count.
- I found no "CADillac" car dataset under that name.

## LEGO model collections (LDraw OMR, Rebrickable, BrickLink Studio, academic datasets)

### Takeaway
The LDraw OMR currently lists **1,470 official sets** as LDraw files, browsable one set at a time. I found no bulk zip. Rebrickable explicitly forbids scraping MOC data and has removed MOCs from its API. Academic options:
- **StableText2Brick**: 47,389 structures, MIT license, ungated on HF, 1x1-height bricks on a 20^3 grid, includes car, bus, train and vessel.
- **BrickNet** (2026): about 320K LDraw samples, with a connector taxonomy covering hinge, axle, pin and Technic ball. It is "available for research purposes", and its terms are unclear.

### Cited Findings
- The LDraw OMR sets list shows "Showing 1 to 25 of 1,470 results". There is no visible bulk download, and you look a set up by number in the search bar. — [library.ldraw.org/omr/sets](https://library.ldraw.org/omr/sets), [library.ldraw.org/omr](https://library.ldraw.org/omr)
- The OMR is a database of LDraw-format files describing models that LEGO released as sets, with rules for submission. — [OMR rules](https://library.ldraw.org/documentation/official-model-repository-omr-standards/rules-and-procedures-for-the-official-model-repository)
- An older unofficial Eurobricks index (2017) listed 2,305 sets made in LDraw, outside the OMR. — [Eurobricks thread](https://www.eurobricks.com/forum/forums/topic/48285-key-topic-official-lego-sets-made-in-ldraw)
- Rebrickable API v3 has "no longer any support for MOCs or MOC inventories" (removed in 2020 because of scraping). Its Terms say downloads include no MOC data, and "under no circumstances are you allowed to automate or manually scrape any MOC related data". — [Rebrickable API docs](https://rebrickable.com/api/v3/docs/), [Rebrickable Terms](https://rebrickable.com/terms/)
- LDraw Parts Library is CC BY 2.0 and downloadable from library.ldraw.org. — [mocforge README](https://github.com/mor64ph/mocforge)
- StableText2Brick: 47,389 rows (42.6k train / 4.79k test) built from over 28,000 unique ShapeNet objects in 21 categories (basket, bed, bench, birdhouse, bookshelf, bottle, bowl, bus, camera, car, chair, guitar, jar, mug, piano, pot, sofa, table, tower, train, vessel). Each brick line is `hxw (x,y,z)` in a 20x20x20 grid. MIT license, 44.3 MB Parquet, no gating. — [HF AvaLovelace/StableText2Brick](https://huggingface.co/datasets/AvaLovelace/StableText2Brick), [BrickGPT paper](https://arxiv.org/pdf/2505.05469)
- The prior dataset, StableLego, has 8,000+ stable structures. StableText2Brick has more than 3x the stable unique objects and more than 5x the stable structures. — [BrickGPT paper](https://arxiv.org/pdf/2505.05469)
- BrickNet (arXiv 2604.22984, Kulits and Schmid): 320,808 samples, 9,743 part variants and 40.5M placed bricks, with an SFT subset of 67,185 samples. The abstract says "over 100,000", which does not match the body. Data is "curated from publicly available online sources" (unnamed). Connector taxonomy: stud variants, hinge, axle (pin/axle/sockets/bar/clip), ball (towball/socket/technic), fixed. "Available for research purposes" via kulits.github.io/BrickNet. — [BrickNet arXiv](https://arxiv.org/html/2604.22984)
- Other LEGO benchmarks: LEGO Co-builder (built from real manuals, where GPT-4o reached about 40% F1 on assembly-state detection), Prompt-to-Parts (LDraw as an intermediate format, cites LEGO-Puzzles), Break and Make (uses LDraw/OMR). — [arXiv 2507.05515](https://arxiv.org/abs/2507.05515), [arXiv 2512.15743](https://arxiv.org/pdf/2512.15743), [arXiv 2207.13738](https://arxiv.org/pdf/2207.13738)

### Inferences
- For mechanism and Technic data, BrickNet is the only large dataset I found with explicit hinge, axle and pin connectors. Check its download page and license before you plan around it.
- OMR Technic sets (part of the 1,470) are a small, clean, legal source of real mechanisms. You would fetch them per set, which is scriptable from the set pages.
- StableText2Brick's car (and bus/train/vessel) subsets give ready-made BrickGPT-format targets at 20^3. To go to 60 studs you need to re-voxelize the source meshes (ShapeNet, gated) or substitute Objaverse meshes.

### Gaps
- I did not determine BrickLink Studio gallery access or bulk download, or its terms.
- I found no OMR bulk-download URL. Per-set file links and the .mpd/.ldr formats were not confirmed on the page.
- I could not verify that names like "LegoACE" or "LEGO-Maker" are real datasets.
- No per-category counts for StableText2Brick (for example, how many car structures).
- The BrickNet license, the hosting location and the share of samples that use Technic parts are unknown.
