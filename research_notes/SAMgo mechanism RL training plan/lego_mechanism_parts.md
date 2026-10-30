# LEGO mechanism parts: hinges, ball joints, Technic running gear (geometry for a verifier)

**Method and primary source.** All geometry below was taken from the official LDraw library. I downloaded the full `complete.zip` from [LDraw library](https://library.ldraw.org/library/updates/complete.zip) (24,985 part files, fetched 2026-10-29). I then flattened each part's subfile tree with a small script and measured the primitive placements and bounding boxes. Individual part files can be checked at `https://library.ldraw.org/library/official/parts/<id>.dat`, for example [3701.dat](https://library.ldraw.org/library/official/parts/3701.dat). The LDraw files do **not** encode rotation limits or click counts, apart from what a part's title says. Those come from the web, and most of them are gaps.

**LDraw frame conventions (used everywhere below).** −Y is up. A part's origin is the centre of its top face, without studs. Studs stick up to y = −4. A plate's bottom is at y = +8 and a brick's bottom is at y = +24. 1 stud = 20 LDU, 1 plate = 8 LDU, 1 brick = 24 LDU, and 1 LDU = 0.4 mm. Unless stated otherwise, "x" is the long axis of a 1×N part.

## Q1. Hinges for car doors: parts, pivot location, axis, range

### Takeaway
A verifier needs three hinge families, each with a pivot fixed in the part frame:
- **Vertical-axis swivels:** the 1×4 hinge plate 2429/2430 and the 1×4 hinge brick 3831/3830. Their pivot is on a long edge, at the joint between the two 1×2 halves.
- **Horizontal-axis click hinges:** 44301/44302, 60471, and the 30364/30365/30386 bricks. Their pivot sits 10 LDU beyond the end or side of the part, at y = 2 on plates and y = 10 on bricks.
- **Classic car doors:** 3821/3822 (1×3×1) and 3188/3189 (1×3×2). These rotate about a vertical axis through the stud at their hinge end.

The swivel hinges and the classic doors are the parts that make a car door open about a vertical axis.

### Cited Findings
Every coordinate below is measured from the LDraw .dat files in [complete.zip](https://library.ldraw.org/library/updates/complete.zip), unless another source is cited.

**Hinge Plate 1×4 swivel: 2429 (Base) and 2430 (Top)** ([2429.dat](https://library.ldraw.org/library/official/parts/2429.dat), [2430.dat](https://library.ldraw.org/library/official/parts/2430.dat))
- **2429 base:**
  - Studs sit at (x, z) = (−30, 10) and (−10, 10).
  - The bounding box is x −40…8, y −4…8 (plate height), z −8…20.
- **2430 top:**
  - Studs sit at (10, 10) and (30, 10).
  - The bounding box is x −8…40, y −4…8, z −8…20.
- **Pivot:** a vertical (Y) axis through (x, z) = (0, 0), common to both halves.
  - That point is on the long edge (z = 0) of a 1×4 footprint that spans z 0…20, at the point where the two 1×2 halves meet.
  - The 8-LDU-radius knuckle around the pivot explains z down to −8.
- **Straight position:** the parts form a plate-height 1×4 with every stud on the 20-LDU grid.
- **Rotation range: not in LDraw (gap).**

**Hinge Brick 1×4 swivel: 3831 (Base) and 3830 (Top)** ([3830.dat](https://library.ldraw.org/library/official/parts/3830.dat), [3831.dat](https://library.ldraw.org/library/official/parts/3831.dat))
- The geometry is the same as 2429/2430, at brick height (y −4…24).
  - 3831 has studs at x = −30 and −10 and spans x −40…8.
  - 3830 has studs at x = 10 and 30 and spans x −8…40.
  - The studs of both are at z = 10.
- **Pivot:** a vertical axis at (x, z) = (0, 0).
- Brickset recommends 1×4 hinge bricks or plates for wall sections that swing open. It notes that "the axis of the hinge has to be outside" the body, or the part blocks itself from opening — [Brickset](https://brickset.com/article/100227/how-to-really-get-into-lego-buildings).

**Hinge 1×2 tilt: 3937 Base + 3938 Top (assembly 3937c01)** ([3937.dat](https://library.ldraw.org/library/official/parts/3937.dat), [3938.dat](https://library.ldraw.org/library/official/parts/3938.dat))
- **3937 base:** occupies y 2…24, x ±20, z ±10. Its bottom is a 1×2 that fits on studs.
- **3938 top:** a 1×2 plate with studs at x = ±10. It spans y −4…16, and its side lugs reach down to y ≈ 15.5.
- **Pivot:** the cylinders in both parts are centred at y = 10, z = 0, with their axis along **X** (the long axis).
  - That is 14 LDU above the bottom of the brick.
  - The assembled hinge is 1 brick tall (24 LDU).
- **Use:** the top tilts sideways about the long axis. It is used for sloped panels and windscreens rather than doors.
- **Range: gap.**

**Click (locking) hinges** ([44301a.dat](https://library.ldraw.org/library/official/parts/44301a.dat), [44302a.dat](https://library.ldraw.org/library/official/parts/44302a.dat), [60471.dat](https://library.ldraw.org/library/official/parts/60471.dat), [30386.dat](https://library.ldraw.org/library/official/parts/30386.dat))

44302a — "Hinge Plate 1 x 2 Locking with Dual Finger on End Vertical"
- Studs at x = ±10, with the plate spanning x ±20.
- The `clh4` click-hinge primitive is placed at (26, 2, 0).
- The finger discs measure x 24…36, y −4…8, so they are 12 LDU in diameter and centred at **(30, 2, 0)**.
- **Pivot axis: Z** (across the 1-wide plate), 10 LDU beyond the end of the plate and 2 LDU below the top face (6 LDU above the bottom).

44301a/44301b — "Locking with Single Finger on End Vertical"
- The finger region is x 21.1…35.35, y −5.38…8 (the teeth add the extra).
- It shares the same pivot at **(30, 2, 0)**.
- **Mating:** a 44301 and a 44302, each with its pivot 10 LDU past its end, join at a common pivot.
  - In the straight position the second plate's origin is at x = 60, rotated 180° about Y. Its studs are then at x = 50 and 70.
  - The pair is 5 studs long and stays on the grid.
- 44301a has a bottom groove and 44301b does not. Both are sold as LEGO element 44301 ("PLATE 1X2 W/STUB VERTICAL/END") — [Brick Architect](https://brickarchitect.com/parts/44301).

60471 — "Hinge Plate 1 x 2 Locking 9-Position with Dual Finger on Side"
- The `clh4` primitive is at (0, 2, −16).
- The fingers measure x ±8.5, y −4…8, z −26…−14.
- **Pivot:** **(z = −20, y = 2)**, with its axis along **X**. That is 10 LDU beyond the long side.
- The 7-position version is **50340**.
- The single-finger side partner is **44567a/b**.

30383 — "Hinge Plate 1 x 2 Locking with Single Finger On Top"
- The finger sits above the stud at x = 10. It spans x 4…16 and z ±4, and reaches up to y = −11.35.
- The pivot axis is along Z at x = 10, at y ≈ −6. **The height is my estimate from the bounding box; I did not check it against the primitive.**

30386 — "Hinge Brick 1 x 2 Locking Double, Two Finger End with 9 Teeth"
- The fingers span x −36…−26 and x 22.9…35.35, with y 4…16 on both ends.
- **Pivots:** (±30, 10, 0), axis along Z. That puts them at the Technic-hole height of a brick, 14 LDU above the bottom.
- Related bricks:
  - **30364** — single finger on end.
  - **30365** — dual finger on end.
  - **30541 / 30540** — horizontal-finger versions, whose axis is vertical.
  - **76385** — single finger on top.
  - **39893** — 7-tooth version.

**Click positions.** LDraw titles separate 7-tooth and 9-tooth versions: 60471 is "9-Position", 50340 is "7-Position", and 54657/54671/54672 are "7 Teeth".
- One hobbyist forum post gives the standard click-hinge step as **22.5°** — [Eurobricks](https://www.eurobricks.com/forum/forums/topic/41521-ldd-design-challenge-2).
- A web search turned up no official angle specification.

**Older friction-hinge plates:** 4275 (1×2, 3 fingers) and 4276 (1×2, 2 fingers), each in a/b stud variants. 2452 is the 1×2 with 3 fingers on the side. All are in the LDraw titles list in [complete.zip](https://library.ldraw.org/library/updates/complete.zip).

**Classic car doors: 3821 (Right) / 3822 (Left), "Door 1 x 3 x 1"** ([3821.dat](https://library.ldraw.org/library/official/parts/3821.dat))
- **Size:** 1 wide (x ±10), 1 brick tall (y −4…24), 3 studs long (z −10…50).
- The **only stud is at the origin**, at the hinge end, and the hinge barrel sits at z ≈ −10…−2.
- **Pivot:** a vertical axis through (x, z) = (0, 0), the centre of the hinge-end stud. The door rotates about a stud of the body and is held under a plate or anti-stud above.
- **Taller version:** 1×3×2 doors **3188 (R) / 3189 (L)**.
- **Newer version:** **92262 (L) / 92263 (R)** "with Hollow Hinge".

**32028 "Plate 1 x 2 with Door Rail"** has a rail block at z ≈ −16, along the side ([32028.dat](https://library.ldraw.org/library/official/parts/32028.dat)). It is used for sliding or framed doors.

**Real-model practice**
- LEGO holds a patent titled "Hinge mechanism for a door of a model car" — [Google Patents US10603598B2](https://patents.google.com/patent/US10603598). I did not read it in detail.
- Large Technic cars open their doors with gear racks and bevel gears; 42173 is an example — [Brothers Brick](https://www.brothers-brick.com/2024/07/20/lego-technic-42173-koenigsegg-jesko-absolut-grey-hypercar-review/).
- Scissor doors are made longer, with the hinge further forward, so the door clears the occupant — [Eurobricks thread](https://www.eurobricks.com/forum/forums/topic/30111-car-door-help/).

### Inferences
- **Swivel hinges as doors.** For a "door swings about a vertical axis" check on a stud-grid body, 2429/2430 (plate) and 3831/3830 (brick) are the most implementable choices. The pivot is a fixed corner point. The door half is a 1×2 that rotates about Y through that corner, so the verifier only has to sweep the door's footprint about that corner and test for collisions.
- **Classic doors.** 3821/3822 rotate about a vertical axis through their own stud centre, which is a grid point. That makes them the simplest door to model.
- **Click hinges.** Every click hinge has its pivot exactly 10 LDU past the part edge, and that holds for both plates and bricks. The pivot height is 2 LDU below the top face on plates and 10 LDU below the top face (the Technic-hole height) on bricks. A verifier can represent each click hinge as an "end connector" with one shared pivot, quantised to discrete angles. The step is about 22.5° (unconfirmed), with 7- and 9-tooth variants.
- **Snapping to the grid.** Grid alignment of 44301+44302 holds only at 0° and 180°. At 90° the second part's studs land 10 LDU off-grid relative to the first part's frame (pivot at x = 30). A verifier must allow off-grid placement downstream of a hinge, or snap only at 0°, 90° and 180°.

### Gaps
- None of these parts has a rotation range or physical end stop in LDraw. Two figures need checking against real parts or BrickLink photos:
  - My estimate for the 2429 and 3830 swivels: about 0–180°, folding to one side only.
  - The 3937/3938 tilt range.
- I found no official click-hinge angle step; 22.5° is from a forum post. The number of positions per tooth count is also unconfirmed.
- I found no source listing which hinge parts Speed Champions or City cars use for opening doors.
- I estimated the 30383 pivot height (≈ −6) and did not check it against the primitive.

## Q2. Ball-and-socket joints

### Takeaway
The 1×2 "Joint-8" ball and socket plates (14417, 22890, 3170, 14418, 14704, 14419) and the 6628 Technic towball all use a ball of **radius 8 LDU (Ø 16 LDU = 6.4 mm in LDraw)**. On the plates the ball or socket centre is at mid-plate height (y = 4), 10 LDU beyond the plate edge. The large Technic ball joint 32474 has a ball of **radius 12.81 LDU (Ø ≈ 10.25 mm)**.

### Cited Findings
Measurements are from the LDraw files ([14417.dat](https://library.ldraw.org/library/official/parts/14417.dat), [14418.dat](https://library.ldraw.org/library/official/parts/14418.dat), [14419.dat](https://library.ldraw.org/library/official/parts/14419.dat), [14704.dat](https://library.ldraw.org/library/official/parts/14704.dat), [6628.dat](https://library.ldraw.org/library/official/parts/6628.dat), [32474.dat](https://library.ldraw.org/library/official/parts/32474.dat), and the primitive [joint8ball.dat](https://library.ldraw.org/library/official/p/joint8ball.dat)).

| LDraw ID | Title | Joint centre (x, y, z) | Notes |
|---|---|---|---|
| 14417 | Plate 1×2 with Ball Joint-8 on Side | (0, 4, −20) | `joint8ball` = an 8-LDU-radius sphere (`8-8sphe` scaled by 8) on a short bar. Bounding box z −28…−12. Studs at x = ±10. |
| 22890 | Plate 1×2 with Ball Joint-8 on End | not measured; by analogy (±30, 4, 0) | **unverified** |
| 3170 | Plate 1×2 with Ball Joint-8 on Both Ends | not measured | — |
| 14418 | Plate 1×2 with Socket Joint-8 with Friction | (30, 4, 0) (`joint8socket1`) | Socket bounding box x 20.68…35 |
| 14704 | Plate 1×2 with Socket Joint-8 with Friction Centre | (0, 4, −20) | Socket on the side, centred |
| 14419 | Plate 1×2 with Socket Joint-8 with Friction and Ball Joint-8 | Socket (30, 4, 0); ball (−30, 4, 0), radius 8 | One end of each |
| 6628 | Technic Pin Towball with Friction | Ball centre (−10, 0, 0), radius 8 | A friction pin (`confric4`) runs from x = 0 to 20 and goes into a Technic pin hole. Bounding box x −18…20. |
| 32474 | Technic Ball Joint with Axlehole Blind | Origin; sphere radius 12.81 | Mounts on an axle |
| 53585 | Technic Ball Joint with Axlehole Open | — | Variant of 32474 |
| 92911 / 92910 | Technic Ball Joint 2.25 Diameter Socket / C-Shape Frame | — | For the 18 mm steel ball (92911c01) |
| 51482 / 30396 | Hinge 1×2 Locking with 2 Fingers and Towball Socket, 7 / 9 Teeth | — | Click hinge combined with a towball socket |

- All the `Joint-8` plates place the joint centre at y = 4, which is the plate's mid-height (from the files above).

### Inferences
- **Attachment.**
  - The plates attach by studs and anti-studs as ordinary 1×2 plates (studs at x = ±10).
  - 6628 attaches by pushing its pin into any Technic round hole: the pin axis is collinear with the hole and the ball sits 10 LDU outside the hole face.
  - 32474 attaches by putting its axle hole on an axle.
- **Modelling the joint.** A verifier can treat ball and socket as one rigid point constraint: ball centre = socket centre. Allowed orientations form a cone around the socket axis.
- **Cone angle.** I have no sourced value for the cone half-angle. Choose one (for example ±30–45° off-axis plus free spin) and flag it as a free parameter.

### Gaps
- I found no published opening angle or rotation range for Joint-8 sockets or the 6628 towball socket. Physical measurement or a BrickLink photo is needed.
- The real ball diameter is unconfirmed. LDraw's 16 LDU equals 6.4 mm; I found no LEGO source giving the real towball size, which is often quoted as about 5.9 mm.
- I did not open 22890 or 3170, so their joint coordinates are inferred.
- I did not determine which socket mates with 32474 (likely the Bionicle and Technic ball cups).

## Q3. Technic running gear: wheels, axles, Technic bricks, mounting

### Takeaway
Technic holes in 1×N bricks are centred **10 LDU below the top face** (14 LDU above the bottom). They sit **between the studs**, at a 20 LDU pitch, so a 1×N brick has N−1 holes. An N-L axle is exactly N×20 LDU long. Wheel and tyre diameters measured from the LDraw geometry match the BrickLink nominal sizes to within about 0.5 mm.

### Cited Findings

#### Technic bricks
Measured from the LDraw files ([3700.dat](https://library.ldraw.org/library/official/parts/3700.dat), [3701.dat](https://library.ldraw.org/library/official/parts/3701.dat), [3894.dat](https://library.ldraw.org/library/official/parts/3894.dat)).

**Hole geometry (all 1×N bricks)**
- **Placement:** every `peghole.dat` is placed at y = 10 on both faces, z = ±10, so the hole axis runs along Z, through the 1-wide brick.
- **Shape:**
  - A counterbore of radius 8, 2 LDU deep.
  - A bore of radius 6 (Ø 12 LDU = 4.8 mm), from [peghole.dat](https://library.ldraw.org/library/official/p/peghole.dat).
- **Hole centre heights:** y = 10 from the top face, which is 14 LDU (5.6 mm in LDraw units) above the bottom of the brick.

**Hole x-positions by part**

| Part | Holes at x = |
|---|---|
| 3700 "Technic Brick 1 x 2 with Hole" | 0 (studs at ±10) |
| 3701 1×4 | −20, 0, 20 |
| 3894 1×6 | −40, −20, 0, 20, 40 |
| 3702 1×8, 2730 1×10, 3895 1×12, 32018 1×14, 3703 1×16 | Same pattern: N−1 holes |
| 32000 "1 x 2 with Holes" | 2 holes under the studs |
| 6541 "1 x 1 with Hole" | (0, 10), under the stud |
| 5565 1×3 | — |
| 32064a "1 x 2 with Axle Hole" | Axle hole (`axlehol5`) at x = 0, y = 10 |

#### Axles
- Each axle is centred at the origin along X. The axle ends (`axleend2`) sit at x = ±10·N, so **length = N × 20 LDU** (measured from [3704.dat](https://library.ldraw.org/library/official/parts/3704.dat) through [3708.dat](https://library.ldraw.org/library/official/parts/3708.dat)).
- **IDs by length:**

| Length | ID |
|---|---|
| 2L | 3704 (32062 = 2L notched) |
| 3L | 4519 |
| 4L | 3705 |
| 5L | 32073 |
| 6L | 3706 |
| 7L | 44294 |
| 8L | 3707 |
| 9L | 60485 |
| 10L | 3737 |
| 11L | 23948 |
| 12L | 3708 |
| 16L | 50451 |
| 32L | 50450 |

  — LDraw titles in [complete.zip](https://library.ldraw.org/library/updates/complete.zip).
- **Axles with a stop:**
  - 24316 — 3L.
  - 87083 — 4L. The stop is a disc of radius 8 at x = 38…40, at one end ([87083.dat](https://library.ldraw.org/library/official/parts/87083.dat)).
  - 15462 — 5L.
  - 55013 — 8L.
  - 99008 — 4L with a middle cylindrical stop.
- **Axle with stud:** 6587 is "Axle 3 with Stud", with the stud at the x = +30 end.

#### Pins, bushes and joiners
All from the LDraw titles in [complete.zip](https://library.ldraw.org/library/updates/complete.zip), with measured bounding boxes.

**Pins**
- 3673 — pin, 40 LDU long.
- 2780 — friction pin.
- 6558 — long friction pin, 60 LDU.
- 32054 — long pin with stop bush.
- 4274 — 1/2 pin.
- 32002 — 3/4 pin.

**Axle pins**
- 3749 — axle pin.
- 43093 — axle pin with friction.
- 18651 — long friction axle pin with a 2L axle.
- 11214 — long friction axle pin with a 2L pin.

**Bushes**
- 3713 — bush with two flanges. Bounding box Ø 18 LDU, 20 LDU long.
- 32123a — half bush. Ø 18 LDU, 10 LDU long.

**Axle joiners and connectors**
- 6538a — axle joiner.
- 6538b — offset axle joiner.
- 59443 — inline smooth joiner.
- 26287 — 3L joiner.
- 32039 — connector (axle/bush).
- 6536 — cross block 1×2 (axle/pin).
- 3651 — connector (pin/bush) with 2 studs.

#### System wheel holders (non-Technic)
- **4600 "Plate 2 x 2 with 2 Wheel Pins":** wheel pins (`wpin2`) at x = ±22, y = 5 ([4600.dat](https://library.ldraw.org/library/official/parts/4600.dat)).
- **6157:** the extended version.
- **2926:** "Plate 1 x 4 with 2 Wheel Pins".

#### Wheels and tyres
Outer diameter and width are measured from the LDraw geometry (bounding box, mm). Each BrickLink name is shown only where I verified it.

| LDraw tyre | LDraw title | Measured OD × width (mm) | BrickLink name (verified) | Typical rim (LDraw measured Ø × width, mm) |
|---|---|---|---|---|
| 6578 | Tyre 14/ 36 x 20 VR | 30.4 × 14.0 | "Tire 30.4 x 14 VR" (alt 75777) — [BrickLink](https://www.bricklink.com/v2/catalog/catalogitem.page?P=6578) | 2994 "Wheel 12 x 20 with Technic Axle Hole and 6 Pegholes" (24.0 × 14.0) **[pairing inferred]** |
| 44309 / 44308 | Tyre 22/ 30 x 30 ZR / H | 43.2 × 21.6 / 22.4 | (43.2 × 22 nominal; not fetched) | 56145 "Wheel Rim 20 x 30 with 6 Dual Spokes…" (30.4 × 20.0) |
| 41897 | Tyre 28/ 23 x 43 ZR Street | 56.0 × 28.0 | (not fetched) | 41896 / 56908 "Wheel Rim 26 x 43…" (43.2 × 26.4) |
| 55976 | Tyre 26/ 52 x 30 Off Road | 56.8 × 25.6 | (not fetched) | 30.4 mm rim |
| 32019 | Tyre 20/ 64 x 37 S | 62.6 × 20.0 | "Tire 62.4 x 20 S" (32019 / 75999) — [Brick Owl](https://www.brickowl.com/catalog/403065), [BrickLink assembly 86652c01](https://www.bricklink.com/v2/catalog/catalogitem.page?P=86652c01) | 86652 "Wheel Rim 18 x 37 … Short Axle Bush" (LDraw bounding box 44.0 × 18.0; BrickLink "Wheel 43.2mm D. x 18mm"); 32020 has the long axle bush |
| 61480 | Tyre 34/ 37 x 43 Off Road | 68.5 × 34.4 | "Tire 68.7 x 34 R" — [BrickLink](https://www.bricklink.com/v2/catalog/catalogitem.page?P=61480) | 43.2 mm rim (41896 / 56908) |
| 44771 | Tyre 35/ 46 x 56 ZR | 68.9 × 35.2 | "Tire 68.8 x 36 ZR" — [BrickLink](https://www.bricklink.com/v2/catalog/catalogitem.page?P=44771) | 44772 / 15038 "Wheel Rim 34 x 56…" (56.0 × 33.6) |
| 56907 | Tyre 35/ 42 x 56 Racing | 81.6 × 35.2 | (BrickLink page not found for this ID) | 56 mm rim (44772 / 15038) |
| 2997 | Tyre 34/ 38 x 56 ZR | 81.1 × 33.6 | (not fetched) | 56 mm rim |
| 45982 | Tyre 38/ 50 x 43 Off Road | 81.9 × 38.4 | (probably "81.6 × 38"; **unverified**) | 43.2 mm rim |
| 92912 | Tyre 38/ 76 x 56 | 94.5 × 38.4 | "Tire 94.3 x 38 R" — [BrickLink](https://www.bricklink.com/v2/catalog/catalogitem.page?P=92912) | 56 mm rim |
| 54120 | Tyre 44/ 44 x 56 Off Road | 94.7 × 44.0 | (not fetched) | 56 mm rim |
| 3739 + 3740 | Wheel 24 x 43 Technic + Tyre 24/80 x 43 | 82.1 × 24.0 | (old Technic) | — |

Other sizes that may be useful:
- 3482 rim (17.6 mm) + 3483 tyre: 24.8 × 7.2 mm.
- 15413: 49.4 × 20.2 mm.
- 2857: 48.8 × 20 mm.

**How LDraw names tyres.** LDraw tyre titles follow a "width / aspect × rim diameter" pattern; for example "Tyre 28/ 23 x 43" is a 56 × 28 tyre that fits a 43 mm rim. This is my inference from comparing titles with the measured sizes.

**Axle holes in the rims.** The rims I checked reference axle-hole primitives: 2994, 55982, 41896, 44772, 86652 and 56908. 56145 shows no `axlehol` reference by name; it probably uses a subpart, so this is unverified.

### Inferences
**Mounting a wheel**
- An axle goes through a **round** Technic hole and spins freely in it. The wheel's **axle-shaped** hole locks it to the axle.
- A typical stub axle passes through two Technic holes, either in bricks stacked on the same hole axis or in a 2-brick-wide frame. Bushes (3713, or the 32123 half bush) hold the axle in place. An axle with a stop (87083, 55013) can replace the inner bush.
- Hole pitch is 20 LDU along the brick. Vertically, hole-to-hole spacing equals stacking height: 1 brick + 2 plates = 40 LDU = 2 studs, so a brick stack can line holes up with the horizontal grid.

**Which wheel suits which car length**
- My rule of thumb: real cars have a wheel diameter of about 0.14–0.17 × length. I did not find a source for this ratio.
- Car lengths in stud units: 20 studs = 160 mm, 32 = 256 mm, 40 = 320 mm, 60 = 480 mm.

| Car length | Target wheel Ø | Candidates |
|---|---|---|
| 20 studs | ≈ 22–27 mm | 3482 + 3483 (24.8 mm), or 30.4 × 14 (6578) for a chunky look |
| 32 studs | ≈ 36–43 mm | 43.2 × 22 (44309 on 56145), or 36.9 × 22.4 (55978) |
| 40 studs | ≈ 45–55 mm | 49.5 × 20 (15413), or 56 × 28 (41897 on 41896) |
| 60 studs | ≈ 68–82 mm | 68.8 × 36 (44771 on 44772), or 81.6 × 35–38 (56907 / 45982) |

**Hole positions in LDU, for collision and alignment checks**
- Hole centres in a brick placed with origin (X, Y, Z) on the stud grid are at (X + 20k − 10(N−2), Y + 10, Z).
- The axle axis is perpendicular to the long axis of the brick.

### Gaps
- **Real hole height:** LDraw puts the hole centre 14 LDU (5.6 mm) above the brick bottom. I found no primary LEGO source for the real-world figure, which is often quoted as 5.8 mm.
- **Unverified BrickLink names:** I did not fetch the BrickLink names for 44309, 41897, 55976, 2997, 45982 or 56907. Rebrickable returned HTTP 403 to automated fetches.
- **Rim pairings:** these are partly inferred (6578 on 2994; 61480 and 45982 on 43.2 mm rims). LEGO's official set pairings were not checked.
- **Real-model wheel sizes:** I found no sourced survey of which wheel sizes official sets use for each car length (for example Speed Champions or Technic 1:8 supercars). The size table above is an inference.
