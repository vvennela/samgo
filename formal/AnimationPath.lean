/-
  The build animation never draws one part through another (samgo/animation.py), proved for all positions and sizes.
  Check with `lean formal/AnimationPath.lean` (Lean 4 core, omega).

  Boxes are integer (units of 0.2 mm; every LEGO dimension the animation uses is a multiple). Two boxes collide when
  their open interiors meet on all three axes (touching faces do not collide). A part arrives along one axis: it is
  drawn at its final place shifted by s * t along that axis, for every t >= 0 down to t = 0 (s = +1: it comes from the
  + side, e.g. dropped from above along z; s = -1: from the - side).

  1. `slide_clear` (samgo.animation.slide_clear): if, for a placed box, the projections on the two other axes are
     disjoint, or the placed box lies entirely on the far side along the axis, the moving box never collides with it
     at any t >= 0. Proved for z from above and for x and y from either side (the remaining cases are the same proof).
  2. `hole_clear` (samgo.animation.hole_clear / on_axle): an axle of half-width r on a line, slid along that line,
     stays inside the line's channel of half-width r; a Technic brick's plastic (or a hub's) lies outside its hole's
     channel of the same line, so the axle never meets it, at any position along the line.
  The Python checks are exactly these hypotheses, so every move the animation draws is covered.
-/

set_option linter.unusedSectionVars false

structure Box where
  x0 : Int
  x1 : Int
  y0 : Int
  y1 : Int
  z0 : Int
  z1 : Int

/-- Open interiors meet on all three axes. -/
def collide (a b : Box) : Prop :=
  a.x0 < b.x1 ∧ b.x0 < a.x1 ∧ a.y0 < b.y1 ∧ b.y0 < a.y1 ∧ a.z0 < b.z1 ∧ b.z0 < a.z1

def shiftZ (a : Box) (d : Int) : Box := { a with z0 := a.z0 + d, z1 := a.z1 + d }
def shiftX (a : Box) (d : Int) : Box := { a with x0 := a.x0 + d, x1 := a.x1 + d }
def shiftY (a : Box) (d : Int) : Box := { a with y0 := a.y0 + d, y1 := a.y1 + d }

/-- Dropped from above along z: placed boxes are beside it (x or y disjoint) or below it. -/
theorem slide_clear_z_down (m p : Box)
    (h : m.x1 ≤ p.x0 ∨ p.x1 ≤ m.x0 ∨ m.y1 ≤ p.y0 ∨ p.y1 ≤ m.y0 ∨ p.z1 ≤ m.z0) :
    ∀ t : Int, 0 ≤ t → ¬ collide (shiftZ m t) p := by
  intro t ht
  simp only [collide, shiftZ]
  omega

/-- Slid in from the + side of x. -/
theorem slide_clear_x_pos (m p : Box)
    (h : m.y1 ≤ p.y0 ∨ p.y1 ≤ m.y0 ∨ m.z1 ≤ p.z0 ∨ p.z1 ≤ m.z0 ∨ p.x1 ≤ m.x0) :
    ∀ t : Int, 0 ≤ t → ¬ collide (shiftX m t) p := by
  intro t ht
  simp only [collide, shiftX]
  omega

/-- Slid in from the - side of x. -/
theorem slide_clear_x_neg (m p : Box)
    (h : m.y1 ≤ p.y0 ∨ p.y1 ≤ m.y0 ∨ m.z1 ≤ p.z0 ∨ p.z1 ≤ m.z0 ∨ m.x1 ≤ p.x0) :
    ∀ t : Int, 0 ≤ t → ¬ collide (shiftX m (-t)) p := by
  intro t ht
  simp only [collide, shiftX]
  omega

/-- Slid in from the + side of y. -/
theorem slide_clear_y_pos (m p : Box)
    (h : m.x1 ≤ p.x0 ∨ p.x1 ≤ m.x0 ∨ m.z1 ≤ p.z0 ∨ p.z1 ≤ m.z0 ∨ p.y1 ≤ m.y0) :
    ∀ t : Int, 0 ≤ t → ¬ collide (shiftY m t) p := by
  intro t ht
  simp only [collide, shiftY]
  omega

/-- Slid in from the - side of y. -/
theorem slide_clear_y_neg (m p : Box)
    (h : m.x1 ≤ p.x0 ∨ p.x1 ≤ m.x0 ∨ m.z1 ≤ p.z0 ∨ p.z1 ≤ m.z0 ∨ m.y1 ≤ p.y0) :
    ∀ t : Int, 0 ≤ t → ¬ collide (shiftY m (-t)) p := by
  intro t ht
  simp only [collide, shiftY]
  omega

/-! ## An axle through a hole

  An axle along x at (y = e, z = c) with half-width r occupies points with |y - e| < r and |z - c| < r, wherever it is
  along x. A Technic brick (or a hub) with a round hole on that same line has no plastic inside the channel
  |y - e| < r, |z - c| < r (the hole is the channel). So no point of the axle is ever in the plastic. -/

/-- A point of an axle on the line (e, c), half-width r, at any position along the line. -/
def inAxle (e c r : Int) (_px py pz : Int) : Prop := e - r < py ∧ py < e + r ∧ c - r < pz ∧ pz < c + r

/-- The plastic of a part with a hole of half-width r on the line (e, c): only outside the hole's channel. -/
def plasticOutsideHole (solid : Int → Int → Int → Prop) (e c r : Int) : Prop :=
  ∀ px py pz, solid px py pz → ¬ (e - r < py ∧ py < e + r ∧ c - r < pz ∧ pz < c + r)

theorem hole_clear (solid : Int → Int → Int → Prop) (e c r : Int) (h : plasticOutsideHole solid e c r) :
    ∀ px py pz, inAxle e c r px py pz → ¬ solid px py pz := by
  intro px py pz ha hs
  exact h px py pz hs ha
