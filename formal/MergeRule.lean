/-
  The merge rule of samgo/consolidate.py, proved correct (Lean 4, core only; check with `lean formal/MergeRule.lean`).

  A piece is an axis-aligned box of LEGO cells: x, y in studs, z in plates; height h = 1 (plate) or 3 (brick).
  The rule: two pieces side by side whose union is a rectangle become one piece, and three plates stacked on one
  footprint become one brick. Proved here, for every position and size:
  1. Geometry is held constant: the merged piece fills exactly the cells of the pieces it replaces, and those
     pieces never shared a cell (so no cell is gained, lost or doubled).
  2. Grip is kept: every stud face that another part could grip on the old pieces (top studs, bottom tubes) is a
     stud face of the merged piece at the same place; the only faces that disappear are internal ones, where the
     replaced pieces gripped only each other.
  What is checked at run time, not here: the merged size is a piece LEGO made in that color (a catalogue lookup),
  the pieces have one color and lie in one rigid body (samgo.kinematics.analyse), and samgo/consolidate.py builds
  exactly these merged pieces (tests/test_consolidate.py compares it with this specification).
-/

set_option linter.unusedSectionVars false

structure Piece where
  x : Int
  y : Int
  z : Int
  w : Int
  l : Int
  h : Int

/-- Cell (cx, cy, cz) is filled by piece p. -/
def occ (p : Piece) (cx cy cz : Int) : Prop :=
  p.x ≤ cx ∧ cx < p.x + p.w ∧ p.y ≤ cy ∧ cy < p.y + p.l ∧ p.z ≤ cz ∧ cz < p.z + p.h

/-- A stud on p's top face at column (sx, sy), level sz (what a part above grips). -/
def top (p : Piece) (sx sy sz : Int) : Prop :=
  p.x ≤ sx ∧ sx < p.x + p.w ∧ p.y ≤ sy ∧ sy < p.y + p.l ∧ sz = p.z + p.h

/-- A tube on p's bottom face at column (sx, sy), level sz (what grips a part below). -/
def bottom (p : Piece) (sx sy sz : Int) : Prop :=
  p.x ≤ sx ∧ sx < p.x + p.w ∧ p.y ≤ sy ∧ sy < p.y + p.l ∧ sz = p.z

/-! ## Beside, along x: p then q -/

/-- consolidate._merged when p.x + p.w = q.x: x = min(p.x, q.x) = p.x, w = p.w + q.w. -/
def mergeX (p q : Piece) : Piece := { p with w := p.w + q.w }

section beside_x
variable (p q : Piece)
  (hy : p.y = q.y) (hl : p.l = q.l) (hz : p.z = q.z) (hh : p.h = q.h) (hx : p.x + p.w = q.x)
  (hpw : 0 < p.w) (hqw : 0 < q.w)
include hy hl hz hh hx hpw hqw

theorem mergeX_cells (cx cy cz : Int) : occ (mergeX p q) cx cy cz ↔ occ p cx cy cz ∨ occ q cx cy cz := by
  simp only [occ, mergeX] <;> omega

theorem mergeX_disjoint (cx cy cz : Int) : ¬ (occ p cx cy cz ∧ occ q cx cy cz) := by
  simp only [occ] <;> omega

theorem mergeX_top (sx sy sz : Int) : top (mergeX p q) sx sy sz ↔ top p sx sy sz ∨ top q sx sy sz := by
  simp only [top, mergeX] <;> omega

theorem mergeX_bottom (sx sy sz : Int) : bottom (mergeX p q) sx sy sz ↔ bottom p sx sy sz ∨ bottom q sx sy sz := by
  simp only [bottom, mergeX] <;> omega
end beside_x

/-! ## Beside, along y: p then q -/

def mergeY (p q : Piece) : Piece := { p with l := p.l + q.l }

section beside_y
variable (p q : Piece)
  (hx : p.x = q.x) (hw : p.w = q.w) (hz : p.z = q.z) (hh : p.h = q.h) (hy : p.y + p.l = q.y)
  (hpl : 0 < p.l) (hql : 0 < q.l)
include hx hw hz hh hy hpl hql

theorem mergeY_cells (cx cy cz : Int) : occ (mergeY p q) cx cy cz ↔ occ p cx cy cz ∨ occ q cx cy cz := by
  simp only [occ, mergeY] <;> omega

theorem mergeY_disjoint (cx cy cz : Int) : ¬ (occ p cx cy cz ∧ occ q cx cy cz) := by
  simp only [occ] <;> omega

theorem mergeY_top (sx sy sz : Int) : top (mergeY p q) sx sy sz ↔ top p sx sy sz ∨ top q sx sy sz := by
  simp only [top, mergeY] <;> omega

theorem mergeY_bottom (sx sy sz : Int) : bottom (mergeY p q) sx sy sz ↔ bottom p sx sy sz ∨ bottom q sx sy sz := by
  simp only [bottom, mergeY] <;> omega
end beside_y

/-- The rule never depends on which of the two pieces is listed first: merging q then p is the x-merge of p, q. -/
theorem mergeX_order (p q : Piece) (cx cy cz : Int) (hx : p.x + p.w = q.x) (hy : p.y = q.y) (hl : p.l = q.l)
    (hz : p.z = q.z) (hh : p.h = q.h) (hpw : 0 < p.w) (hqw : 0 < q.w) :
    occ (mergeX p q) cx cy cz ↔ occ q cx cy cz ∨ occ p cx cy cz := by
  simp only [occ, mergeX] <;> omega

/-! ## Above/below: three plates p1, p2, p3 stacked on one footprint -> one brick -/

/-- consolidate: dict(p1, type="brick"): the bottom plate's footprint and level, height 3. -/
def stack (p1 : Piece) : Piece := { p1 with h := 3 }

section stacked
variable (p1 p2 p3 : Piece)
  (h1 : p1.h = 1) (h2 : p2.h = 1) (h3 : p3.h = 1)
  (x2 : p2.x = p1.x) (x3 : p3.x = p1.x) (y2 : p2.y = p1.y) (y3 : p3.y = p1.y)
  (w2 : p2.w = p1.w) (w3 : p3.w = p1.w) (l2 : p2.l = p1.l) (l3 : p3.l = p1.l)
  (z2 : p2.z = p1.z + 1) (z3 : p3.z = p1.z + 2)
include h1 h2 h3 x2 x3 y2 y3 w2 w3 l2 l3 z2 z3

theorem stack_cells (cx cy cz : Int) :
    occ (stack p1) cx cy cz ↔ occ p1 cx cy cz ∨ occ p2 cx cy cz ∨ occ p3 cx cy cz := by
  simp only [occ, stack] <;> omega

theorem stack_disjoint (cx cy cz : Int) :
    ¬ (occ p1 cx cy cz ∧ occ p2 cx cy cz) ∧ ¬ (occ p2 cx cy cz ∧ occ p3 cx cy cz) ∧
    ¬ (occ p1 cx cy cz ∧ occ p3 cx cy cz) := by
  simp only [occ] <;> omega

/-- What a part above grips: the brick's top studs are exactly the top plate's. -/
theorem stack_top (sx sy sz : Int) : top (stack p1) sx sy sz ↔ top p3 sx sy sz := by
  simp only [top, stack] <;> omega

/-- What grips a part below: the brick's bottom tubes are exactly the bottom plate's. -/
theorem stack_bottom (sx sy sz : Int) : bottom (stack p1) sx sy sz ↔ bottom p1 sx sy sz := by
  simp only [bottom, stack] <;> omega

/-- The faces that disappear are internal: every top stud of p1 or p2 is a bottom tube of the plate right on it,
    so no outside part could grip there (the cell above it is filled by that plate). -/
theorem stack_internal (sx sy sz : Int) :
    (top p1 sx sy sz → bottom p2 sx sy sz) ∧ (top p2 sx sy sz → bottom p3 sx sy sz) := by
  simp only [top, bottom] <;> omega
end stacked
