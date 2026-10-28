"""The SAMgo part model: 22 rectangular bricks and plates on a stud grid.

Units: x and y are studs, z is plate heights (a brick is 3 plates tall).
A part is described by its size, not a catalog ID. There is no rotation
field: a 4x2 is a 2x4 turned sideways, so w (along x) and l (along y) may
come in either order.
"""

from dataclasses import dataclass

HEIGHTS = {"plate": 1, "brick": 3}

# Footprints with the short side first; (w, l) is valid if sorted((w, l)) is here.
FOOTPRINTS = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 6), (1, 8), (2, 2), (2, 3), (2, 4), (2, 6), (2, 8)]

# Used only when exporting parts lists and .ldr files, never for validity.
LDRAW_IDS = {
    ("brick", 1, 1): "3005", ("brick", 1, 2): "3004", ("brick", 1, 3): "3622",
    ("brick", 1, 4): "3010", ("brick", 1, 6): "3009", ("brick", 1, 8): "3008",
    ("brick", 2, 2): "3003", ("brick", 2, 3): "3002", ("brick", 2, 4): "3001",
    ("brick", 2, 6): "2456", ("brick", 2, 8): "3007",
    ("plate", 1, 1): "3024", ("plate", 1, 2): "3023", ("plate", 1, 3): "3623",
    ("plate", 1, 4): "3710", ("plate", 1, 6): "3666", ("plate", 1, 8): "3460",
    ("plate", 2, 2): "3022", ("plate", 2, 3): "3021", ("plate", 2, 4): "3020",
    ("plate", 2, 6): "3795", ("plate", 2, 8): "3034",
}

# name -> (LDraw color code, approximate sRGB for nearest-color mapping).
COLORS = {
    "black": (0, (27, 42, 52)),
    "blue": (1, (30, 90, 168)),
    "green": (2, (0, 133, 43)),
    "red": (4, (180, 0, 0)),
    "yellow": (14, (250, 200, 10)),
    "white": (15, (244, 244, 244)),
    "orange": (25, (214, 121, 35)),
    "tan": (19, (228, 205, 158)),
    "reddish_brown": (70, (95, 49, 9)),
    "light_gray": (71, (150, 150, 150)),
    "dark_gray": (72, (100, 100, 100)),
}

# Target grids store colors as these indices; 0 means empty.
COLOR_INDEX = {name: i + 1 for i, name in enumerate(COLORS)}


class PartError(ValueError):
    pass


@dataclass(frozen=True)
class Piece:
    w: int
    l: int
    type: str
    x: int
    y: int
    z: int
    color: str

    @property
    def h(self) -> int:
        return HEIGHTS[self.type]

    @property
    def cells(self) -> tuple[slice, slice, slice]:
        return (slice(self.x, self.x + self.w), slice(self.y, self.y + self.l), slice(self.z, self.z + self.h))

    @property
    def ldraw_id(self) -> str:
        return LDRAW_IDS[(self.type, *sorted((self.w, self.l)))]


def parse_piece(raw: dict) -> Piece:
    """Turn one model-written piece into a Piece, or raise PartError."""
    if not isinstance(raw, dict):
        raise PartError("piece must be an object")
    fields = ("w", "l", "type", "x", "y", "z", "color")
    missing = [f for f in fields if f not in raw]
    if missing:
        raise PartError(f"missing fields: {', '.join(missing)}")
    for f in ("w", "l", "x", "y", "z"):
        # bool is a subclass of int, so check the exact type.
        if type(raw[f]) is not int:
            raise PartError(f"{f} must be an integer")
    if raw["type"] not in HEIGHTS:
        raise PartError(f"unknown type {raw['type']!r}")
    if tuple(sorted((raw["w"], raw["l"]))) not in FOOTPRINTS:
        raise PartError(f"no {raw['w']}x{raw['l']} {raw['type']} in the palette")
    if raw["color"] not in COLORS:
        raise PartError(f"unknown color {raw['color']!r}")
    return Piece(**{f: raw[f] for f in fields})
