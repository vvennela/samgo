"""Small exact checker for gravity-only, top-down brick/plate assembly certificates.

No solver, floating-point arithmetic or sampled paths are trusted by this checker.
This Python implementation is tested, not itself proved in a theorem prover. Its
claims concern the explicit idealized model below, not manufactured-part safety.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from fractions import Fraction

from .parts import parse_piece

VERSION = "samgo-static-assembly-certificate-v1"
MAX_PARTS = 200
LIMITATIONS = [
    "Axis-aligned standard bricks/plates only; ideal solid envelopes and compatible stud mating.",
    "Flat rigid ground, unit weight per occupied plate-cell at each part's geometric center.",
    "Only vertical compressive forces; no clutch tension, horizontal loads or dynamic motion.",
    "Contact capacity is a declared model parameter, not a measured material-strength guarantee.",
    "Every resting prefix is checked; insertion forces, hand/gripper access and tolerances are not modeled.",
    "No real-world build test or machine-checked proof of this Python checker has been performed.",
]


class CertificateError(ValueError):
    pass


class Unsupported(CertificateError):
    pass


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def rational(value):
    # JSON floats are deliberately not accepted as exact witnesses or policy values.
    if type(value) is int:
        return Fraction(value)
    if not isinstance(value, str) or len(value) > 128:
        raise CertificateError("expected a bounded rational string or integer")
    fields = value.split("/")
    if len(fields) > 2 or any(not x.lstrip("-").isdigit() for x in fields):
        raise CertificateError("invalid rational encoding")
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as e:
        raise CertificateError("invalid rational encoding") from e


@dataclass(frozen=True)
class Policy:
    # In the same abstract force units as a unit-weight plate cell. This is an
    # explicit assumption, NOT the old tensile clutch estimate in stability.py.
    compression_per_stud: str = "250"

    def __post_init__(self):
        if rational(self.compression_per_stud) <= 0:
            raise ValueError("compression capacity must be positive")

    def spec(self):
        return dict(
            version=VERSION,
            policy=asdict(self),
            contact_offset="1/4",
            ground="z=0",
            weight="w*l*height",
            insertion="vertical_from_above",
            temporary_supports=False,
            tensile_forces=False,
            final_components=1,
            limitations=LIMITATIONS,
        )


DEFAULT_POLICY = Policy()


def parse_build(raw):
    if not isinstance(raw, list) or not raw:
        raise CertificateError("assembly must be a nonempty list")
    if len(raw) > MAX_PARTS:
        raise Unsupported(f"certificate scope is limited to {MAX_PARTS} parts")
    pieces = []
    for i, p in enumerate(raw):
        if not isinstance(p, dict):
            raise CertificateError(f"part {i} must be an object")
        if p.get("type") not in ("brick", "plate"):
            raise Unsupported(f"part {i}: type {p.get('type')!r} has no certified geometry/contact model")
        if set(p) != {"type", "w", "l", "x", "y", "z", "color"}:
            raise Unsupported(f"part {i}: extra or missing fields have no certified semantics")
        try:
            part = parse_piece(p)
        except (ValueError, TypeError, KeyError) as e:
            raise CertificateError(f"part {i}: {e}") from e
        if part.z < 0:
            raise CertificateError(f"part {i}: below the ground")
        if max(abs(part.x), abs(part.y), part.z) > 1_000_000:
            raise Unsupported("coordinate resource limit exceeded")
        pieces.append(part)
    return pieces


def xy_overlap(a, b):
    return max(a.x, b.x) < min(a.x + a.w, b.x + b.w) and max(a.y, b.y) < min(a.y + a.l, b.y + b.l)


def validate_geometry(pieces):
    adjacent = [set() for _ in pieces]
    for i, a in enumerate(pieces):
        for j, b in enumerate(pieces[:i]):
            if not xy_overlap(a, b):
                continue
            if max(a.z, b.z) < min(a.z + a.h, b.z + b.h):
                raise CertificateError(f"solid-envelope overlap: parts {j}, {i}")
            if a.z == b.z + b.h or b.z == a.z + a.h:
                adjacent[i].add(j)
                adjacent[j].add(i)
    seen, pending = set(), [0]
    while pending:
        i = pending.pop()
        if i not in seen:
            seen.add(i)
            pending.extend(adjacent[i] - seen)
    if len(seen) != len(pieces):
        raise CertificateError("final build is not one stud-connected assembly")
    if not any(p.z == 0 for p in pieces):
        raise CertificateError("final build has no ground contact")


def validate_order(order, count):
    if (
        not isinstance(order, list)
        or len(order) != count
        or any(type(i) is not int for i in order)
        or sorted(order) != list(range(count))
    ):
        raise CertificateError("order must contain every original part index exactly once")


def check_path(pieces, placed, index):
    """Exact swept-volume condition for the entire vertical approach, not sampled poses.

    Start above every part's top. For intersecting x/y interiors, any obstacle
    whose top exceeds the destination bottom intersects this swept volume.
    Intended stud mating is abstracted into the catalog contact rule.
    """
    p = pieces[index]
    for j in placed:
        if xy_overlap(p, pieces[j]) and pieces[j].z + pieces[j].h > p.z:
            raise CertificateError(f"vertical insertion of part {index} is blocked by part {j}")


def contacts(pieces, placed):
    """Canonical contact order; witnesses cannot invent supports or contact positions.

    Four inset points per shared stud. Ground support uses four inset footprint
    corners. All x/y positions are exact fractions. down=None denotes ground.
    """
    result = []
    quarter = Fraction(1, 4)
    for i in placed:
        a = pieces[i]
        if a.z == 0:
            for x in (a.x + quarter, a.x + a.w - quarter):
                for y in (a.y + quarter, a.y + a.l - quarter):
                    result.append((i, None, x, y, None))
        for j in placed:
            b = pieces[j]
            if i == j or a.z != b.z + b.h or not xy_overlap(a, b):
                continue
            for x in range(max(a.x, b.x), min(a.x + a.w, b.x + b.w)):
                for y in range(max(a.y, b.y), min(a.y + a.l, b.y + b.l)):
                    for dx in (quarter, 1 - quarter):
                        for dy in (quarter, 1 - quarter):
                            result.append((i, j, x + dx, y + dy, (i, j, x, y)))
    return result


def check_forces(pieces, placed, witness, policy):
    cs = contacts(pieces, placed)
    if not isinstance(witness, list) or len(witness) != len(cs):
        raise CertificateError("force witness length differs from derived contacts")
    force = {i: Fraction(0) for i in placed}
    tx, ty, loads = force.copy(), force.copy(), {}
    for contact, encoded in zip(cs, witness):
        up, down, x, y, stud = contact
        f = rational(encoded)
        if f < 0:
            raise CertificateError("tensile force is outside the compression-only model")
        if stud is not None:
            loads[stud] = loads.get(stud, Fraction(0)) + f
        for index, sign in ((up, 1), (down, -1)):
            if index is None:
                continue
            p = pieces[index]
            force[index] += sign * f
            tx[index] += sign * f * (x - p.x - Fraction(p.w, 2))
            ty[index] += sign * f * (y - p.y - Fraction(p.l, 2))
    if any(v > rational(policy.compression_per_stud) for v in loads.values()):
        raise CertificateError("declared per-stud compression capacity exceeded")
    for i in placed:
        p = pieces[i]
        if force[i] != p.w * p.l * p.h or tx[i] != 0 or ty[i] != 0:
            raise CertificateError(f"force/torque balance fails exactly for part {i}")


def check_certificate(raw, certificate, *, policy=DEFAULT_POLICY):
    """Return a verdict only after checking the entire witness against caller policy.

    The certificate's own policy/status fields are never trusted as authority.
    """
    try:
        pieces = parse_build(raw)
        validate_geometry(pieces)
        if not isinstance(certificate, dict) or set(certificate) != {
            "version",
            "build_sha256",
            "spec_sha256",
            "order",
            "stages",
        }:
            raise CertificateError("invalid certificate schema")
        if (
            certificate["version"] != VERSION
            or certificate["build_sha256"] != digest(raw)
            or certificate["spec_sha256"] != digest(policy.spec())
        ):
            raise CertificateError("certificate does not match this build and required specification")
        order = certificate["order"]
        validate_order(order, len(pieces))
        stages = certificate["stages"]
        if not isinstance(stages, list) or len(stages) != len(order):
            raise CertificateError("one checked witness is required per assembly step")
        placed = []
        for i, stage in zip(order, stages):
            if (
                not isinstance(stage, dict)
                or set(stage) != {"part", "forces"}
                or type(stage["part"]) is not int
                or stage["part"] != i
            ):
                raise CertificateError("stage does not match insertion order")
            check_path(pieces, placed, i)
            placed.append(i)
            check_forces(pieces, placed, stage["forces"], policy)
        return dict(
            passed=True, status="certified_under_model", steps=len(order), specification=policy.spec()
        )
    except (ValueError, TypeError, KeyError, OverflowError) as e:
        return dict(passed=False, status="not_certified", issues=[str(e)], specification=policy.spec())
