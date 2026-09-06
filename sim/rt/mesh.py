"""DJI Mini 3 geometry as triangle meshes, written to PLY for Sionna RT.

Everything is in metres, body frame: x forward, y left, z up, origin at the
body centre. Rotor hubs at the four arm tips; each rotor's blades are built in
a local frame at the hub and rotated about +z by the rotor angle before
writing, so a rotor at angle phi is a distinct mesh file.

Dimensions -- the ones that matter for 24 GHz scattering:
  props        6030F: 6 in diameter (R = 76.2 mm), 3 in pitch, 2 blades.
               Chord ~12 mm at 30% span tapering to ~8 mm at the tip,
               ~1.5 mm thick. Pitch angle beta(r) = atan(P / 2 pi r) from the
               3 in pitch: 31 deg at r = 20 mm falling to 9 deg at the tip.
               That twist is what an analytic line-scatterer model cannot
               represent, and at lambda = 12.4 mm a blade chord is ~1 lambda
               wide -- the specular flash is a real, aspect-dependent event.
  motors       1504C: ~18 mm can diameter, ~12 mm tall. Metal. Four of them
               at the arm tips are the drone's most reliable scatterers.
  body         248 g airframe, 148 x 90 x 62 mm folded. The plastic shell is
               nearly transparent at 24 GHz; what reflects is the battery and
               frame inside. Modelled as a metal block 100 x 60 x 35 mm.
  arms         plastic, ~10 mm section, body corner to motor.
  diagonal     247 mm motor-to-motor.

Materials are assigned in scene.py; this file only writes shapes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# ---- DJI Mini 3 (same numbers as sim/drone.py where they overlap) ----
PROP_R = 0.0762
PROP_PITCH = 0.0762                # 3 in
BLADE_ROOT_FRAC = 0.15             # hub/root region not modelled as blade
CHORD_ROOT = 0.012
CHORD_TIP = 0.008
BLADE_THICK = 0.0015
N_BLADES = 2

MOTOR_R = 0.009
MOTOR_H = 0.012
ARM_HALF = 0.005                   # 10 mm square section
ARM_LEN_DIAG = 0.247 / 2           # hub distance from centre
BODY = (0.100, 0.060, 0.035)       # metal internals block, LxWxH
HUB_Z = BODY[2] / 2 + MOTOR_H + 0.004   # prop plane above body top

ROTOR_XY = np.array([[+1, +1], [-1, +1], [-1, -1], [+1, -1]]) * ARM_LEN_DIAG / np.sqrt(2)
ROTOR_SPIN = np.array([+1, -1, +1, -1])          # CCW, CW, CCW, CW


# ------------------------------------------------------------------ PLY
def write_ply(path, verts, faces):
    """Binary little-endian PLY, triangles only. Mitsuba loads this directly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(verts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\nend_header\n"
    ).encode("ascii")
    face_rec = np.empty(len(faces), dtype=[("n", "u1"), ("i", "<i4", (3,))])
    face_rec["n"] = 3
    face_rec["i"] = faces
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(verts.astype("<f4").tobytes())
        fh.write(face_rec.tobytes())
    return path


def _box(center, size):
    cx, cy, cz = center
    sx, sy, sz = np.asarray(size) / 2.0
    v = np.array([[cx - sx, cy - sy, cz - sz], [cx + sx, cy - sy, cz - sz],
                  [cx + sx, cy + sy, cz - sz], [cx - sx, cy + sy, cz - sz],
                  [cx - sx, cy - sy, cz + sz], [cx + sx, cy - sy, cz + sz],
                  [cx + sx, cy + sy, cz + sz], [cx - sx, cy + sy, cz + sz]])
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],   # bottom, top
                  [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],   # sides
                  [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return v, f


def _cylinder(center, radius, height, n=24):
    cx, cy, cz = center
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring_lo = np.stack([cx + radius * np.cos(ang), cy + radius * np.sin(ang),
                        np.full(n, cz - height / 2)], axis=1)
    ring_hi = ring_lo + [0, 0, height]
    v = np.vstack([ring_lo, ring_hi, [[cx, cy, cz - height / 2]], [[cx, cy, cz + height / 2]]])
    lo, hi, cb, ct = np.arange(n), np.arange(n) + n, 2 * n, 2 * n + 1
    f = []
    for i in range(n):
        j = (i + 1) % n
        f += [[lo[i], lo[j], hi[j]], [lo[i], hi[j], hi[i]],   # wall
              [cb, lo[j], lo[i]], [ct, hi[i], hi[j]]]         # caps
    return v, np.array(f)


def _merge(parts):
    verts, faces, off = [], [], 0
    for v, f in parts:
        verts.append(v)
        faces.append(np.asarray(f) + off)
        off += len(v)
    return np.vstack(verts), np.vstack(faces)


# ---------------------------------------------------------------- blades
def blade_mesh(n_span=24, n_chord=2):
    """One blade along +x from the hub, twisted per the 3 in pitch.

    Built as a thin twisted ribbon of quads (two triangles each), both faces,
    so the ray tracer sees a real surface with a chord and a pitch angle.
    Spanwise stations are denser toward the tip where the Doppler lives.
    """
    r = PROP_R * (BLADE_ROOT_FRAC + (1 - BLADE_ROOT_FRAC)
                  * np.linspace(0, 1, n_span) ** 0.8)
    chord = CHORD_ROOT + (CHORD_TIP - CHORD_ROOT) * (r - r[0]) / (r[-1] - r[0])
    beta = np.arctan(PROP_PITCH / (2 * np.pi * r))          # pitch angle
    verts = []
    for ri, ci, bi in zip(r, chord, beta):
        # chord line lies in the y-z plane at station x = ri, rotated by
        # the pitch angle about the spanwise (x) axis
        for s in np.linspace(-0.5, 0.5, n_chord + 1):
            y = s * ci * np.cos(bi)
            z = s * ci * np.sin(bi)
            verts.append([ri, y, z])
    verts = np.array(verts)
    ncol = n_chord + 1
    faces = []
    for i in range(n_span - 1):
        for j in range(n_chord):
            a = i * ncol + j
            b = a + 1
            c = a + ncol
            d = c + 1
            faces += [[a, c, b], [b, c, d]]
    faces = np.array(faces)
    # give the ribbon thickness: duplicate offset along the local normal-ish z
    up = verts + [0, 0, BLADE_THICK / 2]
    dn = verts - [0, 0, BLADE_THICK / 2]
    v = np.vstack([up, dn])
    f = np.vstack([faces, faces[:, ::-1] + len(verts)])     # both sheets
    return v, f


def rotor_mesh(phi, spin=+1, hub=(0.0, 0.0, 0.0)):
    """Two blades at rotor angle phi (rad), about +z at the hub."""
    bv, bf = blade_mesh()
    parts = []
    for k in range(N_BLADES):
        ang = spin * phi + 2 * np.pi * k / N_BLADES
        c, s = np.cos(ang), np.sin(ang)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        parts.append((bv @ R.T + np.asarray(hub), bf))
    return _merge(parts)


# ---------------------------------------------------------------- airframe
def airframe_static_meshes():
    """Body block, four arms, four motor cans -- everything that doesn't spin.

    Returned as {name: (verts, faces)} so scene.py can give the motors metal
    and the arms plastic. The body block is 'metal' on the argument that the
    battery and frame, not the shell, are what reflect at 24 GHz.
    """
    out = {"body": _box((0, 0, 0), BODY)}
    for i, (x, y) in enumerate(ROTOR_XY):
        # arm from body corner toward the hub
        cx, cy = x * 0.55, y * 0.55
        L = np.hypot(x, y) * 0.9
        ang = np.arctan2(y, x)
        v, f = _box((0, 0, 0), (L, 2 * ARM_HALF, 2 * ARM_HALF))
        c, s = np.cos(ang), np.sin(ang)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        out[f"arm{i}"] = (v @ R.T + [cx, cy, 0.0], f)
        out[f"motor{i}"] = _cylinder((x, y, BODY[2] / 2 + MOTOR_H / 2), MOTOR_R, MOTOR_H)
    return out


def rotor_hub(i):
    x, y = ROTOR_XY[i]
    return np.array([x, y, HUB_Z])


def write_drone(out_dir, phi=None):
    """Write all parts. phi: array of 4 rotor angles (rad) or None for zeros."""
    out_dir = Path(out_dir)
    phi = np.zeros(4) if phi is None else np.asarray(phi)
    files = {}
    for name, (v, f) in airframe_static_meshes().items():
        files[name] = write_ply(out_dir / f"{name}.ply", v, f)
    for i in range(4):
        v, f = rotor_mesh(phi[i], ROTOR_SPIN[i], rotor_hub(i))
        files[f"rotor{i}"] = write_ply(out_dir / f"rotor{i}.ply", v, f)
    return files


def write_floor(out_dir, size=30.0, z=0.0):
    """A big square floor slab -- the one multipath surface that always exists."""
    v = np.array([[-size, -size, z], [size, -size, z], [size, size, z], [-size, size, z]])
    f = np.array([[0, 1, 2], [0, 2, 3]])
    return write_ply(Path(out_dir) / "floor.ply", v, f)


def write_calibration_plate(out_dir, side=0.30):
    """Square metal plate, normal along -x (facing a radar on +x).

    Flat-plate RCS at normal incidence is 4 pi A^2 / lambda^2: for a 30 cm
    square at 24.125 GHz that is 4pi * 0.09^2 / 0.012427^2 = 659 m^2 = 28.2 dBsm.
    Big, sharp, and analytic -- the right thing to check the tracer against
    before believing anything it says about a drone.
    """
    h = side / 2
    v = np.array([[0, -h, -h], [0, h, -h], [0, h, h], [0, -h, h]])
    f = np.array([[0, 2, 1], [0, 3, 2], [0, 1, 2], [0, 2, 3]])   # both faces
    return write_ply(Path(out_dir) / "plate.ply", v, f)


def plate_rcs_m2(side, lam):
    return 4 * np.pi * (side ** 2) ** 2 / lam ** 2
