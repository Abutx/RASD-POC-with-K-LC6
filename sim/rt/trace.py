"""Angle-lookup tracer: one rotor revolution per rotor, cached to disk.

A rotor's return depends only on its angle once radar, body and floor are
fixed. So each rotor is traced over half a turn (two identical blades -> the
pattern repeats every pi) at fine angular steps, ONCE, and synth.py drives the
lookup with any RPM profile. The other three rotors are removed while one is
traced, so what is captured is body/arm/motor shadowing of the blades, not
rotor-on-rotor shadowing at artificially equal angles.

Geometry conventions match sim/drone.py: the drone is at the origin in its
body frame with x forward; radar position is given by range, azimuth and
elevation as seen FROM THE DRONE. Elevation 0 = level with the rotor plane,
POSITIVE = radar above the drone, negative = below. (The analytic model is
symmetric in the sign; this one is not, because the body sits under the
rotors and the floor is below everything.)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import mesh, po
from .. import klc6

CACHE_DIR = Path(__file__).resolve().parents[2] / "out" / "rt_cache"


def radar_position(range_m, az_deg=0.0, el_deg=0.0):
    el, az = np.radians(el_deg), np.radians(az_deg)
    return np.array([range_m * np.cos(el) * np.cos(az),
                     range_m * np.cos(el) * np.sin(az),
                     range_m * np.sin(el)])


def _tris(v, f):
    return np.asarray(v, dtype=float)[np.asarray(f)]


class DroneTracer:
    """PO facet model of the Mini 3 at a fixed pose relative to the radar."""

    def __init__(self, range_m=3.0, az_deg=0.0, el_deg=0.0, height_m=1.2,
                 floor=True, pol="TE", yaw_deg=0.0):
        self.range_m = float(range_m)
        self.az_deg, self.el_deg = float(az_deg), float(el_deg)
        self.height_m = float(height_m)
        self.floor = bool(floor)
        self.pol = pol
        self.yaw = np.radians(yaw_deg)
        self.radar = radar_position(range_m, az_deg, el_deg) + [0, 0, height_m]
        # static parts, in world frame
        self.static = {}
        for name, (v, f) in mesh.airframe_static_meshes().items():
            self.static[name] = self._place(_tris(v, f))
        self.static_tris = np.vstack(list(self.static.values()))

    def _place(self, tris):
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return tris @ Rz.T + [0, 0, self.height_m]

    def _eps(self, name):
        return po.PEC if (name == "body" or name.startswith("motor")) else po.EPS_NYLON_GF

    def key(self, n_phi):
        d = dict(range=self.range_m, az=self.az_deg, el=self.el_deg, h=self.height_m,
                 floor=self.floor, pol=self.pol, yaw=float(np.degrees(self.yaw)),
                 n_phi=n_phi, f0=klc6.F_CARRIER, mesh_v=2, occlusion="soft4")
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12], d

    # -------------------------------------------------------- aspect sweep
    def static_yaw_sweep(self, step_deg=5.0):
        """Static-airframe RCS vs drone heading -- the aspect-averaged body number.

        The body block at exact broadside is a specular peak; measured drone
        RCS in the literature is averaged over aspect. This is the fair
        comparison. Rotors excluded (their mean is heading-independent to
        first order).
        """
        yaws = np.arange(0.0, 360.0, step_deg)
        out = np.zeros(len(yaws))
        saved = self.yaw
        for i, y in enumerate(yaws):
            self.yaw = np.radians(y)
            self.static = {n: self._place(_tris(v, f))
                           for n, (v, f) in mesh.airframe_static_meshes().items()}
            self.static_tris = np.vstack(list(self.static.values()))
            out[i] = abs(sum(self.static_return().values())) ** 2
        self.yaw = saved
        self.static = {n: self._place(_tris(v, f))
                       for n, (v, f) in mesh.airframe_static_meshes().items()}
        self.static_tris = np.vstack(list(self.static.values()))
        return yaws, out

    # ------------------------------------------------------------ static
    def static_return(self):
        """Per-part complex RCS amplitude with no rotors present."""
        fz = 0.0 if self.floor else None
        out = {}
        for name, tris in self.static.items():
            others = np.vstack([t for n, t in self.static.items() if n != name])
            out[name] = po.part_return(tris, self.radar, blockers=others,
                                       eps_r=self._eps(name), pol=self.pol,
                                       floor_z=fz, r_ref=self.range_m)
        return out

    # ------------------------------------------------------------ rotors
    def rotor_return(self, i, phi):
        """Complex RCS amplitude of rotor i at angle phi, others absent."""
        v, f = mesh.rotor_mesh(phi, mesh.ROTOR_SPIN[i], mesh.rotor_hub(i))
        tris = self._place(_tris(v, f))
        fz = 0.0 if self.floor else None
        return po.part_return(tris, self.radar, blockers=self.static_tris,
                              eps_r=po.EPS_NYLON_GF, pol=self.pol,
                              floor_z=fz, r_ref=self.range_m)

    def trace(self, n_phi=1440, verbose=True, use_cache=True):
        """Return dict with 'phi' (n_phi,), 'rotor' (4, n_phi) complex, 'static'."""
        key, meta = self.key(n_phi)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = CACHE_DIR / f"trace_{key}.npz"
        if use_cache and path.exists():
            z = np.load(path, allow_pickle=True)
            if verbose:
                print(f"  [cache] {path.name}")
            return {"phi": z["phi"], "rotor": z["rotor"],
                    "static": json.loads(str(z["static_json"])), "meta": meta}
        phi = np.linspace(0.0, np.pi, n_phi, endpoint=False)
        rotor = np.zeros((4, n_phi), dtype=complex)
        if verbose:
            print(f"  tracing 4 rotors x {n_phi} angles (PO facets, "
                  f"{'floor on' if self.floor else 'no floor'}) ...")
        for i in range(4):
            for j, p in enumerate(phi):
                rotor[i, j] = self.rotor_return(i, p)
            if verbose:
                sig = np.mean(np.abs(rotor[i]) ** 2)
                print(f"    rotor{i}: mean RCS {10*np.log10(sig+1e-30):6.1f} dBsm, "
                      f"peak {10*np.log10(np.max(np.abs(rotor[i])**2)+1e-30):6.1f} dBsm")
        static = self.static_return()
        if verbose:
            tot = abs(sum(static.values())) ** 2
            print(f"    static parts: total {10*np.log10(tot+1e-30):6.1f} dBsm  "
                  + "  ".join(f"{k}:{10*np.log10(abs(v)**2+1e-30):.0f}" for k, v in static.items()))
        static_ser = {k: [v.real, v.imag] for k, v in static.items()}
        np.savez_compressed(path, phi=phi, rotor=rotor,
                            static_json=json.dumps(static_ser), meta=json.dumps(meta))
        return {"phi": phi, "rotor": rotor, "static": static_ser, "meta": meta}
