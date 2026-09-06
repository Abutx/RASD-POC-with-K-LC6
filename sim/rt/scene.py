"""Sionna RT scene for a DJI Mini 3 seen by a K-LC6 (or TinyRad) radar.

Materials, antenna patterns and placement. The rotor angles are set by
trace.py; this module only knows how to build the world they spin in.

Scattering model -- read this before trusting any amplitude:
  Sionna RT is a geometric-optics tracer. A blade chord is ~1 wavelength at
  24 GHz, where GO is marginal. Pure specular reflection from such a surface
  connects to a point receiver with probability ~0, so it contributes nothing
  in a shooting-and-bouncing-rays simulation no matter how many rays you
  throw. Physical optics says the real return is a specular LOBE of angular
  width ~lambda/D. Sionna's 'directive' diffuse pattern (Degli-Esposti) is
  exactly that shape, and it connects to the receiver deterministically. So
  the blades and small parts use directive scattering with (S, alpha_r)
  CALIBRATED against the closed-form PO RCS of a blade-sized plate -- see
  scripts/rt_calib.py. That is a calibration, not first-principles physics,
  and it is the accuracy ceiling of this method for lambda-scale features.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import drjit as dr
import mitsuba as mi
from sionna.rt import (load_scene, PlanarArray, Transmitter, Receiver,
                       SceneObject, RadioMaterial, PolarizedAntennaPattern,
                       register_antenna_pattern)

from . import mesh
from .. import klc6

C = klc6.C
F0 = klc6.F_CARRIER

# ---- scattering calibration (filled from scripts/rt_calib.py) ----
# Directive lobe: alpha_r sets the lobe width (larger = narrower), S the share
# of reflected power that goes into the lobe rather than the ideal specular.
BLADE_S = 0.8
BLADE_ALPHA = 4
METAL_S = 0.8
METAL_ALPHA = 4


# ------------------------------------------------------------- materials
def materials():
    """Radio materials for the drone and the room, 24 GHz values.

    nylon + glass fibre (props, arms): eps_r ~3.7, tan-delta ~0.015 ->
        sigma = 2 pi f eps0 eps_r tan-delta = 0.075 S/m
    metal (motors, battery/frame block): PEC-ish
    concrete floor: ITU-R P.2040 at 24 GHz, eps_r 5.24, sigma ~0.8 S/m
    """
    nylon = RadioMaterial(name="nylon_gf", relative_permittivity=3.7,
                          conductivity=0.075, scattering_coefficient=BLADE_S,
                          scattering_pattern="directive", alpha_r=BLADE_ALPHA)
    metal = RadioMaterial(name="drone_metal", relative_permittivity=1.0,
                          conductivity=1e7, scattering_coefficient=METAL_S,
                          scattering_pattern="directive", alpha_r=METAL_ALPHA)
    floor = RadioMaterial(name="concrete_24g", relative_permittivity=5.24,
                          conductivity=0.80, scattering_coefficient=0.4,
                          scattering_pattern="lambertian")
    return {"nylon_gf": nylon, "drone_metal": metal, "concrete_24g": floor}


# ------------------------------------------------------- antenna patterns
def _gaussian_pattern_factory(gain_dbi, bw_az_deg, bw_el_deg):
    """Build a Sionna v_pattern with a Gaussian main lobe.

    Convention (from v_tr38901_pattern): theta is zenith angle, boresight at
    theta = pi/2, phi = 0. -3 dB at half the beamwidth in each plane. The
    pattern returns the FIELD (sqrt of linear gain), peak gain folded in, so
    Sionna's path coefficients carry Gt*Gr and |sum a|^2 is Pr/Pt directly.
    """
    # Plain Python floats: a numpy float64 times a Dr.Jit Float broadcasts to
    # an ndarray and dr.exp() rejects it.
    g_lin = float(10 ** (gain_dbi / 10.0))
    k_el = float(np.log(2.0) / np.radians(bw_el_deg / 2.0) ** 2)   # -3 dB at half-width
    k_az = float(np.log(2.0) / np.radians(bw_az_deg / 2.0) ** 2)
    floor_lin = g_lin * 1e-3                                       # -30 dB back lobe

    def v_pattern(theta: mi.Float, phi: mi.Float) -> mi.Complex2f:
        phi = phi + dr.pi
        phi -= dr.floor(phi / (2.0 * dr.pi)) * 2.0 * dr.pi
        phi -= dr.pi
        el = theta - dr.pi / 2.0
        # power pattern: g * exp(-ln2 * (el/(bw/2))^2) is exactly -3 dB at bw/2
        a = g_lin * dr.exp(-(k_el * el * el + k_az * phi * phi))
        a = dr.maximum(a, floor_lin)
        return mi.Complex2f(dr.sqrt(a), 0.0)

    def factory(polarization="V", polarization_model="tr38901_2"):
        return PolarizedAntennaPattern(v_pattern=v_pattern, polarization=polarization,
                                       polarization_model=polarization_model)
    return factory


_REGISTERED = set()


def register_patterns():
    """K-LC6 (12 x 80 deg, 12.5 dBi) and TinyRad element (76.5 x 17.6, 12.6 dBi).

    The K-LC6 is registered in BOTH mountings, because which lobe is azimuth
    is a mechanical choice (DEMO_TWO_RADAR.md section 6).
    """
    specs = {
        "klc6_wide_az": (12.5, 80.0, 12.0),     # long axis vertical: 80 az x 12 el
        "klc6_narrow_az": (12.5, 12.0, 80.0),   # long axis horizontal
        "tinyrad_elem": (12.6, 76.5, 17.6),
    }
    for name, (g, az, el) in specs.items():
        if name not in _REGISTERED:
            register_antenna_pattern(name, _gaussian_pattern_factory(g, az, el))
            _REGISTERED.add(name)
    return list(specs)


# ------------------------------------------------------------------ scene
class DroneScene:
    """Floor + Mini 3 + radar, with the four rotors replaceable per angle."""

    def __init__(self, work_dir, radar_range_m=3.0, radar_height_m=1.2,
                 drone_height_m=None, radar_el_deg=0.0, radar_az_deg=0.0,
                 pattern="klc6_wide_az", with_floor=True, yaw_deg=0.0):
        """radar at range R from the drone centre, in the drone's x-z plane.

        radar_el_deg: elevation of the radar as seen from the drone (0 = level
        with the rotor plane, the geometry TODAY.md and DEMO_TWO_RADAR insist
        on). radar_az_deg: azimuth around the drone. yaw_deg: drone heading.
        """
        self.work = Path(work_dir)
        self.work.mkdir(parents=True, exist_ok=True)
        register_patterns()
        self.mats = materials()
        self.R = float(radar_range_m)
        self.drone_h = float(radar_height_m if drone_height_m is None else drone_height_m)
        self.radar_h = float(radar_height_m)
        el, az = np.radians(radar_el_deg), np.radians(radar_az_deg)
        # radar position relative to the drone centre
        self.radar_pos = np.array([self.R * np.cos(el) * np.cos(az),
                                   self.R * np.cos(el) * np.sin(az),
                                   self.drone_h + self.R * np.sin(el)])
        self.drone_pos = np.array([0.0, 0.0, self.drone_h])
        self.yaw = np.radians(yaw_deg)

        self.scene = load_scene(None)
        self.scene.frequency = F0
        for m in self.mats.values():
            self.scene.add(m)

        objs = []
        if with_floor:
            fl = mesh.write_floor(self.work, size=25.0, z=0.0)
            objs.append(SceneObject(fname=str(fl), name="floor",
                                    radio_material=self.mats["concrete_24g"]))
        static = mesh.airframe_static_meshes()
        for name, (v, f) in static.items():
            v = self._place(v)
            p = mesh.write_ply(self.work / f"{name}.ply", v, f)
            mat = self.mats["drone_metal"] if (name == "body" or name.startswith("motor")) \
                else self.mats["nylon_gf"]
            objs.append(SceneObject(fname=str(p), name=name, radio_material=mat))
        self.scene.edit(add=objs)
        self._rotor_objs = {}
        self.set_rotor_angles(np.zeros(4))

        self.scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern=pattern,
                                          polarization="V")
        self.scene.rx_array = self.scene.tx_array
        tx = Transmitter("tx", position=self.radar_pos.tolist())
        rx = Receiver("rx", position=(self.radar_pos + [0, 0.0005, 0]).tolist())
        self.scene.add(tx)
        self.scene.add(rx)
        # boresight at the drone
        tx.look_at(self.drone_pos.tolist())
        rx.look_at(self.drone_pos.tolist())

    def _place(self, verts):
        """Body frame -> world: yaw about z, then translate to drone position."""
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return verts @ Rz.T + self.drone_pos

    def set_rotor_angles(self, phi):
        """Replace the four rotor meshes at the given angles (rad)."""
        phi = np.asarray(phi, dtype=float)
        remove = [o for o in self._rotor_objs.values()]
        add = []
        for i in range(4):
            v, f = mesh.rotor_mesh(phi[i], mesh.ROTOR_SPIN[i], mesh.rotor_hub(i))
            v = self._place(v)
            p = mesh.write_ply(self.work / f"rotor{i}.ply", v, f)
            add.append(SceneObject(fname=str(p), name=f"rotor{i}",
                                   radio_material=self.mats["nylon_gf"]))
        if remove:
            self.scene.edit(remove=remove)
        self.scene.edit(add=add)
        self._rotor_objs = {f"rotor{i}": add[i] for i in range(4)}
        return self

    def object_ids(self):
        """name -> object_id, for attributing paths to the part they hit."""
        return {name: int(obj.object_id) for name, obj in self.scene.objects.items()}
