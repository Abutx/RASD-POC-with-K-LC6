#!/usr/bin/env python3
"""Sionna RT smoke test: does the tracer run, load our PLY, and get physics right?

Calibration target is a 2 m square metal plate at 5 m, normal to the radar.
That is far larger than the first Fresnel zone (~0.25 m), so image theory is
exact and the two-way path gain is

    Pr/Pt = lambda^2 / (64 pi^2 R^2)   = -80.1 dB at 5 m, 24.125 GHz

with isotropic antennas and no other paths. If Sionna returns that to within
a dB, the a/tau convention, the PLY loader and the LLVM backend are all fine.
Also dumps the installed antenna-pattern source so the K-LC6 pattern can be
written in the package's own idiom instead of from the docs.
"""
import inspect
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.rt import mesh                                   # noqa: E402

import sionna.rt as rt                                    # noqa: E402
from sionna.rt import (load_scene, PathSolver, PlanarArray, Transmitter,   # noqa: E402
                       Receiver, SceneObject, RadioMaterial)

C = 299_792_458.0
F0 = 24.125e9
LAM = C / F0
OUT = os.path.join(os.environ.get("TEMP", "."), "rt_smoke")
os.makedirs(OUT, exist_ok=True)

print("=" * 72)
print("[0] introspection -- how the package defines a pattern")
print("=" * 72)
import sionna.rt.antenna_pattern as ap                    # noqa: E402
for name in ("v_tr38901_pattern", "v_iso_pattern"):
    fn = getattr(ap, name, None)
    if fn is not None:
        print(f"\n--- {name} ---")
        print(inspect.getsource(fn))
print("\n--- PolarizedAntennaPattern.__init__ ---")
print(inspect.signature(rt.PolarizedAntennaPattern.__init__))
print("\n--- register_antenna_pattern ---")
print(inspect.signature(rt.register_antenna_pattern))
print("\n--- antenna_pattern registry keys ---")
try:
    print(list(ap._antenna_pattern_factories.keys()) if hasattr(ap, "_antenna_pattern_factories")
          else [k for k in dir(ap) if "pattern" in k.lower()][:20])
except Exception as e:
    print("  (registry not exposed)", e)

print("\n" + "=" * 72)
print("[1] plate calibration")
print("=" * 72)
R = 5.0
side = 2.0
plate = mesh.write_calibration_plate(OUT, side=side)
print(f"wrote {plate}")

scene = load_scene(None)                     # empty
scene.frequency = F0
metal = RadioMaterial(name="pec-ish", relative_permittivity=1.0, conductivity=1e7)
scene.add(metal)
obj = SceneObject(fname=str(plate), name="plate", radio_material=metal)
scene.edit(add=obj)
# plate normal is along x at the origin; put the radar on +x at range R
scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
scene.rx_array = scene.tx_array
scene.add(Transmitter("tx", position=[R, 0.0, 0.0], orientation=[np.pi, 0, 0]))
scene.add(Receiver("rx", position=[R, 0.0005, 0.0], orientation=[np.pi, 0, 0]))
print(f"objects: {list(scene.objects.keys())}")

solver = PathSolver()
t0 = time.time()
paths = solver(scene, max_depth=1, los=False, specular_reflection=True,
               diffuse_reflection=False, refraction=False, diffraction=False,
               samples_per_src=200_000)
dt = time.time() - t0
a_re, a_im = paths.a
a = np.asarray(a_re.numpy()) + 1j * np.asarray(a_im.numpy())
tau = np.asarray(paths.tau.numpy())
print(f"trace time {dt*1e3:.0f} ms;  a shape {a.shape};  n_paths {a.shape[-1]}")
print(f"tau (ns): {np.round(tau.ravel()*1e9, 3)}   expected 2R/c = {2*R/C*1e9:.3f} ns")
h = np.sum(a.ravel() * np.exp(-2j * np.pi * F0 * tau.ravel()))
g_db = 20 * np.log10(np.abs(h) + 1e-30)
g_theory = 10 * np.log10(LAM ** 2 / (64 * np.pi ** 2 * R ** 2))
print(f"|sum a e^-j2pi f tau|^2 = {g_db:6.2f} dB")
print(f"image theory            = {g_theory:6.2f} dB")
print(f"difference              = {g_db - g_theory:+6.2f} dB   "
      f"{'OK' if abs(g_db - g_theory) < 1.0 else 'INVESTIGATE'}")
print(f"interactions: {np.asarray(paths.interactions.numpy()).ravel()}")

print("\n" + "=" * 72)
print("[2] a small twisted blade -- does the tracer see it at all?")
print("=" * 72)
bv, bf = mesh.rotor_mesh(0.0, +1, (0.0, 0.0, 0.0))
rotor = mesh.write_ply(os.path.join(OUT, "rotor.ply"), bv, bf)
scene2 = load_scene(None)
scene2.frequency = F0
plastic = RadioMaterial(name="nylon-gf", relative_permittivity=3.7, conductivity=0.075,
                        scattering_coefficient=0.3)
scene2.add(plastic)
scene2.edit(add=SceneObject(fname=str(rotor), name="rotor", radio_material=plastic))
scene2.tx_array = scene.tx_array
scene2.rx_array = scene.tx_array
# radar level with the rotor plane, looking along -x from 3 m
scene2.add(Transmitter("tx", position=[3.0, 0.0, 0.0], orientation=[np.pi, 0, 0]))
scene2.add(Receiver("rx", position=[3.0, 0.0005, 0.0], orientation=[np.pi, 0, 0]))
t0 = time.time()
p2 = solver(scene2, max_depth=1, los=False, specular_reflection=True,
            diffuse_reflection=True, refraction=False, diffraction=False,
            samples_per_src=400_000)
dt = time.time() - t0
a2 = np.asarray(p2.a[0].numpy()) + 1j * np.asarray(p2.a[1].numpy())
tau2 = np.asarray(p2.tau.numpy())
h2 = np.sum(a2.ravel() * np.exp(-2j * np.pi * F0 * tau2.ravel()))
print(f"trace time {dt*1e3:.0f} ms;  n_paths {a2.shape[-1]};  "
      f"|h|^2 = {20*np.log10(np.abs(h2)+1e-30):.1f} dB")
kinds = np.asarray(p2.interactions.numpy()).ravel()
print(f"interaction types present: {sorted(set(kinds.tolist()))}  (1=specular, 2=diffuse)")
sigma_eff = np.abs(h2) ** 2 * (4 * np.pi) ** 3 * 3.0 ** 4 / LAM ** 2
print(f"implied RCS of the rotor at this aspect: {10*np.log10(sigma_eff+1e-30):.1f} dBsm")
print("\nsmoke test complete")
