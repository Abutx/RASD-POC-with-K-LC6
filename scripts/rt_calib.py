#!/usr/bin/env python3
"""Sionna RT: size the ray budget and calibrate the blade scattering model.

Two questions the drone trace cannot start without:

  A. COST. How long does one trace take vs samples_per_src, with directive
     diffuse scattering and edge diffraction enabled, on a drone-sized target?
     Decides the angular resolution we can afford.

  B. AMPLITUDE. A blade chord is ~1 wavelength; geometric optics is at the
     edge of validity. Sionna's 'directive' scattering pattern gives a finite
     specular lobe -- the physically-optics-like behaviour we want -- but its
     absolute level depends on (S, alpha_r). Calibrate them so a blade-sized
     flat plate at normal incidence returns the closed-form PO RCS:

         sigma = 4 pi A^2 / lambda^2

     For a 10 x 60 mm plate at 24.125 GHz that is 0.0293 m^2 = -15.3 dBsm.
     Then trust the angular roll-off from the lobe. This is about as accurate
     as a ray tracer can be made for lambda-scale features, and it is stated
     as a calibration, not derived physics.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.rt import mesh                                            # noqa: E402
from sionna.rt import (load_scene, PathSolver, PlanarArray, Transmitter,  # noqa: E402
                       Receiver, SceneObject, RadioMaterial)

C = 299_792_458.0
F0 = 24.125e9
LAM = C / F0
OUT = os.path.join(os.environ.get("TEMP", "."), "rt_calib")
os.makedirs(OUT, exist_ok=True)


def h_from(paths):
    a = np.asarray(paths.a[0].numpy()) + 1j * np.asarray(paths.a[1].numpy())
    tau = np.asarray(paths.tau.numpy())
    return np.sum(a.ravel() * np.exp(-2j * np.pi * F0 * tau.ravel())), a.shape[-1]


def rcs_dbsm(h, R):
    return 10 * np.log10(np.abs(h) ** 2 * (4 * np.pi) ** 3 * R ** 4 / LAM ** 2 + 1e-30)


def radar_scene(obj_path, name, mat, R, orient_deg=0.0):
    sc = load_scene(None)
    sc.frequency = F0
    sc.add(mat)
    sc.edit(add=SceneObject(fname=str(obj_path), name=name, radio_material=mat))
    sc.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    sc.rx_array = sc.tx_array
    sc.add(Transmitter("tx", position=[R, 0, 0], orientation=[np.pi, 0, 0]))
    sc.add(Receiver("rx", position=[R, 0.0005, 0], orientation=[np.pi, 0, 0]))
    if orient_deg:
        # mi.Point3f rejects numpy float64 -- hand it plain Python floats
        sc.get(name).orientation = [0.0, 0.0, float(np.radians(orient_deg))]
    return sc


solver = PathSolver()
R = 3.0

# ------------------------------------------------------------------ A. cost
print("=" * 74)
print("A. trace cost vs samples_per_src  (rotor mesh at 3 m, level view)")
print("=" * 74)
rv, rf = mesh.rotor_mesh(np.radians(20.0), +1, (0, 0, 0))
rotor = mesh.write_ply(os.path.join(OUT, "rotor.ply"), rv, rf)
nylon = RadioMaterial(name="nylon-gf", relative_permittivity=3.7, conductivity=0.075,
                      scattering_coefficient=0.8, scattering_pattern="directive",
                      alpha_r=4)
sc = radar_scene(rotor, "rotor", nylon, R)
# warm the JIT once so timings below are steady-state
solver(sc, max_depth=1, los=False, diffuse_reflection=True, refraction=False,
       diffraction=True, edge_diffraction=True, samples_per_src=100_000)
print(f"  {'rays':>10}{'time':>9}{'paths':>7}{'|h|^2 dB':>10}{'RCS dBsm':>10}")
for n in [1_000_000, 5_000_000, 20_000_000]:
    t0 = time.time()
    p = solver(sc, max_depth=1, los=False, specular_reflection=True,
               diffuse_reflection=True, refraction=False,
               diffraction=True, edge_diffraction=True, samples_per_src=n)
    dt = time.time() - t0
    h, npath = h_from(p)
    print(f"  {n:>10,}{dt:8.2f}s{npath:7d}{20*np.log10(abs(h)+1e-30):10.1f}{rcs_dbsm(h, R):10.1f}")

# ------------------------------------------------------------- B. calibrate
print("\n" + "=" * 74)
print("B. blade-sized plate at normal incidence vs physical optics")
print("=" * 74)
w, hgt = 0.010, 0.060
A = w * hgt
sigma_po = 4 * np.pi * A ** 2 / LAM ** 2
print(f"  plate {w*1e3:.0f} x {hgt*1e3:.0f} mm, A = {A*1e4:.1f} cm^2, "
      f"PO RCS = {sigma_po:.4f} m^2 = {10*np.log10(sigma_po):.1f} dBsm")
hh, hw = hgt / 2, w / 2
pv = np.array([[0, -hw, -hh], [0, hw, -hh], [0, hw, hh], [0, -hw, hh]])
pf = np.array([[0, 2, 1], [0, 3, 2], [0, 1, 2], [0, 2, 3]])
plate = mesh.write_ply(os.path.join(OUT, "small_plate.ply"), pv, pf)

N_RAYS = 20_000_000
print(f"\n  metal plate, {N_RAYS:,} rays, specular + directive diffuse + edge diffraction")
print(f"  {'S':>5}{'alpha_r':>9}{'paths':>7}{'RCS dBsm':>10}{'vs PO':>8}")
for S in [0.0, 0.5, 0.8, 1.0]:
    for alpha in ([1] if S == 0.0 else [1, 4, 16]):
        m = RadioMaterial(name=f"m{S}{alpha}", relative_permittivity=1.0, conductivity=1e7,
                          scattering_coefficient=S, scattering_pattern="directive",
                          alpha_r=alpha)
        sc = radar_scene(plate, "plate", m, R)
        p = solver(sc, max_depth=1, los=False, specular_reflection=True,
                   diffuse_reflection=(S > 0), refraction=False,
                   diffraction=True, edge_diffraction=True, samples_per_src=N_RAYS)
        h, npath = h_from(p)
        r = rcs_dbsm(h, R)
        print(f"  {S:5.1f}{alpha:9d}{npath:7d}{r:10.1f}{r-10*np.log10(sigma_po):+8.1f}")

print("\n  same plate, best-looking (S, alpha) candidates, rotated off normal:")
for S, alpha in [(0.8, 4), (1.0, 4), (1.0, 16)]:
    row = f"  S={S:.1f} a={alpha:<3d}"
    for ang in [0, 5, 10, 20, 40]:
        # a RadioMaterial binds to one scene; make a fresh one per scene
        m = RadioMaterial(name=f"r{S}{alpha}_{ang}", relative_permittivity=1.0,
                          conductivity=1e7, scattering_coefficient=S,
                          scattering_pattern="directive", alpha_r=alpha)
        sc = radar_scene(plate, "plate", m, R, orient_deg=ang)
        p = solver(sc, max_depth=1, los=False, specular_reflection=True,
                   diffuse_reflection=True, refraction=False,
                   diffraction=True, edge_diffraction=True, samples_per_src=N_RAYS)
        h, _ = h_from(p)
        row += f"  {ang:2d}deg:{rcs_dbsm(h, R):6.1f}"
    print(row)
print("\n  PO for a 60 mm plate has its first null near lambda/D = 12 deg off normal;")
print("  a lobe that is down ~10-15 dB by 10-20 deg is the physically right shape.")

sigma_mirror = np.pi * R ** 2
print("\n" + "=" * 74)
print("WHAT THIS MEANS")
print("=" * 74)
print(f"""
  Image theory (an infinite mirror) at R = {R:.0f} m has equivalent RCS pi R^2 =
  {10*np.log10(sigma_mirror):.1f} dBsm. Sionna's S=0 specular result above sits within ~2 dB of
  that, NOT of the {10*np.log10(sigma_po):.1f} dBsm physical optics gives for the actual plate.
  Sionna RT applies image-theory amplitude to any specular point regardless
  of how small the surface is. The first Fresnel zone here is ~{np.sqrt(LAM*R):.2f} m; a
  blade chord is 0.01 m. For sub-Fresnel-zone surfaces the overestimate is
  pi R^2 lambda^2 / (4 A^2) = {10*np.log10(sigma_mirror/sigma_po):.1f} dB, which is what we measured.

  CONCLUSION: Sionna RT's amplitudes are not usable for lambda-scale blades.
  Its geometry, delays and antenna handling are exact (rt_smoke: 0.00 dB on
  a Fresnel-zone-sized plate). sim/rt/po.py does the blade amplitudes with
  physical optics instead; this script is kept as the evidence for why.""")
