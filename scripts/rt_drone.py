#!/usr/bin/env python3
"""Physical-optics micro-Doppler of a DJI Mini 3 -- trace, synthesise, compare.

  python scripts/rt_drone.py                       # 3 m, level, floor on
  python scripts/rt_drone.py --range 5 --el 30     # radar 30 deg below the rotor plane
  python scripts/rt_drone.py --no-floor --n-phi 720

Writes out/sim/rt_mini3_*.npz + .json (source="sionna" per dataset/README, with
sim_method recorded) and out/demo/rt_*.png. The first thing it prints is the
PO self-test; if that fails nothing after it should be believed.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from klc6 import process                                   # noqa: E402
from sim import capture, drone, klc6                       # noqa: E402
from sim.rt import po, synth, trace                        # noqa: E402

OUT_FIG = "out/demo"
OUT_CAP = "out/sim"


def spec(x, fs, nperseg=4096, overlap=0.85):
    t, v, S = process.spectrogram_mps(x, fs, nperseg=nperseg, overlap=overlap)
    return t, v * klc6.HZ_PER_MPS, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--range", type=float, default=3.0)
    ap.add_argument("--az", type=float, default=0.0)
    ap.add_argument("--el", type=float, default=0.0,
                    help="radar elevation seen from the drone, deg; >0 above, <0 below")
    ap.add_argument("--height", type=float, default=1.2)
    ap.add_argument("--no-floor", action="store_true")
    ap.add_argument("--n-phi", type=int, default=1440)
    ap.add_argument("--rpm", type=float, default=drone.RPM_HOVER)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--fs", type=int, default=50_000)
    ap.add_argument("--if-gain-db", type=float, default=0.0)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--yaw-sweep", action="store_true",
                    help="also compute aspect-averaged static RCS over heading")
    ap.add_argument("--no-figs", action="store_true")
    a = ap.parse_args()

    print("=" * 72)
    print("PO self-test")
    print("=" * 72)
    ok = po.selftest()
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        sys.exit(1)

    print("\n" + "=" * 72)
    print(f"TRACE  range {a.range:g} m, az {a.az:g}, el {a.el:g} deg, "
          f"{'no floor' if a.no_floor else 'concrete floor'}")
    print("=" * 72)
    tr = trace.DroneTracer(range_m=a.range, az_deg=a.az, el_deg=a.el,
                           height_m=a.height, floor=not a.no_floor)
    t0 = time.time()
    tab = tr.trace(n_phi=a.n_phi, use_cache=not a.no_cache)
    print(f"  trace wall time {time.time()-t0:.1f} s")

    rcs = synth.effective_rcs(tab)
    print("\n" + "=" * 72)
    print("EFFECTIVE RCS FROM GEOMETRY (the number every table has been guessing)")
    print("=" * 72)
    print(f"  static airframe (body + arms + motors): {rcs['static_dbsm']:6.1f} dBsm")
    print(f"  four rotors, angle-averaged, summed:     {rcs['rotors_total_dbsm']:6.1f} dBsm")
    print(f"  per rotor: {['%.1f' % r for r in rcs['rotor_dbsm']]}")
    print(f"  blades below body:                       {rcs['blade_below_static_db']:6.1f} dB")
    print(f"\n  analytic model assumed: body {drone.SIGMA_BODY_DBSM:.0f} dBsm, blades "
          f"{drone.SIGMA_BODY_DBSM - drone.BLADE_BELOW_BODY_DB:.0f} dBsm "
          f"({drone.BLADE_BELOW_BODY_DB:.0f} dB below)")
    if a.yaw_sweep:
        yaws, s = tr.static_yaw_sweep(step_deg=5.0)
        s_db = 10 * np.log10(s + 1e-30)
        print(f"\n  static airframe vs heading (aspect average, 5 deg steps):")
        print(f"    mean {10*np.log10(s.mean()):6.1f} dBsm   median {np.median(s_db):6.1f}   "
              f"min {s_db.min():6.1f}   max {s_db.max():6.1f}")
        print(f"    literature: Phantom 4 Pro -12.4 dBsm (25 GHz, aspect-averaged); "
              f"we assumed -17 for a Mini 3")
        rcs["static_yaw_mean_dbsm"] = float(10 * np.log10(s.mean()))
        rcs["static_yaw_median_dbsm"] = float(np.median(s_db))
        rcs["blades_below_yawmean_db"] = float(10 * np.log10(s.mean()) - rcs["rotors_total_dbsm"])

    print("\n" + "=" * 72)
    print("SYNTHESISE")
    print("=" * 72)
    t = np.arange(int(a.seconds * a.fs)) / a.fs
    sig_po = synth.signature(t, tab, a.range, rpm=a.rpm, seed=3)
    sig_an = drone.signature(t, a.range, rpm=a.rpm, el_deg=a.el, seed=3)
    data = capture.render(sig_po, a.fs, if_gain_db=a.if_gain_db, seed=3, iq=True)
    tag = f"r{a.range:g}_el{a.el:g}_{'nofloor' if a.no_floor else 'floor'}_{int(a.rpm)}rpm"
    meta = {"label": "drone_dji_mini3", "source": "sionna", "mode": "cw",
            "session": "rt_mini3_v1", "target": "DJI Mini 3 (PO facet model)",
            "distance_m": a.range, "aspect": "hover", "if_gain_db": a.if_gain_db,
            "sample_rate": a.fs, "channels": ["I", "Q"],
            "sim_method": "physical-optics facets (sim/rt/po.py) + image-method floor; "
                          "Sionna RT used for calibration/validation only (rt_calib.py)",
            "sim_rpm": a.rpm, "sim_elevation_deg": a.el, "sim_floor": not a.no_floor,
            "sim_rcs": rcs, "notes": "SIMULATED, not measured."}
    os.makedirs(OUT_CAP, exist_ok=True)
    p = capture.save(f"{OUT_CAP}/rt_mini3_{tag}", data, a.fs, meta)
    print(f"  wrote {p}  I rms {data[0].std()*1e6:.1f} uV")

    # ---------------------------------------------------------- figures
    tip = drone.rpm_to_tip_hz(a.rpm)
    if a.no_figs:
        _energy_table(a, t, tab, tip)
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(OUT_FIG, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True, sharey=True)
    for ax, (name, x) in zip(axes, [("analytic line-scatterer (sim/drone.py)",
                                     sig_an / np.sqrt(np.mean(np.abs(sig_an) ** 2))),
                                    ("PO facets, twisted blades, floor",
                                     sig_po / np.sqrt(np.mean(np.abs(sig_po) ** 2))),
                                    ("PO through K-LC6 + AD2",
                                     data[0] + 1j * data[1])]):
        tt, f, S = spec(x if name.startswith("PO through") else x, a.fs)
        if name.startswith("PO through"):
            tt, f, S = spec(process.preprocess(x, a.fs, hp_hz=20.0), a.fs)
        ax.pcolormesh(tt, f / 1000, S, vmin=S.max() - 60, vmax=S.max(),
                      cmap="magma", shading="auto")
        ax.axhline(tip / 1e3, color="cyan", ls="--", lw=0.9)
        ax.axhline(-tip / 1e3, color="cyan", ls="--", lw=0.9)
        ax.set_ylim(-16, 16)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel("Doppler (kHz)")
    fig.suptitle(f"DJI Mini 3, {a.rpm:.0f} rpm, radar at {a.range:g} m, "
                 f"{a.el:g} deg elevation. Same kinematics, different scattering.",
                 fontsize=12)
    fp = f"{OUT_FIG}/rt_mini3_{tag}.png"
    fig.savefig(fp, dpi=130)
    plt.close(fig)
    print(f"  figure {fp}")

    # rotor lookup itself: RCS vs blade angle -- the flash structure
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    for i in range(4):
        ax.plot(np.degrees(tab["phi"]), 10 * np.log10(np.abs(tab["rotor"][i]) ** 2 + 1e-30),
                lw=0.8, label=f"rotor {i}")
    ax.set_xlabel("blade angle (deg, period 180)")
    ax.set_ylabel("RCS (dBsm)")
    ax.set_title("Per-rotor RCS vs blade angle -- PO facets. Spikes are specular flashes.")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fp2 = f"{OUT_FIG}/rt_rotor_rcs_vs_angle_{tag}.png"
    fig.savefig(fp2, dpi=130)
    plt.close(fig)
    print(f"  figure {fp2}")

    _energy_table(a, t, tab, tip)


def _energy_table(a, t, tab, tip):
    """Where does the rotor energy land, PO vs analytic -- and how much is unphysical."""
    def band_frac(x, lo, hi):
        w = np.hanning(len(x))
        sp = np.abs(np.fft.fftshift(np.fft.fft(x * w))) ** 2
        f = np.fft.fftshift(np.fft.fftfreq(len(x), 1 / a.fs))
        m = (np.abs(f) >= lo) & (np.abs(f) < hi)
        return sp[m].sum() / sp.sum()

    ang = synth.rotor_angles(t, a.rpm, seed=3)
    po_rot = sum(synth._interp_complex(tab["phi"], tab["rotor"][i], ang[i]) for i in range(4))
    an_rot = drone.micro_doppler(t, rpm=a.rpm, el_deg=a.el, seed=3)
    print("\n  blade energy distribution (rotors only):")
    print(f"  {'band':<18}{'analytic':>10}{'PO facets':>11}")
    for lo, hi in [(0, 1000), (1000, 4000), (4000, tip), (tip, tip * 1.15),
                   (tip * 1.15, a.fs / 2)]:
        print(f"  {lo/1e3:5.1f}-{hi/1e3:5.1f} kHz{100*band_frac(an_rot, lo, hi):9.1f}%"
              f"{100*band_frac(po_rot, lo, hi):10.1f}%")
    above = 100 * band_frac(po_rot, tip * 1.15, a.fs / 2)
    print(f"\n  energy above 1.15x tip Doppler is unphysical for a rotor: {above:.1f}% in PO.")
    print(f"  If that is more than a few percent the angle lookup is under-resolved or")
    print(f"  the shadow boundary is aliasing -- check with --n-phi 2880.")


if __name__ == "__main__":
    main()
