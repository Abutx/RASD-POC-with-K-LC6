#!/usr/bin/env python3
"""Generate DJI Mini 3 micro-Doppler captures as the K-LC6 + AD2 would record them.

Writes real .npz + .json sidecars (source="synth") into out/sim/, so they load
through dataset.manifest / dataset.loader / scripts/analyze.py exactly like
bench captures. Also renders the figures that show what survives the front end.

  python scripts/sim_drone.py                 # captures + all figures
  python scripts/sim_drone.py --figures-only  # don't write captures
  python scripts/sim_drone.py --ranges 1 2 3 5 --seconds 10

The point of these files is to be a rehearsal: build the detector against them
before the bench session, then run the same detector on real captures and see
whether it still works. If it does not, the difference is information.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from klc6 import process                          # noqa: E402
from sim import capture, drone, klc6              # noqa: E402

OUT_CAP = "out/sim"
OUT_FIG = "out/demo"
SESSION = "sim_mini3_v1"


def make_capture(range_m, seconds, fs, rpm, el_deg, if_gain_db, seed,
                 label="drone_dji_mini3", aspect="hover", v_radial=None,
                 out_dir=OUT_CAP, write=True):
    t = np.arange(int(seconds * fs)) / fs
    # signature() is the voltage at the module pin; render() owns gain onwards.
    sig = drone.signature(t, range_m, rpm=rpm, el_deg=el_deg,
                          seed=seed, v_radial=v_radial)
    data = capture.render(sig, fs, if_gain_db=if_gain_db, seed=seed, iq=True)
    meta = {
        "label": label, "source": "synth", "mode": "cw", "session": SESSION,
        "target": "DJI Mini 3 (simulated)", "distance_m": float(range_m),
        "aspect": aspect, "if_gain_db": float(if_gain_db),
        "sample_rate": fs, "channels": ["I", "Q"],
        "sim_rpm": float(rpm), "sim_elevation_deg": float(el_deg),
        "sim_body_rcs_dbsm": drone.SIGMA_BODY_DBSM,
        "sim_blade_below_body_db": drone.BLADE_BELOW_BODY_DB,
        "sim_seed": int(seed),
        "notes": ("SIMULATED, not measured. Physics per sim/drone.py; front end "
                  "per sim/capture.py, calibrated to the measured empty-room "
                  "floor in " + klc6.MEASURED_FLOOR_REF),
    }
    if write:
        stem = (f"{out_dir}/sim_mini3_{aspect}_{range_m:g}m_"
                f"{int(rpm)}rpm_g{int(if_gain_db)}")
        path = capture.save(stem, data, fs, meta)
        print(f"    wrote {path}  ({data.shape[1]/fs:.0f} s, "
              f"I rms {data[0].std()*1e6:.1f} uV)")
    return data, sig, meta


def spec_db(x, fs, nperseg=8192):
    t, v, S = process.spectrogram_mps(x, fs, nperseg=nperseg, overlap=0.75)
    return t, v * klc6.HZ_PER_MPS, S          # convert back to Hz for blade work


def fig_range_ladder(fs, seconds, ranges, if_gain_db, out_dir):
    """The money shot: same drone, increasing range, until it vanishes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(ranges) + 1
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 4.6), constrained_layout=True,
                             sharey=True)
    t = np.arange(int(seconds * fs)) / fs

    # panel 0: the physics alone, no front end at all
    clean = drone.micro_doppler(t, rpm=drone.RPM_HOVER, el_deg=0.0, seed=7)
    tt, f, S = spec_db(clean, fs)
    ax = axes[0]
    ax.pcolormesh(tt, f / 1000, S, vmin=S.max() - 55, vmax=S.max(),
                  cmap="magma", shading="auto")
    ax.set_title("physics only\n(no receiver)", fontsize=10)
    ax.set_ylabel("Doppler (kHz)")
    ax.set_xlabel("time (s)")

    for k, r in enumerate(ranges):
        data, _, _ = make_capture(r, seconds, fs, drone.RPM_HOVER, 0.0,
                                  if_gain_db, seed=10 + k, write=False)
        z = data[0] + 1j * data[1]
        tt, f, S = spec_db(process.preprocess(z, fs, hp_hz=20.0), fs)
        ax = axes[k + 1]
        ax.pcolormesh(tt, f / 1000, S, vmin=S.max() - 55, vmax=S.max(),
                      cmap="magma", shading="auto")
        ax.set_title(f"{r:g} m\nthrough the AD2", fontsize=10)
        ax.set_xlabel("time (s)")

    tip = drone.rpm_to_tip_hz(drone.RPM_HOVER) / 1000
    for ax in axes:
        ax.set_ylim(-16, 16)
        for s in (+1, -1):
            ax.axhline(s * tip, color="cyan", lw=0.8, ls="--", alpha=0.7)
    axes[0].text(0.02, tip + 0.6, f"blade tip {tip*1000:.0f} Hz", color="cyan",
                 fontsize=8, transform=axes[0].get_yaxis_transform(which="grid"))
    fig.suptitle(f"DJI Mini 3 hovering, rotor plane level with the radar "
                 f"(IF gain {if_gain_db:.0f} dB)", fontsize=12)
    p = f"{out_dir}/mini3_range_ladder_g{int(if_gain_db)}.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"    {p}")


def fig_geometry(fs, seconds, out_dir):
    """Elevation dependence -- the single most important mounting decision."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    els = [0.0, 30.0, 60.0, 90.0]
    fig, axes = plt.subplots(1, len(els), figsize=(3.3 * len(els), 4.4),
                             constrained_layout=True, sharey=True)
    t = np.arange(int(seconds * fs)) / fs
    for ax, el in zip(axes, els):
        md = drone.micro_doppler(t, rpm=drone.RPM_HOVER, el_deg=el, seed=5)
        tt, f, S = spec_db(md, fs)
        ax.pcolormesh(tt, f / 1000, S, vmin=S.max() - 55, vmax=S.max(),
                      cmap="magma", shading="auto")
        ax.set_ylim(-16, 16)
        ax.set_title(f"radar {el:.0f}$^\\circ$ below\nrotor plane", fontsize=10)
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel("Doppler (kHz)")
    fig.suptitle("Why the modules must sit at prop height: blade Doppler is "
                 "$\\propto\\cos(\\mathrm{elevation})$", fontsize=12)
    p = f"{out_dir}/mini3_elevation.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"    {p}")


def fig_throttle_and_cadence(fs, seconds, out_dir):
    """Blade line and cadence peak move with RPM -- the classification handle."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    t = np.arange(int(seconds * fs)) / fs
    for j, (name, rpm) in enumerate([("hover", drone.RPM_HOVER),
                                     ("mid", 8000.0),
                                     ("max", drone.RPM_MAX)]):
        md = drone.micro_doppler(t, rpm=rpm, el_deg=0.0, seed=11)
        tt, f, S = spec_db(md, fs)
        ax = axes[0, j]
        ax.pcolormesh(tt, f / 1000, S, vmin=S.max() - 55, vmax=S.max(),
                      cmap="magma", shading="auto")
        tip = drone.rpm_to_tip_hz(rpm) / 1000
        ax.axhline(tip, color="cyan", lw=0.9, ls="--")
        ax.axhline(-tip, color="cyan", lw=0.9, ls="--")
        ax.set_ylim(-16, 16)
        ax.set_title(f"{name}: {rpm:.0f} rpm, tip {tip*1000:.0f} Hz")
        ax.set_xlabel("time (s)")
        if j == 0:
            ax.set_ylabel("Doppler (kHz)")

        cad, cvd = process.cadence_velocity_diagram(S, tt)
        ax = axes[1, j]
        ax.plot(cad, 20 * np.log10(cvd + 1e-12), lw=0.9)
        bpf = drone.herm_spacing_hz(rpm)
        for k in range(1, 4):
            ax.axvline(k * bpf, color="tab:red", ls=":", lw=1)
        ax.set_xlim(0, 900)
        ax.set_xlabel("cadence (Hz)")
        ax.set_title(f"blade-pass {bpf:.0f} Hz (red)", fontsize=10)
        if j == 0:
            ax.set_ylabel("CVD (dB)")
    fig.suptitle("Throttle moves both the blade line and the cadence peak -- "
                 "record throttle on every capture", fontsize=12)
    p = f"{out_dir}/mini3_throttle_cadence.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"    {p}")


def fig_amp_comparison(fs, seconds, out_dir, range_m=2.0):
    """The IF amp argument, in one picture."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gains = [0.0, 20.0, 30.0]
    fig, axes = plt.subplots(1, len(gains), figsize=(3.5 * len(gains), 4.4),
                             constrained_layout=True, sharey=True)
    for ax, g in zip(axes, gains):
        data, _, _ = make_capture(range_m, seconds, fs, drone.RPM_HOVER, 0.0,
                                  g, seed=21, write=False)
        z = data[0] + 1j * data[1]
        tt, f, S = spec_db(process.preprocess(z, fs, hp_hz=20.0), fs)
        ax.pcolormesh(tt, f / 1000, S, vmin=S.max() - 55, vmax=S.max(),
                      cmap="magma", shading="auto")
        ax.set_ylim(-16, 16)
        ax.set_title(f"IF gain {g:.0f} dB", fontsize=11)
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel("Doppler (kHz)")
    fig.suptitle(f"Same Mini 3 at {range_m:g} m. The only thing that changes is "
                 "gain ahead of the ADC.", fontsize=12)
    p = f"{out_dir}/mini3_ifgain_compare.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"    {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranges", type=float, nargs="+", default=[1.0, 2.0, 5.0])
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--fs", type=int, default=50_000)
    ap.add_argument("--if-gain-db", type=float, default=0.0)
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_FIG, exist_ok=True)
    if not args.figures_only:
        os.makedirs(OUT_CAP, exist_ok=True)
        print("captures:")
        for r in args.ranges:
            for g in (0.0, 30.0):
                make_capture(r, args.seconds, args.fs, drone.RPM_HOVER, 0.0, g,
                             seed=int(r * 10 + g))
        # the deliberate negatives, mirroring TODAY.md Blocks 3-4
        make_capture(2.0, args.seconds, args.fs, drone.RPM_HOVER, 90.0, 0.0,
                     seed=90, aspect="overhead-null")
        make_capture(2.0, args.seconds, args.fs, drone.RPM_MAX, 0.0, 0.0,
                     seed=91, aspect="throttle-high")
        make_capture(3.0, args.seconds, args.fs, drone.RPM_HOVER, 0.0, 0.0,
                     seed=92, aspect="approach", v_radial=-1.5)

    if not args.no_figures:
        print("figures:")
        fig_range_ladder(args.fs, args.seconds, args.ranges, args.if_gain_db, OUT_FIG)
        fig_range_ladder(args.fs, args.seconds, args.ranges, 30.0, OUT_FIG)
        fig_geometry(args.fs, args.seconds, OUT_FIG)
        fig_throttle_and_cadence(args.fs, args.seconds, OUT_FIG)
        fig_amp_comparison(args.fs, args.seconds, OUT_FIG)


if __name__ == "__main__":
    main()
