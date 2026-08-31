"""FMCW range profile (SPEC.md 10.8) -- Config A, range only.

    python scripts/range_demo.py                       # measure, save profile
    python scripts/range_demo.py --save empty          # store a background
    python scripts/range_demo.py --background empty    # subtract it

Config A is a 50 Hz sawtooth: 20 ms per chirp, 2000 samples at 100 kSa/s,
slope 1.5e10 Hz/s. PRF 50 Hz gives no usable Doppler, which is fine -- this is
the static range demo.

A static reflector competes with every other static return in the room: walls,
bench, and the module's own ramp feedthrough (~20 mVpp per SPEC 10.6) which
piles into the near bins. Two profiles differenced -- reflector present, then
removed or moved -- isolates it. That is what --background is for, and it is
also how acceptance 10.9.3 gets tested.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from klc6 import acquire as A          # noqa: E402
from klc6 import process as P          # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")


def measure(dev, ramp, spc, n, fs, v_low, v_high, avg=1, verbose=True):
    """Average power range profile over avg acquisitions of n chirps each.

    The buffer holds only 8 chirps of 2000 samples, and 8 chirps averaged left a
    2.36 dB run-to-run floor -- far too coarse to see a few-dB target. Averaging
    many acquisitions drops that floor as sqrt(N).
    """
    A.configure_chirp(dev, ramp, v_low, v_high, shape="sawtooth")
    S = A.chirp_slope(f_ramp=ramp)
    acc, r, last = None, None, None
    total = 0
    for k in range(int(avg)):
        data, _ = A.record_chirps(dev, n, spc, fs=fs, sync=True)
        chirps = data[0]
        last = chirps
        for c in chirps:
            r, mag = P.range_profile(c, fs, S)
            acc = mag ** 2 if acc is None else acc + mag ** 2
            total += 1
        if verbose and avg > 1 and (k + 1) % 10 == 0:
            print(f"    {k+1}/{avg} acquisitions ({total} chirps)", flush=True)
    return r, np.sqrt(acc / total), last


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ramp", type=float, default=50.0)
    ap.add_argument("--spc", type=int, default=2000)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--fs", type=int, default=100_000)
    ap.add_argument("--v-low", type=float, default=0.5)
    ap.add_argument("--v-high", type=float, default=4.5)
    ap.add_argument("--rmax", type=float, default=25.0, help="plot limit, m")
    ap.add_argument("--rmin", type=float, default=1.0,
                    help="ignore below this; feedthrough dominates the near bins")
    ap.add_argument("--save", default=None, help="store this profile under a name")
    ap.add_argument("--background", default=None, help="subtract a stored profile")
    ap.add_argument("--label", default="reflector")
    ap.add_argument("--avg", type=int, default=40,
                    help="acquisitions to average; floor falls as sqrt(N)")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    S = A.chirp_slope(f_ramp=args.ramp)
    print("=" * 70)
    print(f"  FMCW RANGE PROFILE -- {args.ramp:.0f} Hz sawtooth, "
          f"{args.spc} samples/chirp x {args.n}")
    print("=" * 70)
    print(f"  slope {S:.3e} Hz/s   resolution {P.RANGE_RES_M:.2f} m   "
          f"bin {P.beat_to_range(args.fs/args.spc, S):.3f} m")
    print(f"  unambiguous to {P.beat_to_range(args.fs/2, S):.0f} m")

    dev = A.open_device_cfg()
    print(f"  device {A.device_summary(dev)}")
    try:
        r, mag, chirps = measure(dev, args.ramp, args.spc, args.n, args.fs,
                                 args.v_low, args.v_high, avg=args.avg)
    finally:
        try:
            dev.analog_output.channels[0].reset()
        except Exception:
            pass
        dev.close()

    pk = float(np.abs(chirps).max())
    print(f"  IF pk-pk {(chirps.max()-chirps.min())*1e3:.2f} mV, "
          f"rms {chirps.std()*1e6:.1f} uV")

    os.makedirs(OUT, exist_ok=True)
    db = 20 * np.log10(mag + 1e-12)

    bg_db = None
    if args.background:
        bgp = os.path.join(OUT, f"rangeprof_{args.background}.npz")
        if not os.path.exists(bgp):
            raise SystemExit(f"  no stored background at {bgp}")
        b = np.load(bgp)
        if b["mag"].shape != mag.shape:
            raise SystemExit("  background has a different shape; re-record it")
        bg_db = 20 * np.log10(b["mag"] + 1e-12)
        print(f"  subtracting background '{args.background}'")

    if args.save:
        sp = os.path.join(OUT, f"rangeprof_{args.save}.npz")
        np.savez_compressed(sp, r=r, mag=mag, ramp=args.ramp, spc=args.spc,
                            fs=args.fs, slope=S, label=args.label)
        print(f"  wrote {sp}")

    band = (r >= args.rmin) & (r <= args.rmax)
    show = r <= args.rmax
    curve = db if bg_db is None else db - bg_db
    cb = curve[band]
    med = float(np.median(cb))

    order = np.argsort(-cb)
    seen, rows = [], []
    for i in order:
        rr = float(r[band][i])
        if any(abs(rr - s) < P.RANGE_RES_M for s in seen):
            continue
        seen.append(rr); rows.append(i)
        if len(rows) >= 6:
            break

    what = "change vs background" if bg_db is not None else "level"
    print(f"\n  --- strongest returns, {args.rmin:.1f}-{args.rmax:.0f} m "
          f"({what}) ---")
    print(f"  band median {med:.1f} dB")
    print(f"  {'range m':>9} {'dB over med':>12}")
    for i in rows:
        print(f"  {r[band][i]:>9.2f} {cb[i]-med:>11.1f}")

    top_r = float(r[band][rows[0]]); top_d = float(cb[rows[0]] - med)
    print(f"\n  peak at {top_r:.2f} m, {top_d:.1f} dB over the band median")
    if bg_db is None:
        print("  NOTE: no background subtracted -- this peak includes walls,")
        print("  bench and ramp feedthrough. Record one with --save empty after")
        print("  removing the reflector, then re-run with --background empty.")

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    if bg_db is not None:
        ax.plot(r[show], db[show], lw=0.8, color="#888888", label="with target")
        ax.plot(r[show], bg_db[show], lw=0.8, color="#bbbbbb", ls=":",
                label="background")
        ax2 = ax.twinx()
        ax2.plot(r[show], (db - bg_db)[show], lw=1.4, color="#d62728",
                 label="difference")
        ax2.set_ylabel("difference (dB)", color="#d62728")
        ax2.axhline(0, color="#d62728", lw=0.6, ls="--")
    else:
        ax.plot(r[show], db[show], lw=1.0, color="#1f77b4", label="range profile")
    ax.axvspan(0, args.rmin, color="k", alpha=0.12)
    ax.text(args.rmin / 2, ax.get_ylim()[1], " feedthrough", fontsize=8,
            va="top", ha="center", alpha=0.6)
    ax.axvline(top_r, color="g", ls="--", lw=1.0, label=f"peak {top_r:.2f} m")
    ax.set_xlabel("range (m)")
    ax.set_ylabel("dB")
    ax.set_title(f"K-LC6 FMCW range profile -- {args.label} -- "
                 f"{args.ramp:.0f} Hz ramp, {P.RANGE_RES_M:.2f} m resolution")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(OUT, f"range_{args.label}.png")
    fig.savefig(out, dpi=150)
    print(f"\n  wrote {out}")
    print("=" * 70)
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
