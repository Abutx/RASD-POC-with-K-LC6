"""Acceptance criterion 10.9.1 -- do consecutive chirps overlay?

Chirp-synchronous triggering is mandatory for range-Doppler. Without it every
acquisition starts at a random ramp phase, the slow-time phase progression
across chirps is meaningless, and the Doppler FFT integrates noise. The check
is visual and numerical: capture N consecutive chirps, overlay them, and
measure how alike they are.

It captures with sync ON and OFF so the difference is visible rather than
asserted -- if the two look the same, the trigger is not doing anything.

    python scripts/chirp_check.py                 # Config B, 1 kHz ramp
    python scripts/chirp_check.py --ramp 50 --spc 2000    # Config A
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


def similarity(chirps):
    """How alike are the chirps? Returns (mean pairwise corr, shape SNR dB).

    Shape SNR compares the energy of the average chirp against the energy of
    what is left after subtracting it: high means every chirp traces the same
    curve, ~0 dB means they are unrelated.
    """
    x = chirps - chirps.mean(axis=1, keepdims=True)
    mean = x.mean(axis=0)
    resid = x - mean
    p_mean = float(np.mean(mean ** 2))
    p_res = float(np.mean(resid ** 2))
    snr = 10 * np.log10(p_mean / p_res) if p_res > 0 else np.inf

    corrs = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            a, b = x[i], x[j]
            d = a.std() * b.std()
            if d > 0:
                corrs.append(float(np.mean(a * b) / d))
    return (float(np.mean(corrs)) if corrs else np.nan), snr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ramp", type=float, default=1000.0, help="ramp Hz")
    ap.add_argument("--spc", type=int, default=100, help="samples per chirp")
    ap.add_argument("--n", type=int, default=8, help="chirps to overlay")
    ap.add_argument("--fs", type=int, default=100_000)
    ap.add_argument("--v-low", type=float, default=0.5)
    ap.add_argument("--v-high", type=float, default=4.5)
    ap.add_argument("--repeats", type=int, default=6,
                    help="separate acquisitions for the start-phase test")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    S = A.chirp_slope(f_ramp=args.ramp)
    print("=" * 70)
    print(f"  CHIRP SYNC CHECK -- {args.ramp:.0f} Hz sawtooth, "
          f"{args.spc} samples/chirp, {args.n} chirps")
    print("=" * 70)
    print(f"  slope S = {S:.3e} Hz/s   range res {P.RANGE_RES_M:.2f} m")
    print(f"  chirp period {1/args.ramp*1e3:.2f} ms, "
          f"{args.fs/args.ramp:.0f} samples/period at {args.fs:,} Sa/s")
    if abs(args.fs / args.ramp - args.spc) > 1:
        print(f"  !! samples/chirp ({args.spc}) != fs/ramp "
              f"({args.fs/args.ramp:.0f}) -- reshape will drift across chirps")

    dev = A.open_device_cfg()
    print(f"  device {A.device_summary(dev)} "
          f"({dev.analog_input.buffer_size_max:,} sample buffer)")
    results = {}
    try:
        A.configure_chirp(dev, args.ramp, args.v_low, args.v_high,
                          shape="sawtooth")
        for sync in (True, False):
            data, _ = A.record_chirps(dev, args.n, args.spc, fs=args.fs,
                                      sync=sync)
            ch = data[0]
            corr, snr = similarity(ch)
            tag = "TRIGGERED on W1" if sync else "free-running"
            print(f"\n  {tag}: within one acquisition, mean pairwise "
                  f"correlation {corr:+.3f}, shape SNR {snr:+.1f} dB")

            # Within a single acquisition chirps are coherent whether or not we
            # trigger: fs/ramp is an exact integer and both clocks come off the
            # same AD2 oscillator. The trigger's real job is fixing the ramp
            # START phase, which only shows up ACROSS separate acquisitions --
            # measured, triggered holds the ramp peak at sample 96-98 every
            # time while free-running scatters it from 7 to 85.
            firsts, peaks = [], []
            for _ in range(args.repeats):
                d2, _ = A.record_chirps(dev, args.n, args.spc, fs=args.fs,
                                        sync=sync)
                c0 = d2[0][0] - d2[0][0].mean()
                firsts.append(c0)
                peaks.append(int(np.argmax(np.abs(c0))))
            firsts = np.array(firsts)
            cs = []
            for i in range(len(firsts)):
                for j in range(i + 1, len(firsts)):
                    a, b = firsts[i], firsts[j]
                    if a.std() * b.std() > 0:
                        cs.append(float(np.mean(a * b) / (a.std() * b.std())))
            xcorr = float(np.mean(cs)) if cs else float("nan")
            print(f"    across {args.repeats} separate acquisitions: "
                  f"chirp-0 correlation {xcorr:+.3f}, ramp peak at {peaks}")
            results["sync" if sync else "free"] = (ch, corr, snr, xcorr)
    finally:
        try:
            dev.analog_output.channels[0].reset()
        except Exception:
            pass
        dev.close()

    c_sync, corr_s, snr_s, x_s = results["sync"]
    _, corr_f, snr_f, x_f = results["free"]
    print("\n  --- verdict ---")
    print(f"  within one acquisition: triggered {corr_s:+.3f}, "
          f"free-running {corr_f:+.3f}   (expected to match)")
    print(f"  across acquisitions:    triggered {x_s:+.3f}, "
          f"free-running {x_f:+.3f}   (this is the actual trigger test)")
    if corr_s > 0.9 and x_s > 0.9 and x_s - x_f > 0.3:
        print("  PASS -- chirps overlay AND the ramp start phase is locked.")
    elif corr_s > 0.9:
        print("  PARTIAL -- chirps overlay within an acquisition but the start")
        print("  phase drifts between them. Doppler over one CPI is fine; "
              "anything comparing separate captures is not.")
    else:
        print("  FAIL -- chirps do not overlay. Range-Doppler will integrate noise.")
        print("  Check that W1 is on X1 pin 5 and the trigger source is ANALOG_OUT1.")

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    t_us = np.arange(args.spc) / args.fs * 1e6
    for ax, key, title in ((axes[0], "sync", "TRIGGERED on W1"),
                           (axes[1], "free", "free-running")):
        ch, corr, snr, _x = results[key]
        for i, row in enumerate(ch):
            ax.plot(t_us, (row - row.mean()) * 1e3, lw=0.9, alpha=0.8,
                    label=f"chirp {i}" if i < 3 else None)
        ax.set_title(f"{title}\ncorr {corr:+.3f}, shape SNR {snr:+.1f} dB")
        ax.set_xlabel("time within chirp (us)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("IF (mV, per-chirp DC removed)")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"K-LC6 chirp overlay -- {args.ramp:.0f} Hz sawtooth, "
                 f"{args.v_low:.1f}-{args.v_high:.1f} V on W1")
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"chirp_check_{args.ramp:.0f}Hz.png")
    fig.savefig(out, dpi=150)
    print(f"\n  wrote {out}")
    print("=" * 70)
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
