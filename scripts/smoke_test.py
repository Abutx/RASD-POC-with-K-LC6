"""Smoke test: prove the K-LC6 -> AD2 chain streams real data.

Captures, checks integrity, reports level and the dominant Doppler in the
motion band, and writes a two-panel PNG (waveform + spectrogram in m/s).

    python scripts/smoke_test.py                 # 10 s, channel I
    python scripts/smoke_test.py -s 5 --iq       # both channels, if Q is wired
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from klc6 import acquire as A          # noqa: E402
from klc6 import process as P          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "out")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-s", "--seconds", type=float, default=10.0)
    ap.add_argument("--fs", type=int, default=A.FS)
    ap.add_argument("--iq", action="store_true", help="use both I and Q")
    ap.add_argument("--range", type=float, default=5.0,
                    help="AD2 input range Vpp; hardware offers only 5 or 50")
    ap.add_argument("--offset", type=float, default=A.OFFSET_V)
    ap.add_argument("--filter", default="average", choices=("average", "decimate"))
    ap.add_argument("--vlim", type=float, default=6.0)
    ap.add_argument("-o", "--out", default="smoke_test.png")
    ap.add_argument("--npz", default=None)
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    channels = (A.CH_I, A.CH_Q) if args.iq else (A.CH_I,)
    names = A.channel_names(channels)

    print("=" * 70)
    print(f"  K-LC6 SMOKE TEST  |  {args.seconds:.0f} s @ {args.fs:,} Sa/s  "
          f"| channels {names}")
    print("=" * 70)

    dev = A.open_device()
    print(f"  device   {A.device_summary(dev)}")
    try:
        ain = A.configure(dev, fs=args.fs, channels=channels,
                          range_v=args.range, offset_v=args.offset,
                          filter_mode=args.filter)
        ch = ain.channels[channels[0]]
        lsb = ch.range / 2 ** 14
        print(f"  range    +-{ch.range/2:.3f} V ({ch.range:.3f} Vpp), "
              f"offset {ch.offset:+.3f} V, filter {args.filter}")
        print(f"  LSB      {lsb*1e6:.0f} uV")
        print(f"\n  >>> CAPTURING {args.seconds:.0f} s -- MOVE now <<<\n")

        t0 = time.time()
        data, rec = A.record(dev, args.seconds, fs=args.fs, channels=channels,
                             range_v=args.range, offset_v=args.offset,
                             filter_mode=args.filter)
        elapsed = time.time() - t0
    finally:
        dev.close()

    print(f"  captured {data.shape[1]:,} samples/ch in {elapsed:.1f} s, "
          f"lost {rec.lost_samples}, corrupted {rec.corrupted_samples}")
    for n, row in zip(names, data):
        print(f"    {n}: mean {row.mean()*1e3:+8.2f} mV   "
              f"pk-pk {(row.max()-row.min())*1e3:8.3f} mV   "
              f"rms {row.std()*1e6:8.1f} uV   "
              f"{len(np.unique(row))} distinct codes")

    x = P.preprocess(P.to_complex(data, names), args.fs)
    f_hz, v_mps, snr = P.band_peak(x, args.fs, 20.0, 1000.0)
    print(f"\n  dominant motion-band line: {f_hz:+.1f} Hz = {v_mps:+.3f} m/s, "
          f"{snr:.1f} dB over the band median")
    print(f"  {'MOTION DETECTED' if snr > 10 else 'no clear motion line (still room?)'}")

    t, v, S_db = P.spectrogram_mps(x, args.fs, nperseg=8192)
    band = np.abs(v) <= args.vlim
    print(f"  spectrogram {S_db.shape[0]} velocity bins x {S_db.shape[1]} frames, "
          f"{(v[1]-v[0]):.3f} m/s resolution")

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT, exist_ok=True)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 8),
                                   gridspec_kw={"height_ratios": [1, 2]})
    tt = np.arange(data.shape[1]) / args.fs
    step = max(1, data.shape[1] // 200_000)
    for n, row in zip(names, data):
        ax0.plot(tt[::step], row[::step] * 1e3, lw=0.5, label=f"{n} (raw)")
    ax0.set_ylabel("mV")
    ax0.set_xlabel("time (s)")
    ax0.set_title(f"K-LC6 IF, raw volts -- AD2 @ {args.fs/1e3:.0f} kSa/s")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.grid(alpha=0.25)

    vmax = float(S_db[band].max())
    mesh = ax1.pcolormesh(t, v[band], S_db[band], vmin=vmax - 55, vmax=vmax,
                          shading="auto")
    ax1.axhline(0, color="w", lw=0.5, alpha=0.4)
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("radial velocity (m/s)")
    ax1.set_title(f"Doppler spectrogram ({P.HZ_PER_MPS:.1f} Hz per m/s at "
                  f"{P.F_CARRIER/1e9:.3f} GHz)")
    fig.colorbar(mesh, ax=ax1, label="dB")
    fig.tight_layout()

    out = args.out if os.path.isabs(args.out) else os.path.join(OUT, args.out)
    fig.savefig(out, dpi=150)
    print(f"\n  wrote {out}")

    if args.npz:
        npz = args.npz if os.path.isabs(args.npz) else os.path.join(OUT, args.npz)
        np.savez_compressed(npz, data=data, fs=args.fs,
                            channel_names=np.array(names))
        print(f"  wrote {npz}")

    print("=" * 70)
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
