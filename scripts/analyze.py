"""Analyse a capture: interference inventory, spectrogram, cadence diagram.

    python scripts/analyze.py out/20260829_061237_empty_baseline_60s.npz
    python scripts/analyze.py <file.npz> --repair    # trim a leading zero run
                                                     # in place and re-save

`--repair` exists because the AD2's first read leaves one device buffer of
unwritten zeros at the head of a record (8192 samples). Older captures taken
before that was trimmed at the source can be fixed without re-recording.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from klc6 import process as P          # noqa: E402


def load(path):
    d = np.load(path, allow_pickle=False)
    data = np.atleast_2d(d["data"])
    fs = float(d["fs"])
    names = [str(n) for n in d["channel_names"]]
    meta = json.loads(str(d["metadata"])) if "metadata" in d else {}
    ts = str(d["timestamp"]) if "timestamp" in d else ""
    return data, fs, names, meta, ts


def trim_leading_zeros(data, fs, verbose=True):
    allzero = np.all(data == 0.0, axis=0)
    if not allzero.size or not allzero[0]:
        return data, 0
    nz = np.flatnonzero(~allzero)
    lead = int(nz[0]) if nz.size else data.shape[1]
    if verbose:
        print(f"  trimmed {lead:,} leading zero samples "
              f"({lead/fs*1e3:.1f} ms, one device buffer)")
    return data[:, lead:], lead


def inventory(x, fs, top=20, lo=5.0, hi=None, mains=60.0):
    """Averaged spectrum, strongest discrete lines, mains harmonics tagged."""
    hi = hi if hi is not None else fs / 2 * 0.999
    nfft = 1 << 19
    while nfft > len(x) and nfft > 4096:
        nfft >>= 1
    nblk = max(1, len(x) // nfft)
    acc = np.zeros(nfft // 2 + 1)
    win = np.hanning(nfft)
    for i in range(nblk):
        acc += np.abs(np.fft.rfft(x[i*nfft:(i+1)*nfft] * win)) ** 2
    db = 10 * np.log10(acc / nblk + 1e-30)
    fr = np.fft.rfftfreq(nfft, 1 / fs)
    band = (fr >= lo) & (fr <= hi)
    med = float(np.median(db[band]))

    idxs = np.flatnonzero(band)
    order = idxs[np.argsort(-db[idxs])]
    seen, rows = [], []
    for i in order:
        if any(abs(fr[i] - s) < 5 for s in seen):
            continue
        seen.append(float(fr[i])); rows.append(int(i))
        if len(rows) >= top:
            break

    print(f"\n  --- interference inventory ({nblk} x {nfft/fs:.1f} s averages, "
          f"{fs/nfft:.3f} Hz bins, band {lo:.0f}-{hi:.0f} Hz) ---")
    print(f"  band median {med:.1f} dB")
    print(f"  {'Hz':>10} {'m/s':>9} {'dB over med':>12}  origin")
    n_mains = 0
    for i in rows:
        h = float(fr[i]); k = round(h / mains)
        is_m = k >= 1 and abs(h - k * mains) < 2.0
        n_mains += is_m
        print(f"  {h:>10.2f} {h/P.HZ_PER_MPS:>9.3f} {db[i]-med:>11.1f}  "
              f"{'MAINS %gx%d' % (mains, k) if is_m else ''}")
    print(f"\n  {n_mains}/{len(rows)} of the strongest lines are mains harmonics")
    return fr, db, med


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--repair", action="store_true",
                    help="trim leading zeros and overwrite the .npz")
    ap.add_argument("--vlim", type=float, default=6.0)
    ap.add_argument("--nperseg", type=int, default=8192)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    data, fs, names, meta, ts = load(args.path)
    print("=" * 70)
    print(f"  {os.path.basename(args.path)}")
    print(f"  class={meta.get('class','?')}  {data.shape[1]:,} samples/ch "
          f"@ {fs:,.0f} Sa/s = {data.shape[1]/fs:.1f} s  channels {names}")
    if ts:
        print(f"  recorded {ts}")
    print("=" * 70)

    data, lead = trim_leading_zeros(data, fs)
    for n, row in zip(names, data):
        print(f"    {n}: mean {row.mean()*1e3:+8.3f} mV   "
              f"pk-pk {(row.max()-row.min())*1e3:7.3f} mV   "
              f"rms {row.std()*1e6:7.1f} uV   {len(np.unique(row))} codes")

    if args.repair and lead:
        meta["repaired"] = f"trimmed {lead} leading zero samples"
        np.savez_compressed(args.path, data=data, fs=fs,
                            channel_names=np.array(names),
                            timestamp=ts, metadata=json.dumps(meta))
        print(f"  re-saved {args.path} ({data.shape[1]:,} samples/ch)")

    x = P.preprocess(P.to_complex(data, names), fs)
    inventory(x, fs, top=args.top)

    t, v, S_db = P.spectrogram_mps(x, fs, nperseg=args.nperseg)
    cad, cvd = P.cadence_velocity_diagram(S_db, t)

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 9),
                                   gridspec_kw={"height_ratios": [2, 1]})
    band = np.abs(v) <= args.vlim
    vmax = float(S_db[band].max())
    mesh = ax0.pcolormesh(t, v[band], S_db[band], vmin=vmax - 55, vmax=vmax,
                          shading="auto")
    ax0.axhline(0, color="w", lw=0.5, alpha=0.4)
    ax0.set_ylabel("radial velocity (m/s)")
    ax0.set_xlabel("time (s)")
    ax0.set_title(f"{os.path.basename(args.path)} -- {meta.get('class','?')} "
                  f"({P.HZ_PER_MPS:.1f} Hz per m/s)")
    fig.colorbar(mesh, ax=ax0, label="dB")

    if cad.size:
        keep = (cad > 0) & (cad <= 50)
        ax1.plot(cad[keep], 20*np.log10(cvd[keep] + 1e-30), lw=0.9)
        ax1.set_xlabel("cadence (Hz)")
        ax1.set_ylabel("dB")
        ax1.set_title("cadence velocity diagram -- a rotor shows a sharp peak, "
                      "an empty room none")
        ax1.grid(alpha=0.25)

    fig.tight_layout()
    out = os.path.splitext(args.path)[0] + ".png"
    fig.savefig(out, dpi=150)
    print(f"\n  wrote {out}")
    print("=" * 70)
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
