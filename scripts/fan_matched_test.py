"""Duration-matched fan-on vs empty-room test with a same-condition noise floor.

This is the cleanest statistical comparison available in this dataset, because
everything that usually confounds it is controlled:

  * SAME sample rate (50 kSa/s) and SAME duration (60 s) on both sides, so the
    number of averaging blocks is identical and the difference variance is not
    inflated by unequal integration.
  * TWO independent fan-on captures, which give a MEASURED same-condition noise
    floor: whatever fan_on_0 and fan_on_1 disagree about is measurement noise,
    not the fan. Significance is quoted against that, never against an assumed
    floor.
  * Every candidate line is checked against the 60 Hz grid, and the false-tag
    rate of that check is computed rather than hand-waved -- a fan is an
    electrical load as well as a moving target, and its motor raises supply hum
    in exactly the way a real blade line would look.

    python scripts/fan_matched_test.py
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from klc6 import process as P          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def psd(path, fs, nfft, key="data"):
    z = np.load(path)
    x = z[key][0].astype(float)
    x = P.preprocess(x, fs)
    nb = len(x) // nfft
    if nb < 1:
        raise SystemExit(f"  {path}: shorter than one {nfft}-point block")
    acc = np.zeros(nfft // 2 + 1)
    w = np.hanning(nfft)
    for i in range(nb):
        acc += np.abs(np.fft.rfft(x[i * nfft:(i + 1) * nfft] * w)) ** 2
    return np.fft.rfftfreq(nfft, 1 / fs), 10 * np.log10(acc / nb + 1e-30), nb


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fs", type=float, default=50_000.0)
    ap.add_argument("--nfft", type=int, default=1 << 17)
    ap.add_argument("--fmin", type=float, default=5.0)
    ap.add_argument("--fmax", type=float, default=20_000.0)
    ap.add_argument("--mains", type=float, default=60.0)
    ap.add_argument("--mains-tol", type=float, default=1.0)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--no-show", action="store_true", default=True)
    args = ap.parse_args()

    on0 = sorted(glob.glob(f"{ROOT}/out/fan/*fan_on_cw_50k_60s_0.npz"))
    on1 = sorted(glob.glob(f"{ROOT}/out/fan/*fan_on_cw_50k_60s_1.npz"))
    off = sorted(glob.glob(f"{ROOT}/out/baseline/*empty_baseline_60s.npz"))
    if not (on0 and on1 and off):
        raise SystemExit("  missing one of: fan_on_cw_50k_60s_0/_1, empty_baseline_60s")

    f, A, na = psd(on0[0], args.fs, args.nfft)
    _, B, nb = psd(on1[0], args.fs, args.nfft)
    _, O, no = psd(off[0], args.fs, args.nfft)

    print("=" * 74)
    print("  DURATION-MATCHED FAN TEST")
    print("=" * 74)
    print(f"  fan_on_0 {na} blocks | fan_on_1 {nb} blocks | empty {no} blocks"
          f"  ({f[1]:.3f} Hz bins)")

    band = (f >= args.fmin) & (f <= args.fmax)
    same = (A - B)[band]
    floor = float(np.std(same))
    sig = (A + B) / 2 - O
    s = sig[band]
    med = float(np.median(s))

    print(f"\n  same-condition floor (fan_on_0 - fan_on_1): std {floor:.2f} dB")
    print(f"  fan minus empty:                            std {np.std(s):.2f} dB, "
          f"median {med:+.2f} dB")
    excess = float(np.sqrt(max(np.var(s) - floor ** 2, 0.0)))
    print(f"  excess variance beyond the floor:           {excess:.2f} dB")
    if excess < 0.3:
        print("    -> the fan-vs-empty difference is no more structured than two")
        print("       fan-on captures differ from EACH OTHER. No broadband signature.")

    # How often does the mains test fire by chance? 2*tol/spacing of the grid.
    false_tag = 2 * args.mains_tol / args.mains
    nbins = int(band.sum())
    print(f"\n  mains test: within {args.mains_tol:.1f} Hz of a {args.mains:.0f} Hz "
          f"multiple -> {false_tag:.1%} of random frequencies tagged by chance")
    print(f"  searching {nbins:,} bins: a 5-sigma Gaussian outlier is expected "
          f"{nbins * 2.87e-7:.3f} times")

    idx = np.flatnonzero(band)
    order = idx[np.argsort(-sig[idx])]
    seen, rows = [], []
    for i in order:
        if any(abs(f[i] - x) < 2.0 for x in seen):
            continue
        seen.append(float(f[i])); rows.append(int(i))
        if len(rows) >= args.top:
            break

    print(f"\n  {'Hz':>10} {'m/s':>9} {'fan-empty':>10} {'sigma':>7} "
          f"{'|on0-on1|':>10}  origin")
    n_mains = 0
    for i in rows:
        h = float(f[i]); k = round(h / args.mains)
        is_m = k >= 1 and abs(h - k * args.mains) < args.mains_tol
        n_mains += is_m
        repro = abs(A[i] - B[i])
        print(f"  {h:>10.2f} {h/P.HZ_PER_MPS:>9.3f} {sig[i]-med:>+10.2f} "
              f"{(sig[i]-med)/floor:>7.1f} {repro:>10.2f}  "
              f"{'MAINS x%d' % k if is_m else ''}")

    exp = false_tag * len(rows)
    print(f"\n  {n_mains}/{len(rows)} of the strongest lines are mains harmonics; "
          f"{exp:.1f} expected by chance")
    if n_mains > 3 * max(exp, 0.5):
        print("  VERDICT: the fan-on excess is concentrated on the 60 Hz grid.")
        print("  That is the fan MOTOR raising supply hum, not its blades. A fan is")
        print("  an electrical load; switching it on increases mains pickup, which")
        print("  looks exactly like a detection in any off/on comparison.")
    else:
        print("  VERDICT: the excess is not explained by mains alone.")

    non_mains = [i for i in rows
                 if not (round(f[i] / args.mains) >= 1 and
                         abs(f[i] - round(f[i] / args.mains) * args.mains)
                         < args.mains_tol)]
    if non_mains:
        j = non_mains[0]
        print(f"\n  strongest NON-mains line: {f[j]:.1f} Hz = {f[j]/P.HZ_PER_MPS:.2f} m/s, "
              f"{(sig[j]-med)/floor:.1f} sigma, "
              f"reproducibility |on0-on1| = {abs(A[j]-B[j]):.2f} dB "
              f"(floor {floor:.2f})")
        print("  A real line reproduces between the two fan-on captures far better")
        print("  than the floor. Compare those two numbers before believing it.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    a1.semilogx(f[band], ((A + B) / 2)[band], lw=.7, color="#d62728", label="fan ON (2x60 s)")
    a1.semilogx(f[band], O[band], lw=.7, color="#1f77b4", label="empty (60 s)")
    a1.set_ylabel("dB"); a1.legend(); a1.grid(alpha=.25)
    a1.set_title("Duration-matched CW spectra, 50 kSa/s, identical block counts")
    a2.semilogx(f[band], sig[band] - med, lw=.7, color="#d62728", label="fan - empty")
    a2.axhline(3 * floor, color="g", ls=":", label=f"3 sigma ({3*floor:.1f} dB)")
    a2.axhline(0, color="k", lw=.6)
    for i in rows[:10]:
        if round(f[i] / args.mains) >= 1 and abs(f[i] - round(f[i]/args.mains)*args.mains) < args.mains_tol:
            a2.axvline(f[i], color="orange", alpha=.4, lw=.8)
    a2.set_xlabel("Hz"); a2.set_ylabel("dB"); a2.legend(); a2.grid(alpha=.25)
    a2.set_title("Difference; orange = 60 Hz harmonics")
    fig.tight_layout()
    out = f"{ROOT}/out/fan/matched_test.png"
    fig.savefig(out, dpi=150)
    print(f"\n  wrote {out}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
