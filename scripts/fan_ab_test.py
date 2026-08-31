"""Same-session fan ON/OFF/ON A/B — run this first when you wake up.

The 2026-08-30 study returned a solid null (docs/FAN_DETECTION.md) but every
fan-off reference came from the PREVIOUS DAY, at a different ADC operating point.
Fan-on and fan-off differ by a 4-code DC pedestal shift and 0.8-3 dB of broadband
level -- larger than any excess a plastic fan could plausibly add. That confound
caps every absolute-power test at about 1 dB and cannot be removed in software.

This script removes it: three captures back to back, same session, same gain, no
replug, no reconfiguration. ON / OFF / ON, so the two ON records bracket the OFF
record and any monotonic drift cancels. Expected to tighten the floors ~5-10x.

It also grabs a fan-off Config B capture -- the only mode with real blade-Doppler
coverage currently has NO reference at all, which is why the z=8.0 sideband at
+-51.8 Hz could not be attributed.

    python scripts/fan_ab_test.py                    # full protocol, ~7 min
    python scripts/fan_ab_test.py --secs 30          # quicker
    python scripts/fan_ab_test.py --orientation side # label the geometry

GEOMETRY MATTERS -- READ THIS BEFORE RUNNING
--------------------------------------------
If the fan points AT the radar, its blades sweep across the line of sight. Radial
velocity is then ~zero for every blade element and the Doppler is nulled by
geometry, not by sensitivity -- the radar cannot see it no matter how good the
receiver is. Turn the fan SIDE-ON so the blade tips move toward and away from the
module along the boresight. That is the difference between full tip Doppler and
none, and it may be the entire explanation for the null.

Also worth doing while you are there: move it to ~1 m (+16 dB by R^4) and tape
foil to one blade (+10-20 dB RCS, plus a guaranteed once-per-rev flash that makes
template validation unambiguous).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from klc6 import acquire as A          # noqa: E402
from klc6 import process as P          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out", "fan_ab")


def countdown(msg, secs):
    print(f"\n  *** {msg} ***", flush=True)
    for s in range(int(secs), 0, -1):
        if s <= 3 or s % 5 == 0:
            print(f"      {s}...", flush=True)
        time.sleep(1)


def cw_capture(dev, secs, fs, tag, stamp, meta):
    t0 = time.time()
    data, _ = A.record(dev, secs, fs=fs, channels=(A.CH_I,), range_v=5.0,
                       offset_v=A.OFFSET_V, filter_mode="average")
    p = os.path.join(OUT, f"{stamp}_{tag}_cw_{int(fs/1000)}k.npz")
    np.savez_compressed(p, data=data, fs=fs, channel_names=np.array(["I"]),
                        metadata=json.dumps({**meta, "state": tag, "mode": "cw"}))
    x = data[0]
    print(f"  {tag:8s}: {x.size:,} samples ({x.size/fs:.0f} s) in "
          f"{time.time()-t0:.0f} s | rms {x.std()*1e6:7.1f} uV | "
          f"mean {x.mean()*1e3:+7.3f} mV | {len(np.unique(x))} codes", flush=True)
    return p, x


def psd(x, fs, nfft):
    x = P.preprocess(x.astype(float), fs)
    nb = len(x) // nfft
    acc = np.zeros(nfft // 2 + 1)
    w = np.hanning(nfft)
    for i in range(nb):
        acc += np.abs(np.fft.rfft(x[i*nfft:(i+1)*nfft] * w)) ** 2
    return np.fft.rfftfreq(nfft, 1/fs), 10*np.log10(acc/max(nb,1) + 1e-30), nb


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--secs", type=float, default=60.0)
    ap.add_argument("--fs", type=float, default=50_000.0)
    ap.add_argument("--nfft", type=int, default=1 << 17)
    ap.add_argument("--lead", type=float, default=20.0,
                    help="seconds to switch the fan between captures")
    ap.add_argument("--orientation", default="unknown",
                    choices=("unknown", "head-on", "side"),
                    help="side-on is what recovers radial blade Doppler")
    ap.add_argument("--distance-m", type=float, default=2.5)
    ap.add_argument("--foiled", action="store_true")
    ap.add_argument("--skip-cfgb", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {"orientation": args.orientation, "distance_m": args.distance_m,
            "foiled": args.foiled, "protocol": "same-session ON/OFF/ON"}

    print("=" * 74)
    print(f"  SAME-SESSION FAN A/B  |  {args.secs:.0f} s per state @ "
          f"{args.fs/1000:.0f} kSa/s")
    print(f"  orientation={args.orientation}  distance={args.distance_m} m  "
          f"foiled={args.foiled}")
    print("=" * 74)
    if args.orientation == "head-on":
        print("  !! head-on geometry nulls radial blade Doppler. Turn the fan")
        print("     SIDE-ON before concluding anything from this run.")

    dev = A.open_device_cfg()
    print(f"  device {A.device_summary(dev)}")
    caps = {}
    try:
        A.set_dc(dev, 2.5)
        time.sleep(0.3)
        countdown("Fan ON, stand clear", args.lead)
        _, on1 = cw_capture(dev, args.secs, args.fs, "on1", stamp, meta)
        countdown("SWITCH THE FAN OFF -- leave it physically in place", args.lead)
        _, off = cw_capture(dev, args.secs, args.fs, "off", stamp, meta)
        countdown("SWITCH THE FAN BACK ON", args.lead)
        _, on2 = cw_capture(dev, args.secs, args.fs, "on2", stamp, meta)
        caps = {"on1": on1, "off": off, "on2": on2}

        if not args.skip_cfgb:
            print("\n  Config B reference sweep (fan still ON, then OFF)...")
            for state in ("on", "off"):
                if state == "off":
                    countdown("SWITCH THE FAN OFF for the Config B reference",
                              args.lead)
                A.configure_chirp(dev, 1000.0, 0.5, 4.5, shape="sawtooth")
                cpis = []
                for _ in range(150):
                    d, _ = A.record_chirps(dev, 128, 100, fs=100_000, sync=True)
                    cpis.append(d[0].astype(np.float32))
                p = os.path.join(OUT, f"{stamp}_cfgB_{state}.npz")
                np.savez_compressed(p, cpis=np.array(cpis), fs=100_000, spc=100,
                                    nchirps=128, ramp=1000.0,
                                    metadata=json.dumps({**meta, "state": state,
                                                         "mode": "fmcw_cfgB"}))
                print(f"    Config B {state}: {len(cpis)} CPIs -> "
                      f"{os.path.basename(p)}", flush=True)
    finally:
        try:
            dev.analog_output.channels[0].reset()
        except Exception:
            pass
        dev.close()

    # ---- verdict ----
    f, P1, n1 = psd(caps["on1"], args.fs, args.nfft)
    _, P0, n0 = psd(caps["off"], args.fs, args.nfft)
    _, P2, n2 = psd(caps["on2"], args.fs, args.nfft)
    band = (f >= 5) & (f <= 20_000)

    # The two ON captures bracket the OFF capture, so their difference measures
    # session drift PLUS measurement noise -- the honest floor for this test.
    floor = float(np.std((P1 - P2)[band]))
    sig = (P1 + P2) / 2 - P0
    s = sig[band]
    med = float(np.median(s))

    print("\n" + "=" * 74)
    print(f"  blocks {n1}/{n0}/{n2}   bin {f[1]:.3f} Hz")
    print(f"  same-state floor (on1 - on2):  {floor:.2f} dB")
    print(f"  ON minus OFF:  median {med:+.2f} dB, std {np.std(s):.2f} dB")
    excess = float(np.sqrt(max(np.var(s) - floor**2, 0.0)))
    print(f"  excess variance beyond the floor: {excess:.2f} dB")

    idx = np.flatnonzero(band)
    order = idx[np.argsort(-sig[idx])]
    seen, rows = [], []
    for i in order:
        if any(abs(f[i] - x) < 2.0 for x in seen):
            continue
        seen.append(float(f[i])); rows.append(int(i))
        if len(rows) >= 15:
            break
    print(f"\n  {'Hz':>10} {'m/s':>9} {'ON-OFF':>8} {'sigma':>7} "
          f"{'|on1-on2|':>10}  origin")
    n_mains = 0
    for i in rows:
        h = float(f[i]); k = round(h/60.0)
        is_m = k >= 1 and abs(h - k*60.0) < 1.0
        n_mains += is_m
        print(f"  {h:>10.2f} {h/P.HZ_PER_MPS:>9.3f} {sig[i]-med:>+8.2f} "
              f"{(sig[i]-med)/floor:>7.1f} {abs(P1[i]-P2[i]):>10.2f}  "
              f"{'MAINS x%d' % k if is_m else ''}")

    exp = (2*1.0/60.0) * len(rows)
    print(f"\n  {n_mains}/{len(rows)} strongest lines are mains "
          f"({exp:.1f} expected by chance)")
    best = max((sig[i]-med)/floor for i in rows)
    if n_mains > 3*max(exp, 0.5):
        print("  VERDICT: excess is on the 60 Hz grid -> fan MOTOR electrical")
        print("  noise, not blades. Same trap as the 2026-08-30 study.")
    elif best > 5.0:
        print(f"  VERDICT: a non-mains line clears 5 sigma. THIS IS NEW -- the")
        print(f"  2026-08-30 study bounded any line below 8.1 uV rms. Investigate.")
    else:
        print(f"  VERDICT: no line clears 5 sigma (best {best:.1f}). Null confirmed")
        print("  with the session confound removed. If orientation was head-on,")
        print("  re-run side-on before concluding the fan is simply too small.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
