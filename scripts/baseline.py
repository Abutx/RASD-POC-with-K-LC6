"""Long empty-room baseline capture. This is the reference every later class
is judged against, so it is stored raw and its interference content is logged.

    python scripts/baseline.py -s 300
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

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-s", "--seconds", type=float, default=300.0)
    ap.add_argument("--fs", type=int, default=A.FS)
    ap.add_argument("--iq", action="store_true")
    ap.add_argument("--range", type=float, default=5.0)
    ap.add_argument("--offset", type=float, default=A.OFFSET_V)
    ap.add_argument("--filter", default="average")
    ap.add_argument("--notes", default="empty room, stable, nobody present")
    args = ap.parse_args()

    channels = (A.CH_I, A.CH_Q) if args.iq else (A.CH_I,)
    names = A.channel_names(channels)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUT, f"{stamp}_empty_baseline_{int(args.seconds)}s.npz")
    os.makedirs(OUT, exist_ok=True)

    print("=" * 70)
    print(f"  EMPTY-ROOM BASELINE  |  {args.seconds:.0f} s @ {args.fs:,} Sa/s "
          f"| channels {names}")
    print(f"  -> {path}")
    print("=" * 70, flush=True)

    dev = A.open_device()
    print(f"  device {A.device_summary(dev)}", flush=True)

    # Pre-flight: 2 s look before committing to a long run. An abruptly killed
    # process can leave the AD2 holding a frozen channel -- a 300 s capture came
    # back at 10 uV rms and 4 codes, every block exactly constant, and that only
    # showed up after five minutes of holding the room still.
    probe, _ = A.record(dev, 2.0, fs=args.fs, channels=channels,
                        range_v=args.range, offset_v=args.offset,
                        filter_mode=args.filter)
    for n, row in zip(names, probe):
        print(f"  preflight {n}: mean {row.mean()*1e3:+.3f} mV  "
              f"rms {row.std()*1e6:.1f} uV  codes {len(np.unique(row))}", flush=True)
    if probe[0].std() * 1e6 < 30.0:
        dev.close()
        raise SystemExit(
            f"  ABORT: channel {names[0]} is flat ({probe[0].std()*1e6:.1f} uV rms, "
            f"expected ~170).\n"
            f"  The AD2 is holding a frozen channel, or the K-LC6 lost power.\n"
            f"  Unplug/replug the AD2 USB and re-run.")

    # NO progress callback. Passing one to recorder.record() sends dwfpy down a
    # path that never accumulates into the channel buffer: measured side by side
    # at 30 s, no-callback returned 3,000,000 real samples at 176 uV rms, while
    # the callback version returned 25 real samples preceded by 2,999,975 zeros
    # -- and the callback fired exactly once, at the end. Both reported 0 lost
    # and 0 corrupted. There is no safe way to watch a capture in progress here,
    # so the run is silent and verified afterwards instead.
    t0 = time.time()
    print(f"  recording {args.seconds:.0f} s (silent -- no progress callback, "
          f"see comment)...", flush=True)

    try:
        data, rec = A.record(dev, args.seconds, fs=args.fs, channels=channels,
                             range_v=args.range, offset_v=args.offset,
                             filter_mode=args.filter)
    finally:
        dev.close()
    elapsed = time.time() - t0

    print(f"\n  captured {data.shape[1]:,} samples/ch "
          f"({data.shape[1]/args.fs:.1f} s of data) in {elapsed:.1f} s wall",
          flush=True)
    print(f"  lost {rec.lost_samples}, corrupted {rec.corrupted_samples}")

    meta = {
        "class": "empty", "subclass": "", "distance_m": None, "aspect": "static",
        "sample_rate": args.fs, "channels": names, "mode": "cw", "vco_v": None,
        "if_gain_db": 0, "range_v": args.range, "offset_v": args.offset,
        "filter": args.filter, "environment": "indoor", "notes": args.notes,
    }
    np.savez_compressed(path, data=data, fs=args.fs,
                        channel_names=np.array(names),
                        timestamp=_dt.datetime.now().astimezone().isoformat(),
                        metadata=json.dumps(meta))
    size_mb = os.path.getsize(path) / 1e6
    print(f"  wrote {path}  ({size_mb:.1f} MB)")

    # --- what lives in an empty room? this is the false-alarm inventory ---
    for n, row in zip(names, data):
        print(f"    {n}: mean {row.mean()*1e3:+8.3f} mV   "
              f"pk-pk {(row.max()-row.min())*1e3:7.3f} mV   "
              f"rms {row.std()*1e6:7.1f} uV   {len(np.unique(row))} codes")

    x = P.preprocess(P.to_complex(data, names), args.fs)
    nfft = 1 << 20
    nblk = len(x) // nfft
    acc = np.zeros(nfft // 2 + 1)
    for i in range(nblk):
        acc += np.abs(np.fft.rfft(x[i*nfft:(i+1)*nfft] * np.hanning(nfft))) ** 2
    if nblk == 0 or not np.any(acc > 0):
        print("\n  spectrum is identically zero -- the channel was flat. "
              "No inventory to report.")
        return 1
    db = 10 * np.log10(acc / max(nblk, 1) + 1e-30)
    fr = np.fft.rfftfreq(nfft, 1 / args.fs)
    band = (fr >= 5) & (fr <= 20_000)
    med = float(np.median(db[band]))
    print(f"\n  --- empty-room interference inventory "
          f"({nblk} x {nfft/args.fs:.1f} s averages, {args.fs/nfft:.3f} Hz bins) ---")
    print(f"  band median {med:.1f} dB")
    idxs = np.flatnonzero(band)
    order = idxs[np.argsort(-db[idxs])]
    seen, rows = [], []
    for i in order:
        if any(abs(fr[i] - s) < 5 for s in seen):
            continue
        seen.append(fr[i]); rows.append(i)
        if len(rows) >= 20:
            break
    print(f"  {'Hz':>10} {'m/s':>9} {'dB over med':>12}  note")
    for i in rows:
        h = float(fr[i]); k = round(h / 60.0)
        tag = f"MAINS 60x{k}" if k >= 1 and abs(h - k*60) < 2.0 else ""
        print(f"  {h:>10.2f} {h/P.HZ_PER_MPS:>9.3f} {db[i]-med:>11.1f}  {tag}")
    print("=" * 70, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
