"""Collect a multi-modal fan dataset in one pass. Hardware-bound, so this runs
in the main session; every later analysis works from the saved .npz files.

Captures, in order:
  1. CW micro-Doppler   -- W1 held at DC (no sweep), 100 kSa/s, 2 x 25 s.
                           This is the blade-flash / cadence modality.
  2. FMCW Config B      -- 1 kHz sawtooth, 128 chirps x 100 samples per CPI,
                           many CPIs saved raw -> range-Doppler over time.
  3. FMCW Config A      -- 50 Hz sawtooth, 8 x 2000 samples, averaged blocks
                           -> fine range profile.

Record mode at 100 kSa/s only sustains ~30 s on this host (60 s corrupted,
300 s lost 147k samples), so the CW captures are chunked at 25 s.
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
import dwfpy as dwf                    # noqa: E402
from klc6 import acquire as A          # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "out", "fan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="fan_on")
    ap.add_argument("--cw-chunks", type=int, default=2)
    ap.add_argument("--cw-secs", type=float, default=25.0)
    ap.add_argument("--cpis", type=int, default=200)
    ap.add_argument("--cfga-blocks", type=int, default=30)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {"label": args.label, "timestamp": stamp, "notes": args.notes,
            "carrier_hz": A.F_CARRIER, "hz_per_mps": A.HZ_PER_MPS}

    print("=" * 70)
    print(f"  FAN DATASET COLLECTION -- {args.label}")
    print("=" * 70, flush=True)

    # ---------- 1. CW micro-Doppler ----------
    # VCO pin parked at DC so the carrier is fixed: pure Doppler, no sweep.
    dev = A.open_device_cfg()
    try:
        A.set_dc(dev, 2.5)
        time.sleep(0.3)
        for k in range(args.cw_chunks):
            t0 = time.time()
            data, rec = A.record(dev, args.cw_secs, fs=100_000,
                                 channels=(A.CH_I,), range_v=5.0,
                                 offset_v=A.OFFSET_V, filter_mode="average")
            p = os.path.join(OUT, f"{stamp}_{args.label}_cw{k}_100k.npz")
            np.savez_compressed(p, data=data, fs=100_000,
                                channel_names=np.array(["I"]),
                                metadata=json.dumps({**meta, "mode": "cw",
                                                     "vco_v": 2.5}))
            print(f"  CW {k+1}/{args.cw_chunks}: {data.shape[1]:,} samples "
                  f"({data.shape[1]/1e5:.1f} s) in {time.time()-t0:.1f} s, "
                  f"rms {data[0].std()*1e6:.1f} uV -> {os.path.basename(p)}",
                  flush=True)
    finally:
        try:
            dev.analog_output.channels[0].reset()
        except Exception:
            pass
        dev.close()

    # ---------- 2. FMCW Config B, raw CPIs ----------
    dev = A.open_device_cfg()
    fs, spc, nch, ramp = 100_000, 100, 128, 1000.0
    try:
        A.configure_chirp(dev, ramp, 0.5, 4.5, shape="sawtooth")
        cpis, t0 = [], time.time()
        for i in range(args.cpis):
            d, _ = A.record_chirps(dev, nch, spc, fs=fs, sync=True)
            cpis.append(d[0].astype(np.float32))
            if (i + 1) % 50 == 0:
                print(f"    CPI {i+1}/{args.cpis} "
                      f"({time.time()-t0:.0f} s)", flush=True)
        cpis = np.array(cpis)
    finally:
        try:
            dev.analog_output.channels[0].reset()
        except Exception:
            pass
        dev.close()
    p = os.path.join(OUT, f"{stamp}_{args.label}_fmcw_cfgB.npz")
    np.savez_compressed(p, cpis=cpis, fs=fs, spc=spc, nchirps=nch, ramp=ramp,
                        slope_300MHz=300e6 * ramp, slope_180MHz=180e6 * ramp,
                        metadata=json.dumps({**meta, "mode": "fmcw_cfgB"}))
    print(f"  Config B: {cpis.shape} over {time.time()-t0:.1f} s "
          f"-> {os.path.basename(p)}", flush=True)

    # ---------- 3. FMCW Config A, fine range ----------
    dev = A.open_device_cfg()
    fs, spc, ramp = 100_000, 2000, 50.0
    try:
        A.configure_chirp(dev, ramp, 0.5, 4.5, shape="sawtooth")
        blocks = []
        for _ in range(args.cfga_blocks):
            d, _ = A.record_chirps(dev, 8, spc, fs=fs, sync=True)
            blocks.append(d[0].astype(np.float32))
        chirps = np.concatenate(blocks, axis=0)
    finally:
        try:
            dev.analog_output.channels[0].reset()
        except Exception:
            pass
        dev.close()
    p = os.path.join(OUT, f"{stamp}_{args.label}_fmcw_cfgA.npz")
    np.savez_compressed(p, chirps=chirps, fs=fs, spc=spc, ramp=ramp,
                        slope_300MHz=300e6 * ramp, slope_180MHz=180e6 * ramp,
                        metadata=json.dumps({**meta, "mode": "fmcw_cfgA"}))
    print(f"  Config A: {chirps.shape} -> {os.path.basename(p)}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
