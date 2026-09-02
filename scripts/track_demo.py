#!/usr/bin/env python3
"""Live 1D tracker demo (TODAY.md Block 5 test).

  python scripts/track_demo.py                 # live from the AD2
  python scripts/track_demo.py --npz file.npz  # replay a saved cpis capture

Walk 5 m -> 1 m -> back: expect ONE confirmed track, range down then up,
velocity flipping sign at the turn. Empty room: no confirmed tracks.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from klc6 import track as T              # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", help="replay (n_cpi, chirps, spc) instead of live")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--ramp", type=float, default=1000.0)
    ap.add_argument("--chirps", type=int, default=128)
    ap.add_argument("--spc", type=int, default=100)
    ap.add_argument("--fs", type=int, default=100_000)
    ap.add_argument("--bw", type=float, default=180e6,
                    help="measured sweep (uncalibrated range axis)")
    ap.add_argument("--offset-db", type=float, default=10.0)
    args = ap.parse_args()

    tracker = T.Tracker(dt=args.chirps / args.ramp)

    def handle(cube, k):
        dets = T.process_cpi(cube, args.fs, args.ramp, args.bw,
                             offset_db=args.offset_db)
        confirmed = tracker.step(dets)
        line = f"  cpi {k:4d}  {len(dets)} det"
        for tr in confirmed:
            f = tr.features()
            line += (f"  | trk{tr.id} r={tr.x[0]:5.2f} m v={tr.x[1]:+5.2f} m/s "
                     f"age={tr.age} hover={f['hover_frac']:.2f}")
        print(line, flush=True)

    if args.npz:
        cubes = np.load(args.npz)["cpis"]
        for k, cube in enumerate(cubes):
            handle(cube, k)
    else:
        from klc6 import acquire as A
        dev = A.open_device_cfg(A.CFG_BIG_AIN)
        A.configure_chirp(dev, args.ramp, 0.5, 4.5, shape="sawtooth")
        time.sleep(0.5)
        print(f"live for {args.seconds:.0f} s — walk 5 m -> 1 m -> back")
        t0, k = time.time(), 0
        try:
            while time.time() - t0 < args.seconds:
                cube = A.record_chirps(dev, args.chirps, args.spc,
                                       fs=args.fs, sync=True)[0][0]
                handle(cube.reshape(args.chirps, args.spc), k)
                k += 1
        finally:
            dev.analog_output.channels[0].reset()
            dev.close()
    n_conf = len({t.id for t in tracker.tracks if t.confirmed})
    print(f"\nconfirmed tracks this session: "
          f"{n_conf + sum(1 for _ in ())} live now, see per-CPI lines above")


if __name__ == "__main__":
    main()
