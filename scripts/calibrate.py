#!/usr/bin/env python3
"""Record and write the TODAY.md Block 2 calibration files.

  python scripts/calibrate.py dc                    # 10 s, room still
  python scripts/calibrate.py spurs                 # 60 s, room still, wall
  python scripts/calibrate.py iq                    # 30 s, someone WALKING 2-4 m
  python scripts/calibrate.py background            # 30 s, room still
  python scripts/calibrate.py manifest              # write manifest.json
  python scripts/calibrate.py all                   # dc -> spurs -> background (prompts)

Add --infile capture.npz to compute from an existing (2, n) capture instead
of recording. Files land in cal/<YYYYMMDD>/.
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from klc6 import cal as C                # noqa: E402

SECONDS = {"dc": 10, "spurs": 60, "iq": 30, "background": 30}
FS = 50_000                              # >30 s captures per FINDINGS 1.2


def get_capture(args, seconds):
    if args.infile:
        z = np.load(args.infile)
        return np.asarray(z["data"]), float(z["fs"])
    from klc6 import acquire as A
    dev = A.open_device()
    try:
        print(f"  recording {seconds} s at {FS} Sa/s, both channels ...")
        data = A.record(dev, seconds, fs=FS, channels=(0, 1))[0]
        return data, FS
    finally:
        dev.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("dc", "spurs", "iq", "background",
                                     "manifest", "all"))
    ap.add_argument("--infile", help="compute from existing npz instead of recording")
    ap.add_argument("--date", default=datetime.date.today().strftime("%Y%m%d"))
    args = ap.parse_args()

    d = C.CAL_ROOT / args.date
    d.mkdir(parents=True, exist_ok=True)

    steps = ["dc", "spurs", "background"] if args.step == "all" else [args.step]
    for step in steps:
        if step == "manifest":
            files = {"dc": "dc.json", "spurs": "spurs_noamp.json",
                     "iq": "iq.json", "background": "background_cw.npz"}
            man = {}
            for k, fn in files.items():
                if not (d / fn).exists():
                    man[k] = {"file": fn, "status": "missing"}
                elif k == "iq" and not json.loads((d / "iq.json").read_text()).get("verified"):
                    man[k] = {"file": fn, "status": "partial — unverified"}
                else:
                    man[k] = {"file": fn, "status": "pass"}
            for k in ("vco_linearity", "range_axis", "background_fmcw"):
                man[k] = {"status": "blocked — no IF gain"}
            (d / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
            print(f"wrote {d/'manifest.json'}")
            continue

        if args.step == "all":
            input(f"\n[{step}] room ready? ({SECONDS[step]} s capture) — Enter to go ")
        x, fs = get_capture(args, SECONDS[step])

        if step == "dc":
            out = C.compute_dc(x)
            (d / "dc.json").write_text(json.dumps(out, indent=2) + "\n")
            print(f"  DC I {out['I']*1e3:+.3f} mV  Q {out['Q']*1e3:+.3f} mV")
        elif step == "spurs":
            out = C.compute_spurs(x, fs)
            (d / "spurs_noamp.json").write_text(json.dumps(out, indent=2) + "\n")
            print(f"  {len(out['spurs'])} spurs; strongest: "
                  f"{sorted(out['spurs'], key=lambda s: -s['excess_db'])[:5]}")
        elif step == "iq":
            out = C.compute_iq(x)
            (d / "iq.json").write_text(json.dumps(out, indent=2) + "\n")
            print(f"  phase error {out['phase_error_deg']:+.2f} deg, "
                  f"gain ratio {out['gain_ratio']:.3f}  (verified: false)")
        elif step == "background":
            f, p = C.compute_background(x, fs)
            np.savez(d / "background_cw.npz", f=f, p=p)
            print(f"  wrote background_cw.npz ({len(f)} bins)")
    print(f"\ncal dir: {d}")


if __name__ == "__main__":
    main()
