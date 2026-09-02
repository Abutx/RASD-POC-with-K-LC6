#!/usr/bin/env python3
"""Record a labeled training capture (TODAY.md Blocks 3-4) per klc6_capture_spec.md.

Both channels, 50 kSa/s, config-1 buffer, writes <name>.npz + .json sidecar into
out/<class-group>/ so the dataset manifest picks it up. Prints a level + I/Q
sanity check after each capture.

  # human, walking toward
  python scripts/session.py --label human_moving --detail toward_5to1 \
      --aspect approach --distance 5 --seconds 30 --session 20260902_lab

  # drone, held still with props spinning (isolates blade signature)
  python scripts/session.py --label drone_dji_mini --detail propsonly_1m \
      --aspect static-body --distance 1 --throttle mid --seconds 20 \
      --session 20260902_lab --group drone

  python scripts/session.py --list-devices
"""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from klc6 import acquire as A            # noqa: E402

FS = 50_000


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", help="class, e.g. human_moving | drone_dji_mini | empty")
    ap.add_argument("--detail", default="", help="filename detail, e.g. toward_5to1")
    ap.add_argument("--session", required="--list-devices" not in sys.argv,
                    help="session id, e.g. 20260902_lab (splits happen on this)")
    ap.add_argument("--group", default=None,
                    help="out/ subdir (default: inferred from label)")
    ap.add_argument("--aspect", default="", help="approach|recede|cross|hover|static-body|fidget")
    ap.add_argument("--distance", type=float, default=None)
    ap.add_argument("--throttle", default=None, help="drone only")
    ap.add_argument("--target", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--fs", type=int, default=FS)
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    if args.list_devices:
        import dwfpy as dwf
        for d in dwf.Device.enumerate():
            print(d.name, d.serial_number)
        return
    if not args.label:
        ap.error("--label required")

    group = args.group or ("drone" if "drone" in args.label else
                           "human" if "human" in args.label else
                           "bird" if "bird" in args.label else "cw")
    outdir = Path("out") / group
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now()
    detail = f"_{args.detail}" if args.detail else ""
    base = outdir / f"{stamp.strftime('%Y%m%d_%H%M%S')}_{args.label}{detail}"

    dev = A.open_device_cfg(A.CFG_BIG_AIN)
    try:
        print(f"  >>> RECORDING {args.seconds:g} s @ {args.fs} Sa/s, I+Q -- go <<<")
        data = A.record(dev, args.seconds, fs=args.fs, channels=(A.CH_I, A.CH_Q))[0]
    finally:
        dev.close()

    names = A.channel_names((A.CH_I, A.CH_Q))
    for nm, ch in zip(names, data):
        pk = np.abs(ch).max()
        flag = ("  << CLIPPING" if pk > 2.7 else
                "  << very low / unplugged?" if pk < 5e-4 else "")
        print(f"    {nm}: rms {np.std(ch)*1e6:7.1f} uV  peak {pk*1e3:6.2f} mV{flag}")
    # I/Q liveness: correlation magnitude well below 1 means two real channels
    c = np.corrcoef(data[0], data[1])[0, 1]
    print(f"    I/Q correlation {c:+.2f} (near +-1 = degenerate/same pin)")

    np.savez(str(base) + ".npz", data=data, fs=args.fs,
             channel_names=np.array(names))
    meta = {"label": args.label, "source": "bench", "mode": "cw",
            "session": args.session, "target": args.target or args.label,
            "distance_m": args.distance, "aspect": args.aspect,
            "if_gain_db": 0, "notes": args.notes}
    if args.throttle is not None:
        meta["throttle"] = args.throttle
    Path(str(base) + ".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  saved {base}.npz + .json")


if __name__ == "__main__":
    main()
