"""Environment check. Run this FIRST on any new machine.

Every failure mode that has actually cost time on this project, checked in the
order that matters, with the specific fix printed rather than a stack trace:

  1. Python packages          -> pip install -r requirements.txt
  2. WaveForms SDK / dwf.dll  -> dwfpy imports but cannot load the C library
  3. Device enumeration       -> USB unplugged, or WaveForms is holding it
  4. Device configuration 1   -> needed for the 16,384-sample FMCW buffer
  5. Analog in / out present  -> the module only selects; open() must be called
  6. A real capture           -> proves the K-LC6 is powered and wired to CH1

    python scripts/doctor.py
"""
from __future__ import annotations

import importlib
import os
import sys

OK, BAD, WARN = "[ ok ]", "[FAIL]", "[warn]"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    fails = 0
    print("=" * 70)
    print("  K-LC6 ENVIRONMENT CHECK")
    print("=" * 70)
    print(f"  python {sys.version.split()[0]}")
    print(f"  {sys.executable}")
    print(f"  project {ROOT}\n")

    # ---- 1. packages ----
    need = {"numpy": "numpy", "scipy": "scipy", "matplotlib": "matplotlib",
            "PIL": "pillow", "tkinter": "tkinter (system package)",
            "dwfpy": "dwfpy"}
    missing = []
    for mod, pkg in need.items():
        try:
            m = importlib.import_module(mod)
            v = getattr(m, "__version__", "")
            print(f"{OK} {mod:12s} {v}")
        except Exception as exc:
            print(f"{BAD} {mod:12s} MISSING -- {exc}")
            missing.append(pkg)
            fails += 1
    if missing:
        print(f"\n  fix: pip install -r requirements.txt")
        if "tkinter (system package)" in missing:
            print("       tkinter is not pip-installable. python.org builds include")
            print("       it; on Debian/Ubuntu: sudo apt install python3-tk")
        return 1

    # ---- 2. the dwf C library ----
    import dwfpy as dwf
    try:
        ver = dwf.Application.get_version()
        print(f"{OK} {'dwf runtime':12s} {ver}")
    except Exception as exc:
        print(f"{BAD} dwf runtime could not load: {exc}")
        print("\n  dwfpy is only a wrapper -- the WaveForms SDK provides dwf.dll.")
        print("  Install Digilent WaveForms, then re-run:")
        print("  https://digilent.com/reference/software/waveforms/waveforms-3/start")
        return 1

    # ---- 3. enumeration ----
    try:
        devs = dwf.Device.enumerate()
    except Exception as exc:
        print(f"{BAD} enumeration failed: {exc}")
        return 1
    if not devs:
        print(f"{BAD} no Analog Discovery found")
        print("\n  - is the USB cable connected (and a DATA cable, not charge-only)?")
        print("  - is the WaveForms application open? It holds the device")
        print("    exclusively; close it completely.")
        return 1
    for d in devs:
        print(f"{OK} {'device':12s} {d.name} SN {d.serial_number} rev {d.revision}")

    # ---- 4/5. open in configuration 1 ----
    sys.path.insert(0, ROOT)
    from klc6 import acquire as A
    try:
        dev = A.open_device_cfg()
    except Exception as exc:
        print(f"{BAD} could not open in configuration 1: {exc}")
        print("\n  Close WaveForms. If it persists, unplug and replug the USB.")
        return 1
    try:
        ain, aout = dev.analog_input, dev.analog_output
        if ain is None:
            print(f"{BAD} analog_input is None -- the device was selected but not opened")
            fails += 1
        else:
            buf = ain.buffer_size_max
            good = buf >= 16384
            print(f"{OK if good else WARN} {'buffer':12s} {buf:,} samples "
                  f"({'config 1 active' if good else 'expected 16,384 for FMCW'})")
            fails += 0 if good else 1
        print(f"{OK} {'analog out':12s} {len(aout.channels)} channels (W1 drives the VCO)")

        # ---- 6. a real capture ----
        print("\n  capturing 0.5 s from Channel 1 (K-LC6 IF)...")
        data, _ = A.record(dev, 0.5, fs=50_000, channels=(A.CH_I,),
                           range_v=5.0, offset_v=A.OFFSET_V,
                           filter_mode="average")
        x = data[0]
        import numpy as np
        rms = float(x.std()) * 1e6
        codes = len(np.unique(x))
        print(f"  {x.size:,} samples | rms {rms:.1f} uV | mean {x.mean()*1e3:+.2f} mV "
              f"| {codes} distinct ADC codes")
        if rms < 30:
            print(f"{BAD} Channel 1 is flat ({rms:.1f} uV, expect ~150-180).")
            print("       - is the K-LC6 powered (external 5 V on X1 pin 2)?")
            print("       - is CH1 (orange 1+) on X1 pin 3, and 1- on ground?")
            fails += 1
        elif rms > 5000:
            print(f"{WARN} unusually hot ({rms:.0f} uV). Normal for FMCW ramp")
            print("       feedthrough; suspicious if the VCO pin is unconnected.")
        else:
            print(f"{OK} {'IF signal':12s} healthy")
    finally:
        try:
            dev.close()
        except Exception:
            pass

    print("-" * 70)
    if fails == 0:
        print("  ALL CHECKS PASS -- try:  python scripts/live.py --threshold 5")
        print("                          python scripts/ppi.py --rmax 10")
    else:
        print(f"  {fails} CHECK(S) FAILED -- see above")
    print("=" * 70)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
