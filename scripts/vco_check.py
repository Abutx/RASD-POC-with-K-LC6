"""VCO characterisation (SPEC.md 10.1). Do this before any sweep code.

Three things the spec asks for:

  1. Open-circuit voltage at X1 pin 5. The datasheet's 1.65 V is for the 3.3 V
     variant's internal divider; the -00D is the 5 V version with a 4.7k pull-up
     to 5 V, so the real value differs. NEEDS W1 DISCONNECTED and a probe on
     pin 5 -- run with --probe to measure it on Channel 2.
  2. Whether W1 can actually drive the pull-up. Command 2.5 V and read pin 5
     back: ~2.5 V means no buffer needed, noticeably higher means add a
     unity-gain follower.
  3. Confirmation the VCO tunes at all: step W1 as DC across the range with a
     stationary target and watch the IF wander.

Step 3 needs no rewiring and runs by default. Steps 1 and 2 need Channel 2
clipped to X1 pin 5 -- then use --probe.

    python scripts/vco_check.py                  # step 3 only, IF response
    python scripts/vco_check.py --probe          # + measure pin 5 on CH2
    python scripts/vco_check.py --probe --open-circuit   # W1 disconnected
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dwfpy as dwf                    # noqa: E402
from klc6 import acquire as A          # noqa: E402
from klc6 import process as P          # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")


def read_channels(ain, chans, nfft, settle=0.15):
    time.sleep(settle)
    ain.configure(reconfigure=False, start=True)
    ain.wait_for_status(dwf.Status.DONE, read_data=True)
    return {c: np.asarray(ain.channels[c].get_data(), dtype=np.float64)
            for c in chans}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", action="store_true",
                    help="Channel 2 is clipped to X1 pin 5; measure it")
    ap.add_argument("--open-circuit", action="store_true",
                    help="W1 is physically disconnected; just read pin 5")
    ap.add_argument("--v-low", type=float, default=0.5)
    ap.add_argument("--v-high", type=float, default=4.5)
    ap.add_argument("--steps", type=int, default=9)
    ap.add_argument("--fs", type=int, default=100_000)
    ap.add_argument("--nfft", type=int, default=8192)
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    chans = (A.CH_I, A.CH_Q) if args.probe else (A.CH_I,)

    print("=" * 70)
    print("  K-LC6 VCO CHARACTERISATION (SPEC 10.1)")
    print("=" * 70)
    if args.probe:
        print("  Channel 2 assumed on X1 pin 5 (VCO in).")
    else:
        print("  Channel 2 not probing pin 5 -- steps 1 and 2 skipped.")
        print("  Clip CH2 (blue 2+) to X1 pin 5 and re-run with --probe for those.")

    dev = A.open_device_cfg()
    print(f"  device {A.device_summary(dev)} (config {A.CFG_BIG_AIN}, "
          f"{dev.analog_input.buffer_size_max:,} sample buffer)")
    try:
        ain = dev.analog_input
        ain.reset()
        for c in ain.channels:
            c.enabled = False
        # Pin 5 sits near 0-5 V, so no offset on the probe channel; the IF needs
        # its usual -0.2 V pedestal removal.
        ain.setup_channel(A.CH_I, range=5.0, offset=A.OFFSET_V,
                          filter="average", enabled=True)
        if args.probe:
            # Pin 5 swings 0-5 V. The 5 Vpp range centred at 0 only spans
            # +-2.75 V, so everything above that reads back as a flat 2.7434 V
            # and looks exactly like the VCO pin being clamped. Offset the
            # window to +2.5 V so it covers -0.25 to +5.25 V.
            ain.setup_channel(A.CH_Q, range=5.0, offset=2.5,
                              filter="average", enabled=True)
        ain.trigger.source = dwf.TriggerSource.NONE
        ain.setup_acquisition(mode=dwf.AcquisitionMode.SINGLE,
                              sample_rate=args.fs, buffer_size=args.nfft)
        ain.configure(reconfigure=True, start=True)
        ain.wait_for_status(dwf.Status.DONE, read_data=True)

        # ---- step 1: open circuit ----
        if args.open_circuit:
            if not args.probe:
                raise SystemExit("  --open-circuit needs --probe (CH2 on pin 5)")
            d = read_channels(ain, chans, args.nfft, settle=0.3)
            v = d[A.CH_Q]
            print(f"\n  STEP 1 -- open-circuit voltage at X1 pin 5:")
            print(f"    {v.mean():.4f} V  (ripple {v.std()*1e3:.2f} mV rms)")
            print(f"    datasheet Note 3 quotes 1.65 V for the 3.3 V variant; "
                  f"record THIS number instead.")
            return 0

        # ---- steps 2 + 3: drive DC, watch pin 5 and the IF ----
        volts = np.linspace(args.v_low, args.v_high, args.steps)
        print(f"\n  stepping W1 {args.v_low:.2f} -> {args.v_high:.2f} V "
              f"in {args.steps} steps\n")
        hdr = f"  {'W1 cmd':>8} "
        if args.probe:
            hdr += f"{'pin5 meas':>10} {'error':>8} "
        hdr += f"{'IF rms':>10} {'IF mean':>10}  IF spectrum peak"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        rows = []
        for vc in volts:
            A.set_dc(dev, float(vc))
            d = read_channels(ain, chans, args.nfft)
            xi = d[A.CH_I]
            xi_ac = xi - xi.mean()
            sp = np.abs(np.fft.rfft(xi_ac * np.hanning(len(xi_ac))))
            fr = np.fft.rfftfreq(len(xi_ac), 1 / args.fs)
            band = fr >= 20
            pk = float(fr[np.flatnonzero(band)[np.argmax(sp[band])]])
            line = f"  {vc:>8.3f} "
            meas = np.nan
            if args.probe:
                meas = float(d[A.CH_Q].mean())
                line += f"{meas:>10.4f} {meas-vc:>+8.4f} "
            line += (f"{xi_ac.std()*1e6:>9.1f}u {xi.mean()*1e3:>9.3f}m"
                     f"  {pk:>8.1f} Hz")
            print(line, flush=True)
            rows.append((float(vc), meas, float(xi_ac.std()), float(xi.mean()), pk))

        rows = np.array(rows)
        print()
        if args.probe:
            err = rows[:, 1] - rows[:, 0]
            print(f"  STEP 2 -- W1 vs pin 5: max error {np.nanmax(np.abs(err)):+.4f} V")
            if np.nanmax(np.abs(err)) < 0.1:
                print("    W1 drives the 4.7k pull-up fine -- NO buffer needed.")
            else:
                print("    pin 5 does not follow W1 -- add a unity-gain op-amp "
                      "follower between W1 and pin 5.")
        span = rows[:, 3].max() - rows[:, 3].min()
        print(f"  STEP 3 -- IF DC level moved {span*1e3:.2f} mV across the sweep, "
              f"IF rms {rows[:,2].min()*1e6:.0f}-{rows[:,2].max()*1e6:.0f} uV")
        if span * 1e3 > 1.0 or rows[:, 2].max() / max(rows[:, 2].min(), 1e-12) > 1.5:
            print("    the IF responds to the tune voltage -- the VCO is tuning.")
        else:
            print("    the IF barely moved. Either the VCO pin is not connected,")
            print("    W1 cannot drive it, or there is no target in the beam.")

        np.savez_compressed(os.path.join(OUT, "vco_check.npz"),
                            v_cmd=rows[:, 0], v_meas=rows[:, 1],
                            if_rms=rows[:, 2], if_dc=rows[:, 3], if_peak=rows[:, 4])
        print(f"\n  wrote {os.path.join(OUT, 'vco_check.npz')}")
    finally:
        try:
            A.set_dc(dev, 0.0)
            dev.analog_output.channels[0].reset()
        except Exception:
            pass
        dev.close()
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
