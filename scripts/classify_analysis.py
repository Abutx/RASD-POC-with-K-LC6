#!/usr/bin/env python3
"""Drone vs human: which discriminants work, and out to what range.

The uncomfortable fact this script exists to quantify: on this hardware the
range at which you can DETECT and TRACK a target is several times the range at
which you can say WHAT it is. Between those two numbers you have a track with
no label, and pretending otherwise is how a demo turns into a false alarm.

  python scripts/classify_analysis.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim import capture, drone, human, klc6      # noqa: E402

OUT = "out/demo"
RULE = "=" * 78
FS = 50_000
NPERSEG = 8192


def h(t):
    print(f"\n{RULE}\n{t}\n{RULE}")


def spectrum(x, fs=FS):
    w = np.hanning(len(x))
    sp = np.abs(np.fft.fftshift(np.fft.fft(x * w))) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(len(x), 1 / fs))
    return f, sp


def extent_hz(f, sp, frac=0.999):
    """|Doppler| below which `frac` of the energy lies."""
    order = np.argsort(np.abs(f))
    c = np.cumsum(sp[order]) / sp.sum()
    return float(np.abs(f[order])[np.searchsorted(c, frac)])


def physical_separation(dur=4.0):
    h("1. THE PHYSICAL SEPARATION -- no receiver, just kinematics")
    t = np.arange(int(dur * FS)) / FS

    rows = []
    for name, sig in [
        ("human, brisk walk 1.4 m/s", human.micro_doppler(t)),
        ("human, running 4 m/s", human.micro_doppler(t, walk_mps=4.0, f_gait=2.8)),
        ("Mini 3, hover 6040 rpm", drone.micro_doppler(t, rpm=drone.RPM_HOVER)),
        ("Mini 3, max 10700 rpm", drone.micro_doppler(t, rpm=drone.RPM_MAX)),
    ]:
        f, sp = spectrum(sig)
        e99 = extent_hz(f, sp, 0.99)
        e999 = extent_hz(f, sp, 0.999)
        above = sp[np.abs(f) > 1500].sum() / sp.sum()
        rows.append((name, e99, e999, above))

    print(f"\n  {'target':<30}{'99% within':>12}{'99.9% within':>14}"
          f"{'energy >1.5 kHz':>17}")
    print("  " + "-" * 73)
    for name, e99, e999, above in rows:
        print(f"  {name:<30}{e99:9.0f} Hz{e999:11.0f} Hz{above*100:15.1f}%")

    print(f"\n  Fastest Doppler a walking person can produce: "
          f"{human.max_limb_doppler_hz():.0f} Hz")
    print(f"  Mini 3 blade tips at hover:                    "
          f"{drone.rpm_to_tip_hz(drone.RPM_HOVER):.0f} Hz")
    print(f"  Ratio: {drone.rpm_to_tip_hz(drone.RPM_HOVER)/human.max_limb_doppler_hz():.1f}x")
    print("""
  There is nothing on a human body that moves at 48 m/s. The band from about
  1.5 kHz to 14 kHz is EMPTY for any biological target and full for any rotary
  one. That is the discriminant with real physics behind it -- everything else
  in this file is a fallback for when this one runs out of range.""")


def cadence_separation():
    h("2. CADENCE -- two orders of magnitude apart")
    print(f"\n  {'target':<30}{'cadence':>12}  origin")
    print("  " + "-" * 62)
    print(f"  {'human, walking':<30}{human.GAIT_HZ:9.1f} Hz  step rate")
    print(f"  {'human, running':<30}{2.8:9.1f} Hz  step rate")
    for name, rpm in [("Mini 3, hover", drone.RPM_HOVER),
                      ("Mini 3, mid", 8000.0), ("Mini 3, max", drone.RPM_MAX)]:
        print(f"  {name:<30}{drone.herm_spacing_hz(rpm):9.1f} Hz  blade pass")
    print("""
  A human cadence peak and a drone blade-pass peak are ~100x apart and cannot
  be confused by any estimator. FAN_DETECTION.md already reports the human
  half of this measured on real bench data: the envelope-periodicity method
  recovered a 2.09 Hz gait cadence at 5.57 sigma. That is a POSITIVE CONTROL
  we already own, on real hardware, for the discriminant below.""")


def discriminant_ranges():
    h("3. RANGE OF EACH DISCRIMINANT -- where classification actually stops")
    bin_hz = FS / NPERSEG
    tip = drone.rpm_to_tip_hz(drone.RPM_HOVER)
    n_blade = max(int((tip - 4000) / bin_hz), 1)

    sig_body = drone.sigma_body_m2()
    sig_blade = drone.sigma_blades_m2()

    print(f"\n  CW, {FS//1000} kSa/s, {bin_hz:.1f} Hz bins, 5 sigma. Range in metres.\n")
    print(f"  {'discriminant':<38}{'no amp':>10}{'+30 dB':>10}  needs")
    print("  " + "-" * 76)

    out = {}
    # D0 -- can we see the target at all?
    for label, sigma, freq, nb, key, needs in [
        ("DETECT person (bulk line)", 1.0, 250.0, 1, "det_person", "any return"),
        ("DETECT Mini 3 (body line, moving)", sig_body, 800.0, 1, "det_drone",
         "any return"),
        ("CLASSIFY: gait cadence 1.8 Hz", 1.0 * 0.25, 250.0, 1, "gait",
         "bulk line + AM"),
        ("CLASSIFY: blade energy >1.5 kHz", sig_blade, tip, n_blade, "blade",
         "blade pedestal"),
        ("CLASSIFY: blade-pass comb 201 Hz", sig_blade / np.sqrt(n_blade),
         201.0, 1, "herm", "HERM tooth"),
    ]:
        r0 = klc6.detection_range_m(sigma, bin_hz, freq, if_gain_db=0.0,
                                    snr_req_db=14.0, n_spread_bins=nb)
        r30 = klc6.detection_range_m(sigma, bin_hz, freq, if_gain_db=30.0,
                                     snr_req_db=14.0, n_spread_bins=nb)
        out[key] = (r0, r30)
        print(f"  {label:<38}{r0:8.2f} m{r30:8.2f} m  {needs}")

    print(f"""
  THE GAP. With the amp you can track a Mini 3 to {out['det_drone'][1]:.0f} m but only see its
  blades to {out['blade'][1]:.1f} m. Between those you have an unlabelled track. The
  ratio is {out['det_drone'][1]/out['blade'][1]:.1f}x, and it does not close by processing harder --
  blade RCS is 30 dB below body RCS and that is a property of the target.

  A person is detectable to {out['det_person'][1]:.0f} m with the amp, further than the drone,
  because 1 m^2 vs 0.02 m^2 is 17 dB. So the confuser outranges the target.""")
    return out


def rcs_discriminant():
    h("4. RCS AS A DISCRIMINANT -- works at full range, but noisy")
    print(f"""
  Two radars give range, and range plus measured amplitude gives an absolute
  RCS estimate. That is available everywhere the target is detected at all,
  which makes it the only strong discriminant that spans the whole gap.

    moving person   ~   0 dBsm   (1 m^2, RFbeam datasheet p.6)
    DJI Mini 3      ~ -17 dBsm   (0.02 m^2, extrapolated -- see DEMO_TWO_RADAR)
    separation           17 dB

  The catch is fluctuation. Drone RCS swings 10-15 dB with aspect angle and a
  person's does too, so a single look can be wrong by more than half the gap.
  Integrated over a track of a few seconds it separates well; on one CPI it
  does not. Treat it as evidence, never as a verdict, and remember the RCS
  figure it rests on is the weakest number in this whole analysis.""")


def kinematics_discriminant():
    h("5. KINEMATICS -- already implemented, and free")
    print("""
  klc6/track.py already computes exactly the right features per confirmed
  track: mean_abs_v, v_var, hover_frac (fraction of dwell with |v| < 0.2 m/s).

    * hover_frac > 0.8 with a real return  ->  nothing biological does this.
      A person cannot hold station to 0.2 m/s while remaining a strong
      scatterer; they either stand still (and vanish into the clutter notch)
      or they move. A target that hovers AND stays visible is a rotorcraft.
    * smooth velocity, low v_var          ->  rigid body under control
    * |v| oscillating at 1-3 Hz           ->  gait, i.e. human

  This costs nothing -- the tracker is written and the features are already in
  Track.features(). It is the cheapest classification signal available and it
  works to full tracking range.""")


def recommended_classifier():
    h("6. WHAT TO ACTUALLY BUILD")
    print("""
  Do NOT build a blade-micro-Doppler classifier as the primary. It is the most
  convincing evidence and the shortest-ranged, so it cannot carry the system.

  Layered, in the order they run:

  LAYER 1 -- gait cadence, on the bulk return.  Range: full detection range.
    Envelope periodicity on the tracked target's return; look for a 1-3 Hz
    peak. Present -> human, and you are done. This is the workhorse, it needs
    only the bulk line, and FAN_DETECTION.md already demonstrates the method
    at 5.57 sigma on a real walking person.

  LAYER 2 -- kinematics from the tracker.  Range: full tracking range.
    hover_frac, v_var, velocity envelope. Free, already written.

  LAYER 3 -- RCS from range + amplitude.  Range: full detection range.
    17 dB separation, 10-15 dB fluctuation. Evidence, not verdict.

  LAYER 4 -- blade micro-Doppler.  Range: 1.5-5 m with the amp.
    Energy above 1.5 kHz, and a blade-pass comb at 200-360 Hz. When it fires
    it is definitive -- nothing biological produces it. Use it to CONFIRM at
    close range, and to label training data for the other three layers.

  Report the layer that fired. "Drone, confirmed by blade signature at 2 m" and
  "drone, inferred from hover + low RCS at 12 m" are different claims and the
  demo should say which one it is making.

  THE HONEST FAILURE MODE, state it before someone finds it: a person walking
  at constant speed with their arms still, at 12 m, has no gait modulation, no
  hover, and an RCS that could be a large drone. Layers 1-3 all degrade
  together. The system will not reliably tell you what that is, and the fix is
  range -- get the target closer, or get the amp and a longer baseline.""")


def figure(dur=4.0, out_dir=OUT):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from klc6 import process

    os.makedirs(out_dir, exist_ok=True)
    t = np.arange(int(dur * FS)) / FS
    cases = [("walking human, 1.4 m/s", human.micro_doppler(t), 1.0),
             ("running human, 4 m/s",
              human.micro_doppler(t, walk_mps=4.0, f_gait=2.8), 1.0),
             ("DJI Mini 3, hover", drone.micro_doppler(t, rpm=drone.RPM_HOVER),
              drone.sigma_body_m2()),
             ("DJI Mini 3, full throttle",
              drone.micro_doppler(t, rpm=drone.RPM_MAX), drone.sigma_body_m2())]

    fig, axes = plt.subplots(2, len(cases), figsize=(4.0 * len(cases), 8),
                             constrained_layout=True)
    for j, (name, sig, _sigma) in enumerate(cases):
        tt, v, S = process.spectrogram_mps(sig, FS, nperseg=4096, overlap=0.85)
        f = v * klc6.HZ_PER_MPS
        ax = axes[0, j]
        ax.pcolormesh(tt, f / 1000, S, vmin=S.max() - 55, vmax=S.max(),
                      cmap="magma", shading="auto")
        ax.set_ylim(-15, 15)
        ax.axhline(1.5, color="cyan", lw=1.1, ls="--")
        ax.axhline(-1.5, color="cyan", lw=1.1, ls="--")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("time (s)")
        if j == 0:
            ax.set_ylabel("Doppler (kHz)")
            ax.text(0.05, 2.2, "1.5 kHz: nothing human goes above this",
                    color="cyan", fontsize=8)
        # zoomed to the human band
        ax = axes[1, j]
        ax.pcolormesh(tt, f, S, vmin=S.max() - 55, vmax=S.max(),
                      cmap="magma", shading="auto")
        ax.set_ylim(-900, 900)
        ax.set_xlabel("time (s)")
        if j == 0:
            ax.set_ylabel("Doppler (Hz)  [zoom]")
    fig.suptitle(
        "Top: full band. Bottom: zoomed to the human band. Each panel is "
        "micro-Doppler ONLY, normalised to unit power --\n"
        "the bulk/body line is omitted, so this shows SHAPE, not relative "
        "level. The drone fills 1.5-14 kHz and the human\n"
        "cannot reach it; inside the human band the human braids at the gait "
        "rate while the drone shows a flat HERM comb.", fontsize=11)
    p = f"{out_dir}/drone_vs_human.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"\n  figure -> {p}")


def main():
    physical_separation()
    cadence_separation()
    discriminant_ranges()
    rcs_discriminant()
    kinematics_discriminant()
    recommended_classifier()
    figure()


if __name__ == "__main__":
    main()
