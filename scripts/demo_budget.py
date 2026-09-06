#!/usr/bin/env python3
"""Two-radar DJI Mini 3 demo: link budget, detection range, position accuracy.

Prints the whole analysis and writes figures to out/demo/. Every number that
feeds docs/DEMO_TWO_RADAR.md comes from here, so the doc can be regenerated
rather than hand-maintained.

  python scripts/demo_budget.py            # full analysis + figures
  python scripts/demo_budget.py --validate # just the model self-checks
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim import capture, drone, geometry, klc6      # noqa: E402

OUT = "out/demo"
RULE = "=" * 78


def h(title):
    print(f"\n{RULE}\n{title}\n{RULE}")


# ---------------------------------------------------------------- validation
def validate():
    h("MODEL VALIDATION -- does this reproduce things we already know?")

    print("\n[1] RFbeam's own page-6 detection ranges")
    for name, sigma, claim in [("moving person", 1.0, 24.0), ("moving car", 50.0, 62.0)]:
        r_ds = klc6.datasheet_range_m(sigma)
        # our budget, at the datasheet's own reference conditions
        r_us = klc6.detection_range_m(sigma, bin_hz=klc6.REF_BW_HZ, freq_hz=500.0,
                                      if_gain_db=0.0, snr_req_db=klc6.REF_SNR_DB,
                                      module_noise_only=True)
        print(f"    {name:15s} datasheet says >{claim:.0f} m | "
              f"formula {r_ds:5.1f} m | our budget {r_us:5.1f} m")

    print("\n[2] measured empty-room floor vs the model's interpolation")
    base = "out/baseline/20260829_061237_empty_baseline_60s.npz"
    if os.path.exists(base):
        z = np.load(base, allow_pickle=True)
        fs = float(z["fs"])
        x = z["data"][0].astype(float)
        for lo, hi in [(200, 260), (2000, 4000), (7000, 8500), (13000, 14500)]:
            meas = capture.realised_floor_v_rthz(x, fs, lo, hi)
            model = klc6.measured_floor_v_rthz((lo + hi) / 2)
            print(f"    {lo:6d}-{hi:6d} Hz : measured {meas*1e9:7.1f}  "
                  f"model {model*1e9:7.1f} nV/rtHz   "
                  f"({20*np.log10(meas/model):+.2f} dB)")
    else:
        print("    (baseline capture not found -- skipped)")

    print("\n[3] renderer round-trip: synthesise an empty room, measure it back")
    fs = 50_000
    n = int(20 * fs)
    empty = capture.render(np.zeros(n, dtype=complex), fs, seed=1)
    print(f"    rendered rms  {empty[0].std()*1e6:6.1f} uV   "
          f"(bench measured 158.4 uV)")
    print(f"    distinct ADC codes {len(np.unique(np.round(empty[0]/klc6.AD2_LSB_V))):d}"
          f"   (bench measured 4-7)")
    for lo, hi in [(2000, 4000), (7000, 8500)]:
        r = capture.realised_floor_v_rthz(empty[0], fs, lo, hi)
        m = klc6.measured_floor_v_rthz((lo + hi) / 2)
        print(f"    {lo:6d}-{hi:6d} Hz : rendered {r*1e9:7.1f} vs model "
              f"{m*1e9:7.1f} nV/rtHz  ({20*np.log10(r/m):+.2f} dB)")

    print("\n[5] independent cross-check against a published field measurement")
    # Rahman & Robertson, IET Radar Sonar Navig 2019: K-band FMCW, +25 dBm,
    # 24.5 dBi horns, detected DJI Phantom 3 micro-Doppler at 75-95 m. Scale
    # their link to ours and see whether our predicted blade range is consistent.
    their_eirp, their_grx, their_range = 25.0 + 24.5, 24.5, 85.0
    adv_db = ((their_eirp - klc6.EIRP_DBM) + (their_grx - klc6.G_ANT_DBI)
              + klc6.sensitivity_penalty_db(7757.0, 0.0))
    scaled = their_range / 10 ** (adv_db / 40.0)
    print(f"    their EIRP {their_eirp:.1f} dBm vs our {klc6.EIRP_DBM:.0f} dBm, "
          f"Rx gain {their_grx:.1f} vs {klc6.G_ANT_DBI:.1f} dBi,")
    print(f"    plus our {klc6.sensitivity_penalty_db(7757.0, 0.0):.1f} dB AD2 "
          f"penalty  =  {adv_db:.1f} dB total advantage")
    ours = klc6.detection_range_m(drone.sigma_blades_m2(), 6.104, 7757.0,
                                  snr_req_db=14.0, n_spread_bins=615)
    print(f"    their 85 m (Phantom 3) scales to {scaled:.2f} m on our bench")
    print(f"    our budget for a Mini 3 says   {ours:.2f} m")
    print(f"    gap = {40*np.log10(scaled/ours):.1f} dB. A Phantom 3's blades are ~1.7x")
    print( "    longer and its body ~4 dB bigger, which explains ~9 dB of that.")
    print( "    The rest is that their spectrograms integrate far longer than our")
    print( "    6.1 Hz / 5-sigma criterion and they range-gate the clutter away.")
    print( "    CONCLUSION: treat our blade numbers as a conservative LOWER bound.")
    print(f"    Plausible band for blade micro-Doppler, no amp: {ours:.2f} - {scaled:.2f} m.")

    print("\n[4] geometry: GDOP closed form vs sqrt(2)/sin(beta)")
    rig = geometry.Rig(baseline_m=4.0)
    for y in [1.0, 2.0, 4.0, 8.0]:
        p = np.array([0.0, y])
        beta = rig.crossing_angle_deg(p)
        print(f"    y={y:4.1f} m  beta={beta:5.1f} deg  GDOP={rig.gdop(p):5.2f}  "
              f"(sqrt2/sin = {np.sqrt(2)/np.sin(np.radians(beta)):5.2f})")


# ------------------------------------------------------------------- target
def target_summary():
    h("TARGET -- DJI Mini 3, and what its rotors do to 24.125 GHz")
    print(f"\n  mass {drone.MASS_KG*1000:.0f} g | 6030F props, "
          f"{drone.PROP_DIAMETER_M*1000:.1f} mm dia, {drone.N_BLADES} blades, "
          f"{drone.N_ROTORS} rotors")
    print(f"  body RCS  {drone.SIGMA_BODY_DBSM:+.0f} dBsm "
          f"({drone.sigma_body_m2():.4f} m^2)  ESTIMATE, +/-5 dB")
    print(f"  blade RCS {drone.SIGMA_BODY_DBSM - drone.BLADE_BELOW_BODY_DB:+.0f} dBsm "
          f"({drone.sigma_blades_m2():.2e} m^2)  ESTIMATE, lit. 20-40 dB below body")
    print(f"\n  {'condition':<18}{'rpm':>8}{'tip m/s':>10}{'tip Hz':>10}"
          f"{'HERM Hz':>10}{'fs_min':>10}")
    for name, rpm in [("hover", drone.RPM_HOVER), ("mid throttle", 8000.0),
                      ("maximum", drone.RPM_MAX)]:
        tip_hz = drone.rpm_to_tip_hz(rpm)
        print(f"  {name:<18}{rpm:8.0f}{drone.rpm_to_tip_mps(rpm):10.1f}"
              f"{tip_hz:10.0f}{drone.herm_spacing_hz(rpm):10.1f}"
              f"{2*tip_hz/1000:9.1f}k")
    print(f"\n  Nyquist at 50 kSa/s is 25 kHz -- clears even max throttle. "
          f"But note:\n  the K-LC6_V2 (the variant WITH the 20 dB IF amp) rolls "
          f"off at 15 kHz, and\n  max-throttle tips at "
          f"{drone.rpm_to_tip_hz(drone.RPM_MAX):.0f} Hz sit on that shoulder.")


def where_does_blade_energy_land(fs=50_000, dur=2.0, seed=3):
    """Simulate, then MEASURE the spectral distribution rather than assume it."""
    h("WHERE THE BLADE ENERGY ACTUALLY LANDS (measured from the simulation)")
    t = np.arange(int(dur * fs)) / fs
    for label, el in [("radar level with rotors (el=0)", 0.0),
                      ("radar 30 deg below", 30.0),
                      ("radar 60 deg below", 60.0),
                      ("radar directly beneath (el=90)", 90.0)]:
        md = drone.micro_doppler(t, rpm=drone.RPM_HOVER, el_deg=el, seed=seed)
        sp = np.abs(np.fft.fftshift(np.fft.fft(md * np.hanning(len(md))))) ** 2
        f = np.fft.fftshift(np.fft.fftfreq(len(md), 1 / fs))
        tot = sp.sum()
        above4k = sp[np.abs(f) > 4000].sum() / tot
        above6k = sp[np.abs(f) > 6000].sum() / tot
        below1k = sp[np.abs(f) < 1000].sum() / tot
        print(f"  {label:<34} |v|<1k {below1k*100:5.1f}%   "
              f">4k {above4k*100:5.1f}%   >6k {above6k*100:5.1f}%")
    print("\n  Elevation is the whole ballgame. Level with the rotor plane the\n"
          "  blades project their full tip velocity onto the line of sight;\n"
          "  from underneath that motion is tangential and the signature\n"
          "  collapses toward DC, into the worst part of the spectrum.\n"
          "  This is FINDINGS trap 7 reproducing itself, as it must.")
    return None


# ------------------------------------------------------------- link budget
def detection_table():
    h("DETECTION RANGE -- how far can each feature actually be seen?")
    fs = 50_000
    nperseg = 8192
    bin_hz = fs / nperseg
    tip_hz = drone.rpm_to_tip_hz(drone.RPM_HOVER)
    herm_hz = drone.herm_spacing_hz(drone.RPM_HOVER)

    print(f"\n  CW, fs={fs//1000} kSa/s, {nperseg}-pt FFT -> {bin_hz:.2f} Hz bins, "
          f"5 sigma (14 dB) required")
    print(f"  Blade pedestal treated as band energy over 4 kHz..{tip_hz:.0f} Hz "
          f"({int((tip_hz-4000)/bin_hz)} bins)")

    n_blade_bins = max(int((tip_hz - 4000) / bin_hz), 1)
    sig_body = drone.sigma_body_m2()
    sig_blade = drone.sigma_blades_m2()

    rows = [
        ("body line, drone MOVING", sig_body, 800.0, 1, "coherent, bulk Doppler"),
        ("body line, drone HOVERING", sig_body, 30.0, 1, "sits in 1/f + mains"),
        ("HERM comb tooth", herm_hz, herm_hz, 1, "cadence line"),
        ("blade pedestal", sig_blade, tip_hz, n_blade_bins, "band energy"),
    ]

    print(f"\n  {'feature':<28}{'no amp':>12}{'+20 dB':>12}{'+30 dB':>12}"
          f"{'+40 dB':>12}")
    print("  " + "-" * 76)
    for name, sigma, freq, nbins, _note in rows:
        if name.startswith("HERM"):
            # a comb tooth carries roughly 1/N of the blade energy
            sigma = sig_blade / max(n_blade_bins ** 0.5, 1)
            freq = herm_hz
        cells = []
        for g in (0.0, 20.0, 30.0, 40.0):
            r = klc6.detection_range_m(sigma, bin_hz=bin_hz, freq_hz=freq,
                                       if_gain_db=g, snr_req_db=14.0,
                                       n_spread_bins=nbins)
            cells.append(f"{r:10.2f} m")
        print(f"  {name:<28}" + "".join(cells))

    print(f"\n  sensitivity now being lost to the AD2 front end:")
    for f_hz, lbl in [(30.0, "30 Hz (hover body)"), (herm_hz, "201 Hz (HERM)"),
                      (tip_hz, f"{tip_hz:.0f} Hz (blade tip)")]:
        row = "  " + f"{lbl:<24}"
        for g in (0.0, 20.0, 30.0, 40.0):
            row += f"{klc6.sensitivity_penalty_db(f_hz, g):9.1f} dB"
        print(row)
    return bin_hz, tip_hz, n_blade_bins


# ---------------------------------------------------------------- geometry
def geometry_table():
    h("POSITION -- two range circles, and what the geometry costs")

    print("\n  Achievable sweep bandwidth (the AD2's W1 cannot reach 10 V):")
    for lo, hi, note in [(0.5, 4.5, "W1 single-ended, as used today"),
                         (1.0, 5.0, "W1 shifted to the VCO's usable floor"),
                         (1.0, 10.0, "full VCO range -- needs a 10 V level shifter")]:
        bw = klc6.sweep_bandwidth_hz(lo, hi)
        print(f"    {lo:.1f}-{hi:4.1f} V : {bw/1e6:6.1f} MHz -> "
              f"range res {klc6.range_resolution_m(bw):5.2f} m   ({note})")
    print(f"    FINDINGS 5.3 measured ~180 MHz from one eyeballed data point;\n"
          f"    the datasheet's 25 MHz/V over a 4 V drive predicts 100 MHz.\n"
          f"    They disagree by 2.6 dB of range resolution -- worth settling.")

    bw = klc6.sweep_bandwidth_hz(0.5, 4.5)
    print(f"\n  Range accuracy at B={bw/1e6:.0f} MHz "
          f"(res {klc6.range_resolution_m(bw):.2f} m):")
    for snr in [10, 14, 20, 30]:
        print(f"    SNR {snr:2d} dB -> sigma_R = "
              f"{klc6.range_accuracy_m(bw, snr):.3f} m")

    print("\n  GDOP and position accuracy on the centreline "
          "(sigma_R = 0.25 m assumed):")
    sigma_r = 0.25
    print(f"    {'baseline':>10}{'range':>8}{'beta':>9}{'GDOP':>7}"
          f"{'sigma_pos':>11}{'major':>9}{'minor':>9}")
    for b in [2.0, 4.0, 8.0]:
        rig = geometry.Rig(baseline_m=b)
        for y in [1.0, 2.0, 4.0, 8.0, 15.0]:
            p = np.array([0.0, y])
            maj, mino = rig.error_ellipse_m(p, sigma_r)
            print(f"    {b:8.1f} m{y:7.1f} m{rig.crossing_angle_deg(p):8.1f}d"
                  f"{rig.gdop(p):7.2f}{rig.position_sigma_m(p, sigma_r):10.2f} m"
                  f"{maj:8.2f} m{mino:8.2f} m")

    print("\n  Baseline needed to hold GDOP <= 3 at a given range:")
    for y in [2.0, 5.0, 10.0, 20.0]:
        print(f"    {y:5.1f} m  ->  baseline {geometry.baseline_for_range(y, 3.0):5.2f} m")

    print("\n  Elevation coverage -- the narrow 12 deg lobe is a thin sheet:")
    rig = geometry.Rig(baseline_m=4.0)
    for y in [1.0, 2.0, 5.0, 10.0]:
        print(f"    at {y:5.1f} m the covered slab is only "
              f"{rig.elevation_slab_m(y):.2f} m tall")


def interference_note():
    h("TWO MODULES AT ONCE -- mutual interference, and why it is not a problem")
    print("""
  Both modules transmit at 24.125 GHz into the same volume, so radar B's
  receiver sees radar A's carrier directly. That cross-mix is far stronger
  than any target return and would swamp everything -- unless the two
  carriers are offset.

  Offset them with the VCO bias. At 25 MHz/V, 100 mV of DC offset separates
  the carriers by 2.5 MHz. Then:

    * both ramps have IDENTICAL slope and come off the SAME AD2 clock, so the
      cross-mix beat is constant at 2.5 MHz, not swept
    * 2.5 MHz is 100x above the 25 kHz Nyquist of a 50 kSa/s capture, so it
      never aliases into the digitised band -- it is simply not there
    * the wanted self-mix beat is unaffected: it depends on the module's own
      ramp, not on the other module's absolute frequency

  Cost: 100 mV out of the ~4 V drive, i.e. 2.5% of sweep bandwidth. Nothing.

  The alternative, time-multiplexing the two modules, also works and needs no
  offset -- but it halves the update rate and gives up the simultaneity that
  makes the two ranges refer to the same instant. Prefer the offset.

  Channel budget: one AD2 has exactly 2 analog inputs and 2 analog outputs.
  Two radars, I-only, is an exact fit -- W1/W2 drive the two VCOs, 1+/2+ take
  the two IF I pins. Wanting Q on both means a second AD2.""")


# ----------------------------------------------------------------- figures
def figures(out_dir=OUT):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    fs, nperseg = 50_000, 8192
    bin_hz = fs / nperseg
    tip_hz = drone.rpm_to_tip_hz(drone.RPM_HOVER)
    n_blade_bins = max(int((tip_hz - 4000) / bin_hz), 1)

    # --- figure 1: detection range vs IF gain -------------------------------
    gains = np.linspace(0, 45, 100)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    curves = [
        ("body, moving", drone.sigma_body_m2(), 800.0, 1, "tab:blue"),
        ("blade pedestal", drone.sigma_blades_m2(), tip_hz, n_blade_bins, "tab:red"),
        ("body, hovering", drone.sigma_body_m2(), 30.0, 1, "tab:cyan"),
    ]
    for name, sigma, freq, nb, col in curves:
        r = [klc6.detection_range_m(sigma, bin_hz, freq, if_gain_db=g,
                                    snr_req_db=14.0, n_spread_bins=nb) for g in gains]
        ax.plot(gains, r, color=col, lw=2, label=name)
    ax.axvline(0, color="k", ls=":", lw=1)
    ax.axvline(30, color="k", ls="--", lw=1)
    ax.text(30.4, ax.get_ylim()[1] * 0.55, "30 dB IF amp", rotation=90, fontsize=8)
    ax.set_xlabel("external IF gain ahead of the AD2 (dB)")
    ax.set_ylabel("5-sigma detection range (m)")
    ax.set_title("DJI Mini 3 detection range vs IF gain\n"
                 "K-LC6 + AD2, CW, 50 kSa/s, 6.1 Hz bins")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.savefig(f"{out_dir}/detection_vs_ifgain.png", dpi=130)
    plt.close(fig)

    # --- figure 2: GDOP map -------------------------------------------------
    rig = geometry.Rig(baseline_m=4.0)
    gx, gy = np.meshgrid(np.linspace(-8, 8, 400), np.linspace(0.3, 14, 400))
    pts = np.stack([gx, gy], axis=-1)
    g = rig.gdop(pts)
    fig, ax = plt.subplots(figsize=(7.5, 6), constrained_layout=True)
    im = ax.pcolormesh(gx, gy, np.clip(g, 0, 8), cmap="viridis_r", shading="auto")
    cs = ax.contour(gx, gy, g, levels=[1.5, 2, 3, 5], colors="w", linewidths=1)
    ax.clabel(cs, fmt="%.1f", fontsize=8)
    ax.plot(rig.positions[:, 0], rig.positions[:, 1], "r^", ms=11,
            label="K-LC6 modules")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Position dilution of precision, 4 m baseline\n"
                 "sigma_pos = GDOP x sigma_range")
    fig.colorbar(im, ax=ax, label="GDOP")
    ax.legend(loc="upper right")
    fig.savefig(f"{out_dir}/gdop_map.png", dpi=130)
    plt.close(fig)

    # --- figure 3: coverage, both mounting orientations ---------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    gx, gy = np.meshgrid(np.linspace(-9, 9, 500), np.linspace(0.1, 16, 500))
    pts = np.stack([gx, gy], axis=-1)
    cases = [("long axis VERTICAL\n80$^\\circ$ az x 12$^\\circ$ el, no toe-in",
              klc6.BEAM_WIDE_DEG, 0.0),
             ("long axis VERTICAL\n80$^\\circ$ az, 20$^\\circ$ toe-in",
              klc6.BEAM_WIDE_DEG, 20.0),
             ("long axis HORIZONTAL\n12$^\\circ$ az x 80$^\\circ$ el, 20$^\\circ$ toe-in",
              klc6.BEAM_NARROW_DEG, 20.0)]
    for ax, (title, az_bw, toe) in zip(axes, cases):
        rg = geometry.Rig(baseline_m=4.0, toe_in_deg=toe, az_beam_deg=az_bw,
                          el_beam_deg=(klc6.BEAM_NARROW_DEG
                                       if az_bw == klc6.BEAM_WIDE_DEG
                                       else klc6.BEAM_WIDE_DEG))
        both = rg.in_beam(pts)
        g = np.where(both, rg.gdop(pts), np.nan)
        ax.pcolormesh(gx, gy, np.clip(g, 0, 6), cmap="viridis_r", shading="auto")
        ax.contour(gx, gy, both.astype(float), levels=[0.5], colors="k",
                   linewidths=1.2)
        ax.plot(rg.positions[:, 0], rg.positions[:, 1], "r^", ms=10)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (m)")
        # where does the shared coverage begin on the centreline?
        col = both[:, np.argmin(np.abs(gx[0]))]
        ys = gy[:, 0][col]
        if ys.size:
            ax.axhline(ys.min(), color="k", ls="--", lw=1)
            ax.text(-8.6, ys.min() + 0.4, f"overlap starts {ys.min():.1f} m",
                    color="k", fontsize=9,
                    bbox=dict(fc="w", ec="k", lw=0.5, pad=1.5))
    axes[0].set_ylabel("y (m)")
    fig.suptitle("Where the two beams actually converge (4 m baseline). "
                 "Colour = GDOP inside the shared coverage.", fontsize=12)
    fig.savefig(f"{out_dir}/coverage.png", dpi=130)
    plt.close(fig)

    print(f"\n  figures -> {out_dir}/detection_vs_ifgain.png, "
          f"{out_dir}/gdop_map.png, {out_dir}/coverage.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true", help="self-checks only")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    validate()
    if args.validate:
        return
    target_summary()
    where_does_blade_energy_land()
    detection_table()
    geometry_table()
    interference_note()
    if not args.no_figures:
        figures()


if __name__ == "__main__":
    main()
