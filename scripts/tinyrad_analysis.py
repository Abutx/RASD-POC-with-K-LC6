#!/usr/bin/env python3
"""EV-TINYRAD24G: full analysis for the bench rig and the field rig BOMs.

Runs every number that feeds docs/TINYRAD_BENCH.md and docs/TINYRAD_FIELD.md.
Same discipline as demo_budget.py -- datasheet in, physics out, estimates
flagged -- applied to the two BOMs the way FINDINGS.md applies it to bench
claims: each claim in the BOM either survives the arithmetic or it does not.

  python scripts/tinyrad_analysis.py
  python scripts/tinyrad_analysis.py --field   # field-rig sections only
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim import drone, human, klc6, tinyrad as tr        # noqa: E402

RULE = "=" * 78


def h(t):
    print(f"\n{RULE}\n{t}\n{RULE}")


def claim(text, ok, detail=""):
    tag = "SURVIVES " if ok else "FAILS    "
    print(f"  [{tag}] {text}")
    if detail:
        for ln in detail.splitlines():
            print(f"             {ln}")


# =============================================================== BENCH RIG
def validation():
    h("B0. VALIDATION -- does the model reproduce ADI's own numbers?")
    uniq, counts = tr.virtual_array_mm()
    pitch = np.diff(uniq).mean()
    print(f"\n[1] virtual array from UG-1709 Table 2 antenna positions")
    print(f"    {len(uniq)} distinct positions, pitch {pitch:.3f} mm "
          f"(lambda/2 = {tr.LAMBDA*1e3/2:.3f} mm), multiplicity {counts.tolist()}")
    ok = (len(uniq) == 7 and abs(pitch - tr.LAMBDA * 1e3 / 2) < 0.05
          and sorted(counts.tolist()) == [1, 1, 1, 1, 1, 1, 2])
    print(f"    UG says: 'virtual, seven element array with a spacing of lambda/2.\n"
          f"    Two elements overlap.'  -> {'reproduced' if ok else 'NOT REPRODUCED -- check geometry'}")

    print(f"\n[2] ADI's reference measurement: 1 m^2 cube at 5 m -> '100 m for 1 m^2'")
    implied = tr.implied_snr_at_reference_db(14.0)
    model6 = tr.snr_db(tr.REF_RCS_M2, tr.REF_RANGE_M, tr.REF_N * tr.SAMPLE_PERIOD_S,
                       impl_loss_db=6.0)
    model25 = tr.snr_db(tr.REF_RCS_M2, tr.REF_RANGE_M, tr.REF_N * tr.SAMPLE_PERIOD_S,
                        impl_loss_db=25.0)
    print(f"    ADI's two numbers imply SNR at 5 m, one 256 us chirp: >= {implied:.0f} dB (at 5 sigma)")
    print(f"    model, 6 dB impl. loss:  {model6:.0f} dB    model, 25 dB impl. loss: {model25:.0f} dB")
    which = "6 dB" if abs(model6 - implied) < abs(model25 - implied) else "25 dB"
    print(f"    -> the {which} convention is the one consistent with ADI's own claim.")
    print(f"    (Their sweep was 23.95-24.25 GHz, 300 MHz -- 50 MHz outside the ISM\n"
          f"    band. Bench and field numbers here use 250 MHz.)")
    return which


def sensitivity():
    h("B1. SENSITIVITY -- does TinyRad fix the AD2 problem?")
    print(f"""
  K-LC6 + AD2 today: the ADC sits {klc6.sensitivity_penalty_db(7757.0, 0.0):.1f} dB above the module's
  own noise. TinyRad puts a 16-bit sigma-delta with 9-45 dB of LNA/PGA gain
  directly behind a receiver whose noise we know (NF {tr.RX_NF_DB:.0f} dB, gain {tr.RX_GAIN_DB:.0f} dB).
  ADAR7251 noise is 1/f (datasheet Table 1), so the penalty depends on where
  the signal sits in the IF: FMCW beats near 100 kHz, a CW body line near 100 Hz.
""")
    print(f"  {'PGA gain':>10}{'@100 kHz (FMCW)':>17}{'@1 kHz':>10}{'@100 Hz (CW body)':>19}")
    for g in [9, 15, 21, 27, 33, 39, 45]:
        p100k = tr.adc_penalty_db(g, 100e3)
        p1k = tr.adc_penalty_db(g, 1e3)
        p100 = tr.adc_penalty_db(g, 100.0)
        mark = "  <- ADI's reference config" if g == tr.REF_ADC_GAIN_DB else ""
        print(f"  {g:8d} dB{p100k:14.2f} dB{p1k:9.2f} dB{p100:15.2f} dB{mark}")
    print(f"""
  -> At the 21 dB ADI used for their own measurement the ADC costs 0.3 dB on
     an FMCW beat. Even at the worst corner (9 dB gain, 100 Hz) it is under
     5 dB. The 336 uV quantisation wall does not exist on this board. The
     BOM's "no IF amplifier needed" is correct.""")
    claim("BOM: 'IF amplifier not needed -- own IF chain with PGA'", True,
          f"ADC penalty {tr.adc_penalty_db(21, 100e3):.2f} dB at ADI's 21 dB setting, "
          f"vs 25.5 dB on the AD2.")


def link_budget():
    h("B2. LINK BUDGET -- DJI Mini 3, and the BOM's '30-50 m' claim")
    eirp = tr.eirp_dbm()
    print(f"""
  EIRP per Tx: {tr.TX_POWER_DBM:.0f} dBm + {tr.ANT_GAIN_DBI:.1f} dBi (UG-1709 Table 1) = {eirp:.1f} dBm
  K-LC6 was 18 dBm EIRP with 12.5 dBi, so TinyRad transmits ~2.6 dB MORE per
  channel and receives on 4 antennas instead of 1 (+6 dB coherent).
  Receiver: NF {tr.RX_NF_DB:.0f} dB -> {tr.receiver_noise_dbm_per_hz():.0f} dBm/Hz at the antenna.
  Implementation loss: both 6 dB (textbook) and 25 dB (RFbeam-style) shown;
  B0 says which one ADI's own measurement supports.
""")
    sig_body = drone.sigma_body_m2()
    sig_blade = drone.sigma_blades_m2()
    # The blade pedestal is smeared over ~+/-tip Doppler. In a 50 ms fast-chirp
    # dwell (N=32, 54 us) that is ~2 x 7757 Hz / 20 Hz ~ 775 Doppler bins. A
    # band-energy detector recovers only sqrt(N_bins) of that -- the same
    # 14 dB penalty klc6.snr_db charged the K-LC6. Charge it here too.
    n_spread = int(2 * drone.rpm_to_tip_hz(drone.RPM_HOVER) / 20.0)
    spread_db = 10 * np.log10(np.sqrt(n_spread))

    # Two derating conventions, both shown. 6 dB is the textbook allowance
    # (window, straddle, CFAR). 25 dB is what RFbeam's own K-LC6 range formula
    # carries (sim/klc6.py IMPL_MARGIN_DB) and is almost certainly what the
    # BOM author used to get 30-50 m. The bench corner-reflector test decides.
    print(f"  {'target':<26}{'T_coh':>7}{'6 dB loss':>11}{'25 dB loss':>12}{'BOM':>9}")
    print("  " + "-" * 67)
    rows = [
        ("Mini 3 body, moving", sig_body, 0.050, 0.0, "30-50 m"),
        ("Mini 3 body, moving", sig_body, 0.200, 0.0, ""),
        ("Mini 3 blades, spread", sig_blade, 0.050, spread_db, ""),
        ("Mini 3 blades, spread", sig_blade, 0.200, spread_db, ""),
        ("person, 1 m^2", 1.0, 0.050, 0.0, ""),
        ("car, 10 m^2", 10.0, 0.050, 0.0, "100 m"),
    ]
    out = {}
    for name, sigma, tcoh, extra, bom in rows:
        r6 = tr.detection_range_m(sigma, tcoh, impl_loss_db=6.0 + extra)
        r25 = tr.detection_range_m(sigma, tcoh, impl_loss_db=25.0 + extra)
        out[(name, tcoh)] = (r6, r25)
        print(f"  {name:<26}{tcoh*1e3:5.0f} ms{r6:9.0f} m{r25:10.0f} m{bom:>9}")

    r6, r25 = out[("Mini 3 body, moving", 0.050)]
    print(f"""
  Blade rows include a {spread_db:.0f} dB spread penalty ({n_spread} Doppler bins) and
  assume the fast-chirp mode in B3. In the default long-chirp frame the
  blades alias into the clutter notch (FINDINGS 8.3) -- unavailable at ANY range.

  The body row spans {r25:.0f}-{r6:.0f} m depending on the derating convention, on
  top of +/-5 dB body RCS (the antenna gain is now measured, not estimated).
  B0 shows ADI's own 100 m / 1 m^2 claim is consistent with the 6 dB end, so
  the upper figure is the better-supported one -- but it rests on a Mini 3
  RCS extrapolated from a Phantom 4. One corner reflector at 4 m and one
  drone at 5 m settle both.""")
    claim("BOM: 'range on small drone ~30-50 m'", r25 <= 50 <= r6,
          f"{r25:.0f} m with RFbeam-style 25 dB loss, {r6:.0f} m with textbook 6 dB. "
          f"ADI's own\nnumbers support the 6 dB convention, so the BOM is conservative "
          f"by ~2-3x.\nDefensible as a floor, not as an expectation.")
    r_car6, r_car25 = out[("car, 10 m^2", 0.050)]
    claim("BOM: '100 m range' (unspecified target)", r_car25 >= 100,
          f"car reaches {r_car25:.0f}-{r_car6:.0f} m by SNR; the WAVEFORM caps max range "
          f"lower -- see B3.")
    return out


def waveform():
    h("B3. WAVEFORM LIMITS -- the chirp floor vs blade Doppler")
    tip_hover = drone.rpm_to_tip_hz(drone.RPM_HOVER)
    tip_max = drone.rpm_to_tip_hz(drone.RPM_MAX)
    print(f"""
  UG-1709 p.13: sampling is FIXED at 1 MHz, and "N x tS + 20 us [must be]
  smaller than the PWM period." (EngineerZone quotes 22 us.) The PRF is
  therefore bounded by how many range samples you take, and the unambiguous
  Doppler is PRF/2. Mini 3 blade tips: {tip_hover:.0f} Hz hover, {tip_max:.0f} Hz max.
  Sweep 250 MHz (full ISM) unless noted.

  UG-1709 also says N is independent of ramp duration, and "in range doppler
  mode, Np and N are configurable, and arbitrary uniform range doppler
  measurements are possible." So the fast-chirp configs below ARE reachable
  from the stock API (fStart, fStop, tRampUp, Period, N, Seq, FrmSiz).
""")
    print(f"  {'N':>5}{'period':>9}{'PRF':>10}{'+/-Doppler':>12}{'+/-v':>9}"
          f"{'max range':>11}{'bins':>6}  blades?")
    print("  " + "-" * 78)
    for n in [8, 16, 24, 32, 48, 64, 128, 256, 512]:
        per = tr.min_chirp_period_s(n)
        prf = tr.max_prf_hz(n)
        dop = tr.max_unambiguous_doppler_hz(n)
        v = tr.max_unambiguous_velocity_mps(n)
        rmax = tr.max_range_m(250e6, n)
        bins = tr.range_bins(250e6, n)
        ok = ("hover+max" if dop >= tip_max else
              "hover only" if dop >= tip_hover else "NO -- aliases")
        print(f"  {n:5d}{per*1e6:8.0f}us{prf/1e3:8.1f}k{dop/1e3:10.2f}k{v:8.1f}"
              f"{rmax:10.1f} m{bins:6.0f}  {ok}")
    print(f"""
  THE TRADE. Blade micro-Doppler at hover needs PRF >= {2*tip_hover/1e3:.1f} kHz, i.e.
  N <= {int((1/(2*tip_hover) - tr.CHIRP_DEAD_TIME_S)/tr.SAMPLE_PERIOD_S)} samples per chirp -- which caps max range at about
  {tr.max_range_m(250e6, int((1/(2*tip_hover) - tr.CHIRP_DEAD_TIME_S)/tr.SAMPLE_PERIOD_S)):.0f} m. Full throttle needs N <= {int((1/(2*tip_max) - tr.CHIRP_DEAD_TIME_S)/tr.SAMPLE_PERIOD_S)}, max range ~{tr.max_range_m(250e6, max(int((1/(2*tip_max) - tr.CHIRP_DEAD_TIME_S)/tr.SAMPLE_PERIOD_S),1)):.0f} m.
  The {tr.CHIRP_DEAD_TIME_S*1e6:.0f} us dead time is the villain: at N=16 it is {100*tr.CHIRP_DEAD_TIME_S/tr.min_chirp_period_s(16):.0f}% of the period.

  This is SPEC.md 10.3 ("range OR blade classification, not both in one
  waveform") with real numbers attached. The production answer there --
  interleave a long-chirp detect frame with a short-chirp classify burst --
  is exactly what TinyRad's frame-based engine is built to do, IF the
  firmware lets you switch configs between frames. VERIFY on arrival.

  CW mode (park the PLL, stream the ADC) would sidestep all of this and give
  the K-LC6-style micro-Doppler at 1 MSPS. UG-1709 never mentions it; the
  ADF4159 is configured as a single sawtooth triggered by the DSP's PWM. The
  API exposes fStart and fStop -- setting them EQUAL is the obvious thing to
  try, and whether the PLL and the frame engine tolerate a zero-bandwidth
  ramp is the single most valuable question to answer in the first hour.""")
    claim("BOM: 'position from range, velocity, bearing in one unit'", True,
          "true for the body return in a long-chirp frame")
    n_hov = int((1 / (2 * tip_hover) - tr.CHIRP_DEAD_TIME_S) / tr.SAMPLE_PERIOD_S)
    claim("Implicit in BOM: blade micro-Doppler carries over unchanged", False,
          f"only in a fast-chirp mode with N <= {n_hov} and <= ~{tr.max_range_m(250e6, n_hov):.0f} m range, "
          f"or in CW\nmode if the firmware tolerates fStart == fStop. Neither is the default.")


def angle():
    h("B4. ANGLE -- what 4 Rx (and 2x4 MIMO) actually buys")
    print(f"""
  UG-1709 Table 2: Rx pitch 6.2175 mm = lambda/2, Tx pitch 18.654 mm = 3 lambda/2.
  Unambiguous FOV +/-{tr.unambiguous_fov_deg()/2:.0f} deg -- consistent with the 75 deg azimuth beam.
  The 2x4 MIMO forms a 7-ELEMENT virtual array with one overlapping position
  (B0 reproduces this from the coordinates). The overlap is deliberate --
  UG: "to allow the implementation of motion compensation algorithms".
""")
    print(f"  {'array':<28}{'3 dB beamwidth':>16}{'ADI claims':>12}")
    print(f"  {'4 Rx only':<28}{tr.array_beamwidth_deg(4):14.1f} deg{'':>12}")
    print(f"  {'2 Tx x 4 Rx = 7 virtual':<28}{tr.array_beamwidth_deg(7):14.1f} deg"
          f"{tr.ANGLE_RES_CLAIMED_DEG:10.0f} deg")
    print(f"""
  ADI's 20 deg is conservative against the 14.5 deg the geometry gives --
  amplitude taper on the six-patch elements and calibration residuals will
  eat some of the difference. Expect 15-20 deg measured.

  RESOLUTION (separating two targets) is the beamwidth. ACCURACY (locating
  one target) is far better and is what sets position error:
""")
    print(f"  {'range':>7}{'SNR':>6}{'sigma_theta 4el':>17}{'sigma_x 4el':>13}"
          f"{'sigma_theta 7el':>17}{'sigma_x 7el':>13}")
    for r in [2.0, 5.0, 10.0, 20.0, 30.0, 50.0]:
        for snr in [14.0, 24.0]:
            t4 = tr.angle_accuracy_deg(4, snr)
            t7 = tr.angle_accuracy_deg(7, snr)
            print(f"  {r:5.0f} m{snr:5.0f}dB{t4:14.2f} deg{r*np.radians(t4):11.2f} m"
                  f"{t7:14.2f} deg{r*np.radians(t7):11.2f} m")
    print(f"""
  Compare the two-K-LC6 baseline rig: sigma_pos 0.35 m at 2 m, 0.75 m at 8 m,
  degrading as range^2. TinyRad's cross-range error grows only LINEARLY with
  range and stays under 0.5 m to ~30 m at decent SNR. Down-range error is the
  0.6 m cell / sqrt(2 SNR) -- a few cm. **Single-unit position beats the
  two-module baseline beyond ~5 m and never needs a second radar.**

  What it cannot do that the baseline rig could: nothing. What the baseline
  rig could not do that this can: separate two targets at the same range.""")
    claim("BOM: 'position is polar-to-Cartesian from a single radar'", True,
          f"sigma_x ~{30*np.radians(tr.angle_accuracy_deg(7, 20.0)):.2f} m at 30 m "
          f"(7 el, 20 dB); the K-LC6 pair could not reach 30 m at all.")
    claim("BOM: 'angle resolution ~20 deg'", True,
          f"4-el gives {tr.array_beamwidth_deg(4):.0f}, 7-el gives "
          f"{tr.array_beamwidth_deg(7):.0f}; 20 is ADI's conservative figure.")


def data_and_rate():
    h("B5. UPDATE RATE, DATA RATE, COMPUTE")
    n, nchirp = 256, 128                      # a plausible detect-frame config
    frame_s = nchirp * tr.min_chirp_period_s(n)
    raw_bps = tr.N_RX * 2 * n * nchirp / frame_s
    print(f"""
  Detect frame: N={n}, {nchirp} chirps -> {frame_s*1e3:.1f} ms/frame, {1/frame_s:.0f} Hz max
  (ADI: 50 ms frame -> 20 Hz; the USB/DSP path caps it there in practice).
  Raw ADC per frame: {tr.N_RX}ch x 16b x {n} x {nchirp} = {tr.N_RX*2*n*nchirp/1e6:.2f} MB
  -> {raw_bps/1e6:.1f} MB/s if streamed continuously at max rate.

  Range-Doppler-angle per frame: {tr.N_RX} x {nchirp} x {n}-pt range FFT, then
  {n//2} x {tr.N_RX} x {nchirp}-pt Doppler FFT, then 8-pt angle FFT per cell:
  ~{(tr.N_RX*nchirp*5*n*np.log2(n) + (n//2)*tr.N_RX*5*nchirp*np.log2(nchirp) + (n//2)*nchirp*5*8*3)/1e6:.0f} MFLOP/frame -> {(tr.N_RX*nchirp*5*n*np.log2(n) + (n//2)*tr.N_RX*5*nchirp*np.log2(nchirp) + (n//2)*nchirp*5*8*3)*20/1e9:.2f} GFLOP/s at 20 Hz. A Pi 5 does ~30 GFLOP/s in numpy.""")
    claim("BOM (field): 'range-Doppler-angle at a few fps is fine on a Pi 5'", True,
          "an order of magnitude of headroom even at 20 Hz")
    claim("BOM (field): 'push detections, buffer raw locally'", True,
          f"raw is {raw_bps/1e6:.0f} MB/s; a school uplink will not take that.")


def carryover():
    h("B6. WHAT CARRIES OVER -- the BOM's table, checked")
    rows = [
        ("process.py spectrogram, CVD", True,
         "unchanged -- but only useful on fast-chirp or CW data (B3)"),
        ("OS-CFAR", True, "unchanged, 2D on range-Doppler; add angle dimension"),
        ("spur map, background", True, "redo -- new board, new floor. Correct."),
        ("1D Kalman -> (x, y, vx, vy)", True,
         "correct; measurement is now (range, bearing) -> polar Jacobian"),
        ("classifier retrain", True, "correct, and see B7"),
        ("dataset schema + bearing_deg", True, "also add n_rx, n_tx, pga_db, chirp cfg"),
    ]
    for name, ok, note in rows:
        claim(f"BOM: '{name}'", ok, note)
    print("""
  MISSING FROM THE BOM, and it matters:
    * sim/ -- the whole simulator carries over. drone.py, human.py and the
      geometry are front-end-independent; only klc6.py is replaced by
      tinyrad.py. Synthetic captures for the new board are a one-line change.
    * The measured empty-room floor does NOT carry over. Nor does the mains
      comb -- TinyRad is USB-powered from a laptop and has its own supply
      regulation, so the 60 Hz picture will be different (probably better,
      possibly with USB switching spurs instead). Re-measure on day one.
    * The K-LC6's 25 dB RFbeam derating does not transfer -- B0 shows ADI's
      own 100 m claim is consistent with ~6 dB, not 25. If the corner
      reflector at 4 m lands more than ~6 dB below the 6 dB-loss prediction,
      something is off (NF, cable, calibration) -- measure it, don't argue.""")


def classification_carryover():
    h("B7. CLASSIFICATION ON TINYRAD")
    print(f"""
  The layered classifier from DEMO_TWO_RADAR.md section 9 survives intact, and
  gets a fifth layer for free:

    1. gait cadence on the bulk line          -- any frame type, full range
    2. kinematics (hover_frac, v_var)          -- now in 2D, much stronger
    3. RCS from range + amplitude              -- full range
    4. blade micro-Doppler                     -- fast-chirp frames only, ~12 m
    5. NEW: angular extent                     -- a car is 10+ deg wide at 20 m;
                                                  a drone or a person is a point

  Blade detection range on TinyRad, fast-chirp N=32, 50 ms coherent, with the
  same 14 dB spread penalty B2 charges:
    {tr.detection_range_m(drone.sigma_blades_m2(), 0.050, impl_loss_db=6.0 + 10*np.log10(np.sqrt(775))):.1f} m (6 dB loss) / {tr.detection_range_m(drone.sigma_blades_m2(), 0.050, impl_loss_db=25.0 + 10*np.log10(np.sqrt(775))):.1f} m (25 dB loss)
    vs {klc6.detection_range_m(drone.sigma_blades_m2(), 6.1, 7757.0, if_gain_db=30.0, snr_req_db=14.0, n_spread_bins=615):.1f} m for K-LC6 + 30 dB amp
  and the waveform caps it at {tr.max_range_m(250e6, 32):.1f} m regardless. On TinyRad the
  link and the chirp floor land in the same place for blades, ~4-10 m. That
  is 3-6x the K-LC6-with-amp figure and still short of demo distances.""")


# =============================================================== FIELD RIG
def regulatory():
    h("F1. REGULATORY -- 47 CFR 15.245 vs 15.249 at 24 GHz")
    e_tiny = tr.eirp_dbm()
    l249 = tr.field_to_eirp_dbm(tr.FCC_15249_FIELD_V_M)
    l245 = tr.field_to_eirp_dbm(tr.FCC_15245_FIELD_V_M)
    print(f"""
  TinyRad EIRP per Tx (8 dBm + 12.6 dBi): {e_tiny:+.1f} dBm
  15.249 general, 24.0-24.25 GHz:     250 mV/m @ 3 m = {l249:+.1f} dBm EIRP  -> OVER by {e_tiny-l249:.0f} dB
  15.245 field-disturbance sensor:   2500 mV/m @ 3 m = {l245:+.1f} dBm EIRP  -> under by {l245-e_tiny:.0f} dB
      ...but 15.245 is 24.075-24.175 GHz ONLY: {(tr.FCC_15245_BAND_HZ[1]-tr.FCC_15245_BAND_HZ[0])/1e6:.0f} MHz, not 250.

  Consequence: to operate as a 15.245 sensor the sweep must stay inside
  24.075-24.175 GHz. Range resolution goes from {tr.range_resolution_m(250e6):.2f} m to
  {tr.range_resolution_m(100e6):.2f} m. Everything in B3 built on 250 MHz shifts by 2.5x
  in range cell size (max range for a given N goes UP by 2.5x, which helps).

  This is an EVALUATION BOARD with no FCC ID of its own. Indoor bench use is
  the normal engineering case. A fixed outdoor installation radiating over a
  school pickup lane is a different thing, and someone should make a
  deliberate decision about it before install day -- this is an engineering
  reading of the rules, not legal advice.""")
    claim("BOM (field): silent on regulatory", False,
          "add a line item: decide 15.245 (100 MHz, 1.5 m res) vs seek\n"
          "authorisation; do not default to the full 250 MHz sweep outdoors.")


def radome():
    h("F2. RADOME -- polycarbonate at 24 GHz")
    hw = tr.half_wave_thickness_mm()
    print(f"""
  Half-wave thickness in polycarbonate (eps_r 2.75): {hw:.2f} mm -- the
  reflectionless thickness. One-way loss vs thickness (normal incidence):
""")
    print(f"  {'material':<22}{'t (mm)':>8}{'one-way':>10}{'two-way':>10}")
    for name, t, er, td in [("polycarbonate", 2.0, 2.75, 0.008),
                            ("polycarbonate (BOM)", 3.0, 2.75, 0.008),
                            ("polycarbonate", hw, 2.75, 0.008),
                            ("polycarbonate", 4.5, 2.75, 0.008),
                            ("acrylic (PMMA)", 3.0, 2.6, 0.02),
                            ("window glass", 3.0, 6.0, 0.02),
                            ("ABS (the box wall!)", 2.5, 2.8, 0.01)]:
        l = tr.radome_loss_db(t * 1e-3, er, td)
        print(f"  {name:<22}{t:8.2f}{l:9.2f} dB{2*l:9.2f} dB")
    print(f"""
  3 mm polycarbonate is a good choice: {2*tr.radome_loss_db(3e-3):.2f} dB two-way. Moving to
  {hw:.1f} mm would make it {2*tr.radome_loss_db(hw*1e-3):.2f} dB, which is not worth chasing.
  The BOM's "not acrylic" is over-cautious -- 3 mm PMMA costs {2*tr.radome_loss_db(3e-3, 2.6, 0.02):.2f} dB
  two-way -- but "not glass" is right ({2*tr.radome_loss_db(3e-3, 6.0, 0.02):.1f} dB).

  What the BOM misses: WATER. A 0.5 mm film of rain on the window costs
  several dB at 24 GHz and a wet radome is the single largest weather loss
  outdoors, far above rain in the path. Tilt the window >= 15 deg from
  vertical, extend the enclosure lid as a drip shield, and consider a
  hydrophobic coating. Rain in the 50 m path itself:""")
    for rate in [5, 25, 50]:
        print(f"    {rate:3d} mm/hr: {tr.rain_loss_db_per_km(rate):.2f} dB/km "
              f"-> {2*0.05*tr.rain_loss_db_per_km(rate):.3f} dB two-way at 50 m. Negligible.")
    claim("BOM (field): '3 mm polycarbonate, not acrylic, not glass'", True,
          "polycarbonate correct; acrylic would also have been fine; glass is not")
    claim("BOM (field): radome guidance covers wet-window loss", False,
          "add: tilt, drip shield, hydrophobic coating. Biggest outdoor loss.")


def site_geometry():
    h("F3. SITE GEOMETRY -- the vehicle placement vs the drone placement")
    el = tr.BEAM_EL_DEG
    print(f"""
  BOM: radar at 2-3 m height, tilted slightly DOWN toward the pickup lane.
  Elevation beam is {el:.0f} deg (3 dB). That is a thin sheet, exactly like the
  K-LC6 mounted for wide azimuth:
""")
    print(f"  {'horiz. range':>13}{'sheet height':>14}   drone must be within")
    for r in [10, 20, 30, 50]:
        hgt = 2 * r * np.tan(np.radians(el / 2))
        print(f"  {r:11.0f} m{hgt:12.1f} m   +/-{hgt/2:.1f} m of radar height")
    print(f"""
  A drone at 15-30 m altitude over a school is OUTSIDE the elevation beam
  entirely from a 2.5 m mount tilted down. And if you tilt UP to catch it,
  two things happen: (a) the pickup lane leaves the beam, and (b) you are
  now looking at the rotor disc from below -- the cos(elevation) null from
  DEMO_TWO_RADAR.md section 4. From 25 m altitude at 30 m horizontal range
  the radar sees the drone at {np.degrees(np.arctan(25/30)):.0f} deg elevation: blade energy above
  4 kHz drops to about {0.3:.0f}-{15:.0f}% of the level-view value (measured in the sim
  at 30-60 deg).

  THE VEHICLE-COUNTING PLACEMENT AND THE DRONE-CAPTURE PLACEMENT ARE
  INCOMPATIBLE WITH ONE UNIT. The BOM's "drone captures happen wherever they
  fly -- coordinate a schedule" hides this. Either:
    (a) two units, one per job -- the BOM already allows "a second one";
    (b) one unit, and the drone team flies IN THE SHEET: low, level with the
        radar, across the lane when it is empty. Useful for the dataset,
        not representative of the product's overhead-drone case;
    (c) mount high (roof edge, ~8-10 m) tilted down at the lane; drones
        crossing at 5-15 m altitude then pass through the sheet with the
        radar roughly LEVEL with them. This is the only geometry that
        serves both, and the BOM's pole mount does not reach it.""")
    claim("BOM (field): one unit at 2-3 m serves vehicles AND drone captures", False,
          "15 deg elevation sheet + cos(el) null. Pick (a), (b) or (c) above.")


def vehicle_counting():
    h("F4. VEHICLE COUNTING -- the job the placement is actually good for")
    for r in [20, 30, 50]:
        cell = r * np.radians(tr.ANGLE_RES_CLAIMED_DEG)
        print(f"  at {r:2d} m: angle cell {cell:5.1f} m wide, range cell "
              f"{tr.range_resolution_m(100e6):.1f} m (15.245) / {tr.range_resolution_m(250e6):.1f} m (ISM)")
    print(f"""
  Bearing cannot separate two cars side by side at 30 m (cell {30*np.radians(20):.0f} m wide).
  Range can. A single-file pickup lane is a 1D problem and 0.6-1.5 m of
  range resolution resolves cars 4-5 m long with margin. Car RCS >= 10 m^2
  detects at {tr.detection_range_m(10.0, 0.050):.0f}+ m by SNR -- the waveform's max range is the limit,
  not the link. At 20 Hz and 5 m/s a car moves 0.25 m per frame: the tracker
  will have 20+ hits per car. Fine.

  Confusers here are pedestrians (gait, layer 1) and -- new -- BIRDS. K-band
  bird RCS is -20 to -30 dBsm, wingbeat 2-10 Hz. A bird and a Mini 3 have
  similar RCS; the wingbeat vs blade-pass cadence (10 Hz vs 200+ Hz) is the
  separator, and it needs the fast-chirp frame (B3) to see it.""")
    claim("BOM (field): '1D range plus bearing is plenty' for the lane", True,
          "range does the work; bearing is too coarse to matter at 30 m")


def power_and_thermal():
    h("F5. POWER, PoE, ENCLOSURE THERMAL")
    p_tiny = tr.POWER_W                   # UG-1709 Table 3: 5 V, 780 mA, all Rx on
    p_pi = 6.0
    print(f"""
  TinyRad: UG-1709 Table 3, 5 V x 780 mA = {p_tiny:.1f} W with all four Rx enabled.
  Pi 5 under load ~{p_pi:.0f} W. Total ~{p_tiny+p_pi:.0f} W. 802.3at delivers 25.5 W at the PD.
  Note 780 mA is above a USB 2.0 port's 500 mA -- the Pi 5's USB-C/USB 3
  ports supply 1.6 A shared, so budget for it and do not hang anything else
  off that bus.""")
    claim("BOM (field): 'TinyRad + Pi 5 under 15 W, 802.3at comfortable'", True,
          f"~{p_tiny+p_pi:.0f} W estimated; ~15 W margin")
    print(f"""
  Thermal: ~{p_tiny+p_pi:.0f} W inside a sealed IP66 ABS box in Texas sun. A 250x200x100
  box has ~0.19 m^2 surface; at ~5 W/m^2/K natural convection that is ~{(p_tiny+p_pi)/(0.19*5):.0f} K
  rise over ambient BEFORE solar gain, which on a dark box can add 15-25 K.
  A Pi 5 throttles at 85 C. On a 40 C day the inside could reach 70-80 C.""")
    claim("BOM (field): enclosure thermal design", False,
          "add: light-coloured box, sun shield, Pi 5 heatsink or active cooler,\n"
          "temperature in the heartbeat. Desiccant handles humidity, not heat.\n"
          f"TinyRad carries an {tr.ONBOARD_TEMP_SENSOR} temperature sensor on its own PCB\n"
          "(UG-1709 BOM, U12) -- read it too.")


def edge_linux():
    h("F6. THE LINUX QUESTION -- now answerable from the user guide")
    print(f"""
  UG-1709 p.1: EQUIPMENT NEEDED -- "PC with Windows 7 (or more recent version)".
  UG-1709 p.13: the Python tree contains a DLL directory holding "{tr.PYTHON_USB_LAYER}";
  the MATLAB tree uses a "{tr.MATLAB_USB_LAYER}". The board enumerates as
  "{tr.USB_DEVICE_NAME}" and needs a manual driver install from Demo_Driver.zip.

  So: THE STOCK PYTHON LIBRARY IS WINDOWS-ONLY AS SHIPPED. It calls into a
  Windows DLL. There is no Linux path in the box.

  Two ways forward, in order of risk:
    (a) x86 mini-PC running Windows in the enclosure. Works day one. +$100-120
        over a Pi 5, ~10 W more, and a 12 V PoE splitter.
    (b) Re-implement the usb.dll transport over libusb/pyusb. The device is a
        plain bulk endpoint, the command set is whatever the Python class
        sends, and the Blackfin firmware does not care what OS is on the other
        end. Feasible; a few days if the protocol is simple, longer if not.
        Check EngineerZone for prior art before starting.

  The BOM's sequencing stands: do not order field compute until the bench rig
  has settled this. But go in expecting (a), and treat (b) as the upgrade.""")
    claim("BOM (field): 'verify Linux before ordering the edge computer'", True,
          "correct -- and the user guide says to expect the answer to be 'no'.")
    claim("BOM (field): 'Raspberry Pi 5 if Python runs on Linux'", False,
          "it does not, as shipped. Plan the x86 path; treat a libusb port as\n"
          "a project, not a checkbox.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", action="store_true")
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    do_bench = a.bench or not a.field
    do_field = a.field or not a.bench
    if do_bench:
        validation()
        sensitivity()
        link_budget()
        waveform()
        angle()
        data_and_rate()
        carryover()
        classification_carryover()
    if do_field:
        regulatory()
        radome()
        site_geometry()
        vehicle_counting()
        power_and_thermal()
        edge_linux()


if __name__ == "__main__":
    main()
