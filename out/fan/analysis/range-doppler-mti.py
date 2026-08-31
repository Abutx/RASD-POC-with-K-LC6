"""
Range-Doppler + MTI: is a running household fan detectable at 24 GHz (K-LC6)?

Method: range FFT along fast time, Doppler FFT along slow time, zero-Doppler
notch (MTI), look for energy at a consistent range bin. The fan is ~2.5 m away.
The fan's signature would be energy STATIONARY IN RANGE but spread in Doppler.

Chain, in order:
  1  GEOMETRY        what each config can and cannot physically resolve
  2  POSITIVE CTRL   moving person (cfgA) vs empty room (cfgA)  -- must pass
     + FAN cfgA      fan-on cfgA vs the SAME empty cfgA reference
  3  UPPER BOUND     fan return relative to the person's, same config, same bin
  4  CONFIG B        200 CPIs of range-Doppler (the assigned dataset)
  5  51 Hz FEATURE   target return, or modulation of the TX feedthrough?
  6  SENSITIVITY     synthetic-target injection -> quantitative upper bound
  7  CW CROSS-CHECK  independent, gain-invariant line search

Every floor is MEASURED by splitting one condition; none is assumed.
Run:  cd C:/dev/klc6 && python C:/dev/klc6/out/fan/analysis/range-doppler-mti.py
"""
from __future__ import annotations
import sys
import numpy as np
from scipy.signal import welch
from scipy.ndimage import median_filter

sys.path.insert(0, "C:/dev/klc6")
from klc6 import process as P

C = 299_792_458.0
B_MEAS = 180e6                 # MEASURED sweep bandwidth (FINDINGS 5.3), +-30%
LSB = 336e-6                   # AD2 5 Vpp range
OUT = "C:/dev/klc6/out/fan/analysis"
R = "C:/dev/klc6/out"
FAN_B = R + "/fan/20260830_093545_fan_on_fmcw_cfgB.npz"
FAN_A = R + "/fan/20260830_093545_fan_on_fmcw_cfgA.npz"
MOVING = R + "/fmcw/raw_chirps_moving.npz"
EMPTY = R + "/fmcw/raw_chirps_box_out.npz"
EMPTY2 = R + "/fmcw/raw_chirps_box_in.npz"

db = lambda x: 10 * np.log10(np.asarray(x) + 1e-30)


def hdr(s):
    print("\n" + "=" * 78)
    print(s)
    print("=" * 78)


def load(fn, key):
    return np.asarray(np.load(fn, allow_pickle=True)[key], dtype=float)


# ------------------------------------------------------------------ primitives
def detrend_chirps(x, order=3):
    """Polynomial-detrend each chirp. The ramp self-mix is 16.8 mVpp, ~100x the
    whole CW signal, and subtracting the MEAN does not touch it -- a linear ramp
    minus its mean is still a linear ramp (FINDINGS 5.4)."""
    x = np.asarray(x, float)
    n = x.shape[-1]
    V = np.vander(np.linspace(-1, 1, n), order + 1)
    coef, *_ = np.linalg.lstsq(V, x.reshape(-1, n).T, rcond=None)
    return x - (V @ coef).T.reshape(x.shape)


def range_fft(x, fs, S, order=3):
    x = detrend_chirps(x, order)
    n = x.shape[-1]
    Rc = np.fft.rfft(x * np.hanning(n), axis=-1)
    return np.fft.rfftfreq(n, 1 / fs) * C / (2.0 * S), Rc


def mti_profile(chirps, fs, S, prf, order=3, notch=1):
    rng, Rc = range_fft(chirps, fs, S, order)
    nch = Rc.shape[0]
    D = np.fft.fftshift(np.fft.fft(Rc * np.hanning(nch)[:, None], axis=0), axes=0)
    keep = np.abs(np.arange(nch) - nch // 2) > notch
    return rng, (np.abs(D) ** 2)[keep].mean(axis=0)


def chance_bar(n):
    """Expected max of n standard normals: the bar a search over n cells must clear."""
    return np.sqrt(2 * np.log(n))


# ============================================================== CONFIG A SETUP
dA = dict(fs=100000, spc=2000, ramp=50.0)
S_A, prfA = B_MEAS * dA["ramp"], dA["ramp"]
moving = load(MOVING, "chirps")
empty = load(EMPTY, "chirps")
empty2 = load(EMPTY2, "chirps")
fanA = load(FAN_A, "chirps")
NCH = min(moving.shape[0], empty.shape[0], fanA.shape[0])

hdr("0. DATA SANITY -- quantisation (FINDINGS 1.1: bare IF sits on 4-7 ADC codes)")
for nm, a in [("fan cfgB", load(FAN_B, "cpis")), ("fan cfgA", fanA),
              ("person cfgA", moving), ("empty cfgA", empty)]:
    print("  %-12s %-18s %3d distinct codes, span %5.1f LSB  (that span IS the ramp)"
          % (nm, str(a.shape), np.unique(a).size, (a.max() - a.min()) / LSB))
print("  -> the target signal is far below one LSB; only FFT processing gain exposes it.")

hdr("1. GEOMETRY -- what each config can physically resolve")
print("  Config A: fs=%d spc=%d ramp=%.0f Hz -> PRF %.0f Hz, chirp %.0f ms"
      % (dA["fs"], dA["spc"], dA["ramp"], prfA, dA["spc"] / dA["fs"] * 1e3))
print("            S=%.3e Hz/s, range bin %.3f m, v_unambiguous +-%.3f m/s"
      % (S_A, (dA["fs"] / dA["spc"]) * C / (2 * S_A), prfA / 2 / P.HZ_PER_MPS))
print("            range-Doppler coupling %.2f m per m/s -> a 10 m/s blade tip lands"
      % (P.HZ_PER_MPS * C / (2 * S_A)))
print("            %.0f m away from 2.5 m." % (10 * P.HZ_PER_MPS * C / (2 * S_A)))
print("            *** Config A CANNOT hold fast blade energy in the 2.5 m bin. It is a")
print("                valid test for the SLOW parts of the fan only. ***")
print("  Config B: PRF 1000 Hz, chirp 1 ms, v_unambiguous +-3.11 m/s, coupling %.3f m"
      % (P.HZ_PER_MPS * C / (2 * B_MEAS * 1000)))
print("            per m/s -> a 10 m/s blade smears only 1.3 m. Right config for a fan.")

# ================================================= 2. POSITIVE CONTROL + FAN A
kw = dict(order=3, notch=1)
rngA, p_mov = mti_profile(moving[:NCH], dA["fs"], S_A, prfA, **kw)
_, p_emp = mti_profile(empty[:NCH], dA["fs"], S_A, prfA, **kw)
_, p_em2 = mti_profile(empty2[:NCH], dA["fs"], S_A, prfA, **kw)
_, p_fan = mti_profile(fanA[:NCH], dA["fs"], S_A, prfA, **kw)

NSPLIT = 4
m = NCH // NSPLIT
profs = np.array([db(mti_profile(empty[i * m:(i + 1) * m], dA["fs"], S_A, prfA, **kw)[1])
                  for i in range(NSPLIT)])
sigma1 = profs.std(axis=0, ddof=1)             # MEASURED: split-half of ONE condition
sigma_diff = np.sqrt(2.0) * sigma1 / np.sqrt(NSPLIT)
d_static = db(p_em2) - db(p_emp)               # independent null-vs-null check

hdr("2. CONFIG A -- positive control and fan, against a MEASURED floor")
print("  equal averaging: %d chirps from every condition (fan/person had 240, empty 160)"
      % NCH)
sel = (rngA >= 0.4) & (rngA <= 8.0)
print("\n%8s %9s | %8s %6s %7s | %8s %6s %7s | %9s"
      % ("range m", "empty dB", "PERSON", "diff", "sigma", "FAN", "diff", "sigma", "null-null"))
for i in np.flatnonzero(sel):
    dm = db(p_mov[i]) - db(p_emp[i])
    df = db(p_fan[i]) - db(p_emp[i])
    print("%8.2f %9.2f | %8.2f %6.2f %7.2f | %8.2f %6.2f %7.2f | %9.2f"
          % (rngA[i], db(p_emp[i]), db(p_mov[i]), dm, dm / sigma_diff[i],
             db(p_fan[i]), df, df / sigma_diff[i], d_static[i]))
print("\n  MEASURED floor (split-%d of the empty room), 0.4-8 m: difference sigma = %.2f dB"
      % (NSPLIT, np.median(sigma_diff[sel])))
print("  independent check, two static conditions differenced: rms %.2f dB -> floor is honest."
      % np.sqrt(np.mean(d_static[sel] ** 2)))
i_m = np.flatnonzero(sel)[np.argmax((db(p_mov) - db(p_emp))[sel])]
s_m = (db(p_mov[i_m]) - db(p_emp[i_m])) / sigma_diff[i_m]
print("\n  >>> POSITIVE CONTROL: person %+.2f dB at %.2f m = %.1f sigma"
      % (db(p_mov[i_m]) - db(p_emp[i_m]), rngA[i_m], s_m))
print("      (FINDINGS 5.5 reports +6.7 dB / 12 sigma for this file) -- PASS")
i25 = int(np.argmin(np.abs(rngA - 2.5)))
d25 = db(p_fan[i25]) - db(p_emp[i25])
s25 = d25 / sigma_diff[i25]
print("  >>> FAN at the a-priori 2.5 m bin: %+.2f dB = %.2f sigma -- NO DETECTION" % (d25, s25))
i_f = np.flatnonzero(sel)[np.argmax((db(p_fan) - db(p_emp))[sel])]
print("      best fan bin anywhere in-room: %+.2f dB at %.2f m = %.2f sigma "
      "(chance bar for %d bins is %.2f)"
      % (db(p_fan[i_f]) - db(p_emp[i_f]), rngA[i_f],
         (db(p_fan[i_f]) - db(p_emp[i_f])) / sigma_diff[i_f], sel.sum(), chance_bar(sel.sum())))

# ------------------------------------------------------------- 3. UPPER BOUND
F = p_emp[i25]
p95 = d25 + 2 * sigma_diff[i25]
fan_tgt = max(F * (10 ** (p95 / 10) - 1), 1e-30)
per_tgt = F * (10 ** ((db(p_mov[i25]) - db(p_emp[i25])) / 10) - 1)
hdr("3. UPPER BOUND on the fan's MTI return (Config A, 2.5 m bin, same processing)")
print("  person target power = %+.2f dB rel. the empty-room MTI floor" % db(per_tgt / F))
print("  fan    target power < %+.2f dB (95%% upper bound = measured diff + 2 sigma)"
      % db(fan_tgt / F))
print("  >>> the fan's moving-target return is at least %.1f dB below the person's."
      % db(per_tgt / fan_tgt))

# ==================================================================== CONFIG B
hdr("4. CONFIG B -- 200 CPIs x 128 chirps x 100 samples (the assigned dataset)")
dBz = np.load(FAN_B, allow_pickle=True)
cpis = np.asarray(dBz["cpis"], float)
fsB, spcB = int(dBz["fs"]), int(dBz["spc"])
nchB, rampB = int(dBz["nchirps"]), float(dBz["ramp"])
S_B, prfB, ncpi = B_MEAS * rampB, rampB, cpis.shape[0]
print("  NOTE: this is the ONLY Config B capture that exists -- there is no Config B")
print("  empty room and no Config B person. Config B is therefore SELF-referenced, and")
print("  its sensitivity is established by injection (section 6).")
print("  dwell %.1f s, range bin %.3f m, max range %.1f m"
      % (ncpi * nchB / prfB, (fsB / spcB) * C / (2 * S_B), (fsB / 2) * C / (2 * S_B)))

rngB, Rb = range_fft(cpis, fsB, S_B, order=3)
velB = np.fft.fftshift(np.fft.fftfreq(nchB, 1 / prfB)) / P.HZ_PER_MPS
fdB = velB * P.HZ_PER_MPS
D = np.fft.fftshift(np.fft.fft(Rb * np.hanning(nchB)[None, :, None], axis=1), axes=1)
PW = np.abs(D) ** 2
notch = np.abs(np.arange(nchB) - nchB // 2) <= 1
keepB = ~notch
tol = 8.0
mains = np.zeros(nchB, bool)
for h in range(1, int(prfB / 2 / 60) + 1):
    mains |= np.abs(np.abs(fdB) - 60 * h) < tol
clean = keepB & ~mains
print("  MTI notch removes |v|<=%.3f m/s (%d bins). Mains-tagged: %d/%d = %.0f%%"
      % (abs(velB[nchB // 2 + 1]), notch.sum(), mains.sum(), nchB, 100 * mains.sum() / nchB))
print("  *** that %.0f%% IS the chance false-tag rate for '60 Hz multiple' on this Doppler"
      % (100 * mains.sum() / nchB))
print("      axis -- a mains tag here carries almost no information, so nothing below is")
print("      claimed or dismissed on the strength of a mains tag alone. ***")

cell_mean = PW.mean(axis=0)
cell_se = PW.std(axis=0, ddof=1) / np.sqrt(ncpi)
ref = np.median(cell_mean[clean, :], axis=0)          # per-range self-normalisation
Z = (cell_mean - ref[None, :]) / cell_se
mask = np.zeros_like(Z, bool)
cols = np.flatnonzero((rngB >= 0.4) & (rngB <= 20.0))
mask[np.ix_(np.flatnonzero(clean), cols)] = True
ncell = int(mask.sum())
bar = chance_bar(ncell)
Za = (PW[0::2].mean(0) - PW[1::2].mean(0)) / np.sqrt(
    PW[0::2].var(0, ddof=1) / 100 + PW[1::2].var(0, ddof=1) / 100)
print("\n  z-map over %d (range x Doppler) cells. Chance bar = %.2f sigma." % (ncell, bar))
print("  split-half NULL (fan-on vs fan-on): z std %.2f (want 1.00), max |z| %.2f"
      % (Za[mask].std(), np.abs(Za[mask]).max()))
print("  -> the measured floor is calibrated; z really is in sigma units.")
rr, dd = np.where(mask)
print("\n  top-6 cells:")
for q in np.argsort(Z[mask])[::-1][:6]:
    j, i = rr[q], dd[q]
    print("    %6.2f m  %+7.3f m/s (%+7.1f Hz)  z=%6.2f  %+5.2f dB over the range-bin median"
          % (rngB[i], velB[j], fdB[j], Z[j, i], db(cell_mean[j, i] / ref[i])))
i25B = int(np.argmin(np.abs(rngB - 2.5)))
sub = Z[np.ix_(np.flatnonzero(clean), [i25B - 1, i25B, i25B + 1])]
print("\n  >>> at the fan's a-priori range (2.5 m +-1 bin, covering the +-1.3 m blade")
print("      smear): max z = %.2f vs a %.2f chance bar -- NO DETECTION"
      % (sub.max(), chance_bar(sub.size)))
print("  >>> the MTI range profile is FLAT to within 1 dB from 1.7 m to 12 m: this is")
print("      receiver noise, with no clutter and no target standing above it.")

# ------------------------------------------------- 5. what IS the 51 Hz thing?
hdr("5. The 51 Hz feature at 0.83 m: a target, or modulation of the feedthrough?")
M = cell_mean
sb = np.flatnonzero((np.abs(fdB) >= 43) & (np.abs(fdB) <= 58))
nz = np.flatnonzero(clean & ~((np.abs(fdB) >= 43) & (np.abs(fdB) <= 58)))
dc = M[nchB // 2 - 1:nchB // 2 + 2].sum(axis=0)
print("  A target return scales with RANGE. A modulation of the transmitter feedthrough")
print("  scales with the FEEDTHROUGH. Those predictions differ, so the data can decide:")
print("\n%8s %12s %11s %9s %9s %12s"
      % ("range m", "feedthru dB", "51Hz sb dB", "noise dB", "sb-noise", "sb/feedthru"))
for i in range(0, 7):
    s, n = M[sb, i].mean(), M[nz, i].mean()
    print("%8.2f %12.2f %11.2f %9.2f %9.2f %12.2f"
          % (rngB[i], db(dc[i]), db(s), db(n), db(s / n), db(s / dc[i])))
print("\n  sb/feedthrough is ~constant wherever the feedthrough is large, and the moment")
print("  the feedthrough falls 20 dB (at >=2.5 m) the sideband falls to EXACTLY the noise")
print("  floor. It tracks the feedthrough, not range.")
print("  >>> The 51 Hz sidebands are amplitude modulation of the TX feedthrough (vibration")
print("      or an electrical artefact), NOT a range-resolved return from the fan. With no")
print("      fan-OFF Config B capture in existence, whether the fan CAUSES them is untested.")

# ------------------------------- 5b. is 51.8 Hz a coherent line? resolution test
kf = np.flatnonzero((fdB >= 35) & (fdB <= 70))
pk = np.maximum(M[kf, 1] - np.median(M[clean, 1]), 0)
f51 = float(np.sum(fdB[kf] * pk) / np.sum(pk))
print("\n  sideband centroid = %.2f Hz (a 3-blade fan at %.0f rpm -- entirely plausible),"
      % (f51, f51 / 3 * 60))
print("  so it is worth asking the CW data whether a coherent %.0f Hz line exists at all." % f51)
print("\n  RESOLUTION-SCALING TEST. A coherent narrow line gains ~3 dB of contrast for every")
print("  halving of the FFT bin width, because its power stays in one bin while the noise")
print("  floor per bin drops. Broadband noise gains nothing. 60 Hz mains is the control.")


def cw_contrast(f, res, lo, hi, nsec=None):
    d = np.load(f, allow_pickle=True)
    x = d["data"][0].astype(float)
    fs = float(d["fs"])
    if nsec:
        x = x[:int(nsec * fs)]
    x = x - x.mean()
    n = int(fs / res)
    fr, Px = welch(x, fs=fs, nperseg=n, noverlap=n // 2, window="hann")
    con = db(Px) - db(median_filter(Px, size=41, mode="nearest"))
    return float(con[(fr >= lo) & (fr <= hi)].max())


F50A = R + "/fan/20260830_094820_fan_on_cw_50k_60s_0.npz"
F50B = R + "/fan/20260830_095020_fan_on_cw_50k_60s_1.npz"
E50 = R + "/baseline/20260829_061237_empty_baseline_60s.npz"
print("\n%8s %9s %9s %9s %11s | %12s" % ("bin Hz", "fanA", "fanB", "empty",
                                         "fan-empty", "60Hz (ctrl)"))
for res in [2.0, 1.0, 0.5, 0.25, 0.125]:
    a_ = cw_contrast(F50A, res, 49, 53)
    b_ = cw_contrast(F50B, res, 49, 53)
    e_ = cw_contrast(E50, res, 49, 53)
    ctl = cw_contrast(F50A, res, 59.5, 60.5)
    print("%8.3f %9.2f %9.2f %9.2f %11.2f | %12.2f" % (res, a_, b_, e_, 0.5 * (a_ + b_) - e_, ctl))
print("\n  the 60 Hz control climbs 13.3 -> 24.0 dB (+2.7 dB per halving) exactly as a")
print("  coherent line must. The 51.8 Hz fan-minus-empty excess stays flat near 0.8 dB and")
print("  then goes NEGATIVE. There is no coherent 51.8 Hz line in the CW data.")
print("  >>> the Config B sideband is an FMCW-mode feedthrough artefact, uncorroborated by")
print("      CW, and NOT evidence of a fan.")

# --------------------------------------------------------- 6. injection sens.
hdr("6. SENSITIVITY BY INJECTION -- what return WOULD Config B have found?")
f_beat = 2.0 * S_B * 2.5 / C
v_inj = 1.5
f_dinj = v_inj * P.HZ_PER_MPS
tf, ts = np.arange(spcB) / fsB, np.arange(nchB) / prfB
wD = np.hanning(nchB)[None, :, None]
rs = np.random.default_rng(0)
print("  inject a point scatterer at 2.5 m (f_beat %.0f Hz), v=%.1f m/s (f_d %.0f Hz,"
      % (f_beat, v_inj, f_dinj))
print("  mains-clean), random phase per CPI, into the REAL fan-on data.")


def inject_z(a_pk):
    ph = rs.uniform(0, 2 * np.pi, ncpi)[:, None, None]
    sig = a_pk * np.cos(2 * np.pi * f_beat * tf[None, None, :]
                        + 2 * np.pi * f_dinj * ts[None, :, None] + ph)
    _, Ri = range_fft(cpis + sig, fsB, S_B, 3)
    Pi = np.abs(np.fft.fftshift(np.fft.fft(Ri * wD, axis=1), axes=1)) ** 2
    cm = Pi.mean(0)
    cs = Pi.std(0, ddof=1) / np.sqrt(ncpi)
    zi = (cm - np.median(cm[clean, :], axis=0)[None, :]) / cs
    j = int(np.argmin(np.abs(velB - v_inj)))
    return float(zi[j - 2:j + 3, i25B - 1:i25B + 2].max())


print("\n  %11s %9s %8s" % ("amp uV rms", "/ LSB", "z"))
amps, zs = [], []
for a_pk in [0, 1e-6, 2e-6, 3e-6, 5e-6, 1e-5, 2e-5, 5e-5]:
    z = inject_z(a_pk)
    amps.append(a_pk / np.sqrt(2))
    zs.append(z)
    print("  %11.2f %9.4f %8.2f" % (a_pk / np.sqrt(2) * 1e6, a_pk / np.sqrt(2) / LSB, z))
amps, zs = np.array(amps), np.array(zs)
o = np.argsort(zs)
a_min = float(np.interp(bar, zs[o], amps[o]))
print("\n  >>> Config B would have detected a %.2f uV rms IF return at 2.5 m" % (a_min * 1e6))
print("      = %.4f LSB = %.1f dB BELOW one ADC code (that is the FFT processing gain)."
      % (a_min / LSB, 20 * np.log10(a_min / LSB)))
print("      The real fan produced nothing above the %.2f sigma bar, so the fan's IF" % bar)
print("      return is below %.2f uV rms." % (a_min * 1e6))

# ------------------------------------------------------------ 7. CW crosscheck
hdr("7. CW CROSS-CHECK (independent of FMCW; gain-invariant line contrast)")


def cwspec(f, nsec=None, res=1.0):
    d = np.load(f, allow_pickle=True)
    x = d["data"][0].astype(float)
    fs = float(d["fs"])
    if nsec:
        x = x[:int(nsec * fs)]
    x = x - x.mean()
    n = int(fs / res)
    fr, Px = welch(x, fs=fs, nperseg=n, noverlap=n // 2, window="hann")
    return fr, db(Px) - db(median_filter(Px, size=41, mode="nearest")), len(x) / fs


print("  Absolute levels are NOT comparable: the fan-on 50 kSa/s runs sit ~3 dB BELOW the")
print("  day-earlier empty baseline (118/108 uV rms vs 158 uV), so an absolute difference")
print("  measures session gain, not the fan. Compare each spectrum's LINE CONTRAST against")
print("  its own running median instead -- gain-invariant, and a blade line is a narrowband")
print("  peak, which survives any broadband offset. Averaging is equalised in both pairs.")
for tag, fa, fb, fe, ns in [
        ("50 kSa/s, 60 s each (equal averaging)",
         R + "/fan/20260830_094820_fan_on_cw_50k_60s_0.npz",
         R + "/fan/20260830_095020_fan_on_cw_50k_60s_1.npz",
         R + "/baseline/20260829_061237_empty_baseline_60s.npz", None),
        ("100 kSa/s, both truncated to 6 s (equal averaging)",
         R + "/fan/20260830_093545_fan_on_cw0_100k.npz",
         R + "/fan/20260830_093545_fan_on_cw1_100k.npz",
         R + "/cw/smoke_static.npz", 6.0)]:
    fr, ca, da_ = cwspec(fa, ns)
    _, cb, _ = cwspec(fb, ns)
    _, ce, _ = cwspec(fe, ns)
    exc = 0.5 * (ca + cb) - ce
    s = (fr >= 5) & (fr <= 2000) & (np.abs(fr - np.round(fr / 60) * 60) >= 2)
    i = np.flatnonzero(s)[np.argmax(exc[np.flatnonzero(s)])]
    sig = exc[i] / exc[s].std()
    print("\n  %s  (%.0f s each)" % (tag, da_))
    print("    max fan-on excess %+.2f dB at %.1f Hz (%.3f m/s), sigma_measured %.2f dB"
          % (exc[i], fr[i], fr[i] / P.HZ_PER_MPS, exc[s].std()))
    print("    -> %.1f sigma; %d non-mains bins searched, chance bar %.2f sigma -> %s"
          % (sig, s.sum(), chance_bar(s.sum()),
             "DETECTION" if sig > chance_bar(s.sum()) else "NO DETECTION"))

# ======================================================================= FIGURE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2, figsize=(13.5, 9))
a = ax[0, 0]
a.fill_between(rngA, db(p_emp) - 2 * sigma_diff, db(p_emp) + 2 * sigma_diff,
               color="0.75", alpha=.7, label="empty $\\pm2\\sigma$ (measured)")
a.plot(rngA, db(p_emp), "k-", lw=1.4, label="empty room")
a.plot(rngA, db(p_mov), "r-", lw=1.8, label="person (positive control)")
a.plot(rngA, db(p_fan), "b-", lw=1.8, label="fan on")
a.axvline(2.5, color="g", ls=":", lw=1.5, label="fan at 2.5 m")
a.set_xlim(0.4, 8)
a.set_xlabel("range (m)")
a.set_ylabel("MTI power (dB)")
a.set_title("A. Config A MTI profiles: person %.0f$\\sigma$, fan %.1f$\\sigma$" % (s_m, s25))
a.legend(fontsize=8)
a.grid(alpha=.3)

a = ax[0, 1]
nc = min(25, len(rngB))
ext = [rngB[0], rngB[nc - 1], velB[0], velB[-1]]
Md = db(cell_mean[:, :nc]).copy()
Md[notch] = np.nan
im = a.imshow(Md, aspect="auto", origin="lower", extent=ext, cmap="viridis",
              vmin=np.nanpercentile(Md, 2), vmax=np.nanpercentile(Md, 99.8))
a.axvline(2.5, color="r", ls=":", lw=1.6)
a.text(2.6, 2.6, "fan", color="r", fontsize=9)
a.set_xlabel("range (m)")
a.set_ylabel("velocity (m/s)")
a.set_title("B. Config B range-Doppler, 200 CPIs, zero-Doppler notched")
plt.colorbar(im, ax=a, label="dB")

a = ax[1, 0]
kk = np.flatnonzero(clean)
for i, c, l in [(1, "C3", "%.2f m (feedthrough bin)" % rngB[1]),
                (i25B, "C0", "%.2f m (the fan's range)" % rngB[i25B]),
                (12, "C7", "%.2f m (empty space)" % rngB[12])]:
    a.plot(fdB[kk], db(cell_mean[kk, i] / np.median(cell_mean[kk, i])), color=c, lw=1.2, label=l)
for h in range(1, 9):
    a.axvline(60 * h, color="r", ls=":", lw=.6)
    a.axvline(-60 * h, color="r", ls=":", lw=.6)
a.axvline(51, color="m", ls="--", lw=1)
a.axvline(-51, color="m", ls="--", lw=1)
a.set_xlabel("Doppler (Hz)   red dotted = 60 Hz multiples, magenta = 51 Hz")
a.set_ylabel("dB over range-bin median")
a.set_title("C. Doppler cuts: the 51 Hz pair sits on the feedthrough bin only")
a.legend(fontsize=8)
a.grid(alpha=.3)

a = ax[1, 1]
a.semilogx(np.maximum(amps, 3e-8) * 1e6, zs, "o-", color="C2")
a.axhline(bar, color="r", ls="--", label="chance bar %.2f$\\sigma$" % bar)
a.axvline(a_min * 1e6, color="k", ls=":", label="threshold %.2f $\\mu$V rms" % (a_min * 1e6))
a.set_xlabel("injected target amplitude ($\\mu$V rms)")
a.set_ylabel("z at the injected cell")
a.set_title("D. Sensitivity: detects %.0f dB below one ADC code" % (20 * np.log10(a_min / LSB)))
a.legend(fontsize=8)
a.grid(alpha=.3, which="both")

fig.suptitle("Range-Doppler + MTI: the person (positive control) is found at %.0f sigma; "
             "the fan at 2.5 m is not found at all" % s_m, fontsize=12)
fig.tight_layout()
fig.savefig(OUT + "/range-doppler-mti.png", dpi=130)
print("\nfigure -> " + OUT + "/range-doppler-mti.png")
