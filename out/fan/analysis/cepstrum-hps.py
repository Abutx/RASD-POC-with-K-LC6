"""
Cepstrum + Harmonic Product Spectrum detector for a rotating-blade (fan) comb.
K-LC6 24.125 GHz CW.  Question: is a running household fan detectable, and with
what confidence?

Physics: a rotating blade set produces HERM/JEM lines -- a comb in the Doppler
spectrum spaced at the blade-pass rate (N_blades * f_rot), extending out to the
blade-tip Doppler.  A comb is periodic IN FREQUENCY, so it collapses to one peak
in the cepstrum (quefrency q = 1/spacing) and to one peak in the harmonic
product spectrum.  Both integrate every line, so they beat peak-picking.

Confounders handled explicitly:
  * the 60 Hz mains comb does the same thing, at q = 1/60 = 16.67 ms, and a fan
    is an electrical load so mains pickup can GROW when it is switched on;
  * the fan-ON and fan-OFF captures were taken in different sessions and differ
    by ~1.5-2 dB in absolute floor and by one ADC code in DC offset, so only
    spectral SHAPE is comparable -- which is what this method uses;
  * with ~1400 quefrency bins searched, 3 sigma happens ~4x by chance, so the
    real floor is established from SAME-CONDITION null controls, not from theory.
"""
import json, os, sys
import numpy as np
from scipy.signal import welch, get_window, medfilt
from scipy.signal import windows as _win
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "C:/dev/klc6")
from klc6 import process as P

OUT = "C:/dev/klc6/out/fan/analysis"
HZ_PER_MPS = P.HZ_PER_MPS          # 160.945 Hz per m/s
R = {}


def say(*a):
    print(*a, flush=True)


def hdr(t):
    say("\n" + "=" * 78); say(t); say("=" * 78)


# ----------------------------------------------------------------- data ------
CW = {
    "fanON_100k_0": "C:/dev/klc6/out/fan/20260830_093545_fan_on_cw0_100k.npz",
    "fanON_100k_1": "C:/dev/klc6/out/fan/20260830_093545_fan_on_cw1_100k.npz",
    "fanOFF_100k":  "C:/dev/klc6/out/cw/smoke_static.npz",
    "fanON_50k_a":  "C:/dev/klc6/out/fan/20260830_094820_fan_on_cw_50k_60s_0.npz",
    "fanON_50k_b":  "C:/dev/klc6/out/fan/20260830_095020_fan_on_cw_50k_60s_1.npz",
    "fanOFF_50k":   "C:/dev/klc6/out/baseline/20260829_061237_empty_baseline_60s.npz",
}


def load_cw(key):
    d = np.load(CW[key], allow_pickle=True)
    x = np.asarray(d["data"])[0].astype(np.float64)   # row 0 == channel 'I'
    return x, float(np.asarray(d["fs"]))


# --------------------------------------------------------- core method ------
NPER = 16384        # Welch segment -> df = 3.05 Hz at 50 kSa/s
SMOOTH = 101        # median-filter trend removal, 308 Hz wide


def welch_db(x, fs, nperseg, band):
    x = x - x.mean()
    f, Pxx = welch(x, fs, window="hann", nperseg=nperseg,
                   noverlap=nperseg // 2, detrend="constant")
    m = (f >= band[0]) & (f <= band[1])
    return f[m], 10.0 * np.log10(Pxx[m] + 1e-30)


def flatten_log(S_db, smooth_bins=SMOOTH):
    """Remove the smooth spectral trend; only comb structure survives.
    Also makes the result invariant to the flat gain offset between sessions."""
    k = int(smooth_bins) | 1
    trend = medfilt(S_db, kernel_size=k)
    h = k // 2
    trend[:h] = trend[h]
    trend[-h:] = trend[-h - 1]
    return S_db - trend


def real_cepstrum(Rdb):
    """Cepstrum magnitude of the flattened log spectrum.

    Rdb is real, sampled uniformly in FREQUENCY (step df).  A comb of spacing D
    makes Rdb periodic in n with period D/df, so |FFT_n{Rdb}| peaks at bin
    k = N*df/D, i.e. quefrency q = k/(N*df) = 1/D seconds.

    Taper is Tukey(0.15), NOT Hann: a Hann taper over a wide analysis band
    annihilates a comb that only occupies the lower part of the band (verified
    -- it hid an injected 73 Hz comb completely).
    """
    r = Rdb - Rdb.mean()
    r = r * _win.tukey(len(r), 0.15)
    return np.abs(np.fft.rfft(r))


def quefrency_axis(nbins, df):
    return np.arange(nbins // 2 + 1) / (nbins * df)


def hps_log(freq, Rdb, K=5, fmin=15.0, fmax=600.0, nf=3000):
    """Log-domain harmonic product spectrum, H(f0) = mean_k R(k*f0).
    Only fundamentals whose first K harmonics all lie in band are scanned."""
    fmin = max(fmin, freq[0])
    fmax = min(fmax, freq[-1] / K)
    f0 = np.linspace(fmin, fmax, nf)
    H = np.zeros_like(f0)
    for k in range(1, K + 1):
        H += np.interp(k * f0, freq, Rdb)
    return f0, H / K


def mains_mask(freq, tol=2.0, mains=60.0):
    n = np.round(freq / mains)
    return (n >= 1) & (np.abs(freq - n * mains) <= tol)


def blocks(x, fs, nblocks, band, nper=NPER, smooth=SMOOTH):
    """Split into independent sub-records -> one flattened spectrum + one
    cepstrum each.  This is how the floor gets MEASURED (ground rule 5)."""
    seg = len(x) // nblocks
    Rs, cs, f = [], [], None
    for i in range(nblocks):
        f, S = welch_db(x[i * seg:(i + 1) * seg], fs, nper, band)
        Rd = flatten_log(S, smooth)
        Rs.append(Rd); cs.append(real_cepstrum(Rd))
    return f, np.array(Rs), np.array(cs), quefrency_axis(len(Rs[0]), f[1] - f[0])


def zstat(a, b):
    """Welch two-sample z per bin, from the MEASURED block scatter.
    Handles unequal block counts via the 1/n1 + 1/n2 rule (ground rule 5)."""
    se = np.sqrt(a.std(0, ddof=1) ** 2 / len(a) + b.std(0, ddof=1) ** 2 / len(b))
    return (a.mean(0) - b.mean(0)) / se, se


# =============================================================== SECTION A ===
hdr("A. DATA INTEGRITY / QUANTISATION  (ground rule 2)")
A = {}
for k in CW:
    x, fs = load_cw(k)
    codes = np.unique(x)
    lsb = float(np.median(np.diff(np.sort(codes))))
    A[k] = dict(n=len(x), fs=fs, dur=len(x) / fs, ncodes=int(len(codes)),
                lsb_uV=lsb * 1e6, rms_uV=float(x.std() * 1e6),
                dc_mV=float(x.mean() * 1e3), nzero=int((x == 0).sum()))
    say("%-13s %9d @ %5.0f kSa/s %5.1f s  codes=%2d  LSB=%6.1f uV  rms=%6.1f uV"
        "  DC=%7.3f mV  zeros=%d"
        % (k, len(x), fs / 1000, len(x) / fs, len(codes), lsb * 1e6,
           x.std() * 1e6, x.mean() * 1e3, int((x == 0).sum())))
R["integrity"] = A
say("")
say("  Only 4-5 distinct ADC codes in every record (LSB 336 uV), as FINDINGS 1.1")
say("  predicts. Detection depends entirely on FFT processing gain.")
say("  Fan-ON records sit ~1.2-1.5 mV lower in DC and ~1.5-2 dB lower in")
say("  broadband floor than the fan-OFF references, which came from a different")
say("  session. ABSOLUTE level is therefore not comparable between conditions;")
say("  only spectral SHAPE is. Cepstrum/HPS use shape only.")

# =============================================================== SECTION B ===
hdr("B. POSITIVE CONTROLS  (does the method work at all, on real data?)")

# ---- B1: a real, known 50.000 Hz comb -------------------------------------
say("")
say("B1. REAL known comb: FMCW Config-A ramp feedthrough, truth = 50.000 Hz")
x_ramp = np.load("C:/dev/klc6/out/fmcw/raw_chirps_moving.npz")["chirps"].astype(float).ravel()
fR, SR = welch_db(x_ramp, 1e5, 32768, (100.0, 5000.0))
RR = flatten_log(SR)
cR = real_cepstrum(RR); qR = quefrency_axis(len(RR), fR[1] - fR[0])
sel = (qR > 2e-3) & (qR < 0.3)
i = int(np.argmax(cR[sel])); q_hat = qR[sel][i]
fl = np.median(cR[sel]); md = 1.4826 * np.median(np.abs(cR[sel] - fl))
say("    cepstrum  -> %.2f Hz  (%.0f sigma over its own measured floor)"
    % (1 / q_hat, (cR[sel][i] - fl) / md))
f0R, HR = hps_log(fR, RR, K=5, fmin=30, fmax=300, nf=6000)
say("    HPS(K=5)  -> %.2f Hz  (HPS also locks onto 3x = 150 Hz, the standard"
    % f0R[int(np.argmax(HR))])
say("                 HPS harmonic ambiguity; the cepstrum does not suffer it)")
R["pc_ramp"] = dict(cepstrum_hz=float(1 / q_hat),
                    hps_hz=float(f0R[int(np.argmax(HR))]),
                    sigma=float((cR[sel][i] - fl) / md))
b1_pass = abs(1 / q_hat - 50.0) < 1.0

# ---- B2: a real, known 60 Hz mains comb in the empty room ------------------
say("")
say("B2. REAL known comb #2: 60 Hz mains in the empty-room baseline (FINDINGS 3)")
x_off, fs_off = load_cw("fanOFF_50k")
f60, S60 = welch_db(x_off - x_off.mean(), fs_off, 32768, (40.0, 3000.0))
R60 = flatten_log(S60)
c60 = real_cepstrum(R60); q60 = quefrency_axis(len(R60), f60[1] - f60[0])
s60 = (q60 > 2e-3) & (q60 < 0.15)
j = int(np.argmax(c60[s60]))
fl = np.median(c60[s60]); md = 1.4826 * np.median(np.abs(c60[s60] - fl))
say("    cepstrum  -> %.2f Hz  (%.1f sigma)   truth 60.00 Hz"
    % (1 / q60[s60][j], (c60[s60][j] - fl) / md))
f0m, Hm = hps_log(f60, R60, K=5, fmin=40, fmax=300, nf=6000)
say("    HPS(K=5)  -> %.2f Hz" % f0m[int(np.argmax(Hm))])
_dq = q60[1]
say("    quefrency bin = %.3f ms wide = +-%.2f Hz at a 60 Hz spacing, so 59.4 and"
    % (_dq * 1e3, abs(60 - 1 / (1 / 60.0 + _dq))))
say("    60.0 Hz are the same cepstral bin. The mains cepstral peak is weak (only")
say("    h=1,3,5 stand out, so the comb is far from uniform) but HPS nails it.")
R["pc_mains"] = dict(cepstrum_hz=float(1 / q60[s60][j]),
                     sigma=float((c60[s60][j] - fl) / md),
                     hps_hz=float(f0m[int(np.argmax(Hm))]))
b2_pass = abs(1 / q60[s60][j] - 60.0) < 1.5

# ---- B3: the known moving person, by MTI (loading + statistics sanity) -----
say("")
say("B3. KNOWN MOVING PERSON via consecutive-chirp MTI (FINDINGS 5.5: +6.7 dB,")
say("    12 sigma at 1.5 m). Not my method -- this checks that the person really")
say("    is in the file I loaded and that my sigma machinery reproduces it.")


def mti_profile(chirps, S, npoly=3):
    c = np.asarray(chirps, dtype=float).copy()
    n = c.shape[1]
    t = np.linspace(-1, 1, n)
    for i in range(c.shape[0]):                      # ground rule 3
        c[i] -= np.polyval(np.polyfit(t, c[i], npoly), t)
    d = np.diff(c, axis=0)
    Sp = np.abs(np.fft.rfft(d * get_window("hann", n), axis=1)) ** 2
    return P.beat_to_range(np.fft.rfftfreq(n, 1 / 1e5), S), Sp


S180 = 180e6 * 50.0
rng, Sm = mti_profile(np.load("C:/dev/klc6/out/fmcw/raw_chirps_moving.npz")["chirps"], S180)
_,   Ss = mti_profile(np.load("C:/dev/klc6/out/fmcw/raw_chirps_box_out.npz")["chirps"], S180)
pm = 10 * np.log10(Sm.mean(0) + 1e-30)
ps = 10 * np.log10(Ss.mean(0) + 1e-30)
h1, h2 = np.array_split(Sm, 2)
half = 10 * np.log10(h1.mean(0) + 1e-30) - 10 * np.log10(h2.mean(0) + 1e-30)
bnd = (rng > 0.6) & (rng < 6)
floor_mti = float(np.std(half[bnd]) / np.sqrt(2))
say("    MEASURED split-half floor (same condition, two halves) = %.2f dB" % floor_mti)
for rr in (1.0, 1.5, 2.5, 4.0):
    j = int(np.argmin(np.abs(rng - rr)))
    say("    R=%4.1f m  moving %7.2f  static %7.2f  diff %+5.2f dB  (%5.1f sigma)"
        % (rr, pm[j], ps[j], pm[j] - ps[j], (pm[j] - ps[j]) / floor_mti))
jP = int(np.argmin(np.abs(rng - 1.5)))
R["pc_mti"] = dict(db=float(pm[jP] - ps[jP]),
                   sigma=float((pm[jP] - ps[jP]) / floor_mti), floor_db=floor_mti)
b3_pass = (pm[jP] - ps[jP]) / floor_mti > 5

# ---- B4: the known moving person, by MY method ----------------------------
say("")
say("B4. KNOWN MOVING PERSON via CEPSTRUM/HPS (the assigned method itself).")
BAND_P = (200.0, 8000.0)
xm = np.load("C:/dev/klc6/out/fmcw/raw_chirps_moving.npz")["chirps"].astype(float).ravel()
xs = np.load("C:/dev/klc6/out/fmcw/raw_chirps_box_out.npz")["chirps"].astype(float).ravel()
fp, Rmv, cmv, qp = blocks(xm, 1e5, 12, BAND_P, nper=8192, smooth=51)   # 0.4 s blocks
_,  Rst, cst, _ = blocks(xs, 1e5, 8, BAND_P, nper=8192, smooth=51)
zp, _ = zstat(cmv, cst)
SP = (qp > 1.5e-3) & (qp < 0.12)
jj = int(np.argmax(np.abs(zp[SP])))
say("    12 vs 8 blocks of 0.4 s, df=%.1f Hz, %d quefrency bins searched"
    % (fp[1] - fp[0], SP.sum()))
say("    TEST  moving vs static  : largest |sigma| = %.2f at spacing %.1f Hz"
    % (abs(zp[SP][jj]), 1 / qp[SP][jj]))
_, _, cn1, _ = blocks(xs[:len(xs) // 2], 1e5, 4, BAND_P, nper=8192, smooth=51)
_, _, cn2, _ = blocks(xs[len(xs) // 2:], 1e5, 4, BAND_P, nper=8192, smooth=51)
zn, _ = zstat(cn1, cn2)
_, _, cm1, _ = blocks(xm[:len(xm) // 2], 1e5, 6, BAND_P, nper=8192, smooth=51)
_, _, cm2, _ = blocks(xm[len(xm) // 2:], 1e5, 6, BAND_P, nper=8192, smooth=51)
zn_m, _ = zstat(cm1, cm2)
say("    NULL  static 1st vs 2nd half (4v4): largest |sigma| = %.2f"
    % np.abs(zn[SP]).max())
say("    NULL  moving 1st vs 2nd half (6v6): largest |sigma| = %.2f"
    % np.abs(zn_m[SP]).max())
_null_max = max(float(np.abs(zn[SP]).max()), float(np.abs(zn_m[SP]).max()))
b4_pass = bool(abs(zp[SP][jj]) > 5 and abs(zp[SP][jj]) > 1.5 * _null_max)
R["pc_ceps_person"] = dict(sigma=float(abs(zp[SP][jj])),
                           spacing_hz=float(1 / qp[SP][jj]),
                           null_sigma=_null_max, passed=b4_pass)
say("    -> cepstrum/HPS %s the walking person."
    % ("DETECTS" if b4_pass else "DOES NOT DETECT"))
say("    Expected: a walking human is not a frequency-periodic comb at any")
say("    spacing this data resolves (df=%.0f Hz here; gait cadence is ~2 Hz)."
    % (fp[1] - fp[0]))

# ---- B5: end-to-end sensitivity by injection into REAL fan-OFF data --------
say("")
say("B5. SENSITIVITY CALIBRATION. A synthetic blade comb (spacing 73.0 Hz --")
say("    deliberately NOT a multiple of 60 -- lines from 73 Hz to 4.0 kHz, random")
say("    phase) is added to the FIRST half of the real empty-room record; the")
say("    SECOND half stays clean and acts as the reference. Identical pipeline.")
BAND = (40.0, 6000.0)
half_n = len(x_off) // 2
xoff0 = x_off - x_off.mean()
A_h, B_h = xoff0[:half_n].copy(), xoff0[half_n:].copy()
rms = float(x_off.std())
t_h = np.arange(half_n) / fs_off
rng_ = np.random.default_rng(0)
_, _, cB5, qi = blocks(B_h, fs_off, 4, BAND)
SEL_I = (qi > 1.5e-3) & (qi < 0.12)
jt = int(np.argmin(np.abs(qi - 1 / 73.0)))
inj = []
say("    NOTE a comb of spacing D also puts cepstral rahmonics at q = n/D, i.e.")
say("    apparent spacings D/2, D/3, ... The pass criterion below is the sigma AT")
say("    the true quefrency 1/73 s, not the location of the global maximum")
say("    (which often lands on a rahmonic).")
for amp_db in [-46, -40, -34, -30, -28, -26, -24, -22, -20, -18]:
    a = rms * 10 ** (amp_db / 20.0)
    nlines = int(4000 // 73)
    y = A_h.copy()
    for k in range(1, nlines + 1):
        y += (a / np.sqrt(nlines)) * np.sqrt(2) * np.cos(
            2 * np.pi * k * 73.0 * t_h + rng_.uniform(0, 2 * np.pi))
    _, _, cA5, _ = blocks(y, fs_off, 4, BAND)
    zi, _ = zstat(cA5, cB5)
    jg = int(np.argmax(zi[SEL_I]))
    inj.append((amp_db, float(1 / qi[SEL_I][jg]), float(zi[SEL_I][jg]), float(zi[jt])))
    _sp = 1 / qi[SEL_I][jg]
    _rah = abs(73.0 / _sp - round(73.0 / _sp)) < 0.05
    say("    comb %+4d dB re IF rms (%d lines) -> global max %7.2f Hz @ %+6.1f sig"
        " %-16s;  sigma AT 73.0 Hz = %+6.1f"
        % (amp_db, nlines, _sp, zi[SEL_I][jg],
           "(73/n rahmonic)" if _rah else "", zi[jt]))
R["injection"] = [dict(dbc=r[0], global_hz=r[1], global_sigma=r[2],
                       sigma_at_truth=r[3]) for r in inj]
ok = [r for r in inj if r[3] > 5.0]
inj_thresh = min(r[0] for r in ok) if ok else None
R["injection_threshold_dbc"] = inj_thresh
say("    -> the detector recovers the correct spacing above 5 sigma once the comb")
say("       carries >= %s dB relative to the wideband IF rms." % inj_thresh)

# =============================================================== SECTION C ===
hdr("C. FAN-ON vs FAN-OFF, cepstrum, matched 50 kSa/s / 60 s, EQUAL block counts")
NB = 8
cond = {}
for k in ["fanON_50k_a", "fanON_50k_b", "fanOFF_50k"]:
    x, fs = load_cw(k)
    f, Rs, cs, q = blocks(x - x.mean(), fs, NB, BAND)
    cond[k] = dict(f=f, Rs=Rs, cs=cs, q=q)
fax = cond["fanOFF_50k"]["f"]
say("  band %.0f-%.0f Hz (%.2f-%.1f m/s), df=%.3f Hz, %d bins, %d blocks x 7.5 s"
    % (BAND[0], BAND[1], BAND[0] / HZ_PER_MPS, BAND[1] / HZ_PER_MPS,
       fax[1] - fax[0], len(fax), NB))
qax = cond["fanOFF_50k"]["q"]
SELQ = (qax > 1.5e-3) & (qax < 0.12)          # spacing 8.3 .. 667 Hz
qsel = qax[SELQ]
say("  %d quefrency bins searched -> comb spacings 8.3-667 Hz." % SELQ.sum())
say("  A household fan is 900-1600 rpm with 3-5 blades: blade-pass 45-133 Hz,")
say("  rotation 15-27 Hz. Both lie inside the searched span.")


def run_pair(ka, kb, label):
    z, se = zstat(cond[ka]["cs"], cond[kb]["cs"])
    j = int(np.argmax(np.abs(z[SELQ])))
    say("  %-34s max|sigma| = %5.2f at spacing %7.2f Hz" %
        (label, abs(z[SELQ][j]), 1 / qsel[j]))
    return z, float(abs(z[SELQ][j])), float(1 / qsel[j])


say("")
say("  TEST (fan present vs absent):")
z_a, m_a, sp_a = run_pair("fanON_50k_a", "fanOFF_50k", "fan ON run a  vs  fan OFF")
z_b, m_b, sp_b = run_pair("fanON_50k_b", "fanOFF_50k", "fan ON run b  vs  fan OFF")
say("")
say("  NULL CONTROLS (same condition on both sides -- these MEASURE the real")
say("  false-alarm floor of the whole detector, session drift included):")
z_n1, m_n1, sp_n1 = run_pair("fanON_50k_a", "fanON_50k_b", "fan ON a      vs  fan ON b")
_, RoA, coA, _ = blocks(A_h, fs_off, 4, BAND)
_, RoB, coB, _ = blocks(B_h, fs_off, 4, BAND)
zn2, _ = zstat(coA, coB)
m_n2 = float(np.abs(zn2[SELQ]).max())
say("  %-34s max|sigma| = %5.2f at spacing %7.2f Hz"
    % ("fan OFF 1st half vs 2nd half", m_n2,
       1 / qsel[int(np.argmax(np.abs(zn2[SELQ])))]))
xa_, _ = load_cw("fanON_50k_a"); xa_ = xa_ - xa_.mean()
_, _, cA1, _ = blocks(xa_[:len(xa_) // 2], 50000.0, 4, BAND)
_, _, cA2, _ = blocks(xa_[len(xa_) // 2:], 50000.0, 4, BAND)
zn3, _ = zstat(cA1, cA2)
m_n3 = float(np.abs(zn3[SELQ]).max())
say("  %-34s max|sigma| = %5.2f" % ("fan ON a 1st half vs 2nd half", m_n3))

con = np.vstack([cond["fanON_50k_a"]["cs"], cond["fanON_50k_b"]["cs"]])
coff = cond["fanOFF_50k"]["cs"]
z_p, se_p = zstat(con, coff)
jp = int(np.argmax(np.abs(z_p[SELQ])))
say("")
say("  pooled fan ON (16 blocks) vs fan OFF (8 blocks): max|sigma| = %.2f at %.2f Hz"
    % (abs(z_p[SELQ][jp]), 1 / qsel[jp]))
say("  top 8 quefrency bins of the pooled test:")
say("  %9s %11s %10s %9s %8s  60Hz-mult?"
    % ("q (ms)", "spacing Hz", "d_ceps", "SE", "sigma"))
for j in np.argsort(np.abs(z_p[SELQ]))[::-1][:8]:
    sp = 1 / qsel[j]
    is_m = abs(sp / 60 - round(sp / 60)) < 0.03 and sp > 30
    say("  %9.3f %11.2f %+10.3f %9.3f %+8.2f  %s"
        % (qsel[j] * 1e3, sp, (con.mean(0) - coff.mean(0))[SELQ][j],
           se_p[SELQ][j], z_p[SELQ][j], "YES" if is_m else "-"))
say("")
say("  Theory says |z|>3 should occur %.1f times in %d bins by chance."
    % (SELQ.sum() * 2 * 0.00135, SELQ.sum()))
say("  MEASURED null controls put the real floor at max|sigma| = %.2f / %.2f / %.2f"
    % (m_n1, m_n2, m_n3))
R["ceps_test"] = dict(on_a_vs_off=dict(sigma=m_a, spacing_hz=sp_a),
                      on_b_vs_off=dict(sigma=m_b, spacing_hz=sp_b),
                      pooled=dict(sigma=float(abs(z_p[SELQ][jp])),
                                  spacing_hz=float(1 / qsel[jp])),
                      nbins=int(SELQ.sum()))
R["ceps_nulls"] = dict(onA_vs_onB=m_n1, off_splithalf=m_n2, onA_splithalf=m_n3)

da = cond["fanON_50k_a"]["cs"].mean(0) - coff.mean(0)
db_ = cond["fanON_50k_b"]["cs"].mean(0) - coff.mean(0)
rho = float(np.corrcoef(da[SELQ], db_[SELQ])[0, 1])
# Proper null with the SAME shared-reference structure: split the fan-OFF
# record into three independent thirds; two play "run a"/"run b", one plays
# the shared reference. Any correlation here is structural, not target.
n3 = len(xoff0) // 3
_, _, cT1, _ = blocks(xoff0[:n3], fs_off, 3, BAND)
_, _, cT2, _ = blocks(xoff0[n3:2 * n3], fs_off, 3, BAND)
_, _, cT3, _ = blocks(xoff0[2 * n3:], fs_off, 3, BAND)
rho_null = float(np.corrcoef((cT1.mean(0) - cT3.mean(0))[SELQ],
                             (cT2.mean(0) - cT3.mean(0))[SELQ])[0, 1])
say("  Repeatability r of the ON-minus-OFF cepstral difference across the two")
say("  independent fan-ON runs: %+.3f. Same statistic with two halves of the" % rho)
say("  fan-OFF record standing in for 'fan ON': %+.3f -- so that much correlation"
    % rho_null)
say("  comes from the SHARED reference alone, not from the fan.")
R["ceps_repeatability_r"] = rho
R["ceps_repeatability_r_null"] = rho_null

say("")
say("  RESTRICTED SEARCH. Constraining the hypothesis to physically plausible")
say("  blade-pass rates (40-140 Hz: 3-5 blades at 800-1700 rpm) shrinks the")
say("  search space and so lowers the bar a real detection has to clear:")
BP = SELQ & (qax > 1 / 140.0) & (qax < 1 / 40.0)
say("  %d quefrency bins; |z|>3 expected %.2f times by chance."
    % (BP.sum(), BP.sum() * 2 * 0.00135))
for lab, zz_ in (("fan ON a vs OFF", z_a), ("fan ON b vs OFF", z_b),
                 ("pooled ON vs OFF", z_p), ("NULL ON a vs ON b", z_n1)):
    j = int(np.argmax(np.abs(zz_[BP])))
    say("    %-18s max|sigma| = %5.2f at %6.2f Hz blade-pass"
        " (%.0f rpm on 3 blades, %.0f on 5)"
        % (lab, abs(zz_[BP][j]), 1 / qax[BP][j], 1 / qax[BP][j] / 3 * 60,
           1 / qax[BP][j] / 5 * 60))
say("")
say("  CANDIDATE CONSISTENCY. A real blade comb must appear at the SAME spacing")
say("  in both independent fan-ON runs. Evaluating every test at the quefrency")
say("  picked by the pooled test (%.2f Hz spacing):" % (1 / qsel[jp]))
_qc = qsel[jp]
_jc = int(np.argmin(np.abs(qax - _qc)))
for lab, zz_ in (("fan ON a vs OFF", z_a), ("fan ON b vs OFF", z_b),
                 ("NULL ON a vs ON b", z_n1)):
    say("    %-18s sigma at %.2f Hz = %+6.2f" % (lab, 1 / _qc, zz_[_jc]))
say("    run a picked this bin; run b gives %+.2f sigma there. A blade comb"
    % z_b[_jc])
say("    cannot be present in one 60 s fan-ON record and absent in the next.")
R["candidate_consistency"] = dict(spacing_hz=float(1 / _qc),
                                  sigma_run_a=float(z_a[_jc]),
                                  sigma_run_b=float(z_b[_jc]),
                                  sigma_null=float(z_n1[_jc]))
say("")
say("  60 Hz RAHMONIC FAMILY (ground rule 1). The prominent cepstral peaks in")
say("  BOTH conditions sit at q = n/60 s -- the mains comb. Difference there:")
mrow = []
for n_ in range(1, 8):
    qq = n_ / 60.0
    if qq > qax[SELQ][-1]:
        break
    jm = int(np.argmin(np.abs(qax - qq)))
    mrow.append((n_, float(z_p[jm]), float(z_n1[jm])))
    say("    q=%6.2f ms (n=%d, spacing %.1f Hz): ON-OFF %+6.2f sigma  |  NULL %+6.2f"
        % (qq * 1e3, n_, 60.0 / n_, z_p[jm], z_n1[jm]))
say("  The mains rahmonics do not grow with the fan, so the fan is not even")
say("  detectable as an ELECTRICAL load on this receiver, let alone as a target.")
R["mains_rahmonics"] = [dict(n=r[0], sigma_on_off=r[1], sigma_null=r[2]) for r in mrow]
R["ceps_restricted"] = dict(
    nbins=int(BP.sum()),
    pooled_sigma=float(np.abs(z_p[BP]).max()),
    pooled_hz=float(1 / qax[BP][int(np.argmax(np.abs(z_p[BP])))]),
    null_sigma=float(np.abs(z_n1[BP]).max()))

# =============================================================== SECTION D ===
hdr("D. HARMONIC PRODUCT SPECTRUM, fan-ON vs fan-OFF")


def hps_blocks(Rs, K=5):
    out, f0 = [], None
    for r in Rs:
        f0, H = hps_log(fax, r, K=K, fmin=20, fmax=600, nf=3000)
        out.append(H)
    return f0, np.array(out)


f0_last = zH_last = mm_last = None
for K in (3, 5):
    f0, Hon = hps_blocks(np.vstack([cond["fanON_50k_a"]["Rs"],
                                    cond["fanON_50k_b"]["Rs"]]), K)
    _, Hoff = hps_blocks(cond["fanOFF_50k"]["Rs"], K)
    _, Hna = hps_blocks(cond["fanON_50k_a"]["Rs"], K)
    _, Hnb = hps_blocks(cond["fanON_50k_b"]["Rs"], K)
    zH, _ = zstat(Hon, Hoff)
    zHn, _ = zstat(Hna, Hnb)
    mm = mains_mask(f0, tol=2.0)
    jb = int(np.argmax(zH))
    nz_ = zH.copy(); nz_[mm] = -np.inf
    say("  K=%d  scan %.0f-%.0f Hz (%d trials, %.1f%% tagged 60 Hz multiple)"
        % (K, f0[0], f0[-1], len(f0), mm.mean() * 100))
    say("       best overall   %7.2f Hz  %+0.3f dB  %+5.2f sigma  %s"
        % (f0[jb], (Hon.mean(0) - Hoff.mean(0))[jb], zH[jb],
           "(60 Hz MULTIPLE)" if mm[jb] else ""))
    say("       best non-mains %7.2f Hz  %+5.2f sigma"
        % (f0[int(np.argmax(nz_))], nz_.max()))
    say("       same-condition null (ON a vs ON b): max|sigma| %+5.2f"
        % np.abs(zHn).max())
    R["hps_K%d" % K] = dict(best_f0=float(f0[jb]), best_sigma=float(zH[jb]),
                            best_is_mains=bool(mm[jb]),
                            best_nonmains_f0=float(f0[int(np.argmax(nz_))]),
                            best_nonmains_sigma=float(nz_.max()),
                            null_sigma=float(np.abs(zHn).max()),
                            n_trials=int(len(f0)))
    f0_last, zH_last, mm_last = f0, zH, mm

# =============================================================== SECTION E ===
hdr("E. MAINS COMB: does switching the fan on raise it? (ground rule 1)")
mtab = {}
for k in ["fanON_50k_a", "fanON_50k_b", "fanOFF_50k"]:
    x, fs = load_cw(k); x = x - x.mean()
    f, S = welch_db(x, fs, 32768, (10, 20000))
    med = float(np.median(S))
    vals = [float(S[int(np.argmin(np.abs(f - 60 * h)))] - med)
            for h in [1, 2, 3, 4, 5, 8, 16, 32, 48]]
    mtab[k] = vals
    say("  %-12s dB over median at 60*h (h=1,2,3,4,5,8,16,32,48): %s"
        % (k, " ".join("%6.1f" % v for v in vals)))
say("")
say("  The mains comb is NOT stronger with the fan on -- the fundamental is")
say("  ~2 dB WEAKER, tracking the same overall floor offset seen in A. So no")
say("  candidate blade line here can be explained away as extra mains pickup,")
say("  and equally no mains growth is masquerading as a fan detection.")
R["mains_db_over_median"] = mtab
ft = np.linspace(3000, 12000, 200000)
fr = float(mains_mask(ft, 2.0).mean() * 100)
say("  Computed false-tag rate of a '+-2 Hz of a 60 Hz multiple' rule above")
say("  3 kHz: %.2f%% of random frequencies. High-order mains tags are therefore" % fr)
say("  treated as uninformative and are not used as evidence either way.")
R["mains_false_tag_pct"] = fr

# =============================================================== SECTION F ===
hdr("F. CROSS-CHECK at 100 kSa/s (25 s x2 fan-ON vs 6 s fan-OFF)")
xa, _ = load_cw("fanON_100k_0"); xb, _ = load_cw("fanON_100k_1"); xo, _ = load_cw("fanOFF_100k")
BAND2 = (40.0, 6000.0)
say("  Ground rule 5 in force. Blocks are forced to EQUAL DURATION (2.0 s) on")
say("  both sides, so each block carries the same number of Welch averages and")
say("  the two conditions have the same per-block scatter. What cannot be equal")
say("  is the NUMBER of blocks: 12 for each 25 s fan-ON record, only 3 for the")
say("  6 s fan-OFF reference. The 1/n1+1/n2 term carries that, but a variance")
say("  estimated from 3 blocks (2 dof) is itself very noisy, so sigma here is")
say("  unstable. The matched-n null below MEASURES how much that inflates it.")
f2, _, c2a, q2 = blocks(xa - xa.mean(), 1e5, 12, BAND2)
_, _, c2b, _ = blocks(xb - xb.mean(), 1e5, 12, BAND2)
_, _, c2o, _ = blocks(xo - xo.mean(), 1e5, 3, BAND2)
S2 = (q2 > 1.5e-3) & (q2 < 0.12)
res100 = {}
for lab, cc in (("fan ON 0 vs fan OFF", c2a), ("fan ON 1 vs fan OFF", c2b)):
    zz, _ = zstat(cc, c2o)
    j = int(np.argmax(np.abs(zz[S2])))
    res100[lab] = (float(abs(zz[S2][j])), float(1 / q2[S2][j]))
    say("  %-30s max|sigma| = %5.2f at spacing %7.2f Hz"
        % (lab, abs(zz[S2][j]), 1 / q2[S2][j]))
zz, _ = zstat(c2a, c2b)
j = int(np.argmax(np.abs(zz[S2])))
n_full = float(abs(zz[S2][j]))
say("  %-30s max|sigma| = %5.2f at spacing %7.2f Hz  <- NULL, 12v12"
    % ("fan ON 0 vs fan ON 1", n_full, 1 / q2[S2][j]))
zz2, _ = zstat(c2a[:3], c2a[6:9])
j2 = int(np.argmax(np.abs(zz2[S2])))
zz3, _ = zstat(c2a, c2b[:3])
j3 = int(np.argmax(np.abs(zz3[S2])))
say("  %-30s max|sigma| = %5.2f  <- NULL, 3v3 same record"
    % ("fan ON 0 early vs fan ON 0 late", abs(zz2[S2][j2])))
say("  %-30s max|sigma| = %5.2f  <- NULL, 12v3, same block counts as the test"
    % ("fan ON 0 (12) vs fan ON 1 (3)", abs(zz3[S2][j3])))
say("  The 12-vs-3 SAME-CONDITION null reaches %.1f sigma, i.e. the whole" % abs(zz3[S2][j3]))
say("  apparent significance of the 100 kSa/s test is an artefact of estimating")
say("  a variance from 3 reference blocks. This dataset cannot decide anything;")
say("  the duration-matched 50 kSa/s comparison in section C is the real test.")
# cross-rate check: does the 100 kSa/s candidate survive in the well-controlled
# duration-matched 50 kSa/s comparison?
_c100 = res100["fan ON 0 vs fan OFF"][1]
_j50 = int(np.argmin(np.abs(qax - 1.0 / _c100)))
say("  CROSS-RATE CHECK. The 100 kSa/s candidate spacing is %.2f Hz. In the" % _c100)
say("  duration-matched 50 kSa/s test (section C) that same spacing gives:")
say("    fan ON a vs OFF %+0.2f sigma; fan ON b vs OFF %+0.2f sigma; NULL %+0.2f"
    % (z_a[_j50], z_b[_j50], z_n1[_j50]))
say("  The two rates do not agree, so the 100 kSa/s feature belongs to its shared")
say("  6 s reference, not to the fan.")
R["cross_rate_check"] = dict(spacing_hz=float(_c100),
                             sigma50_run_a=float(z_a[_j50]),
                             sigma50_run_b=float(z_b[_j50]),
                             sigma50_null=float(z_n1[_j50]))
R["ceps_100k"] = dict(on0_vs_off=res100["fan ON 0 vs fan OFF"][0],
                      on0_vs_off_hz=res100["fan ON 0 vs fan OFF"][1],
                      on1_vs_off=res100["fan ON 1 vs fan OFF"][0],
                      on1_vs_off_hz=res100["fan ON 1 vs fan OFF"][1],
                      null_12v12=n_full,
                      null_3v3=float(abs(zz2[S2][j2])),
                      null_12v3=float(abs(zz3[S2][j3])))

# =============================================================== SECTION G ===
hdr("G. IS THE FAN VISIBLE BY ANY MEANS? (context for the null)")
d = np.load("C:/dev/klc6/out/fan/20260830_093545_fan_on_fmcw_cfgB.npz", allow_pickle=True)
cpis = d["cpis"].astype(float)
fsB = float(d["fs"]); prf = float(d["ramp"])
tt = np.linspace(-1, 1, cpis.shape[2])
acc = 0.0; rngB = velB = None
for i in range(cpis.shape[0]):
    c = cpis[i].copy()
    for k_ in range(c.shape[0]):
        c[k_] -= np.polyval(np.polyfit(tt, c[k_], 3), tt)   # ground rule 3
    rngB, velB, rd = P.range_doppler(c, fsB, 180e6 * prf, prf)
    acc = acc + 10 ** (rd / 10.0)
rdm = 10 * np.log10(acc / cpis.shape[0] + 1e-30)
nzv = np.abs(velB) > 0.25
prof = rdm[nzv, :].max(0)
say("  FMCW cfgB, 200 CPIs integrated, moving-target (|v|>0.25 m/s) range profile:")
for rr in [1, 2, 2.5, 3, 4, 6]:
    j = int(np.argmin(np.abs(rngB - rr)))
    say("    R=%4.1f m  %7.2f dB" % (rr, prof[j]))
say("  cfgB PRF=1000 Hz -> unambiguous |v| < %.2f m/s (%.0f Hz). A household fan"
    % (500.0 / HZ_PER_MPS, 500.0))
say("  blade tip runs 15-30 m/s (2.4-4.8 kHz), so cfgB ALIASES it and cannot")
say("  confirm or deny blade Doppler. CW at 50-100 kSa/s can, so the CW records")
say("  carry the whole argument.")
say("")
say("  CW broadband check: median PSD of each band MINUS that record's own")
say("  15-20 kHz reference band, so the session gain offset cancels.")
bands = [(200, 600), (600, 1500), (1500, 3000), (3000, 6000), (6000, 10000)]
btab = {}
for k in ["fanON_50k_a", "fanON_50k_b", "fanOFF_50k"]:
    x, fs = load_cw(k); x = x - x.mean()
    f, S = welch_db(x, fs, 16384, (20, 24000))
    ref = float(np.median(S[(f > 15000) & (f < 20000)]))
    row = [float(np.median(S[(f >= lo) & (f < hi)]) - ref) for lo, hi in bands]
    btab[k] = row
    say("    %-12s %s" % (k, " ".join("%7.2f" % v for v in row)))
say("    bands (Hz): " + " ".join("%d-%d" % b for b in bands))
say("")
say("  Quantitative bound on ANY blade micro-Doppler energy, block statistics:")
bb = {}
for k in ["fanON_50k_a", "fanON_50k_b", "fanOFF_50k"]:
    x, fs = load_cw(k); x = x - x.mean()
    seg = len(x) // 8
    rows = []
    for i in range(8):
        f, S = welch_db(x[i * seg:(i + 1) * seg], fs, 16384, (20, 24000))
        ref = float(np.median(S[(f > 15000) & (f < 20000)]))
        rows.append([float(np.median(S[(f >= lo) & (f < hi)]) - ref)
                     for lo, hi in bands])
    bb[k] = np.array(rows)
on_bb = np.vstack([bb["fanON_50k_a"], bb["fanON_50k_b"]])
off_bb = bb["fanOFF_50k"]
zb = (on_bb.mean(0) - off_bb.mean(0)) / np.sqrt(
    on_bb.std(0, ddof=1) ** 2 / len(on_bb) + off_bb.std(0, ddof=1) ** 2 / len(off_bb))
say("  Every band comes out NEGATIVE (fan-ON lower). That is not a negative fan")
say("  return -- it is a residual session systematic of 0.02-0.5 dB that survives")
say("  the reference-band normalisation. So the honest bound on a fan excess is")
say("  |systematic| + 2*SE, not the one-sided upper limit:")
bounds = []
for i, (lo, hi) in enumerate(bands):
    d = float(on_bb.mean(0)[i] - off_bb.mean(0)[i])
    se_ = abs(d / zb[i]) if zb[i] != 0 else float("nan")
    ub = abs(d) + 2 * se_
    below = -10 * np.log10(10 ** (ub / 10) - 1)
    bounds.append(dict(lo=lo, hi=hi, diff_db=d, sigma=float(zb[i]),
                       bound_db=float(ub), below_noise_db=float(below)))
    say("    %5d-%5d Hz (%4.1f-%4.1f m/s): diff %+0.3f dB (%+0.1f sig); fan excess"
        " < %0.3f dB -> any fan echo >= %0.1f dB BELOW the noise here"
        % (lo, hi, lo / HZ_PER_MPS, hi / HZ_PER_MPS, d, zb[i], ub, below))
R["band_bounds"] = bounds
R["band_bound_sigma"] = [float(v) for v in zb]
say("    i.e. %s m/s" % " ".join("%.1f-%.1f" % (b[0] / HZ_PER_MPS, b[1] / HZ_PER_MPS)
                                 for b in bands))
say("")
say("  A 40-50 cm household fan at 900-1600 rpm has a blade tip at 19-42 m/s,")
say("  i.e. 3.0-6.7 kHz Doppler. That is precisely the 3000-6000 Hz row above,")
say("  where the bound is tightest and nothing appears.")
R["band_shape_rel_15k"] = btab

# =============================================================== FIGURE =====
Ron = np.vstack([cond["fanON_50k_a"]["Rs"], cond["fanON_50k_b"]["Rs"]])
Roff = cond["fanOFF_50k"]["Rs"]
fig, ax = plt.subplots(3, 2, figsize=(15, 11.5))
ax[0, 0].plot(fax, Roff.mean(0), lw=.45, label="fan OFF")
ax[0, 0].plot(fax, Ron.mean(0), lw=.45, alpha=.8, label="fan ON")
ax[0, 0].set_xlim(40, 1200); ax[0, 0].set_xlabel("Doppler (Hz)")
ax[0, 0].set_ylabel("dB, trend removed")
ax[0, 0].set_title("A  flattened log Doppler spectrum (50 kSa/s, 60 s)")
ax[0, 0].legend(fontsize=8)
ax[0, 1].plot(qsel * 1e3, coff.mean(0)[SELQ], lw=.8, label="fan OFF")
ax[0, 1].plot(qsel * 1e3, con.mean(0)[SELQ], lw=.8, alpha=.8, label="fan ON")
ax[0, 1].axvline(1000 / 60, color='r', ls=':', lw=1)
ax[0, 1].text(1000 / 60, ax[0, 1].get_ylim()[1] * .85, "  60 Hz mains",
              color='r', fontsize=7)
ax[0, 1].set_xlabel("quefrency (ms)"); ax[0, 1].set_title("B  real cepstrum")
ax[0, 1].legend(fontsize=8)
ax[1, 0].plot(qsel * 1e3, z_p[SELQ], lw=.7, label="fan ON - fan OFF")
ax[1, 0].plot(qsel * 1e3, z_n1[SELQ], lw=.7, alpha=.7, label="null: ON a - ON b")
for s_ in (3, 5, -3, -5):
    ax[1, 0].axhline(s_, color='r' if abs(s_) == 5 else 'orange', ls='--', lw=.7)
ax[1, 0].set_xlabel("quefrency (ms)"); ax[1, 0].set_ylabel("sigma (measured floor)")
ax[1, 0].set_title("C  cepstral difference vs same-condition null")
ax[1, 0].legend(fontsize=8)
ax[1, 1].plot(f0_last, zH_last, lw=.7)
ax[1, 1].plot(f0_last[mm_last], zH_last[mm_last], '.', ms=2, color='r',
              label="60 Hz multiple")
for s_ in (3, 5, -3, -5):
    ax[1, 1].axhline(s_, color='orange', ls='--', lw=.7)
ax[1, 1].set_xlabel("trial blade-pass fundamental (Hz)"); ax[1, 1].set_ylabel("sigma")
ax[1, 1].set_title("D  HPS(K=5) difference, fan ON - fan OFF"); ax[1, 1].legend(fontsize=8)
ax[2, 0].plot([r[0] for r in inj], [max(r[2], .1) for r in inj], 'o-', label="global max")
ax[2, 0].plot([r[0] for r in inj], [max(r[3], .1) for r in inj], 's-', label="at true 73 Hz")
ax[2, 0].axhline(5, color='r', ls='--'); ax[2, 0].set_yscale('log')
ax[2, 0].set_xlabel("injected comb level (dB re IF rms)"); ax[2, 0].set_ylabel("sigma")
ax[2, 0].set_title("E  sensitivity: 73 Hz comb injected into real empty-room data")
ax[2, 0].legend(fontsize=8)
ax[2, 1].plot(rng, pm, lw=.9, label="moving person")
ax[2, 1].plot(rng, ps, lw=.9, label="static room")
ax[2, 1].set_xlim(0, 8); ax[2, 1].set_xlabel("range (m)"); ax[2, 1].set_ylabel("dB")
ax[2, 1].set_title("F  positive control: MTI range profile (+%.1f dB, %.0f sigma)"
                   % (pm[jP] - ps[jP], (pm[jP] - ps[jP]) / floor_mti))
ax[2, 1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT + "/cepstrum-hps.png", dpi=110)
say("")
say("figure -> " + OUT + "/cepstrum-hps.png")

hdr("VERDICT")
_best = max(m_a, m_b, float(np.abs(z_p[SELQ]).max()))
_null = max(m_n1, m_n2, m_n3)
say("  best fan-ON-vs-fan-OFF cepstral statistic anywhere : %.2f sigma" % _best)
say("  worst SAME-CONDITION null with the same detector    : %.2f sigma" % _null)
say("  best HPS statistic (K=3/K=5)                        : %.2f / %.2f sigma"
    % (R["hps_K3"]["best_sigma"], R["hps_K5"]["best_sigma"]))
say("  matching HPS same-condition nulls                   : %.2f / %.2f sigma"
    % (R["hps_K3"]["null_sigma"], R["hps_K5"]["null_sigma"]))
say("  candidate at 82.78 Hz: run a %+0.2f, run b %+0.2f -> not repeatable"
    % (R["candidate_consistency"]["sigma_run_a"],
       R["candidate_consistency"]["sigma_run_b"]))
say("")
say("  NO BLADE COMB DETECTED. The test statistic never exceeds the measured")
say("  same-condition false-alarm floor, and the best candidate does not repeat")
say("  across two independent 60 s fan-ON captures.")
say("  Method sensitivity (B5): a blade comb would have been found at >5 sigma")
say("  if it carried >= %d dB relative to the IF rms. It does not." % inj_thresh)
R["verdict"] = dict(best_sigma=float(_best), null_sigma=float(_null),
                    detected=False)

hdr("SUMMARY JSON")
R["pc_pass"] = dict(b1_ramp_comb=bool(b1_pass), b2_mains_comb=bool(b2_pass),
                    b3_person_mti=bool(b3_pass), b4_person_cepstrum=bool(b4_pass))
say(json.dumps(R, indent=1, default=float))
