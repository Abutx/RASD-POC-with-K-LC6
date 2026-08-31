"""
Statistical detectors (fan-on vs fan-off), K-LC6 24 GHz.

Method: NO peak picking. Compare DISTRIBUTIONS of block-wise spectral power
between conditions -- per-bin mean shift, per-bin variance ratio, kurtosis,
two-sample KS -- with every significance measured against a floor obtained by
splitting a single condition in half (never an assumed floor).

Run:  cd C:/dev/klc6 && python C:/dev/klc6/out/fan/analysis/statistical.py
"""
from __future__ import annotations
import sys, os, json
import numpy as np
from scipy import stats
from scipy.signal import welch, decimate, get_window, butter, sosfiltfilt
sys.path.insert(0, r'C:/dev/klc6')
from klc6 import process as P

OUT = r'C:/dev/klc6/out/fan/analysis'
HZ_PER_MPS = P.HZ_PER_MPS          # 160.945
RNG = np.random.default_rng(0xC0FFEE)

F = dict(
    fan_cw0=r'C:/dev/klc6/out/fan/20260830_093545_fan_on_cw0_100k.npz',
    fan_cw1=r'C:/dev/klc6/out/fan/20260830_093545_fan_on_cw1_100k.npz',
    fan_fmcwA=r'C:/dev/klc6/out/fan/20260830_093545_fan_on_fmcw_cfgA.npz',
    fan_fmcwB=r'C:/dev/klc6/out/fan/20260830_093545_fan_on_fmcw_cfgB.npz',
    off_smoke=r'C:/dev/klc6/out/cw/smoke_static.npz',
    off_base60=r'C:/dev/klc6/out/baseline/20260829_061237_empty_baseline_60s.npz',
    fmcw_box=r'C:/dev/klc6/out/fmcw/raw_chirps_box_out.npz',
    fmcw_move=r'C:/dev/klc6/out/fmcw/raw_chirps_moving.npz',
)
LOG = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    LOG.append(s)


def cw(path):
    d = np.load(path, allow_pickle=True)
    return d['data'][0].astype(float), float(d['fs'])


# ======================================================================
# 0. Data hygiene: how many ADC codes are we actually working with?
# ======================================================================
say('=' * 78)
say('0. QUANTISATION / DATA HYGIENE')
say('=' * 78)
codeinfo = {}
for k in ('fan_cw0', 'fan_cw1', 'off_smoke', 'off_base60'):
    x, fs = cw(F[k])
    u, c = np.unique(x, return_counts=True)
    p = c / c.sum()
    H = -np.sum(p[p > 0] * np.log2(p[p > 0]))
    codeinfo[k] = dict(n_codes=int(u.size), entropy_bits=float(H),
                       rms_uV=float(x.std() * 1e6), dc_mV=float(x.mean() * 1e3),
                       dur_s=float(x.size / fs), fs=fs)
    say(f'{k:12s} fs={fs:7.0f} dur={x.size/fs:5.1f}s  codes={u.size}  '
        f'code-entropy={H:.3f} bit  rms={x.std()*1e6:6.1f} uV  DC={x.mean()*1e3:6.3f} mV')
    say(f'{"":12s}   code occupancy {np.round(p,4)}')
say('NOTE: fan-on sits on a DIFFERENT code grid (DC 11.85 mV) than either')
say('      fan-off reference (13.00 / 13.25 mV) and is more concentrated on one')
say('      code -> its quantisation-noise floor is intrinsically different.')
say('      Any raw broadband level comparison is confounded; all tests below')
say('      normalise each record by its own out-of-band floor.')


# ======================================================================
# 1. Block-PSD engine
# ======================================================================
def block_psd(x, fs, nper=16384, nblocks=None, start=0):
    """Non-overlapping (independent) periodogram blocks -> (f, Pblocks[nb,nf])."""
    x = np.asarray(x, float)
    w = get_window('hann', nper)
    U = (w ** 2).sum()
    nb = (x.size - start) // nper
    if nblocks:
        nb = min(nb, nblocks)
    seg = x[start:start + nb * nper].reshape(nb, nper)
    seg = seg - seg.mean(axis=1, keepdims=True)
    Xf = np.fft.rfft(seg * w, axis=1)
    Pb = (np.abs(Xf) ** 2) / (fs * U)
    f = np.fft.rfftfreq(nper, 1 / fs)
    return f, Pb


NORM_LO, NORM_HI = 15000., 24000.   # 93-149 m/s: far above any fan blade tip


def normalise(f, Pb, lo=None, hi=None):
    """Divide every block by its own high-band median power -> removes global
    gain / quantiser-floor differences between records, keeps spectral SHAPE."""
    lo = NORM_LO if lo is None else lo
    hi = NORM_HI if hi is None else hi
    m = (f >= lo) & (f <= hi)
    ref = np.median(Pb[:, m], axis=1, keepdims=True)
    return Pb / ref, ref[:, 0]


def mains_mask(f, tol=2.0, fmax=None):
    h = np.abs(f - np.round(f / 60.) * 60.)
    m = (h <= tol) & (f > 30)
    if fmax is not None:
        m &= (f <= fmax)
    return m


# ======================================================================
# 2. The statistic set (all distribution-based, no peak picking)
# ======================================================================
def per_bin_stats(PA, PB):
    LA, LB = 10 * np.log10(PA + 1e-30), 10 * np.log10(PB + 1e-30)
    d = dict()
    d['mean_shift_db'] = LA.mean(0) - LB.mean(0)
    t, p = stats.ttest_ind(LA, LB, axis=0, equal_var=False)   # Welch: handles unequal n/var
    d['t'] = t
    d['p_t'] = p
    d['var_ratio_db'] = 10 * np.log10(LA.var(0, ddof=1) / (LB.var(0, ddof=1) + 1e-30))
    ks = np.empty(PA.shape[1])
    pks = np.empty(PA.shape[1])
    for i in range(PA.shape[1]):
        ks[i], pks[i] = stats.ks_2samp(PA[:, i], PB[:, i])
    d['ks'] = ks
    d['p_ks'] = pks
    return d


def band_summary(f, d, bands):
    say(f'  {"band Hz":>14s} {"m/s":>13s} {"dmean dB":>9s} {"dvar dB":>8s} '
        f'{"medKS":>6s} {"KSp<.01 %":>9s} {"nbin":>5s}')
    rows = []
    for lo, hi in bands:
        m = (f >= lo) & (f < hi)
        if m.sum() < 2:
            continue
        row = dict(lo=lo, hi=hi, n=int(m.sum()),
                   dmean=float(np.median(d['mean_shift_db'][m])),
                   dvar=float(np.median(d['var_ratio_db'][m])),
                   ks=float(np.median(d['ks'][m])),
                   ksfrac=float(np.mean(d['p_ks'][m] < 0.01) * 100))
        rows.append(row)
        say(f'  {lo:6.0f}-{hi:<7.0f} {lo/HZ_PER_MPS:5.2f}-{hi/HZ_PER_MPS:<7.2f} '
            f'{row["dmean"]:+9.2f} {row["dvar"]:+8.2f} {row["ks"]:6.3f} '
            f'{row["ksfrac"]:9.1f} {row["n"]:5d}')
    return rows


BANDS = [(20, 60), (60, 120), (120, 250), (250, 500), (500, 1000), (1000, 2000),
         (2000, 4000), (4000, 8000), (8000, 15000)]

# ======================================================================
# 3. POSITIVE CONTROL -- FMCW cfgA, moving person vs empty (box_out)
# ======================================================================
say('')
say('=' * 78)
say('3. POSITIVE CONTROL: moving person (FMCW cfgA) -- distribution test')
say('=' * 78)
B_MEAS = 180e6


def mti_range_frames(path, order=3, nfft=4096, ramp=50.0):
    d = np.load(path, allow_pickle=True)
    ch = d['chirps'].astype(float)
    fs = float(d['fs'])
    S = B_MEAS * ramp
    t = np.arange(ch.shape[1])
    V = np.vander(t / ch.shape[1], order + 1)
    coef, *_ = np.linalg.lstsq(V, ch.T, rcond=None)
    ch = ch - (V @ coef).T                     # per-chirp poly detrend (ramp feedthrough)
    mti = np.diff(ch, axis=0)                  # consecutive-chirp cancellation
    w = np.hanning(mti.shape[1])
    X = np.fft.rfft(mti * w, n=nfft, axis=1)
    fb = np.fft.rfftfreq(nfft, 1 / fs)
    rng = P.beat_to_range(fb, S)
    return rng, np.abs(X) ** 2


rng, Pmove = mti_range_frames(F['fmcw_move'])
_, Pbox = mti_range_frames(F['fmcw_box'])
sel = (rng >= 0.4) & (rng <= 9.0)
rsel = rng[sel]
Pm, Pb_ = Pmove[:, sel], Pbox[:, sel]
say(f'frames: moving {Pm.shape[0]}, empty {Pb_.shape[0]}; {sel.sum()} range bins 0.4-9.0 m')

Lm, Lb = 10 * np.log10(Pm + 1e-30), 10 * np.log10(Pb_ + 1e-30)
shift = Lm.mean(0) - Lb.mean(0)


def splithalf_floor(L, nrep=400):
    n = L.shape[0]
    out = []
    for _ in range(nrep):
        idx = RNG.permutation(n)
        a, b = idx[:n // 2], idx[n // 2:2 * (n // 2)]
        out.append(L[a].mean(0) - L[b].mean(0))
    return np.array(out)


half_box = splithalf_floor(Lb)
scale = np.sqrt((1 / Pm.shape[0] + 1 / Pb_.shape[0]) / (2 / (Pb_.shape[0] // 2)))
floor_db = half_box.std(0) * scale
sig = shift / floor_db
say(f'measured split-half floor (empty condition, rescaled to nA/nB): '
    f'median {np.median(floor_db):.3f} dB')
i = int(np.argmax(sig))
say(f'BEST BIN: {rsel[i]:.2f} m  shift {shift[i]:+.2f} dB  floor {floor_db[i]:.2f} dB'
    f'  -> {sig[i]:.1f} sigma   ({sel.sum()} bins searched)')
for r0 in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 7.0):
    j = int(np.argmin(np.abs(rsel - r0)))
    say(f'   r={rsel[j]:4.2f} m  shift {shift[j]:+6.2f} dB  {sig[j]:6.1f} sigma')

j15 = int(np.argmin(np.abs(rsel - 1.5)))
ks, pks = stats.ks_2samp(Pm[:, j15], Pb_[:, j15])
ad = stats.anderson_ksamp([Pm[:, j15], Pb_[:, j15]])
say(f'  KS at 1.5 m: D={ks:.3f} p={pks:.2e}   Anderson-Darling A2k={ad.statistic:.2f} '
    f'p={ad.pvalue:.4f}')
say(f'  var ratio at 1.5 m: {10*np.log10(Lm[:,j15].var()/Lb[:,j15].var()):+.2f} dB')
say(f'  kurtosis at 1.5 m: moving {stats.kurtosis(Pm[:,j15]):+.2f}  '
    f'empty {stats.kurtosis(Pb_[:,j15]):+.2f}')

# aggregate KS -- sizes MUST be matched, KS D scales as sqrt(1/nA+1/nB)
NKS = 70   # frames drawn per side, identical for observation and null
def agg_ks(PA, PB, nrep=150):
    out = []
    for _ in range(nrep):
        ia = RNG.choice(PA.shape[0], NKS, replace=False)
        ib = RNG.choice(PB.shape[0], NKS, replace=False)
        out.append(np.mean([stats.ks_2samp(PA[ia, kk], PB[ib, kk]).statistic
                            for kk in range(0, PA.shape[1], 3)]))
    return np.array(out)
obs_ks_dist = agg_ks(Pm, Pb_)
null_agg = []
for _ in range(150):
    idx = RNG.permutation(Pb_.shape[0])
    a, b = idx[:NKS], idx[NKS:2 * NKS]
    null_agg.append(np.mean([stats.ks_2samp(Pb_[a, kk], Pb_[b, kk]).statistic
                             for kk in range(0, Pb_.shape[1], 3)]))
null_agg = np.array(null_agg)
obs_agg = obs_ks_dist.mean()
sig_agg = (obs_agg - null_agg.mean()) / null_agg.std()
say(f'  AGGREGATE mean-KS over all range bins: obs {obs_agg:.3f} vs null '
    f'{null_agg.mean():.3f} +/- {null_agg.std():.3f}  -> {sig_agg:.1f} sigma')
POS_PASS = bool(sig[i] >= 5.0 and 1.0 <= rsel[i] <= 2.5)
say(f'POSITIVE CONTROL PASSED = {POS_PASS}')

# ======================================================================
# 4. FAN, FMCW cfgA vs the SAME empty reference (box_out)
# ======================================================================
say('')
say('=' * 78)
say('4. FAN (FMCW cfgA) vs empty (box_out) -- identical pipeline to sec.3')
say('=' * 78)
_, Pfan = mti_range_frames(F['fan_fmcwA'])
Pf = Pfan[:, sel]
Lf = 10 * np.log10(Pf + 1e-30)
shift_f = Lf.mean(0) - Lb.mean(0)
scale_f = np.sqrt((1 / Pf.shape[0] + 1 / Pb_.shape[0]) / (2 / (Pb_.shape[0] // 2)))
floor_f = half_box.std(0) * scale_f
sig_f = shift_f / floor_f
say(f'frames: fan {Pf.shape[0]}, empty {Pb_.shape[0]}; floor median {np.median(floor_f):.3f} dB')
k = int(np.argmax(np.abs(sig_f)))
say(f'BEST |sigma| BIN: {rsel[k]:.2f} m  shift {shift_f[k]:+.2f} dB -> {sig_f[k]:+.1f} sigma'
    f'   ({sel.sum()} bins searched; |3 sigma| expected ~{sel.sum()*0.0027:.1f}x by chance)')
for r0 in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0):
    j = int(np.argmin(np.abs(rsel - r0)))
    ksj = stats.ks_2samp(Pf[:, j], Pb_[:, j])
    say(f'   r={rsel[j]:4.2f} m  shift {shift_f[j]:+6.2f} dB  {sig_f[j]:+6.1f} sigma  '
        f'KS={ksj.statistic:.3f} p={ksj.pvalue:.3f}')
obs_fan = agg_ks(Pf, Pb_).mean()
sig_fan_agg = (obs_fan - null_agg.mean()) / null_agg.std()
say(f'  AGGREGATE mean-KS: obs {obs_fan:.3f} vs null {null_agg.mean():.3f} '
    f'+/- {null_agg.std():.3f}  -> {sig_fan_agg:+.1f} sigma')

# ---- cfgB internal Doppler test (no cfgB fan-off reference exists) ----
d = np.load(F['fan_fmcwB'], allow_pickle=True)
cpis = d['cpis'].astype(float)
S_B = B_MEAS * float(d['ramp'])
fs_B = float(d['fs'])
prf = float(d['ramp'])
tt = np.arange(cpis.shape[2])
V = np.vander(tt / cpis.shape[2], 4)
flat = cpis.reshape(-1, cpis.shape[2])
coef, *_ = np.linalg.lstsq(V, flat.T, rcond=None)
flat = flat - (V @ coef).T
cp = flat.reshape(cpis.shape)
rd = []
for c in cp:
    r_m, vel, db = P.range_doppler(c, fs_B, S_B, prf, range_nfft=256, dopp_nfft=128)
    rd.append(db)
rd = np.array(rd)
rmask = (r_m >= 0.8) & (r_m <= 6.0)
lin = 10 ** (rd / 10.)
mov = (np.abs(vel) > 0.25) & (np.abs(vel) <= 2.5)
farv = np.abs(vel) > 2.5
E_mov = lin[:, mov][:, :, rmask].mean(axis=(1, 2))
E_noi = lin[:, farv][:, :, rmask].mean(axis=(1, 2))
ratio_db = 10 * np.log10(E_mov.mean() / E_noi.mean())
h = []
for _ in range(300):
    idx = RNG.permutation(len(E_mov))
    a, b = idx[:len(E_mov) // 2], idx[len(E_mov) // 2:]
    h.append(10 * np.log10(E_mov[a].mean() / E_mov[b].mean()))
say(f'  cfgB internal: Doppler band 0.25-2.5 m/s over the >2.5 m/s noise Doppler bins '
    f'(0.8-6 m) = {ratio_db:+.2f} dB; split-half floor {np.std(h):.3f} dB -> '
    f'{ratio_db/np.std(h):+.1f} sigma  [no cfgB fan-off reference exists: weak, internal only]')

# ======================================================================
# 5. CW statistical detectors -- fan-on vs fan-off
# ======================================================================
say('')
say('=' * 78)
say('5. CW: block-PSD distribution tests @ 100 kSa/s')
say('=' * 78)
NPER = 16384
x_f0, fs100 = cw(F['fan_cw0'])
x_f1, _ = cw(F['fan_cw1'])
x_off, _ = cw(F['off_smoke'])
x_b60, fs50 = cw(F['off_base60'])

f100, Pf0 = block_psd(x_f0, fs100, NPER)
_, Pf1 = block_psd(x_f1, fs100, NPER)
_, Poff = block_psd(x_off, fs100, NPER)
Pf0n, r0_ = normalise(f100, Pf0)
Pf1n, r1_ = normalise(f100, Pf1)
Poffn, ro = normalise(f100, Poff)
say(f'blocks @100kSa/s, {NPER} pt ({NPER/fs100*1e3:.0f} ms, {fs100/NPER:.2f} Hz bins): '
    f'fan_cw0 {Pf0.shape[0]}, fan_cw1 {Pf1.shape[0]}, fan_off_smoke {Poff.shape[0]}')
say(f'raw out-of-band floor (15-24 kHz): fan0 {10*np.log10(r0_.mean()):.2f} dB, '
    f'fan1 {10*np.log10(r1_.mean()):.2f} dB, off {10*np.log10(ro.mean()):.2f} dB '
    f'-- fan-on floor is {10*np.log10(r0_.mean()/ro.mean()):+.2f} dB vs off, an '
    f'instrument-state offset, normalised out below')

mm = mains_mask(f100, 2.0, 15000.)
say(f'mains tag: {mm.sum()} of {(f100<15000).sum()} bins below 15 kHz within 2 Hz of a '
    f'60 Hz multiple = {100*mm.sum()/max(1,(f100<15000).sum()):.2f}% FALSE-TAG RATE '
    f'(bin width {fs100/NPER:.2f} Hz)')

Pfan_all = np.vstack([Pf0n, Pf1n])
NB = Poffn.shape[0]


def run_compare(name, PA, PB, match=True):
    if match and PA.shape[0] > PB.shape[0]:
        PA = PA[RNG.choice(PA.shape[0], PB.shape[0], replace=False)]
    say(f'-- {name}:  nA={PA.shape[0]} nB={PB.shape[0]}')
    d = per_bin_stats(PA, PB)
    rows = band_summary(f100, d, BANDS)
    return d, rows


say('')
say('NULL 1 (same condition, same session, minutes apart): fan_cw0 vs fan_cw1')
dn1, rn1 = run_compare('null fan0 vs fan1', Pf0n, Pf1n)
say('')
say('NULL 2 (same condition, split in half): fan-off smoke first vs second half')
hh = Poffn.shape[0] // 2
dn2, rn2 = run_compare('null off-half vs off-half', Poffn[:hh], Poffn[hh:2 * hh])
say('')
say('TEST 1: FAN-ON (25+25 s) vs FAN-OFF smoke (6 s), block-count matched to 9 vs 9')
dt1, rt1 = run_compare('fan_on vs fan_off', Pfan_all, Poffn)
say('')
say('TEST 1b: same, using ALL 76 fan blocks (Welch t handles the unequal n)')
dt1b = per_bin_stats(Pfan_all, Poffn)
rt1b = band_summary(f100, dt1b, BANDS)


def band_scalar(f, PA, PB, lo, hi):
    m = (f >= lo) & (f < hi) & ~mains_mask(f, 2.0)
    return (10 * np.log10(PA[:, m].mean(1)), 10 * np.log10(PB[:, m].mean(1)), m)


MOTION = [('slow  0.12-0.75 m/s (20-120 Hz)', 20, 120),
          ('walk  0.75-3.1 m/s (120-500 Hz)', 120, 500),
          ('fast  3.1-12.4 m/s (0.5-2 kHz)', 500, 2000),
          ('blades 12.4-62 m/s (2-10 kHz)', 2000, 10000)]
say('')
say('Scalar band statistic (mains-notched), sigma vs MEASURED same-condition floor:')
say(f'  {"band":34s} {"fanON-fanOFF":>13s} {"floor":>8s} {"sigma":>7s}')
scalar_rows = []
pool = Pfan_all
for lab, lo, hi in MOTION:
    m = (f100 >= lo) & (f100 < hi) & ~mains_mask(f100, 2.0)
    a_on = 10 * np.log10(pool[:, m].mean(1))
    b_off = 10 * np.log10(Poffn[:, m].mean(1))
    obs = a_on.mean() - b_off.mean()
    nulls = []
    for _ in range(3000):
        ja = RNG.permutation(pool.shape[0])
        A = 10 * np.log10(pool[ja[:NB]][:, m].mean(1))
        B = 10 * np.log10(pool[ja[NB:2 * NB]][:, m].mean(1))
        nulls.append(A.mean() - B.mean())
    nulls = np.array(nulls)
    nulls2 = []
    for _ in range(3000):
        jb = RNG.permutation(Poffn.shape[0])
        h2 = Poffn.shape[0] // 2
        A = 10 * np.log10(Poffn[jb[:h2]][:, m].mean(1))
        B = 10 * np.log10(Poffn[jb[h2:2 * h2]][:, m].mean(1))
        nulls2.append(A.mean() - B.mean())
    nulls2 = np.array(nulls2)
    fl = np.sqrt(nulls.std() ** 2 + nulls2.std() ** 2)
    say(f'  {lab:34s} {obs:+13.3f} {fl:8.3f} {obs/fl:+7.2f}')
    scalar_rows.append(dict(band=lab, lo=lo, hi=hi, obs_db=float(obs),
                            floor_db=float(fl), sigma=float(obs / fl)))

say('')
say('MAINS bins (flagged as an ELECTRICAL-LOAD artefact, NOT a target):')
mains_rows = []
for hn in (1, 2, 3, 4, 5, 6, 8):
    fc = 60. * hn
    m = np.abs(f100 - fc) <= 1.6 * (fs100 / NPER)   # >= 1 bin width, else empty
    on = 10 * np.log10(Pfan_all[:, m].mean())
    off = 10 * np.log10(Poffn[:, m].mean())
    n0 = 10 * np.log10(Pf0n[:, m].mean())
    n1 = 10 * np.log10(Pf1n[:, m].mean())
    mains_rows.append(dict(hz=fc, mps=fc / HZ_PER_MPS, on=float(on), off=float(off),
                           delta=float(on - off), null=float(n0 - n1)))
    say(f'   {fc:5.0f} Hz ({fc/HZ_PER_MPS:6.3f} m/s): fan_on {on:+7.2f} dB, fan_off '
        f'{off:+7.2f} dB, delta {on-off:+6.2f} dB  (same-cond null {n0-n1:+.2f} dB)')

# ======================================================================
# 6. Cross-session control at 50 kSa/s
# ======================================================================
say('')
say('=' * 78)
say('6. CROSS-SESSION CONTROL @ 50 kSa/s (decimated) -- the honest floor')
say('=' * 78)


def dec2(x):
    return decimate(x, 2, ftype='fir', zero_phase=True)


xf_50 = np.concatenate([dec2(x_f0), dec2(x_f1)])
xo_50 = dec2(x_off)
f50, Pfan50 = block_psd(xf_50, fs50, 8192)
_, Poff50 = block_psd(xo_50, fs50, 8192)
_, Pb60 = block_psd(x_b60, fs50, 8192)


def nrm(f, Pb):
    m = (f >= 8000.) & (f <= 12000.)
    return Pb / np.median(Pb[:, m], axis=1, keepdims=True)


Pfan50n, Poff50n, Pb60n = nrm(f50, Pfan50), nrm(f50, Poff50), nrm(f50, Pb60)
say(f'blocks (8192 pt, {8192/fs50*1e3:.0f} ms): fan50 {Pfan50n.shape[0]}, '
    f'off_smoke50 {Poff50n.shape[0]}, base60 {Pb60n.shape[0]}')
MOT50 = [('20-120 Hz', 20, 120), ('120-500 Hz', 120, 500),
         ('500-2000 Hz', 500, 2000), ('2-10 kHz', 2000, 10000)]


def scal(f, Pb, lo, hi):
    m = (f >= lo) & (f < hi) & ~mains_mask(f, 2.0)
    return 10 * np.log10(Pb[:, m].mean(1))


say(f'  {"band":12s} {"OFFvsOFF x-session":>19s} {"FAN vs BASE60":>14s} {"FAN vs SMOKE":>13s}')
xrows = []
for lab, lo, hi in MOT50:
    a = scal(f50, Poff50n, lo, hi)
    b = scal(f50, Pb60n, lo, hi)
    c = scal(f50, Pfan50n, lo, hi)
    off_off = a.mean() - b.mean()
    fan_b60 = c.mean() - b.mean()
    fan_sm = c.mean() - a.mean()
    xrows.append(dict(band=lab, off_off=float(off_off), fan_b60=float(fan_b60),
                      fan_smoke=float(fan_sm)))
    say(f'  {lab:12s} {off_off:+19.3f} {fan_b60:+14.3f} {fan_sm:+13.3f}')
say('  If |FAN vs OFF| is not clearly larger than |OFF vs OFF across sessions|,')
say('  the fan-on/off difference is session drift, not the fan.')

# ======================================================================
# 6b. LONG-RECORD, MATCHED-fs comparison: fan (50 s) vs base60 (60 s)
#     -- the only pair with comparable averaging on both sides.
#     Cross-session drift is bounded by smoke50 vs base60 (both fan-OFF).
# ======================================================================
say('')
say('=' * 78)
say('6b. LONG-RECORD @50 kSa/s: fan(305 blk) vs base60(365 blk), with the')
say('    fan-OFF/fan-OFF cross-session pair as the drift reference')
say('=' * 78)
NM = 36     # match every comparison to the smallest available sample (smoke50)


def matched(PA, PB, n=NM, nrep=60):
    """KS and mean-shift with strictly matched sample sizes."""
    ksm, dm = [], []
    for _ in range(nrep):
        ia = RNG.choice(PA.shape[0], n, replace=False)
        ib = RNG.choice(PB.shape[0], n, replace=False)
        A, B = PA[ia], PB[ib]
        ksm.append(np.array([stats.ks_2samp(A[:, kk], B[:, kk]).statistic
                             for kk in range(0, PA.shape[1], 4)]))
        dm.append(10 * np.log10(A.mean(0)) - 10 * np.log10(B.mean(0)))
    return np.array(ksm).mean(0), np.array(dm).mean(0)


def selfnull(PA, n=NM, nrep=60):
    ksm = []
    for _ in range(nrep):
        idx = RNG.permutation(PA.shape[0])
        A, B = PA[idx[:n]], PA[idx[n:2 * n]]
        ksm.append(np.array([stats.ks_2samp(A[:, kk], B[:, kk]).statistic
                             for kk in range(0, PA.shape[1], 4)]))
    return np.array(ksm).mean(0)


ks_fan_b60, d_fan_b60 = matched(Pfan50n, Pb60n)
ks_off_b60, d_off_b60 = matched(Poff50n, Pb60n)
ks_self = selfnull(Pb60n)
fsub = f50[::4]
say(f'  {"band":12s} {"KS fan/base":>12s} {"KS off/base":>12s} {"KS base/base":>13s}'
    f' {"dB fan/base":>12s} {"dB off/base":>12s}')
lr_rows = []
for lab, lo, hi in [('20-120 Hz', 20, 120), ('120-500 Hz', 120, 500),
                    ('500-2000 Hz', 500, 2000), ('2-10 kHz', 2000, 10000),
                    ('10-24 kHz', 10000, 24000)]:
    m = (fsub >= lo) & (fsub < hi) & ~mains_mask(fsub, 2.0)
    md = (f50 >= lo) & (f50 < hi) & ~mains_mask(f50, 2.0)
    lr_rows.append(dict(band=lab, ks_fan=float(ks_fan_b60[m].mean()),
                        ks_off=float(ks_off_b60[m].mean()),
                        ks_self=float(ks_self[m].mean()),
                        db_fan=float(np.median(d_fan_b60[md])),
                        db_off=float(np.median(d_off_b60[md]))))
    say(f'  {lab:12s} {ks_fan_b60[m].mean():12.4f} {ks_off_b60[m].mean():12.4f}'
        f' {ks_self[m].mean():13.4f} {np.median(d_fan_b60[md]):+12.3f}'
        f' {np.median(d_off_b60[md]):+12.3f}')
say('  KS fan/base is compared against KS off/base -- BOTH are cross-session, so any')
say('  fan signal must show as fan/base > off/base. base/base is the within-session floor.')

# ---- cadence-velocity structure test (blade-pass periodicity) ----
say('')
say('Cadence-velocity structure test (FFT along the TIME axis of the spectrogram):')
say('a rotating fan should put a sharp line at the blade-pass rate; empty room should not.')
cvd_rows = []
for nm, xs, fsx in (('fan_on_50s', xf_50, fs50), ('empty_base60', x_b60, fs50),
                    ('empty_smoke6s', xo_50, fs50)):
    xp = P.preprocess(xs, fsx, hp_hz=20.0)
    t_, v_, Sdb = P.spectrogram_mps(xp, fsx, nperseg=2048, overlap=0.75)
    cad, C = P.cadence_velocity_diagram(Sdb, t_)
    ok = (cad > 0.3) & (cad < 60)
    Cn = C[ok] / np.median(C[ok])
    j = int(np.argmax(Cn))
    # measured floor: std of the cadence spectrum itself (many bins => multiple comparison)
    z = (Cn[j] - 1.0) / Cn.std()
    cvd_rows.append(dict(rec=nm, cad_hz=float(cad[ok][j]), excess=float(Cn[j]), z=float(z),
                         nbins=int(ok.sum())))
    say(f'  {nm:14s} strongest cadence {cad[ok][j]:7.3f} Hz  x{Cn[j]:5.2f} over median  '
        f'z={z:5.2f}  ({ok.sum()} cadence bins searched)')
say('  A genuine blade-pass line would be a large, record-specific z present ONLY in')
say('  fan_on. Compare the three rows.')

# ======================================================================
# 6c. THE STRONGEST CW TEST: two 60 s / 50 kSa/s fan-ON records that are
#     duration- and rate-MATCHED to the 60 s empty baseline.
#     Same-condition null = fan_A vs fan_B (identical n, same session).
# ======================================================================
say('')
say('=' * 78)
say('6c. MATCHED 60 s @ 50 kSa/s: fan_on x2 vs empty_baseline_60s')
say('=' * 78)
xa, _ = cw(r'C:/dev/klc6/out/fan/20260830_094820_fan_on_cw_50k_60s_0.npz')
xb, _ = cw(r'C:/dev/klc6/out/fan/20260830_095020_fan_on_cw_50k_60s_1.npz')
for nm, xx in (('fan60_A', xa), ('fan60_B', xb), ('empty60', x_b60)):
    u, c = np.unique(xx, return_counts=True)
    say(f'  {nm:9s} N={xx.size} dur={xx.size/fs50:.1f}s rms={xx.std()*1e6:6.1f} uV '
        f'DC={xx.mean()*1e3:6.3f} mV codes={u.size} occ={np.round(c/c.sum(),4)}')
NP2 = 8192
fM, PA_ = block_psd(xa, fs50, NP2)
_, PB_ = block_psd(xb, fs50, NP2)
_, PE_ = block_psd(x_b60, fs50, NP2)


def nrm2(f, Pb):
    m = (f >= 8000.) & (f <= 12000.)
    return Pb / np.median(Pb[:, m], axis=1, keepdims=True)


PAn, PBn, PEn = nrm2(fM, PA_), nrm2(fM, PB_), nrm2(fM, PE_)
NM2 = min(PAn.shape[0], PBn.shape[0], PEn.shape[0])
say(f'  blocks: fanA {PAn.shape[0]}, fanB {PBn.shape[0]}, empty {PEn.shape[0]}; '
    f'using {NM2} per side, {NP2} pt = {fs50/NP2:.2f} Hz bins')
PAn, PBn, PEn = PAn[:NM2], PBn[:NM2], PEn[:NM2]
PFn = np.vstack([PAn, PBn])

BANDS50 = [('0.12-0.75 m/s   20-120 Hz', 20, 120),
           ('0.75-3.11 m/s  120-500 Hz', 120, 500),
           ('3.1-12.4 m/s  0.5-2 kHz', 500, 2000),
           ('12.4-31 m/s     2-5 kHz', 2000, 5000),
           ('31-62 m/s      5-10 kHz', 5000, 10000),
           ('62-124 m/s    10-20 kHz', 10000, 20000)]
say('')
say(f'  {"band":28s} {"fan-empty dB":>12s} {"null A-B dB":>11s} {"floor":>7s} {"sigma":>7s}'
    f' {"KS f/e":>7s} {"KS A/B":>7s}')
m60_rows = []
for lab, lo, hi in BANDS50:
    m = (fM >= lo) & (fM < hi) & ~mains_mask(fM, 2.0)
    obs = 10 * np.log10(PFn[:, m].mean(1)).mean() - 10 * np.log10(PEn[:, m].mean(1)).mean()
    nullv = 10 * np.log10(PAn[:, m].mean(1)).mean() - 10 * np.log10(PBn[:, m].mean(1)).mean()
    # measured floor: bootstrap same-condition splits on BOTH sides, matched n
    r1, r2 = [], []
    for _ in range(400):
        j = RNG.permutation(2 * NM2)
        r1.append(10 * np.log10(PFn[j[:NM2]][:, m].mean(1)).mean()
                  - 10 * np.log10(PFn[j[NM2:2 * NM2]][:, m].mean(1)).mean())
        j2 = RNG.permutation(NM2)
        r2.append(10 * np.log10(PEn[j2[:NM2 // 2]][:, m].mean(1)).mean()
                  - 10 * np.log10(PEn[j2[NM2 // 2:]][:, m].mean(1)).mean())
    fl = np.sqrt(np.std(r1) ** 2 + (np.std(r2) / np.sqrt(2)) ** 2)
    ksfe = np.mean([stats.ks_2samp(PFn[:NM2, kk], PEn[:, kk]).statistic
                    for kk in np.flatnonzero(m)[::16]])
    ksab = np.mean([stats.ks_2samp(PAn[:, kk], PBn[:, kk]).statistic
                    for kk in np.flatnonzero(m)[::16]])
    m60_rows.append(dict(band=lab, obs_db=float(obs), null_db=float(nullv),
                         floor_db=float(fl), sigma=float(obs / fl),
                         ks_fan_empty=float(ksfe), ks_fan_fan=float(ksab)))
    say(f'  {lab:28s} {obs:+12.3f} {nullv:+11.3f} {fl:7.3f} {obs/fl:+7.2f}'
        f' {ksfe:7.4f} {ksab:7.4f}')
say('  NOTE the "null A-B" column: two fan-ON records taken minutes apart differ by')
say('  a comparable amount, which is what the floor is measuring.')

# per-bin scan across the whole 20 Hz - 20 kHz band, with the empirical
# null distribution taken from fan_A vs fan_B (identical n, same condition)
say('')
LF = 10 * np.log10(np.vstack([PAn, PBn]) + 1e-30)
LE = 10 * np.log10(PEn + 1e-30)
t_obs, _ = stats.ttest_ind(LF, LE, axis=0, equal_var=False)
t_null, _ = stats.ttest_ind(10 * np.log10(PAn + 1e-30), 10 * np.log10(PBn + 1e-30),
                            axis=0, equal_var=False)
scan = (fM >= 20) & (fM <= 20000)
nomains = scan & ~mains_mask(fM, 2.0)
say(f'PER-BIN SCAN 20 Hz-20 kHz, {scan.sum()} bins ({nomains.sum()} after mains notch):')
say(f'  |t| > 5 (fan vs empty), mains-notched : {int((np.abs(t_obs[nomains])>5).sum())} bins'
    f'   [expected by chance at 5 sigma: {nomains.sum()*5.7e-7:.4f}]')
say(f'  |t| > 5 (fan_A vs fan_B NULL), notched: {int((np.abs(t_null[nomains])>5).sum())} bins')
say(f'  |t| > 5 on MAINS bins (fan vs empty)  : '
    f'{int((np.abs(t_obs[scan & mains_mask(fM,2.0)])>5).sum())} of '
    f'{int((scan & mains_mask(fM,2.0)).sum())}')
worst = np.flatnonzero(nomains)[np.argsort(-np.abs(t_obs[nomains]))[:8]]
say('  strongest NON-mains bins (fan vs empty):')
for w in worst:
    say(f'     {fM[w]:9.2f} Hz  {fM[w]/HZ_PER_MPS:7.3f} m/s  t={t_obs[w]:+7.2f}  '
        f'null t at same bin = {t_null[w]:+6.2f}  d={LF[:,w].mean()-LE[:,w].mean():+6.2f} dB')
say('  If the null column is comparable, the bin is instrument state, not the fan.')
excess_bins = int((np.abs(t_obs[nomains]) > 5).sum())
null_bins = int((np.abs(t_null[nomains]) > 5).sum())

# ======================================================================
# 7. Non-Gaussianity / entropy / kurtosis
# ======================================================================
say('')
say('=' * 78)
say('7. SPECTRAL ENTROPY / KURTOSIS (structure tests)')
say('=' * 78)


def spec_entropy(Pb, f, lo, hi):
    m = (f >= lo) & (f < hi) & ~mains_mask(f, 2.0)
    p = Pb[:, m].mean(0)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)) / np.log(m.sum()))


say(f'  {"band":12s} {"H_fanON":>9s} {"H_fanOFF":>9s} {"H_fan0":>8s} {"H_fan1":>8s}'
    f'   (1.0 = featureless)')
ent_rows = []
for lab, lo, hi in MOT50:
    hf = spec_entropy(Pfan_all, f100, lo, hi)
    ho = spec_entropy(Poffn, f100, lo, hi)
    h0 = spec_entropy(Pf0n, f100, lo, hi)
    h1 = spec_entropy(Pf1n, f100, lo, hi)
    ent_rows.append(dict(band=lab, H_fan=hf, H_off=ho))
    say(f'  {lab:12s} {hf:9.5f} {ho:9.5f} {h0:8.5f} {h1:8.5f}')
say('  (H_off uses 9 blocks vs 76 for the fan; fewer averages ALWAYS lowers')
say('   entropy, so H_off < H_fan is expected even with no fan.)')

say('')
say(f'  {"band":14s} {"kurt fanON":>11s} {"kurt fanOFF":>12s} {"kurt base60":>12s}')
kurt_rows = []
for lab, lo, hi in MOT50:
    sos = butter(4, [lo, hi], btype='band', fs=fs100, output='sos')
    kon = stats.kurtosis(sosfiltfilt(sos, x_f0))
    koff = stats.kurtosis(sosfiltfilt(sos, x_off))
    sos5 = butter(4, [lo, min(hi, 20000)], btype='band', fs=fs50, output='sos')
    kb = stats.kurtosis(sosfiltfilt(sos5, x_b60))
    kurt_rows.append(dict(band=lab, on=float(kon), off=float(koff), base=float(kb)))
    say(f'  {lab:14s} {kon:+11.4f} {koff:+12.4f} {kb:+12.4f}')

# ======================================================================
# 7b. WHAT WE COULD HAVE SEEN -- 5-sigma sensitivity limits
# ======================================================================
say('')
say('=' * 78)
say('7b. SENSITIVITY: 5-sigma upper limits on any fan return')
say('=' * 78)
say('FMCW cfgA (MTI, 0.4-9 m): measured split-half floor = '
    f'{np.median(floor_f):.3f} dB per range bin.')
say(f'  -> 5-sigma upper limit on fan MTI excess = {5*np.median(floor_f):.2f} dB.')
say(f'  -> the moving person gave {shift[i]:+.2f} dB in the same bins, i.e. the fan '
    f'returns at least {shift[i]-5*np.median(floor_f):.1f} dB less MTI energy than a walking person.')
lim_rows = []
for r in scalar_rows:
    lim_rows.append(dict(band=r['band'], limit5_db=float(5 * r['floor_db'])))
    say(f'CW {r["band"]:34s}: 5-sigma limit {5*r["floor_db"]:5.2f} dB band-power excess')
# equivalent single-tone sensitivity in the CW record
mband = (f100 >= 500) & (f100 < 2000) & ~mains_mask(f100, 2.0)
noise_bin = Pfan_all[:, mband].mean() * (ro.mean())      # de-normalise to V^2/Hz
enbw = fs100 / NPER * 1.5
say(f'CW per-bin noise (0.5-2 kHz) = {10*np.log10(noise_bin):.1f} dB(V^2/Hz), ENBW {enbw:.1f} Hz;')
say(f'  with {Pfan_all.shape[0]} averaged blocks a coherent tone of amplitude '
    f'{np.sqrt(2*noise_bin*enbw*5/np.sqrt(Pfan_all.shape[0]))*1e6:.2f} uV would reach 5 sigma '
    f'(record rms is {x_f0.std()*1e6:.0f} uV, LSB 336 uV).')

# mains-harmonic significance, measured against the same-condition null
say('')
say('Mains-harmonic deltas vs the same-condition (fan0 vs fan1) null spread:')
nullm = []
for hn in (1, 2, 3, 4, 5, 6, 8):
    fc = 60. * hn
    m = np.abs(f100 - fc) <= 1.6 * (fs100 / NPER)
    if m.sum() == 0:
        continue
    reps = []
    for _ in range(1000):
        ja = RNG.permutation(Pfan_all.shape[0])
        A = 10 * np.log10(Pfan_all[ja[:NB]][:, m].mean())
        B = 10 * np.log10(Pfan_all[ja[NB:2 * NB]][:, m].mean())
        reps.append(A - B)
    reps2 = []
    for _ in range(1000):
        jb = RNG.permutation(Poffn.shape[0])
        h2 = Poffn.shape[0] // 2
        reps2.append(10 * np.log10(Poffn[jb[:h2]][:, m].mean())
                     - 10 * np.log10(Poffn[jb[h2:2 * h2]][:, m].mean()))
    fl = np.sqrt(np.std(reps) ** 2 + np.std(reps2) ** 2)
    dlt = 10 * np.log10(Pfan_all[:, m].mean()) - 10 * np.log10(Poffn[:, m].mean())
    nullm.append(dict(hz=fc, delta=float(dlt), floor=float(fl), sigma=float(dlt / fl)))
    say(f'   {fc:5.0f} Hz ({fc/HZ_PER_MPS:6.3f} m/s): delta {dlt:+6.2f} dB, floor {fl:5.2f} dB'
        f' -> {dlt/fl:+6.2f} sigma')
say('   (7 harmonics tested; these are MAINS bins -- a fan is an electrical load,')
say('    so a rise here is conducted/radiated supply pickup, NOT a Doppler return.)')

# ======================================================================
# 8. VERDICT
# ======================================================================
say('')
say('=' * 78)
say('8. VERDICT')
say('=' * 78)
best = max(scalar_rows, key=lambda r: abs(r['sigma']))
say(f'CW best band {best["band"]}: {best["obs_db"]:+.2f} dB, {best["sigma"]:+.2f} sigma')
say(f'FMCW fan best bin: {shift_f[k]:+.2f} dB at {rsel[k]:.2f} m -> {sig_f[k]:+.1f} sigma')
say(f'FMCW positive control: {shift[i]:+.2f} dB at {rsel[i]:.2f} m -> {sig[i]:.1f} sigma')

# ---------------- figure ----------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 2, figsize=(14, 9))
a = ax[0, 0]
a.plot(rsel, shift, label=f'moving person (peak {sig[i]:.0f}$\\sigma$)', lw=1.6)
a.plot(rsel, shift_f, label=f'fan ON (peak {sig_f[k]:+.1f}$\\sigma$)', lw=1.6)
a.fill_between(rsel, -3 * floor_f, 3 * floor_f, alpha=.25, color='k',
               label='$\\pm3\\sigma$ measured floor')
a.set_xlabel('range m')
a.set_ylabel('MTI power vs empty room, dB')
a.set_title('FMCW cfgA: distribution mean-shift vs empty (box_out)')
a.legend(fontsize=8)
a.grid(alpha=.3)
a = ax[0, 1]
a.semilogx(f100[1:], 10 * np.log10(Pfan_all.mean(0)[1:]), lw=.7, label='fan ON (76 blk)')
a.semilogx(f100[1:], 10 * np.log10(Poffn.mean(0)[1:]), lw=.7, label='fan OFF (9 blk)')
for hn in range(1, 9):
    a.axvline(60 * hn, color='r', ls=':', lw=.6)
a.set_xlim(10, 25000)
a.set_xlabel('Hz')
a.set_ylabel('dB rel. own 15-24 kHz floor')
a.set_title('CW normalised block-mean PSD (red = 60 Hz mains)')
a.legend(fontsize=8)
a.grid(alpha=.3)
a = ax[1, 0]
a.semilogx(fM[1:], t_obs[1:], lw=.5, color='C3', label='fan ON (2x60 s) vs empty 60 s')
a.semilogx(fM[1:], t_null[1:], lw=.5, color='C0', alpha=.8, label='NULL fan_A vs fan_B')
for lv in (5, -5):
    a.axhline(lv, color='k', ls='--', lw=.7)
a.set_xlim(1, 22000)
a.set_ylim(-10, 10)
a.axhline(0, color='k', lw=.5)
a.set_xlabel('Hz')
a.set_ylabel('Welch t (dB units)')
a.set_title('Matched 60 s @50 kSa/s per-bin t: all excursions NEGATIVE', fontsize=10)
a.legend(fontsize=8)
a.grid(alpha=.3)
a = ax[1, 1]
bl = [r['band'].split()[-2] + ' ' + r['band'].split()[-1] for r in m60_rows]
yv = [r['sigma'] for r in m60_rows]
a.barh(range(len(yv)), yv, color=['C3' if v < 0 else 'C2' for v in yv])
a.axvline(5, color='k', ls='--', lw=.8)
a.axvline(-5, color='k', ls='--', lw=.8)
a.set_yticks(range(len(yv)))
a.set_yticklabels(bl, fontsize=8)
a.set_xlabel('sigma (fan ON - empty), measured floor')
a.set_title('Matched 60 s band statistic: no positive excess in any band')
a.grid(alpha=.3, axis='x')
fig.suptitle('K-LC6 fan detection: statistical detectors vs baseline', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'statistical.png'), dpi=110)
say(f'figure -> {OUT}/statistical.png')
with open(os.path.join(OUT, 'statistical_log.txt'), 'w') as fh:
    fh.write('\n'.join(LOG))
json.dump(dict(scalar=scalar_rows, matched60=m60_rows, scan_excess_bins=excess_bins, scan_null_bins=null_bins, longrec=lr_rows, cvd=cvd_rows, mains_sigma=nullm, limits=lim_rows, xsession=xrows, entropy=ent_rows, kurt=kurt_rows,
               mains=mains_rows,
               pos_sigma=float(sig[i]), pos_range=float(rsel[i]), pos_shift=float(shift[i]),
               fan_fmcw_sigma=float(sig_f[k]), fan_fmcw_range=float(rsel[k]),
               fan_fmcw_shift=float(shift_f[k]), codes=codeinfo),
          open(os.path.join(OUT, 'statistical.json'), 'w'), indent=1)
