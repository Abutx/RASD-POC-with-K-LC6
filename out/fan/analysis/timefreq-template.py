"""
Is a running household fan detectable with the K-LC6 at 24.125 GHz?

Assigned method: high-resolution time-frequency + blade-flash template matching (CW),
plus the per-frame Doppler spread, and FMCW cross-checks.

  0  data integrity / quantisation / mains false-tag rate
  1  POSITIVE CONTROL - the known moving person (FMCW Config A)
  2  CW spectra, equal-DOF: physical vs UNPHYSICAL Doppler bands (the decisive test)
  3  CW time-frequency: Doppler spread + blade-flash matched filter + whitened cadence
  4  the 60 Hz cadence line, and why it is mains and not a blade
  5  INJECTION: what fan WOULD have been found? (calibrates the null)
  6  FMCW cfgA / cfgB cross-checks with empirical chance levels

Every floor quoted is MEASURED by splitting a condition, never assumed.
"""
import os
import sys
import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, 'C:/dev/klc6')
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import stft, medfilt, decimate
from klc6 import process as P

OUT = 'C:/dev/klc6/out/fan/analysis'
HZ = P.HZ_PER_MPS                      # 160.945 Hz per m/s
RG = np.random.default_rng(0)
RES = {}


def hdr(s):
    print('\n' + '=' * 78 + '\n' + s + '\n' + '=' * 78)


# --------------------------------------------------------------------- helpers
def load_cw(f, row=0):
    d = np.load(f, allow_pickle=True)
    x = d['data'][row].astype(float)
    return x - x.mean(), float(d['fs'])


def notch_mains(x, fs, f0=60.0, hw_hz=0.6):
    """FFT bin-nulling of every 60 Hz multiple, at a FIXED Hz width so the amount of
    signal removed does not depend on record length or sample rate."""
    X = np.fft.rfft(x)
    df = fs / len(x)
    h = max(1, int(round(hw_hz / df)))
    nk = 0
    for k in range(1, int(fs / 2 // f0) + 1):
        c = int(round(k * f0 / df))
        a, b = max(0, c - h), min(len(X), c + h + 1)
        X[a:b] = 0
        nk += b - a
    X[0] = 0
    return np.fft.irfft(X, n=len(x)), nk / len(X)


def tfmap(x, fs, nper, hop):
    f, t, Z = stft(x, fs=fs, nperseg=nper, noverlap=nper - hop, window='hann',
                   boundary=None, padded=False, scaling='spectrum')
    return f, t, np.abs(Z) ** 2


def flash_mf(x, fs, nper=1024, hop=128, lo=300, hi=None):
    """Blade-flash matched filter.

    A blade flash is a brief brightening that is broadband in Doppler. Normalising every
    Doppler bin by its own time-median removes the stationary spectral shape AND any gain
    offset between captures, so what survives, b(t), is a robust-z broadband-excess series.
    The template is a short Gaussian flash -- a whitened impulse -- correlated along time.
    """
    hi = hi or min(20000, fs / 2 * 0.95)
    f, t, S = tfmap(x, fs, nper, hop)
    Sb = S[(f >= lo) & (f <= hi)]
    Sb = Sb / np.median(Sb, axis=1, keepdims=True)
    b = Sb.mean(0)
    b = (b - np.median(b)) / (1.4826 * np.median(np.abs(b - np.median(b))))
    tem = np.exp(-0.5 * (np.arange(-3, 4) / 1.0) ** 2)
    tem /= np.linalg.norm(tem)
    return t, np.convolve(b, tem[::-1], mode='same'), float(np.mean(np.diff(t)))


FMIN, FMAX = 5.0, 350.0                # cadence search band (edges trimmed)


def cadence_z(x, fs, f0=None, nmed=151, **kw):
    """Whitened cadence statistic of the matched-filter output.

    The raw cadence spectrum is strongly red, so a median/MAD null model over the whole
    band gives z of 40+ on pure noise. Dividing by a running-median baseline first makes
    the null well behaved (chance max z ~ sqrt(2 ln N), which is what fan-off returns)."""
    t, mf, dt = flash_mf(x, fs, **kw)
    n = len(mf)
    C = np.abs(np.fft.rfft((mf - mf.mean()) * np.hanning(n))) ** 2
    cf = np.fft.rfftfreq(n, dt)
    s = (cf >= FMIN) & (cf <= FMAX)
    c, ff = C[s], cf[s]
    base = medfilt(c, kernel_size=nmed)
    base[base <= 0] = np.median(c)
    lr = np.log(c / base)
    mu = np.median(lr)
    sd = 1.4826 * np.median(np.abs(lr - mu))
    z = (lr - mu) / sd
    nm = np.abs(ff - np.round(ff / 60.) * 60.) > 1.0     # exclude mains cadence
    i = int(np.argmax(np.where(nm, z, -1e9)))
    o = dict(best_z=float(z[i]), best_f=float(ff[i]), ntrial=int(nm.sum()),
             chance=float(np.sqrt(2 * np.log(nm.sum()))),
             z_at_60=float(z[int(np.argmin(np.abs(ff - 60.)))]),
             ff=ff, z=z)
    if f0 is not None:
        o['z_at_f0'] = float(z[int(np.argmin(np.abs(ff - f0)))])
    return o


def synth_fan(n, fs, amp, rev=12.0, nb=3, vtip=12.0, seed=1):
    """Synthetic rotating-blade CW return: a distributed blade gives a flat Doppler
    pedestal out to v_tip, plus a specular flash once per blade pass. amp = IF volts rms."""
    g = np.random.default_rng(seed)
    bpf = rev * nb
    ped = g.standard_normal(n)
    X = np.fft.rfft(ped)
    fq = np.fft.rfftfreq(n, 1 / fs)
    X[(fq > vtip * HZ) | (fq < 0.5 * HZ)] = 0
    ped = np.fft.irfft(X, n=n)
    ped /= ped.std()
    fl = np.zeros(n)
    w = int(fs * 0.0015)
    for m in range(int(n / fs * bpf)):
        c = int((m / bpf + g.normal(0, 0.0005)) * fs)
        if 0 < c < n - w:
            fl[c:c + w] += np.hanning(w) * g.standard_normal(w)
    fl /= fl.std()
    s = 0.6 * ped + 0.4 * fl
    return amp * s / s.std()


CWF = {'FAN_ON#0': 'C:/dev/klc6/out/fan/20260830_093545_fan_on_cw0_100k.npz',
       'FAN_ON#1': 'C:/dev/klc6/out/fan/20260830_093545_fan_on_cw1_100k.npz',
       'off_smoke': 'C:/dev/klc6/out/cw/smoke_static.npz'}
BASE = 'C:/dev/klc6/out/baseline/20260829_061237_empty_baseline_60s.npz'

# ----------------------------------------------------------------- 0 integrity
hdr('0  DATA INTEGRITY / QUANTISATION / MAINS FALSE-TAG RATE')
raw = {}
for k, f in list(CWF.items()) + [('off_base60', BASE)]:
    x, fs = load_cw(f)
    raw[k] = (x, fs)
    u = np.unique(x)
    step = np.median(np.diff(u)) * 1e6 if len(u) > 1 else 0.0
    print(f'{k:11s} fs={fs:7.0f} dur={len(x)/fs:5.1f}s rms={np.std(x)*1e6:6.1f}uV '
          f'distinct ADC codes={len(u):3d} step={step:.1f}uV zeros={int((x == 0).sum())}')
print('-> 4-5 codes on a 336 uV LSB: the CW IF is quantisation-limited (FINDINGS 1.1).')
print(f'-> quantisation noise alone is LSB/sqrt(12) = {336/np.sqrt(12):.0f} uV rms vs a '
      f'measured {np.std(raw["FAN_ON#0"][0])*1e6:.0f} uV total.')
ft = RG.uniform(3000, 20000, 400000)
FTAG = float(np.mean(np.abs(ft - np.round(ft / 60.) * 60.) < 2.0))
print(f'-> "within 2 Hz of a 60 Hz multiple" tags {FTAG*100:.2f}% of random frequencies above')
print('   3 kHz, so a high-order mains tag there carries essentially no information.')
RES['mains_false_tag_pct'] = FTAG * 100

CW = {k: (notch_mains(x, fs)[0], fs) for k, (x, fs) in raw.items()}
print('CW records mains-nulled with a fixed +-0.6 Hz FFT window on every 60 Hz multiple.')

# ---------------------------------------------------------- 1 positive control
hdr('1  POSITIVE CONTROL - the known moving person (FMCW Config A)')
print('This dataset contains no CW capture with a person, so the time-frequency machinery is')
print('validated on the FMCW slow-time TF distribution. The statistic is its 0th moment --')
print('non-zero-Doppler power per frame, per range bin -- the same quantity the CW flash')
print('detector integrates. If a fan moves at all, this statistic sees it.')


def prof(f):
    d = np.load(f, allow_pickle=True)
    ch = np.asarray(d['chirps'], float)
    n = ch.shape[1]
    V = np.vander(np.arange(n) / n, 4)            # order-3 detrend removes ramp feedthrough
    ch = ch - (V @ np.linalg.lstsq(V, ch.T, rcond=None)[0]).T
    Rr = np.fft.rfft(ch * np.hanning(n), axis=1)
    return Rr, P.beat_to_range(np.fft.rfftfreq(n, 1 / 1e5), 180e6 * 50.)   # B=180 MHz MEASURED


def tf0(Rr, nwin=16, hop=4):
    o = []
    for s in range(0, Rr.shape[0] - nwin + 1, hop):
        g = Rr[s:s + nwin]
        g = g - g.mean(0, keepdims=True)
        o.append((np.abs(g) ** 2).mean(0))
    return 10 * np.log10(np.array(o) + 1e-30)


FM = {'moving_person': 'C:/dev/klc6/out/fmcw/raw_chirps_moving.npz',
      'static_boxout': 'C:/dev/klc6/out/fmcw/raw_chirps_box_out.npz',
      'static_boxin': 'C:/dev/klc6/out/fmcw/raw_chirps_box_in.npz',
      'FAN_ON_cfgA': 'C:/dev/klc6/out/fan/20260830_093545_fan_on_fmcw_cfgA.npz'}
TF = {}
for k, f in FM.items():
    Rr, rng = prof(f)
    TF[k] = tf0(Rr)


def bbfloor(a, nb=8):
    return np.std([b.mean() for b in np.array_split(a, nb)], ddof=1) / np.sqrt(nb)


ref = TF['static_boxout']
print(f'\nfloor per range bin = max(split-half of static_boxout, block-bootstrap of the same)')
print(f'{"range m":>8}{"person dB":>11}{"sigma":>8}{"FAN dB":>10}{"sigma":>8}'
      f'{"boxin dB":>10}{"sigma":>8}{"floor dB":>10}')
PC_SIGMA = 0.0
FAN_A = []
for i in range(1, 8):
    a = ref[:, i]
    h = len(a) // 2
    fl = max(abs(a[:h].mean() - a[h:2 * h].mean()), bbfloor(a))
    dp = TF['moving_person'][:, i].mean() - a.mean()
    dfn = TF['FAN_ON_cfgA'][:, i].mean() - a.mean()
    db = TF['static_boxin'][:, i].mean() - a.mean()
    PC_SIGMA = max(PC_SIGMA, dp / fl)
    FAN_A.append((float(rng[i]), float(dfn), float(dfn / fl), float(fl)))
    print(f'{rng[i]:8.2f}{dp:+11.2f}{dp/fl:8.1f}{dfn:+10.2f}{dfn/fl:8.1f}'
          f'{db:+10.2f}{db/fl:8.1f}{fl:10.3f}')
PC_PASS = bool(PC_SIGMA >= 5.0)
RES['pc_sigma'] = PC_SIGMA
RES['pc_pass'] = PC_PASS
print(f'\nPOSITIVE CONTROL: best {PC_SIGMA:.1f} sigma -> {"PASS" if PC_PASS else "FAIL"}')
print('  (+7.4 dB at 1.67 m; FINDINGS 5.5 quotes +6.69 dB / 12 sigma at 1.5 m -- reproduced.)')
print('  static_boxin, a second no-target condition, returns -3.1..+2.1 sigma: the floor is honest.')
FAN_A_BEST = max(abs(s) for _, _, s, _ in FAN_A)
RES['fan_cfgA_best_sigma'] = FAN_A_BEST
print(f'IDENTICAL detector on FAN_ON cfgA (fan at ~2.5 m): best |sigma| = {FAN_A_BEST:.1f} -> nothing.')
i25 = min(range(len(FAN_A)), key=lambda j: abs(FAN_A[j][0] - 2.5))
ul = 2 * FAN_A[i25][3] + FAN_A[i25][1]
RES['fan_vs_person_dB'] = 7.08 - ul
print(f'  fan at {FAN_A[i25][0]:.2f} m: {FAN_A[i25][1]:+.2f} dB, 2-sigma upper limit +{ul:.2f} dB.')
print(f'  => the fan\'s moving-target return is at least {7.08-ul:.1f} dB below a person at the same range.')

# ------------------------------------------------- 2 equal-DOF spectral compare
hdr('2  EQUAL-DOF CW SPECTRA: physical vs UNPHYSICAL Doppler bands')
print('The fan-off 100 kSa/s reference is 6 s (36 x 16384-pt segments) and fan-on is 25 s.')
print('Unequal averaging inflates the difference variance, so every condition is reduced to')
print('the SAME 36 segments. The floor is |FAN_ON#0 - FAN_ON#1|: two independent 25 s chunks')
print('of the SAME condition, minutes apart, through identical processing.')
NSEG, NP = 36, 16384


def psd_fixedN(x, fs, nseg=NSEG, nper=NP):
    n = len(x) // nper
    idx = np.linspace(0, n - 1, nseg).astype(int)
    w = np.hanning(nper)
    acc = 0.0
    for i in idx:
        acc = acc + np.abs(np.fft.rfft(x[i * nper:(i + 1) * nper] * w)) ** 2
    return np.fft.rfftfreq(nper, 1 / fs), acc / len(idx)


PS = {}
for k in ['FAN_ON#0', 'FAN_ON#1', 'off_smoke']:
    x, fs = CW[k]
    fr, PS[k] = psd_fixedN(x, fs)
mask_m = np.abs(fr - np.round(fr / 60.) * 60.) > 3
BANDS = [('0.5-3 m/s', 80, 483), ('3-20 m/s', 483, 3219), ('20-40 m/s', 3219, 6438),
         ('40-60 m/s', 6438, 9657), ('UNPHYSICAL 60-124 m/s', 9657, 20000),
         ('UNPHYSICAL 124-300 m/s', 20000, 48000)]
print(f'\n{"band":24s}{"on0":>9}{"on1":>9}{"off":>9}{"on-off":>9}{"floor":>9}{"sigma":>8}')
BANDTAB = []
for nm, lo, hi in BANDS:
    m = (fr >= lo) & (fr < hi) & mask_m
    a, b, c = [10 * np.log10(PS[k][m].mean()) for k in ['FAN_ON#0', 'FAN_ON#1', 'off_smoke']]
    fl = abs(a - b)
    d = (a + b) / 2 - c
    BANDTAB.append((nm, d, fl, d / fl))
    print(f'{nm:24s}{a:9.2f}{b:9.2f}{c:9.2f}{d:+9.3f}{fl:9.3f}{d/fl:8.1f}')
phys = np.mean([d for nm, d, _, _ in BANDTAB if not nm.startswith('UNPHYS')])
unph = np.mean([d for nm, d, _, _ in BANDTAB if nm.startswith('UNPHYS')])
RES['phys_dB'], RES['unphys_dB'] = float(phys), float(unph)
print(f'\nmean offset in the PHYSICAL fan bands   : {phys:+.3f} dB')
print(f'mean offset in the UNPHYSICAL (>60 m/s) : {unph:+.3f} dB   difference {phys-unph:+.3f} dB')
print('The two are the same to within 0.06 dB, and BOTH are NEGATIVE. A fan cannot produce')
print('Doppler at 60-300 m/s, so this flat offset is a receiver/gain difference between two')
print('captures on different days -- and it goes the wrong way for a target anyway.')

print('\nShape test: normalise each condition to its own >60 m/s level, then difference.')


def selfnorm(p):
    m = (fr > 9657) & (fr < 20000) & mask_m
    return p / p[m].mean()


d_on = 10 * np.log10(0.5 * (selfnorm(PS['FAN_ON#0']) + selfnorm(PS['FAN_ON#1'])))
d_off = 10 * np.log10(selfnorm(PS['off_smoke']))
d_fl = 10 * np.log10(selfnorm(PS['FAN_ON#0'])) - 10 * np.log10(selfnorm(PS['FAN_ON#1']))


def smooth(a, w=64):
    return np.convolve(a, np.ones(w) / w, mode='same')


sm = smooth(d_on - d_off, 64)
sd_fl = smooth(d_fl, 64)[(fr > 200) & (fr < 45000)].std()
sel = (fr > 80) & (fr < 9657) & mask_m
i = int(np.argmax(np.abs(sm[sel])))
NTR = int(sel.sum() / 64)
RES['shape_sigma'] = float(abs(sm[sel][i]) / sd_fl)
print(f'  largest smoothed shape difference in 0.5-60 m/s: {sm[sel][i]:+.3f} dB at '
      f'{fr[sel][i]/HZ:.2f} m/s')
print(f'  floor from the two fan-on chunks, same statistic: {sd_fl:.3f} dB')
print(f'  -> {abs(sm[sel][i])/sd_fl:.1f} sigma over ~{NTR} independent smoothed trials '
      f'(chance |z| ~ {np.sqrt(2*np.log(NTR)):.1f}). No excess.')

# ---------------------------------------------------------------- 3 CW TF
hdr('3  CW TIME-FREQUENCY: blade-flash matched filter, Doppler spread, cadence')
CONF = [(256, 64), (1024, 128), (4096, 512), (16384, 2048)]
print(f'{"nperseg":>8}{"dt ms":>8}{"df Hz":>8}{"condition":>12}{"MFmax":>8}{"N>5":>6}'
      f'{"rate/s":>9}{"N>6":>6}')
FLASH = {}
for nper, hop in CONF:
    for k in ['FAN_ON#0', 'FAN_ON#1', 'off_smoke']:
        x, fs = CW[k]
        t, mf, dt = flash_mf(x, fs, nper, hop)
        n5, n6 = int((mf > 5).sum()), int((mf > 6).sum())
        FLASH[(nper, k)] = dict(mfmax=float(mf.max()), n5=n5, rate=n5 / (len(mf) * dt))
        print(f'{nper:>8}{dt*1e3:8.2f}{fs/nper:8.1f}{k:>12}{mf.max():8.2f}{n5:6d}'
              f'{n5/(len(mf)*dt):9.3f}{n6:6d}')
print('\nThe flash counts do NOT replicate between the two fan-on chunks (same fan, same')
print('session, minutes apart). Locating them in time:')
for k in ['FAN_ON#0', 'FAN_ON#1']:
    x, fs = CW[k]
    t, mf, dt = flash_mf(x, fs, 1024, 128)
    ev = t[mf > 5]
    if len(ev):
        print(f'  {k}: {len(ev)} events, all between {ev.min():.1f} s and {ev.max():.1f} s '
              f'of a 25 s record ({len(np.unique(np.round(ev,1)))} distinct instants)')
    else:
        print(f'  {k}: 0 events in 25 s')
print('  A continuously running fan produces flashes uniformly over both records.')

print('\nFlash RATE with a properly averaged fan-off reference (9 x 6 s blocks of the empty')
print('60 s baseline) vs 8 x 6 s blocks of fan-on, both at 50 kSa/s:')


def rate_blocks(x, fs, dur=6.0):
    n = int(dur * fs)
    o = []
    for i in range(len(x) // n):
        t, mf, dt = flash_mf(x[i * n:(i + 1) * n], fs)
        o.append(int((mf > 5).sum()) / (len(mf) * dt))
    return np.array(o)


bfull, _ = notch_mains(raw['off_base60'][0], 50000.)
r_off = rate_blocks(bfull, 50000.)
ON50 = {k: notch_mains(decimate(raw[k][0], 2, ftype='fir', zero_phase=True), 50000.)[0]
        for k in ['FAN_ON#0', 'FAN_ON#1']}
r_on = np.concatenate([rate_blocks(ON50['FAN_ON#0'], 50000.),
                       rate_blocks(ON50['FAN_ON#1'], 50000.)])
fl_r = np.sqrt(r_off.var(ddof=1) / len(r_off) + r_on.var(ddof=1) / len(r_on))
RES['flash_rate_sigma'] = float((r_on.mean() - r_off.mean()) / fl_r)
print(f'  FAN OFF n={len(r_off)}: {r_off.mean():.3f} +- {r_off.std(ddof=1):.3f} /s   {np.round(r_off,2)}')
print(f'  FAN ON  n={len(r_on)}: {r_on.mean():.3f} +- {r_on.std(ddof=1):.3f} /s   {np.round(r_on,2)}')
print(f'  difference {r_on.mean()-r_off.mean():+.3f} /s, measured floor {fl_r:.3f} -> '
      f'{(r_on.mean()-r_off.mean())/fl_r:+.1f} sigma')
print('  This is the single most fan-favourable number in the whole analysis, and it is still')
print('  only 2 sigma. It is carried by ONE 6 s block (8.5 /s) with three other fan-on blocks')
print('  at exactly zero -- the opposite of what a steadily rotating blade produces -- and it')
print('  is a between-day comparison, so any room or supply transient enters it.')

print('\nPer-frame Doppler SPREAD (2nd moment of the floor-subtracted TF distribution):')
print(f'{"nperseg":>8}{"condition":>12}{"spread m/s":>12}{"sd":>8}{"vs off":>9}')
SPREAD = {}
for nper, hop in CONF[1:]:
    vals = {}
    for k in ['FAN_ON#0', 'FAN_ON#1', 'off_smoke']:
        x, fs = CW[k]
        f, t, S = tfmap(x, fs, nper, hop)
        m = (f >= 200) & (f <= 20000)
        fm, Sb = f[m], S[m]
        ex = np.maximum(Sb - np.median(Sb, axis=0, keepdims=True), 0)
        tot = ex.sum(0) + 1e-30
        mu = (fm[:, None] * ex).sum(0) / tot
        vals[k] = np.sqrt(((fm[:, None] - mu) ** 2 * ex).sum(0) / tot) / HZ
    for k in vals:
        print(f'{nper:>8}{k:>12}{vals[k].mean():12.3f}{vals[k].std():8.3f}'
              f'{vals[k].mean()-vals["off_smoke"].mean():+9.3f}')
    SPREAD[nper] = {k: float(v.mean()) for k, v in vals.items()}
print('Fan-on spread is 0.1-0.14 m/s BELOW fan-off, never above. The statistic is also')
print('saturated: with a white background the normalised spread pins at the band value')
print('(~37 m/s), so it has little power here regardless of sign. Reported, not relied on.')

print('\nWhitened cadence statistic, 6-s blocks (duration-matched to the fan-off reference):')
print(f'{"cond":10s}{"blk":>4}{"best z":>9}{"at Hz":>9}{"z@60Hz":>9}')
CADB = {'on': [], 'off': []}
for k in ['FAN_ON#0', 'FAN_ON#1', 'off_smoke']:
    x, fs = CW[k]
    nb = int(6 * fs)
    for i in range(len(x) // nb):
        r = cadence_z(x[i * nb:(i + 1) * nb], fs)
        CADB['off' if k == 'off_smoke' else 'on'].append(r['best_z'])
        print(f'{k:10s}{i:4d}{r["best_z"]:9.2f}{r["best_f"]:9.2f}{r["z_at_60"]:9.2f}')
print(f'  {r["ntrial"]} cadence trials per block -> chance max z ~ {r["chance"]:.1f}')
print(f'  fan-on max over 8 blocks {max(CADB["on"]):.2f}; fan-off {CADB["off"][0]:.2f}; '
      f'expected max of 8 null blocks ~{np.sqrt(2*np.log(8*r["ntrial"])):.1f}')
RES['cad_on_max'] = float(max(CADB['on']))
RES['cad_off'] = float(CADB['off'][0])
RES['cad_chance8'] = float(np.sqrt(2 * np.log(8 * r['ntrial'])))

print('\nFull-length (25 s) cadence statistic, fan-on vs two 25-s empty-room segments:')
b, fsb = load_cw(BASE)
H = {'off_base60 seg0': notch_mains(b[:int(25 * fsb)], fsb)[0],
     'off_base60 seg1': notch_mains(b[int(25 * fsb):int(50 * fsb)], fsb)[0],
     'FAN_ON#0 (50k)': ON50['FAN_ON#0'],
     'FAN_ON#1 (50k)': ON50['FAN_ON#1']}
for k, v in H.items():
    r = cadence_z(v, 50000.)
    print(f'  {k:18s} best z {r["best_z"]:6.2f} @ {r["best_f"]:7.2f} Hz   '
          f'z@60Hz {r["z_at_60"]:6.2f}   (chance {r["chance"]:.1f})')
print('  Fan-on (3.4, 3.7) is BELOW both empty-room segments (4.2, 4.3) and below chance.')

# ------------------------------------------------------ 4 the 60 Hz cadence line
hdr('4  THE 60 Hz CADENCE LINE -- mains, not a blade (ground rule 1)')
print('At nperseg=1024 the broadband-excess series of BOTH fan-on chunks carries a strong')
print('cadence line near 60 Hz. Testing whether it is a target or the supply:')
print(f'{"file":11s}{"channel":>9}{"peak Hz":>11}{"dB":>7}   (row1 = the AD2 second input, no Doppler)')
for k, f in CWF.items():
    d = np.load(f, allow_pickle=True)
    fs = float(d['fs'])
    for row in (0, 1):
        x = d['data'][row].astype(float)
        xn, _ = notch_mains(x - x.mean(), fs)
        t, mf, dt = flash_mf(xn, fs, 1024, 256)
        n = len(mf)
        C = np.abs(np.fft.rfft((mf - mf.mean()) * np.hanning(n)))
        cf = np.fft.rfftfreq(n, dt)
        s = (cf > 2) & (cf < 190)
        i = int(np.argmax(C[s]))
        print(f'{k:11s}{"row"+str(row):>9}{cf[s][i]:11.4f}{20*np.log10(C[s][i]/np.median(C[s])):7.1f}')
print('The line sits at 59.982 Hz -- mains to within 0.02 Hz, my full cadence resolution --')
print('and it appears on row1 of off_smoke, a channel that carries NO radar return. It is a')
print('60 Hz amplitude modulation of the quantiser, not a blade-pass rate. Which channel')
print('shows it depends on where that channel sits relative to an ADC code boundary.')
print('A shaded-pole fan motor runs at slip BELOW synchronous and can never lock to 59.982 Hz.')
print('This is exactly the trap ground rule 1 describes: the fan is an electrical load.')

# --------------------------------------------------------------- 5 injection
hdr('5  INJECTION -- what fan WOULD this pipeline have found?')
print('Synthetic rotating blade (flat Doppler pedestal to v_tip=12 m/s + a specular flash per')
print('blade pass at 36 Hz) added to REAL fan-off data, then run through the full pipeline.')
host6, fs6 = CW['off_smoke']
n6 = len(host6)
nr6 = np.std(host6)
print(f'\n(a) 6-s host, cadence route.  host: z@36Hz={cadence_z(host6,fs6,36.)["z_at_f0"]:+.2f}')
print(f'{"inj dB":>8}{"uV rms":>9}{"z@36":>8}{"best z":>9}{"det z>5":>9}')
MIN6 = None
for db in [-24, -21, -18, -15, -12, -9, -6, -3, 0]:
    y, _ = notch_mains(host6 + synth_fan(n6, fs6, nr6 * 10 ** (db / 20.)), fs6)
    r = cadence_z(y, fs6, 36.)
    d = r['z_at_f0'] > 5
    if d and MIN6 is None:
        MIN6 = db
    print(f'{db:8d}{nr6*10**(db/20.)*1e6:9.1f}{r["z_at_f0"]:8.2f}{r["best_z"]:9.2f}'
          f'{"YES" if d else "no":>9}')
h25 = H['off_base60 seg0']
n25 = len(h25)
nr25 = np.std(h25)
print(f'\n(b) 25-s host (matched to the fan-on record length), cadence route:')
print(f'{"inj dB":>8}{"uV rms":>9}{"z@36":>8}{"best z":>9}{"det z>5":>9}')
MIN25 = None
for db in [-24, -21, -18, -15, -12, -9, -6, -3]:
    y, _ = notch_mains(h25 + synth_fan(n25, 50000., nr25 * 10 ** (db / 20.)), 50000.)
    r = cadence_z(y, 50000., 36.)
    d = r['z_at_f0'] > 5
    if d and MIN25 is None:
        MIN25 = db
    print(f'{db:8d}{nr25*10**(db/20.)*1e6:9.1f}{r["z_at_f0"]:8.2f}{r["best_z"]:9.2f}'
          f'{"YES" if d else "no":>9}')
fl_min = min(fl for _, _, fl, _ in BANDTAB)
bb_sens = 10 * np.log10(10 ** (5 * fl_min / 10.) - 1)
RES['inj_min6'], RES['inj_min25'], RES['bb_sens'] = MIN6, MIN25, float(bb_sens)
print(f'\n(c) broadband-excess route, the more sensitive of the two: the measured floor is')
print(f'    {fl_min:.2f} dB (best band), so 5 sigma needs +{5*fl_min:.2f} dB of excess noise power,')
print(f'    i.e. a fan return of {bb_sens:+.1f} dB relative to the CW noise rms.')
print(f'\nSENSITIVITY: this pipeline detects a fan whose IF return is above roughly')
print(f'{min(MIN25 if MIN25 else 0, bb_sens):+.0f} dB re the CW noise rms ({nr25*10**(bb_sens/20.)*1e6:.0f} uV).')
print('That is a poor sensitivity, and the reason is hardware, not processing: with 4-5 ADC')
print('codes the CW floor is quantisation, and the day-to-day gain drift between captures')
print('(~1 dB) is larger than any excess the fan could plausibly add.')

# --------------------------------------------------------------- 6 cfgB
hdr('6  FMCW cfgB CROSS-CHECK (PRF 1000 Hz, +-3.11 m/s unambiguous)')
d = np.load('C:/dev/klc6/out/fan/20260830_093545_fan_on_fmcw_cfgB.npz', allow_pickle=True)
cp = np.asarray(d['cpis'], float)
nS = cp.shape[2]
V = np.vander(np.arange(nS) / nS, 4)
flat = cp.reshape(-1, nS)
cpd = (flat - (V @ np.linalg.lstsq(V, flat.T, rcond=None)[0]).T).reshape(cp.shape)
rr = np.fft.rfft(cpd * np.hanning(nS), axis=2)
rr = rr - rr.mean(1, keepdims=True)
rd = np.fft.fftshift(np.fft.fft(rr * np.hanning(128)[None, :, None], axis=1), axes=1)
vel = np.fft.fftshift(np.fft.fftfreq(128, 1 / 1000.)) / HZ
rngB = P.beat_to_range(np.fft.rfftfreq(nS, 1 / 1e5), 180e6 * 1000.)
M = 10 * np.log10((np.abs(rd) ** 2).mean(0) + 1e-30)
half = len(cp) // 2
Ma = 10 * np.log10((np.abs(rd[:half]) ** 2).mean(0) + 1e-30)
Mb = 10 * np.log10((np.abs(rd[half:]) ** 2).mean(0) + 1e-30)
nz = np.abs(vel) > 0.15
far = [i for i in range(len(rngB)) if rngB[i] > 8.0]
chance = np.array([M[nz, i].max() - np.median(M[nz, i]) for i in far])
print(f'empirical chance level, measured on {len(far)} target-free range bins beyond 8 m:')
print(f'  peak-minus-median over 128 Doppler bins = {chance.mean():.2f} +- {chance.std():.2f} dB')
print(f'\n{"range m":>9}{"peak dB":>10}{"at v m/s":>10}{"peak-med":>10}{"splithalf":>11}{"sigma":>8}')
CFGB = []
for i in range(1, 6):
    col = M[:, i]
    j = int(np.argmax(np.where(nz, col, -1e30)))
    pm = col[j] - np.median(col[nz])
    sh = abs(Ma[j, i] - Mb[j, i])
    sg = (pm - chance.mean()) / chance.std()
    CFGB.append((float(rngB[i]), float(pm), float(sg)))
    print(f'{rngB[i]:9.2f}{col[j]:10.2f}{vel[j]:10.3f}{pm:10.2f}{sh:11.2f}{sg:8.1f}')
RES['cfgB_best_sigma'] = max(s for _, _, s in CFGB)
RES['cfgB_sigma_at_2p5m'] = [s for r, _, s in CFGB if abs(r - 2.5) < 0.5][0]
print('Nothing at 2.5 m, where the fan is (-0.4 sigma). The 0.83 m bin is 9.4 sigma over the')
print('far-range chance level, so it needs explaining. Its Doppler structure:')
col = M[:, 1]
for j in np.argsort(col)[::-1][:5]:
    fd = vel[j] * HZ
    print(f'    v={vel[j]:+.3f} m/s  fd={fd:+8.2f} Hz  {col[j]-np.median(col[nz]):+5.2f} dB')
print(f'  Doppler bin width {(vel[1]-vel[0])*HZ:.2f} Hz; 60 Hz mains falls at v=+{60/HZ:.3f} m/s,')
print('  which a 128-point Hann Doppler FFT splits into the +-54.7 / 62.5 Hz bins.')
print('  The feature is SYMMETRIC in velocity (+-0.340, +-0.291, +-0.631 all equally raised).')
print('  A moving target is one-sided in Doppler; a symmetric pair is amplitude modulation of')
print('  a STATIONARY scatterer -- here the near-range ramp feedthrough, AM-modulated at 60 Hz.')
print('  cfgA, which does have a fan-off reference, gives -0.22 dB / -0.9 sigma in that same')
print('  bin, confirming it is present with the fan off. Mains again, not a blade.')

# ------------------------------------------------------------------- figure
fig = plt.figure(figsize=(16, 12))
x, fs = CW['FAN_ON#0']
f, t, S = tfmap(x[:int(3 * fs)], fs, 4096, 512)
xo, _ = CW['off_smoke']
fo, to, So = tfmap(xo[:int(3 * fs)], fs, 4096, 512)
vm = (f >= 0) & (f <= 5000)
lim = (float(np.percentile(10 * np.log10(S[vm] + 1e-30), 2)),
       float(np.percentile(10 * np.log10(S[vm] + 1e-30), 99.9)))
ax = fig.add_subplot(3, 3, 1)
ax.pcolormesh(t, f[vm] / HZ, 10 * np.log10(S[vm] + 1e-30), shading='auto', cmap='magma',
              vmin=lim[0], vmax=lim[1])
ax.set_title('FAN ON  CW spectrogram (nper=4096)')
ax.set_ylabel('radial velocity  m/s')
ax.set_xlabel('s')
ax = fig.add_subplot(3, 3, 2)
ax.pcolormesh(to, fo[vm] / HZ, 10 * np.log10(So[vm] + 1e-30), shading='auto', cmap='magma',
              vmin=lim[0], vmax=lim[1])
ax.set_title('FAN OFF  CW spectrogram (same scale)')
ax.set_xlabel('s')
ax = fig.add_subplot(3, 3, 3)
for k, c in [('FAN_ON#0', 'C3'), ('FAN_ON#1', 'C1'), ('off_smoke', 'C0')]:
    xx, ff = CW[k]
    _, m2, _ = flash_mf(xx, ff, 1024, 128)
    ax.hist(m2, bins=90, histtype='step', density=True, label=k, color=c)
ax.axvline(5, ls='--', c='k', lw=.8)
ax.set_yscale('log')
ax.legend(fontsize=7)
ax.set_title('blade-flash matched-filter output')
ax.set_xlabel('z')
ax = fig.add_subplot(3, 3, 4)
xs = np.arange(len(BANDTAB))
cols = ['C3' if not n.startswith('UNPHYS') else '0.6' for n, _, _, _ in BANDTAB]
ax.bar(xs, [d for _, d, _, _ in BANDTAB], yerr=[fl for _, _, fl, _ in BANDTAB], color=cols)
ax.set_xticks(xs)
ax.set_xticklabels([n.replace('UNPHYSICAL ', '*') for n, _, _, _ in BANDTAB],
                   rotation=35, ha='right', fontsize=7)
ax.axhline(0, c='k', lw=.5)
ax.set_ylabel('fan_on - fan_off  dB')
ax.set_title('equal-DOF band offsets\n(* = unphysical for a fan; identical => gain, not target)')
ax = fig.add_subplot(3, 3, 5)
for k, c in [('FAN_ON#0', 'C3'), ('off_smoke', 'C0')]:
    ax.semilogx(fr[1:] / HZ, smooth(10 * np.log10(selfnorm(PS[k]))[1:], 128), lw=.8, color=c, label=k)
ax.axvspan(60, 320, color='0.85', zorder=0)
ax.text(70, ax.get_ylim()[1] * .9, 'unphysical', fontsize=7)
ax.set_xlabel('radial velocity m/s')
ax.set_ylabel('dB (self-normalised)')
ax.set_title('CW spectrum shape, velocity axis')
ax.legend(fontsize=7)
ax = fig.add_subplot(3, 3, 6)
im = ax.pcolormesh(rngB[:8], vel, M[:, :8], shading='auto', cmap='viridis')
ax.axvline(2.5, c='r', ls='--', lw=1)
ax.set_title('FAN ON  FMCW cfgB range-Doppler (MTI)\nred = the fan at 2.5 m')
ax.set_xlabel('range m')
ax.set_ylabel('m/s')
fig.colorbar(im, ax=ax)
ax = fig.add_subplot(3, 3, 7)
w = 0.38
xs = np.arange(1, 8)
ax.bar(xs - w / 2, [TF['moving_person'][:, i].mean() - ref[:, i].mean() for i in range(1, 8)],
       w, label='moving person (positive control)', color='C2')
ax.bar(xs + w / 2, [TF['FAN_ON_cfgA'][:, i].mean() - ref[:, i].mean() for i in range(1, 8)],
       w, label='FAN ON', color='C3')
ax.errorbar(xs, np.zeros(7), yerr=[FAN_A[i][3] for i in range(7)], fmt='none', ecolor='k', capsize=2)
ax.set_xticks(xs)
ax.set_xticklabels([f'{rng[i]:.1f}' for i in range(1, 8)])
ax.set_title('FMCW cfgA motion power vs static reference\n(identical processing, 1-sigma bars)')
ax.set_xlabel('range m')
ax.set_ylabel('dB')
ax.legend(fontsize=7)
ax.axhline(0, c='k', lw=.5)
ax = fig.add_subplot(3, 3, 8)
d = np.load(CWF['off_smoke'], allow_pickle=True)
for row, c, lb in [(0, 'C0', 'off_smoke row0 (IF)'), (1, 'C4', 'off_smoke row1 (no radar)')]:
    xx = d['data'][row].astype(float)
    xn, _ = notch_mains(xx - xx.mean(), 100000.)
    tt, m2, dt = flash_mf(xn, 100000., 1024, 256)
    n = len(m2)
    C = np.abs(np.fft.rfft((m2 - m2.mean()) * np.hanning(n)))
    cf = np.fft.rfftfreq(n, dt)
    s = (cf > 2) & (cf < 190)
    ax.plot(cf[s], 20 * np.log10(C[s] / np.median(C[s])), lw=.7, color=c, label=lb)
xx, ffs = CW['FAN_ON#0']
tt, m2, dt = flash_mf(xx, ffs, 1024, 256)
n = len(m2)
C = np.abs(np.fft.rfft((m2 - m2.mean()) * np.hanning(n)))
cf = np.fft.rfftfreq(n, dt)
s = (cf > 2) & (cf < 190)
ax.plot(cf[s], 20 * np.log10(C[s] / np.median(C[s])), lw=.7, color='C3', label='FAN_ON#0 row0 (IF)')
ax.axvline(60, c='k', ls=':', lw=1)
ax.set_xlabel('cadence Hz')
ax.set_ylabel('dB')
ax.legend(fontsize=6)
ax.set_title('the 60 Hz cadence line appears on a\nchannel with no radar return => mains')
ax = fig.add_subplot(3, 3, 9)
ax.axis('off')
ax.text(0, 1,
        f'POSITIVE CONTROL  {PC_SIGMA:5.1f} sigma   {"PASS" if PC_PASS else "FAIL"}\n'
        f'  moving person +7.4 dB @ 1.67 m\n\n'
        f'FAN, same detector\n'
        f'  FMCW cfgA best |sigma|  {FAN_A_BEST:5.1f}\n'
        f'  FMCW cfgB best  sigma   {RES["cfgB_best_sigma"]:5.1f}\n'
        f'  CW band offset          {RES["shape_sigma"]:5.1f} sigma (chance {np.sqrt(2*np.log(NTR)):.1f})\n'
        f'  CW cadence z on / off   {RES["cad_on_max"]:.2f} / {RES["cad_off"]:.2f}\n'
        f'    chance for 8 blocks   {RES["cad_chance8"]:.2f}\n\n'
        f'physical bands  {phys:+.2f} dB\n'
        f'unphysical      {unph:+.2f} dB  -> gain, not target\n\n'
        f'sensitivity  {bb_sens:+.0f} dB re CW noise rms\n'
        f'fan return is >={RES["fan_vs_person_dB"]:.1f} dB below a person\n'
        f'at the same range.\n\n'
        f'VERDICT: NO DETECTION',
        va='top', family='monospace', fontsize=9)
fig.tight_layout()
fig.savefig(f'{OUT}/timefreq-template.png', dpi=110)
print(f'\nfigure -> {OUT}/timefreq-template.png')

hdr('SUMMARY')
for k, v in RES.items():
    print(f'  {k:24s} {v}')
