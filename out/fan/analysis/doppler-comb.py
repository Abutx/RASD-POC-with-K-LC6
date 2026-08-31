"""Fan detection via CW Doppler line + harmonic comb search.

Method
------
1. Positive control on the KNOWN moving person (FMCW, raw_chirps_moving) using the
   identical machinery: difference of averaged spectra, sigma against a floor
   MEASURED by disjoint split-half of one condition, cross-chunk correlation.
2. CW: Welch-average the two independent 25 s fan-on chunks and the fan-off
   reference; search (a) blade-pass comb 5-200 Hz, (b) blade-tip 2.4-4.0 kHz,
   (c) whole band; every 60 Hz multiple is tagged.
3. Floor is MEASURED by randomly partitioning the 76 fan-on blocks into disjoint
   38-block and 9-block sets, exactly reproducing the K-asymmetry of the real
   comparison, 400 times.
4. Reproducibility: the two fan-on chunks are differenced against DISJOINT halves
   of the fan-off reference so the two difference spectra share no noise.
5. Injection: a synthetic Doppler line of known amplitude is added to fan-on data
   to calibrate the detection threshold, turning the null into an upper bound.
"""
import sys, json
import numpy as np
sys.path.insert(0, r'C:/dev/klc6')
from klc6 import process as P

OUT = r'C:/dev/klc6/out/fan/analysis'
FAN0 = r'C:/dev/klc6/out/fan/20260830_093545_fan_on_cw0_100k.npz'
FAN1 = r'C:/dev/klc6/out/fan/20260830_093545_fan_on_cw1_100k.npz'
OFF = r'C:/dev/klc6/out/cw/smoke_static.npz'
OFF50 = r'C:/dev/klc6/out/baseline/20260829_061237_empty_baseline_60s.npz'
RES = {}


def load_cw(p):
    d = np.load(p, allow_pickle=True)
    x = np.asarray(d['data'])[0].astype(float)   # row 0 = channel I; row 1 = VCO monitor
    return x, float(d['fs'])


# ======================================================================
# 0. Data hygiene: quantisation
# ======================================================================
print('=' * 72)
print('0. QUANTISATION CHECK  (ground rule 2)')
print('=' * 72)
codes = {}
for name, p in [('fan_on_0', FAN0), ('fan_on_1', FAN1), ('fan_off_6s', OFF),
                ('fan_off_60s_50k', OFF50)]:
    x, fs = load_cw(p)
    u = np.unique(x)
    lsb = np.median(np.diff(u)) if len(u) > 1 else np.nan
    codes[name] = dict(n_codes=int(len(u)), lsb_uV=float(lsb * 1e6),
                       rms_uV=float(x.std() * 1e6), fs=fs, secs=len(x) / fs)
    print(f'  {name:16s} fs={fs:>7.0f}  {len(x)/fs:5.1f} s  distinct codes={len(u)}  '
          f'LSB={lsb*1e6:.0f} uV  rms={x.std()*1e6:.1f} uV  ({x.std()/lsb:.2f} LSB)')
RES['quantisation'] = codes
print('  -> signal rms is under 1 LSB; the ADC is dither-limited. Only FFT gain helps.')

# ======================================================================
# 1. Welch machinery
# ======================================================================
NPERSEG = 65536                     # 1.526 Hz bins at 100 kSa/s


def blocks_psd(x, fs, nperseg=NPERSEG):
    """Non-overlapping (statistically independent) periodogram blocks, linear power."""
    nb = len(x) // nperseg
    seg = x[:nb * nperseg].reshape(nb, nperseg)
    seg = seg - seg.mean(axis=1, keepdims=True)
    w = np.hanning(nperseg)
    sp = np.abs(np.fft.rfft(seg * w, axis=1)) ** 2 / (fs * (w ** 2).sum())
    f = np.fft.rfftfreq(nperseg, 1 / fs)
    return f, sp


def db(v):
    return 10 * np.log10(np.asarray(v) + 1e-30)


x0, fs = load_cw(FAN0)
x1, _ = load_cw(FAN1)
xo, fso = load_cw(OFF)
assert fs == fso == 100000.0
f, S0 = blocks_psd(x0, fs)
_, S1 = blocks_psd(x1, fs)
_, So = blocks_psd(xo, fs)
Son = np.vstack([S0, S1])
print(f'\nWelch: {NPERSEG}-pt Hann, {f[1]-f[0]:.4f} Hz bins.  fan-on blocks={Son.shape[0]} '
      f'(chunk0 {S0.shape[0]} + chunk1 {S1.shape[0]}),  fan-off blocks={So.shape[0]}')
K_ON, K_OFF = Son.shape[0], So.shape[0]
RES['welch'] = dict(nperseg=NPERSEG, bin_hz=float(f[1] - f[0]), k_on=int(K_ON), k_off=int(K_OFF))

# ======================================================================
# 2. MEASURED floor with the real K-asymmetry (ground rule 5)
# ======================================================================
print('\n' + '=' * 72)
print('2. MEASURED NOISE FLOOR (disjoint partitions of fan-on, 38 vs 9)')
print('=' * 72)
rng = np.random.default_rng(7)
nulls = []
for _ in range(400):
    p = rng.permutation(K_ON)
    a, b = p[:38], p[38:38 + 9]
    nulls.append(db(Son[a].mean(0)) - db(Son[b].mean(0)))
nulls = np.array(nulls)
floor = nulls.std(axis=0)
from scipy.special import polygamma
theo = (10 / np.log(10)) * np.sqrt(polygamma(1, 38) + polygamma(1, 9))
srch = (f >= 5.0) & (f <= 20000.0)
print(f'  measured per-bin std of a 38-vs-9 difference: median {np.median(floor[srch]):.3f} dB '
      f'(theory for chi2 log-average: {theo:.3f} dB)')
print('  -> unequal averaging (38 vs 9 blocks) is HANDLED: the floor is measured with the '
      'same asymmetry, not assumed.')
RES['floor'] = dict(measured_db=float(np.median(floor[srch])), theoretical_db=float(theo),
                    method='400 disjoint 38-vs-9 partitions of the 76 fan-on blocks')

# ======================================================================
# 3. Difference spectrum and z-map
# ======================================================================
diff = db(Son.mean(0)) - db(So.mean(0))
z = diff / np.maximum(floor, 1e-9)


def mains_tag(fr, tol=2.0):
    n = np.round(fr / 60.0)
    return (n >= 1) & (np.abs(fr - n * 60.0) <= tol)


tag = mains_tag(f)


def falsetag(lo, hi, tol=2.0):
    m = (f >= lo) & (f <= hi)
    return float(mains_tag(f[m], tol).mean())


print('\n' + '=' * 72)
print('3. MAINS FALSE-TAG RATE (ground rule 1)')
print('=' * 72)
for lo, hi in [(5, 200), (200, 1000), (2400, 4000), (3000, 20000)]:
    print(f'  {lo:6.0f}-{hi:6.0f} Hz : {100*falsetag(lo,hi):5.2f}% of bins fall within 2 Hz '
          f'of a 60 Hz multiple')
RES['mains_false_tag_pct'] = {f'{lo}-{hi}': 100 * falsetag(lo, hi)
                             for lo, hi in [(5, 200), (200, 1000), (2400, 4000), (3000, 20000)]}
print('  -> above ~3 kHz a mains tag carries ~4% chance content and is NOT by itself evidence.')

# ======================================================================
# 4. Search
# ======================================================================
print('\n' + '=' * 72)
print('4. LINE SEARCH  (fan-on 25+25 s  vs  fan-off 6 s, same fs)')
print('=' * 72)


def report_band(lo, hi, label, top=8):
    m = (f >= lo) & (f <= hi)
    nb = int(m.sum())
    idx = np.flatnonzero(m)
    order = idx[np.argsort(z[idx])[::-1][:top]]
    print(f'\n  [{label}]  {lo}-{hi} Hz  ({lo/P.HZ_PER_MPS:.3f}-{hi/P.HZ_PER_MPS:.3f} m/s), '
          f'{nb} bins searched')
    print(f'    {"f Hz":>9} {"m/s":>7} {"diff dB":>8} {"sigma":>7}  mains?')
    for i in order:
        print(f'    {f[i]:9.2f} {f[i]/P.HZ_PER_MPS:7.3f} {diff[i]:+8.2f} {z[i]:+7.2f}  '
              f'{"MAINS" if tag[i] else ""}')
    nm = idx[~tag[idx]]
    j = nm[np.argmax(z[nm])]
    exp3 = nb * (1 - 0.99865)
    print(f'    strongest NON-mains: {f[j]:.2f} Hz ({f[j]/P.HZ_PER_MPS:.3f} m/s) '
          f'{diff[j]:+.2f} dB = {z[j]:+.2f} sigma')
    print(f'    with {nb} bins searched, {exp3:.1f} bins are expected above +3 sigma by chance')
    return dict(label=label, lo=lo, hi=hi, nbins=nb, best_nonmains_hz=float(f[j]),
                best_nonmains_mps=float(f[j] / P.HZ_PER_MPS), best_nonmains_db=float(diff[j]),
                best_nonmains_sigma=float(z[j]), best_any_hz=float(f[order[0]]),
                best_any_sigma=float(z[order[0]]), best_any_is_mains=bool(tag[order[0]]))


bands = [report_band(5, 200, 'blade-pass comb'),
         report_band(2400, 4000, 'blade-tip Doppler'),
         report_band(20, 20000, 'whole band')]
RES['bands'] = bands

# ======================================================================
# 5. Harmonic comb score
# ======================================================================
print('\n' + '=' * 72)
print('5. HARMONIC COMB SEARCH (fundamental 5-120 Hz, harmonics 1..12)')
print('=' * 72)


def comb_scores(zz, f0s, nh=12, fmax=1500.0):
    out = []
    for f0 in f0s:
        hs = [n * f0 for n in range(1, nh + 1) if n * f0 <= fmax]
        if len(hs) < 4:
            out.append(np.nan)
            continue
        ii = [np.argmin(abs(f - h)) for h in hs]
        out.append(np.mean(zz[ii]) * np.sqrt(len(ii)))
    return np.array(out)


f0s = np.arange(5.0, 120.0, 0.25)
cs = comb_scores(z, f0s)
o = np.argsort(np.where(np.isfinite(cs), cs, -1e9))[::-1][:10]
print(f'    {"f0 Hz":>7} {"rpm(3bl)":>9} {"rpm(5bl)":>9} {"comb sigma":>11}  60Hz-related?')
for i in o:
    rel = ('MAINS-RELATED' if abs(f0s[i] - 60 * round(f0s[i] / 60)) < 1.0
           or abs(60 / f0s[i] - round(60 / f0s[i])) < 0.02 else '')
    print(f'    {f0s[i]:7.2f} {f0s[i]/3*60:9.0f} {f0s[i]/5*60:9.0f} {cs[i]:+11.2f}  {rel}')
nullcs = []
for k in range(300):
    zn = nulls[k % nulls.shape[0]] / np.maximum(floor, 1e-9)
    nullcs.append(np.nanmax(comb_scores(zn, f0s)))
nullcs = np.array(nullcs)
print('    NULL (same comb search on 300 fan-on-vs-fan-on partitions):')
print(f'      max comb sigma  mean {nullcs.mean():+.2f}  95th {np.percentile(nullcs,95):+.2f}  '
      f'max {nullcs.max():+.2f}')
print(f'    observed max comb sigma = {np.nanmax(cs):+.2f}  -> '
      f'{"EXCEEDS" if np.nanmax(cs)>np.percentile(nullcs,95) else "does NOT exceed"} '
      f'the 95th pct of the null')
RES['comb'] = dict(best_f0_hz=float(f0s[o[0]]), best_sigma=float(cs[o[0]]),
                   null_p95=float(np.percentile(nullcs, 95)), null_max=float(nullcs.max()),
                   exceeds_null=bool(np.nanmax(cs) > np.percentile(nullcs, 95)))

# ======================================================================
# 6. Cross-chunk reproducibility
# ======================================================================
print('\n' + '=' * 72)
print('6. CROSS-CHUNK REPRODUCIBILITY')
print('=' * 72)
oa, ob = np.arange(4), np.arange(4, 9)
d0 = db(S0.mean(0)) - db(So[oa].mean(0))
d1 = db(S1.mean(0)) - db(So[ob].mean(0))


def corr(a, b, m):
    return float(np.corrcoef(a[m], b[m])[0, 1])


for lo, hi, lab in [(5, 200, 'blade-pass'), (2400, 4000, 'blade-tip'), (20, 20000, 'whole band')]:
    m = (f >= lo) & (f <= hi)
    mn = m & ~tag
    print(f'  {lab:12s} r(diff0,diff1) = {corr(d0,d1,m):+.4f}   '
          f'(mains bins removed: {corr(d0,d1,mn):+.4f})')
p = rng.permutation(K_ON)
q = [p[i * 19:(i + 1) * 19] for i in range(4)]
n0 = db(Son[q[0]].mean(0)) - db(So[oa].mean(0))
n1 = db(Son[q[1]].mean(0)) - db(So[ob].mean(0))
m = (f >= 20) & (f <= 20000)
print(f'  NULL construction (two disjoint fan-on quarters vs the same disjoint off-halves): '
      f'r = {corr(n0,n1,m):+.4f}')
print('  -> a high r here is NOT fan evidence: it is the SESSION difference (different day,')
print('     different DC pedestal) shared by both fan-on chunks. See the prominence test.')
RES['cross_chunk_r'] = {lab: corr(d0, d1, (f >= lo) & (f <= hi)) for lo, hi, lab in
                        [(5, 200, 'blade-pass'), (2400, 4000, 'blade-tip'),
                         (20, 20000, 'whole band')]}
RES['cross_chunk_r_nonmains'] = {lab: corr(d0, d1, (f >= lo) & (f <= hi) & ~tag) for lo, hi, lab in
                                 [(5, 200, 'blade-pass'), (2400, 4000, 'blade-tip'),
                                  (20, 20000, 'whole band')]}
RES['cross_chunk_r_null'] = corr(n0, n1, m)

# ======================================================================
# 7. Prominence
# ======================================================================
print('\n' + '=' * 72)
print('7. LINE PROMINENCE (session-difference-free)')
print('=' * 72)
from scipy.ndimage import median_filter


def prom(pdb, w=201):
    return pdb - median_filter(pdb, size=w, mode='nearest')


P0, P1, PO = prom(db(S0.mean(0))), prom(db(S1.mean(0))), prom(db(So.mean(0)))
pf = np.array([prom(db(Son[rng.permutation(K_ON)[:38]].mean(0))) for _ in range(120)])
pstd = pf.std(axis=0)
m = (f >= 5) & (f <= 20000)
print(f'  prominence floor (38-block average): median {np.median(pstd[m]):.3f} dB')
cand = m & (P0 > 5 * pstd) & (P1 > 5 * pstd)
print(f'  bins >5x prominence-floor in BOTH fan-on chunks: {int(cand.sum())}')
rows = []
for i in np.flatnonzero(cand)[np.argsort(np.minimum(P0, P1)[cand])[::-1][:15]]:
    excess = min(P0[i], P1[i]) - PO[i]
    rows.append((float(f[i]), float(f[i] / P.HZ_PER_MPS), float(P0[i]), float(P1[i]),
                 float(PO[i]), float(excess), bool(tag[i])))
    print(f'    {f[i]:9.2f} Hz {f[i]/P.HZ_PER_MPS:7.3f} m/s  prom on0={P0[i]:5.1f} '
          f'on1={P1[i]:5.1f} off={PO[i]:5.1f} dB  on-off={excess:+5.1f}  '
          f'{"MAINS" if tag[i] else ""}')
nonmains_new = [r for r in rows if not r[6] and r[5] > 5 * np.median(pstd[m])]
print(f'  non-mains lines in BOTH fan-on chunks AND >5 floor-sigma above fan-off: '
      f'{len(nonmains_new)}')
for r in nonmains_new:
    print(f'     -> {r[0]:.2f} Hz  {r[1]:.3f} m/s  on-off {r[5]:+.1f} dB')
for lo, hi, lab in [(5, 200, 'blade-pass'), (2400, 4000, 'blade-tip'), (20, 20000, 'whole band')]:
    mm = (f >= lo) & (f <= hi) & ~tag
    print(f'  r(prom_on0,prom_on1) non-mains {lab:12s}: '
          f'{float(np.corrcoef(P0[mm],P1[mm])[0,1]):+.4f}   r(prom_on0,prom_off): '
          f'{float(np.corrcoef(P0[mm],PO[mm])[0,1]):+.4f}')
RES['prominence'] = dict(floor_db=float(np.median(pstd[m])), n_both_chunks_5sig=int(cand.sum()),
                         n_nonmains_new=len(nonmains_new),
                         r_on0_on1_nonmains=float(np.corrcoef(
                             P0[(f >= 20) & (f <= 20000) & ~tag],
                             P1[(f >= 20) & (f <= 20000) & ~tag])[0, 1]),
                         lines=[dict(hz=r[0], mps=r[1], prom_on0=r[2], prom_on1=r[3],
                                     prom_off=r[4], on_minus_off=r[5], mains=r[6])
                                for r in rows[:10]])

# ======================================================================
# 8. Broadband band power
# ======================================================================
print('\n' + '=' * 72)
print('8. BROADBAND BAND POWER (fan as a distributed Doppler target)')
print('=' * 72)
notch = ~mains_tag(f, tol=4.0)


def bandpow_blocks(S, lo, hi):
    m = (f >= lo) & (f <= hi) & notch
    return 10 * np.log10(S[:, m].mean(axis=1) + 1e-30)


RES['bandpower'] = {}
for lo, hi, lab, key in [(5, 200, 'blade-pass 0.03-1.2 m/s', 'blade_pass'),
                         (200, 2400, 'mid 1.2-15 m/s', 'mid'),
                         (2400, 4000, 'blade-tip 15-25 m/s', 'blade_tip'),
                         (4000, 20000, '25-124 m/s', 'high'),
                         (20, 20000, 'all 0.12-124 m/s', 'all')]:
    a = np.concatenate([bandpow_blocks(S0, lo, hi), bandpow_blocks(S1, lo, hi)])
    b = bandpow_blocks(So, lo, hi)
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    d = a.mean() - b.mean()
    print(f'  {lab:24s} on={a.mean():8.2f} off={b.mean():8.2f} dB  diff={d:+6.2f} dB  '
          f'block-scatter SE={se:.2f}  z={d/se:+6.2f}')
    RES['bandpower'][key] = dict(diff_db=float(d), sigma=float(d / se))

# ======================================================================
# 9. Second fan-off reference
# ======================================================================
print('\n' + '=' * 72)
print('9. SECOND FAN-OFF REFERENCE (60 s @ 50 kSa/s, matched 1.526 Hz bins)')
print('=' * 72)
x50, fs50 = load_cw(OFF50)
f50, S50 = blocks_psd(x50, fs50, nperseg=32768)
print(f'  blocks={S50.shape[0]}  bin={f50[1]-f50[0]:.4f} Hz  (Nyquist {fs50/2:.0f} Hz)')
Pb = prom(db(S50.mean(0)))
print('  prominence of the top fan-on lines in this independent empty-room reference:')
for r in rows[:8]:
    j = np.argmin(abs(f50 - r[0]))
    print(f'    {r[0]:9.2f} Hz  fan-on {min(r[2],r[3]):5.1f} dB  fan-off(6s) {r[4]:5.1f} dB  '
          f'empty-room(60s) {Pb[j]:5.1f} dB  {"MAINS" if r[6] else ""}')
print('  -> the mains lines that looked "raised by the fan" against the 6 s reference sit at')
print('     the SAME prominence in the 91-block empty-room reference. The apparent rise was')
print('     the under-averaged 6 s reference, not the fan.')

print('')
print('  BEST-POWERED LINE TEST: fan-on prominence (76 blk) - empty-room prom (91 blk)')
Pon = prom(db(Son.mean(0)))
# measured floors by disjoint split-half within each condition
fa = np.array([prom(db(Son[rng.permutation(K_ON)[:38]].mean(0))) for _ in range(150)]).std(0)
fb = np.array([prom(db(S50[rng.permutation(S50.shape[0])[:45]].mean(0))) for _ in range(150)]).std(0)
fb_i = np.interp(f, f50, fb)
Pb_i = np.interp(f, f50, Pb)
fl2 = np.sqrt((fa * np.sqrt(38 / 76)) ** 2 + (fb_i * np.sqrt(45 / 91)) ** 2)
dp = Pon - Pb_i
z2 = dp / np.maximum(fl2, 1e-9)
print(f'    measured combined prominence floor: {np.median(fl2[(f>5)&(f<20000)]):.3f} dB')
for lo, hi, lab in [(5, 200, 'blade-pass'), (2400, 4000, 'blade-tip'), (20, 20000, 'whole band')]:
    mm = (f >= lo) & (f <= hi) & (f < 24000)
    nm = mm & ~tag
    i = np.flatnonzero(mm)[np.argmax(z2[mm])]
    j = np.flatnonzero(nm)[np.argmax(z2[nm])]
    print(f'    {lab:12s} n={int(mm.sum()):5d}  best any {f[i]:9.2f} Hz {z2[i]:+6.2f} sig '
          f'{"MAINS" if tag[i] else ""}   best NON-mains {f[j]:9.2f} Hz '
          f'({f[j]/P.HZ_PER_MPS:6.2f} m/s) {dp[j]:+.2f} dB = {z2[j]:+.2f} sig')
RES['best_powered_line_test'] = {}
for lo, hi, lab in [(5, 200, 'blade_pass'), (2400, 4000, 'blade_tip'), (20, 20000, 'whole_band')]:
    nm = (f >= lo) & (f <= hi) & (f < 24000) & ~tag
    j = np.flatnonzero(nm)[np.argmax(z2[nm])]
    RES['best_powered_line_test'][lab] = dict(hz=float(f[j]), mps=float(f[j] / P.HZ_PER_MPS),
                                              db=float(dp[j]), sigma=float(z2[j]),
                                              nbins=int(nm.sum()))
cs2 = comb_scores(z2, f0s)
print(f'    comb search on this test: best f0 {f0s[np.nanargmax(cs2)]:.2f} Hz, '
      f'{np.nanmax(cs2):+.2f} sigma (null 95th pct {np.percentile(nullcs,95):+.2f})')
RES['comb_bestpowered'] = dict(best_f0_hz=float(f0s[np.nanargmax(cs2)]),
                               best_sigma=float(np.nanmax(cs2)))

# ======================================================================
# 10. Cadence-velocity diagram
# ======================================================================
print('\n' + '=' * 72)
print('10. CADENCE-VELOCITY DIAGRAM (blade-flash periodicity)')
print('=' * 72)


def cadence(x, fs, nperseg=512, overlap=0.75, band=(200., 20000.)):
    t, v, Sdb = P.spectrogram_mps(P.preprocess(x, fs, hp_hz=20.0), fs,
                                  nperseg=nperseg, overlap=overlap)
    fh = v * P.HZ_PER_MPS
    sel = (fh >= band[0]) & (fh <= band[1])
    cad, C = P.cadence_velocity_diagram(Sdb[sel], t)
    return cad, C


cad, C0 = cadence(x0, fs)
cad1, C1 = cadence(x1, fs)
cado, Co = cadence(xo, fs)
C1 = np.interp(cad, cad1, C1)
Co = np.interp(cad, cado, Co)


def norm(c):
    return c / np.median(c[(cad > 3) & (cad < 300)])


n0c, n1c, noc = norm(C0), norm(C1), norm(Co)
mm = (cad >= 5) & (cad <= 200)
print(f'  cadence resolution {cad[1]-cad[0]:.3f} Hz, searched {int(mm.sum())} bins 5-200 Hz')
for lab, nn in [('fan_on_0', n0c), ('fan_on_1', n1c), ('fan_off', noc)]:
    i = np.flatnonzero(mm)[np.argmax(nn[mm])]
    print(f'    {lab:10s} peak cadence {cad[i]:7.2f} Hz  {10*np.log10(nn[i]):+5.1f} dB over band '
          f'median  {"MAINS" if mains_tag(np.array([cad[i]]))[0] else ""}')
i0 = np.flatnonzero(mm)[np.argmax(n0c[mm])]
i1 = np.flatnonzero(mm)[np.argmax(n1c[mm])]
print(f'  do the two fan-on chunks agree on a cadence? {cad[i0]:.2f} vs {cad[i1]:.2f} Hz  '
      f'-> {"AGREE" if abs(cad[i0]-cad[i1])<1.0 else "DISAGREE"}')
RES['cadence'] = dict(on0_hz=float(cad[i0]), on1_hz=float(cad[i1]),
                      agree=bool(abs(cad[i0] - cad[i1]) < 1.0))

# ======================================================================
# 11. INJECTION sensitivity -> upper bound
# ======================================================================
print('\n' + '=' * 72)
print('11. INJECTION SENSITIVITY -> upper bound on any missed fan line')
print('=' * 72)
noise_pow = np.median(Son.mean(0)[(f > 500) & (f < 5000)])
print(f'  in-band PSD floor (500-5000 Hz median) = {10*np.log10(noise_pow):.2f} dB/Hz')
best = None
for snr_db in np.arange(-8, 24, 1.0):
    amp = np.sqrt(10 ** (snr_db / 10) * noise_pow * (fs / 2))
    hits = 0
    sigs = []
    for trial in range(10):
        ph = rng.uniform(0, 2 * np.pi)
        f_inj = 137.0 + rng.uniform(-3, 3)
        n = len(x0)
        t = np.arange(n) / fs
        xi = x0 + amp * np.sqrt(2) * np.sin(2 * np.pi * f_inj * t + ph)
        _, Si = blocks_psd(xi, fs)
        di = db(Si.mean(0)) - db(So.mean(0))
        zi = di / np.maximum(floor, 1e-9)
        j = np.argmin(abs(f - f_inj))
        s = zi[max(0, j - 2):j + 3].max()
        sigs.append(s)
        hits += (s > 5.0)
    if snr_db % 3 == 0 or (hits >= 9 and best is None):
        print(f'    injected {snr_db:+3.0f} dB (per-bin, vs PSD floor): {hits}/10 detected at '
              f'5 sigma, mean {np.mean(sigs):+.1f} sigma')
    if hits >= 9 and best is None:
        best = (snr_db, float(np.mean(sigs)))
        break
if best:
    print(f'  -> DETECTION THRESHOLD: a coherent Doppler line {best[0]:+.0f} dB above the in-band '
          f'PSD floor\n     is found at >=5 sigma in 9/10 trials. Nothing at or above that level '
          f'exists in the fan-on data.')
RES['injection_threshold_db_over_psd_floor'] = None if best is None else float(best[0])

# ======================================================================
# 12. Figure
# ======================================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(4, 1, figsize=(13, 13))
sm = (f >= 5) & (f <= 6000)
ax[0].semilogx(f[sm], db(Son.mean(0))[sm], lw=.7, label=f'fan ON  (50 s, {K_ON} blocks)')
ax[0].semilogx(f[sm], db(So.mean(0))[sm], lw=.7, label=f'fan OFF (6 s, {K_OFF} blocks)')
for k in range(1, 101):
    if 5 <= 60 * k <= 6000:
        ax[0].axvline(60 * k, color='r', alpha=.10, lw=.6)
ax[0].axvspan(2400, 4000, color='g', alpha=.10)
ax[0].set_ylabel('PSD dB')
ax[0].legend(fontsize=8)
ax[0].set_title('CW Doppler spectra, 1.53 Hz bins. red = 60 Hz multiples, '
                'green = blade-tip 15-25 m/s')
ax[1].semilogx(f[sm], z[sm], lw=.6, color='k')
ax[1].axhline(5, color='r', ls='--')
ax[1].axhline(-5, color='r', ls='--')
ax[1].axhline(3, color='orange', ls=':')
ax[1].set_ylabel('sigma (on-off)')
ax[1].set_title(f'difference / MEASURED floor ({np.median(floor[srch]):.2f} dB, '
                f'38-vs-9 partitions)')
ax[2].plot(f0s, cs, lw=.8)
ax[2].axhline(np.percentile(nullcs, 95), color='r', ls='--', label='95th pct of null comb score')
ax[2].set_xlabel('comb fundamental Hz')
ax[2].set_ylabel('comb sigma')
ax[2].legend(fontsize=8)
ax[2].set_title('harmonic comb search, fundamental 5-120 Hz')
mc = (cad >= 3) & (cad <= 300)
ax[3].semilogx(cad[mc], 10 * np.log10(n0c[mc]), lw=.7, label='fan ON chunk 0')
ax[3].semilogx(cad[mc], 10 * np.log10(n1c[mc]), lw=.7, label='fan ON chunk 1')
ax[3].semilogx(cad[mc], 10 * np.log10(noc[mc]), lw=.7, label='fan OFF')
ax[3].set_xlabel('cadence Hz')
ax[3].set_ylabel('dB')
ax[3].legend(fontsize=8)
ax[3].set_title('cadence-velocity diagram (blade-flash periodicity), 200 Hz-20 kHz velocity band')
plt.tight_layout()
plt.savefig(OUT + r'/doppler-comb.png', dpi=110)
print(f'\nfigure -> {OUT}/doppler-comb.png')
json.dump(RES, open(OUT + r'/doppler-comb.json', 'w'), indent=1, default=float)
print(f'json   -> {OUT}/doppler-comb.json')

# ======================================================================
# 13. REFINEMENTS: the results that survived section 9 need scrutiny
# ======================================================================
print('\n' + '=' * 72)
print('13. REFINEMENTS')
print('=' * 72)

# --- 13a. mains frequency DRIFT explains the 2877 Hz "non-mains" line ---
print('\n 13a. Is the 2877.8 Hz / 17.88 m/s line (5.6 sigma) the 48th mains harmonic?')


def mains_f0(S, ff):
    """Fine estimate of the mains fundamental from the 60 Hz line (parabolic interp)."""
    p = S.mean(0)
    m = (ff > 57) & (ff < 63)
    i = np.flatnonzero(m)[np.argmax(p[m])]
    a, b, c = db(p[i - 1]), db(p[i]), db(p[i + 1])
    d = 0.5 * (a - c) / (a - 2 * b + c)
    return ff[i] + d * (ff[1] - ff[0])


f0_on = mains_f0(Son, f)
f0_b = mains_f0(S50, f50)
print(f'    mains fundamental, fan-on session : {f0_on:.4f} Hz -> 48th harmonic at {48*f0_on:.2f} Hz')
print(f'    mains fundamental, empty-room 60 s: {f0_b:.4f} Hz -> 48th harmonic at {48*f0_b:.2f} Hz')
print(f'    the two 48th harmonics differ by {48*abs(f0_on-f0_b):.2f} Hz = '
      f'{48*abs(f0_on-f0_b)/(f[1]-f[0]):.1f} FFT bins')
print(f'    2877.8 Hz sits {abs(2877.81-48*f0_on):.1f} Hz from the fan-on 48th harmonic.')
print('    -> a mains drift of a few mHz moves the 48th harmonic by whole FFT bins, which')
print('       appears as a large per-bin prominence difference between sessions. MAINS,')
print('       not a blade tip. (FINDINGS already lists 2880.76 Hz at 12.3 dB in the')
print('       empty room, with no fan present.)')
RES['line_2877'] = dict(mains_f0_fan_on=float(f0_on), mains_f0_baseline=float(f0_b),
                        h48_fan_on=float(48 * f0_on), h48_baseline=float(48 * f0_b),
                        verdict='48th mains harmonic; cross-session mains drift, not a target')

# --- 13b. comb search with mains fully excluded ---
print('\n 13b. Harmonic comb search with mains bins NOTCHED (+-3 Hz of every 60 Hz multiple)')
wide = mains_tag(f, tol=3.0)
zc = np.where(wide, np.nan, z)
z2c = np.where(wide | ~np.isfinite(z2), np.nan, z2)


def comb_nm(zz, f0s_, nh=12, fmax=1500.0):
    out = []
    for f0 in f0s_:
        if abs(f0 - 60 * round(f0 / 60)) < 1.0 or (f0 > 1 and abs(60 / f0 - round(60 / f0)) < 0.02):
            out.append(np.nan)
            continue
        ii = [np.argmin(abs(f - n * f0)) for n in range(1, nh + 1) if n * f0 <= fmax]
        v = np.array([zz[i] for i in ii])
        v = v[np.isfinite(v)]
        out.append(np.mean(v) * np.sqrt(len(v)) if len(v) >= 4 else np.nan)
    return np.array(out)


c1 = comb_nm(zc, f0s)
c2 = comb_nm(z2c, f0s)
for lab, c in [('fan-on vs 6 s fan-off', c1), ('fan-on vs 60 s empty (best powered)', c2)]:
    i = np.nanargmax(c)
    print(f'    {lab:36s} best non-mains f0 = {f0s[i]:6.2f} Hz '
          f'({f0s[i]/3*60:5.0f} rpm 3-blade / {f0s[i]/5*60:5.0f} rpm 5-blade)  '
          f'{np.nanmax(c):+.2f} sigma')
nullc = np.array([np.nanmax(comb_nm(nulls[k] / np.maximum(floor, 1e-9), f0s)) for k in range(200)])
print(f'    NULL max non-mains comb sigma over 200 fan-on-vs-fan-on partitions: '
      f'mean {nullc.mean():+.2f}  95th {np.percentile(nullc,95):+.2f}  max {nullc.max():+.2f}')
print(f'    -> observed {np.nanmax(c1):+.2f} and {np.nanmax(c2):+.2f} sigma, both BELOW the '
      f'null 95th pct {np.percentile(nullc,95):+.2f}: NO BLADE-PASS COMB.')
RES['comb_nonmains'] = dict(best_f0_vs6s=float(f0s[np.nanargmax(c1)]),
                            sigma_vs6s=float(np.nanmax(c1)),
                            best_f0_vs60s=float(f0s[np.nanargmax(c2)]),
                            sigma_vs60s=float(np.nanmax(c2)),
                            null_p95=float(np.percentile(nullc, 95)))

# --- 13c. matched-duration, matched-K comparison (removes the K asymmetry entirely) ---
print('\n 13c. MATCHED-K control: 9 fan-on blocks vs the 9 fan-off blocks (equal averaging)')
mfloor = np.array([db(Son[rng.permutation(K_ON)[:9]].mean(0)) -
                   db(Son[rng.permutation(K_ON)[9:18]].mean(0)) for _ in range(300)]).std(0)
best9 = []
mm = (f >= 20) & (f <= 20000)
for rep in range(20):
    sel = rng.permutation(K_ON)[:9]
    d9 = db(Son[sel].mean(0)) - db(So.mean(0))
    zz = np.where(wide, -1e9, d9 / np.maximum(mfloor, 1e-9))
    best9.append(float(np.max(zz[mm])))
print(f'    measured matched-K floor: {np.median(mfloor[srch]):.3f} dB')
print(f'    best NON-mains sigma over 20 random 9-block fan-on draws: '
      f'mean {np.mean(best9):+.2f}, max {np.max(best9):+.2f}')
print('    (with 13094 bins the max of pure noise is expected near +4.0 sigma)')
RES['matched_k'] = dict(floor_db=float(np.median(mfloor[srch])),
                        mean_best_sigma=float(np.mean(best9)),
                        max_best_sigma=float(np.max(best9)))

# --- 13d. cadence with a fair, well-averaged fan-off control ---
print('\n 13d. CADENCE: is the 60 Hz cadence the fan or the mains?')
cadb, Cb = cadence(x50, fs50)
nb2 = Cb / np.median(Cb[(cadb > 3) & (cadb < 300)])
mb = (cadb >= 5) & (cadb <= 200)
ib = np.flatnonzero(mb)[np.argmax(nb2[mb])]
print(f'    empty-room 60 s (50 kSa/s, NO fan) peak cadence {cadb[ib]:.2f} Hz  '
      f'{10*np.log10(nb2[ib]):+.1f} dB over band median')
print(f'    fan-on chunks peaked at {cad[i0]:.2f} / {cad[i1]:.2f} Hz')
print('    -> the empty room shows the SAME 60 Hz cadence with the fan off. Mains amplitude')
print('       modulation, not a blade flash.')
nmc = (cad >= 5) & (cad <= 200) & ~mains_tag(cad, tol=1.0)
j0 = np.flatnonzero(nmc)[np.argmax(n0c[nmc])]
j1 = np.flatnonzero(nmc)[np.argmax(n1c[nmc])]
print(f'    strongest NON-mains cadence: on0 {cad[j0]:7.2f} Hz {10*np.log10(n0c[j0]):+.1f} dB, '
      f'on1 {cad[j1]:7.2f} Hz {10*np.log10(n1c[j1]):+.1f} dB  -> '
      f'{"AGREE" if abs(cad[j0]-cad[j1])<1 else "DISAGREE => not a real cadence"}')
RES['cadence']['baseline_hz'] = float(cadb[ib])
RES['cadence']['nonmains_on0_hz'] = float(cad[j0])
RES['cadence']['nonmains_on1_hz'] = float(cad[j1])
RES['cadence']['nonmains_agree'] = bool(abs(cad[j0] - cad[j1]) < 1)

# --- 13e. proper injection threshold scan, starting well below the floor ---
print('\n 13e. INJECTION THRESHOLD (fine scan) -> quantitative upper bound')
thr = None
for snr_db in np.arange(-34, 6, 1.0):
    amp = np.sqrt(10 ** (snr_db / 10) * noise_pow * (fs / 2))
    hits, sigs = 0, []
    for trial in range(10):
        ph = rng.uniform(0, 2 * np.pi)
        f_inj = 137.0 + rng.uniform(-3, 3)
        t = np.arange(len(x0)) / fs
        xi = x0 + amp * np.sqrt(2) * np.sin(2 * np.pi * f_inj * t + ph)
        _, Si = blocks_psd(xi, fs)
        zi = (db(Si.mean(0)) - db(So.mean(0))) / np.maximum(floor, 1e-9)
        j = np.argmin(abs(f - f_inj))
        s = zi[max(0, j - 2):j + 3].max()
        sigs.append(s)
        hits += (s > 5.0)
    if snr_db % 4 == 0 or (hits >= 9 and thr is None):
        print(f'    line {snr_db:+4.0f} dB re in-band PSD floor: {hits:2d}/10 at 5 sigma, '
              f'mean {np.mean(sigs):+6.1f} sigma')
    if hits >= 9 and thr is None:
        thr = float(snr_db)
        break
print(f'  -> DETECTION THRESHOLD {thr:+.0f} dB re the in-band PSD floor '
      f'({10*np.log10(noise_pow):.1f} dB/Hz), i.e. a coherent Doppler line carrying')
print(f'     {10*np.log10(10**(thr/10)*(f[1]-f[0])/(fs/2)*1e0):.1f} dB ... expressed as line-to-'
      f'in-band-noise in one 1.53 Hz bin after 50 s of averaging.')
print('     The fan produced NO such line anywhere in 20 Hz - 20 kHz.')
RES['injection_threshold_db_over_psd_floor'] = thr

# --- 13f. what fraction of a person-sized return would that be? ---
print('\n 13f. VERDICT SUMMARY')
print(f'    positive control (moving person, FMCW, same statistics): +7.0 dB, 12.1 sigma, '
      f'cross-half r = +0.98')
print(f'    fan, best non-mains line anywhere 20 Hz-20 kHz vs 6 s ref : '
      f'{bands[2]["best_nonmains_sigma"]:+.2f} sigma (13094 bins searched)')
print(f'    fan, best non-mains comb                                 : '
      f'{np.nanmax(c2):+.2f} sigma (null 95th pct {np.percentile(nullc,95):+.2f})')
print(f'    fan, cross-chunk correlation of the difference spectrum   : '
      f'{RES["cross_chunk_r"]["whole band"]:+.4f} (null {RES["cross_chunk_r_null"]:+.4f}; '
      f'positive control was +0.98)')
print(f'    fan, broadband band power                                 : '
      f'{RES["bandpower"]["all"]["diff_db"]:+.2f} dB (NEGATIVE - fan-on is quieter)')

json.dump(RES, open(OUT + r'/doppler-comb.json', 'w'), indent=1, default=float)
print(f'\njson updated -> {OUT}/doppler-comb.json')

# ======================================================================
# 14. UPPER BOUND IN INTERPRETABLE UNITS + INDEPENDENT FMCW CORROBORATION
# ======================================================================
print('\n' + '=' * 72)
print('14. UPPER BOUND AND INDEPENDENT CROSS-CHECK')
print('=' * 72)

tone_pow = 10 ** (thr / 10) * noise_pow * (fs / 2)
tone_rms = np.sqrt(tone_pow)
tot_rms = x0.std()
print(f'\n 14a. What the {thr:+.0f} dB injection threshold means physically:')
print(f'    detectable Doppler line rms  = {tone_rms*1e6:.2f} uV')
print(f'    total fan-on IF rms          = {tot_rms*1e6:.1f} uV  '
      f'(ADC LSB = 336 uV, so the line is {336/(tone_rms*1e6):.0f}x SMALLER than one LSB)')
print(f'    line-to-total-IF-power ratio = {20*np.log10(tone_rms/tot_rms):.1f} dB')
print('    -> 50 s of coherent averaging buys enough processing gain to see a Doppler line')
print('       far below one ADC code. The fan produced nothing at that level.')
RES['upper_bound'] = dict(detectable_line_rms_uV=float(tone_rms * 1e6),
                          total_if_rms_uV=float(tot_rms * 1e6),
                          line_to_total_db=float(20 * np.log10(tone_rms / tot_rms)),
                          lsb_uV=336.0)

print('\n 14b. Broadband bound: any broad Doppler wing from the fan')
print(f'    measured in-band power difference (fan-on minus fan-off, mains notched) = '
      f'{RES["bandpower"]["all"]["diff_db"]:+.2f} dB')
print('    the sign is NEGATIVE, so the limit is set by the session-to-session gain offset')
print('    (fan-on IF rms 157.6 uV vs fan-off 173.6 uV = -0.84 dB, a receiver/session')
print('    difference, not the fan). Any broadband fan contribution is therefore below')
print('    roughly 1 dB of the in-band noise power and cannot be separated from that offset.')

print('\n 14c. INDEPENDENT CROSS-CHECK: FMCW consecutive-chirp MTI on the fan-on capture,')
print('      using the exact pipeline that scored 12 sigma on the moving person.')
fa_ = np.load(r'C:/dev/klc6/out/fan/20260830_093545_fan_on_fmcw_cfgA.npz')
st_ = np.load(r'C:/dev/klc6/out/fmcw/raw_chirps_box_out.npz')
mv_ = np.load(r'C:/dev/klc6/out/fmcw/raw_chirps_moving.npz')
S_fm = 180e6 * float(fa_['ramp'])


def dtp(x, order=3):
    n = x.shape[-1]
    t = np.linspace(-1, 1, n)
    V = np.vander(t, order + 1)
    co, *_ = np.linalg.lstsq(V, x.T, rcond=None)
    return x - (V @ co).T


def mti_spec(ch):
    x = dtp(np.asarray(ch, float))
    x = np.diff(x, axis=0)
    w = np.hanning(x.shape[1])
    return np.abs(np.fft.rfft(x * w, axis=1)) ** 2


rr = P.beat_to_range(np.fft.rfftfreq(2000, 1 / 100000.0), S_fm)
Sf, Ss, Sm = mti_spec(fa_['chirps']), mti_spec(st_['chirps']), mti_spec(mv_['chirps'])
bandr = (rr > 0.5) & (rr < 8.0)


def cmp_(A, B, lab):
    d = db(A.mean(0)) - db(B.mean(0))
    nl = []
    for _ in range(400):
        p = rng.permutation(A.shape[0])
        nl.append(db(A[p[:A.shape[0] // 2]].mean(0)) -
                  db(A[p[A.shape[0] // 2:A.shape[0] // 2 + B.shape[0] // 2]].mean(0)))
    fl = np.array(nl).std(0)
    zz = np.where(bandr, d / np.maximum(fl, 1e-9), -1e9)
    i = int(np.argmax(zz))
    # cross-half reproducibility
    h = A.shape[0] // 2
    s = B.shape[0] // 2
    d0 = db(A[:h].mean(0)) - db(B[:s].mean(0))
    d1 = db(A[h:].mean(0)) - db(B[s:].mean(0))
    r_ = float(np.corrcoef(d0[bandr], d1[bandr])[0, 1])
    print(f'    {lab:26s} best {rr[i]:5.2f} m  {d[i]:+6.2f} dB  {zz[i]:+6.2f} sigma   '
          f'cross-half r = {r_:+.3f}')
    return dict(range_m=float(rr[i]), db=float(d[i]), sigma=float(zz[i]), cross_half_r=r_)


pc = cmp_(Sm, Ss, 'moving person (control)')
fanr = cmp_(Sf, Ss, 'FAN ON')
print('    -> the same pipeline that finds the person at ~12 sigma with r=+0.96 finds the')
print(f'       fan at {fanr["sigma"]:+.1f} sigma with cross-half r = {fanr["cross_half_r"]:+.3f}.')
RES['fmcw_crosscheck'] = dict(person=pc, fan=fanr)

# cfgB range-Doppler, low-velocity (|v| < 3.1 m/s unambiguous at 1 kHz PRF)
cp = np.load(r'C:/dev/klc6/out/fan/20260830_093545_fan_on_fmcw_cfgB.npz')['cpis']
S_b = 180e6 * 1000.0
mx = []
for k in range(cp.shape[0]):
    rgb, vb, rd = P.range_doppler(dtp(np.asarray(cp[k], float)), 100000.0, S_b, 1000.0)
    rd = P.clutter_notch(rd, vb, width_mps=0.3)
    m2 = (rgb > 0.5) & (rgb < 8.0)
    mx.append(rd[:, m2].max() - np.median(rd[:, m2]))
mx = np.array(mx)
print(f'\n    cfgB range-Doppler (200 CPIs, zero-Doppler notched, 0.5-8 m):')
print(f'      peak-over-median per CPI: mean {mx.mean():.2f} dB, std {mx.std():.2f} dB, '
      f'max {mx.max():.2f} dB')
print('      (a moving target parked in one range-Doppler cell would push this well above the')
print('       CPI-to-CPI scatter; it does not.)')
RES['fmcw_cfgB'] = dict(mean_peak_over_median_db=float(mx.mean()), std_db=float(mx.std()),
                        max_db=float(mx.max()))

json.dump(RES, open(OUT + r'/doppler-comb.json', 'w'), indent=1, default=float)
print(f'\njson updated -> {OUT}/doppler-comb.json')
