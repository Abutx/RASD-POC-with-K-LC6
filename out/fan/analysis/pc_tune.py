"""Tune the CVD statistic until it detects the KNOWN moving person."""
import sys
import numpy as np
sys.path.insert(0, 'C:/dev/klc6')
from scipy.signal import detrend
from scipy.ndimage import median_filter
from klc6 import process as P

MOVING = 'C:/dev/klc6/out/fmcw/raw_chirps_moving.npz'
BOXOUT = 'C:/dev/klc6/out/fmcw/raw_chirps_box_out.npz'
FANFMA = 'C:/dev/klc6/out/fan/20260830_093545_fan_on_fmcw_cfgA.npz'
S_SLOPE = 180e6 * 50.0
PRF = 50.0


def rangemap(f, mti=False, nfft=4096):
    d = np.load(f, allow_pickle=True)
    ch = d['chirps'].astype(float)
    n = ch.shape[1]
    tt = np.linspace(-1, 1, n)
    V = np.vander(tt, 4)
    co, *_ = np.linalg.lstsq(V, ch.T, rcond=None)
    ch = ch - (V @ co).T
    R = np.fft.rfft(ch * np.hanning(n)[None, :], n=nfft, axis=1)
    rng = P.beat_to_range(np.fft.rfftfreq(nfft, 1 / 1e5), S_SLOPE)
    if mti:
        R = np.diff(R, axis=0)
    return rng, np.abs(R)


def cvd(A, sel, block, hop, dt, norm=True):
    M = A[:, sel].T.astype(float)
    if norm:
        M = M / np.maximum(M.mean(axis=1, keepdims=True), 1e-30) - 1.0
    w = np.hanning(block)
    wn = (w ** 2).sum()
    cad = np.fft.rfftfreq(block, dt)
    out = []
    for s0 in range(0, M.shape[1] - block + 1, hop):
        B = detrend(M[:, s0:s0 + block], axis=1, type='linear')
        out.append((np.abs(np.fft.rfft(B * w[None, :], axis=1)) ** 2 / wn).mean(axis=0))
    return cad, np.asarray(out)


def zband(kA, kB, sel):
    a = kA[:, sel].mean(axis=1)
    b = kB[:, sel].mean(axis=1)
    return (a.mean() - b.mean()) / np.sqrt(a.std(ddof=1) ** 2 / len(a) + b.std(ddof=1) ** 2 / len(b))


for mti in [False, True]:
    rng, Am = rangemap(MOVING, mti)
    _, As = rangemap(BOXOUT, mti)
    _, Af = rangemap(FANFMA, mti)
    sel = (rng >= 0.6) & (rng <= 4.0)
    for norm in [True, False]:
        for block, hop in [(64, 16), (96, 16), (128, 16), (160, 8)]:
            cad, km = cvd(Am, sel, block, hop, 1 / PRF, norm)
            _, ks = cvd(As, sel, block, hop, 1 / PRF, norm)
            _, kf = cvd(Af, sel, block, hop, 1 / PRF, norm)
            bb = (cad >= 0.8) & (cad <= 25)
            lo = (cad >= 0.8) & (cad <= 4.0)
            zm = zband(km, ks, bb)
            zf = zband(kf, ks, bb)
            zml = zband(km, ks, lo)
            zfl = zband(kf, ks, lo)
            # sharp-line statistic
            mm = km.mean(axis=0)
            ss = km.std(axis=0, ddof=1) / np.sqrt(km.shape[0])
            base = median_filter(mm, 21, mode='nearest')
            zpk = np.max(((mm - base) / np.maximum(ss, 1e-30))[bb])
            print('mti=%-5s norm=%-5s block=%3d K=%2d/%2d | person band %6.2f sig, lowcad %6.2f sig | '
                  'FAN band %6.2f sig, lowcad %6.2f sig | person sharp-line %5.2f'
                  % (mti, norm, block, km.shape[0], ks.shape[0], zm, zml, zf, zfl, zpk))
    print()
