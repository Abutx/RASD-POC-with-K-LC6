import sys
import numpy as np
sys.path.insert(0, 'C:/dev/klc6')
from scipy.signal import spectrogram, detrend
from klc6 import process as P

FANON = ['C:/dev/klc6/out/fan/20260830_093545_fan_on_cw0_100k.npz',
         'C:/dev/klc6/out/fan/20260830_093545_fan_on_cw1_100k.npz']
OFF100 = 'C:/dev/klc6/out/cw/smoke_static.npz'
OFF50 = 'C:/dev/klc6/out/baseline/20260829_061237_empty_baseline_60s.npz'


def load(f):
    d = np.load(f, allow_pickle=True)
    return d['data'][0].astype(float), float(d['fs'])


def cvd(f, nper, hop, flo, fhi, block, norm=True, hp=True):
    x, fs = load(f)
    x = x - x.mean()
    if fs < 75000:
        nper //= 2
        hop //= 2
    fr, t, S = spectrogram(x, fs=fs, nperseg=nper, noverlap=nper - hop, window='hann',
                           scaling='spectrum', mode='magnitude', detrend=False)
    sel = (fr >= flo) & (fr <= fhi)
    S = S[sel]
    if norm:
        S = S / np.maximum(S.mean(axis=1, keepdims=True), 1e-30) - 1.0
    dt = float(t[1] - t[0])
    w = np.hanning(block)
    wn = (w ** 2).sum()
    cad = np.fft.rfftfreq(block, dt)
    out = []
    for s0 in range(0, S.shape[1] - block + 1, block // 2):
        B = detrend(S[:, s0:s0 + block], axis=1, type='linear')
        out.append((np.abs(np.fft.rfft(B * w[None, :], axis=1)) ** 2 / wn).mean(axis=0))
    return cad, np.asarray(out), int(sel.sum())


def z(a, b):
    return (a.mean() - b.mean()) / np.sqrt(a.std(ddof=1) ** 2 / len(a) + b.std(ddof=1) ** 2 / len(b))


print('band Hz        (m/s)         nper block  Kon/Koff nbins | band-int z(on-off) | best single-bin z @ Hz')
for flo, fhi in [(20, 200), (20, 1000), (200, 2000), (200, 20000), (1000, 8000),
                 (3000, 20000), (20, 49000), (2000, 6000)]:
    for nper, block in [(1024, 512), (512, 512), (256, 1024)]:
        ks = []
        for f in FANON:
            cad, k, nb = cvd(f, nper, nper // 4, flo, fhi, block)
            ks.append(k)
        kon = np.vstack(ks)
        cad, koff, nb = cvd(OFF100, nper, nper // 4, flo, fhi, block)
        bb = (cad >= 2) & (cad <= min(190, cad[-1]))
        zb = z(kon[:, bb].mean(axis=1), koff[:, bb].mean(axis=1))
        mA = kon.mean(0); sA = kon.std(0, ddof=1) / np.sqrt(len(kon))
        mB = koff.mean(0); sB = koff.std(0, ddof=1) / np.sqrt(len(koff))
        zz = (mA - mB) / np.sqrt(sA ** 2 + sB ** 2)
        i = int(np.argmax(np.where(bb, zz, -1e9)))
        print('%5.0f-%-6.0f (%5.2f-%6.1f) %5d %5d  %3d/%2d %5d | %+7.2f | %+6.2f @ %7.2f Hz  max|z| %5.2f'
              % (flo, fhi, flo / P.HZ_PER_MPS, fhi / P.HZ_PER_MPS, nper, block,
                 len(kon), len(koff), nb, zb, zz[i], cad[i], np.abs(zz[bb]).max()))
    print()
