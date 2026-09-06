"""Physical-optics facet scattering for the drone mesh.

Why this exists: scripts/rt_calib.py measured Sionna RT's geometric-optics
specular return from a blade-sized plate at +28 dB over physical optics --
image theory applied to a surface far smaller than the Fresnel zone. For a
target whose parts are ~1 wavelength across, PO on the actual facets is the
standard high-frequency method (POFACETS and every RCS code like it), and it
reproduces the closed-form plate RCS exactly by construction.

What is in here:
  * Gordon's closed-form PO integral over a planar polygon (edge sum), with a
    numerical self-test against brute-force quadrature.
  * Kirchhoff bistatic facet amplitude with the PEC or Fresnel-dielectric
    reflection coefficient.
  * Vectorised Moller-Trumbore occlusion so a blade behind the body is dark.
  * Four-path floor multipath by the image method with the complex Fresnel
    coefficient of concrete -- the one multipath surface every bench has.

Phase convention: exp(-j 2 k R) for a round trip of length 2R, matching
sim/drone.py, so decreasing range -> increasing phase -> positive Doppler.

Limits, stated: PO is good for facets >= ~lambda/2 near specular and degrades
toward grazing and at edges (PTD would be the next refinement). A blade chord
is ~1 lambda. This is the accuracy ceiling of the method, and it is far above
GO's for this target.
"""
from __future__ import annotations

import numpy as np

from .. import klc6

C = klc6.C
F0 = klc6.F_CARRIER
LAM = C / F0
K = 2.0 * np.pi / LAM

EPS0 = 8.854187817e-12
# complex relative permittivity at 24.125 GHz: eps' - j sigma/(omega eps0)
EPS_NYLON_GF = 3.7 - 1j * 0.075 / (2 * np.pi * F0 * EPS0)
EPS_CONCRETE = 5.24 - 1j * 0.80 / (2 * np.pi * F0 * EPS0)
PEC = None


# --------------------------------------------------------------- geometry
def tri_props(tris):
    """(N,3,3) -> centroids (N,3), unit normals (N,3), areas (N,)."""
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    n = np.cross(e1, e2)
    area2 = np.linalg.norm(n, axis=1)
    normals = n / np.maximum(area2, 1e-30)[:, None]
    return tris.mean(axis=1), normals, 0.5 * area2


def gordon_integral(tris, normals, w):
    """I = integral over each polygon of exp(j w . r) dA, w in the plane.

    Gordon (1975) via the 2-D divergence theorem:
        I = (1 / (j |w|^2)) sum_m [(n x w) . e_m] exp(j w . c_m) sinc(w . e_m / 2)
    with e_m the edge vectors, c_m the edge midpoints. For |w| -> 0, I -> area.
    tris (N,3,3), normals (N,3), w (N,3). Returns complex (N,).
    """
    N = tris.shape[0]
    wn = np.linalg.norm(w, axis=1)
    out = np.zeros(N, dtype=complex)
    small = wn < 1e-6
    # small-|w| limit: the area
    if small.any():
        _, _, a = tri_props(tris[small])
        out[small] = a
    big = ~small
    if big.any():
        T, nn, ww = tris[big], normals[big], w[big]
        nxw = np.cross(nn, ww)
        acc = np.zeros(big.sum(), dtype=complex)
        for m in range(3):
            p0, p1 = T[:, m], T[:, (m + 1) % 3]
            e = p1 - p0
            c = 0.5 * (p0 + p1)
            x = 0.5 * np.einsum("ij,ij->i", ww, e)
            sinc = np.where(np.abs(x) < 1e-9, 1.0, np.sin(x) / np.where(x == 0, 1, x))
            acc += np.einsum("ij,ij->i", nxw, e) * np.exp(1j * np.einsum("ij,ij->i", ww, c)) * sinc
        out[big] = acc / (1j * wn[big] ** 2)
    return out


def fresnel_gamma(cos_i, eps_r, pol="TE"):
    """Complex reflection coefficient of a half-space, incidence cosine cos_i.

    eps_r None -> PEC (Gamma = -1). pol 'TE' (E perpendicular to the plane of
    incidence), 'TM', or 'avg' (power-average of the two, phase of TE).
    """
    if eps_r is None:
        return -np.ones_like(np.asarray(cos_i, dtype=complex))
    cos_i = np.clip(np.asarray(cos_i, dtype=float), 0.0, 1.0)
    sin2 = 1.0 - cos_i ** 2
    root = np.sqrt(eps_r - sin2 + 0j)
    g_te = (cos_i - root) / (cos_i + root)
    g_tm = (eps_r * cos_i - root) / (eps_r * cos_i + root)
    if pol == "TE":
        return g_te
    if pol == "TM":
        return g_tm
    mag = np.sqrt(0.5 * (np.abs(g_te) ** 2 + np.abs(g_tm) ** 2))
    return mag * np.exp(1j * np.angle(g_te))


# ----------------------------------------------------------------- facets
def facet_amplitude(tris, normals, k_in, k_out, eps_r, pol="TE"):
    """Kirchhoff/PO bistatic amplitude per facet, sqrt(m^2), complex.

    k_in: unit vector of propagation FROM source TO facet (N,3).
    k_out: unit vector FROM facet TO observer (N,3).
    Monostatic: k_out = -k_in. Normalised so |amp|^2 = 4 pi A^2 / lambda^2 for
    a PEC facet at normal incidence. Only the lit side scatters: facets whose
    normal faces away from the source return zero.
    """
    cos_i = -np.einsum("ij,ij->i", normals, k_in)          # incidence cosine
    cos_s = np.einsum("ij,ij->i", normals, k_out)
    lit = cos_i > 1e-6
    # tangential part of k (k_out - k_in): w = K * ((k_out - k_in) - n (n.(k_out - k_in)))
    d = k_out - k_in
    w = K * (d - normals * np.einsum("ij,ij->i", normals, d)[:, None])
    # SHAPE FACTOR ONLY: integrate over the facet in coordinates centred on its
    # own centroid. The position phase is applied separately by the caller as
    # exp(-j 2 k R_centroid). Evaluating Gordon in world coordinates instead
    # puts exp(j w . c) -- the plane-wave position phase -- into I as well, and
    # the spatial phase is then counted twice: every facet's phase advances at
    # 2x the physical rate and rotor energy lands at up to 2x the tip Doppler.
    # That bug survived every origin-centred self-test; see selftest() item 4.
    cen = tris.mean(axis=1)
    I = gordon_integral(tris - cen[:, None, :], normals, w)
    obliq = 0.5 * (np.clip(cos_i, 0, 1) + np.clip(cos_s, 0, 1))
    gamma = fresnel_gamma(cos_i, eps_r, pol)
    amp = (K / np.sqrt(np.pi)) * obliq * I * gamma
    return np.where(lit, amp, 0.0)


def _segments_blocked(orig, dirs, dist, blockers, chunk=256):
    """Moller-Trumbore: True where segment orig -> orig + dirs*dist hits a blocker."""
    N = orig.shape[0]
    v0 = blockers[:, 0]
    e1 = blockers[:, 1] - v0
    e2 = blockers[:, 2] - v0
    blocked = np.zeros(N, dtype=bool)
    for s in range(0, N, chunk):
        o = orig[s:s + chunk]
        dd = dirs[s:s + chunk]
        L = dist[s:s + chunk]
        h = np.cross(dd[:, None, :], e2[None, :, :])            # (F, T, 3)
        a = np.einsum("tj,ftj->ft", e1, h)
        ok = np.abs(a) > 1e-12
        f = 1.0 / np.where(ok, a, 1.0)
        sv = o[:, None, :] - v0[None, :, :]
        u = f * np.einsum("ftj,ftj->ft", sv, h)
        q = np.cross(sv, e1[None, :, :])
        v = f * np.einsum("fj,ftj->ft", dd, q)
        t = f * np.einsum("tj,ftj->ft", e2, q)
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6) & (t < L[:, None] - 1e-6)
        blocked[s:s + chunk] = hit.any(axis=1)
    return blocked


def occlusion_mask(tris, normals, src, blockers, soft=True):
    """Visibility of each facet from src, in [0, 1].

    Binary per-facet visibility flickers as a blade edge grazes a motor top
    between adjacent rotor angles, and that flicker is broadband noise in the
    spectrogram that no real radar sees. With soft=True the facet is sampled
    at its centroid and at three points 2/3 of the way to each vertex, and the
    visible fraction is returned -- a crude but effective anti-aliasing of the
    shadow boundary. Each sample ray starts a hair off the facet toward the
    source so the facet cannot occlude itself.
    """
    cen = tris.mean(axis=1)
    if soft:
        pts = [cen] + [cen + (tris[:, k] - cen) * (2.0 / 3.0) for k in range(3)]
    else:
        pts = [cen]
    vis = np.zeros(len(tris))
    for p in pts:
        d = src[None, :] - p
        dist = np.linalg.norm(d, axis=1)
        dirs = d / dist[:, None]
        side = np.sign(np.einsum("ij,ij->i", normals, dirs))[:, None]
        orig = p + normals * side * 1e-4
        vis += ~_segments_blocked(orig, dirs, dist, blockers)
    return vis / len(pts)


# ---------------------------------------------------------------- target
def part_return(tris, radar, blockers=None, eps_r=PEC, pol="TE",
                floor_z=None, eps_floor=EPS_CONCRETE, r_ref=None):
    """Complex 'RCS amplitude' of one part (m), referenced to range r_ref.

    Sums lit, unoccluded facets with the exact two-way phase. With floor_z set,
    adds the three ground-bounce paths of the image method: source->facet via
    ground then back direct, direct out then ground back, and ground both ways,
    each with the concrete Fresnel coefficient at the grazing angle it uses.

    |return|^2 is the effective RCS in m^2 at the reference range, and the
    phase is what a CW radar's I/Q would show.
    """
    radar = np.asarray(radar, dtype=float)
    cen, nrm, _ = tri_props(tris)
    r_ref = float(np.linalg.norm(radar - cen.mean(axis=0))) if r_ref is None else r_ref
    if blockers is not None and len(blockers):
        vis = occlusion_mask(tris, nrm, radar, blockers)      # fraction in [0, 1]
    else:
        vis = np.ones(len(tris))

    def leg(src):
        d = cen - src[None, :]
        R = np.linalg.norm(d, axis=1)
        return d / R[:, None], R                             # k_in from src, range

    k_dir, R_dir = leg(radar)
    total = np.zeros(len(tris), dtype=complex)

    # direct-direct
    amp = facet_amplitude(tris, nrm, k_dir, -k_dir, eps_r, pol)
    total += amp * np.exp(-1j * K * 2 * R_dir) * (r_ref / R_dir) ** 2

    if floor_z is not None:
        img = radar.copy()
        img[2] = 2 * floor_z - radar[2]
        k_img, R_img = leg(img)
        # grazing geometry for the ground reflection on the image leg
        cos_g = np.abs(k_img[:, 2])                          # cos of incidence at the floor
        g = fresnel_gamma(cos_g, eps_floor, pol)
        # ground out, direct back
        a1 = facet_amplitude(tris, nrm, k_img, -k_dir, eps_r, pol)
        total += g * a1 * np.exp(-1j * K * (R_img + R_dir)) * (r_ref ** 2 / (R_img * R_dir))
        # direct out, ground back
        a2 = facet_amplitude(tris, nrm, k_dir, -k_img, eps_r, pol)
        total += g * a2 * np.exp(-1j * K * (R_dir + R_img)) * (r_ref ** 2 / (R_img * R_dir))
        # ground both ways
        a3 = facet_amplitude(tris, nrm, k_img, -k_img, eps_r, pol)
        total += g * g * a3 * np.exp(-1j * K * 2 * R_img) * (r_ref / R_img) ** 2

    # visibility is a fraction of the facet AREA seen, so it scales amplitude
    # by that fraction (the PO integral is linear in the illuminated area)
    return complex(np.sum(total * vis))


# --------------------------------------------------------------- selftest
def selftest(verbose=True):
    """Gordon vs an exact closed form, and the flat-plate RCS check.

    A rectangle a x b centred on the origin has the exact PO integral
        I = a b sinc(w_x a / 2) sinc(w_y b / 2),   sinc(x) = sin(x)/x.
    Split it into two triangles; Gordon summed over both must match to
    machine precision at any w, including where the integral has cancelled
    to a tiny fraction of the area (a grid quadrature cannot get there).
    """
    rng = np.random.default_rng(1)
    ok = True
    a_, b_ = 0.012, 0.060                              # a blade-sized rectangle
    ha, hb = a_ / 2, b_ / 2
    rect = np.array([[[-ha, -hb, 0], [ha, -hb, 0], [ha, hb, 0]],
                     [[-ha, -hb, 0], [ha, hb, 0], [-ha, hb, 0]]], dtype=float)
    nrm = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

    def sinc(x):
        return np.where(np.abs(x) < 1e-12, 1.0, np.sin(x) / np.where(x == 0, 1, x))

    for _ in range(6):
        wx, wy = rng.uniform(-3 * K, 3 * K), rng.uniform(-3 * K, 3 * K)
        w = np.array([[wx, wy, 0.0], [wx, wy, 0.0]])
        Ig = gordon_integral(rect, nrm, w).sum()
        Ie = a_ * b_ * sinc(wx * ha) * sinc(wy * hb)
        err = abs(Ig - Ie) / max(abs(Ie), 1e-15)
        ok &= err < 1e-6
        if verbose:
            print(f"  gordon vs exact sinc product |w|={np.hypot(wx, wy):7.1f}: "
                  f"{abs(Ig):.4e} vs {abs(Ie):.4e}  rel err {err:.1e}")
    # 2. plate RCS at normal incidence = 4 pi A^2 / lambda^2
    side = 0.06
    h = side / 2
    plate = np.array([[[0, -h, -h], [0, h, -h], [0, h, h]],
                      [[0, -h, -h], [0, h, h], [0, -h, h]]], dtype=float)
    A = side ** 2
    sigma_po = 4 * np.pi * A ** 2 / LAM ** 2
    amp = part_return(plate, radar=np.array([3.0, 0.0, 0.0]), eps_r=PEC)
    sigma = abs(amp) ** 2
    err_db = 10 * np.log10(sigma / sigma_po)
    ok &= abs(err_db) < 0.1
    if verbose:
        print(f"  6 cm PEC plate, normal incidence: PO {10*np.log10(sigma_po):.2f} dBsm, "
              f"facets {10*np.log10(sigma):.2f} dBsm  ({err_db:+.2f} dB)")
    # 3. 10 deg off normal should be well down (first null ~ lambda/D = 11.9 deg)
    R = 3.0
    ang = np.radians(10.0)
    radar = np.array([R * np.cos(ang), R * np.sin(ang), 0.0])
    amp10 = part_return(plate, radar=radar, eps_r=PEC)
    if verbose:
        print(f"  same plate 10 deg off normal: {10*np.log10(abs(amp10)**2+1e-30):.1f} dBsm "
              f"({10*np.log10(abs(amp10)**2/sigma):+.1f} dB re normal)")
    # 4. TRANSLATION INVARIANCE. Move plate and radar together by an arbitrary
    #    offset: |RCS| must not change, and the phase must not change either
    #    (same range, same aspect). This is the test that catches a spatial
    #    phase being counted twice -- everything above sits at the origin and
    #    cannot see it.
    shift = np.array([0.137, -0.291, 0.462])
    amp_a = part_return(plate, radar=radar, eps_r=PEC, r_ref=R)
    amp_b = part_return(plate + shift, radar=radar + shift, eps_r=PEC, r_ref=R)
    mag_db = 20 * np.log10(abs(amp_b) / max(abs(amp_a), 1e-30))
    dphase = np.angle(amp_b / amp_a)
    ok &= abs(mag_db) < 1e-6 and abs(dphase) < 1e-6
    if verbose:
        print(f"  translation invariance (plate+radar shifted {np.linalg.norm(shift):.3f} m): "
              f"|amp| {mag_db:+.2e} dB, phase {dphase:+.2e} rad")
    # 5. Facet subdivision: one plate as 2 triangles vs the same plate as 32
    #    triangles must agree -- the PO integral is additive over the surface
    #    ONLY if each facet carries its own position phase correctly.
    n = 4
    xs = np.linspace(-h, h, n + 1)
    fine = []
    for i in range(n):
        for j in range(n):
            y0, y1, z0, z1 = xs[i], xs[i + 1], xs[j], xs[j + 1]
            fine.append([[0, y0, z0], [0, y1, z0], [0, y1, z1]])
            fine.append([[0, y0, z0], [0, y1, z1], [0, y0, z1]])
    fine = np.array(fine, dtype=float)
    amp_fine = part_return(fine, radar=radar, eps_r=PEC, r_ref=R)
    sub_db = 20 * np.log10(abs(amp_fine) / max(abs(amp10), 1e-30))
    ok &= abs(sub_db) < 0.2
    if verbose:
        print(f"  subdivision (2 vs {len(fine)} facets, 10 deg off normal): {sub_db:+.3f} dB")
    return ok
