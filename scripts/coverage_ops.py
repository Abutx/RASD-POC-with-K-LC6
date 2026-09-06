#!/usr/bin/env python3
"""How to catch every drone at DKR without losing range or blowing the budget.

Uses the surveyed DKR footprint (OpenStreetMap relation 1639372) and the same
beam model as the live sim to score operating concepts on the metric that
matters for a perimeter system: of drones that fly in from outside, what
fraction is inside SOME node's beam at SOME point before the wall, and how far
out. Range is held fixed (TinyRad, Mini 3 body) so the comparison isolates
COVERAGE. Then: what it takes to push TinyRad to 150 m, rain at 24 vs 77 GHz,
and the licence question.

  python scripts/coverage_ops.py [path/to/dkr_map.json]
"""
import json, math, os, sys, random
from pathlib import Path

import numpy as np

MAP = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "viz" / "dkr" / "dkr_map.json"
M = json.loads(MAP.read_text(encoding="utf-8"))
OUTER = np.array(M["stadium"][0], dtype=float)           # x east, y north, closed ring
CEN = OUTER.mean(axis=0)
random.seed(7); np.random.seed(7)

# ------------------------------------------------------------- geometry
def signed_area(r):
    x, y = r[:, 0], r[:, 1]
    return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
CCW = signed_area(OUTER) > 0

def point_in_ring(px, py, r=OUTER):
    inside = np.zeros_like(px, dtype=bool)
    x, y = r[:, 0], r[:, 1]
    n = len(r)
    j = n - 1
    for i in range(n):
        xi, yi, xj, yj = x[i], y[i], x[j], y[j]
        cond = ((yi > py) != (yj > py)) & (px < (xj - xi) * (py - yi) / ((yj - yi) if yj != yi else 1e-12) + xi)
        inside ^= cond
        j = i
    return inside

def dist_to_ring(px, py, r=OUTER):
    best = np.full_like(px, 1e18, dtype=float)
    for i in range(len(r) - 1):
        ax, ay = r[i]; bx, by = r[i + 1]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy or 1e-9
        t = np.clip(((px - ax) * dx + (py - ay) * dy) / L2, 0, 1)
        d2 = (ax + t * dx - px) ** 2 + (ay + t * dy - py) ** 2
        best = np.minimum(best, d2)
    return np.sqrt(best)

PERIM = float(np.sum(np.hypot(np.diff(OUTER[:, 0]), np.diff(OUTER[:, 1]))))

# ------------------------------------------------------------ occlusion
# Buildings are opaque prisms (OSM footprint + height). A target is blocked when the
# straight line from the radar passes below a roof anywhere inside a footprint. Per
# radar position we pre-cast one ray per 0.5 deg azimuth bin and record, for every
# building it crosses, (r_near, r_far, roof). The line's height is linear in range,
# so its minimum over the crossing is at r_near or r_far.
OCCLUDE = os.environ.get("DKR_NO_OCCLUSION", "") == ""
N_AZ = 720
def _closed(p):
    p = np.array(p, dtype=float)
    return p if np.allclose(p[0], p[-1]) else np.vstack([p, p[:1]])
BUILDINGS = [(_closed(b["p"]), min(float(b["h"]), 70.0)) for b in M.get("buildings", []) if len(b["p"]) >= 4]
_OCC = {}
class Occluder:
    def __init__(self, x, y, h):
        self.x, self.y, self.h = x, y, h
        az = (np.arange(N_AZ) + 0.5) * (2 * np.pi / N_AZ)
        dx, dy = np.cos(az), np.sin(az)
        rn, rf, hb = [], [], []
        for poly, hh in BUILDINGS:
            if point_in_ring(np.array([x]), np.array([y]), poly)[0]:
                continue                                    # part of the structure the node sits on
            ax, ay = poly[:-1, 0], poly[:-1, 1]; ex, ey = np.diff(poly[:, 0]), np.diff(poly[:, 1])
            den = dx[:, None] * ey[None, :] - dy[:, None] * ex[None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = ((ax[None, :] - x) * ey[None, :] - (ay[None, :] - y) * ex[None, :]) / den
                s = ((ax[None, :] - x) * dy[:, None] - (ay[None, :] - y) * dx[:, None]) / den
            hit = (np.abs(den) > 1e-12) & (s >= 0) & (s <= 1) & (t > 0)
            cnt = hit.sum(axis=1)
            if not (cnt >= 2).any():
                continue
            near = np.where(hit, t, np.inf).min(axis=1); far = np.where(hit, t, -np.inf).max(axis=1)
            ok = cnt >= 2
            rn.append(np.where(ok, near, np.inf)); rf.append(np.where(ok, far, np.inf)); hb.append(np.full(N_AZ, hh))
        self.k = len(rn)
        if self.k:
            self.rn = np.stack(rn, axis=1); self.rf = np.stack(rf, axis=1); self.hb = np.stack(hb, axis=1)   # (N_AZ, K)
    def blocked(self, P):
        if not self.k:
            return np.zeros(P.shape[:-1], dtype=bool)
        vx, vy, zt = P[..., 0] - self.x, P[..., 1] - self.y, P[..., 2]
        r = np.hypot(vx, vy); r = np.where(r < 1e-6, 1e-6, r)
        b = (np.floor((np.arctan2(vy, vx) % (2 * np.pi)) / (2 * np.pi / N_AZ)).astype(int)) % N_AZ
        RN, RF, HB = self.rn[b], self.rf[b], self.hb[b]                   # (..., K)
        rr = r[..., None]; dz = (zt - self.h)[..., None]
        with np.errstate(invalid="ignore"):                 # inf * 0 where no building: never < roof
            zn = self.h + dz * RN / rr; zf = self.h + dz * np.minimum(RF, rr) / rr
        return ((RN < rr) & (np.minimum(zn, zf) < HB)).any(axis=-1)
def occluder(x, y, h):
    key = (round(x, 1), round(y, 1), round(h, 1))
    if key not in _OCC:
        _OCC[key] = Occluder(x, y, h)
    return _OCC[key]

def wall_points(n):
    pts = []
    seg_len = np.hypot(np.diff(OUTER[:, 0]), np.diff(OUTER[:, 1]))
    cum = np.concatenate([[0], np.cumsum(seg_len)])
    for k in range(n):
        d = (k + 0.5) / n * PERIM
        i = int(np.searchsorted(cum, d, side="right") - 1)
        i = min(i, len(seg_len) - 1)
        t = (d - cum[i]) / seg_len[i]
        a, b = OUTER[i], OUTER[i + 1]
        x, y = a + (b - a) * t
        ex, ey = b - a
        nx, ny = (ey, -ex) if not CCW else (-ey, ex)
        L = math.hypot(nx, ny); nx /= L; ny /= L
        if (x + nx - CEN[0]) ** 2 + (y + ny - CEN[1]) ** 2 < (x - CEN[0]) ** 2 + (y - CEN[1]) ** 2:
            nx, ny = -nx, -ny
        pts.append((x, y, nx, ny))
    return pts

# ---------------------------------------------------------------- beams
class Node:
    def __init__(self, x, y, h, nx, ny, tilt_deg, az_bw, el_bw, R):
        self.p = np.array([x, y, h]); self.yaw = math.atan2(ny, nx)
        self.tilt = math.radians(tilt_deg); self.haz = math.radians(az_bw / 2); self.hel = math.radians(el_bw / 2); self.R = R
        cy, sy, ct, st = math.cos(self.yaw), math.sin(self.yaw), math.cos(self.tilt), math.sin(self.tilt)
        self.f = np.array([cy * ct, sy * ct, st]); self.r = np.array([-sy, cy, 0.0]); self.u = np.cross(self.r, self.f)
        self.occ = occluder(x, y, h) if OCCLUDE else None
    def covers(self, P):                              # P (...,3): x, y, z(up)
        v = P - self.p
        d2 = np.sum(v * v, axis=-1)
        lx = v @ self.f; ly = v @ self.u; lz = v @ self.r
        az = np.arctan2(lz, np.maximum(lx, 1e-9)); el = np.arctan2(ly, np.hypot(lx, lz))
        seen = (d2 <= self.R ** 2) & (lx > 0) & (np.abs(az) <= self.haz) & (np.abs(el) <= self.hel)
        if self.occ is not None and seen.any():
            seen &= ~self.occ.blocked(P)
        return seen

def sector_tilts(k, h, depth, el_bw, street=True, top_alt=60.0):
    """Operating concept: ONE sector looks down at the street mid-zone, the rest
    are spread over the drone band (5 m at the zone edge .. 60 m at 40 m out).
    k = 1 -> a single sheet aimed at the centre of the drone band."""
    d_lo = math.degrees(math.atan2(5.0 - h, depth))          # low drone, far
    d_hi = math.degrees(math.atan2(top_alt - h, 40.0))       # high drone, near
    if k == 1:
        return [0.5 * (d_lo + d_hi)]
    out = []
    kd = k
    if street:
        out.append(math.degrees(math.atan2(1.5 - h, 60.0)))  # street 60 m out (covers ~35-150 m of street)
        kd = k - 1
    if kd == 1:
        out.append(0.5 * (d_lo + d_hi))
    elif kd > 1:
        out += list(np.linspace(d_lo + el_bw / 2, d_hi - el_bw / 2, kd))
    return out

def build(n_nodes, heights, k_sectors, az_bw, el_bw, R, depth, tilt_mode="spread", yaw_stagger=0.0, street=True):
    nodes = []
    for h in heights:
        for i, (x, y, nx, ny) in enumerate(wall_points(n_nodes)):
            if yaw_stagger:
                a = math.atan2(ny, nx) + math.radians(yaw_stagger) * (1 if i % 2 else -1); nx, ny = math.cos(a), math.sin(a)
            if tilt_mode == "spread":
                tilts = sector_tilts(k_sectors, h, depth, el_bw, street=street)
            else:
                tilts = [tilt_mode]
            for t in tilts:
                nodes.append(Node(x, y, h, nx, ny, t, az_bw, el_bw, R))
    return nodes

def miss_profile(nodes, runs):
    """Where do the uncaught tracks live? Altitude histogram of misses."""
    bins = [5, 15, 25, 35, 45, 60.01]; miss = np.zeros(len(bins) - 1); tot = np.zeros(len(bins) - 1)
    for xs, ys, alt in runs:
        P = np.stack([xs, ys, np.full_like(xs, alt)], axis=-1)
        inside = point_in_ring(xs, ys); d = dist_to_ring(xs, ys); valid = (~inside) & (d > 6)
        seen = np.zeros(len(xs), dtype=bool)
        for n in nodes: seen |= n.covers(P)
        b = np.searchsorted(bins, alt, side="right") - 1; b = min(max(b, 0), len(tot) - 1)
        tot[b] += 1; miss[b] += 0 if (seen & valid).any() else 1
    return [(bins[i], bins[i + 1], miss[i], tot[i]) for i in range(len(tot))]

# ----------------------------------------------------------- metrics
def zone_grid(depth, cell=6.0):
    ext = float(np.max(np.abs(OUTER - CEN))) + depth + 20
    xs = np.arange(CEN[0] - ext, CEN[0] + ext, cell); ys = np.arange(CEN[1] - ext, CEN[1] + ext, cell)
    X, Y = np.meshgrid(xs, ys); X = X.ravel(); Y = Y.ravel()
    m = (~point_in_ring(X, Y)) & (dist_to_ring(X, Y) <= depth)
    return X[m], Y[m]

def volume_coverage(nodes, X, Y, alts):
    hit = np.zeros(len(X), dtype=bool); per_alt = []
    for a in alts:
        P = np.stack([X, Y, np.full_like(X, a)], axis=-1)
        h = np.zeros(len(X), dtype=bool)
        for n in nodes: h |= n.covers(P)
        per_alt.append(h.mean()); hit |= h
    return float(np.mean(per_alt)), per_alt

def street_coverage(nodes, X, Y):
    P = np.stack([X, Y, np.full_like(X, 1.5)], axis=-1)
    h = np.zeros(len(X), dtype=bool)
    for n in nodes: h |= n.covers(P)
    return float(h.mean())

def approaches(n_runs, alt_lo, alt_hi, start_out=320.0, speed=12.0, dt=0.25):
    """Straight-line inbound drone tracks from start_out beyond the wall to the wall."""
    runs = []
    for _ in range(n_runs):
        b = random.uniform(0, 2 * math.pi); alt = random.uniform(alt_lo, alt_hi)
        ext = float(np.max(np.abs(OUTER - CEN)))
        sx, sy = CEN + (ext + start_out) * np.array([math.cos(b), math.sin(b)]) + np.random.uniform(-60, 60, 2)
        tx, ty = CEN + np.random.uniform(-40, 40, 2)
        L = math.hypot(tx - sx, ty - sy); steps = int(L / (speed * dt))
        xs = np.linspace(sx, tx, steps); ys = np.linspace(sy, ty, steps)
        runs.append((xs, ys, alt))
    return runs

def approach_stats(nodes, runs):
    caught = 0; warn = []
    for xs, ys, alt in runs:
        P = np.stack([xs, ys, np.full_like(xs, alt)], axis=-1)
        inside = point_in_ring(xs, ys); d = dist_to_ring(xs, ys)
        valid = (~inside) & (d > 6)
        seen = np.zeros(len(xs), dtype=bool)
        for n in nodes: seen |= n.covers(P)
        seen &= valid
        if seen.any():
            caught += 1; warn.append(float(d[np.argmax(seen)]))
        else:
            warn.append(0.0)
    return caught / len(runs), float(np.mean(warn)), float(np.median([w for w in warn if w > 0] or [0]))

# ------------------------------------------------------------------ main
def main():
    depth = 150.0
    X, Y = zone_grid(depth)
    alts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    runs = approaches(600, 5, 60)
    AZ, EL = 75.0, 15.0
    R_TEXT, R_CONS, R_ENH = 113.0, 38.0, 155.0
    CAR_SCALE_LOCAL = (10 / 0.02) ** 0.25
    MODULE, SERVO, MOUNT = 400.0, 35.0, 120.0

    print("=" * 96)
    print("DKR PERIMETER -- WHY DRONES ARE MISSED, AND WHAT CATCHES THEM  (zone 150 m, drones 5-60 m)")
    print("=" * 96)
    print(f"footprint perimeter {PERIM:.0f} m, {len(X)} zone cells, {len(runs)} inbound drone tracks at 12 m/s\n")

    # elevation demand
    for h in (45, 25, 12):
        print(f"  mount {h:2d} m: street at zone edge {math.degrees(math.atan2(1.5-h,150)):+6.1f} deg | "
              f"street 40 m out {math.degrees(math.atan2(1.5-h,40)):+6.1f} | drone 60 m @100 m {math.degrees(math.atan2(60-h,100)):+6.1f} | "
              f"drone 60 m @40 m {math.degrees(math.atan2(60-h,40)):+6.1f}  -> needs ~{math.degrees(math.atan2(60-h,40))-math.degrees(math.atan2(1.5-h,150)):.0f} deg of elevation; one sheet is {EL:.0f}")
    print()

    # sim's 'split' aim from the roof: tilt to hit 12 m altitude at 60% of range
    split_tilt = math.degrees(math.atan2(12 - 45, 0.6 * R_TEXT))
    configs = [
        # label, nodes, heights, sectors, R, stagger, extra(list of (n, h, k, az, el, R, street))
        ("A  sim as shipped: 12 roof nodes, 1 sheet, split aim (%.0f deg)" % split_tilt, 12, [45], split_tilt, R_TEXT, 0, None),
        ("A2 12 roof nodes, 1 sheet aimed at the drone band (+3 deg)", 12, [45], 1, R_TEXT, 0, None),
        ("B  12 roof, nodding 2 sectors (1 street + 1 drone)",           12, [45], 2, R_TEXT, 0, None),
        ("C  12 roof, nodding 3 sectors (1 street + 2 drone)",           12, [45], 3, R_TEXT, 0, None),
        ("D  12 roof, nodding 4 sectors (1 street + 3 drone)",           12, [45], 4, R_TEXT, 0, None),
        ("D2 12 roof, 4 sectors ALL drone (no street look)",             12, [45], 4, R_TEXT, 0, "nostreet"),
        ("E  16 roof, 4 sectors",                                        16, [45], 4, R_TEXT, 0, None),
        ("F  24 roof, 1 sheet aimed at the drone band",                  24, [45], 1, R_TEXT, 0, None),
        ("G  two rings 45 m + 20 m, 12 each, 2 sectors each",            12, [45, 20], 2, R_TEXT, 0, None),
        ("I  12 nodes at mid-height 25 m, 4 sectors",                    12, [25], 4, R_TEXT, 0, None),
        ("K  D at CONSERVATIVE range 38 m",                              12, [45], 4, R_CONS, 0, None),
        ("L  D at ENHANCED range 155 m (7-el MIMO + 100 ms CPI)",        12, [45], 4, R_ENH, 0, None),
        ("M  L + cheap low ring: 12 x K-LC6+amp at 8 m, 18 m range, level", 12, [45], 4, R_ENH, 0, (12, 8, 1, 80, 12, 18, False)),
        ("N  16 roof, 4 sectors, 155 m",                                 16, [45], 4, R_ENH, 0, None),
        ("O  N + cheap low ring",                                        16, [45], 4, R_ENH, 0, (16, 8, 1, 80, 12, 18, False)),
        ("P  20 roof, 4 sectors, 155 m",                                 20, [45], 4, R_ENH, 0, None),
        ("Q  16 roof, 5 sectors (1 street + 4 drone), 155 m",            16, [45], 5, R_ENH, 0, None),
        ("R  16 roof, 4 sectors, 155 m, yaw staggered +/-15",            16, [45], 4, R_ENH, 15, None),
        ("S  20 roof, 5 sectors, 155 m",                                 20, [45], 5, R_ENH, 0, None),
    ]
    print(f"  {'config':<66}{'mods':>5}{'cost':>8}{'vol%':>6}{'street%':>8}{'CAUGHT':>8}{'warn m':>8}{'rate':>7}")
    print("  " + "-" * 110)
    rows = []
    for label, n, hs, k, R, stag, extra in configs:
        if isinstance(k, float):                     # explicit tilt (the sim's split aim)
            nodes = build(n, hs, 1, AZ, EL, R, depth, tilt_mode=k)
            kk = 1
        else:
            nodes = build(n, hs, k, AZ, EL, R, depth, yaw_stagger=stag, street=(extra != "nostreet"))
            kk = k
        mods = n * len(hs); cost = mods * MODULE + mods * MOUNT + (mods * SERVO if kk > 1 else 0)
        if isinstance(extra, tuple):
            en, eh, ek, eaz, eel, eR, est = extra
            low = build(en, [eh], ek, eaz, eel, eR, depth, tilt_mode=0.0 if ek == 1 else "spread", street=est)
            nodes += low; mods += en; cost += en * 190.0   # K-LC6 $50 + amp $32 + ADC/MCU ~$110
        vol, per = volume_coverage(nodes, X, Y, alts)
        st = street_coverage(nodes, X, Y)
        caught, warn, wmed = approach_stats(nodes, runs)
        rate = 20.0 / kk
        rows.append((label, mods, cost, vol, st, caught, warn, rate, per, nodes))
        print(f"  {label:<66}{mods:>5}{cost:>8.0f}{100*vol:>6.0f}{100*st:>8.0f}{100*caught:>7.0f}%{warn:>8.0f}{rate:>6.1f}Hz")
    print("\n  vol% = drone airspace (5-60 m) in the zone inside a beam; street% = zone ground at 1.5 m inside a beam;")
    print("  CAUGHT = inbound tracks seen at least once before the wall; warn = mean distance from the wall at first")
    print("  detection (0 for tracks never seen); rate = revisit per elevation sector; cost = hardware per node, USD.")

    print("\n  drone-airspace coverage by altitude (% of zone), A vs D vs L:")
    print("   alt m : " + " ".join(f"{a:>4d}" for a in alts))
    for lab, idx in (("A", 0), ("D", 4), ("L", 11)):
        print(f"   {lab:<5} : " + " ".join(f"{100*v:>4.0f}" for v in rows[idx][8]))

    print("\n  where the remaining misses live (uncaught / total tracks by altitude band):")
    for lab, idx in (("D", 4), ("L", 11), ("N", 13), ("P", 15), ("Q", 16), ("S", 18)):
        prof = miss_profile(rows[idx][9], runs)
        print(f"   {lab}: " + "  ".join(f"{lo:.0f}-{hi:.0f} m: {int(m)}/{int(t)}" for lo, hi, m, t in prof))

    # ------------------------------------------------------------ range to 150 m
    print("\n" + "=" * 96)
    print("GETTING TINYRAD TO 150 m ON A MINI 3 BODY (textbook 113 m at 50 ms, 4 receivers)")
    print("=" * 96)
    base = 113.0
    steps = [("baseline: 50 ms CPI, 4 Rx", 0.0),
             ("use the 7-element MIMO virtual array (2 Tx x 4 Rx) instead of 4", 10*math.log10(7/4)),
             ("CPI 50 -> 100 ms (drone walks 1.2 m in a 0.6 m cell: ~1 dB loss, net +2 dB)", 2.0),
             ("CPI 50 -> 200 ms with range-migration compensation (keystone)", 6.0),
             ("15.245 EIRP headroom used: +12 dB (needs more Tx power; PA is fixed at 8 dBm)", 12.0)]
    acc = 0.0
    for lbl, db in steps:
        acc += db; print(f"  {lbl:<82} {'+%.1f dB'%db if db else '':>9} -> {base*10**(acc/40):5.0f} m")
    print("\n  Realistic on the stock board: 7-element MIMO + 100 ms CPI = +4.4 dB -> ~146 m; with keystone at 200 ms -> ~160 m.")
    print("  150 m is reachable in software. It rests on the textbook (6 dB) loss convention that ADI's own")
    print("  100 m / 1 m^2 figure supports; under RFbeam-style derating the same chain gives ~52 m.")

    # ---------------------------------------------------- production vs Magos
    print("\n" + "=" * 96)
    print("PRODUCTION NODE (BOM) vs MAGOS, ON THE DKR FOOTPRINT  (zone 400 m, drones 5-60 m, inbound from 650 m)")
    print("=" * 96)
    depth2 = 400.0
    X2, Y2 = zone_grid(depth2, cell=10.0)
    runs2 = approaches(600, 5, 60, start_out=650.0)
    def beams(n, h, spec):
        """spec: list of (az, el, tilt_deg or 'sectorsK', Rdrone, Rcar). Rcar = 0 -> air only."""
        out = []
        for i, (x, y, nx, ny) in enumerate(wall_points(n)):
            for az, el, tilt, Rd, Rc in spec:
                tilts = sector_tilts(int(tilt[7:]), h, depth2, el) if isinstance(tilt, str) else [tilt]
                for t in tilts:
                    out.append((Node(x, y, h, nx, ny, t, az, el, Rd), Node(x, y, h, nx, ny, t, az, el, Rc) if Rc else None))
        return out
    def score(pairs):
        dn = [p[0] for p in pairs]; cn = [p[1] for p in pairs if p[1]]
        vol, _ = volume_coverage(dn, X2, Y2, alts)
        st = street_coverage(cn, X2, Y2) if cn else 0.0
        caught, warn, _ = approach_stats(dn, runs2)
        return vol, st, caught, warn
    # Mini 3 body ranges: BOM 500 m @0.01 m^2 -> x(0.02/0.01)^0.25 = 595; cone 110 -> 131; Magos AR-300 300 m @~0.03 -> 271
    FAN = (90.0, 15.0); CONE = (120.0, 120.0, 90.0, 131.0, 0.0)
    AR300 = (120.0, 120.0, 30.0, 271.0, 0.0); SR500 = (120.0, 30.0, -15.0, 100.0, 600.0)
    # Pricing. Ours: BOM cost x (1 + 110% margin). Magos publishes no list price; anchors are a
    # $4,000 used SR-500F resale (Green Wave Electronics), SpotterRF C40 $12k MSRP, Echodyne
    # ~$40k/panel. Per node-pair (AR-300 + SR-500): mid $31k, range $23k-$43k, hardware only.
    SELL = 2.10
    COST_PROTO, COST_VOL, COST_TINY = 2129.0, 686.0, 555.0
    MAGOS_PAIR = (31000.0, 23000.0, 43000.0)          # (mid, low, high) sell price, AR-300 + SR-500
    prod_cfgs = [
        ("P1 production x4, roof 45 m, fan FIXED +10 (as BOM), cone",           4, 45, [(*FAN, 10.0, 595.0, 2800.0), CONE], "ours"),
        ("P2 production x4, roof, fan fixed -8 (aimed at the drone band)",      4, 45, [(*FAN, -8.0, 595.0, 2800.0), CONE], "ours"),
        ("P3 production x4, roof, fan nodding 3 sectors, cone",                 4, 45, [(*FAN, "sectors3", 595.0, 2800.0), CONE], "ours"),
        ("P4 production x8, roof, fan nodding 3 sectors, cone",                 8, 45, [(*FAN, "sectors3", 595.0, 2800.0), CONE], "ours"),
        ("P5 production x4, LOW mount 8 m, fan +10 (BOM's intended geometry)",  4,  8, [(*FAN, 10.0, 595.0, 2800.0), CONE], "ours"),
        ("P6 production x4 low 8 m, fan +4",                                    4,  8, [(*FAN, 4.0, 595.0, 2800.0), CONE], "ours"),
        ("P7 production x8 low 8 m, fan nodding 3 sectors",                     8,  8, [(*FAN, "sectors3", 595.0, 2800.0), CONE], "ours"),
        ("M1 Magos AR-300 x4 + SR-500 x4, roof (as Magos recommends)",         4, 45, [AR300, SR500], "magos"),
        ("M2 Magos AR-300 x8 + SR-500 x8, roof",                                8, 45, [AR300, SR500], "magos"),
        ("M3 Magos AR-300 x4 + SR-500 x4, low 8 m",                             4,  8, [AR300, SR500], "magos"),
        ("T  TinyRad x20, roof, 4 sectors, 155 m (for scale, zone 400)",       20, 45, [(75.0, 15.0, "sectors4", 155.0, 155.0 * CAR_SCALE_LOCAL)], "tiny"),
    ]
    print(f"  sell price = cost x {SELL:.2f} (110% margin). Ours: proto ${COST_PROTO:,.0f} -> ${COST_PROTO*SELL:,.0f}/node, "
          f"volume ${COST_VOL:,.0f} -> ${COST_VOL*SELL:,.0f}/node. Magos pair est. ${MAGOS_PAIR[0]:,.0f} (${MAGOS_PAIR[1]:,.0f}-${MAGOS_PAIR[2]:,.0f}).")
    print(f"  {'config':<64}{'nodes':>6}{'CAUGHT':>8}{'warn m':>8}{'site $ proto':>14}{'site $ volume':>15}{'site $ range':>20}")
    print("  " + "-" * 135)
    for label, n, h, spec, who in prod_cfgs:
        vol, st, caught, warn = score(beams(n, h, spec))
        if who == "ours":
            p1, p2, rng = f"${n*COST_PROTO*SELL:,.0f}", f"${n*COST_VOL*SELL:,.0f}", ""
        elif who == "tiny":
            p1, p2, rng = f"${n*COST_TINY*SELL:,.0f}", "", ""
        else:
            p1, p2, rng = f"${n*MAGOS_PAIR[0]:,.0f}", "", f"${n*MAGOS_PAIR[1]:,.0f}-{n*MAGOS_PAIR[2]:,.0f}"
        print(f"  {label:<64}{n:>6}{100*caught:>7.0f}%{warn:>8.0f}{p1:>14}{p2:>15}{rng:>20}")
    # break-even: what Magos would have to charge per pair to match our 8-node site price
    for tag, c in (("proto", COST_PROTO), ("volume", COST_VOL)):
        print(f"  Magos would have to sell an AR-300 + SR-500 pair for ${8*c*SELL/8:,.0f} to match our 8-node {tag} site price "
              f"(vs ~$4,000 for a USED SR-500 alone).")
    print("\n  Production fan fixed at +10 deg from a roof sees altitudes 49-77 m at 100 m out and nothing lower; the cone")
    print("  sees only what is nearly overhead. The BOM's 500 m only buys coverage if the fan is aimed where drones are.")
    print("  Magos AR-300's 120 deg elevation fan is the reason it catches everything inside its (shorter) range.")

    # -------------------------------------------------------------- Pd math
    print("\n" + "=" * 96)
    print("WHY 'INSIDE THE BEAM' IS THE WHOLE GAME (cumulative detection over an approach)")
    print("=" * 96)
    for pd in (0.3, 0.5, 0.8):
        looks_1s = 20 / 3      # 3 sectors, 20 Hz frames
        t_in = (113 - 6) / 12.0
        looks = looks_1s * t_in
        print(f"  single-look Pd {pd:.1f}: a 12 m/s drone spends {t_in:.0f} s inside 113 m, gets ~{looks:.0f} looks per sector")
        print(f"      -> cumulative Pd = 1 - (1-{pd})^{looks:.0f} = {1-(1-pd)**looks:.6f}")
    print("  Any drone that is geometrically inside a sector for even ~2 s is detected with near certainty.")
    print("  Every miss in the sim is a drone that was never inside any sector. Coverage, not SNR.")

    # ----------------------------------------------------------------- rain
    print("\n" + "=" * 96)
    print("RAIN: 24 GHz vs 77 GHz  (ITU-R P.838, two-way path)")
    print("=" * 96)
    coef = {"24 GHz": (0.124, 1.061), "77 GHz": (0.55, 0.93)}
    print(f"  {'rate':>10} {'':>3} " + "".join(f"{f+' @150 m':>16}{f+' @500 m':>16}" for f in coef))
    for rate, name in ((5, "light"), (25, "heavy"), (50, "violent")):
        row = f"  {rate:>4d} mm/h {name:>8} "
        for f, (k, a) in coef.items():
            g = k * rate ** a
            row += f"{2*0.150*g:>13.2f} dB{2*0.5*g:>13.2f} dB"
        print(row)
    print("\n  At 150 m, 77 GHz loses ~2-3 dB more than 24 GHz in a downpour -- real, not fatal. At 500 m it is")
    print("  ~6 dB more (range x0.7 in the worst storm), plus rain backscatter as clutter and a wet radome.")
    print("  Your concern is right at 500 m; at 150 m it is not the deciding factor.")

    # -------------------------------------------------------------- licence
    print("\n" + "=" * 96)
    print("LICENSING COST, HONESTLY")
    print("=" * 96)
    print("""  Part 90 radiolocation (24.05-24.25 GHz): the FCC application fee and frequency coordination are on
  the order of hundreds to low thousands of dollars per site, renewable on a 10-year term -- cheap next
  to the hardware. The expensive part is that the EQUIPMENT must be certified for Part 90. But a custom
  board sold as a product needs certification under Part 15 too. The marginal cost of the licensed
  path is therefore mostly coordination and paperwork, not a second certification. Verify current fee
  schedules with a compliance consultant; this is an engineering read, not legal advice.

  The cheaper unlicensed lever stays: 15.245 (24.075-24.175 GHz, +32.7 dBm EIRP) vs 15.249 (+12.7 dBm).
  That 20 dB is worth 3.2x range and costs a 100 MHz sweep instead of 250 (1.5 m cells vs 0.6 m).""")

if __name__ == "__main__":
    main()
