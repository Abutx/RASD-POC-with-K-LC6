"""OpenStreetMap -> compact local-metre JSON for the DKR perimeter render.

Projection: equirectangular about the stadium centroid (error < 0.1 m over
1 km). x = east, y = north, metres. The HTML maps y -> -z (Three.js z south).
"""
import json, math, sys, re
from pathlib import Path

SRC = Path(__file__).with_name("dkr_osm.json")
OUT = SRC.with_name("dkr_map.json")
d = json.load(open(SRC, encoding="utf-8"))
els = d["elements"]
STADIUM_ID = int(sys.argv[1]) if len(sys.argv) > 1 else None
STADIUM_TYPE = sys.argv[2] if len(sys.argv) > 2 else "way"

def pts_of(e):
    if e["type"] == "way":
        return [(p["lon"], p["lat"]) for p in e.get("geometry", [])]
    return []

def rel_outer_rings(e):
    """Stitch outer member ways of a multipolygon into closed rings."""
    segs = [[(p["lon"], p["lat"]) for p in m.get("geometry", [])]
            for m in e.get("members", []) if m.get("role") in ("outer", "") and m.get("geometry")]
    rings = []
    while segs:
        ring = segs.pop(0)
        changed = True
        while changed and ring[0] != ring[-1]:
            changed = False
            for i, s in enumerate(segs):
                if s[0] == ring[-1]: ring += s[1:]; segs.pop(i); changed = True; break
                if s[-1] == ring[-1]: ring += s[-2::-1]; segs.pop(i); changed = True; break
                if s[-1] == ring[0]: ring = s[:-1] + ring; segs.pop(i); changed = True; break
                if s[0] == ring[0]: ring = s[::-1][:-1] + ring; segs.pop(i); changed = True; break
        rings.append(ring)
    return rings

# ---- find the stadium ----
stad = None
if STADIUM_ID is not None:
    stad = next(e for e in els if e["id"] == STADIUM_ID and e["type"] == STADIUM_TYPE)
else:
    cands = [e for e in els if e.get("tags", {}).get("building") == "stadium" or e.get("tags", {}).get("leisure") == "stadium"]
    named = [e for e in cands if re.search(r"Royal|Memorial Stadium|DKR", e.get("tags", {}).get("name", ""))]
    pool = named or cands
    def area(e):
        rings = [pts_of(e)] if e["type"] == "way" else rel_outer_rings(e)
        a = 0
        for r in rings:
            for i in range(len(r) - 1):
                a += r[i][0] * r[i + 1][1] - r[i + 1][0] * r[i][1]
        return abs(a)
    stad = max(pool, key=area)
stad_rings = [pts_of(stad)] if stad["type"] == "way" else rel_outer_rings(stad)
allp = [p for r in stad_rings for p in r]
lon0 = sum(p[0] for p in allp) / len(allp); lat0 = sum(p[1] for p in allp) / len(allp)
R = 6371008.8
kx = math.cos(math.radians(lat0)) * math.pi / 180 * R
ky = math.pi / 180 * R
def proj(p): return (round((p[0] - lon0) * kx, 2), round((p[1] - lat0) * ky, 2))
def projring(r): return [proj(p) for p in r]

print(f"stadium: {stad['type']} {stad['id']} {stad.get('tags',{}).get('name')}  rings {len(stad_rings)}  centre {lat0:.6f}, {lon0:.6f}")

def height_of(t):
    if "height" in t:
        m = re.match(r"([\d.]+)", t["height"]);
        if m: return float(m.group(1))
    if "building:levels" in t:
        try: return float(t["building:levels"]) * 3.5 + 1.0
        except ValueError: pass
    b = t.get("building", "yes")
    return {"garage": 14.0, "dormitory": 20.0, "university": 16.0, "stadium": 40.0, "roof": 4.0, "shelter": 3.0}.get(b, 8.0)

out = {"centre": {"lat": lat0, "lon": lon0}, "stadium": [projring(r) for r in stad_rings],
       "stadium_name": stad.get("tags", {}).get("name", "stadium"),
       "buildings": [], "roads": [], "paths": [], "fences": [], "parking": [], "grass": [], "pitches": []}
for e in els:
    t = e.get("tags", {})
    if e is stad: continue
    if e["type"] == "relation":
        if t.get("building"):
            for r in rel_outer_rings(e):
                if len(r) >= 4: out["buildings"].append({"n": t.get("name", ""), "h": height_of(t), "p": projring(r)})
        continue
    pts = pts_of(e)
    if len(pts) < 2: continue
    P = projring(pts)
    if t.get("building"):
        if len(P) >= 4: out["buildings"].append({"n": t.get("name", ""), "h": height_of(t), "p": P})
    elif "highway" in t:
        hw = t["highway"]
        rec = {"n": t.get("name", ""), "c": hw, "p": P}
        if hw in ("motorway", "motorway_link", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "service", "living_street"):
            out["roads"].append(rec)
        elif hw in ("footway", "pedestrian", "cycleway", "path", "steps"):
            out["paths"].append(rec)
    elif t.get("barrier") in ("fence", "wall", "retaining_wall", "guard_rail"):
        out["fences"].append({"c": t["barrier"], "p": P})
    elif t.get("amenity") == "parking":
        out["parking"].append(P)
    elif t.get("landuse") == "grass" or t.get("leisure") in ("park", "garden"):
        out["grass"].append(P)
    elif t.get("leisure") in ("pitch", "track"):
        out["pitches"].append({"c": t["leisure"], "p": P})

def perim(r):
    return sum(math.dist(r[i], r[i + 1]) for i in range(len(r) - 1))
xs = [p[0] for r in out["stadium"] for p in r]; ys = [p[1] for r in out["stadium"] for p in r]
print(f"stadium footprint: {max(xs)-min(xs):.0f} m E-W x {max(ys)-min(ys):.0f} m N-S, perimeter {sum(perim(r) for r in out['stadium']):.0f} m, {sum(len(r) for r in out['stadium'])} vertices")
print("counts:", {k: len(v) for k, v in out.items() if isinstance(v, list)})
names = sorted({r["n"] for r in out["roads"] if r["n"]})
print("roads named:", len(names), names[:30])
s = json.dumps(out, separators=(",", ":"))
OUT.write_text(s, encoding="utf-8")
print(f"wrote {OUT.name}: {len(s)/1024:.0f} KB")
