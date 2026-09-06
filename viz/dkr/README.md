# DKR perimeter coverage model

Interactive 3D model of radar node coverage around Darrell K Royal–Texas
Memorial Stadium (exterior perimeter, nodes on the roof rim facing out), plus
the surveyed map data it runs on. Companion to `docs/COVERAGE_OPS.md` and
`scripts/coverage_ops.py`, which score the same beam geometry offline.

| file | what |
|---|---|
| `dkr_perimeter.html` | model source; `const MAP=/*MAPDATA*/null;` is the injection point |
| `dkr_perimeter_built.html` | built page with the map inlined — open directly in a browser |
| `build_perimeter.py` | inlines `dkr_map.json` into the source → `_built.html` |
| `dkr_map.json` | stadium rings, 159 buildings (footprint + height), roads, paths, fences, parking, grass — local metres |
| `dkr_osm.json` | raw OpenStreetMap Overpass export (relation 1639372 and surroundings) |
| `osm_prep.py` | `dkr_osm.json` → `dkr_map.json` (`python osm_prep.py 1639372 relation`) |
| `dbf_elev.py` | servo vs elevation-DBF comparison for the production fan (§9 of COVERAGE_OPS) |
| `dkr_coverage.html` | earlier interior-coverage render, kept for reference |

Sensors in the model: TinyRad (4-ch), K-LC6 with/without IF amp, custom,
FCC-capped receive array, the production two-tier node (BOM), and Magos
AR-300 + SR-500. Controls: node count, mount height, nodding elevation
sectors, zone depth, fan tilt / elevation FOV (DBF), building line-of-sight
occlusion, live traffic, drop-in drones and swarms with altitude.

Rebuild after editing the source:

    python viz/dkr/build_perimeter.py

Map data © OpenStreetMap contributors, ODbL. Three.js r128 from cdnjs.
