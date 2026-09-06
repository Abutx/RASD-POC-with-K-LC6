# Catching Every Drone — Coverage Operating Concept for DKR

**Question:** the perimeter sim misses most drones. How do we get to *every*
drone at ≥150 m with the TinyRad test platform, without losing range or
inflating cost — and without Part 90 or 77 GHz?

**Reproduce with `python scripts/coverage_ops.py`.** The study uses the
surveyed DKR footprint (OSM relation 1639372, 880 m perimeter) and the same
beam model as the live sim, scoring each operating concept on the metric a
perimeter system is judged by: of 600 drones flying in from 320 m out at
12 m/s and 5–60 m altitude, what fraction is inside *some* node's beam before
it reaches the wall, and how far out.

---

## 1. VERDICT

**The misses are geometric, not sensitivity. Range was never the problem.**

From a 45 m roof, a drone at 60 m altitude 100 m out is at +8.5° elevation;
the street below it is at −24°. One TinyRad sheet is 15° tall. **A single
fixed sheet covers under half the elevations a perimeter needs**, and near
the wall the demand exceeds 60°. Every missed drone flew over or under the
sheet.

| operating concept | modules | hardware | **caught** | warning | revisit |
|---|---|---|---|---|---|
| **A · sim as shipped** — 12 roof, 1 sheet | 12 | $6,240 | **53%** | 38 m | 20 Hz |
| F · 24 roof, 1 sheet | 24 | $12,480 | 40% | 42 m | 20 Hz |
| D · 12 roof, **nodding 4 sectors** | 12 | $6,660 | 90% | 85 m | 5 Hz |
| **L · D + software range to 155 m** | 12 | **$6,660** | **94%** | **134 m** | 5 Hz |
| N · 16 roof, 4 sectors, 155 m | 16 | $8,880 | 98% | 141 m | 5 Hz |
| **P · 20 roof, 4 sectors, 155 m** | 20 | **$11,100** | **100%** | **144 m** | 5 Hz |

Hardware = TinyRad $400 + mount $120 + nodding servo $35 per node.

Three findings that decide how to spend:

1. **Doubling the nodes does nothing.** 24 single-sheet nodes catch *fewer*
   drones (40%) than 12. Every added node has the same blind elevation.
2. **A $35 servo per node nearly doubles the catch rate.** Nodding the module
   between four tilt positions — one down at the street, three across the
   drone band — takes 12 nodes from 53% to 90% at the same range.
3. **The rest is software and node density.** Using the 7-element MIMO array
   and a 100 ms CPI lifts range to ~155 m for free (94%, 134 m warning);
   going from 16 to 20 nodes closes the last azimuth gaps (100%).

---

## 2. Why the sim misses — elevation demand vs a 15° sheet

| radar mount | street 40 m out | street at 150 m | drone 60 m @100 m | drone 60 m @40 m | elevation span needed |
|---|---|---|---|---|---|
| 45 m (roof) | −47° | −16° | +8.5° | +21° | **~37°** |
| 25 m (mid) | −30° | −9° | +19° | +41° | ~50° |
| 12 m (wall) | −15° | −4° | +26° | +50° | ~54° |

Drone-airspace coverage by altitude, % of the 150 m zone:

| alt (m) | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 | 55 | 60 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A · one sheet, split aim | 26 | 30 | 27 | 22 | 15 | 9 | 4 | 1 | 0 | 0 | 0 | 0 |
| D · 4 sectors | 20 | 16 | 12 | 19 | 27 | 35 | 40 | 43 | 44 | 44 | 41 | 38 |
| L · 4 sectors, 155 m | 20 | 31 | 42 | 51 | 58 | 65 | 69 | 73 | 74 | 73 | 71 | 67 |

The shipped sim sees **nothing above 40 m**. That is the whole story.

---

## 3. The operating concept: nod, don't add

**Nodding mount.** A servo cycles each module through K tilt positions. Each
dwell is a full coherent frame, so **per-look range is unchanged** — the cost
is revisit rate, 20 Hz / K. At K = 4 that is 5 Hz per sector; a 12 m/s drone
moves 2.4 m between looks at the same sector. Fine for detection; tracking
sees the target in ~0.2 s bursts every 0.8 s, which is enough for a perimeter
alarm and thinner than ideal for kinematic classification.

**Sector allocation** (from height H over a zone of depth Z):
- one sector down at the street 60 m out — covers roughly 35–150 m of road
- K − 1 sectors spread from "5 m drone at the zone edge" to "60 m drone 40 m out"

Removing the street sector (D2) *lowers* drone catch to 68% — the down-look
also catches low drones. Keep it.

**Why cumulative detection makes "inside the beam" the whole game.** A drone
at 12 m/s spends ~9 s inside 113 m and gets ~59 looks per sector. Even at a
poor single-look Pd of 0.3, cumulative Pd = 1 − 0.7⁵⁹ ≈ 1.000000. Any drone
that is geometrically inside a sector for two seconds is caught. Every miss is
a drone that was never inside any sector.

---

## 4. Full results

| # | concept | modules | $ | vol% | street% | caught | warn m |
|---|---|---|---|---|---|---|---|
| A | 12 roof, 1 sheet, split aim (−26°) — **the sim as shipped** | 12 | 6,240 | 11 | 22 | **53%** | 38 |
| A2 | 12 roof, 1 sheet aimed at the drone band | 12 | 6,240 | 14 | 0 | 37% | 38 |
| B | 12 roof, 2 sectors | 12 | 6,660 | 20 | 23 | 82% | 60 |
| C | 12 roof, 3 sectors | 12 | 6,660 | 29 | 23 | 88% | 79 |
| D | 12 roof, 4 sectors | 12 | 6,660 | 32 | 23 | 90% | 85 |
| D2 | 12 roof, 4 sectors, no street look | 12 | 6,660 | 26 | 0 | 68% | 71 |
| E | 16 roof, 4 sectors | 16 | 8,880 | 37 | 26 | 98% | 92 |
| F | 24 roof, 1 sheet | 24 | 12,480 | 18 | 0 | 40% | 42 |
| G | two rings 45 + 20 m, 12 each, 2 sectors | 24 | 13,320 | 25 | 39 | 86% | 66 |
| I | 12 at mid-height 25 m, 4 sectors | 12 | 6,660 | 36 | 31 | 91% | 91 |
| K | D at **conservative** 38 m range | 12 | 6,660 | 2 | 0 | 31% | 9 |
| **L** | **D at 155 m (software)** | 12 | **6,660** | 58 | 23 | **94%** | **134** |
| M | L + cheap K-LC6 low ring | 24 | 8,940 | 58 | 23 | 94% | 134 |
| N | 16 roof, 4 sectors, 155 m | 16 | 8,880 | 67 | 27 | 98% | 141 |
| O | N + cheap low ring | 32 | 11,920 | 67 | 27 | 98% | 141 |
| **P** | **20 roof, 4 sectors, 155 m** | 20 | **11,100** | 70 | 29 | **100%** | **144** |
| Q | 16 roof, 5 sectors, 155 m | 16 | 8,880 | 67 | 27 | 98% | 141 |
| R | 16 roof, 4 sectors, 155 m, yaw staggered ±15° | 16 | 8,880 | 65 | 27 | 99% | 140 |
| S | 20 roof, 5 sectors, 155 m | 20 | 11,100 | 70 | 29 | 100% | 144 |

vol% = drone airspace (5–60 m) inside a beam; street% = zone ground at 1.5 m inside a beam;
caught = inbound tracks seen before the wall; warn = mean distance from the wall at first detection.

**Where the last misses live** (uncaught / total by altitude):

| | 5–15 m | 15–25 | 25–35 | 35–45 | 45–60 |
|---|---|---|---|---|---|
| D (12 nodes) | 9/110 | 15/128 | 10/102 | 11/93 | 17/167 |
| L (12, 155 m) | 9/110 | 4/128 | 8/102 | 7/93 | 10/167 |
| N (16, 155 m) | 0/110 | 0/128 | 3/102 | 4/93 | 3/167 |
| **P (20, 155 m)** | **0** | **0** | **0** | **0** | **0** |

At 16 nodes the low band is clean; the ten remaining misses are mid/high
tracks arriving **between nodes** close to the wall. A fifth elevation sector
(Q) does not touch them; four more nodes (P) or yaw-staggering (R, 99%) do.
**Node spacing ≈ 44 m on the 880 m perimeter is the closing condition.**

**The cheap low ring adds nothing** (M = L, O = N). A K-LC6 at 18 m range and
12° elevation, mounted at 8 m, never sees a drone at 12 m altitude 15 m out.
Save the money.

---

## 5. Range: 150 m from TinyRad is software

Textbook Mini 3 body range is 113 m at 50 ms CPI on 4 receivers. Cumulative:

| step | gain | range |
|---|---|---|
| baseline | — | 113 m |
| use the **7-element MIMO virtual array** (2 Tx × 4 Rx) | +2.4 dB | 130 m |
| **CPI 50 → 100 ms** (drone walks 1.2 m in a 0.6 m cell, ~1 dB loss) | +2.0 dB | **146 m** |
| CPI 200 ms with range-migration (keystone) compensation | +6.0 dB | 206 m |
| spend 15.245 EIRP headroom (+12 dB; needs more PA than the board has) | +12 dB | 411 m |

**MIMO + 100 ms gets ~146–155 m on the stock board with no hardware change.**
It rests on the textbook loss convention, which ADI's own 1 m² / 100 m figure
supports (`TINYRAD_BENCH.md` §2). Under RFbeam-style derating the same chain
gives ~52 m — and row K shows what that would mean: 31% caught. **Confirming
the range convention on the bench is the first thing to do with the board.**

---

## 6. Rain, 24 vs 77 GHz — the numbers

ITU-R P.838, two-way path:

| rain | 24 GHz @150 m | 24 GHz @500 m | 77 GHz @150 m | 77 GHz @500 m |
|---|---|---|---|---|
| 5 mm/h light | 0.2 dB | 0.7 dB | 0.7 dB | 2.5 dB |
| 25 mm/h heavy | 1.1 dB | 3.8 dB | 3.3 dB | **11.0 dB** |
| 50 mm/h violent | 2.4 dB | 7.9 dB | 6.3 dB | **20.9 dB** |

Your instinct is right for the **500 m product**: 77 GHz gives up 7–13 dB more
than 24 GHz in a storm — range × 0.5–0.7 — plus rain backscatter as clutter
and a wet radome. At **150 m** the penalty is 2–4 dB: real, not decisive.
Staying at 24 GHz is the right call for the 500 m goal; it is not forced at
150 m.

---

## 7. Licensing cost, honestly

Part 90 radiolocation (24.05–24.25 GHz): FCC application fee plus frequency
coordination run on the order of **hundreds to low thousands of dollars per
site**, on a ten-year term — small next to the hardware. The expensive part is
equipment certification, **but a product sold under Part 15 needs
certification too.** The marginal cost of the licensed path is paperwork and
coordination, not a second certification. Verify current fees with a
compliance consultant; this is an engineering read, not legal advice.

The unlicensed lever remains **15.245** (24.075–24.175 GHz, +32.7 dBm EIRP)
over 15.249 (+12.7 dBm): 20 dB = 3.2× range for a 100 MHz sweep (1.5 m cells).

---

## 8. Recommendation

**For the TinyRad test system, in order:**

1. **Nodding mounts on every node, 4 sectors** (1 street + 3 drone). ~$35
   each. This alone takes 53% → 90%.
2. **Software range: 7-element MIMO processing + 100 ms CPI.** Free. → 94%,
   warning distance 134 m. **Verify the textbook range convention on the bench
   first** — row K is what happens if it does not hold.
3. **20 nodes on the roof rim (44 m spacing).** → 100% of 600 tracks, 144 m
   mean warning. $11.1k in modules, mounts and servos.
4. Do **not** buy a low ring of cheap nodes, and do not add nodes without
   adding elevation sectors.

**For the final board**, the same study says what to design for: an elevation
FOV of ≥40° (electronically, so nothing nods and revisit stays at frame rate),
receive aperture rather than transmit gain (EIRP is capped whatever you do),
and 15.245 or a site licence to spend the EIRP headroom. Those three choices,
not a different band, are what carry the 150 m test system to a 500 m product.

---

## 9. Production node (BOM) vs Magos on the DKR footprint

Inputs: `production_node_bom.pdf` (two-tier 24 GHz node) and Magos' published
AR-300 / SR-series datasheets. Same footprint, same 600 inbound DJI Mini 3
tracks (5–60 m altitude, from 650 m out), zone widened to 400 m so the longer
sensors have room. Both are selectable in the 3D model (sensor menu →
"Production node" / "Magos"); each node carries two beams.

**What the BOM claims, and whether the radar equation agrees**

| tier | beam | claim | check |
|---|---|---|---|
| 1 · 16-ch fan, PA, §15.245 +32.7 dBm | 90° × 15°, +10° uptilt | 500 m crossing / 750 m hover @0.01 m², 660 m @0.03 m², 1.5 m res, ~1 Hz | reproduced at ~13 dBi per subarray with 100–500 ms CPI; Mini 3 body (0.02 m²) → **~600 m** |
| 2 · 4-ch overhead, 9.5 dBi | 120° × 120°, straight up | 110 m | matches no-PA, 50 ms CPI; Mini 3 → **~130 m** |

The link budgets hold. The geometry does not: a 15° fan fixed at +10° from
a 45 m roof covers elevations +2.5° … +17.5°, i.e. **49–77 m altitude at
100 m out, and nothing below the rim** — the same elevation blindness as the
single-sheet TinyRad, now with a longer beam that points over every drone in
the 5–60 m band. The cone only fills the last ~130 m overhead. The BOM's
"4 nodes for 360°" also leaves the corners of a ~200 × 250 m footprint with
one node each and 90° between neighbours.

**Magos, from published figures only** (no public pricing):

| unit | band | FOV | ranges | notes |
|---|---|---|---|---|
| AR-300 | X-band | 120° × 120° | >300 m DJI Mavic (≈0.03 m²) → **~270 m** Mini 3; air only | 7 Hz, 35 W, 0.1 m res; Magos recommends four + PTZ for 360° |
| SR-500 | 5.8 GHz | 120° × 30° | vehicle 600 m, person 400 m, **drone ~100 m** | 8 scans/s, 11 W; SR-1000 800/1000 m |

**Results** (`scripts/coverage_ops.py`, section "PRODUCTION NODE vs MAGOS"):

| config | nodes | cost/node | drone vol. | street | caught | mean warn |
|---|---|---|---|---|---|---|
| P1 production ×4, roof, fan fixed +10° **(as BOM)** | 4 | 2,129 | 12% | 0% | **25%** | 81 m |
| P2 production ×4, roof, fan fixed −8° | 4 | 2,129 | 47% | 57% | 71% | 372 m |
| P3 production ×4, roof, fan nodding 3 sectors | 4 | 2,129 | 52% | 6% | 82% | 471 m |
| P4 production ×8, roof, fan nodding 3 sectors | 8 | 2,129 | 68% | 15% | **97%** | **538 m** |
| P5 production ×4, low mount 8 m, fan +10° | 4 | 2,129 | 53% | 0% | 83% | 390 m |
| P6 production ×4, low 8 m, fan +4° | 4 | 2,129 | 61% | 63% | 86% | 473 m |
| P7 production ×8, low 8 m, fan nodding 3 sectors | 8 | 2,129 | 83% | 84% | **100%** | **541 m** |
| M1 Magos AR-300 ×4 + SR-500 ×4, roof (as Magos recommends) | 4+4 | n/p | 33% | 87% | 85% | 210 m |
| M2 Magos AR-300 ×8 + SR-500 ×8, roof | 8+8 | n/p | 45% | 92% | 98% | 249 m |
| M3 Magos AR-300 ×4 + SR-500 ×4, low 8 m | 4+4 | n/p | 37% | 89% | 90% | 220 m |
| T  TinyRad ×20, roof, 4 sectors, 155 m (§4 winner, 400 m zone) | 20 | 555 | 11% | 15% | 99% | 103 m |

cost/node is the BOM's own prototype figure ($686 at volume); sell prices
are below. "vol." is the fraction of the 400 m × 5–60 m zone volume inside
at least one beam. **Free-space numbers** — with the 159 surrounding
buildings blocking line of sight (§10) the roof rows drop 1–6 points
(P4 97→94%, M2 98→96%) and the **low-mount rows collapse** (P5 83→47%,
P6 86→48%, P7 100→86%, M3 90→55%).

**Reading it**

1. **Built as drawn, the production node loses to Magos and to the TinyRad
   test rig** — 25% caught at 81 m warning, because the fan is aimed above the
   drones. Nothing about the RF is wrong; the +10° is.
2. **Aimed properly it beats Magos by 2× on warning distance.** Nodding the
   fan (P3/P4) gives 400–540 m mean warning against Magos' 190–250 m. That
   is the BOM's 500 m range actually reaching drones, and it is the reason to
   build at 24 GHz with §15.245's +32.7 dBm rather than buy Magos' 5.8 GHz/
   X-band pair. Mounting low with a shallow tilt (P6/P7) looks as good in
   free space but not once buildings are in the way — see §10.
3. **Magos' strength is elevation, not range.** AR-300's 120° × 120° cone
   sees everything inside 270 m regardless of tilt or mount height, which is
   why M1 catches 85% with four units and never needs a nodding mount. The
   production node gets the same effect only from the 130 m cone, and only
   overhead.
4. **Street coverage is a separate beam.** The fan cannot look at the street
   and the drone band at once from a roof; Magos pairs a ground radar for
   exactly this. P6/P7 (low mount) or P2 (fan aimed down at −8°) get both from
   one node; the roof-mounted nodding configs sacrifice street for air.
5. **Four nodes is not enough on this footprint whoever makes them** — the
   best 4-node case is 90% (M3). Eight is the number that reaches 97–100%
   for either vendor.

**Price, at 110% margin on each node**

Our sell price = BOM cost × 2.10. Magos has no public list price anywhere
(their own site, distributors, GSA, IPVM's report — which redacts its
figures — all say "request a quote"). The anchors that do exist:

- a **used SR-500F resells for $4,000** (Green Wave Electronics; ibdglobal
  lists it "login to see price") → new SR-500 ≈ $5–8k, we use $6k
- drone-radar tier: SpotterRF C40 was $12k MSRP (2012); Echodyne's small
  panels are ~$40k, EchoGuard $50k+ → AR-300 ≈ $18–35k, we use $25k
- per AR-300 + SR-500 pair: **~$31k, range $23–43k**, hardware only (Magos'
  MASS software licence, PTZ and integration are on top)

| | per node | 4 nodes | 8 nodes (97–100% catch) |
|---|---|---|---|
| Production node, prototype cost $2,129 | **$4,470** | $17.9k | **$35.8k** |
| Production node, volume cost $686 | **$1,440** | $5.8k | **$11.5k** |
| TinyRad rig ($555 cost), 20 nodes | $1,166 | — | $23.3k (20 nodes, 99%) |
| Magos AR-300 + SR-500 pair (est.) | ~$31k ($23–43k) | ~$124k ($92–172k) | **~$248k ($184–344k)** |

At equal catch rate (8 nodes each, 97% vs 98%) the production node at
110% margin is **~7× cheaper than Magos at prototype cost and ~20× at
volume**, with 2× the warning distance. Break-even: Magos would have to
sell an AR-300 + SR-500 pair for $4,470 (proto) or $1,440 (volume) — below
what a used SR-500 alone fetches. The margin could go well past 110% and
still undercut them; the number that matters commercially is the 8-node site
price, and $36k for a stadium is a rounding error next to a $250k Magos
quote. These Magos figures are estimates — get a real quote before putting
them in front of a customer.

**Servo vs beamforming in elevation.** The BOM's Tier 1 is "8 vertical
subarrays × 16 az channels": each channel is an 8-patch column combined in
analogue, so the array beamforms in azimuth only and the 15° elevation is
fixed by the column. To steer or multi-beam in elevation electronically the
columns have to be split into digitised sub-columns — 2 per column (32 ch)
gives 30° of elevation, 3 (48 ch) 45°, 4 (64 ch) 60° — and every elevation
beam then exists at once, at full aperture gain, 20 Hz, no moving parts.
Same footprint, 8 roof nodes, buildings on (`viz/dkr/dbf_elev.py`):

| Tier 1 elevation | ch | drone vol. | street | caught | mean warn | revisit/sector |
|---|---|---|---|---|---|---|
| BOM: 15° fixed +10° | 16 | 17% | 0% | 29% | 105 m | 20 Hz |
| servo, 15° × 3 sectors | 16 | 58% | 5% | 96% | 493 m | 6.7 Hz |
| servo, 15° × 4 sectors | 16 | 58% | 5% | 96% | 493 m | 5 Hz |
| DBF 30°, fixed +7° | 32 | 62% | 13% | 91% | 493 m | 20 Hz |
| DBF 45°, fixed +7° | 48 | 66% | 23% | 92% | 496 m | 20 Hz |
| **DBF 60°, fixed +7°** | **64** | **67%** | **27%** | **94%** | **497 m** | **20 Hz** |
| DBF 30° + 2-position servo | 32 | 64% | 19% | 96% | 497 m | 10 Hz |

Catch rates within ±2 points are the same (600 tracks). What DBF buys is
volume coverage (67% vs 58%), street coverage, and a 20 Hz revisit on every
elevation — the servo's 4–7 Hz per sector is what lets a 12 m/s drone move
2–3 m between looks. What it costs: 3× the receive silicon (an ADF5904 +
ADAR7251 pair per 4 channels; roughly +$500–700 per node at volume, so
~$1,200–1,400 instead of $686), 64 × 4 MSPS × 16 bit ≈ 4 Gb/s into the
ZU3EG, and the TX array must illuminate 60° instead of 15° — its gain drops
~6 dB (20 → 14 dBi), so the PA has to make it up: the BOM's open item "TX
PA, P1dB ≥ +17 dBm" becomes **P1dB ≥ +21 dBm** to hold +26 dBm EIRP, and
that is still 6.7 dB under the §15.245 ceiling. Range is preserved because
the receive aperture is unchanged and EIRP is unchanged; under an EIRP cap,
spreading the transmit beam costs PA power, not range.

**Changes to the BOM this implies**: make the fan tilt a parameter (nodding
mount, or an electronically steered elevation with ≥3 states) rather than a
fixed +10°; plan eight nodes for a stadium of this size, not four; and keep
the nodes on the roof — fence-line masts lose a third of the tracks to the
buildings across the street (§10). None of this changes the $686 volume
cost by more than the mount.

---

## 10. Buildings block line of sight — and what else the sim leaves out

The results above were "too good" for one obvious reason: through §9 the
sim was free space. The 159 buildings around DKR (OSM footprints and
heights, 1–50 m, mean 14 m) were drawn but not in the physics. They are
now, in both `coverage_ops.py` and the 3D model (checkbox "Buildings block
line of sight", on by default): a building is an opaque prism, and a target
is lost when the straight line from the radar passes below a roof anywhere
inside a footprint. Same 600 tracks, free space vs occluded:

| config | caught, free | caught, buildings | warn, free → occl. |
|---|---|---|---|
| A  12 roof, 1 sheet (as shipped) | 53% | 49% | 38 → 33 m |
| D  12 roof, 4 sectors | 90% | 88% | 85 → 82 m |
| L  D at 155 m | 94% | 92% | 134 → 126 m |
| N  16 roof, 4 sectors, 155 m | 98% | 98% | 141 → 133 m |
| **P  20 roof, 4 sectors, 155 m** | 100% | **100%** | 144 → 135 m |
| G  two rings 45 m + 20 m | 86% | 85% | — |
| P1 production ×4, fan +10° as BOM | 25% | 24% | 81 → 80 m |
| P4 production ×8, roof, nodding | 97% | **94%** | 538 → 473 m |
| **P5 production ×4, low 8 m, +10°** | 83% | **47%** | 390 → 86 m |
| **P6 production ×4, low 8 m, +4°** | 86% | **48%** | 473 → 91 m |
| **P7 production ×8, low 8 m, nodding** | 100% | **86%** | 541 → 269 m |
| M1 Magos ×4 + ×4, roof | 85% | 81% | 210 → 193 m |
| M2 Magos ×8 + ×8, roof | 98% | 96% | 249 → 230 m |
| **M3 Magos ×4 + ×4, low 8 m** | 90% | **55%** | 220 → 90 m |
| T  TinyRad ×20, 400 m zone | 99% | 97% | 103 → 99 m |

Reading it:

- **Roof nodes barely notice the buildings.** From 45 m the line to a
  drone at 5–60 m clears every 14 m roof until it is well behind it, and the
  streets ringing DKR (San Jacinto, Red River, 23rd, MLK) are open. Cost:
  1–4 points of catch rate and 5–10% of warning distance. The 20-node roof
  config still catches 600/600.
- **Low mounts die.** At 8 m the beam is *below* the roofs across the
  street; PCL (18.5 m), Jester (16 m), the garages (22–29 m) each shadow
  a whole sector. Half the tracks come in behind a building. This removes
  the fence-mast option from §9 and is the reason the production node has to
  live on the roof — which is exactly the mount its fixed +10° fan is wrong
  for.
- **Warning distance shrinks more than catch rate**, because the long
  first look (400–500 m out) is what a building takes away; the drone is
  still seen once it clears the roofline closer in. The production node's
  8-roof-node warning goes 538 → 473 m; still 2× Magos.

What the sim still does *not* model, all of which push the other way:

1. **Detection inside the beam is certain.** A drone inside a beam is
   counted, full stop. Real single-look Pd at the range edge is 0.3–0.8;
   §4 shows cumulative Pd over an approach is ≈1 anyway, but a drone that
   clips the very edge of a beam for one frame is credited here and might
   not be in life.
2. **No trees, cranes, light masts, signage.** OSM has none of these for
   DKR; the live-oak canopy along San Jacinto is 10–15 m of foliage that
   is partly transparent at 24 GHz but not free.
3. **No clutter or false alarms.** Nothing about cars, birds, flags, or
   the crowd generating tracks. Catch rate is only half of the product; the
   other half (`docs/DEMO_TWO_RADAR.md` §classification) is not scored here.
4. **Range is a number, not a distribution.** 155 m / 600 m are the
   Mini 3 body at the stated RCS in clear air; a smaller drone, a
   nose-on aspect, or 25 mm/h rain (§6) shortens it.
5. **Building heights are OSM's.** Many are level-count estimates; a
   roof plant or parapet adds 2–4 m. Heights are capped at 70 m in the
   model.
6. **The stadium itself only blocks by facing.** Nodes face outward, so the
   bowl behind them is never in the beam; the sim does not model the
   scoreboard, light towers or the north-end structure as blockers for
   neighbouring nodes.

Net: treat the roof-node results as an upper bound that is probably within
~5 points of reality on coverage, and the low-mount results as the ones to
stop trusting.

---

## 11. Sources

- `scripts/coverage_ops.py` — the study; `sim/rt/`, `sim/tinyrad.py` for the
  beam and range models; `docs/TINYRAD_BENCH.md` §2 for the range convention
- OpenStreetMap relation 1639372 (DKR footprint), © OSM contributors, ODbL
- ITU-R P.838-3 rain attenuation coefficients
- 47 CFR 15.245, 15.249; Part 90 Subpart F (radiolocation)
- `production_node_bom.pdf` (planned two-tier node); Magos AR-300 and
  SR-series public datasheets (FOV, range, update rate, power — no pricing)
