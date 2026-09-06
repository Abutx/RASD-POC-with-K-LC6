# BOM — TinyRad Field Rig

Outdoor deployment at Harmony School of Endeavor. One TinyRad in a weatherproof box on a pole, an edge computer streaming to the network, powered over Ethernet.

**Do not order the field parts until the bench rig has proven two things:** the Python acquisition works, and it runs on whatever edge computer you pick. Both are verified in the bench BOM's first-tasks list.

---

## Bill of materials

### Radar and compute

| Item | Qty | Cost | Notes |
|---|---|---|---|
| EV-TINYRAD24G | 1 | ~$400 | Same unit as the bench rig, or a second one |
| Edge computer | 1 | $80–200 | **See §Edge computer** — depends on Linux support |
| USB-C cable, short | 1 | in box | TinyRad to edge computer |
| microSD, 64 GB, high-endurance | 1 | $15 | If Raspberry Pi. Local buffer for captures. |

### Enclosure

| Item | Qty | Cost | Notes |
|---|---|---|---|
| IP66 enclosure, ~250 × 200 × 100 mm | 1 | $35 | ABS or polycarbonate body |
| Polycarbonate sheet, 3 mm | 1 | $15 | Radome window. **Not acrylic, not glass** — both are lossy at 24 GHz. |
| Silicone sealant, clear | 1 | $6 | Seal the window |
| Cable glands, PG9 or M16 | 2 | $8 | Ethernet in, spare |
| Desiccant packs | 4 | $5 | Condensation |
| Standoffs, M3, nylon | set | $6 | Mount the boards off the enclosure floor |

**Radome spacing:** the K-LC6 datasheet's radome guidance applies — window at least 6 mm in front of the antenna face, uniform thickness across the beam. Use the same rule for TinyRad.

### Power and network

| Item | Qty | Cost | Notes |
|---|---|---|---|
| PoE splitter, 802.3at, 5 V / 4 A, USB-C out | 1 | $20 | Powers the edge computer |
| PoE injector, 802.3at | 1 | $25 | At the building end, or a PoE switch port |
| Outdoor Cat6, shielded, direct-burial | 30–50 m | $30 | Run from the nearest network closet |
| Ethernet surge protector, outdoor | 1 | $20 | Both ends ideally |
| RJ45 waterproof coupler | 1 | $8 | At the enclosure gland |

### Mounting

| Item | Qty | Cost | Notes |
|---|---|---|---|
| Pole mount bracket, tilt-adjustable | 1 | $25 | Or wall mount, depending on the site |
| Hose clamps or U-bolts | 2 | $8 | |
| Mounting plate for the enclosure | 1 | $10 | |

### Totals

| | Cost |
|---|---|
| Radar | $400 |
| Edge computer | $80–200 |
| Enclosure | $75 |
| Power and network | $103 |
| Mounting | $43 |
| **Total** | **$700–820** |

---

## Edge computer — decide after the bench test

TinyRad's vendor software is Windows-first. Whether the Python acquisition library runs on Linux determines which box goes in the enclosure.

| If Python runs on | Buy | Cost | Notes |
|---|---|---|---|
| Linux / ARM | **Raspberry Pi 5, 8 GB** | $80 | Runs acquisition and streams. Range-Doppler-angle processing at a few fps is fine on a Pi 5. Classifier inference too. |
| Windows only | **Intel NUC or mini PC, x86** | $180–200 | Adds size and power draw. PoE splitter needs to supply 12 V instead — pick a splitter to match. |

**Check first:** ADI EngineerZone, search "TinyRad Linux" and "TinyRad Python Raspberry Pi." If someone has done it, follow their path. If not, budget a day to try before falling back to x86.

---

## Site placement

From the survey:

- **Point it at the pickup lane** for vehicle counting. Cars pass one at a time through a line — 1D range plus bearing is plenty.
- **Radar at 2–3 m height**, tilted slightly down toward the lane. Drone captures from the drone team happen wherever they fly — coordinate a schedule.
- **Nothing between the radar and the lane** — no trees, no fence posts. Foliage costs a couple of dB per meter and moves in wind.
- **Within 50 m of a network drop.** Longer runs need a PoE extender.
- **Power budget:** TinyRad plus Pi 5 is under 15 W. 802.3at PoE delivers 25 W. Comfortable.

---

## Data path

```
TinyRad ──USB-C──► edge computer ──Ethernet/PoE──► school network ──► Supabase
                        │
                        └── local microSD buffer, 24 h rolling
```

- Edge computer runs `acquire.py`, applies calibration, runs range-Doppler-angle and the tracker
- Pushes detections and tracks to Supabase continuously
- Buffers raw captures locally; uploads on a schedule or on-demand to keep bandwidth reasonable
- Heartbeat every minute so you know it's alive

**The school's IT will want to know:** what traffic it generates, that it's outbound-only, and that it stores nothing about people. Have that answer written before you install.

---

## Install-day checklist

1. Bench-test the complete assembled enclosure indoors first — radar, edge computer, PoE, all in the box, streaming to Supabase, for 24 hours.
2. Site survey with a tape: mounting point, cable route, distance to network drop.
3. Mount, cable, power.
4. Corner reflector at a measured distance in the lane — range and angle check on site.
5. Confirm heartbeat and data arriving in Supabase from the school network.
6. Leave the corner reflector in place for the first day as a known target.
7. Come back after 48 hours with the drone team.

---

## What it needs from the school

- A network drop within cable reach and a PoE port or wall outlet for the injector
- Permission to mount on a pole or wall
- A one-paragraph host letter from the school leader — this is the artifact for the SBIR application
- A drone-team flight schedule for labelled captures

---

## Redo when

- Pi swaps or reflashes — recalibrate spur map
- Enclosure opened — recheck radome seal
- Moved — re-run the corner reflector check
- Any change to TinyRad chirp config — re-run background reference
