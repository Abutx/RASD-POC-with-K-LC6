# Ray-Traced Micro-Doppler — Sionna RT and Physical Optics

**Ask:** use NVIDIA Sionna RT for the Doppler simulations, as accurately as
possible. **Outcome:** Sionna RT installed, validated, and shown — with
numbers — to be the wrong amplitude model for wavelength-scale rotor blades;
a physical-optics facet engine built in its place, validated to machine
precision and to translation invariance; the drone's RCS derived from
geometry instead of guessed, and it confirms the guesses.
Reproduce with `scripts/rt_smoke.py`, `scripts/rt_calib.py`, `scripts/rt_drone.py`,
`scripts/rt_sweep.py`. Code in `sim/rt/`.

Every prior table in this repo used a DJI Mini 3 RCS **extrapolated from a
Phantom 4** (−17 dBsm body, blades 30 dB below), flagged each time as the
weakest number in the analysis. §6 replaces it with a geometric result.

---

## 1. VERDICT

| question | answer |
|---|---|
| Does Sionna RT run here? | **Yes.** 2.0.1 on Python 3.13, CPU (LLVM) backend. The `jitc_llvm_init` warning is benign. |
| Is its geometry/phase right? | **Exactly.** 2 m plate at 5 m: −80.10 dB vs −80.10 dB theory; delay to the picosecond. |
| Are its amplitudes right for blades? | **No — by ~28–36 dB.** It applies image theory to surfaces far smaller than the Fresnel zone. |
| So what does the accurate sim use? | **Physical-optics facets** on the real twisted-blade mesh: Gordon's exact polygon integral, Fresnel dielectric, soft occlusion, image-method floor. |
| Is *that* validated? | Gordon vs closed form **1e-16**; plate RCS **−0.01 dB**; translation invariance **3e-14 dB**; 2 vs 32 facets **0.03 dB**; angular convergence at 2× **<0.4%**. |
| What is the Mini 3's RCS? | **Body −13.4 dBsm** aspect-averaged (Phantom 4 Pro measured: −12.4). **Rotors −48.6 dBsm** at level view, rising to −33 at 60°. |
| Were the old assumptions right? | **Yes.** Body within 4 dB, blades within 2 dB. |
| What did geometry add that the analytic model missed? | At level view **the blade return comes from the inner span, not the tip**: only 19% of blade energy sits near the tip Doppler, 68% in 1–4 kHz. |

Sionna keeps two honest roles: the calibration harness that quantified its
own limitation, and the phase/delay reference every other model must match.
The scene builder (`sim/rt/scene.py`, K-LC6 pattern at 12.5 dBi gain /
15.7 dBi directivity in Sionna's own check) stays for room-scale multipath
work, where Sionna is the right tool.

---

## 2. What the Sionna tutorial offers, and why it does not fit

The Mobility tutorial's fast path assigns a `velocity` vector to a scene
object and synthesises Doppler in `paths.cfr()` — a **rigid-translation
model** that the tutorial itself says "degrades over longer horizons." A
rotor blade's velocity varies along its span and reverses every half-turn.
The correct use of Sionna for a rotor is the tutorial's other path: **time-step
the geometry and re-trace**, via `SceneObject.orientation`.

`sim/rt/trace.py` does that with one saving observation: a rotor's return is a
function of its angle alone once radar and body are fixed. Each rotor is
traced over half a turn (two identical blades → period π) at 0.125° steps,
**once**, and the lookup is driven by any RPM profile — four independent
rotors, jitter, throttle — at zero further tracing cost.

---

## 3. Sionna RT — validated, then measured against physical optics

### 3.1 Smoke test (`scripts/rt_smoke.py`) — passes

| | result |
|---|---|
| backend | traces on LLVM CPU; first call 0.9 s (JIT), steady state **0.06 s at 20 M rays** |
| PLY loader | our numpy-generated meshes load as `SceneObject` |
| 2 m PEC plate at 5 m, iso antennas | **−80.10 dB** vs **−80.10 dB** image theory λ²/(64π²R²) |
| delay | 33.356 ns vs 2R/c = 33.356 ns |
| convention | `a` is the passband coefficient; baseband = Σ a·e^{−j2πfτ}, as documented |

### 3.2 Calibration (`scripts/rt_calib.py`) — the limitation, quantified

A **10 × 60 mm PEC plate at 3 m** — blade-sized. Physical optics at normal
incidence: σ = 4πA²/λ² = **−15.3 dBsm**.

| Sionna setting | RCS returned | vs PO |
|---|---|---|
| pure specular (S=0) | **+12.5 dBsm** | **+27.9 dB** |
| S=0.5, directive α=1…16 | +14.0 to +14.2 | +29.3 to +29.5 |
| S=0.8 | +16.3 to +16.8 | +31.7 to +32.1 |
| S=1.0 | +19.9 to +20.4 | +35.2 to +35.7 |

Off-normal the same plate returned 16.5 / 18.7 / 18.7 / 11.1 dBsm at
0 / 5 / 10 / 20° — **flat**, where physical optics has its first null at
λ/D ≈ 12° (our PO engine: −16 dB at 10°). No aperture-dependent lobe exists
in Sionna's model for a surface this size.

**Diagnosis.** Image theory for an infinite mirror at 3 m has equivalent RCS
πR² = +14.5 dBsm — within 2 dB of Sionna's specular result. Sionna applies
image-theory amplitude to any specular point regardless of the surface's
extent. For a sub-Fresnel-zone surface the overestimate is
πR²λ²/(4A²) = **+29.8 dB** predicted; +28–32 dB measured. The rotor mesh
itself: 0 paths at 400 k rays, 22 at 20 M, RCS still climbing 10 dB per 4×
rays — hit-count-limited and unconverged.

**Conclusion:** a geometric-optics ray tracer cannot give trustworthy
amplitudes for λ-scale blades. Its geometry, delays and antenna handling
are exact.

---

## 4. Physical-optics facets (`sim/rt/po.py`)

The standard high-frequency RCS method for this size regime. Each facet
contributes

    amp = (k/√π) · ½(cos θᵢ + cos θₛ) · I(w) · Γ

with I(w) the PO integral of e^{j w·r} over the facet **in coordinates centred
on the facet** (the shape factor), and the position phase applied separately
as e^{−j2kR_centroid}. I(w) is Gordon's closed-form edge sum; Γ is the Fresnel
coefficient (PEC = −1; nylon/glass-fibre ε_r = 3.7 − j0.06, ≈ −10 dB re
metal). Also: **soft occlusion** (four sample rays per facet, so shadow
boundaries do not flicker between rotor angles) and **four-path floor
multipath** by the image method with concrete's complex Fresnel coefficient
(ε_r = 5.24 − j0.60) at each path's grazing angle.

### Validation

| test | result |
|---|---|
| Gordon vs exact sinc product, six random w to 3k | **rel. err. ≤ 6×10⁻¹⁶** |
| 6 cm PEC plate, normal incidence | −0.01 dB vs 4πA²/λ² |
| 10° off normal | −16.0 dB re normal (first null at 11.9°) |
| **translation invariance** — plate and radar shifted 0.56 m together | **+3×10⁻¹⁴ dB, −5×10⁻¹⁶ rad** |
| **subdivision** — 2 vs 32 facets, 10° off normal | **+0.03 dB** |
| **angular convergence** — el 10°, 1440 vs 2880 angles | rotors −44.9 dBsm both; band fractions within 0.4% |
| lookup spectrum vs physical ceiling 2k·r_tip | **0.2–0.3%** beyond it; |R| autocorrelation 2.75–4.4°, as λ/D predicts |

### The bug the last three tests exist for

The first version evaluated Gordon in world coordinates. e^{j w·r} then
already carried each facet's position phase, and the separate e^{−j2kR}
counted it again: every facet's phase advanced at 2× the physical rate,
67% of rotor energy landed above the tip Doppler, blade-flash fringes came
out at half their true angular period, and the lookup was spiky at the
0.125° sample scale. Every origin-centred self-test passed. Translation
invariance and subdivision would have failed immediately; they are in the
suite now, and the lookup-spectrum ceiling check is what diagnosed it.

### Limits, stated

PO is reliable for facets ≳ λ/2 near specular and weakens toward grazing and
away from the main lobe; it omits edge diffraction (PTD is the next
refinement). A blade chord is ~1 λ. At **level elevation** the pitched blade
faces sit at 60–80° incidence — PO's weakest regime — so the level-view
blade amplitude is the least certain number here. The body is a PEC box, so
its broadside specular peak (−3.7 dBsm) is sharper than a real rounded
airframe's; the aspect-averaged figure is the one to quote.

---

## 5. The Mini 3 as geometry (`sim/rt/mesh.py`)

| part | model | material |
|---|---|---|
| props | 6030F: R = 76.2 mm, 2 blades, chord 12 → 8 mm, 1.5 mm thick, **twist from the 3 in pitch** (31° at r = 20 mm → 9° at the tip) | nylon + glass fibre |
| motors | 4 × Ø18 × 12 mm at the arm tips | metal |
| body | 100 × 60 × 35 mm block — battery and frame; the plastic shell is nearly transparent at 24 GHz | metal |
| arms | 10 mm square section | nylon |
| geometry | 247 mm motor-to-motor; hubs 33.5 mm above body centre | |

Radar placed by range, azimuth and **elevation as seen from the drone:
positive above the rotor plane, negative below** (the analytic model is
symmetric in this sign; the PO model is not — the body is under the rotors
and the floor under everything). Concrete floor at z = 0, drone at 1.2 m.

---

## 6. Results — 3 m range, 6040 rpm, 1440 angles per rotor

### 6.1 The RCS, from geometry

| quantity | assumed (Phantom-4 extrapolation) | **PO** |
|---|---|---|
| **body, aspect-averaged over 360° heading** | −17 dBsm | **−13.4 dBsm** (median −27.0; Phantom 4 Pro measured −12.4) |
| body, broadside specular peak, level | — | −3.7 dBsm |
| body, level, off-broadside (±10°, 30°) | — | −19 to −30 dBsm |
| **four rotors summed, level view** | −47 dBsm | **−48.6 dBsm** |
| each rotor, level, angle-averaged | — | −53 to −56 dBsm |
| each rotor, flash peak | — | −38 to −43 dBsm |
| motors (4, metal) | — | −29 to −46 dBsm |
| arms | — | below −50 dBsm |

**The extrapolation every table flagged as its weakest number was right**:
body within 4 dB, blades within 2 dB. `DEMO_TWO_RADAR.md` and the TinyRad
analyses stand as written.

### 6.2 Elevation — two effects the analytic model cannot see

| radar elevation | rotors summed | body (this heading) | blades below body | blade energy 4–7.8 kHz | 1–4 kHz |
|---|---|---|---|---|---|
| **0° (level)** | **−48.6 dBsm** | −3.7 | 45.0 dB | **19%** | **68%** |
| +10° above | −44.9 | −19.2 | 25.7 dB | 24% | 65% |
| **−10° below** | −43.3 | −29.8 | **13.5 dB** | 27% | 62% |
| +30° above | −44.7 | −25.6 | 19.2 dB | 13% | 60% |
| +60° above | **−32.9** | −18.8 | 14.1 dB | 0.2% | 84% |
| analytic, any el | −47 | −17 | 30 dB | 63% (level) | 36% |

Two findings, both robust across floor on/off and the 2× convergence run:

**1. Blade RCS is minimum at level view and rises off-level — 16 dB higher at
60°.** The 3 in pitch tilts the blade faces 9–31° from horizontal; a level
radar sees them at 60–80° incidence (grazing), a radar 60° above sees them
near normal. Blade *Doppler* collapses in exactly the direction blade *RCS*
grows (the cos(el) kinematic null). **Level view maximises micro-Doppler
bandwidth and minimises micro-Doppler power; steep elevation does the
reverse.** No prior table had this trade; the analytic model has neither half
of it.

**2. At level view the blade return comes from the inner span, not the tip.**
Root facets (pitch 31°) are seen at 59° incidence, tip facets (9°) at 81°. So
**68% of blade energy sits in 1–4 kHz and only 19% near the 7.76 kHz tip
line**, against 63% near the tip in the line-scatterer model. Consequences
for the demo: a detector tuned to the tip line is looking in the wrong place
for a level-mounted radar; the blade energy lands closer to the mains comb
(dead by ~6 kHz — `DEMO_TWO_RADAR.md` §1) and inside the fan's 2.4–4 kHz
band. The band-energy detector in `sim/klc6.py` should integrate from
**~1 kHz**, not 4 kHz.

**3. Slightly below the rotor plane is the most blade-favourable geometry**
found: at −10° the blades are only 13.5 dB under the body. That is the
natural geometry for a ground radar and a low drone.

### 6.3 Floor bounce

Level view, concrete floor vs none: rotors −48.6 vs −47.7 dBsm; body −3.7 vs
−3.6. **Under 1 dB** at 3 m and 1.2 m height; it mostly adds a little
low-Doppler energy (12% vs 6% below 1 kHz). Real, small, included.

### 6.4 Figures

- `out/demo/rt_mini3_r3_el{0,10,-10,30,60}_floor_6040rpm.png` — analytic vs
  PO vs PO-through-K-LC6+AD2, same kinematics
- `out/demo/rt_rotor_rcs_vs_angle_*.png` — the flash structure: a clean
  specular event at 90° (blade broadside), ~15° wide, −38 to −43 dBsm peak
- captures in `out/sim/rt_mini3_*.npz` (`source: "sionna"`, `sim_method`
  recorded); they index and load through `dataset.manifest` / `loader`
  unchanged

---

## 7. How to use it

```bash
python scripts/rt_smoke.py                      # Sionna backend + phase convention
python scripts/rt_calib.py                      # the +30 dB finding, reproducible
python scripts/rt_drone.py --el 0 --yaw-sweep   # trace, synthesise, figures, RCS
python scripts/rt_drone.py --el -10             # radar below the rotor plane
python scripts/rt_sweep.py                      # elevations + convergence + floor
```

Traces cache in `out/rt_cache/` keyed on every geometric parameter; **purge
the directory after any change to `po.py` or `mesh.py`** — the key does not
carry a physics version. One 1440-angle trace is ~6 min on this machine.
The kinematics are `sim/drone.py`'s unchanged, so every difference between the
analytic and PO spectrograms is a **scattering** effect, by construction.

---

## 8. Sources

- NVIDIA Sionna RT 2.0.1 — Mobility tutorial; `Paths`, `SceneObject`,
  `PathSolver`, `RadioMaterial`, antenna-pattern API docs
- Gordon, W. B., "Far-field approximations to the Kirchhoff-Helmholtz
  representations of scattered fields," IEEE Trans. AP-23, 1975
- ITU-R P.2040 material constants (concrete at 24 GHz)
- Semkin et al., arXiv:1911.05926 — Phantom 4 Pro −12.4 dBsm at 25 GHz
- `docs/DEMO_TWO_RADAR.md` §3 (the extrapolation this confirms),
  `sim/drone.py`, `sim/klc6.py`
