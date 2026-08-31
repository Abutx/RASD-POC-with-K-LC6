# Fan Detection — Synthesized Verdict

**Date:** 2026-08-30
**Input:** 8 independent DSP analyses of the fan-on captures (CW 50/100 kSa/s, FMCW Config A/B) against the 2026-08-29 fan-off references.
**Scripts/figures/logs:** `C:/dev/klc6/out/fan/analysis/` (one `.py`/`.png`/`.log`/`.json` set per method).

---

## 1. VERDICT

**The fan was NOT detected. Confidence ~0.9 that no fan signature exists in this data above the receiver's measured sensitivity floor.** All 8 methods returned nulls; zero claimed a detection; every candidate peak above 5σ resolved to 60 Hz mains or session/quantiser artefacts under scrutiny. This is a **well-characterised, sensitivity-limited null**, not an inconclusive shrug: the six methods that passed their positive control all reproduce the documented moving-person detection (+6.7–7.9 dB at 1.5–2.5 m, 9.6–20.9σ, FINDINGS 5.5) through byte-identical pipelines against the same references, and injection calibrations bound any fan line at astonishingly low levels — a coherent CW Doppler line above **8.1 µV rms** (41× below one ADC LSB), a Config B moving scatterer above **1.44 µV rms per cell**, or an AM depth above **~2–3%** would have been found at 5σ. Nothing was. The fan's moving-target return is bounded **≥6.4–11.7 dB below a walking person at the same 2.5 m range** on the same hardware.

**What would overturn this:** a same-session fan-on/fan-off A/B pair (every off reference here is from the previous day, at a different ADC operating point, 0.8–3 dB gain offset); a fan-off Config B capture (the one config with real blade-Doppler coverage has no reference at all); or a repeat with the fan side-on to the boresight (if the rotation axis pointed at the radar, blade motion was tangential and the radial Doppler is nulled by geometry — the null would then be about geometry, not sensitivity). None of these is expected to flip the verdict, but any of them could.

---

## 2. What each method found

| Method | Control | Detected | Best fan-candidate evidence | Significance | Survived verification |
|---|---|---|---|---|---|
| Doppler comb (CW) | PASS (+12.1σ person; injection 8.1 µV @5σ) | **No** | 2877.8 Hz "blade-tip" line, +5.61σ | = 48th mains harmonic (fitted f₀ drift 0.39 Hz explains it); cross-chunk r = +0.015 vs +0.98 control | n/a — null |
| Cadence-velocity (CW) | **FAIL** (CVD can't see the 4.8 s person; MTI arm passes at 13.6σ) | **No** | 4.75σ at 127.8 Hz over 962 bins | Same-condition nulls reach 4.69–5.61σ — indistinguishable from drift | n/a — null |
| Cepstrum / HPS (CW) | **FAIL** on person (expected — no comb in gait); PASS on 2 real combs (50 Hz feedthrough @313σ, mains) + injection | **No** | 82.78 Hz cepstral spacing, 4.42σ | Direct comb-energy test: +0.012 dB / +0.29σ; doesn't repeat in run 2 (2.21σ) | n/a — null |
| Envelope periodicity (CW) | PASS (2.09 Hz gait cadence, 5.57σ; mains AM @228σ) | **No** | 3.906 Hz line in both fan-on runs, 3.18σ | Empty room shows 3.74σ at the *same bin* | n/a — null |
| Range-Doppler MTI (FMCW B) | PASS (person +7.7–7.9 dB, 9.6–13.3σ) | **No** | ±51.83 Hz sideband pair on feedthrough bin, z = 8.0 | AM of TX leakage: tracks feedthrough not range; no CW counterpart under resolution scaling | Rejected as fan return |
| Timing alignment / coherent integration (FMCW) | PASS (person 11.5σ coherent, 13.8σ incoherent) | **No** | Config B cell at 1.30 m / −0.29 m/s, 7.54σ | Symmetric in ±Doppler, inside 60 Hz Hann mainlobe, tracks feedthrough | Rejected as mains AM |
| Statistical detectors (CW+FMCW) | PASS (person 11.9σ per-bin, 8.8σ aggregate no-peak-picking) | **No** | +2.2 dB at 180 Hz, +2.5 dB at 300 Hz, 2.1–2.3σ | Both mains harmonics (electrical-load trap), sub-threshold anyway | n/a — null |
| Time-freq / blade-flash template (CW) | PASS (person 13.3–20.9σ) | **No** | Flash-rate excess +2.0σ | Carried by one 6 s burst; 5× disagreement between fan-on chunks; steady rotor ⇒ constant rate | n/a — null |

Tally: 8/8 ran, 6/8 passed the positive control, 0/8 detections, 0 candidates survived adversarial checks. The two control failures (CVD, cepstrum-on-person) are documented method limitations — a 4.8 s walking human has no stable periodicity for those estimators — and both methods separately validated their comb/periodicity machinery by injection into real data, so their fan nulls remain informative *for a periodic rotor*, which is what a fan is.

---

## 3. Where the methods agree and disagree

**Agreement — the load-bearing evidence:**

1. **The validated MTI/range-Doppler detector, run by five independent implementations against the same fan-off Config A reference, gives −0.9 to +0.6 dB (|σ| ≤ 1.8) at the fan's 2.5 m bin**, where the same code finds the person at +6.7 to +7.9 dB, 9.6–13.8σ. Five independent codebases, same answer, same range, same reference. This is the strongest single fact in the study.
2. **Fan-on is broadband QUIETER than fan-off by 0.8–1.2 dB in every band, including 60–300 m/s where no fan can radiate.** Four methods independently measured this and all attributed it to the same cause: session gain / ADC-pedestal offset (fan-on IF 108–158 µV rms on 1–2 dominant codes; fan-off 158–174 µV straddling two codes). The sign is wrong for a target.
3. **Every >5σ feature resolved to 60 Hz mains, by structurally independent tests:** fitted mains-fundamental drift (48th harmonic at 2877.8 Hz), empty-room presence at equal strength (180 Hz, 60 Hz cadence), ±Doppler symmetry (Config B sidebands = AM of a static scatterer, not motion), and appearance on an AD2 channel carrying no radar signal (60 Hz quantiser AM). Three methods hit the *same* seductive trap — a mains-related line at a plausible blade-pass or blade-tip value (51.8 Hz ↔ 1037 rpm 3-blade; 60 Hz ↔ 901–1201 rpm; 2878 Hz ↔ 17.9 m/s tip) — and all three killed it independently.
4. **Cross-chunk / cross-run reproducibility fails everywhere.** The two independent fan-on captures never agree on any non-mains feature (comb spacing, ACF lag, cadence, flash rate), while the positive control reproduces at r = +0.96–0.98. A real rotor cannot be present in one 25–60 s capture and absent two minutes later.
5. **Blade-tip bands (12–62 m/s) are the emptiest part of the spectrum:** −0.02 to −0.10 dB against measured floors of 0.03–0.05 dB.

**Disagreement:** none on the verdict. The only spread is in *scope*: the two failed-control methods can only claim "no blade comb / no cadence" rather than "no fan", and the timing-alignment analysis correctly notes that heavily-aliased blade micro-Doppler smeared across apparent range in FMCW is bounded (max 2.9σ over 492 bins) rather than excluded — but the CW whole-band search, which has no such ambiguity, covers exactly that case down to 8.1 µV rms and finds nothing.

**One loose end, flagged not resolved:** the Config B ±51.8 Hz feedthrough sideband is statistically real (z = 8.0). It is definitively *not* a range-resolved radar return, but with no fan-off Config B capture it cannot be attributed. If it is the fan mechanically vibrating the module, that is a non-radar coupling with no range information — an interesting curiosity, not a detection.

---

## 4. Fan parameters, if detected

Not applicable — no detection. For the record, the search space covered blade counts 2–5, shaft speeds ~200–13000 rpm (comb spacings 8.3–667 Hz), and blade-tip velocities out to 124 m/s; a plausible household fan (900–1600 rpm, 15–30 m/s tips) could not have fallen outside it. Upper bounds instead of parameters:

- Coherent Doppler line anywhere in 20 Hz–20 kHz: **< 8.1 µV rms** at the IF (−23 dB re in-band PSD floor).
- Blade-comb total power: **< ~10 µV rms** (−24 dB re 158 µV IF rms).
- Blade-flash AM depth: **< 2–3%** of the IF envelope over 60 s.
- Config B per-cell moving return at 2.5 m: **< 1.44 µV rms** (0.0043 LSB).
- MTI excess at 2.5 m: **< 2.0–2.9 dB** (5σ), vs the person's +6.8–7.9 dB ⇒ fan return ≥ 6.4–11.7 dB below a person.

---

## 5. Why it is hard on this hardware

- **Quantisation is the wall.** The AD2's smallest range is 5 Vpp → **336 µV LSB**; the bare CW IF occupies **4–7 distinct ADC codes** with AC rms ~0.47 LSB (~97 µV of the measured 158 µV rms is quantisation noise alone). Everything ever detected on this bench got there purely on FFT processing gain. The same rig produced an honest 1.1σ null on a **2.7 m² corner reflector** — a plastic-bladed fan at 2.5 m is a far smaller RCS.
- **Energy dilution.** A rotor doesn't concentrate its return in one Doppler bin: it smears from 0 to tip speed (≈4 kHz at 25 m/s), across ~2.6 m of range-Doppler-coupled apparent range in FMCW, and aliases in both configs (Config A unambiguous |v| < 0.155 m/s; Config B < 3.1 m/s). Per-cell energy is therefore even further below the 336 µV floor than the total return.
- **Mains at 35.9 dB** over the noise floor, with harmonics landing exactly on plausible blade signatures (60 Hz ↔ 0.373 m/s; 2880 Hz ↔ 17.9 m/s "blade tip"). Cross-session mains drift of a few mHz moves high-order harmonics by whole FFT bins, manufacturing 5σ+ per-bin differences (the 2877.8 Hz trap). High-order mains tags are near-worthless: ±2 Hz tags 6–7% of random bins above 3 kHz, and on Config B's aliased Doppler axis a mains tag hits 25% of bins by chance.
- **Ramp feedthrough (FMCW): 16.8 mV pk-pk, ~100× the CW signal.** The 49–61 code span of the chirp records is feedthrough, not signal; its 60 Hz AM produced both fake "blade" sidebands. Handled by order-3 polynomial detrend + consecutive-chirp MTI throughout.
- **Session confound.** No same-session fan-off exists in any mode. Fan-on vs fan-off differ by a 4-code DC pedestal shift and 0.8–3 dB of broadband level — larger than any excess a plastic fan could plausibly add — forcing every test onto gain-invariant shape/prominence statistics.
- **Real-only IF** (Q channel unwired): one-sided spectrum, approach folds onto recede, direction information lost.

---

## 6. What to do next (ranked)

1. **IF amplifier (FINDINGS open item 3).** The single change that settles everything. Gain of 30–40 dB before the ADC moves the IF from ~0.5 LSB rms to full-range, converting the 336 µV quantisation wall into a thermal-noise-limited floor — a **~30–40 dB sensitivity gain**, dwarfing anything software can do. Every method's caveat list converges on this.
2. **Same-session fan A/B capture** (10 minutes): fan-on 60 s → fan physically in place but OFF 60 s → on again, same rate, no replug. Removes the session/pedestal confound that currently caps every absolute-power test at ~1 dB, tightening those floors by **~5–10×** and separating "fan invisible" from "fan not in beam".
3. **Fan-off Config B capture (30–60 s).** The only config with usable blade-Doppler coverage has zero references. Directly attributes or kills the z = 8.0 ±51.8 Hz feedthrough sideband and enables a real A/B in range-Doppler.
4. **Reorient the fan side-on** so blade tips sweep along the line of sight, and move it to 1 m. Side-on geometry recovers the full radial tip velocity (potentially the whole signature if the null is geometric); 2.5 m → 1 m buys **+16 dB** (R⁴). Optional: foil tape on one blade for a further ~10–20 dB RCS boost and a guaranteed once-per-rev flash for template validation.
5. **Wire the Q channel (open item 4).** +3 dB SNR, unfolds approach/recede, makes a blade wing one-sided and far more distinctive against symmetric mains AM.
6. **Record a CW capture with a person walking.** Closes the one validation gap every CW method flagged: no CW positive control exists in the dataset. Cheap, and it converts future CW nulls from "injection-validated" to "end-to-end validated".
7. **Higher PRF / longer contiguous dwell for blade micro-Doppler** (Config B at max PRF, or CW at ≥50 kSa/s with the IF amp): unaliased tip Doppler needs ±4 kHz; coherent dwell beyond 128 ms is currently pointless (measured scene coherence ratio 7.3–7.6 dB vs 6.9 dB noise expectation) but becomes the integration lever once the quantiser is no longer the floor.

---

*Synthesis of 8 independent analyses; all scripts, figures and logs under `C:/dev/klc6/out/fan/analysis/`. Verdict written per ground rule 6: this null is the answer, not a failure to find one.*
