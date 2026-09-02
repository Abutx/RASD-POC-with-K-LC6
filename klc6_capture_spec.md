# K-LC6 Capture Spec

How every training capture is recorded and labeled, so bench, field, and sim
data all land in one queryable form (`dataset/`). Referenced by TODAY.md Blocks 3–4.

## §1 Modes

| mode | what | key in npz |
|---|---|---|
| `cw` | VCO parked at DC 2.5 V, Doppler only | `data` (2, n) |
| `fmcw_chirps` | Config A/B raw chirps | `chirps` (n, spc) |
| `fmcw_cpis` | stacked CPIs | `cpis` (n, ch, spc) |

CW is the modality for micro-Doppler / blade work (no PRF, no blind speeds).

## §2 Standing settings

- Both channels `(0, 1)` = I, Q. Device configuration 1 (16,384 buffer).
- **50 kSa/s** for anything over 30 s (FINDINGS §1.2); 100 kSa/s only under 30 s.
- No amplifier: IF straight to AD2. Record `if_gain_db: 0`.
- Every capture runs through `apply_calibration` on load (DC → I/Q → spurs).

## §3 The two-file rule

Each capture is `<name>.npz` + `<name>.json` sidecar. The npz is raw volts and
sacred; the sidecar carries the label. Nothing is training data without a sidecar.

## §4 Filename

`<YYYYMMDD>_<HHMMSS>_<class>_<detail>.npz`
e.g. `20260902_143210_drone_hover_1m_throttle_mid.npz`

## §5 Sidecar fields (required unless noted)

```json
{
  "label": "drone_dji_mini",     // see dataset/README class list
  "source": "bench",             // bench | field | sionna | synth
  "mode": "cw",
  "session": "20260902_lab",     // REQUIRED — splits happen on this
  "target": "DJI Mini 2",
  "distance_m": 1.0,
  "aspect": "hover",             // approach | recede | cross | hover | static-body | fidget
  "if_gain_db": 0,
  "throttle": "mid",             // drone only — record EVERY time; blade line moves with RPM
  "notes": ""
}
```

## §6 Per-class capture plan (this session)

Human (Block 3) and drone (Block 4) tables live in `TODAY.md`. Rules that bind
every capture:

- **Label at capture time**, never after.
- **Nothing else moving in the 80° beam.** Clear it.
- Drone: **radar at prop height**, prop guards on, separate pilot.
- Record throttle setting on every drone capture.
- Include the deliberate negatives (person standing still, drone body held with
  props spinning) — they define the amp-less floor and isolate the blade signature.
