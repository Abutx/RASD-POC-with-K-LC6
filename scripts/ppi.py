"""Live PPI-style radar scope. Moving targets appear as arcs at their range.

    python scripts/ppi.py                 # 300 MHz assumed sweep
    python scripts/ppi.py --bw 180e6      # calibrated sweep (see below)
    python scripts/ppi.py --rmax 15       # zoom the scope in

WHY ARCS AND NOT DOTS
---------------------
A dot on a scope carries range AND bearing. The K-LC6 has one transmit and one
receive antenna, so there is no angle information anywhere in the data -- a
target at +10 deg and one at -10 deg produce identical samples. Drawing a dot
would mean inventing a bearing. Instead each detection is drawn as an arc across
the antenna's real beamwidth, which is what an azimuth-unresolved target
actually looks like. Beamforming across multiple RX channels is what collapses
that arc into a dot; that is a hardware change, not a processing one.

RANGE SCALE
-----------
Range comes from R = f_beat * c / (2*S), S = B * f_ramp, so every range depends
on the sweep bandwidth B. The datasheet figure is 300 MHz. Measured against an
operator standing at a known 2.5 m, the peak landed at 1.5 m, implying the real
sweep is nearer 180 MHz. Pass --bw to set it; the scope prints what it is using.

PROCESSING
----------
Config B: 1 kHz sawtooth, 100 samples/chirp, 128 chirps per CPI (12,800 samples,
which needs device configuration 1's 16,384-sample buffer).

  range FFT (fast time) -> Doppler FFT (slow time) -> notch zero Doppler

The zero-Doppler notch is what makes this work at all. The ramp self-mixes and
puts ~16 mV pk-pk of feedthrough into the IF, roughly 100x the entire CW signal
level, and the walls and bench add more. All of it is stationary, so all of it
lands in the zero-Doppler column and is removed in one step. Measured, this is
worth about 30 dB: a static corner reflector was undetectable at 1.1 sigma,
while a moving person on the same hardware came in at 12 sigma.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
import tkinter as tk

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dwfpy as dwf                    # noqa: E402
from klc6 import acquire as A          # noqa: E402
from klc6 import process as P          # noqa: E402

SCOPE = 620          # scope canvas size, px
WFALL_W = 420        # waterfall width, px


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fs", type=int, default=100_000)
    ap.add_argument("--ramp", type=float, default=1000.0)
    ap.add_argument("--spc", type=int, default=100)
    ap.add_argument("--chirps", type=int, default=128)
    ap.add_argument("--bw", type=float, default=180e6,
                    help="sweep bandwidth Hz; 180e6 is MEASURED, 300e6 is the datasheet")
    ap.add_argument("--beamwidth", type=float, default=25.0,
                    help="azimuth beamwidth in degrees, for the arc span")
    ap.add_argument("--rmax", type=float, default=12.0, help="scope range, m")
    ap.add_argument("--rmin", type=float, default=0.4,
                    help="ignore below this; residual feedthrough lives here")
    ap.add_argument("--calib", type=float, default=3.0, help="baseline seconds")
    ap.add_argument("--threshold", type=float, default=4.0,
                    help="dB over the empty-room baseline to declare a target")
    ap.add_argument("--notch", type=int, default=3,
                    help="Doppler bins either side of zero to suppress")
    ap.add_argument("--mains", type=float, default=60.0,
                    help="mains frequency; its harmonics land on the Doppler axis")
    ap.add_argument("--mains-tol", type=float, default=0.0,
                    help="Hz tolerance for a mains row; 0 = half a Doppler bin")
    ap.add_argument("--mains-bins", type=int, default=0,
                    help="extra Doppler bins either side of each mains row")
    ap.add_argument("--persist", type=float, default=2.5, help="trail seconds")
    ap.add_argument("--v-low", type=float, default=0.5)
    ap.add_argument("--v-high", type=float, default=4.5)
    args = ap.parse_args()

    S = args.bw * args.ramp
    fs, spc, nch = args.fs, args.spc, args.chirps
    r = P.beat_to_range(np.fft.rfftfreq(spc, 1 / fs), S)
    f_d = np.fft.fftshift(np.fft.fftfreq(nch, 1 / args.ramp))
    vel = f_d / P.HZ_PER_MPS
    keep = (r >= args.rmin) & (r <= args.rmax)
    r_k = r[keep]
    win_r = np.hanning(spc)[None, :]
    win_d = np.hanning(nch)[:, None]
    notch = np.abs(np.arange(nch) - nch // 2) <= args.notch

    # Mains lands ON the Doppler axis. PRF 1 kHz spans +-500 Hz, so 60 Hz and its
    # harmonics sit at 0.37, 0.75, 1.12, 1.49, 1.86, 2.24, 2.61, 2.98 m/s -- inside
    # the display, in EVERY range bin. The empty room puts mains 35.9 dB over the
    # floor (FINDINGS section 3), and the 2026-08-30 fan study found mains was the
    # only thing that ever cleared 5 sigma. Unnotched, these are permanent false
    # targets smeared across the whole scope.
    # Tolerance must be HALF A BIN, not a fixed Hz figure. Doppler bins here are
    # PRF/nchirps = 7.81 Hz and the 60 Hz grid does not align to them, so a wider
    # tolerance grabs several rows per harmonic: at 12 Hz this notched 87 of 128
    # rows -- 68% of the velocity axis, including the 1.5 m/s a walking person
    # occupies. Half a bin claims exactly the one nearest row per harmonic.
    dop_bin = args.ramp / nch
    tol = args.mains_tol if args.mains_tol > 0 else dop_bin / 2.0
    mains_rows = np.zeros(nch, bool)
    if args.mains > 0:
        k = np.round(np.abs(f_d) / args.mains)
        on_h = (k >= 1) & (np.abs(np.abs(f_d) - k * args.mains) <= tol)
        for i in np.flatnonzero(on_h):
            lo = max(0, i - args.mains_bins)
            mains_rows[lo:i + args.mains_bins + 1] = True
    notch = notch | mains_rows

    print("=" * 70)
    print("  K-LC6 PPI SCOPE")
    print("=" * 70)
    print(f"  sweep {args.bw/1e6:.0f} MHz -> resolution {P.C/(2*args.bw):.2f} m, "
          f"slope {S:.2e} Hz/s")
    print(f"  {nch} chirps x {spc} samples = {nch*spc:,} samples per CPI "
          f"({nch/args.ramp*1e3:.0f} ms)")
    print(f"  range {args.rmin:.1f}-{args.rmax:.0f} m in {keep.sum()} bins, "
          f"velocity +-{np.abs(vel).max():.2f} m/s")
    print(f"  beamwidth {args.beamwidth:.0f} deg -- NO bearing information, "
          f"targets drawn as arcs")
    nm = int(mains_rows.sum())
    mv = sorted({round(abs(v), 2) for v in vel[mains_rows]})
    print(f"  Doppler notch: {int(notch.sum())}/{nch} rows "
          f"({int((notch & ~mains_rows).sum())} zero-Doppler + {nm} mains)")
    if mv:
        print(f"    {dop_bin:.2f} Hz Doppler bins, tol {tol:.2f} Hz -> mains rows at "
              f"+-{', '.join('%.2f' % v for v in mv)} m/s")
    frac = notch.sum() / nch
    if frac > 0.3:
        print(f"    !! {frac:.0%} of the velocity axis notched -- too much. "
              f"Lower --mains-tol / --mains-bins.")

    dev = A.open_device_cfg()
    print(f"  device {A.device_summary(dev)}")
    A.configure_chirp(dev, args.ramp, args.v_low, args.v_high, shape="sawtooth")

    def cpi():
        data, _ = A.record_chirps(dev, nch, spc, fs=fs, sync=True)
        x = data[0]
        x = x - x.mean(axis=1, keepdims=True)
        rng = np.fft.rfft(x * win_r, axis=1)
        rd = np.fft.fftshift(np.fft.fft(rng * win_d, axis=0), axes=0)
        mag = np.abs(rd)
        mag[notch, :] = 0.0                    # kill stationary clutter
        prof = mag[:, keep].max(axis=0)        # strongest mover per range bin
        vidx = mag[:, keep].argmax(axis=0)
        return 20 * np.log10(prof + 1e-12), vel[vidx]

    print(f"\n  CALIBRATING {args.calib:.0f} s -- keep still\n", flush=True)
    cal, t0 = [], time.time()
    while time.time() - t0 < args.calib:
        p, _ = cpi()
        cal.append(p)
    cal = np.array(cal)
    base = np.median(cal, axis=0)
    spread = float(np.median(np.std(cal, axis=0)))
    print(f"  baseline set over {len(cal)} CPIs, per-bin spread {spread:.2f} dB")
    print(f"  threshold {args.threshold:.1f} dB "
          f"({args.threshold/max(spread,1e-6):.1f} sigma)\n", flush=True)

    # ---------------- display ----------------
    root = tk.Tk()
    root.title("K-LC6 radar scope")
    root.configure(bg="#05140a")
    frame = tk.Frame(root, bg="#05140a"); frame.pack(padx=8, pady=8)
    cv = tk.Canvas(frame, width=SCOPE, height=SCOPE, bg="#03210f",
                   highlightthickness=0)
    cv.grid(row=0, column=0)
    wf = tk.Canvas(frame, width=WFALL_W, height=SCOPE, bg="#03120a",
                   highlightthickness=0)
    wf.grid(row=0, column=1, padx=(8, 0))
    status = tk.Label(root, text="", font=("Consolas", 11), fg="#7fff9f",
                      bg="#05140a", anchor="w")
    status.pack(fill="x", padx=12, pady=(0, 8))

    cx, cy = SCOPE / 2, SCOPE - 40          # origin: bottom centre
    span = min(SCOPE / 2 - 20, cy - 20)
    half = args.beamwidth / 2.0

    def rr(rng):                             # range -> pixels
        return span * rng / args.rmax

    # static scope furniture
    for gr in range(1, int(args.rmax) + 1):
        if args.rmax > 15 and gr % 2:
            continue
        pr = rr(gr)
        cv.create_arc(cx - pr, cy - pr, cx + pr, cy + pr,
                      start=90 - half, extent=args.beamwidth,
                      style=tk.ARC, outline="#0d7a3a")
        cv.create_text(cx + 6, cy - pr, text=f"{gr} m", fill="#0d7a3a",
                       anchor="sw", font=("Consolas", 8))
    for ang in (-half, 0, half):
        a = math.radians(90 + ang)
        cv.create_line(cx, cy, cx + span * math.cos(a), cy - span * math.sin(a),
                       fill="#0d7a3a", dash=(3, 4))
    cv.create_text(cx, 16, text=f"beamwidth {args.beamwidth:.0f}° "
                                f"- no bearing resolution",
                   fill="#0d7a3a", font=("Consolas", 9))
    cv.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill="#7fff9f", outline="")

    wf_img = np.zeros((SCOPE, int(keep.sum())))
    from PIL import Image, ImageTk
    import matplotlib
    cmap = (matplotlib.colormaps["inferno"](np.linspace(0, 1, 256))[:, :3]
            * 255).astype(np.uint8)
    photo = ImageTk.PhotoImage(Image.new("RGB", (WFALL_W, SCOPE)))
    wf_item = wf.create_image(0, 0, anchor="nw", image=photo)
    wf.create_text(WFALL_W - 6, 12, anchor="ne", fill="#7fff9f",
                   font=("Consolas", 9),
                   text=f"range->  {args.rmin:.1f}-{args.rmax:.0f} m   time v")

    trails = []          # (range, strength_db, velocity, timestamp)
    running = {"go": True}
    root.protocol("WM_DELETE_WINDOW", lambda: running.update(go=False))

    print("  LIVE -- walk toward and away. Close the window to stop.\n", flush=True)
    frames, t_start, t_fps, fps_n = 0, time.time(), time.time(), 0
    ndet, t_last_print = 0, 0.0
    try:
        while running["go"]:
            prof, vprof = cpi()
            frames += 1; fps_n += 1
            excess = prof - base
            now = time.time()

            hits = np.flatnonzero(excess > args.threshold)
            for i in hits:
                trails.append((float(r_k[i]), float(excess[i]),
                               float(vprof[i]), now))
            trails[:] = [t for t in trails if now - t[3] < args.persist]

            cv.delete("blip")
            for rng, st, vv, ts in trails:
                age = (now - ts) / args.persist
                pr = rr(rng)
                inten = max(0.0, 1.0 - age)
                # approaching = warm, receding = cool; brightness = strength
                if vv < 0:
                    col = f"#{int(90+165*inten):02x}{int(30*inten):02x}{int(30*inten):02x}"
                else:
                    col = f"#{int(30*inten):02x}{int(120+135*inten):02x}{int(90+80*inten):02x}"
                w = max(2, int(2 + 5 * inten * min(st / 12.0, 1.0)))
                cv.create_arc(cx - pr, cy - pr, cx + pr, cy + pr,
                              start=90 - half, extent=args.beamwidth,
                              style=tk.ARC, outline=col, width=w, tags="blip")

            col_db = np.clip((excess - 0) / 18.0, 0, 1)
            wf_img[1:, :] = wf_img[:-1, :]
            wf_img[0, :] = col_db
            rgb = cmap[(wf_img * 255).astype(np.uint8)]
            photo.paste(Image.fromarray(rgb, "RGB").resize((WFALL_W, SCOPE),
                                                           Image.NEAREST))

            if hits.size:
                j = hits[int(np.argmax(excess[hits]))]
                msg = (f"TARGET  {r_k[j]:5.2f} m   {excess[j]:+5.1f} dB   "
                       f"{vprof[j]:+5.2f} m/s   ({len(hits)} bins)")
                # Also print it. The status label is invisible to anything
                # reading the log, so a session run in the background looked
                # like it had zero detections when it may have had many.
                ndet += 1
                if now - t_last_print > 0.4:
                    t_last_print = now
                    print(f"    t={now-t_start:6.1f}s  TARGET {r_k[j]:5.2f} m  "
                          f"{excess[j]:+5.1f} dB  {vprof[j]:+5.2f} m/s  "
                          f"({len(hits)} bins)", flush=True)
            else:
                msg = "no movers                                        "
            if now - t_fps >= 2.0:
                rate = fps_n / (now - t_fps)
                msg += f"   |  {rate:.1f} CPI/s"
                print(f"    [{rate:4.1f} CPI/s | {ndet} detections so far | "
                      f"peak excess {excess.max():+5.1f} dB @ "
                      f"{r_k[int(np.argmax(excess))]:.2f} m]", flush=True)
                t_fps, fps_n = now, 0
            status.configure(text="  " + msg)

            root.update_idletasks(); root.update()
    except (KeyboardInterrupt, tk.TclError):
        pass
    finally:
        try:
            dev.analog_output.channels[0].reset()
        except Exception:
            pass
        dev.close()
        print(f"\n  stopped: {frames} CPIs in {time.time()-t_start:.1f} s")
        try:
            root.destroy()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
