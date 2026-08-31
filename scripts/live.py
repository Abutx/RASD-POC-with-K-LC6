"""Live motion detector. Big banner goes red when it sees Doppler.

    python scripts/live.py                  # hand waves and walking
    python scripts/live.py --threshold 4    # less twitchy
    python scripts/live.py --vlim 40        # include fan/rotor velocities
    python scripts/live.py --nfft 4096      # 24 fps ceiling, coarser velocity

Why this is a raw Tkinter window and not matplotlib
---------------------------------------------------
Benchmarked on this machine, redrawing the same three-panel figure:

    draw_idle + pause, constrained_layout    1.6 fps
    draw_idle + pause, tight_layout          3.8 fps
    blit, tight_layout                       6.2 fps   <- best matplotlib
    this Tk + PIL display                   >60 fps

matplotlib's Agg rasteriser re-scales the spectrogram to screen pixels in
software every frame, and TkAgg then copies the whole canvas. Blitting helped
but not nearly enough. Rendering the spectrogram as a small PIL image and
letting Tk scale it once is two orders of magnitude cheaper, which puts the
frame rate back where it belongs: limited by acquisition, not by drawing.

Acquisition is SINGLE-shot; record mode runs ~2x wall clock and leaves 8192
unwritten samples at the head of every buffer. Frame time is nfft/fs, which is
also 1/(velocity resolution): 8192 pt at 100 kSa/s is 82 ms, a 12 fps ceiling.
Going to 4096 doubles the rate but halves resolution AND lets more of the huge
low-frequency content leak into the detect band -- measured, the empty-room
band-power spread went from 0.83 dB to 9.42 dB, which destroys the threshold.
A high-pass and a Blackman-Harris window are applied per frame to hold that
leakage down.

Mains harmonics are notched before detection: the measured baseline puts 60 Hz
at 0.373 m/s, 180 Hz at 1.119 m/s, 300 Hz at 1.864 m/s -- inside the walking
band, 36 dB over the floor.

KEEP THE ROOM EMPTY for the calibration countdown at the start.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import tkinter as tk

import numpy as np
from scipy.signal import butter, sosfilt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dwfpy as dwf                    # noqa: E402
from klc6 import acquire as A          # noqa: E402
from klc6 import process as P          # noqa: E402

W, H_SPEC, H_MET = 960, 380, 130


def build_mask(freqs, lo_hz, hi_hz, mains_hz, tol_hz):
    m = (np.abs(freqs) >= lo_hz) & (np.abs(freqs) <= hi_hz)
    if mains_hz > 0:
        k = np.round(np.abs(freqs) / mains_hz)
        m &= ~((k >= 1) & (np.abs(np.abs(freqs) - k * mains_hz) <= tol_hz))
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fs", type=int, default=100_000)
    ap.add_argument("--nfft", type=int, default=8192)
    ap.add_argument("--vlim", type=float, default=6.0)
    ap.add_argument("--vmin-detect", type=float, default=0.25)
    ap.add_argument("--vmax-detect", type=float, default=6.0)
    ap.add_argument("--mains", type=float, default=60.0)
    ap.add_argument("--mains-tol", type=float, default=3.0)
    ap.add_argument("--hp", type=float, default=20.0, help="high-pass, Hz")
    ap.add_argument("--calib", type=float, default=3.0)
    ap.add_argument("--threshold", type=float, default=2.0,
                    help="dB over calibrated empty room; 0 = auto (6x spread)")
    ap.add_argument("--hold", type=float, default=0.6)
    ap.add_argument("--ncols", type=int, default=240)
    ap.add_argument("--range", type=float, default=5.0)
    ap.add_argument("--offset", type=float, default=A.OFFSET_V)
    ap.add_argument("--dyn", type=float, default=45.0, help="colour range, dB")
    args = ap.parse_args()

    from PIL import Image, ImageTk
    import matplotlib

    fs, nfft = args.fs, args.nfft
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    vel = freqs / P.HZ_PER_MPS
    # Blackman-Harris: ~-92 dB sidelobes vs Hann's -31, so the huge DC/low-freq
    # content of the K-LC6 IF does not leak into the detect band.
    win = np.blackman(nfft)
    sos = butter(4, args.hp, btype="highpass", fs=fs, output="sos")

    detect_mask = build_mask(freqs, args.vmin_detect * P.HZ_PER_MPS,
                             args.vmax_detect * P.HZ_PER_MPS,
                             args.mains, args.mains_tol)
    show = vel <= args.vlim
    nshow = int(show.sum())
    raw_band = ((np.abs(freqs) >= args.vmin_detect * P.HZ_PER_MPS) &
                (np.abs(freqs) <= args.vmax_detect * P.HZ_PER_MPS))

    print("=" * 70)
    print("  K-LC6 LIVE MOTION DETECTOR")
    print("=" * 70)
    print(f"  {fs:,} Sa/s, {nfft} pt -> {fs/nfft:.1f} Hz "
          f"({fs/nfft/P.HZ_PER_MPS:.3f} m/s) bins")
    print(f"  frame {nfft/fs*1e3:.0f} ms -> {fs/nfft:.0f} fps acquisition ceiling")
    print(f"  detect {args.vmin_detect:.2f}-{args.vmax_detect:.2f} m/s, "
          f"{int(detect_mask.sum())} bins "
          f"({int(raw_band.sum()-detect_mask.sum())} notched)")

    dev = A.open_device()
    print(f"  device {A.device_summary(dev)}")
    ain = dev.analog_input
    ain.reset()
    ain.setup_channel(0, range=args.range, offset=args.offset,
                      filter="average", enabled=True)
    ain.setup_acquisition(mode=dwf.AcquisitionMode.SINGLE, sample_rate=fs,
                          buffer_size=nfft)
    ain.configure(reconfigure=True, start=True)
    ain.wait_for_status(dwf.Status.DONE, read_data=True)

    def frame():
        ain.configure(reconfigure=False, start=True)
        ain.wait_for_status(dwf.Status.DONE, read_data=True)
        x = np.asarray(ain.channels[0].get_data(), dtype=np.float64)
        x = sosfilt(sos, x - x.mean())
        return np.abs(np.fft.rfft(x * win)) ** 2

    print(f"\n  CALIBRATING {args.calib:.0f} s -- keep the room EMPTY and still\n",
          flush=True)
    cal, t0 = [], time.time()
    while time.time() - t0 < args.calib:
        cal.append(10 * np.log10(frame()[detect_mask].sum() + 1e-30))
    cal = np.array(cal)
    base, spread = float(np.median(cal)), float(np.std(cal))
    thr = args.threshold if args.threshold > 0 else max(8.0, 6.0 * spread)
    print(f"  empty-room band power {base:.1f} dB, spread {spread:.2f} dB "
          f"over {len(cal)} frames")
    print(f"  threshold {thr:.1f} dB ({thr/max(spread,1e-6):.1f} sigma)\n", flush=True)

    # ---------------- Tk display ----------------
    root = tk.Tk()
    root.title("K-LC6 live motion detector")
    root.configure(bg="#101010")

    banner = tk.Label(root, text="CLEAR", font=("Segoe UI", 46, "bold"),
                      fg="white", bg="#2e7d32", height=2, width=28)
    banner.pack(fill="x", padx=8, pady=(8, 4))

    cv_spec = tk.Canvas(root, width=W, height=H_SPEC, bg="#101010",
                        highlightthickness=0)
    cv_spec.pack(padx=8)
    spec_item = cv_spec.create_image(0, 0, anchor="nw")

    cv_met = tk.Canvas(root, width=W, height=H_MET, bg="#181818",
                       highlightthickness=0)
    cv_met.pack(padx=8, pady=(4, 8))
    status = tk.Label(root, text="", font=("Consolas", 11), fg="#bbbbbb",
                      bg="#101010", anchor="w")
    status.pack(fill="x", padx=10, pady=(0, 8))

    # metric panel: fixed guides drawn once, polyline updated per frame
    met_lo, met_hi = -4.0, max(20.0, thr * 4)
    def met_y(v):
        return H_MET - (v - met_lo) / (met_hi - met_lo) * H_MET
    cv_met.create_line(0, met_y(0), W, met_y(0), fill="#555555")
    cv_met.create_line(0, met_y(thr), W, met_y(thr), fill="#e53935", dash=(5, 3))
    cv_met.create_text(6, met_y(thr) - 9, anchor="w", fill="#e53935",
                       font=("Consolas", 10), text=f"threshold {thr:.1f} dB")
    trace = cv_met.create_line(0, met_y(0), W, met_y(0), fill="#4fc3f7", width=2)

    # velocity gridlines over the spectrogram
    for gv in np.arange(1.0, args.vlim + 0.01, 1.0):
        y = H_SPEC - gv / args.vlim * H_SPEC
        cv_spec.create_line(0, y, W, y, fill="#ffffff", stipple="gray25")
        cv_spec.create_text(W - 6, y - 8, anchor="e", fill="#dddddd",
                            font=("Consolas", 9), text=f"{gv:.0f} m/s")

    cmap = (matplotlib.colormaps["viridis"](np.linspace(0, 1, 256))[:, :3]
        * 255).astype(np.uint8)
    img_db = np.full((nshow, args.ncols), -140.0)
    hist = np.zeros(args.ncols)
    # One PhotoImage, updated in place with paste(). Building a new PhotoImage
    # every frame re-converts the bitmap into Tk's internal format and
    # re-registers it with the interpreter -- measured at ~55 of the 65 ms
    # per-frame draw cost.
    photo = ImageTk.PhotoImage(Image.new("RGB", (W, H_SPEC)))
    cv_spec.itemconfig(spec_item, image=photo)

    running = {"go": True}
    root.protocol("WM_DELETE_WINDOW", lambda: running.update(go=False))

    print("  LIVE -- wave a hand or walk in front of the module. "
          "Close the window or Ctrl+C to stop.\n", flush=True)
    last_hit, hits, frames = -1e9, 0, 0
    t_start = t_fps = time.time()
    fps_n, t_acq, t_draw = 0, 0.0, 0.0

    try:
        while running["go"]:
            ta = time.time()
            sp = frame()
            tb = time.time(); t_acq += tb - ta
            frames += 1; fps_n += 1

            level = 10 * np.log10(sp[detect_mask].sum() + 1e-30)
            excess = level - base
            now = time.time()
            if excess > thr:
                last_hit = now; hits += 1
                pv = float(vel[detect_mask][np.argmax(sp[detect_mask])])
                print(f"    t={now-t_start:6.1f}s  MOTION  {excess:+5.1f} dB  "
                      f"peak {pv:+.2f} m/s", flush=True)
            live = (now - last_hit) < args.hold

            banner.configure(text="MOTION DETECTED" if live else "CLEAR",
                             bg="#c62828" if live else "#2e7d32")

            img_db[:, :-1] = img_db[:, 1:]
            img_db[:, -1] = 10 * np.log10(sp[show] + 1e-30)
            top = img_db.max()
            norm = np.clip((img_db - (top - args.dyn)) / args.dyn, 0, 1)
            rgb = cmap[(norm * 255).astype(np.uint8)]          # (nshow, ncols, 3)
            photo.paste(Image.fromarray(rgb[::-1], "RGB")
                        .resize((W, H_SPEC), Image.NEAREST))

            hist[:-1] = hist[1:]; hist[-1] = excess
            xs = np.linspace(0, W, args.ncols)
            pts = []
            for xv, yv in zip(xs, np.clip(hist, met_lo, met_hi)):
                pts += [float(xv), float(met_y(yv))]
            cv_met.coords(trace, *pts)

            root.update_idletasks(); root.update()
            t_draw += time.time() - tb

            if now - t_fps >= 3.0:
                el = now - t_fps
                msg = (f"{fps_n/el:.1f} fps   acquire "
                       f"{t_acq/max(fps_n,1)*1e3:.1f} ms   draw "
                       f"{t_draw/max(fps_n,1)*1e3:.1f} ms   "
                       f"level {excess:+.1f} dB   detections {hits}")
                status.configure(text="  " + msg)
                print(f"    [{msg}]", flush=True)
                t_fps, fps_n, t_acq, t_draw = now, 0, 0.0, 0.0
    except (KeyboardInterrupt, tk.TclError):
        pass
    finally:
        dev.close()
        el = time.time() - t_start
        print(f"\n  stopped: {frames} frames in {el:.1f} s "
              f"({frames/max(el,1e-9):.1f} fps), {hits} detections")
        try:
            root.destroy()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
