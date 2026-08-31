"""Analog Discovery 2 acquisition for the K-LC6 24 GHz Doppler module.

Streams the K-LC6 IF output through the AD2 in record mode and hands back raw
volts. Nothing here processes the signal -- every downstream idea depends on
having the original samples, so only volts are ever stored.

Two constraints from the spec that bite immediately:
  * WaveForms and a script cannot hold the device at the same time.
  * The default 8192-sample buffer is 82 ms at 100 kSa/s. Anything longer needs
    record mode, which streams continuously to the host.
"""
from __future__ import annotations

import datetime as _dt
import json
import os

import numpy as np

import dwfpy as dwf

# ---- physical constants (24.125 GHz CW) ----
C = 299_792_458.0
F_CARRIER = 24.125e9
LAMBDA = C / F_CARRIER            # 0.012427 m
HZ_PER_MPS = 2.0 / LAMBDA         # 160.95 Hz per m/s

# ---- defaults ----
FS = 100_000                      # >= 48 kSa/s or blade content aliases
RANGE_V = 0.5                     # 0.5 bare module, 0.05 if amplified
OFFSET_V = -0.2                   # K-LC6 IF sits on a +0.2 V DC pedestal
CH_I = 0                          # orange 1+ -> X1 pin 3
CH_Q = 1                          # blue   2+ -> X1 pin 1


class DeviceBusy(RuntimeError):
    pass


def open_device():
    """Open the first Analog Discovery. Caller is responsible for closing it."""
    devices = dwf.Device.enumerate()
    if not devices:
        raise DeviceBusy(
            "No Analog Discovery found.\n"
            "  - is the USB cable connected?\n"
            "  - is WaveForms open? It holds the device exclusively; close it.")
    device = devices[0]
    try:
        device.open()          # dwfpy opens in place and returns None
    except Exception as exc:   # dwf reports device-busy as a generic error
        raise DeviceBusy(
            f"Found {device.name} (SN {device.serial_number}) but could not "
            f"open it: {exc}\nClose the WaveForms application and try again.") from None
    return device


def device_summary(device) -> str:
    return (f"{device.name} (SN {device.serial_number}, rev {device.revision})")


def configure(device, fs=FS, channels=(CH_I,), range_v=RANGE_V,
              offset_v=OFFSET_V, filter_mode="average"):
    """Enable channels, set range/offset/rate. Returns the analog-in module."""
    ain = device.analog_input
    ain.reset()
    for ch in ain.channels:
        ch.enabled = False
    for idx in channels:
        # range <= 5 V selects the AD2's low-gain path; offset recentres the
        # K-LC6's DC pedestal so a DC-coupled front end does not sit on a rail.
        # filter must go through setup_channel: the .filter property wants a
        # FilterMode enum, while setup_channel accepts the string form.
        ain.setup_channel(idx, range=range_v, offset=offset_v,
                          filter=filter_mode, enabled=True)
    ain.setup_acquisition(mode=dwf.AcquisitionMode.RECORD, sample_rate=fs)
    return ain


def record(device, duration_s, fs=FS, channels=(CH_I,), range_v=RANGE_V,
           offset_v=OFFSET_V, filter_mode="average", progress=None):
    """Stream `duration_s` seconds. Returns (n_channels, n_samples) float64 volts.

    Raises on any lost or corrupted sample. A silently gap-filled array produces
    spectrogram artifacts that look exactly like real Doppler content, so a
    partial capture is worse than no capture.
    """
    ain = configure(device, fs=fs, channels=channels, range_v=range_v,
                    offset_v=offset_v, filter_mode=filter_mode)
    recorder = ain.record(sample_rate=fs, length=duration_s,
                          configure=True, start=True)

    # WARNING: `progress` is retained only for experiments. Do not use it for a
    # capture you intend to keep. Passing a callback to recorder.record() sends
    # dwfpy down a path that never fills the channel buffer -- a 30 s capture
    # returned 25 real samples after 2,999,975 zeros, with the callback firing
    # once, at the end, and lost/corrupted both reporting clean.
    if progress is None:
        recorder.record()
    else:
        recorder.record(callback=progress)

    if recorder.lost_samples or recorder.corrupted_samples:
        raise RuntimeError(
            f"capture integrity failure: {recorder.lost_samples} lost, "
            f"{recorder.corrupted_samples} corrupted samples.\n"
            f"The host could not keep up. Lower fs, or shorten the capture.")

    data = np.array([np.asarray(c.data_samples, dtype=np.float64)
                     for c in recorder.channels])

    # The lost/corrupted counters do NOT catch an unfilled buffer. A 300 s
    # capture once returned 29,999,966 exact zeros followed by 34 real samples
    # and still reported 0 lost, 0 corrupted -- the accumulated chunks had been
    # consumed by reading data_samples from inside the progress callback. A real
    # signal always varies over a whole second, so a dead-flat head means the
    # buffer was never populated.
    # The first read leaves one device-buffer's worth of unwritten zeros at the
    # head -- measured as exactly 8192 samples (163.8 ms at 50 kSa/s), a single
    # contiguous run starting at index 0, with lost/corrupted both reporting
    # clean. Trim it: 8192 zero-valued samples are a broadband impulse that
    # smears across the whole spectrum.
    lead = 0
    if data.shape[1]:
        allzero = np.all(data == 0.0, axis=0)
        if allzero[0]:
            nz = np.flatnonzero(~allzero)
            lead = int(nz[0]) if nz.size else data.shape[1]
    if lead:
        if lead > 4 * 8192:
            raise RuntimeError(
                f"capture starts with {lead:,} zero samples "
                f"({lead/fs*1e3:.0f} ms) -- far more than one device buffer. "
                f"The buffer was not populated; do not pass a progress callback "
                f"to recorder.record().")
        data = data[:, lead:]

    head = data[:, :min(int(fs), data.shape[1])]
    if data.shape[1] >= fs and np.all(head.std(axis=1) == 0):
        raise RuntimeError(
            "capture buffer was not populated: the first second is dead flat "
            "while lost/corrupted report clean.\n"
            "Do not read recorder.channels[].data_samples inside the record "
            "callback -- that consumes the accumulated chunks.")
    if np.any(np.all(data == 0.0, axis=0)):
        n0 = int(np.sum(np.all(data == 0.0, axis=0)))
        raise RuntimeError(
            f"{n0:,} zero samples remain after trimming the leading buffer. "
            f"Dropouts mid-record produce impulse artifacts; discard this capture.")
    return data, recorder


def channel_names(channels) -> list:
    names = {CH_I: "I", CH_Q: "Q"}
    return [names.get(c, f"ch{c}") for c in channels]


def capture_to_file(path, duration_s, fs=FS, channels=(CH_I,), range_v=RANGE_V,
                    offset_v=OFFSET_V, metadata=None, device=None):
    """Record and write an .npz of raw volts plus provenance."""
    owned = device is None
    device = device or open_device()
    try:
        data, rec = record(device, duration_s, fs=fs, channels=channels,
                           range_v=range_v, offset_v=offset_v)
    finally:
        if owned:
            device.close()

    meta = dict(metadata or {})
    meta.setdefault("sample_rate", fs)
    meta.setdefault("channels", channel_names(channels))
    meta.setdefault("range_v", range_v)
    meta.setdefault("offset_v", offset_v)
    meta.setdefault("mode", "cw")

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        data=data,
        fs=fs,
        channel_names=np.array(channel_names(channels)),
        timestamp=_dt.datetime.now().astimezone().isoformat(),
        metadata=json.dumps(meta),
    )
    return path, data


# =====================================================================
# FMCW (SPEC.md section 10) -- W1 -> X1 pin 5 (VCO in)
# =====================================================================

# The 128-chirp CPI of Config B is 12,800 samples, which does not fit the AD2's
# default 8,192-sample analog-in buffer. Device configuration 1 trades
# analog-out buffer (16,384 -> 1,024, ample for a sawtooth) for 16,384
# analog-in samples, so a whole CPI lands in one triggered acquisition.
CFG_BIG_AIN = 1

SWEEP_BW_HZ = 300e6               # K-LC6 VCO span, per datasheet
RANGE_RES_M = C / (2 * SWEEP_BW_HZ)   # 0.4997 m


def open_device_cfg(configuration=CFG_BIG_AIN):
    """Open the AD2 in a specific device configuration."""
    try:
        dev = dwf.AnalogDiscovery2(configuration=configuration)
        # The constructor only selects; __enter__ (or open()) actually opens it,
        # and until then device.analog_input is None.
        if not dev.is_open:
            dev.open()
    except Exception as exc:
        raise DeviceBusy(
            f"could not open the Analog Discovery in configuration "
            f"{configuration}: {exc}\nClose WaveForms and try again.") from None
    return dev


def configure_chirp(device, f_ramp, v_low, v_high, shape="sawtooth",
                    channel=0, start=True):
    """Drive W1 with the VCO ramp. Returns the analog-out channel.

    Sawtooth (RAMP_UP at 100% symmetry), not triangle: a triangle's down-ramp
    inverts the sign of the beat frequency, so half of every period has to be
    discarded or processed separately.
    """
    ch = device.analog_output.channels[channel]
    ch.reset()
    fn = {"sawtooth": dwf.Function.RAMP_UP,
          "triangle": dwf.Function.TRIANGLE,
          "dc": dwf.Function.DC}[shape]
    amplitude = (v_high - v_low) / 2.0
    offset = (v_high + v_low) / 2.0
    kw = dict(function=fn, offset=offset, enabled=True,
              configure=True, start=start)
    if shape != "dc":
        kw.update(frequency=float(f_ramp), amplitude=amplitude, symmetry=100.0)
    else:
        kw.update(amplitude=0.0)
    ch.setup(**kw)
    return ch


def set_dc(device, volts, channel=0):
    """Hold W1 at a fixed voltage (VCO characterisation, SPEC 10.1)."""
    ch = device.analog_output.channels[channel]
    ch.reset()
    ch.setup(function=dwf.Function.DC, amplitude=0.0, offset=float(volts),
             enabled=True, configure=True, start=True)
    return ch


def record_chirps(device, n_chirps, samples_per_chirp, fs=FS,
                  channels=(CH_I,), range_v=RANGE_V, offset_v=OFFSET_V,
                  filter_mode="average", sync=True):
    """Capture n_chirps x samples_per_chirp, triggered on the ramp.

    Chirp-synchronous triggering is mandatory: without it every acquisition
    starts at a random ramp phase, the slow-time phase progression across
    chirps is meaningless, and the Doppler FFT integrates noise instead of a
    target. The trigger source is the analog-out channel driving W1.

    Returns (data, ain) with data shaped (n_channels, n_chirps, samples_per_chirp).
    """
    total = int(n_chirps) * int(samples_per_chirp)
    ain = device.analog_input
    ain.reset()
    for ch in ain.channels:
        ch.enabled = False
    for idx in channels:
        ain.setup_channel(idx, range=range_v, offset=offset_v,
                          filter=filter_mode, enabled=True)
    if total > ain.buffer_size_max:
        raise ValueError(
            f"{n_chirps} x {samples_per_chirp} = {total:,} samples exceeds the "
            f"{ain.buffer_size_max:,}-sample buffer of this device "
            f"configuration. Open with open_device_cfg({CFG_BIG_AIN}) for "
            f"16,384, or lower n_chirps.")

    ain.setup_acquisition(mode=dwf.AcquisitionMode.SINGLE, sample_rate=fs,
                          buffer_size=total)
    if sync:
        ain.trigger.source = dwf.TriggerSource.ANALOG_OUT1
        ain.trigger.position = 0.0
    else:
        ain.trigger.source = dwf.TriggerSource.NONE

    ain.configure(reconfigure=True, start=True)
    ain.wait_for_status(dwf.Status.DONE, read_data=True)

    rows = []
    for idx in channels:
        x = np.asarray(ain.channels[idx].get_data(), dtype=np.float64)
        if x.size != total:
            raise RuntimeError(
                f"expected {total:,} samples, got {x.size:,}; the reshape into "
                f"chirps would be wrong")
        rows.append(x.reshape(int(n_chirps), int(samples_per_chirp)))
    return np.array(rows), ain


def chirp_slope(bw_hz=SWEEP_BW_HZ, f_ramp=1000.0):
    """Sweep slope S in Hz/s, from bandwidth and ramp rate."""
    return bw_hz * float(f_ramp)
