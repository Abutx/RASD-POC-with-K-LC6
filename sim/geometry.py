"""Two-radar geometry: coverage, crossing angle, position accuracy.

The demo puts two K-LC6 modules a baseline `b` apart, both looking into a
shared volume, and fixes the target by intersecting their two range circles.
This module answers the three questions that follow from that:

  1. where do the two beams actually overlap (coverage)
  2. how does range error turn into position error there (GDOP)
  3. how far out is the answer still worth having

There is NO bearing information in a single K-LC6 -- one Tx, one Rx, so a
target at +10 deg and one at -10 deg give identical samples (FINDINGS 6). The
second radar is not a refinement, it IS the angular measurement: the entire
cross-range accuracy of the demo comes from the baseline, and nothing else.
"""
from __future__ import annotations

import numpy as np

from . import klc6


class Rig:
    """Two radars on a baseline, both boresighted into the shared volume.

    Coordinates: baseline along x, radars at (-b/2, 0) and (+b/2, 0), both
    looking toward +y. A target at (x, y) with y > 0 is in front of both.
    """

    def __init__(self, baseline_m=4.0, toe_in_deg=0.0,
                 az_beam_deg=klc6.BEAM_WIDE_DEG, el_beam_deg=klc6.BEAM_NARROW_DEG):
        self.baseline_m = float(baseline_m)
        self.toe_in_deg = float(toe_in_deg)
        self.az_beam_deg = float(az_beam_deg)
        self.el_beam_deg = float(el_beam_deg)

    @property
    def positions(self):
        b = self.baseline_m / 2.0
        return np.array([[-b, 0.0], [+b, 0.0]])

    @property
    def boresights(self):
        """Unit boresight vectors, toed in toward the centreline by toe_in_deg."""
        t = np.radians(self.toe_in_deg)
        return np.array([[np.sin(t), np.cos(t)], [-np.sin(t), np.cos(t)]])

    # ---------------------------------------------------------- measurement
    def ranges(self, xy):
        """True range from each radar to points `xy` of shape (..., 2)."""
        xy = np.asarray(xy, dtype=float)
        d = xy[..., None, :] - self.positions            # (..., 2 radars, 2)
        return np.linalg.norm(d, axis=-1)

    def los_units(self, xy):
        """Unit vectors radar -> target, shape (..., 2 radars, 2)."""
        xy = np.asarray(xy, dtype=float)
        d = xy[..., None, :] - self.positions
        return d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)

    def crossing_angle_deg(self, xy):
        """Angle subtended at the target by the two radars.

        This single number governs position quality. 90 deg is ideal; as it
        collapses toward 0 the two range circles become tangent and the
        cross-range coordinate stops being observable at all.
        """
        u = self.los_units(xy)
        cosb = np.clip(np.sum(u[..., 0, :] * u[..., 1, :], axis=-1), -1.0, 1.0)
        return np.degrees(np.arccos(cosb))

    # ---------------------------------------------------------------- GDOP
    def gdop(self, xy):
        """sqrt(trace((H^T H)^-1)) for H = the two unit LOS vectors.

        Works out to sqrt(2)/|sin(beta)| exactly, with beta the crossing angle.
        """
        beta = np.radians(self.crossing_angle_deg(xy))
        return np.sqrt(2.0) / np.maximum(np.abs(np.sin(beta)), 1e-9)

    def position_sigma_m(self, xy, sigma_range_m):
        """1-sigma position error from equal, independent range errors."""
        return np.asarray(sigma_range_m, dtype=float) * self.gdop(xy)

    def error_ellipse_m(self, xy, sigma_range_m):
        """(semi-major, semi-minor) of the 1-sigma position error ellipse.

        Eigenvalues of (H^T H)^-1 are 1/(1 -+ |cos beta|), so the ellipse is
        stretched along the bisector of the two lines of sight -- error grows
        in the down-range direction, not across it. Worth knowing for a demo:
        the track will look tight left-right and loose in and out.
        """
        beta = np.radians(self.crossing_angle_deg(xy))
        c = np.abs(np.cos(beta))
        s = np.asarray(sigma_range_m, dtype=float)
        return s / np.sqrt(np.maximum(1.0 - c, 1e-9)), s / np.sqrt(1.0 + c)

    # ------------------------------------------------------------ coverage
    def in_beam(self, xy, n_beamwidths=1.0):
        """Boolean: is the point inside BOTH azimuth main lobes (and in front)?

        Elevation is a separate and, with the narrow lobe, much harsher
        constraint -- see elevation_slab_m().
        """
        xy = np.asarray(xy, dtype=float)
        u = self.los_units(xy)
        bs = self.boresights
        half = np.radians(self.az_beam_deg / 2.0) * n_beamwidths
        ok = np.ones(xy.shape[:-1], dtype=bool)
        for i in range(2):
            cos_off = np.clip(np.sum(u[..., i, :] * bs[i], axis=-1), -1.0, 1.0)
            ok &= cos_off >= np.cos(half)
        return ok

    def elevation_slab_m(self, range_m):
        """Vertical extent of the elevation beam at a given range.

        With the module mounted for a wide 80 deg azimuth fan, elevation is the
        narrow 12 deg lobe and the covered volume is a thin horizontal sheet:
        0.21 m tall per metre of range. A drone that climbs out of that sheet
        vanishes, and the operator will read it as a detection failure.
        """
        return 2.0 * np.asarray(range_m, dtype=float) * np.tan(
            np.radians(self.el_beam_deg / 2.0))

    def two_way_beam_loss_db(self, xy):
        """Combined off-boresight loss for both radars, dB (two-way each)."""
        u = self.los_units(xy)
        bs = self.boresights
        loss = np.zeros(np.asarray(xy).shape[:-1])
        for i in range(2):
            cos_off = np.clip(np.sum(u[..., i, :] * bs[i], axis=-1), -1.0, 1.0)
            off_deg = np.degrees(np.arccos(cos_off))
            loss += klc6.beam_gain_db(off_deg, self.az_beam_deg)
        return loss


def solve_position(rig, r1, r2, forward=True):
    """Intersect two range circles -> (x, y). The classic two-sphere solution.

    Two circles meet at two points, mirrored about the baseline. They are a
    genuine ambiguity that no amount of processing removes; the demo resolves
    it by construction, because both modules only look into y > 0. Say so out
    loud rather than pretending the geometry is unambiguous.
    """
    b = rig.baseline_m
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    x = (r1 ** 2 - r2 ** 2) / (2.0 * b)          # radars at -b/2 and +b/2
    y2 = r1 ** 2 - (x + b / 2.0) ** 2
    y = np.sqrt(np.maximum(y2, 0.0))
    return np.stack([x, y if forward else -y], axis=-1)


def optimal_range_m(baseline_m):
    """Range on the centreline where the crossing angle is exactly 90 deg.

    beta = 2*arctan((b/2)/y) = 90 deg  =>  y = b/2. Position accuracy is best
    at half a baseline out and degrades as (range/baseline)^2 beyond it, which
    is the fact that sizes the rig: to hold quality at range you move the
    modules apart, you do not improve the radar.
    """
    return np.asarray(baseline_m, dtype=float) / 2.0


def baseline_for_range(range_m, gdop_max=3.0):
    """Baseline needed to keep GDOP below `gdop_max` at `range_m` on the centreline.

    GDOP = sqrt(2)/sin(beta), beta = 2*arctan(b/(2y)).
    """
    beta = np.arcsin(np.clip(np.sqrt(2.0) / gdop_max, 0.0, 1.0))
    return 2.0 * np.asarray(range_m, dtype=float) * np.tan(beta / 2.0)
