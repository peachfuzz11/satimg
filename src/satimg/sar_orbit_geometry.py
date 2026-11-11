"""Scene geometry of a SAR product whose sensor orbit is known.

:class:`SAROrbitGeometry` is the :class:`~satimg.scene_geometry.SceneGeometry`
of a SAR scene plus the moving-target Doppler correction, worked out from
the sensor's own Earth-fixed state vectors (``attrs["orbit"]``) rather than
from the image's pixel frame -- so one implementation serves every such
product, map-projected (Capella, Umbra) or in radar geometry (ICEYE),
right- or left-looking, broadside or squinted::

    with satimg.open(path) as product:
        stored = product.geometry.to_dict()

    geometry = SAROrbitGeometry.from_dict(stored)
    geometry.correct_position(lat, lon, speed_mps, heading_deg)

``attrs["orbit"]`` is ``{"epoch", "times", "positions", "velocities"}``:
``epoch`` the scene centre's imaging time (ISO 8601, UTC), ``times``
seconds from it, ``positions`` / ``velocities`` the ECEF state (metres,
m/s) at each. See :mod:`satimg.sar` for the physics.
"""

from __future__ import annotations

from functools import cached_property

import numpy

from satimg import sar
from satimg.scene_geometry import SceneGeometry


class SAROrbitGeometry(SceneGeometry):
    """Pixel <-> lat/lon conversion, metadata and the moving-target SAR
    geometry of one scene with an ``orbit`` attr (see the module docs).

    Every target is taken as imaged when the sensor sees it at the scene
    centre's Doppler cone angle -- its zero-Doppler time for a broadside
    collect, its aperture centre for a squinted one.
    """

    @staticmethod
    def _scalar_or_array(value, single: bool):
        return float(numpy.asarray(value).ravel()[0]) if single else numpy.asarray(value)

    @cached_property
    def _orbit(self) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        orbit = self._attrs.get("orbit")
        if not orbit:
            raise ValueError("this scene's metadata has no sensor orbit: no moving-target geometry")
        return (
            numpy.asarray(orbit["times"], dtype=float),
            numpy.asarray(orbit["positions"], dtype=float),
            numpy.asarray(orbit["velocities"], dtype=float),
        )

    @cached_property
    def _cone_cosine(self) -> float:
        """The scene centre's Doppler cone cosine at the orbit's epoch."""
        lat, lon = self.transformer.rowcol_to_latlon(((self.height - 1) / 2, (self.width - 1) / 2))[0]
        position, velocity = sar.interpolate_orbit(*self._orbit, 0.0)
        return float(sar.doppler_cone_cosine(position, velocity, sar.geodetic_to_ecef(lat, lon))[0])

    def _sensor_state(self, latlon: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        """The sensor's ECEF ``(position, velocity)`` when it images each
        ``(lat, lon)``, ``(N, 3)`` each."""
        target = sar.geodetic_to_ecef(latlon[:, 0], latlon[:, 1])
        t = sar.imaging_time(*self._orbit, target, self._cone_cosine)
        return sar.interpolate_orbit(*self._orbit, t)

    def _shift(self, latlon: numpy.ndarray, speed, heading_deg):
        position, velocity = self._sensor_state(latlon)
        return sar.moving_target_shift(latlon[:, 0], latlon[:, 1], position, velocity, speed, heading_deg)

    def _latlon(self, rowcol) -> numpy.ndarray:
        return self.transformer.rowcol_to_latlon(rowcol)

    # -- public -------------------------------------------------------------
    def heading_to_los(self, rowcol, heading_deg):
        """A compass heading relative to the ground-projected radar line of
        sight at ``rowcol``, ``[-180, 180)``: ``0`` moving straight away from
        the sensor, ``+-180`` straight towards it, near ``+-90`` along track
        (little or no Doppler shift). Same convention as
        :func:`satimg.sar.heading_to_los`, with the look direction taken from
        the orbit, whichever side the radar looks."""
        latlon = self._latlon(rowcol)
        position, _ = self._sensor_state(latlon)
        look = sar.look_bearing(latlon[:, 0], latlon[:, 1], position)
        rel = (numpy.asarray(heading_deg, dtype=float) - look + 180.0) % 360.0 - 180.0
        return self._scalar_or_array(rel, numpy.ndim(rowcol) == 1)

    def doppler_azimuth_shift_m(self, rowcol, speed: float, heading_deg: float):
        """Ground displacement (metres) of a target moving at ``speed`` (m/s)
        on compass ``heading_deg`` at pixel ``rowcol``, square to the line of
        sight: positive towards the way the sensor flies, which a target
        closing on it is shifted towards. ``rowcol`` is a ``(row, col)`` pair
        or an ``(N, 2)`` array. See :func:`satimg.sar.moving_target_shift`."""
        shift, _ = self._shift(self._latlon(rowcol), speed, heading_deg)
        return self._scalar_or_array(shift, numpy.ndim(rowcol) == 1)

    def doppler_pixel_shift(self, rowcol, speed: float, heading_deg: float):
        """:meth:`doppler_azimuth_shift_m` as an image displacement: a
        ``(d_row, d_col)`` pair (an ``(N, 2)`` array for many ``rowcol``) --
        where the target appears minus where it really is."""
        rc = numpy.atleast_2d(numpy.asarray(rowcol, dtype=float))
        latlon = self._latlon(rc)
        shift, bearing = self._shift(latlon, speed, heading_deg)
        seen = sar.offset_latlon(latlon[:, 0], latlon[:, 1], shift, bearing)
        out = self.transformer.latlon_to_rowcol(seen) - rc
        return out[0] if numpy.ndim(rowcol) == 1 else out

    def correct_position(self, lat: float, lon: float, speed: float, heading_deg: float):
        """A moving target's true ``(lat, lon)`` from where it was read off
        the image, given its ``speed`` (m/s) and compass ``heading_deg`` --
        the inverse of :meth:`doppler_pixel_shift`. Same signature as
        :meth:`satimg.products.sentinel1.grd.sentinel1_grd.Sentinel1GRD.correct_position`."""
        seen = numpy.array([[lat, lon]], dtype=float)
        true = seen
        # the shift barely varies over its own length: a few fixed-point
        # steps make this the exact inverse of doppler_pixel_shift
        for _ in range(3):
            shift, bearing = self._shift(true, speed, heading_deg)
            true = sar.offset_latlon(seen[:, 0], seen[:, 1], -shift, bearing)
        return float(true[0, 0]), float(true[0, 1])
