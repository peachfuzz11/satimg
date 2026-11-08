"""NISAR GCOV scene geometry, detached from the product file.

:class:`NisarGeometry` is the :class:`~satimg.scene_geometry.SceneGeometry` of a
geocoded (map-projected, north-up) NISAR GCOV scene, plus the moving-target SAR
geometry::

    with satimg.open(path) as product:
        stored = product.geometry.to_dict()

    geometry = NisarGeometry.from_dict(stored)
    geometry.correct_position(lat, lon, speed_mps, heading_deg)

Unlike Sentinel-1 GRD, a GCOV raster is already on a map grid: neither rotated
into the flight direction nor mirrored. A moving target's Doppler displacement
still happens along the satellite's ground track, which now crosses the image
at an angle -- so the shift is a ``(row, col)`` pixel vector, not a row count.
Its direction and size come straight from the product's own radar-grid cube
(line-of-sight and along-track unit vectors, slant range), so the left-looking
NISAR needs no look-side assumption baked in.
"""

from __future__ import annotations

import numpy

from satimg.scene_geometry import SceneGeometry


def _scalar_or_array(value, rowcol):
    return float(numpy.asarray(value).ravel()[0]) if numpy.ndim(rowcol) == 1 else numpy.asarray(value)


class NisarGeometry(SceneGeometry):
    """Pixel <-> lat/lon conversion, per-pixel radar-grid metadata and the
    moving-target SAR geometry of one NISAR GCOV scene.

    Needs the ``los_east`` / ``los_north`` / ``along_track_east`` /
    ``along_track_north`` / ``slant_range`` fields and the
    ``platform_velocity`` attr :class:`~satimg.products.nisar.NisarGCOVProduct`
    reads.
    """

    # -- headings ---------------------------------------------------------
    def heading_in_image(self, rowcol, heading_deg):
        """A compass heading (degrees clockwise from true north) as a
        direction in the image (degrees clockwise from image "up"), at pixel
        ``rowcol`` -- the compass heading turned by the grid convergence (see
        :meth:`~satimg.transform.base.Transformer.north_angle`). The image is a
        plain map, so this is a rotation only: never mirrored."""
        north = self.transformer.north_angle(rowcol)
        return _scalar_or_array((numpy.asarray(heading_deg, dtype=float) + north) % 360.0, rowcol)

    def heading_from_image(self, rowcol, image_heading_deg):
        """The inverse of :meth:`heading_in_image`: a direction in the image
        back to a compass heading."""
        north = self.transformer.north_angle(rowcol)
        return _scalar_or_array((numpy.asarray(image_heading_deg, dtype=float) - north) % 360.0, rowcol)

    def heading_to_los(self, rowcol, heading_deg):
        """A compass heading relative to the ground-projected radar line of
        sight at ``rowcol``, ``[-180, 180)``: ``0`` moving straight away from
        the satellite, ``+-180`` straight towards it, ``+-90`` along track (no
        Doppler shift). Same convention as
        :func:`satimg.sar_utils.heading_to_los`, but the look direction is the
        product's own LOS vector, whichever side the radar looks."""
        m = self.metadata
        # the cube's LOS points target -> sensor; away from it is the look
        look = numpy.rad2deg(numpy.arctan2(-m.los_east.at(rowcol), -m.los_north.at(rowcol)))
        rel = (numpy.asarray(heading_deg, dtype=float) - look + 180.0) % 360.0 - 180.0
        return _scalar_or_array(rel, rowcol)

    # -- Doppler ------------------------------------------------------------
    def doppler_azimuth_shift_m(self, rowcol, speed: float, heading_deg: float):
        """Along-track displacement (metres) of a target moving at ``speed``
        (m/s) on compass ``heading_deg`` at pixel ``rowcol``: positive along
        the flight direction, which a target closing on the satellite is
        shifted towards. See :func:`satimg.sar_utils.azimuth_shift_m` for the
        physics; here the range rate is the target's velocity projected onto
        the product's own line-of-sight vector."""
        m = self.metadata
        heading = numpy.deg2rad(numpy.asarray(heading_deg, dtype=float))
        v_east = numpy.asarray(speed, dtype=float) * numpy.sin(heading)
        v_north = numpy.asarray(speed, dtype=float) * numpy.cos(heading)
        # LOS points target -> sensor, so moving along it closes the range
        range_rate = -(v_east * m.los_east.at(rowcol) + v_north * m.los_north.at(rowcol))
        shift = -range_rate * m.slant_range.at(rowcol) / self._attrs["platform_velocity"]
        return _scalar_or_array(shift, rowcol)

    def doppler_pixel_shift(self, rowcol, speed: float, heading_deg: float):
        """:meth:`doppler_azimuth_shift_m` as an image displacement: a
        ``(d_row, d_col)`` pair (an ``(N, 2)`` array for many ``rowcol``) --
        where the target appears minus where it really is."""
        m = self.metadata
        shift_m = numpy.atleast_1d(self.doppler_azimuth_shift_m(rowcol, speed, heading_deg))
        track = numpy.rad2deg(numpy.arctan2(
            numpy.atleast_1d(m.along_track_east.at(rowcol)), numpy.atleast_1d(m.along_track_north.at(rowcol))
        ))
        direction = numpy.deg2rad(numpy.atleast_1d(self.heading_in_image(rowcol, track)))
        t = self.transformer.transform
        d_row = -shift_m * numpy.cos(direction) / abs(t.e)
        d_col = shift_m * numpy.sin(direction) / abs(t.a)
        out = numpy.column_stack((d_row, d_col))
        return out[0] if numpy.ndim(rowcol) == 1 else out

    def correct_position(self, lat: float, lon: float, speed: float, heading_deg: float):
        """A moving target's true ``(lat, lon)`` from where it was read off
        the image -- the inverse of :meth:`doppler_pixel_shift`. Same
        signature as
        :meth:`satimg.products.sentinel1.Sentinel1Product.correct_position`."""
        rowcol = self.transformer.latlon_to_rowcol((lat, lon))[0]
        true_rowcol = rowcol - self.doppler_pixel_shift(rowcol, speed, heading_deg)
        true_lat, true_lon = self.transformer.rowcol_to_latlon(true_rowcol)[0]
        return float(true_lat), float(true_lon)

