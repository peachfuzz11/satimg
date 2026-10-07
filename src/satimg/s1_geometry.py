"""Sentinel-1 scene geometry, detached from the product files.

:class:`Sentinel1Geometry` is the :class:`~satimg.scene_geometry.SceneGeometry`
of a Sentinel-1 GRD scene: built from (and stored as) the annotation's
geolocation grid points, the scene-wide attrs and the image shape, plus the SAR
moving-target geometry -- so Doppler shifts can be estimated after the product
file is gone::

    with satimg.open(path) as product:
        stored = product.geometry.to_dict()

    geometry = SceneGeometry.from_dict(stored)    # a Sentinel1Geometry
    shift_px = geometry.doppler_azimuth_shift(rowcol, speed, heading)
"""

from __future__ import annotations

import numpy

from satimg import s1_utils, sar_utils
from satimg.metadata import grid_from_points
from satimg.scene_geometry import SceneGeometry


class Sentinel1Geometry(SceneGeometry):
    """Pixel <-> lat/lon conversion, per-pixel geolocation metadata and the
    moving-target SAR geometry of one Sentinel-1 GRD scene.

    ``points`` and ``attrs`` are as returned by
    :func:`satimg.s1_utils.read_geolocation`; ``shape`` is the full image's
    ``(height, width)``.
    """

    type_name = "sentinel1"

    def __init__(self, points: list[dict], attrs: dict, shape: tuple[int, int]):
        self._points = points
        keys = tuple(s1_utils.GEOLOC_FIELDS)
        grid = grid_from_points(points, keys)
        fields = {
            name: (grid["rows"], grid["cols"], grid[name], s1_utils.GEOLOC_FIELDS[name][1])
            for name in keys
        }
        super().__init__(s1_utils.build_transformer(points), fields, attrs, shape)

    # -- serialisation --------------------------------------------------
    def to_dict(self) -> dict:
        """Stored as the geolocation grid points the GCP transformer and the
        fields are both built from."""
        return {
            "type": self.type_name,
            "points": [dict(p) for p in self._points],
            "attrs": dict(self._attrs),
            "shape": list(self._shape),
        }

    @classmethod
    def _from_dict(cls, data: dict) -> "Sentinel1Geometry":
        return cls(data["points"], data["attrs"], tuple(data["shape"]))

    # -- SAR geometry ---------------------------------------------------
    def _local_platform_heading(self, rowcol):
        """Satellite ground-track heading at ``rowcol``'s own latitude (rather
        than the single scene-wide ``platform_heading`` attr), via
        :func:`satimg.sar_utils.ground_track_heading`."""
        lat = self.transformer.rowcol_to_latlon(rowcol)[:, 0]
        ascending = self._attrs["pass"].lower() == "ascending"
        heading = sar_utils.ground_track_heading(
            lat, self._attrs["orbit_inclination"], ascending
        )
        return float(heading[0]) if numpy.ndim(rowcol) == 1 else heading

    def heading_to_los(self, rowcol, heading_deg):
        """See :meth:`satimg.products.sentinel1.Sentinel1Product.heading_to_los`."""
        rel = sar_utils.heading_to_los(heading_deg, self._local_platform_heading(rowcol))
        return float(rel) if numpy.ndim(rowcol) == 1 else numpy.asarray(rel)

    def heading_in_image(self, rowcol, heading_deg):
        """See :meth:`satimg.products.sentinel1.Sentinel1Product.heading_in_image`."""
        result = sar_utils.heading_in_image(heading_deg, self._local_platform_heading(rowcol))
        return float(result) if numpy.ndim(rowcol) == 1 else numpy.asarray(result)

    def doppler_azimuth_shift(self, rowcol, speed: float, heading_deg: float):
        """See :meth:`satimg.products.sentinel1.Sentinel1Product.doppler_azimuth_shift`."""
        m = self.metadata
        incidence = m.incidence_angle.at(rowcol)
        slant_range_m = m.slant_range_time.at(rowcol) * sar_utils.SPEED_OF_LIGHT / 2
        shift_m = sar_utils.azimuth_shift_m(
            speed, heading_deg, self._local_platform_heading(rowcol), incidence,
            slant_range_m, self._attrs["platform_velocity"],
        )
        shift_px = shift_m / self._attrs["azimuth_pixel_spacing"]
        return float(shift_px) if numpy.ndim(rowcol) == 1 else numpy.asarray(shift_px)

    def correct_position(self, lat: float, lon: float, speed: float, heading_deg: float):
        """See :meth:`satimg.products.sentinel1.Sentinel1Product.correct_position`."""
        rowcol = self.transformer.latlon_to_rowcol((lat, lon))[0]
        shift_px = self.doppler_azimuth_shift(rowcol, speed, heading_deg)
        corrected_rowcol = (rowcol[0] - shift_px, rowcol[1])
        corrected_lat, corrected_lon = self.transformer.rowcol_to_latlon(corrected_rowcol)[0]
        return float(corrected_lat), float(corrected_lon)
