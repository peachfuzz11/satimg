"""Sentinel-1 GRD (IW and EW) reader."""

from __future__ import annotations

import datetime
import os
import re

import numpy
import PIL.Image
import xarray

from satimg import s1_utils, sar_utils
from satimg.metadata import Metadata, grid_from_points
from satimg.product import Product
from satimg.readers import keep_open, load_thumbnail, merge_bands
from satimg.registry import register
from satimg.source import Source
from satimg.tiling import as_band_yx, label_bands


@register(r"^S1[ABCD]_(IW_GRDH|EW_GRDM)_1SD[HV]_\d{8}T\d{6}_\d{8}T\d{6}.*\.SAFE$")
class Sentinel1Product(Product):
    """Ground-range-detected Sentinel-1, either acquisition mode.

    ``raw`` holds the calibrated backscatter (one band per polarisation);
    ``visual`` is the usual dB -> sigmoid stretch as a single greyscale band.
    ``metadata`` carries the ``geolocationGridPoint`` fields (``incidence_angle``,
    ``elevation_angle``, ``slant_range_time``, ``height``). :meth:`heading_to_los`
    and :meth:`doppler_azimuth_shift` estimate the SAR moving-target azimuth-shift
    effect for an object detected in the image, in pixels; :meth:`heading_in_image`
    re-expresses a compass heading in this GRD product's own (rotated) pixel frame;
    :meth:`correct_position` combines both to recover a moving target's true
    lat/lon from its as-detected position. SAFE parsing itself lives in
    :mod:`satimg.s1_utils`; the SAR geometry behind the last four methods lives
    in :mod:`satimg.sar_utils`.
    """

    def __init__(self, source: Source):
        super().__init__(source)
        meta = s1_utils.read_manifest(self._source)
        self._timestamp = meta["timestamp"]
        self._footprint = meta["footprint"]
        annotation = s1_utils.annotation_file(self._source)
        points, attrs, shape = s1_utils.read_geolocation(
            self._source, annotation, self._timestamp
        )
        self._geoloc_points = points
        self._geoloc_attrs = attrs
        self._geoloc_shape = shape
        self._transformer = s1_utils.build_transformer(points)

    @property
    def mode(self) -> str:
        m = re.search(r"_(IW|EW)_", os.path.basename(self._path))
        return m.group(1) if m else "IW"

    def _open_raw(self, tile: int | tuple[int, int]) -> xarray.DataArray:
        self._require_extracted("raw")
        measurement = os.path.join(self._path, "measurement")
        pols = {}
        for f in os.listdir(measurement):
            m = re.search(r"-(vv|vh|hh|hv)-", f)
            if m:
                pols[os.path.join(measurement, f)] = m.group(1)
        files = sorted(pols, key=lambda f: s1_utils.POL_ORDER[pols[f]])
        merged = merge_bands(files, tile=tile)
        da = label_bands(as_band_yx(merged), [pols[f].upper() for f in files])
        return keep_open(da, merged)

    def _render_visual(
        self, raw: xarray.DataArray, tile: int | tuple[int, int]
    ) -> xarray.DataArray:
        self._require_extracted("visual")
        db = (10 * numpy.log10(raw.where(raw > 0))).fillna(0).mean("band")
        u8 = (255 / (1 + numpy.exp(-((db - 20) * 0.18)))).clip(0, 255).astype("uint8")
        return label_bands(as_band_yx(u8), ("amplitude",))

    def _read_metadata(self) -> Metadata:
        keys = tuple(s1_utils.GEOLOC_FIELDS)
        grid = grid_from_points(self._geoloc_points, keys)
        fields = {
            name: self._field(
                grid["rows"], grid["cols"], grid[name],
                name=name, units=s1_utils.GEOLOC_FIELDS[name][1],
            )
            for name in keys
        }
        return Metadata(fields, self._geoloc_attrs)

    @property
    def height(self) -> int:
        """From the annotation XML's ``numberOfLines`` (matches ``raw``
        exactly) rather than the base ``int(self.raw.sizes["y"])`` -- so
        ``metadata``'s fields get a real ``.grid`` / ``.corners()`` even
        zip-native, with no ``raw`` open needed."""
        return self._geoloc_shape[0]

    @property
    def width(self) -> int:
        """See :attr:`height`; from ``numberOfSamples``."""
        return self._geoloc_shape[1]

    @property
    def transformer(self):
        return self._transformer

    @property
    def timestamp(self) -> datetime.datetime:
        return self._timestamp

    @property
    def footprint(self) -> dict:
        return self._footprint

    def thumbnail(self) -> PIL.Image.Image:
        """The shipped preview PNG, forced to greyscale: SAR quick-looks ship
        as an RGB polarimetric composite (e.g. VV/VH/ratio as R/G/B) on some
        products, but the plain, dark, single-band amplitude look -- matching
        :attr:`visual` -- is the more useful thumbnail for a SAR scene."""
        for name in ("thumbnail.png", "quick-look.png"):
            relpath = f"preview/{name}"
            if self._source.exists(relpath):
                return load_thumbnail(self._source, relpath, greyscale=True)
        raise FileNotFoundError(f"no preview image under {self._source.name}")

    def _local_platform_heading(self, rowcol):
        """Satellite ground-track heading at ``rowcol``'s own latitude (rather
        than the single scene-wide ``platform_heading`` attr), via
        :func:`satimg.sar_utils.ground_track_heading`."""
        lat = self.transformer.rowcol_to_latlon(rowcol)[:, 0]
        ascending = self.metadata.attrs["pass"].lower() == "ascending"
        heading = sar_utils.ground_track_heading(
            lat, self.metadata.attrs["orbit_inclination"], ascending
        )
        return float(heading[0]) if numpy.ndim(rowcol) == 1 else heading

    def heading_to_los(self, rowcol, heading_deg):
        """Convert a compass heading (degrees clockwise from true north) into the
        object's bearing relative to the radar line of sight at ``rowcol``.

        ``rowcol`` is a ``(row, col)`` pixel pair (or a list of pairs / an
        ``(N, 2)`` array, matching :meth:`~satimg.metadata.Field.at`) -- needed
        because the local satellite heading varies (slightly) with latitude across
        a scene. See :func:`satimg.sar_utils.heading_to_los` for the convention and
        equations.
        """
        rel = sar_utils.heading_to_los(heading_deg, self._local_platform_heading(rowcol))
        return float(rel) if numpy.ndim(rowcol) == 1 else numpy.asarray(rel)

    def heading_in_image(self, rowcol, heading_deg):
        """Convert a compass heading (degrees clockwise from true north) into
        this GRD product's own pixel frame, at pixel ``rowcol``.

        Unlike an orthorectified product, GRD imagery is still in native
        sensor geometry: row increases with azimuth time, i.e. towards the
        direction the platform is heading (see the sign convention in
        :func:`satimg.sar_utils.azimuth_shift_m`), so the image's "up"
        (decreasing row) points the *opposite* way -- the local platform
        heading plus 180 degrees. This is why an ascending-pass Sentinel-1
        GRD scene looks upside-down (south-up) relative to a map, and a
        descending-pass one looks right-side up.

        ``0``/``360`` means the object points towards the top of the image
        (decreasing row), ``90`` towards the right -- handy for e.g. drawing a
        detected ship's heading as an arrow directly on the raster. This is a
        SAR-only concern: a map-projected, north-up optical product needs no
        such conversion at all.
        """
        up_heading = (self._local_platform_heading(rowcol) + 180.0) % 360.0
        result = sar_utils.heading_in_image(heading_deg, up_heading)
        return float(result) if numpy.ndim(rowcol) == 1 else numpy.asarray(result)

    def doppler_azimuth_shift(self, rowcol, speed: float, heading_deg: float):
        """Azimuth-direction pixel displacement of a moving object at ``rowcol``.

        ``rowcol`` is a ``(row, col)`` pixel pair (or a list of pairs / an ``(N, 2)``
        array, matching :meth:`~satimg.metadata.Field.at`), ``speed`` is the
        object's ground speed in m/s, and ``heading_deg`` its compass heading in
        degrees clockwise from true north. Returns the estimated shift in pixels
        along the azimuth (row) axis -- positive towards higher row indices. See
        :func:`satimg.sar_utils.azimuth_shift_m` for the underlying physics and sign
        convention.
        """
        m = self.metadata
        attrs = m.attrs
        incidence = m.incidence_angle.at(rowcol)
        slant_range_m = m.slant_range_time.at(rowcol) * sar_utils.SPEED_OF_LIGHT / 2
        shift_m = sar_utils.azimuth_shift_m(
            speed, heading_deg, self._local_platform_heading(rowcol), incidence,
            slant_range_m, attrs["platform_velocity"],
        )
        shift_px = shift_m / attrs["azimuth_pixel_spacing"]
        return float(shift_px) if numpy.ndim(rowcol) == 1 else numpy.asarray(shift_px)

    def correct_position(self, lat: float, lon: float, speed: float, heading_deg: float):
        """Recover a moving target's true ``(lat, lon)`` from its as-detected
        position in this GRD product.

        ``lat``/``lon`` is where the target was read off the image, and ``speed``
        (m/s) / ``heading_deg`` (degrees clockwise from true north) are the same
        inputs as :meth:`doppler_azimuth_shift`. The target's own motion displaces
        it along the azimuth (row) axis of the focused image by
        :meth:`doppler_azimuth_shift`'s pixel shift; this undoes that displacement
        to recover the position the target actually occupied at acquisition time.
        """
        rowcol = self.transformer.latlon_to_rowcol((lat, lon))[0]
        shift_px = self.doppler_azimuth_shift(rowcol, speed, heading_deg)
        corrected_rowcol = (rowcol[0] - shift_px, rowcol[1])
        corrected_lat, corrected_lon = self.transformer.rowcol_to_latlon(corrected_rowcol)[0]
        return float(corrected_lat), float(corrected_lon)
