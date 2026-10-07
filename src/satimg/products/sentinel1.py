"""Sentinel-1 GRD (IW and EW) reader."""

from __future__ import annotations

import datetime
import os
import re

import numpy
import PIL.Image
import xarray

from satimg import s1_utils
from satimg.metadata import Metadata
from satimg.product import Product
from satimg.readers import keep_open, load_thumbnail, merge_bands
from satimg.registry import register
from satimg.s1_geometry import Sentinel1Geometry
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
    re-expresses a compass heading in this GRD product's own (rotated and mirrored) pixel frame;
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
        self._geometry = Sentinel1Geometry.from_points(points, attrs, shape)

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
        return self._geometry.metadata

    @property
    def height(self) -> int:
        """From the annotation XML's ``numberOfLines`` (matches ``raw``
        exactly) rather than the base ``int(self.raw.sizes["y"])`` -- so
        ``metadata``'s fields get a real ``.grid`` / ``.corners()`` even
        zip-native, with no ``raw`` open needed."""
        return self._geometry.height

    @property
    def width(self) -> int:
        """See :attr:`height`; from ``numberOfSamples``."""
        return self._geometry.width

    @property
    def transformer(self):
        return self._geometry.transformer

    @property
    def geometry(self) -> Sentinel1Geometry:
        """As :attr:`Product.geometry`, as a
        :class:`~satimg.s1_geometry.Sentinel1Geometry` -- which the
        transformer, metadata and SAR geometry methods below delegate to."""
        return self._geometry

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

    def heading_to_los(self, rowcol, heading_deg):
        """Convert a compass heading (degrees clockwise from true north) into the
        object's bearing relative to the radar line of sight at ``rowcol``.

        ``rowcol`` is a ``(row, col)`` pixel pair (or a list of pairs / an
        ``(N, 2)`` array, matching :meth:`~satimg.metadata.Field.at`) -- needed
        because the local satellite heading varies (slightly) with latitude across
        a scene. See :func:`satimg.sar_utils.heading_to_los` for the convention and
        equations.
        """
        return self._geometry.heading_to_los(rowcol, heading_deg)

    def heading_in_image(self, rowcol, heading_deg):
        """Convert a compass heading (degrees clockwise from true north) into
        this GRD product's own pixel frame, at pixel ``rowcol``.

        Unlike an orthorectified product, GRD imagery is still in native
        sensor geometry: row increases with azimuth time, i.e. towards the
        direction the platform is heading (see the sign convention in
        :func:`satimg.sar_utils.azimuth_shift_m`), so the image's "up"
        (decreasing row) points the *opposite* way -- the local platform
        heading plus 180 degrees. Column increases with ground range, i.e.
        towards the look direction, which for the right-looking Sentinel-1 is
        the platform heading plus 90 degrees -- so "right" sits *counter-
        clockwise* of "up", where on a map it sits clockwise. The image is
        therefore a **mirror image** of the map, not merely a rotated one:
        an ascending-pass scene is flipped top-to-bottom (north at the bottom,
        east on the right) and a descending-pass one left-to-right (north at
        the top, east on the left). Headings are converted accordingly, sense
        of rotation reversed.

        ``0``/``360`` means the object points towards the top of the image
        (decreasing row), ``90`` towards the right -- handy for e.g. drawing a
        detected ship's heading as an arrow directly on the raster. This is a
        SAR-only concern: a map-projected, north-up optical product needs no
        such conversion at all.
        """
        return self._geometry.heading_in_image(rowcol, heading_deg)

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
        return self._geometry.doppler_azimuth_shift(rowcol, speed, heading_deg)

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
        return self._geometry.correct_position(lat, lon, speed, heading_deg)
