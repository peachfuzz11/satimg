"""Sentinel-2 Level-1C reader."""

from __future__ import annotations

import datetime
import os
import re
import warnings
from functools import cached_property
from xml.etree import ElementTree

import numpy
import rioxarray
import xarray
from rasterio.crs import CRS
from rasterio.transform import Affine

from satimg.metadata import Metadata, fill_nan_nearest, regular_axis
from satimg.product import Product
from satimg.readers import keep_open, merge_bands
from satimg.registry import register
from satimg.source import Source
from satimg.tiling import as_band_yx, label_bands
from satimg.transform import Transformer

#: spacing of the MTD_TL.xml angle grids, metres.
_ANGLE_STEP_M = 5000.0

#: Sentinel-2 L1C per-band native ground sample distance, metres -- picks the
#: ``Tile_Geocoding/Geoposition`` (and matching ``Size``) that
#: ``band_names[1]`` (the :func:`~satimg.readers.merge_bands` ``match=``
#: target) was actually shot at, so the zip-native transformer/grid-shape
#: agree exactly with opening that band's own JP2.
_BAND_GSD_M = {
    "B01": 60, "B02": 10, "B03": 10, "B04": 10, "B05": 20, "B06": 20,
    "B07": 20, "B08": 10, "B8A": 20, "B09": 60, "B10": 60, "B11": 20, "B12": 20,
}


def _pad_band(name: str) -> str:
    """``B1`` -> ``B01``; leaves ``B8A`` / ``B10`` .. ``B12`` unchanged."""
    return re.sub(r"^B(\d)$", r"B0\1", name)


def _parse_mtd(source: Source, path: str) -> dict:
    """``path`` (a real filesystem path, unlike ``source``) is only used to
    build ``reflectance``/``tci``'s raster paths -- raster-only, so still a
    plain path even when ``source`` is zip-native (where those two entries
    are then unused, since ``raw``/``visual`` are guarded off)."""
    mtd = source.find_file("MTD_MSIL1C.xml")
    if mtd is None:
        raise FileNotFoundError(f"no MTD_MSIL1C.xml under {source.name}")
    with source.open(mtd) as f:
        root = ElementTree.parse(f).getroot()

    physical, wavelengths = {}, {}
    for info in root.findall(".//Spectral_Information"):
        bid = int(info.get("bandId"))
        physical[bid] = info.get("physicalBand")
        central = info.find(".//CENTRAL")
        if central is not None:
            wavelengths[bid] = float(central.text)

    reflectance, tci = [], None
    for band_id, elem in enumerate(root.iter("IMAGE_FILE")):
        file_path = os.path.join(path, elem.text) + ".jp2"
        if physical.get(band_id) is None:
            tci = file_path
        else:
            reflectance.append(file_path)

    n = len(reflectance)
    band_names = [_pad_band(physical[i]) for i in range(n)]
    band_wavelengths = [wavelengths.get(i, numpy.nan) for i in range(n)]

    ext = [float(v) for v in root.find(".//EXT_POS_LIST").text.split()]
    ring = [(ext[i + 1], ext[i]) for i in range(0, len(ext), 2)]
    return {
        "reflectance": reflectance,
        "band_names": band_names,
        "wavelengths": band_wavelengths,
        "tci": tci,
        "footprint": {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        },
    }


def _angle_grid(node: ElementTree.Element) -> numpy.ndarray:
    """The ``Values_List`` of a Zenith/Azimuth node as a 2-D array (``NaN`` ok)."""
    rows = node.find("Values_List").findall("VALUES")
    return numpy.array([[float(v) for v in row.text.split()] for row in rows])


def _mean_grids(grids: list[numpy.ndarray], *, circular: bool) -> numpy.ndarray:
    """NaN-aware mean of the per-detector grids; ``circular`` for azimuths. Cells
    that no detector covers (tile corners) are then nearest-filled."""
    stack = numpy.stack(grids)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices
        if not circular:
            merged = numpy.nanmean(stack, axis=0)
        else:
            rad = numpy.deg2rad(stack)
            merged = numpy.rad2deg(
                numpy.arctan2(numpy.nanmean(numpy.sin(rad), 0), numpy.nanmean(numpy.cos(rad), 0))
            ) % 360.0
    return fill_nan_nearest(merged)


def _parse_tl(source: Source) -> dict:
    """Acquisition time, the sun / viewing angle grids, and the
    ``Tile_Geocoding`` CRS + per-resolution affine transform / pixel-grid
    shape (for the zip-native transformer/metadata-grid-shape, in place of
    opening a band's own JP2), all from ``MTD_TL.xml``."""
    tl = source.find_file("MTD_TL.xml")
    if tl is None:
        raise FileNotFoundError(f"no MTD_TL.xml under {source.name}")
    with source.open(tl) as f:
        root = ElementTree.parse(f).getroot()

    timestamp = datetime.datetime.strptime(
        root.find(".//SENSING_TIME").text, "%Y-%m-%dT%H:%M:%S.%fZ"
    )

    geocoding = root.find(".//Tile_Geocoding")
    crs = CRS.from_string(geocoding.find("HORIZONTAL_CS_CODE").text)
    transforms = {
        int(g.get("resolution")): Affine(
            float(g.find("XDIM").text), 0, float(g.find("ULX").text),
            0, float(g.find("YDIM").text), float(g.find("ULY").text),
        )
        for g in geocoding.findall("Geoposition")
    }
    pixel_shapes = {
        int(s.get("resolution")): (int(s.find("NROWS").text), int(s.find("NCOLS").text))
        for s in geocoding.findall("Size")
    }

    pixel_m = min(transforms)
    step_px = _ANGLE_STEP_M / pixel_m

    sun = root.find(".//Sun_Angles_Grid")
    grids = {
        "sun_zenith": _angle_grid(sun.find("Zenith")),
        "sun_azimuth": _angle_grid(sun.find("Azimuth")),
    }
    view = root.findall(".//Viewing_Incidence_Angles_Grids")
    grids["view_zenith"] = _mean_grids(
        [_angle_grid(g.find("Zenith")) for g in view], circular=False
    )
    grids["view_azimuth"] = _mean_grids(
        [_angle_grid(g.find("Azimuth")) for g in view], circular=True
    )

    shape = grids["sun_zenith"].shape
    axes = (regular_axis(shape[0], step_px), regular_axis(shape[1], step_px))

    mean_sun = root.find(".//Mean_Sun_Angle")
    attrs = {
        "mean_sun_zenith": float(mean_sun.find("ZENITH_ANGLE").text),
        "mean_sun_azimuth": float(mean_sun.find("AZIMUTH_ANGLE").text),
    }
    return {
        "timestamp": timestamp, "axes": axes, "grids": grids, "attrs": attrs,
        "crs": crs, "transforms": transforms, "pixel_shapes": pixel_shapes,
    }


@register(r"^S2[ABCD]_MSIL1C_\d{8}T\d{6}_.*_\d{8}T\d{6}\.SAFE$")
class Sentinel2L1CProduct(Product):
    """``raw`` is the 13 reflectance bands resampled to the 10 m grid; ``visual``
    is the shipped 3-band True Colour Image. ``metadata`` carries the ``MTD_TL.xml``
    angle grids (``sun_zenith``, ``sun_azimuth``, ``view_zenith``, ``view_azimuth``);
    the viewing grids are the per-detector grids merged."""

    def __init__(self, path: str, source: Source | None = None):
        super().__init__(path, source)
        self._meta = _parse_mtd(self._source, self._path)
        self._tl = _parse_tl(self._source)
        self._timestamp = self._tl["timestamp"]
        self._gsd = _BAND_GSD_M[self._meta["band_names"][1]]
        self._transformer = Transformer(self._tl["transforms"][self._gsd], self._tl["crs"])

    def _open_raw(self, tile: int | tuple[int, int]) -> xarray.DataArray:
        self._require_extracted("raw")
        merged = merge_bands(self._meta["reflectance"], match=1, tile=tile)
        da = label_bands(as_band_yx(merged), self._meta["band_names"])
        da = da.assign_coords(wavelength_nm=("band", self._meta["wavelengths"]))
        return keep_open(da, merged)

    def _render_visual(
        self, raw: xarray.DataArray, tile: int | tuple[int, int]
    ) -> xarray.DataArray:
        self._require_extracted("visual")
        src = rioxarray.open_rasterio(self._meta["tci"])
        tw, th = (tile, tile) if isinstance(tile, int) else tile
        # a plain (unchunked) DataArray's .astype() computes eagerly -- chunk
        # first so this stays lazy and windowed like every other view.
        tci = as_band_yx(src.chunk({"x": tw, "y": th}).astype("uint8"))
        return keep_open(label_bands(tci, ("red", "green", "blue")), src)

    def _read_metadata(self) -> Metadata:
        rows, cols = self._tl["axes"]
        fields = {
            name: self._field(rows, cols, grid, name=name, units="degrees")
            for name, grid in self._tl["grids"].items()
        }
        return Metadata(fields, self._tl["attrs"])

    def _metadata_grid_shape(self) -> tuple[int, int]:
        return self._tl["pixel_shapes"][self._gsd]

    @property
    def transformer(self) -> Transformer:
        return self._transformer

    @property
    def timestamp(self) -> datetime.datetime:
        return self._timestamp

    @property
    def footprint(self) -> dict:
        return self._meta["footprint"]

    def thumbnail(self):
        import PIL.Image

        name = next(o for o in self._source.listdir() if o.endswith("-ql.jpg"))
        with self._source.open(name) as f:
            img = PIL.Image.open(f)
            img.load()
            return img
