"""Sentinel-2 Level-1C reader."""

from __future__ import annotations

import datetime
import os
import warnings
from functools import cached_property
from xml.etree import ElementTree

import numpy
import rasterio
import xarray

from satimg.metadata import Metadata, fill_nan_nearest, regular_axis
from satimg.product import Product
from satimg.raster import Raster
from satimg.readers import find_file, merge_bands, open_band
from satimg.registry import register
from satimg.transform import Transformer

#: spacing of the MTD_TL.xml angle grids, metres.
_ANGLE_STEP_M = 5000.0


def _parse_mtd(path: str) -> dict:
    mtd = find_file(path, "MTD_MSIL1C.xml")
    if mtd is None:
        raise FileNotFoundError(f"no MTD_MSIL1C.xml under {path}")
    root = ElementTree.parse(mtd).getroot()

    physical = {}
    for info in root.findall(".//Spectral_Information"):
        physical[int(info.get("bandId"))] = info.get("physicalBand")

    reflectance, tci = [], None
    for band_id, elem in enumerate(root.iter("IMAGE_FILE")):
        file_path = os.path.join(path, elem.text) + ".jp2"
        if physical.get(band_id) is None:
            tci = file_path
        else:
            reflectance.append(file_path)

    ext = [float(v) for v in root.find(".//EXT_POS_LIST").text.split()]
    ring = [(ext[i + 1], ext[i]) for i in range(0, len(ext), 2)]
    return {
        "reflectance": reflectance,
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


def _parse_tl(path: str) -> dict:
    """Acquisition time + the sun / viewing angle grids from ``MTD_TL.xml``."""
    tl = find_file(path, "MTD_TL.xml")
    if tl is None:
        raise FileNotFoundError(f"no MTD_TL.xml under {path}")
    root = ElementTree.parse(tl).getroot()

    timestamp = datetime.datetime.strptime(
        root.find(".//SENSING_TIME").text, "%Y-%m-%dT%H:%M:%S.%fZ"
    )

    pixel_m = min(
        abs(float(g.find("XDIM").text))
        for g in root.iter("Geoposition")
    )
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
    return {"timestamp": timestamp, "axes": axes, "grids": grids, "attrs": attrs}


@register(r"^S2[ABCD]_MSIL1C_\d{8}T\d{6}_.*_\d{8}T\d{6}\.SAFE$")
class Sentinel2L1CProduct(Product):
    """``raw`` is the 13 reflectance bands resampled to the 10 m grid; ``visual``
    is the shipped 3-band True Colour Image. ``metadata`` carries the ``MTD_TL.xml``
    angle grids (``sun_zenith``, ``sun_azimuth``, ``view_zenith``, ``view_azimuth``);
    the viewing grids are the per-detector grids merged."""

    def __init__(self, path: str):
        super().__init__(path)
        self._meta = _parse_mtd(path)
        self._tl = _parse_tl(path)
        self._timestamp = self._tl["timestamp"]
        with rasterio.open(self._meta["reflectance"][1]) as src:
            self._transformer = Transformer(src.transform, src.crs)

    @cached_property
    def raw(self) -> Raster:
        da = merge_bands(self._meta["reflectance"], match=1)
        return Raster(da, transform=self._transformer, name="reflectance")

    def _render_visual(self) -> Raster:
        tci = open_band(self._meta["tci"]).astype("uint8")
        return Raster(tci, transform=self._transformer, name="tci")

    def _read_metadata(self) -> Metadata:
        rows, cols = self._tl["axes"]
        fields = {
            name: self._field(rows, cols, grid, name=name, units="degrees")
            for name, grid in self._tl["grids"].items()
        }
        return Metadata(fields, self._tl["attrs"])

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

        name = next(o for o in os.listdir(self._path) if o.endswith("-ql.jpg"))
        return PIL.Image.open(os.path.join(self._path, name))
