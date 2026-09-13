"""Sentinel-2 Level-1C reader."""

from __future__ import annotations

import datetime
import os
import re
import warnings
from functools import cached_property
from xml.etree import ElementTree

import numpy
import pyproj
import rioxarray
import xarray
from affine import Affine

from satimg.metadata import Metadata, fill_nan_nearest, regular_axis
from satimg.product import Product
from satimg.readers import find_file, keep_open, merge_bands
from satimg.registry import register
from satimg.tiling import as_band_yx, label_bands
from satimg.transform import Transformer

#: spacing of the MTD_TL.xml angle grids, metres.
_ANGLE_STEP_M = 5000.0


def _pad_band(name: str) -> str:
    """``B1`` -> ``B01``; leaves ``B8A`` / ``B10`` .. ``B12`` unchanged."""
    return re.sub(r"^B(\d)$", r"B0\1", name)


def _parse_mtd(path: str) -> dict:
    mtd = find_file(path, "MTD_MSIL1C.xml")
    if mtd is None:
        raise FileNotFoundError(f"no MTD_MSIL1C.xml under {path}")
    root = ElementTree.parse(mtd).getroot()

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


def _geo_transform(root: ElementTree.Element, resolution: str = "10") -> tuple[Affine, pyproj.CRS]:
    """Affine transform + CRS of the ``resolution`` m grid, from ``MTD_TL.xml``'s
    ``<Tile_Geocoding>`` block -- no raster file needs to be opened."""
    geo = root.find(".//Tile_Geocoding")
    pos = next(g for g in geo.findall("Geoposition") if g.get("resolution") == resolution)
    ulx, uly = float(pos.find("ULX").text), float(pos.find("ULY").text)
    xdim, ydim = float(pos.find("XDIM").text), float(pos.find("YDIM").text)
    epsg = geo.findtext("HORIZONTAL_CS_CODE")
    return Affine(xdim, 0, ulx, 0, ydim, uly), pyproj.CRS.from_user_input(epsg)


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
    transform, crs = _geo_transform(root)

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
    return {
        "timestamp": timestamp, "transform": transform, "crs": crs,
        "axes": axes, "grids": grids, "attrs": attrs,
    }


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
        self._transformer = Transformer(self._tl["transform"], self._tl["crs"])

    @cached_property
    def raw(self) -> xarray.DataArray:
        merged = merge_bands(self._meta["reflectance"], match=1)
        da = label_bands(as_band_yx(merged), self._meta["band_names"])
        da = da.assign_coords(wavelength_nm=("band", self._meta["wavelengths"]))
        return keep_open(da, merged)

    def _render_visual(self) -> xarray.DataArray:
        src = rioxarray.open_rasterio(self._meta["tci"])
        tci = as_band_yx(src.astype("uint8"))
        return keep_open(label_bands(tci, ("red", "green", "blue")), src)

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