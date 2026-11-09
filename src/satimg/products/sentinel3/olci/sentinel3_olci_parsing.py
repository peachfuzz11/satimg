"""Sentinel-3 OLCI Level-1B full resolution (``OL_1_EFR___``) parsing.

A product is a ``.SEN3`` folder of netCDF4 files: ``Oa01_radiance.nc`` ..
``Oa21_radiance.nc`` (uint16, ``scale_factor`` per band, mW / m^2 / sr /
nm), ``tie_geo_coordinates.nc`` / ``tie_geometries.nc`` (latitude /
longitude and sun / view angles on every row and every
``ac_subsampling_factor``-th column) and ``xfdumanifest.xml``.

Geolocation comes from the tie points, not ``geo_coordinates.nc``: the
per-pixel positions there are terrain corrected -- displaced by hundreds of
metres over mountains, which no smooth interpolation follows (a node grid
chasing them grew to millions of nodes) -- while the tie points are smooth.
The two agree exactly where the altitude is zero; at sea, where the
per-pixel altitude is the geoid (~20 m), they sit ~100 m apart (median, a
third of a pixel), more than parallax explains. Checked against the
coastline on a real product, both agree with the land mask equally well
(75.2 % vs 74.7 % of clear coastal pixels), so the tie points are kept: fast,
smooth, and a few hundred bytes of nodes.
"""

from __future__ import annotations

import datetime
import re
from xml.etree import ElementTree

import numpy

from satimg.products import swath_nodes
from satimg.readers.hdf5 import open_h5, read_scaled
from satimg.source.source import Source

BANDS = tuple(f"Oa{i:02d}" for i in range(1, 22))

#: central wavelength (nm) of each band, OLCI's nominal band set
WAVELENGTHS_NM = (
    400.0, 412.5, 442.5, 490.0, 510.0, 560.0, 620.0, 665.0, 673.75, 681.25, 708.75,
    753.75, 761.25, 764.375, 767.5, 778.75, 865.0, 885.0, 900.0, 940.0, 1020.0,
)

#: nominal solar irradiance (mW / m^2 / nm) for the visual's true colour bands
SOLAR_IRRADIANCE = {"Oa04": 1916.0, "Oa06": 1824.0, "Oa08": 1525.0}

#: tie_geometries variable -> metadata field
ANGLES = {
    "SZA": "solar_zenith",
    "SAA": "solar_azimuth",
    "OZA": "satellite_zenith",
    "OAA": "satellite_azimuth",
}

#: rows kept of the every-row tie-point grid for the metadata fields
TIE_ROW_STEP = 64

_NAME = re.compile(
    r"^(?P<platform>S3[A-D])_OL_1_EFR____(?P<start>\d{8}T\d{6})_(?P<stop>\d{8}T\d{6})_\d{8}T\d{6}"
    r"_\d{4}_(?P<cycle>\d{3})_(?P<orbit>\d{3})_(?P<frame>\d{4})_(?P<centre>\w{3})_O_(?P<timeliness>NR|ST|NT)_\d{3}"
)

def band_file(band: str) -> str:
    return f"{band}_radiance.nc"


def gdal_path(raster_path: str, band: str) -> str:
    return f'HDF5:"{raster_path}"://{band}_radiance'


def read_name(name: str) -> dict:
    m = _NAME.match(name)
    if not m:
        return {}
    return {
        "satellite": f"sentinel-3{m['platform'][-1].lower()}",
        "cycle": int(m["cycle"]),
        "relative_orbit": int(m["orbit"]),
        "frame": int(m["frame"]),
        "timeliness": {"NR": "nrt", "ST": "stc", "NT": "ntc"}[m["timeliness"]],
    }


def read_footprint(source: Source) -> dict | None:
    """The manifest's ``gml:posList`` footprint (lat lon pairs) as GeoJSON."""
    manifest = source.find_file("xfdumanifest.xml")
    if manifest is None:
        return None
    with source.open(manifest) as f:
        root = ElementTree.parse(f).getroot()
    for el in root.iter():
        if el.tag.endswith("posList") and el.text:
            values = [float(v) for v in el.text.split()]
            ring = [(lon, lat) for lat, lon in zip(values[0::2], values[1::2])]
            if ring[0] != ring[-1]:
                ring.append(ring[0])
            return {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]}}
    return None


def _time(value) -> datetime.datetime:
    text = value.decode() if isinstance(value, bytes) else str(value)
    return datetime.datetime.fromisoformat(text.rstrip("Z"))


def read_times(h5) -> tuple[datetime.datetime, datetime.datetime]:
    return _time(numpy.asarray(h5.attrs["start_time"]).ravel()[0]), _time(numpy.asarray(h5.attrs["stop_time"]).ravel()[0])


def read_scales(source: Source, raster_dir: str) -> tuple[list[float], list[float]]:
    """Each band's ``scale_factor`` / ``add_offset``, in :data:`BANDS` order."""
    scales, offsets = [], []
    for band in BANDS:
        with open_h5(source, f"{raster_dir}{band_file(band)}") as h5:
            attrs = h5[f"{band}_radiance"].attrs
            scales.append(float(numpy.ravel(attrs["scale_factor"])[0]))
            offsets.append(float(numpy.ravel(attrs.get("add_offset", [0.0]))[0]))
    return scales, offsets


def read_shape(h5) -> tuple[int, int]:
    """The image's ``(rows, columns)`` from a radiance file's dataset."""
    name = next(n for n in h5 if n.endswith("_radiance"))
    return tuple(h5[name].shape)


def read_geolocation(h5) -> dict:
    """The tie-point (ellipsoid) latitude / longitude from
    ``tie_geo_coordinates.nc``, scaled, fills ``NaN``, with the tie grid's
    pixel-centre ``rows`` / ``cols`` (tie column ``k`` is image column ``k x
    ac_subsampling_factor`` -- the file's own ``tie_columns`` is not filled
    in)."""
    step_ac = int(numpy.ravel(h5.attrs["ac_subsampling_factor"])[0])
    step_al = int(numpy.ravel(h5.attrs["al_subsampling_factor"])[0])
    lat, lon = read_scaled(h5["latitude"]), read_scaled(h5["longitude"])
    return {
        "lat": lat, "lon": lon,
        "rows": numpy.arange(lat.shape[0]) * step_al + 0.5,
        "cols": numpy.arange(lat.shape[1]) * step_ac + 0.5,
    }


def nodes(geolocation: dict, tolerance_m: float) -> dict:
    """A :class:`~satimg.transform.swath_transformer.SwathTransformer` node grid: every tie
    column, and the fewest tie rows between which linear interpolation down
    each column stays within ``tolerance_m``. No scans: OLCI is a
    push-broom."""
    lat, lon = geolocation["lat"], geolocation["lon"]
    keep = swath_nodes.node_columns(lat.T, lon.T, tolerance_m)
    return {"rows": geolocation["rows"][keep], "cols": geolocation["cols"], "lat": lat[keep], "lon": lon[keep]}


def read_angles(h5) -> dict:
    """The tie-point angles on every :data:`TIE_ROW_STEP`-th row (and the
    last), as ``{"rows", "cols", field: values}`` in pixel-centre
    coordinates; tie column ``k`` is image column ``k x ac_subsampling_factor``
    (the file's own ``tie_columns`` coordinate is not filled in)."""
    step_ac = int(numpy.ravel(h5.attrs["ac_subsampling_factor"])[0])
    step_al = int(numpy.ravel(h5.attrs["al_subsampling_factor"])[0])
    n_rows, n_cols = h5["SZA"].shape
    keep = numpy.unique(numpy.r_[numpy.arange(0, n_rows, TIE_ROW_STEP), n_rows - 1])
    out = {"rows": keep * step_al + 0.5, "cols": numpy.arange(n_cols) * step_ac + 0.5}
    for variable, field in ANGLES.items():
        out[field] = read_scaled(h5[variable])[keep]
    return out
