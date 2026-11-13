"""Sentinel-1 SAFE parsing: manifest, annotation XML, geolocation grid, and
the GCP transformer built from it.

Pure functions taking a :class:`~satimg.source.source.Source` (or already-parsed XML)
and returning plain dicts/lists/a :class:`~satimg.transform.gcp_transformer.GCPTransformer` --
no dependency on :class:`~satimg.product.Product`, so they're testable
independently of a full product open. See
:class:`~satimg.products.sentinel1.grd.sentinel1_grd.Sentinel1GRD` for the product-facing
entry points that call these.
"""

from __future__ import annotations

import datetime
import re
from xml.etree import ElementTree

from rasterio.control import GroundControlPoint
from rasterio.crs import CRS

from satimg.metadata.grids import grid_from_points
from satimg.source.source import Source
from satimg.transform.gcp_transformer import GCPTransformer

#: measurement-file polarisation -> sort order (co-pol before cross-pol).
POL_ORDER = {"vv": 0, "vh": 1, "hh": 2, "hv": 3}

GEOLOC_FIELDS = {
    "incidence_angle": ("incidenceAngle", "degrees"),
    "elevation_angle": ("elevationAngle", "degrees"),
    "slant_range_time": ("slantRangeTime", "seconds"),
    "height": ("height", "metres"),
}

_NS = {
    "safe": "http://www.esa.int/safe/sentinel-1.0",
    "gml": "http://www.opengis.net/gml",
}

def read_manifest(source: Source) -> dict:
    manifest = source.find_file("manifest.safe")
    if manifest is None:
        raise FileNotFoundError(f"no manifest.safe under {source.name}")
    with source.open(manifest) as f:
        root = ElementTree.parse(f).getroot()
    start = datetime.datetime.strptime(
        root.find(".//safe:startTime", _NS).text, "%Y-%m-%dT%H:%M:%S.%f"
    )
    stop = datetime.datetime.strptime(
        root.find(".//safe:stopTime", _NS).text, "%Y-%m-%dT%H:%M:%S.%f"
    )
    coords = [
        (float(lon), float(lat))
        for pair in root.find(".//gml:coordinates", _NS).text.split()
        for lat, lon in [pair.split(",")]
    ]
    return {
        "timestamp": start + (stop - start) / 2,
        "footprint": {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coords]},
        },
    }


def annotation_file(source: Source) -> str:
    """Root-relative path of the first product-annotation XML (the
    ``calibration/`` ones are excluded)."""
    files = sorted(f for f in source.listdir("annotation") if f.endswith(".xml"))
    if not files:
        raise FileNotFoundError(f"no annotation XML under {source.name}/annotation")
    return f"annotation/{files[0]}"


def orbit(root, epoch: datetime.datetime) -> dict:
    """The sensor orbit as
    :class:`~satimg.sar_orbit_geometry.SAROrbitGeometry` reads it: the
    annotation's ``<orbit>`` state vectors (Earth-fixed, every 10 s, over a
    few minutes around the scene), timed from ``epoch``."""
    states = sorted(
        (
            (datetime.datetime.fromisoformat(o.findtext("time")) - epoch).total_seconds(),
            [float(o.findtext(f"position/{c}")) for c in "xyz"],
            [float(o.findtext(f"velocity/{c}")) for c in "xyz"],
        )
        for o in root.findall(".//orbit")
    )
    return {
        "epoch": epoch.isoformat() + "Z",
        "times": [s[0] for s in states],
        "positions": [s[1] for s in states],
        "velocities": [s[2] for s in states],
    }


def read_geolocation(
    source: Source, relpath: str, epoch: datetime.datetime
) -> tuple[list[dict], dict, tuple[int, int]]:
    """The annotation XML's geolocation grid points (row/col plus each
    :data:`GEOLOC_FIELDS` value, and -- for building GCPs, not exposed in
    :class:`~satimg.metadata.metadata.Metadata` -- latitude/longitude/height), the
    scene-wide attrs (``orbit`` timed from ``epoch``, see :func:`orbit`), and
    the full measurement grid's ``(lines, samples)`` shape
    (``imageAnnotation/imageInformation``)."""
    with source.open(relpath) as f:
        root = ElementTree.parse(f).getroot()
    points = []
    for gp in root.findall(".//geolocationGridPoint"):
        pt = {
            "row": int(gp.findtext("line")),
            "col": int(gp.findtext("pixel")),
            "latitude": float(gp.findtext("latitude")),
            "longitude": float(gp.findtext("longitude")),
            "height": float(gp.findtext("height")),
        }
        for name, (tag, _units) in GEOLOC_FIELDS.items():
            pt[name] = float(gp.findtext(tag))
        points.append(pt)
    attrs = {
        "incidence_angle_mid_swath": float(root.findtext(".//incidenceAngleMidSwath")),
        "platform_heading": float(root.findtext(".//platformHeading")),
        "pass": root.findtext(".//pass"),
        "azimuth_pixel_spacing": float(root.findtext(".//azimuthPixelSpacing")),
        "orbit": orbit(root, epoch),
    }
    shape = (
        int(root.findtext(".//numberOfLines")),
        int(root.findtext(".//numberOfSamples")),
    )
    return points, attrs, shape


def build_transformer(points: list[dict]) -> GCPTransformer:
    """A :class:`~satimg.transform.gcp_transformer.GCPTransformer` (WGS84) from
    :func:`read_geolocation`'s geolocation grid points."""
    gcps = [
        GroundControlPoint(
            row=p["row"], col=p["col"], x=p["longitude"], y=p["latitude"], z=p["height"],
        )
        for p in points
    ]
    return GCPTransformer(gcps, CRS.from_epsg(4326))


def build_fields(points: list[dict]) -> dict:
    """:func:`read_geolocation`'s grid points as
    :class:`~satimg.scene_geometry.SceneGeometry` fields: each
    :data:`GEOLOC_FIELDS` value on the geolocation grid."""
    grid = grid_from_points(points, tuple(GEOLOC_FIELDS))
    return {name: (grid["rows"], grid["cols"], grid[name], units) for name, (_tag, units) in GEOLOC_FIELDS.items()}


#: ``S1A_IW_GRDH_1SDV_<start>_<stop>_<orbit>_<datatake>_<product unique id>[_COG][.SAFE]``
_SCENE_NAME = re.compile(r"^(S1[A-D]_.+_[0-9A-F]{6})_[0-9A-F]{4}(?:_COG)?(?:\.SAFE)?$")


def scene_key(name: str) -> str | None:
    """``name`` without the product unique id (and ``_COG`` / ``.SAFE``), which
    change every time ESA reprocesses the scene -- or None if ``name`` is not a
    Sentinel-1 product name."""
    match = _SCENE_NAME.match(name)
    return match.group(1) if match else None
