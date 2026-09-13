"""Sentinel-1 SAFE parsing: manifest, annotation XML, geolocation grid, and
the GCP transformer built from it.

Pure functions taking a :class:`~satimg.source.Source` (or already-parsed XML)
and returning plain dicts/lists/a :class:`~satimg.transform.GCPTransformer` --
no dependency on :class:`~satimg.product.Product`, so they're testable
independently of a full product open. See
:class:`~satimg.products.sentinel1.Sentinel1Product` for the product-facing
entry points that call these.
"""

from __future__ import annotations

import datetime
from xml.etree import ElementTree

import numpy
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS

from satimg.source import Source
from satimg.transform import GCPTransformer

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

#: Sentinel-1's sun-synchronous orbit inclination, degrees -- actively maintained,
#: so effectively constant across the whole constellation and mission (ESA mission
#: documentation). Used to derive the local ground-track heading at a target's own
#: latitude; see :func:`satimg.sar_utils.ground_track_heading`.
ORBIT_INCLINATION_DEG = 98.1813


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


def _nearest_orbit_velocity(root, at: datetime.datetime) -> float:
    """Platform speed (m/s) from the ``<orbit>`` state vector closest to ``at``.
    Sentinel-1's orbit is near-circular, so a single vector's speed is
    representative of the whole (~25s) scene -- no need to interpolate."""
    best_dt, best_velocity = None, None
    for orb in root.findall(".//orbit"):
        t = datetime.datetime.fromisoformat(orb.findtext("time"))
        dt = abs((t - at).total_seconds())
        if best_dt is None or dt < best_dt:
            velocity = numpy.array([float(orb.findtext(f"velocity/{c}")) for c in "xyz"])
            best_dt, best_velocity = dt, velocity
    return float(numpy.linalg.norm(best_velocity))


def read_geolocation(
    source: Source, relpath: str, at: datetime.datetime
) -> tuple[list[dict], dict, tuple[int, int]]:
    """The annotation XML's geolocation grid points (row/col plus each
    :data:`GEOLOC_FIELDS` value, and -- for building GCPs, not exposed in
    :class:`~satimg.metadata.Metadata` -- latitude/longitude/height), the
    scene-wide attrs, and the full measurement grid's ``(lines, samples)``
    shape (``imageAnnotation/imageInformation``)."""
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
        "platform_velocity": _nearest_orbit_velocity(root, at),
        "orbit_inclination": ORBIT_INCLINATION_DEG,
        "azimuth_pixel_spacing": float(root.findtext(".//azimuthPixelSpacing")),
    }
    shape = (
        int(root.findtext(".//numberOfLines")),
        int(root.findtext(".//numberOfSamples")),
    )
    return points, attrs, shape


def build_transformer(points: list[dict]) -> GCPTransformer:
    """A :class:`~satimg.transform.GCPTransformer` (WGS84) from
    :func:`read_geolocation`'s geolocation grid points."""
    gcps = [
        GroundControlPoint(
            row=p["row"], col=p["col"], x=p["longitude"], y=p["latitude"], z=p["height"],
        )
        for p in points
    ]
    return GCPTransformer(gcps, CRS.from_epsg(4326))
