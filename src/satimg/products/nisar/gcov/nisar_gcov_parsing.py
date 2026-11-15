"""NISAR L2 GCOV HDF5 parsing: identification, the frequency-A image grid,
the radar-grid metadata cube and the orbit.

Paths and semantics follow the NISAR L2 GCOV product specification (JPL
D-102274). Pure functions over an open ``h5py.File`` (see :func:`satimg.readers.hdf5.open_h5`),
returning plain dicts / arrays -- no dependency on
:class:`~satimg.product.Product`. See
:class:`~satimg.products.nisar.gcov.nisar_gcov.NisarGCOV` for the product-facing entry
points.
"""

from __future__ import annotations

import datetime
import re

import numpy
from rasterio.crs import CRS
from rasterio.transform import Affine

from satimg.source.local_source import FileSource
from satimg.source.source import Source

#: the L-band science group (an S-band product has ``SSAR`` instead).
ROOT = "/science/LSAR"
IDENTIFICATION = f"{ROOT}/identification"
GRID = f"{ROOT}/GCOV/grids/frequencyA"
RADAR_GRID = f"{ROOT}/GCOV/metadata/radarGrid"
ORBIT = f"{ROOT}/GCOV/metadata/orbit"

#: the real-valued (diagonal) covariance terms, i.e. backscatter power per
#: polarisation, in the order ``raw`` stacks the ones present.
DIAGONAL_TERMS = ("HHHH", "HVHV", "VHVH", "VVVV", "RHRH", "RVRV")

#: radarGrid cube -> (metadata field name, units). Sampled at sea level (see
#: :func:`read_radar_grid`).
CUBE_FIELDS = {
    "incidenceAngle": ("incidence_angle", "degrees"),
    "elevationAngle": ("elevation_angle", "degrees"),
    "slantRange": ("slant_range", "metres"),
    "losUnitVectorX": ("los_east", ""),
    "losUnitVectorY": ("los_north", ""),
    "alongTrackUnitVectorX": ("along_track_east", ""),
    "alongTrackUnitVectorY": ("along_track_north", ""),
}

#: 2-D radarGrid layers (no height axis) -> (metadata field name, units).
PLANE_FIELDS = {
    "groundTrackVelocity": ("ground_track_velocity", "m/s"),
}

#: the most cube nodes kept per axis: the cube is posted every few hundred
#: metres, far finer than these smooth fields need.
MAX_CUBE_NODES = 128


def find_h5(source: Source) -> str:
    """Root-relative path of the product's ``.h5``: the source itself when it
    is one file (named with ``.h5`` or not: ``download`` writes it to whatever
    path it is given), else the first ``.h5`` at its root (a zip holding one)."""
    if isinstance(source, FileSource) or (source.name.endswith(".h5") and source.exists(source.name)):
        return source.name
    for name in sorted(source.listdir()):
        if name.endswith(".h5"):
            return name
    raise FileNotFoundError(f"no .h5 under {source.name}")


def gdal_path(source: Source, relpath: str, dataset: str) -> str:
    """The GDAL (``rasterio``) path of HDF5 ``dataset`` in ``relpath``."""
    return f'HDF5:"{source.raster_path(relpath)}":/{dataset}'


def _str(value) -> str:
    if isinstance(value, numpy.ndarray):
        value = value.item() if value.shape == () else value[0]
    return value.decode() if isinstance(value, bytes) else str(value)


def _strs(value) -> list[str]:
    return [_str(v) for v in numpy.atleast_1d(value)]


def parse_time(text: str) -> datetime.datetime:
    """A NISAR UTC timestamp (``YYYY-mm-ddTHH:MM:SS[.fffffffff]``, often
    nanoseconds) as a naive ``datetime``, truncated to microseconds."""
    text = text.strip().rstrip("Z")
    m = re.match(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?", text)
    if not m:
        raise ValueError(f"not a NISAR timestamp: {text!r}")
    return datetime.datetime.fromisoformat(f"{m.group(1)}.{(m.group(2) or '0')[:6].ljust(6, '0')}")


def _epoch(units: str) -> datetime.datetime:
    """``"seconds since 2025-01-01T00:00:00"`` -> that datetime."""
    return parse_time(units.split("since", 1)[1])


def parse_polygon(wkt: str) -> dict:
    """The identification ``boundingPolygon`` (WKT ``POLYGON`` or ``POLYGON Z``,
    lon lat [h]) as a GeoJSON Feature."""
    body = wkt[wkt.index("((") + 2: wkt.index("))")]
    coords = []
    for vertex in body.split(","):
        lon, lat = (float(v) for v in vertex.split()[:2])
        coords.append((lon, lat))
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [coords]}}


def read_identification(h5) -> dict:
    ident = h5[IDENTIFICATION]
    start = parse_time(_str(ident["zeroDopplerStartTime"][()]))
    stop = parse_time(_str(ident["zeroDopplerEndTime"][()]))
    attrs = {
        "granule_id": _str(ident["granuleId"][()]),
        "look_direction": _str(ident["lookDirection"][()]).lower(),
        "pass": _str(ident["orbitPassDirection"][()]).lower(),
        "absolute_orbit": int(ident["absoluteOrbitNumber"][()]),
        "track": int(ident["trackNumber"][()]),
        "frame": int(ident["frameNumber"][()]),
    }
    return {
        "start": start,
        "stop": stop,
        "timestamp": start + (stop - start) / 2,
        "footprint": parse_polygon(_str(ident["boundingPolygon"][()])),
        "attrs": attrs,
    }


def read_grid(h5) -> dict:
    """The frequency-A image grid: its shape, affine transform and CRS (from
    the CF pixel-centre ``x/yCoordinates``), the diagonal covariance terms it
    holds, and its pixel spacing."""
    grid = h5[GRID]
    xs = grid["xCoordinates"][()]
    ys = grid["yCoordinates"][()]
    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])  # negative: north-up
    transform = Affine(dx, 0.0, float(xs[0]) - dx / 2, 0.0, dy, float(ys[0]) - dy / 2)
    projection = grid["projection"]
    epsg = int(projection.attrs.get("epsg_code", projection[()]))
    present = set(grid)
    listed = _strs(grid["listOfCovarianceTerms"][()]) if "listOfCovarianceTerms" in present else DIAGONAL_TERMS
    terms = [t for t in DIAGONAL_TERMS if t in present and t.upper() in {s.upper() for s in listed}]
    if not terms:
        raise ValueError("GCOV frequencyA has no diagonal covariance terms")
    return {
        "shape": (len(ys), len(xs)),
        "transform": transform,
        "crs": CRS.from_epsg(epsg),
        "terms": terms,
        "pixel_spacing": (abs(dy), abs(dx)),
    }


def _subsample(n: int) -> numpy.ndarray:
    """At most :data:`MAX_CUBE_NODES` evenly spread indices into ``n``, both
    ends kept."""
    if n <= MAX_CUBE_NODES:
        return numpy.arange(n)
    return numpy.unique(numpy.linspace(0, n - 1, MAX_CUBE_NODES).round().astype(int))


def _at_sea_level(cube: numpy.ndarray, heights: numpy.ndarray) -> numpy.ndarray:
    """``cube`` (height, y, x) linearly interpolated to height 0 -- ships sit
    on the sea surface -- or its nearest layer if 0 is outside the cube."""
    order = numpy.argsort(heights)
    heights, cube = heights[order], cube[order]
    if len(heights) == 1 or not heights[0] <= 0.0 <= heights[-1]:
        return cube[int(numpy.abs(heights).argmin())]
    i = min(int(numpy.searchsorted(heights, 0.0, side="right")) - 1, len(heights) - 2)
    t = (0.0 - heights[i]) / (heights[i + 1] - heights[i])
    return cube[i] * (1 - t) + cube[i + 1] * t


def read_radar_grid(h5, transform: Affine) -> dict:
    """The radarGrid metadata cube as coarse grids in image pixel
    coordinates: ``{"rows", "cols", <field>: (rows, cols) values}`` for every
    :data:`CUBE_FIELDS` / :data:`PLANE_FIELDS` dataset present, the 3-D cubes
    sampled at sea level. ``transform`` is the image grid's
    (:func:`read_grid`) -- the cube shares its CRS."""
    radar = h5[RADAR_GRID]
    xs = radar["xCoordinates"][()]
    ys = radar["yCoordinates"][()]
    heights = radar["heightAboveEllipsoid"][()]
    yi, xi = _subsample(len(ys)), _subsample(len(xs))
    inverse = ~transform
    cols = numpy.array([(inverse * (x, ys[0]))[0] for x in xs[xi]])
    rows = numpy.array([(inverse * (xs[0], y))[1] for y in ys[yi]])
    # Field wants ascending axes; the cube is north-up like the image
    row_order, col_order = numpy.argsort(rows), numpy.argsort(cols)
    out = {"rows": rows[row_order], "cols": cols[col_order]}
    for name, (field, _units) in {**CUBE_FIELDS, **PLANE_FIELDS}.items():
        if name not in radar:
            continue
        data = radar[name]
        if data.ndim == 3:
            values = _at_sea_level(data[:, yi, :][:, :, xi].astype(float), heights)
        else:
            values = data[yi, :][:, xi].astype(float)
        out[field] = values[row_order][:, col_order]
    return out


def orbit(h5, epoch: datetime.datetime) -> dict:
    """The sensor orbit as
    :class:`~satimg.sar_orbit_geometry.SAROrbitGeometry` reads it: the
    product's orbit state vectors (Earth-fixed), timed from ``epoch``."""
    group = h5[ORBIT]
    times = group["time"]
    offset = (_epoch(_str(times.attrs["units"])) - epoch).total_seconds()
    return {
        "epoch": epoch.isoformat() + "Z",
        "times": (times[()] + offset).tolist(),
        "positions": numpy.asarray(group["position"][()], dtype=float).tolist(),
        "velocities": numpy.asarray(group["velocity"][()], dtype=float).tolist(),
    }
