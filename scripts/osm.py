"""Shared bits of the scripts building satimg's bundled data from OpenStreetMap.

QLever's SPARQL endpoint over the whole OpenStreetMap planet answers a planet-wide
query in seconds to minutes, where Overpass would have to be crawled tile by tile.
"""

import gzip
import json
import math
import os
import re

import requests
import shapely
import shapely.affinity
import shapely.ops

QLEVER_URL = "https://qlever.dev/api/osm-planet"
PREFIXES = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX osm: <https://www.openstreetmap.org/>
PREFIX osmkey: <https://www.openstreetmap.org/wiki/Key:>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
"""
KEY_PREFIX = "<https://www.openstreetmap.org/wiki/Key:"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "satimg", "data")


def select(query) -> list[list[str]]:
    """The rows of a QLever query, as RDF terms: <iri>, "value", "value"@lang,
    "value"^^<datatype> or a bare WKT point. An element matching several UNION
    branches comes back once for each."""
    response = requests.post(QLEVER_URL, data={"query": PREFIXES + query},
                             headers={"Accept": "text/tab-separated-values"}, timeout=900)
    response.raise_for_status()
    # QLever sends no charset, and splitlines() would also split on the Unicode line
    # separators some names contain
    return [line.split("\t") for line in response.content.decode("utf-8").rstrip("\n").split("\n")[1:]]


def literal(term) -> str:
    """The value of an RDF literal term."""
    m = re.match(r'"(.*)"', term)
    return m.group(1) if m else term


def tags(query) -> dict[str, dict[str, str]]:
    """{element: {key: value}} from a ``SELECT ?x ?k ?v`` query."""
    result = {}
    for element, key, value in select(query):
        result.setdefault(element, {})[key[len(KEY_PREFIX):-1]] = literal(value)
    return result


def polygonal(geom):
    """The polygons of ``geom`` (made valid), dropping its points and lines."""
    geom = shapely.make_valid(geom)
    return shapely.ops.unary_union([g for g in getattr(geom, "geoms", [geom])
                                    if g.geom_type in ("Polygon", "MultiPolygon")])


def disc(lon, lat, meters):
    """A 16-sided circle of ``meters`` radius around a point, in lon/lat degrees."""
    circle = shapely.Point(lon, lat).buffer(meters / 111_320, quad_segs=4)
    return shapely.affinity.scale(circle, xfact=1 / math.cos(math.radians(lat)), origin=(lon, lat))


def widen(geom, meters):
    """``geom`` grown by ``meters`` on every side, in lon/lat degrees."""
    centre = geom.centroid
    xfact = math.cos(math.radians(centre.y))
    # buffer in a frame where a degree of longitude is as long as one of latitude
    squared = shapely.affinity.scale(geom, xfact=xfact, origin=centre)
    return shapely.affinity.scale(squared.buffer(meters / 111_320, quad_segs=2), xfact=1 / xfact, origin=centre)


def write_features(name, features):
    """Write ``features`` as a gzipped FeatureCollection into satimg/data. mtime=0
    keeps a rebuild from the same data byte-identical, so git sees no change."""
    with gzip.GzipFile(os.path.join(DATA_DIR, name), "wb", mtime=0) as f:
        f.write(json.dumps({"type": "FeatureCollection", "features": features}, separators=(",", ":")).encode())


def feature(properties, geom):
    """A GeoJSON feature, its coordinates rounded to ~1 m."""
    return {"type": "Feature", "properties": properties,
            "geometry": json.loads(shapely.to_geojson(shapely.set_precision(geom, 0.00001)))}
