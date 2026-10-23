"""Rebuild ``src/satimg/data/ports.json.gz`` from the OpenStreetMap port and harbour
areas of the whole planet, each classed as a commercial port or a small-boat harbor.

    uv run python scripts/build_ports.py
"""

import gzip
import json
import math
import os
import re

import requests
import shapely
from shapely.ops import unary_union

OUT = os.path.join(os.path.dirname(__file__), "..", "src", "satimg", "data", "ports.json.gz")

# QLever's SPARQL endpoint over the whole OpenStreetMap planet answers these in a few
# minutes, where Overpass would have to be crawled tile by tile. The geometries and the tags
# are asked for apart: one query with an OPTIONAL per tag runs QLever out of memory.
QLEVER_URL = "https://qlever.dev/api/osm-planet"
PREFIXES = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX osm: <https://www.openstreetmap.org/>
PREFIX osmkey: <https://www.openstreetmap.org/wiki/Key:>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
"""
AREAS = """
  { ?x osmkey:industrial "port" } UNION { ?x osmkey:landuse "harbour" } UNION
  { ?x osmkey:harbour "yes" } UNION { ?x <https://www.openstreetmap.org/wiki/Key:seamark:type> "harbour" }
  ?x rdf:type ?type . FILTER(?type IN (osm:way, osm:relation))
"""
GEOMETRY_QUERY = PREFIXES + "SELECT ?x ?wkt WHERE {" + AREAS + "  ?x geo:hasGeometry/geo:asWKT ?wkt .\n}"
TAG_QUERY = PREFIXES + "SELECT ?x ?k ?v WHERE {" + AREAS + """
  VALUES ?k { osmkey:name osmkey:industrial osmkey:harbour osmkey:leisure osmkey:port
              <https://www.openstreetmap.org/wiki/Key:seamark:harbour:category> }
  ?x ?k ?v .
}"""
KEY_PREFIX = "<https://www.openstreetmap.org/wiki/Key:"


def select(query) -> list[list[str]]:
    """The rows of a QLever query, as RDF terms: <iri>, "value", "value"@lang or "value"^^<datatype>.
    An area matching several of the AREAS patterns comes back once for each."""
    response = requests.post(QLEVER_URL, data={"query": query},
                             headers={"Accept": "text/tab-separated-values"}, timeout=900)
    response.raise_for_status()
    # QLever sends no charset, and splitlines() would also split on the Unicode line
    # separators some names contain
    return [line.split("\t") for line in response.content.decode("utf-8").rstrip("\n").split("\n")[1:]]


COMMERCIAL = {"cargo", "container", "bulk", "tanker", "roro", "ferry", "passenger", "naval", "shipyard",
              "offshore_support", "port_support", "port_support_base", "timber", "oil_terminal"}


def tag_kind(tags) -> str | None:
    """"port" or "harbor" when the tags say so, None when the area could be either."""
    categories = {c.strip() for c in tags.get("seamark:harbour:category", "").split(";") if c.strip()}
    if tags.get("leisure") == "marina" or any("marina" in c for c in categories) or tags.get("harbour") == "slipway":
        return "harbor"
    if tags.get("industrial") == "port" or "port" in tags or categories & COMMERCIAL:
        return "port"
    return None


def main():
    tags = {}
    for area, key, value in select(TAG_QUERY):
        tags.setdefault(area, {})[key[len(KEY_PREFIX):-1]] = re.match(r'"(.*)"', value).group(1)
    geometries = {area: wkt for area, wkt in select(GEOMETRY_QUERY)}

    areas = []
    for area, wkt in geometries.items():
        geom = shapely.make_valid(shapely.from_wkt(re.match(r'"(.*)"', wkt).group(1)))
        # unclosed ways come back as lines
        geom = unary_union([g for g in getattr(geom, "geoms", [geom]) if g.geom_type in ("Polygon", "MultiPolygon")])
        geom = shapely.set_precision(geom.simplify(0.0001, preserve_topology=True), 0.00001)
        if not geom.is_empty:
            areas.append((tags.get(area, {}), geom, tag_kind(tags.get(area, {}))))

    # An area whose tags don't tell is a port when it lies next to port terminal land or
    # covers 20 ha or more (tagged marinas are rarely over 7 ha, and a marina inside a
    # big basin still makes its points harbor points), else a small-boat harbor.
    port_tree = shapely.STRtree([g for _, g, kind in areas if kind == "port"])
    features = []
    for tags, geom, kind in areas:
        if kind is None:
            hectares = geom.area * 111.32e3 ** 2 * math.cos(math.radians(geom.centroid.y)) / 1e4
            if len(port_tree.query(geom, predicate="dwithin", distance=0.001)) or hectares >= 20:
                kind = "port"
            else:
                kind = "harbor"
        features.append({
            "type": "Feature",
            "properties": {"name": tags.get("name"), "kind": kind},
            "geometry": json.loads(shapely.to_geojson(geom)),
        })

    # gzipped: the planet's ports are ~8 MB of JSON. mtime=0 keeps a rebuild from the
    # same data byte-identical, so git sees no change.
    with gzip.GzipFile(OUT, "wb", mtime=0) as f:
        f.write(json.dumps({"type": "FeatureCollection", "features": features}, separators=(",", ":")).encode())


if __name__ == "__main__":
    main()
