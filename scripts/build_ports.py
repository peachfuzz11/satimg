"""Rebuild ``src/satimg/data/ports.json.gz`` from the OpenStreetMap port and harbour
areas of the whole planet, each classed as a commercial port or a small-boat harbor.
Ports are widened by 50 m to take in the ships at berth.

    uv run python scripts/build_ports.py
"""

import math

import shapely

from osm import feature, literal, polygonal, select, tags, widen, write_features

# a port's outline often runs along the quay, with the ships at berth just outside it
PORT_METERS = 50

AREAS = """
  { ?x osmkey:industrial "port" } UNION { ?x osmkey:landuse "harbour" } UNION
  { ?x osmkey:harbour "yes" } UNION { ?x <https://www.openstreetmap.org/wiki/Key:seamark:type> "harbour" }
  ?x rdf:type ?type . FILTER(?type IN (osm:way, osm:relation))
"""
# the geometries and the tags are asked for apart: one query with an OPTIONAL per tag
# runs QLever out of memory
GEOMETRY_QUERY = "SELECT ?x ?wkt WHERE {" + AREAS + "  ?x geo:hasGeometry/geo:asWKT ?wkt .\n}"
TAG_QUERY = "SELECT ?x ?k ?v WHERE {" + AREAS + """
  VALUES ?k { osmkey:name osmkey:industrial osmkey:harbour osmkey:leisure osmkey:port
              <https://www.openstreetmap.org/wiki/Key:seamark:harbour:category> }
  ?x ?k ?v .
}"""

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
    area_tags = tags(TAG_QUERY)
    geometries = {area: wkt for area, wkt in select(GEOMETRY_QUERY)}

    areas = []
    for area, wkt in geometries.items():
        # unclosed ways come back as lines
        geom = polygonal(shapely.from_wkt(literal(wkt)))
        geom = shapely.set_precision(geom.simplify(0.0001, preserve_topology=True), 0.00001)
        if not geom.is_empty:
            areas.append((area_tags.get(area, {}), geom, tag_kind(area_tags.get(area, {}))))

    # An area whose tags don't tell is a port when it lies next to port terminal land or
    # covers 20 ha or more (tagged marinas are rarely over 7 ha, and a marina inside a
    # big basin still makes its points harbor points), else a small-boat harbor.
    port_tree = shapely.STRtree([g for _, g, kind in areas if kind == "port"])
    features = []
    for area_tag, geom, kind in areas:
        if kind is None:
            hectares = geom.area * 111.32e3 ** 2 * math.cos(math.radians(geom.centroid.y)) / 1e4
            if len(port_tree.query(geom, predicate="dwithin", distance=0.001)) or hectares >= 20:
                kind = "port"
            else:
                kind = "harbor"
        if kind == "port":
            geom = widen(geom, PORT_METERS)
        features.append(feature({"name": area_tag.get("name"), "kind": kind}, geom))

    # gzipped: the planet's ports are ~8 MB of JSON
    write_features("ports.json.gz", features)


if __name__ == "__main__":
    main()
