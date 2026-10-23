"""Rebuild ``src/satimg/data/fishfarms.json.gz`` from EMODnet's finfish farms and the
planet's offshore fish farms in OpenStreetMap.

    uv run python scripts/build_fishfarms.py

A farm's net cages and feed barge span a few hundred metres around the point most farms
are mapped as, so a point becomes a 250 m disc: EMODnet's farms at sea (Europe, mostly
salmon) and OSM's. OSM's marine farms (``seamark:type=marine_farm``) and aquaculture
areas (``landuse=aquaculture``) count when they lie at sea and are for fish or of no
stated kind; an outline is widened by 50 m.
"""

import requests
import shapely

from osm import disc, feature, literal, polygonal, select, tags, widen, write_features
from satimg.landmask import LandMask

EMODNET_URL = ("https://ows.emodnet-humanactivities.eu/wfs?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature"
               "&typeNames=emodnet:finfish&outputFormat=application/json")
SITE_METERS = 250
OUTLINE_METERS = 50

FARMS = """
  { ?x <https://www.openstreetmap.org/wiki/Key:seamark:type> "marine_farm" } UNION { ?x osmkey:landuse "aquaculture" }
"""
GEOMETRY_QUERY = "SELECT ?x ?wkt WHERE {" + FARMS + "  ?x geo:hasGeometry/geo:asWKT ?wkt .\n}"
TAG_QUERY = "SELECT ?x ?k ?v WHERE {" + FARMS + """
  VALUES ?k { osmkey:name osmkey:aquaculture <https://www.openstreetmap.org/wiki/Key:seamark:marine_farm:category> }
  ?x ?k ?v .
}"""


def main():
    land = LandMask()
    features = []

    for f in requests.get(EMODNET_URL, timeout=300).json()["features"]:
        properties = f["properties"]
        if properties["position_c"] == "At sea" and properties["status"] != "Inactive":
            lon, lat = f["geometry"]["coordinates"][:2]
            features.append(feature({"name": properties["owner_name"], "source": "emodnet"},
                                    disc(lon, lat, SITE_METERS)))

    farm_tags = tags(TAG_QUERY)
    for farm, wkt in dict(select(GEOMETRY_QUERY)).items():
        farm_tag = farm_tags.get(farm, {})
        kind = farm_tag.get("seamark:marine_farm:category") or farm_tag.get("aquaculture")
        if kind and "fish" not in kind:
            continue
        geom = shapely.from_wkt(literal(wkt))
        outline = polygonal(geom)
        point = outline.representative_point() if not outline.is_empty else geom.centroid
        if land.contains(point.y, point.x):
            continue
        geom = disc(point.x, point.y, SITE_METERS) if outline.is_empty else widen(outline, OUTLINE_METERS)
        features.append(feature({"name": farm_tag.get("name"), "source": "osm"}, geom))

    write_features("fishfarms.json.gz", features)


if __name__ == "__main__":
    main()
