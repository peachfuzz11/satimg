"""Rebuild ``src/satimg/data/platforms.json.gz`` from the planet's offshore platforms
in OpenStreetMap (``man_made=offshore_platform`` or ``seamark:type=platform``).

    uv run python scripts/build_platforms.py

Most platforms are mapped as points, which a detection's centre never quite hits: a
point becomes a 200 m disc -- a platform is 50-100 m across -- and an outline is
widened by 50 m.
"""

import shapely

from osm import disc, feature, literal, polygonal, select, tags, widen, write_features

POINT_METERS = 200
OUTLINE_METERS = 50

PLATFORMS = """
  { ?x osmkey:man_made "offshore_platform" } UNION { ?x <https://www.openstreetmap.org/wiki/Key:seamark:type> "platform" }
"""
GEOMETRY_QUERY = "SELECT ?x ?wkt WHERE {" + PLATFORMS + "  ?x geo:hasGeometry/geo:asWKT ?wkt .\n}"
TAG_QUERY = "SELECT ?x ?k ?v WHERE {" + PLATFORMS + """
  VALUES ?k { osmkey:name <https://www.openstreetmap.org/wiki/Key:seamark:platform:category> }
  ?x ?k ?v .
}"""


def main():
    platform_tags = tags(TAG_QUERY)
    features = []
    for platform, wkt in dict(select(GEOMETRY_QUERY)).items():
        geom = shapely.from_wkt(literal(wkt))
        outline = polygonal(geom)
        if outline.is_empty:
            centre = geom.centroid
            geom = disc(centre.x, centre.y, POINT_METERS)
        else:
            geom = widen(outline, OUTLINE_METERS)
        platform_tag = platform_tags.get(platform, {})
        features.append(feature({"name": platform_tag.get("name"),
                                 "category": platform_tag.get("seamark:platform:category")}, geom))

    write_features("platforms.json.gz", features)


if __name__ == "__main__":
    main()
