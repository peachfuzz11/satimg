"""Rebuild ``src/satimg/data/windfarms.json.gz`` from EMODnet's wind farms and the
planet's offshore wind farms in OpenStreetMap.

    uv run python scripts/build_windfarms.py

EMODnet's farms (Europe) carry the year they were built. Outside them, OSM's farm
outlines are taken as they are, and the offshore turbines OSM maps as points --
most farms have no outline -- each become a 50 m disc.
"""

import numpy
import requests
import shapely

from osm import disc, feature, literal, polygonal, select, tags, write_features
from satimg.landmask import LandMask

EMODNET_URL = ("https://ows.emodnet-humanactivities.eu/wfs?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature"
               "&typeNames=emodnet:windfarmspoly&outputFormat=application/json")
STANDING = {"Production", "Construction", "Test site"}
TURBINE_METERS = 50

FARMS = """
  ?x osmkey:power "plant" . ?x <https://www.openstreetmap.org/wiki/Key:plant:source> "wind" .
"""
FARM_QUERY = "SELECT ?x ?wkt WHERE {" + FARMS + "  ?x geo:hasGeometry/geo:asWKT ?wkt .\n}"
FARM_NAME_QUERY = "SELECT ?x ?k ?v WHERE {" + FARMS + "  VALUES ?k { osmkey:name } ?x ?k ?v .\n}"
TURBINE_QUERY = """SELECT ?wkt WHERE {
  ?x <https://www.openstreetmap.org/wiki/Key:generator:source> "wind" . ?x rdf:type osm:node .
  ?x geo:hasGeometry/geo:asWKT ?wkt .
}"""


def main():
    land = LandMask()
    features = []

    emodnet = [f for f in requests.get(EMODNET_URL, timeout=300).json()["features"]
               if f["properties"]["status"] in STANDING]
    emodnet_geoms = [shapely.geometry.shape(f["geometry"]) for f in emodnet]
    for f, geom in zip(emodnet, emodnet_geoms):
        year = f["properties"]["year"]
        features.append(feature({"name": f["properties"]["name"], "year": int(year) if year else None,
                                 "source": "emodnet"}, geom))
    covered = shapely.STRtree(emodnet_geoms)

    names = tags(FARM_NAME_QUERY)
    outlines = []
    for farm, wkt in dict(select(FARM_QUERY)).items():
        geom = polygonal(shapely.from_wkt(literal(wkt)))
        point = geom.representative_point() if not geom.is_empty else None
        if point and not land.contains(point.y, point.x) and not len(covered.query(geom, predicate="intersects")):
            outlines.append(geom)
            features.append(feature({"name": names.get(farm, {}).get("name"), "year": None, "source": "osm"}, geom))

    turbines = shapely.from_wkt([literal(wkt) for wkt, in select(TURBINE_QUERY)])
    lon, lat = shapely.get_x(turbines), shapely.get_y(turbines)
    offshore = ~land.contains(lat, lon)
    inside, _ = shapely.STRtree(emodnet_geoms + outlines).query(turbines, predicate="within")
    keep = offshore & ~numpy.isin(numpy.arange(len(turbines)), inside)
    turbine_areas = shapely.union_all([disc(x, y, TURBINE_METERS) for x, y in zip(lon[keep], lat[keep])])
    for geom in getattr(turbine_areas, "geoms", [turbine_areas]):
        features.append(feature({"name": None, "year": None, "source": "osm turbines"}, geom))

    write_features("windfarms.json.gz", features)


if __name__ == "__main__":
    main()
