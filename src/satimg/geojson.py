"""Reading GeoJSON, whatever wraps the geometry."""

from __future__ import annotations


def geometry_of(geojson: dict) -> dict:
    """The geometry of a GeoJSON geometry, Feature or (single-feature)
    FeatureCollection -- the first feature's, for a collection."""
    if geojson.get("type") == "FeatureCollection":
        geojson = geojson["features"][0]
    if geojson.get("type") == "Feature":
        geojson = geojson["geometry"]
    return geojson


def bbox(geojson: dict) -> tuple[float, float, float, float]:
    """``(min_lon, min_lat, max_lon, max_lat)`` of a Point / Polygon / ...
    geometry, Feature or single-feature FeatureCollection (see
    :func:`geometry_of`) -- of its first ring, for a multi-part geometry."""
    coords = geometry_of(geojson)["coordinates"]
    if not isinstance(coords[0], (list, tuple)):
        coords = [coords]
    while isinstance(coords[0][0], (list, tuple)):
        coords = coords[0]
    lons, lats = [c[0] for c in coords], [c[1] for c in coords]
    return min(lons), min(lats), max(lons), max(lats)
