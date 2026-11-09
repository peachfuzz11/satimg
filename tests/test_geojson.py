import numpy
import pytest
import xarray

from satimg.geojson import bbox, geometry_of
from satimg.readers.thumbnail import THUMBNAIL_SIZE, thumbnail_from_visual
from satimg.readers.vrt import gcps
from satimg.transform.affine_transformer import AffineTransformer

POLYGON = {"type": "Polygon", "coordinates": [[[1, 2], [3, 2], [3, 5], [1, 5], [1, 2]]]}


@pytest.mark.parametrize(
    "geojson",
    [
        POLYGON,
        {"type": "Feature", "geometry": POLYGON},
        {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": POLYGON}]},
    ],
)
def test_bbox_unwraps_features(geojson):
    assert geometry_of(geojson) == POLYGON
    assert bbox(geojson) == (1, 2, 3, 5)


def test_bbox_of_point_and_multipolygon():
    assert bbox({"type": "Point", "coordinates": [7, 8]}) == (7, 8, 7, 8)
    multi = {"type": "MultiPolygon", "coordinates": [POLYGON["coordinates"]]}
    assert bbox(multi) == (1, 2, 3, 5)


@pytest.mark.parametrize("bands, mode", [(1, "L"), (3, "RGB")])
def test_thumbnail_from_visual(bands, mode):
    visual = xarray.DataArray(numpy.zeros((bands, 900, 1300), dtype="uint8"), dims=("band", "y", "x"))
    img = thumbnail_from_visual(visual)
    assert (img.mode, img.size) == (mode, THUMBNAIL_SIZE)


def test_gcps_span_the_image():
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    t = AffineTransformer(Affine(0.1, 0, 10, 0, -0.1, 60), CRS.from_epsg(4326))
    points = gcps(t, (100, 200))
    assert len(points) == 25
    assert (points[0].row, points[0].col, points[0].x, points[0].y) == (0, 0, 10, 60)
    assert (points[-1].row, points[-1].col) == (100, 200)
    assert points[-1].x == pytest.approx(30) and points[-1].y == pytest.approx(50)
