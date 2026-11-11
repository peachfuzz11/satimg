import json

import numpy
import pytest

import satimg
from satimg import SAROrbitGeometry, SceneGeometry
from satimg.products.landsat.c2l1.landsat_c2l1 import GEOMETRY_GRID
from satimg.transform.affine_transformer import AffineTransformer
from satimg.transform.transformer_factory import load_transformer
from satimg.transform.gcp_transformer import GCPTransformer
from tests.test_zip_native import _build_zip

PRODUCTS = ["sentinel1_iw", "sentinel2_l1c", "landsat"]


@pytest.fixture(scope="module", params=PRODUCTS)
def product(request):
    return request.getfixturevalue(request.param)


def _stored(product):
    # through strict JSON, the way an app (e.g. PostgreSQL jsonb) would persist it
    return json.loads(json.dumps(product.geometry.to_dict(), allow_nan=False))


def _load(product, stored):
    cls = SAROrbitGeometry if isinstance(product.geometry, SAROrbitGeometry) else SceneGeometry
    return cls.from_dict(stored)


def _rowcols(product):
    h, w = product.height, product.width
    return numpy.array([(h // 4, w // 4), (h // 2, w // 2), (3 * h // 4, 3 * w // 4)])


def test_round_trips_through_strict_json(product):
    stored = _stored(product)
    assert _load(product, stored).to_dict() == stored


def test_sentinel1_loads_with_its_gcp_transformer(sentinel1_iw):
    geometry = SAROrbitGeometry.from_dict(_stored(sentinel1_iw))
    assert type(geometry) is SAROrbitGeometry
    assert type(geometry.transformer) is GCPTransformer


def test_shape_and_transformer_match(product):
    geometry = _load(product, _stored(product))
    assert (geometry.height, geometry.width) == (product.height, product.width)
    rowcols = _rowcols(product)
    latlons = product.transformer.rowcol_to_latlon(rowcols)
    numpy.testing.assert_allclose(geometry.transformer.rowcol_to_latlon(rowcols), latlons)
    numpy.testing.assert_allclose(
        geometry.transformer.latlon_to_rowcol(latlons), product.transformer.latlon_to_rowcol(latlons)
    )


def test_metadata_matches(product):
    geometry = _load(product, _stored(product))
    assert geometry.metadata.fields == product.metadata.fields
    assert geometry.metadata.attrs == product.metadata.attrs
    rowcols = _rowcols(product)
    # Landsat's fields are coarsened to GEOMETRY_GRID; angles are smooth
    atol = 0.05 if type(product).__name__ == "LandsatC2L1" else 1e-9
    for name in geometry.metadata.fields:
        numpy.testing.assert_allclose(
            geometry.metadata[name].at(rowcols), product.metadata[name].at(rowcols), atol=atol
        )


def test_landsat_fields_are_coarsened(landsat):
    stored = _stored(landsat)
    for field in stored["fields"].values():
        assert len(field["rows"]) <= GEOMETRY_GRID and len(field["cols"]) <= GEOMETRY_GRID
    assert len(json.dumps(stored)) < 200_000


def test_landsat_shape_from_mtl_matches_raw(landsat):
    assert (landsat.height, landsat.width) == (landsat.raw.sizes["y"], landsat.raw.sizes["x"])


def test_nan_is_stored_as_null():
    geometry = SceneGeometry(
        _affine_transformer(),
        {"f": ([0, 10], [0, 10], [[1.0, numpy.nan], [2.0, 3.0]], "degrees")},
        {},
        (11, 11),
    )
    stored = json.loads(json.dumps(geometry.to_dict(), allow_nan=False))
    assert stored["fields"]["f"]["values"] == [[1.0, None], [2.0, 3.0]]
    assert numpy.isnan(SceneGeometry.from_dict(stored).metadata.f._values[0, 1])


def _affine_transformer():
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    return AffineTransformer(Affine(10, 0, 300000, 0, -10, 6200000), CRS.from_epsg(32633))


@pytest.mark.parametrize("key", ["sentinel1_iw", "sentinel2_l1c", "landsat"])
def test_zip_native_geometry_matches(key, request, tmp_path_factory):
    with satimg.open(_build_zip(key, tmp_path_factory)) as zip_product:
        stored = json.loads(json.dumps(zip_product.geometry.to_dict(), allow_nan=False))
    assert stored == _stored(request.getfixturevalue(key))


@pytest.mark.parametrize("key, kind", [("sentinel1_iw", GCPTransformer), ("sentinel2_l1c", AffineTransformer)])
def test_transformer_round_trips_through_dict(key, kind, request):
    product = request.getfixturevalue(key)
    stored = json.loads(json.dumps(product.transformer.to_dict(), allow_nan=False))
    transformer = kind.from_dict(stored)
    assert type(transformer) is kind
    assert type(load_transformer(stored)) is kind
    assert transformer.to_dict() == stored
    rowcols = _rowcols(product)
    numpy.testing.assert_allclose(
        transformer.rowcol_to_latlon(rowcols), product.transformer.rowcol_to_latlon(rowcols)
    )


def test_load_transformer_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="not a stored transformer"):
        load_transformer({"polynomial": []})
