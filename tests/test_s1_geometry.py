import json

import numpy
import pytest

from satimg import Sentinel1Geometry


@pytest.fixture(scope="module")
def stored(sentinel1_iw):
    # through JSON, the way an app would persist it
    return json.loads(json.dumps(sentinel1_iw.geometry.to_dict()))


@pytest.fixture(scope="module")
def geometry(stored):
    return Sentinel1Geometry.from_dict(stored)


def _rowcols(product):
    h, w = product.height, product.width
    return numpy.array([(h // 4, w // 4), (h // 2, w // 2), (3 * h // 4, 3 * w // 4)])


def test_round_trips_through_dict(geometry, stored):
    assert geometry.to_dict() == stored


def test_shape_matches(sentinel1_iw, geometry):
    assert (geometry.height, geometry.width) == (sentinel1_iw.height, sentinel1_iw.width)


def test_transformer_matches(sentinel1_iw, geometry):
    rowcols = _rowcols(sentinel1_iw)
    latlons = sentinel1_iw.transformer.rowcol_to_latlon(rowcols)
    numpy.testing.assert_allclose(geometry.transformer.rowcol_to_latlon(rowcols), latlons)
    numpy.testing.assert_allclose(
        geometry.transformer.latlon_to_rowcol(latlons), sentinel1_iw.transformer.latlon_to_rowcol(latlons)
    )


def test_metadata_matches(sentinel1_iw, geometry):
    assert geometry.metadata.fields == sentinel1_iw.metadata.fields
    assert geometry.metadata.attrs == sentinel1_iw.metadata.attrs
    rowcols = _rowcols(sentinel1_iw)
    for name in geometry.metadata.fields:
        numpy.testing.assert_allclose(
            getattr(geometry.metadata, name).at(rowcols), getattr(sentinel1_iw.metadata, name).at(rowcols)
        )
    assert geometry.metadata.incidence_angle.corners() == sentinel1_iw.metadata.incidence_angle.corners()


@pytest.mark.parametrize("heading", [0.0, 73.0, 190.0, 300.0])
def test_sar_geometry_matches(sentinel1_iw, geometry, heading):
    rowcol = tuple(_rowcols(sentinel1_iw)[1])
    assert geometry.doppler_azimuth_shift(rowcol, 7.5, heading) == pytest.approx(
        sentinel1_iw.doppler_azimuth_shift(rowcol, 7.5, heading)
    )
    assert geometry.heading_to_los(rowcol, heading) == pytest.approx(sentinel1_iw.heading_to_los(rowcol, heading))
    assert geometry.heading_in_image(rowcol, heading) == pytest.approx(
        sentinel1_iw.heading_in_image(rowcol, heading)
    )
    assert geometry.heading_from_image(rowcol, heading) == pytest.approx(
        sentinel1_iw.heading_from_image(rowcol, heading)
    )
    lat, lon = sentinel1_iw.transformer.rowcol_to_latlon(rowcol)[0]
    assert geometry.correct_position(lat, lon, 7.5, heading) == pytest.approx(
        sentinel1_iw.correct_position(lat, lon, 7.5, heading)
    )


def test_vectorised_doppler_shift(sentinel1_iw, geometry):
    rowcols = _rowcols(sentinel1_iw)
    shifts = geometry.doppler_azimuth_shift(rowcols, 7.5, 73.0)
    assert shifts.shape == (len(rowcols),)
    for rowcol, shift in zip(rowcols, shifts):
        assert shift == pytest.approx(geometry.doppler_azimuth_shift(tuple(rowcol), 7.5, 73.0))
