"""Landsat's true-colour ``visual``: reflectance, ratio-Brovey, per-scene haze correction.

The minified fixture has all-zero pixels, so the render is checked on small synthetic rasters.
"""

import math

import numpy
import pytest
import xarray

from satimg.products.landsat.c2l1 import landsat_c2l1_parsing
from satimg.products.landsat.c2l1.landsat_c2l1 import GAMMA, HAZE_FRACTION, WHITE_POINT
from satimg.products.landsat.c2l1.landsat_c2l1_parsing import BANDS, reflectance_coefficients


def _dn(product, rho: float) -> int:
    """The DN of every optical band that gives top-of-atmosphere reflectance ``rho``."""
    mult, add = product._reflectance["red"]
    return round((rho * product._sin_sun - add) / mult)


def _raw(product, rhos) -> xarray.DataArray:
    """A raw cube, 1 row, one column per entry of ``rhos`` (``None`` = no data), every band equal."""
    dn = numpy.array([0 if r is None else _dn(product, r) for r in rhos], dtype="uint16")
    data = numpy.broadcast_to(dn[None, None, :], (len(BANDS), 1, len(rhos))).copy()
    return xarray.DataArray(data, dims=("band", "y", "x"), coords={"band": list(BANDS)})


class TestReflectanceCoefficients:
    def test_read_from_the_mtl(self):
        mtl = {"LEVEL1_RADIOMETRIC_RESCALING": {"REFLECTANCE_MULT_BAND_4": "2.5E-05", "REFLECTANCE_ADD_BAND_4": "-0.2"}}
        coeffs = reflectance_coefficients(mtl)
        assert coeffs["red"] == (2.5e-5, -0.2)
        assert coeffs["blue"] == landsat_c2l1_parsing.DEFAULT_REFLECTANCE  # not in this MTL

    def test_product_carries_them(self, landsat):
        assert set(landsat._reflectance) == set(landsat_c2l1_parsing.BAND_NUMBER)
        assert landsat._sin_sun == pytest.approx(math.sin(math.radians(26.76645117)))


class TestDarkObject:
    def test_no_data_scene_has_no_haze(self, landsat):
        assert landsat._dark_object().tolist() == [0.0, 0.0, 0.0]

    def test_computed_once(self, landsat):
        assert landsat._dark_object() is landsat._dark_object()


class TestRender:
    @pytest.fixture
    def product(self, landsat, monkeypatch):
        monkeypatch.setattr(landsat, "_dark", numpy.zeros(3, dtype="float32"))
        return landsat

    def test_shape_dtype_and_bands(self, product):
        out = product._render_visual(_raw(product, [0.1, 0.2]), 2)
        assert out.dtype == numpy.uint8
        assert out.dims == ("band", "y", "x") and out.shape == (3, 1, 2)
        assert [str(b) for b in out.band.values] == ["red", "green", "blue"]

    def test_white_point_maps_to_255_and_follows_the_gamma(self, product):
        # DNs are whole numbers, so test just above the white point (it clips) and at half of it
        out = product._render_visual(_raw(product, [WHITE_POINT * 1.05, WHITE_POINT / 2]), 2).values
        assert (out[:, 0, 0] == 255).all()
        assert abs(int(out[0, 0, 1]) - 255 * 0.5 ** (1 / GAMMA)) <= 1

    def test_brighter_reflectance_is_brighter(self, product):
        out = product._render_visual(_raw(product, [0.02, 0.1, 0.2, 0.3]), 4).values[0, 0]
        assert (numpy.diff(out.astype(int)) > 0).all()

    def test_no_data_is_black(self, product):
        out = product._render_visual(_raw(product, [None, 0.2]), 2).values
        assert (out[:, 0, 0] == 0).all() and (out[:, 0, 1] > 0).all()

    def test_haze_is_subtracted_per_scene(self, landsat, monkeypatch):
        haze = 0.1
        monkeypatch.setattr(landsat, "_dark", numpy.full(3, haze, dtype="float32"))
        # at HAZE_FRACTION * haze the pixel is black; above it, it is not
        out = landsat._render_visual(_raw(landsat, [HAZE_FRACTION * haze, 0.2]), 2).values
        assert (out[:, 0, 0] == 0).all() and (out[:, 0, 1] > 0).all()

    def test_pan_sharpening_keeps_each_colour_magnitude(self, product):
        raw = _raw(product, [0.2])
        # pan twice as bright as the colours: the render brightens, but colours stay in the same ratio
        plain = product._render_visual(raw, 1).values[:, 0, 0].astype(int)
        raw.loc[dict(band="pan")] = _dn(product, 0.4)
        bright = product._render_visual(raw, 1).values[:, 0, 0].astype(int)
        assert (bright > plain).all()
        assert len(set(plain.tolist())) == 1 and len(set(bright.tolist())) == 1  # equal-DN colours stay grey
