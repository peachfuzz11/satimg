"""Landsat's true-colour ``visual``: reflectance, ratio-Brovey, fixed tone curve.

The minified fixture has all-zero pixels, so the render is checked on small synthetic rasters.
"""

import math

import numpy
import pytest
import xarray

from satimg.products.landsat.c2l1 import landsat_c2l1_parsing
from satimg.products.landsat.c2l1.landsat_c2l1 import GAMMA, RAYLEIGH_TAU, WHITE_POINT
from satimg.products.landsat.c2l1.landsat_c2l1_parsing import BANDS, reflectance_coefficients


def _dn(product, rho: float) -> int:
    """The DN of every optical band that gives top-of-atmosphere reflectance ``rho``."""
    mult, add = product._reflectance["red"]
    return round((rho * product._sin_sun - add) / mult)


def _path(product) -> numpy.ndarray:
    """The Rayleigh path reflectance (red, green, blue) the render subtracts for this scene's sun."""
    mu = product._sin_sun
    return numpy.array(RAYLEIGH_TAU) * 0.75 * (1 + mu**2) / (4 * mu)


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


class TestRender:
    def test_shape_dtype_and_bands(self, landsat):
        out = landsat._render_visual(_raw(landsat, [0.1, 0.2]), 2)
        assert out.dtype == numpy.uint8
        assert out.dims == ("band", "y", "x") and out.shape == (3, 1, 2)
        assert [str(b) for b in out.band.values] == ["red", "green", "blue"]

    def test_white_point_maps_to_255_and_follows_the_gamma(self, landsat):
        # DNs are whole numbers, so test just above the white point (it clips) and at half of it
        out = landsat._render_visual(_raw(landsat, [WHITE_POINT * 1.05, WHITE_POINT / 2]), 2).values
        assert (out[:, 0, 0] == 255).all()
        path = _path(landsat)  # equal colours and pan: pan-sharpening changes nothing
        half = 255 * (numpy.clip(WHITE_POINT / 2 - path, 0, None) / (WHITE_POINT - path)) ** (1 / GAMMA)
        assert numpy.abs(out[:, 0, 1].astype(float) - half).max() <= 1

    def test_the_rayleigh_path_reflectance_is_removed(self, landsat):
        path = _path(landsat)
        below = landsat._render_visual(_raw(landsat, [path.min() * 0.99]), 1).values
        assert (below == 0).all()  # under every colour's path reflectance
        above = landsat._render_visual(_raw(landsat, [path.max() * 2]), 1).values
        assert (above > 0).all()
        assert path[0] < path[1] < path[2]  # the short wavelengths scatter most: red < green < blue

    def test_brighter_reflectance_is_brighter(self, landsat):
        out = landsat._render_visual(_raw(landsat, [0.02, 0.1, 0.2, 0.3]), 4).values[0, 0]
        assert (numpy.diff(out.astype(int)) > 0).all()

    def test_no_data_is_black(self, landsat):
        out = landsat._render_visual(_raw(landsat, [None, 0.2]), 2).values
        assert (out[:, 0, 0] == 0).all() and (out[:, 0, 1] > 0).all()

    def test_a_pixel_depends_only_on_its_own_reflectance(self, landsat):
        # nothing is computed from the rest of the scene: a pixel renders the same next to any others
        alone = landsat._render_visual(_raw(landsat, [0.2]), 1).values[:, 0, 0]
        crowded = landsat._render_visual(_raw(landsat, [0.01, 0.2, 0.3, None]), 4).values[:, 0, 1]
        assert alone.tolist() == crowded.tolist()

    def test_a_brighter_pan_band_brightens_every_colour(self, landsat):
        raw = _raw(landsat, [0.2])
        plain = landsat._render_visual(raw, 1).values[:, 0, 0].astype(int)
        raw.loc[dict(band="pan")] = _dn(landsat, 0.4)
        bright = landsat._render_visual(raw, 1).values[:, 0, 0].astype(int)
        assert (bright > plain).all()
