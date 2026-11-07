import datetime
import io
import json
import zipfile

import h5py
import numpy
import pytest

import satimg
from satimg import Sentinel3Geometry
from satimg.connectors import Sentinel3Connector, get_connector
from satimg.landmask import LandMask
from satimg.products import Sentinel3OLCIProduct
from satimg.registry import resolve
from satimg.transform import SwathTransformer
from tests import minify_sentinel3 as fx
from tests.conftest import _PRODUCTS

NAME = "S3A_OL_1_EFR____20240916T101050_20240916T101350_20240917T104150_0179_117_065_1800_PS1_O_NT_004.SEN3"
FOLDER = _PRODUCTS / NAME

pytestmark = pytest.mark.skipif(not FOLDER.exists(), reason="OLCI test product missing")


@pytest.fixture(scope="module")
def product():
    with satimg.open(str(FOLDER)) as p:
        yield p


@pytest.fixture(scope="module")
def truth():
    """The product's own per-pixel geolocation and a sea mask for it."""
    with h5py.File(FOLDER / "geo_coordinates.nc") as g:
        lat, lon = g["latitude"][()] * 1e-6, g["longitude"][()] * 1e-6
    sea = ~LandMask().contains(lat.ravel(), lon.ravel()).reshape(lat.shape)
    return lat, lon, sea


def _metres(a, b):
    a, b = numpy.deg2rad(a), numpy.deg2rad(b)
    h = numpy.sin((b[:, 0] - a[:, 0]) / 2) ** 2 + numpy.cos(a[:, 0]) * numpy.cos(b[:, 0]) * numpy.sin((b[:, 1] - a[:, 1]) / 2) ** 2
    return 2 * 6_371_000 * numpy.arcsin(numpy.sqrt(h))


@pytest.mark.parametrize("name", [
    NAME, NAME[:-5],
    "S3B_OL_1_EFR____20261105T105357_20261105T105657_20261105T131948_0180_126_265_1980_ESA_O_NR_004.SEN3",
])
def test_resolves(name):
    assert resolve(name) is Sentinel3OLCIProduct


def test_shape_bands_time_attrs(product):
    assert product.shape == (21, fx.ROWS, fx.COLS)
    assert product.raw.band.values[0] == "Oa01" and product.raw.band.values[-1] == "Oa21"
    assert product.raw.wavelength_nm.sel(band="Oa17") == 865.0
    assert product.dtype == numpy.uint16
    assert product.timestamp == datetime.datetime(2024, 9, 16, 10, 12, 20, 407740)
    attrs = product.metadata.attrs
    assert (attrs["satellite"], attrs["timeliness"], attrs["relative_orbit"], attrs["frame"]) == ("sentinel-3a", "ntc", 65, 1800)
    assert len(attrs["radiance_scales"]) == 21


def test_raw_and_radiance(product):
    raw = product.raw.sel(band="Oa08").values
    assert raw[fx.TARGET] == fx.TARGET_DN
    assert raw[fx.FILL] == 65535
    with h5py.File(FOLDER / "Oa08_radiance.nc") as f:
        scale = float(f["Oa08_radiance"].attrs["scale_factor"][0])
    rad = product.radiance(product.raw.sel(band=["Oa08"])).values[0]
    assert rad[fx.TARGET] == pytest.approx(fx.TARGET_DN * scale, rel=1e-6)
    assert numpy.isnan(rad[fx.FILL])


def test_visual_true_colour(product):
    v = product.visual.values
    assert v.shape == (3, fx.ROWS, fx.COLS) and v.dtype == numpy.uint8
    assert list(product.visual.band.values) == ["red", "green", "blue"]
    assert (v[:, fx.TARGET[0], fx.TARGET[1]] == 255).all()
    assert (v[:, fx.FILL[0], fx.FILL[1]] == 0).all()
    assert product.thumbnail().size == satimg.readers.THUMBNAIL_SIZE  # no quicklook in the fixture: rendered


def test_metadata_angles_from_the_tie_points(product):
    with h5py.File(FOLDER / "tie_geometries.nc") as f:
        sza = f["SZA"][()] * 1e-6
    assert set(product.metadata.fields) == {"solar_zenith", "solar_azimuth", "satellite_zenith", "satellite_azimuth"}
    # tie column 1 is image column 64; row 64 is one of the kept tie rows
    assert product.metadata.solar_zenith.at((64.5, 64.5)) == pytest.approx(sza[64, 1], abs=1e-3)


def test_footprint_is_the_manifest_s(product, truth):
    lat, lon, _sea = truth
    ring = product.footprint["geometry"]["coordinates"][0]
    assert ring[0] == pytest.approx((lon[0, 0], lat[0, 0]), abs=1e-4)
    assert len(ring) == 5


def test_transformer_matches_sea_level_pixels(product, truth):
    lat, lon, sea = truth
    rr, cc = numpy.nonzero(sea)
    assert len(rr) > 1000  # the window is mostly the Gulf of Bothnia
    rc = numpy.column_stack((rr + 0.5, cc + 0.5))
    error = _metres(product.transformer.rowcol_to_latlon(rc), numpy.column_stack((lat[rr, cc], lon[rr, cc])))
    # the per-pixel positions are terrain corrected and, at sea, sit ~100 m
    # (a third of a pixel) from the tie points -- see satimg.s3_utils
    assert numpy.median(error) < 150
    assert numpy.percentile(error, 99) < 250


def test_transformer_inverse(product, truth):
    lat, lon, sea = truth
    rr, cc = numpy.nonzero(sea)
    pick = numpy.arange(0, len(rr), 97)
    rc = numpy.column_stack((rr[pick] + 0.5, cc[pick] + 0.5))
    back = product.transformer.latlon_to_rowcol(product.transformer.rowcol_to_latlon(rc))
    assert back == pytest.approx(rc, abs=1e-3)


def test_patches_and_zip(product, tmp_path):
    patch = next(iter(product.patches_at([(fx.TARGET[1], fx.TARGET[0])], 16)))
    assert (patch.visual.values[:, 8, 8] == 255).all()
    archive = tmp_path / (NAME[:-5] + ".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as z:
        for f in FOLDER.iterdir():
            z.write(f, f"{NAME}/{f.name}")
    with satimg.open(str(archive)) as p:
        assert isinstance(p, Sentinel3OLCIProduct)
        assert p.raw.sel(band="Oa08").values[fx.TARGET] == fx.TARGET_DN


def test_geometry_round_trips(product):
    stored = json.loads(json.dumps(product.geometry.to_dict(), allow_nan=False))
    geometry = Sentinel3Geometry.from_dict(stored)
    assert isinstance(geometry.transformer, SwathTransformer)
    rc = (64.5, 128.5)
    assert _metres(geometry.transformer.rowcol_to_latlon(rc), product.transformer.rowcol_to_latlon(rc))[0] < 100
    assert geometry.metadata.solar_zenith.at(rc) == pytest.approx(product.metadata.solar_zenith.at(rc), abs=0.1)


# -- connector ------------------------------------------------------------------
def test_connector_timeliness():
    assert get_connector("sentinel-3-olci-efr").collection == "sentinel-3-olci-1-efr-nrt"
    assert get_connector("sentinel-3-olci-efr", timeliness="ntc").collection == "sentinel-3-olci-1-efr-ntc"
    with pytest.raises(ValueError, match="timeliness"):
        Sentinel3Connector(timeliness="fast")


def test_connector_downloads_the_lowercase_product_asset(tmp_path):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as z:
        z.writestr(f"{NAME}/xfdumanifest.xml", "<x/>")

    class Response:
        status_code = 200
        content = payload.getvalue()
        headers = {}

    class Session:
        def get(self, url, allow_redirects=True):
            assert url == "https://download.example/odata/product"
            return Response()

    connector = Sentinel3Connector()
    connector._get_session = lambda: Session()
    item = {"id": NAME[:-5], "assets": {"product": {"href": "https://download.example/odata/product"}}}
    out = connector.download(item, str(tmp_path / "olci.zip"))
    assert zipfile.ZipFile(out).namelist() == [f"{NAME}/xfdumanifest.xml"]
