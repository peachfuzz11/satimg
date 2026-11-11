import datetime
import zipfile

import numpy
import pytest

import satimg
from satimg.products.capella.gec.capella_gec import CapellaGEC
from satimg.products.capella.gec.capella_gec_connector import CapellaGECConnector
from satimg.readers.thumbnail import THUMBNAIL_SIZE
from satimg.registry import resolve
from tests import capella_fixture as fx


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return fx.build(tmp_path_factory.mktemp("capella"))


@pytest.fixture(scope="module")
def product(folder):
    with satimg.open(str(folder)) as p:
        yield p


@pytest.mark.parametrize("name", [
    fx.NAME,
    f"{fx.NAME}.tif",
    "CAPELLA_C02_SS_GEC_HH_20210308122926_20210308122942",
    "CAPELLA_C03_SM_GEC_VV_20210329185145_20210329185152",
])
def test_resolves(name):
    assert resolve(name) is CapellaGEC


def test_other_product_types_do_not_resolve():
    with pytest.raises(satimg.UnknownProductError):
        resolve(fx.NAME.replace("_GEC_", "_GEO_"))


def test_opens_the_product_folder(product):
    assert isinstance(product, CapellaGEC)
    assert product.shape == (1, fx.HEIGHT, fx.WIDTH)
    assert product.dtype == numpy.uint16
    assert list(product.raw.band.values) == ["HH"]
    assert product.timestamp == datetime.datetime(2026, 1, 16, 1, 41, 14, 75999)
    attrs = product.metadata.attrs
    assert (attrs["platform"], attrs["mode"], attrs["look_direction"], attrs["pass"]) == (
        "capella-13", "spotlight", "left", "ascending")
    assert attrs["scale_factor"] == fx.SCALE
    assert attrs["pixel_spacing"] == [fx.SPACING, fx.SPACING]
    assert product.transformer.crs.to_epsg() == fx.EPSG


def test_footprint_is_the_grid_outline(product):
    ring = product.footprint["geometry"]["coordinates"][0]
    assert len(ring) == 5 and ring[0] == ring[-1]
    corners = product.transformer.latlon_to_rowcol([(lat, lon) for lon, lat in ring[:4]])
    expected = [[0, 0], [0, fx.WIDTH], [fx.HEIGHT, fx.WIDTH], [fx.HEIGHT, 0]]
    assert corners.ravel() == pytest.approx(numpy.ravel(expected), abs=1e-6)


def test_sigma0_is_calibrated(product):
    s0 = product.sigma0(product.raw).values[0]
    assert 10 * numpy.log10(s0[fx.SHIP]) == pytest.approx(fx.SHIP_DB, abs=0.01)
    assert 10 * numpy.log10(s0[40, 40]) == pytest.approx(fx.SEA_DB, abs=0.05)
    assert numpy.isnan(s0[0, 0])


def test_visual_sea_dark_ship_bright(product):
    v = product.visual.values[0]
    assert product.visual.dtype == numpy.uint8
    assert v[fx.SHIP] > 240
    assert v[40, 40] < 20
    assert v[0, 0] == 0


def test_overviews_and_rendered_thumbnail(product):
    assert product.visual_overviews == [1, 2, 4]
    assert product.thumbnail().size == THUMBNAIL_SIZE


def test_shipped_thumbnail_wins(tmp_path):
    with satimg.open(str(fx.build(tmp_path, thumbnail=True))) as p:
        thumb = p.thumbnail()
    assert thumb.size == THUMBNAIL_SIZE
    assert set(thumb.getdata()) == {77}


def test_opens_a_bare_tif_and_a_zip(folder, tmp_path):
    with satimg.open(str(folder / f"{fx.NAME}.tif")) as p:
        assert p.shape == (1, fx.HEIGHT, fx.WIDTH)
    archive = tmp_path / "collect.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as z:
        for f in folder.iterdir():
            z.write(f, f"{fx.NAME}/{f.name}")
    with satimg.open(str(archive)) as p:
        assert isinstance(p, CapellaGEC)
        assert p.visual.values[0][fx.SHIP] > 240


# -- connector ------------------------------------------------------------------
def test_wants_only_gec_items():
    connector = CapellaGECConnector()
    assert connector._wanted(f"https://x/{fx.NAME}/{fx.NAME}.json")
    assert not connector._wanted("https://x/CAPELLA_C13_SP_SLC_HH_1_2/CAPELLA_C13_SP_SLC_HH_1_2.json")


def test_download_takes_image_metadata_and_thumbnail(tmp_path, monkeypatch):
    base = "https://capella-open-data.s3.amazonaws.com/data/2026/1/16"
    geo = fx.NAME.replace("_GEC_", "_GEO_")
    item = {
        "id": fx.NAME,
        "assets": {
            "HH": {"href": f"{base}/{fx.NAME}/{fx.NAME}.tif", "roles": ["data"]},
            "metadata": {"href": f"{base}/{fx.NAME}/{fx.NAME}_extended.json", "roles": ["metadata"]},
            "preview": {"href": f"{base}/{geo}/{geo}_preview.tif", "roles": ["overview"]},
            "thumbnail": {"href": f"{base}/{geo}/{geo}_thumb.png", "roles": ["thumbnail"]},
        },
    }
    fetched = {}

    def stream_to_zip(session, members, save_path):
        fetched.update(members)
        return save_path

    monkeypatch.setattr("satimg.products.capella.gec.capella_gec_connector.stream_to_zip", stream_to_zip)
    CapellaGECConnector().download(item, str(tmp_path / "c.zip"))
    assert fetched == {
        f"{fx.NAME}/{fx.NAME}.tif": item["assets"]["HH"]["href"],
        f"{fx.NAME}/{fx.NAME}_extended.json": item["assets"]["metadata"]["href"],
        f"{fx.NAME}/{geo}_thumb.png": item["assets"]["thumbnail"]["href"],
    }
