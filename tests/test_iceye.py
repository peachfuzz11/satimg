import datetime
import zipfile

import numpy
import pytest

import satimg
from satimg.products.iceye.grd.iceye_grd import IceyeGRD
from satimg.products.iceye.grd.iceye_grd_connector import IceyeGRDConnector
from satimg.products.iceye.grd.iceye_grd_parsing import gcp_grid
from satimg.readers.thumbnail import THUMBNAIL_SIZE
from satimg.registry import resolve
from satimg.transform.swath_transformer import SwathTransformer
from tests import iceye_fixture as fx


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return fx.build(tmp_path_factory.mktemp("iceye"))


@pytest.fixture(scope="module")
def product(folder):
    with satimg.open(str(folder)) as p:
        yield p


@pytest.mark.parametrize("name", [
    fx.NAME,
    f"{fx.NAME}_GRD.tif",
    "ICEYE_ETH0SK_20260219T021322Z_8871777_X53_SM",
    "ICEYE_67Z6P0_20260818T063421Z_10334436_X49_SCW_GRD.tif",
])
def test_resolves(name):
    assert resolve(name) is IceyeGRD


def test_other_product_types_do_not_resolve():
    with pytest.raises(satimg.UnknownProductError):
        resolve(f"{fx.NAME}_SLC.tif")


def test_opens_the_product_folder(product):
    assert isinstance(product, IceyeGRD)
    assert product.shape == (1, fx.HEIGHT, fx.WIDTH)
    assert product.dtype == numpy.uint16
    assert list(product.raw.band.values) == ["VV"]
    assert product.timestamp == datetime.datetime(2025, 11, 10, 12, 44, 42, 38000)
    attrs = product.metadata.attrs
    assert (attrs["platform"], attrs["mode"], attrs["imaging_mode"], attrs["look_direction"], attrs["pass"]) == (
        "iceye-x42", "spotlight", "SLEDF", "right", "descending")
    assert attrs["calibration_factor"] == fx.CF
    assert attrs["pixel_spacing"] == pytest.approx([fx.ROW_M, fx.COL_M], rel=1e-3)


def test_transformer_interpolates_the_gcp_grid(product):
    t = product.transformer
    assert isinstance(t, SwathTransformer)
    # a GCP's pixel index is a pixel centre
    assert t.rowcol_to_latlon([(85.5, 133.5)])[0] == pytest.approx(fx.latlon(85, 133), abs=1e-9)
    assert t.rowcol_to_latlon([(40.5, 100.5)])[0] == pytest.approx(fx.latlon(40, 100), abs=1e-7)
    rowcol = numpy.array([[10.0, 20.0], [200.0, 150.0]])
    assert t.latlon_to_rowcol(t.rowcol_to_latlon(rowcol)) == pytest.approx(rowcol, abs=1e-6)


def test_footprint_is_the_grid_outline(product):
    ring = product.footprint["geometry"]["coordinates"][0]
    assert len(ring) == 5 and ring[0] == ring[-1]
    lat, lon = fx.latlon(-0.5, -0.5)  # the top-left pixel's outer corner
    assert ring[0] == pytest.approx((lon, lat), abs=1e-7)


def test_sigma0_unpacks_and_calibrates(product):
    s0 = product.sigma0(product.raw).values[0]
    assert 10 * numpy.log10(s0[fx.SHIP]) == pytest.approx(fx.SHIP_DB, abs=0.01)
    assert 10 * numpy.log10(s0[40, 40]) == pytest.approx(fx.SEA_DB, abs=0.05)
    assert numpy.isnan(s0[0, 0])  # no data
    assert numpy.isnan(s0[fx.FLOOR])  # below the packing's floor


def test_visual_sea_dark_ship_bright(product):
    v = product.visual.values[0]
    assert product.visual.dtype == numpy.uint8
    assert v[fx.SHIP] > 240
    assert v[40, 40] < 20
    assert v[0, 0] == 0


def test_overviews_and_thumbnail(product):
    assert product.visual_overviews == [1, 2, 4]
    # the overviews keep the full-resolution band's packing
    assert product.visual_overview(2).values[0][fx.SHIP[0] // 2, fx.SHIP[1] // 2] > 200
    assert product.thumbnail().size == THUMBNAIL_SIZE


def test_opens_a_bare_tif_and_a_zip(folder, tmp_path):
    with satimg.open(str(folder / f"{fx.NAME}_GRD.tif")) as p:
        assert p.shape == (1, fx.HEIGHT, fx.WIDTH)
    archive = tmp_path / "image.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as z:
        for f in folder.iterdir():
            z.write(f, f"{fx.NAME}/{f.name}")
    with satimg.open(str(archive)) as p:
        assert isinstance(p, IceyeGRD)
        assert p.visual.values[0][fx.SHIP] > 240


def test_missing_corner_gcps_are_filled_in(tmp_path):
    missing = ((0.0, 133.0), (0.0, 199.0))
    rows, cols, lat, lon = gcp_grid(fx.gcps(skip=missing))
    assert lat.shape == (4, 4)
    for r, c in missing:
        i, j = list(rows).index(r), list(cols).index(c)
        # linear from the column below; the fixture's grid is linear too
        assert (lat[i, j], lon[i, j]) == pytest.approx(fx.latlon(r, c), abs=1e-9)
    with satimg.open(str(fx.build(tmp_path, skip_gcps=missing))) as p:
        assert p.shape == (1, fx.HEIGHT, fx.WIDTH)


# -- connector ------------------------------------------------------------------
def test_wants_image_items_only():
    connector = IceyeGRDConnector()
    assert connector._wanted(f"https://x/stac-items/2025/11/{fx.NAME}.json")
    assert not connector._wanted("https://x/stac-items/000-iceye-sar-collection-footprints.json")


def test_download_takes_the_grd(tmp_path, monkeypatch):
    base = f"https://iceye-open-data-catalog.s3.amazonaws.com/data/dwell-fine/{fx.NAME}/{fx.NAME}"
    item = {
        "id": fx.NAME,
        "assets": {
            "grd-cog": {"href": f"{base}_GRD.tif", "roles": ["data"]},
            "grd-json": {"href": f"{base}_GRD.json", "roles": ["metadata"]},
            "slc-cog": {"href": f"{base}_SLC.tif", "roles": ["data"]},
        },
    }
    fetched = {}

    def stream_to_zip(session, members, save_path):
        fetched.update(members)
        return save_path

    monkeypatch.setattr("satimg.products.iceye.grd.iceye_grd_connector.stream_to_zip", stream_to_zip)
    IceyeGRDConnector().download(item, str(tmp_path / "i.zip"))
    assert fetched == {f"{fx.NAME}/{fx.NAME}_GRD.tif": f"{base}_GRD.tif"}
