import datetime
import zipfile

import numpy
import pytest

import satimg
from satimg.products.umbra.gec.umbra_gec import UmbraGEC
from satimg.products.umbra.gec.umbra_gec_connector import UmbraGECConnector
from satimg.readers.thumbnail import THUMBNAIL_SIZE
from satimg.registry import resolve
from tests import umbra_fixture as fx


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return fx.build(tmp_path_factory.mktemp("umbra"))


@pytest.fixture(scope="module")
def product(folder):
    with satimg.open(str(folder)) as p:
        yield p


@pytest.mark.parametrize("name", [fx.NAME, f"{fx.NAME}_GEC.tif", "/data/2023-04-16-02-14-35_UMBRA-04"])
def test_resolves(name):
    assert resolve(name) is UmbraGEC


def test_opens_the_collect_folder(product):
    assert isinstance(product, UmbraGEC)
    assert product.shape == (1, fx.HEIGHT, fx.WIDTH)
    assert product.dtype == numpy.uint8
    assert list(product.raw.band.values) == ["VV"]
    assert product.timestamp == datetime.datetime(2025, 1, 31, 14, 10, 48, 900000)
    assert product.bounds() == pytest.approx((-58.42, -34.611, -58.419, -34.61))
    attrs = product.metadata.attrs
    assert (attrs["platform"], attrs["mode"], attrs["look_direction"], attrs["pass"]) == (
        "umbra-08", "spotlight", "left", "descending")
    assert attrs["incidence_angle"] == pytest.approx(32.8)
    # ~0.2 m pixels: the rotated degree grid measured in metres
    assert attrs["pixel_spacing"] == pytest.approx([0.20, 0.22], abs=0.02)


def test_rotated_grid_round_trips(product):
    rowcol = numpy.array([[10.0, 20.0], [200.0, 150.0]])
    back = product.transformer.latlon_to_rowcol(product.transformer.rowcol_to_latlon(rowcol))
    assert back == pytest.approx(rowcol)
    assert product.transformer.rowcol_to_latlon([(0, 0)])[0] == pytest.approx((-34.61, -58.42))


def test_visual_water_dark_ship_bright(product):
    visual = product.visual
    assert visual.dtype == numpy.uint8 and visual.sizes["band"] == 1
    v = visual.values[0]
    assert v[fx.SHIP] > 240
    assert v[40, 40] < 10
    assert v[0, 0] == 0  # no data stays black


def test_overviews_and_thumbnail(product):
    assert product.visual_overviews == [1, 2, 4]
    assert product.visual_overview(2).sizes["x"] == fx.WIDTH // 2
    assert product.thumbnail().size == THUMBNAIL_SIZE


def test_opens_a_bare_tif_and_a_zip(folder, tmp_path):
    with satimg.open(str(folder / f"{fx.NAME}_GEC.tif")) as p:
        assert p.shape == (1, fx.HEIGHT, fx.WIDTH)
    archive = tmp_path / "collect.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as z:
        for f in folder.iterdir():
            z.write(f, f"{fx.NAME}/{f.name}")
    with satimg.open(str(archive)) as p:
        assert isinstance(p, UmbraGEC)
        assert p.visual.values[0][fx.SHIP] > 240


def test_older_metadata_json(tmp_path):
    with satimg.open(str(fx.build(tmp_path, v1=True))) as p:
        assert p.timestamp == datetime.datetime(2025, 1, 31, 14, 10, 48, 900000)
        attrs = p.metadata.attrs
        assert (attrs["platform"], attrs["look_direction"], attrs["pass"]) == ("umbra-08", "left", "descending")
        assert (attrs["resolution_range"], attrs["resolution_azimuth"]) == (0.37, 0.36)
        assert p.bounds() == pytest.approx((-58.42, -34.611, -58.419, -34.61))


def test_missing_metadata_is_a_clear_error(tmp_path):
    folder = fx.build(tmp_path)
    (folder / f"{fx.NAME}.stac.v2.json").unlink()
    with pytest.raises(FileNotFoundError, match="METADATA"):
        satimg.open(str(folder))


# -- connector ------------------------------------------------------------------
class _Response:
    def __init__(self, content=b"", status=200):
        self.content = content
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def _listing(keys):
    body = "".join(f"<Contents><Key>{k}</Key></Contents>" for k in keys)
    return (
        '<?xml version="1.0"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        f"<IsTruncated>false</IsTruncated>{body}</ListBucketResult>"
    ).encode()


class _Session:
    def __init__(self, keys):
        self.keys = keys
        self.calls = []

    def get(self, url, params=None, **kwargs):
        self.calls.append((url, params))
        return _Response(_listing([k for k in self.keys if k.startswith(params["prefix"])]))


def test_collect_files_picks_the_nearest_collect_of_the_satellite():
    task = fx.STAC["properties"]["umbra:task_id"]
    prefix = f"sar-data/task-data/{task}/"
    keys = [
        # the item's collect, 2 s off the item time, as the bucket names them
        f"{prefix}2025-01-31-14-10-46_UMBRA-08/2025-01-31-14-10-46_UMBRA-08_GEC.tif",
        f"{prefix}2025-01-31-14-10-46_UMBRA-08/2025-01-31-14-10-46_UMBRA-08.stac.v2.json",
        f"{prefix}2025-01-31-14-10-46_UMBRA-08/2025-01-31-14-10-46_UMBRA-08_SICD.nitf",
        # another satellite, and another day, in the same task
        f"{prefix}2025-01-31-14-10-48_UMBRA-09/2025-01-31-14-10-48_UMBRA-09_GEC.tif",
        f"{prefix}2025-02-07-14-10-48_UMBRA-08/2025-02-07-14-10-48_UMBRA-08_GEC.tif",
    ]
    folder, files = UmbraGECConnector().collect_files(_Session(keys), fx.STAC)
    assert folder == "2025-01-31-14-10-46_UMBRA-08"
    assert sorted(f.rsplit("/", 1)[1] for f in files) == [
        "2025-01-31-14-10-46_UMBRA-08.stac.v2.json", "2025-01-31-14-10-46_UMBRA-08_GEC.tif"]


def test_collect_files_refuses_a_distant_collect():
    prefix = f"sar-data/task-data/{fx.STAC['properties']['umbra:task_id']}/"
    keys = [f"{prefix}2025-02-07-14-10-48_UMBRA-08/2025-02-07-14-10-48_UMBRA-08_GEC.tif"]
    with pytest.raises(FileNotFoundError, match="no collect near"):
        UmbraGECConnector().collect_files(_Session(keys), fx.STAC)


def test_download_zips_the_collect(tmp_path, monkeypatch):
    connector = UmbraGECConnector()
    folder = "2025-01-31-14-10-46_UMBRA-08"
    keys = [f"x/{folder}/{folder}_GEC.tif", f"x/{folder}/{folder}.stac.v2.json"]
    monkeypatch.setattr(connector, "collect_files", lambda session, item: (folder, keys))
    fetched = {}

    def stream_to_zip(session, members, save_path):
        fetched.update(members)
        return save_path

    monkeypatch.setattr("satimg.products.umbra.gec.umbra_gec_connector.stream_to_zip", stream_to_zip)
    connector.download(fx.STAC, str(tmp_path / "c.zip"))
    assert fetched == {
        f"{folder}/{folder}_GEC.tif": f"https://umbra-open-data-catalog.s3.amazonaws.com/x/{folder}/{folder}_GEC.tif",
        f"{folder}/{folder}.stac.v2.json": f"https://umbra-open-data-catalog.s3.amazonaws.com/x/{folder}/{folder}.stac.v2.json",
    }
