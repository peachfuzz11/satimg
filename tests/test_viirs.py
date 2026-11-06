import datetime
import io
import json
import os
import zipfile

import h5py
import numpy
import pytest

import satimg
from satimg import ViirsGeometry
from satimg.connectors import ViirsDNBConnector, get_connector
from satimg.connectors.viirs import HttpRangeFile, rings_intersect
from satimg.prediction import ViirsBoatDetector
from satimg.products import ViirsDNBProduct
from satimg.registry import resolve
from satimg.transform import SwathTransformer
from satimg.viirs_utils import granule_key, granule_times, node_columns
from tests import minify_viirs as fx
from tests.conftest import _PRODUCTS

NAME = "SVDNB_j01_d20261101_t1630064_e1631309_b46398_c20261101171011472000_oebc_ops"
FOLDER = _PRODUCTS / NAME
SDR = FOLDER / f"{NAME}.h5"
GEO = next(FOLDER.glob("GDNBO_*.h5")) if FOLDER.exists() else None

pytestmark = pytest.mark.skipif(not FOLDER.exists(), reason="VIIRS test granule missing")


@pytest.fixture(scope="module")
def product():
    with satimg.open(str(FOLDER)) as p:
        yield p


@pytest.fixture(scope="module")
def truth():
    with h5py.File(GEO) as g:
        return g["All_Data/VIIRS-DNB-GEO_All/Latitude"][()].astype(float), g["All_Data/VIIRS-DNB-GEO_All/Longitude"][()].astype(float)


def _metres(a, b):
    a, b = numpy.deg2rad(a), numpy.deg2rad(b)
    h = numpy.sin((b[:, 0] - a[:, 0]) / 2) ** 2 + numpy.cos(a[:, 0]) * numpy.cos(b[:, 0]) * numpy.sin((b[:, 1] - a[:, 1]) / 2) ** 2
    return 2 * 6_371_000 * numpy.arcsin(numpy.sqrt(h))


# -- names ----------------------------------------------------------------------
@pytest.mark.parametrize("name", [NAME, f"{NAME}.h5", "SVDNB_npp_d20261101_t0000035_e0001277_b77784_c20261101004425633000_oebc_ops.h5"])
def test_resolves(name):
    assert resolve(name) is ViirsDNBProduct


def test_granule_times_roll_over_midnight():
    start, end = granule_times("SVDNB_j01_d20261101_t2359301_e0000546_b46400_c1_oebc_ops.h5")
    assert start == datetime.datetime(2026, 11, 1, 23, 59, 30, 100000)
    assert end == datetime.datetime(2026, 11, 2, 0, 0, 54, 600000)


def test_granule_key_pairs_sdr_and_geo():
    assert granule_key(SDR.name) == granule_key(GEO.name)


# -- product ----------------------------------------------------------------------
@pytest.mark.parametrize("how", ["folder", "bare_sdr", "zip"])
def test_opens_every_layout(how, tmp_path):
    if how == "folder":
        path = FOLDER
    elif how == "bare_sdr":
        path = SDR
    else:
        path = tmp_path / f"{NAME}.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
            for f in (SDR, GEO):
                z.write(f, f"{NAME}/{f.name}")
    with satimg.open(str(path)) as p:
        assert isinstance(p, ViirsDNBProduct)
        assert p.raw.values[0, fx.BOATS[0][0], fx.BOATS[0][1]] > 50e-9


def test_shape_time_attrs(product):
    assert product.shape == (1, fx.ROWS, fx.COLS)
    assert list(product.raw.band.values) == ["DNB"]
    assert product.dtype == numpy.float32
    assert product.timestamp.replace(microsecond=0) == datetime.datetime(2026, 11, 1, 16, 30, 48)
    attrs = product.metadata.attrs
    assert (attrs["satellite"], attrs["orbit"], attrs["day_night"], attrs["pass"]) == ("noaa-20", 46398, "night", "descending")
    assert 0 < attrs["moon_illumination"] < 1


def test_raw_fill_is_nan(product):
    raw = product.raw.values[0]
    assert numpy.isnan(raw[fx.FILL])
    assert numpy.isnan(raw).sum() == 1


def test_visual_stretches_lights_bright(product):
    v = product.visual.values[0]
    assert v.dtype == numpy.uint8
    # 60 nW on the 0.5 - 500 nW log stretch: ~70% up; the sea is near the bottom
    assert v[fx.BOATS[0]] > 150
    assert numpy.median(v) < 100
    assert v[fx.FILL] == 0
    assert product.thumbnail().size == satimg.readers.THUMBNAIL_SIZE


def test_metadata_is_night(product):
    m = product.metadata
    assert set(m.fields) == {
        "satellite_zenith", "satellite_azimuth", "solar_zenith", "solar_azimuth", "lunar_zenith", "lunar_azimuth",
    }
    assert m.solar_zenith.at((32, 128)) > 100


# -- transformer ------------------------------------------------------------------
def test_transformer_forward_within_tolerance(product, truth):
    lat, lon = truth
    rr, cc = numpy.meshgrid(numpy.arange(fx.ROWS), numpy.arange(fx.COLS), indexing="ij")
    rc = numpy.column_stack((rr.ravel() + 0.5, cc.ravel() + 0.5))
    error = _metres(product.transformer.rowcol_to_latlon(rc), numpy.column_stack((lat.ravel(), lon.ravel())))
    assert error.max() < 100  # TRANSFORMER_TOLERANCE_M = 50, plus the row direction


def test_transformer_inverse_finds_a_pixel_seeing_the_point(product, truth):
    lat, lon = truth
    rr, cc = numpy.meshgrid(numpy.arange(0, fx.ROWS, 3), numpy.arange(0, fx.COLS, 7), indexing="ij")
    target = numpy.column_stack((lat[rr, cc].ravel(), lon[rr, cc].ravel()))
    rc = product.transformer.latlon_to_rowcol(target)
    assert not numpy.isnan(rc).any()
    # where scans overlap (the bow-tie) another pixel may be returned -- but
    # one that sees the same spot
    assert _metres(product.transformer.rowcol_to_latlon(rc), target).max() < 100
    exact = numpy.hypot(rc[:, 0] - (rr.ravel() + 0.5), rc[:, 1] - (cc.ravel() + 0.5)) < 0.1
    assert exact.mean() > 0.8


def test_transformer_never_interpolates_across_scans(product, truth):
    # the last pixel of scan 0 and first of scan 1: each from its own scan's nodes
    lat, lon = truth
    for row in (15, 16, 31, 32):
        got = product.transformer.rowcol_to_latlon((row + 0.5, 3.5))
        assert _metres(got, numpy.array([[lat[row, 3], lon[row, 3]]]))[0] < 100


def test_transformer_off_swath_is_nan(product):
    assert numpy.isnan(product.transformer.latlon_to_rowcol((0.0, 0.0))).all()


def test_swath_transformer_across_the_antimeridian():
    rows, cols = numpy.array([0.5, 9.5]), numpy.array([0.5, 9.5])
    t = SwathTransformer(rows, cols, [[10.0, 10.0], [9.0, 9.0]], [[179.5, -179.5], [179.5, -179.5]])
    lat, lon = t.rowcol_to_latlon((5.0, 5.0))[0]
    assert lat == pytest.approx(9.5, abs=1e-3)
    assert abs(lon) == pytest.approx(180.0, abs=1e-3)
    assert t.latlon_to_rowcol((lat, lon))[0] == pytest.approx((5.0, 5.0), abs=1e-6)


def test_node_columns_meet_their_tolerance(truth):
    lat, lon = truth
    loose, tight = node_columns(lat[:2], lon[:2], 250.0), node_columns(lat[:2], lon[:2], 25.0)
    assert loose[0] == 0 and loose[-1] == fx.COLS - 1
    assert len(loose) < len(tight)


def test_geometry_round_trips(product, truth):
    stored = json.loads(json.dumps(product.geometry.to_dict(), allow_nan=False))
    geometry = ViirsGeometry.from_dict(stored)
    assert isinstance(geometry.transformer, SwathTransformer)
    lat, lon = truth
    got = geometry.transformer.rowcol_to_latlon((40.5, 180.5))
    assert _metres(got, numpy.array([[lat[40, 180], lon[40, 180]]]))[0] < 300
    assert geometry.metadata.solar_zenith.at((32, 128)) == pytest.approx(product.metadata.solar_zenith.at((32, 128)), abs=0.5)


# -- boat lights ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def detections(product):
    return ViirsBoatDetector(product).detect()


def _pixels(detections):
    return {(int(d.bbox.center[1]), int(d.bbox.center[0])) for d in detections}


def test_finds_every_injected_boat(detections):
    assert set(fx.BOATS) <= _pixels(detections)


def test_drops_the_particle_hit(detections):
    assert fx.PARTICLE not in _pixels(detections)


def test_skips_the_noisy_edge_columns(detections):
    assert all(col >= 8 for _, col in _pixels(detections))


def test_detections_are_tagged_and_located(detections, product):
    d = next(d for d in detections if (int(d.bbox.center[1]), int(d.bbox.center[0])) == fx.BOATS[0])
    assert d.label.name == "boat" and 0.5 <= d.label.confidence <= 1
    expected = product.transformer.rowcol_to_latlon((fx.BOATS[0][0] + 0.5, fx.BOATS[0][1] + 0.5))[0]
    assert (d.coordinate.lat, d.coordinate.lon) == pytest.approx(tuple(expected))
    assert isinstance(d.tags, list)


def test_daytime_granule_has_no_boat_lights(product, monkeypatch):
    monkeypatch.setitem(product.metadata.attrs, "day_night", "day")
    assert ViirsBoatDetector(product).detect() == []


# -- connector ------------------------------------------------------------------------
def test_connector_factory_is_offline():
    assert get_connector("viirs-dnb").satellite == "noaa-20"
    assert get_connector("viirs-dnb", satellite="snpp")._bucket == "noaa-nesdis-snpp-pds"
    with pytest.raises(ValueError, match="unknown satellite"):
        ViirsDNBConnector("goes-16")


def test_connector_needs_a_time_window():
    with pytest.raises(ValueError, match="start_datetime"):
        ViirsDNBConnector().search()


def test_rings_intersect():
    square = [(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)]
    assert rings_intersect(square, [(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)])
    assert rings_intersect(square, [(0.5, 0.5), (1, 0.5), (1, 1), (0.5, 0.5)])  # inside
    assert not rings_intersect(square, [(5, 5), (6, 5), (6, 6), (5, 5)])
    east = [(179, 0), (179.9, 0), (179.9, 1), (179, 1), (179, 0)]
    west = [(-179.95, 0.5), (-179, 0.5), (-179, 0.8), (-179.95, 0.5)]
    assert not rings_intersect(east, west)
    across = [(179.5, 0.2), (-179.5, 0.2), (-179.5, 0.6), (179.5, 0.6), (179.5, 0.2)]
    assert rings_intersect(east, across)


class _FileSession:
    """Stands in for a requests session over a local file, honouring Range."""

    def __init__(self, path):
        self.data = open(path, "rb").read()
        self.requests = 0

    def head(self, url, timeout=None):
        return _Response(b"", {"Content-Length": str(len(self.data))})

    def get(self, url, headers=None, timeout=None):
        self.requests += 1
        start, stop = (int(v) for v in headers["Range"].split("=")[1].split("-"))
        return _Response(self.data[start:stop + 1], {})


class _Response:
    def __init__(self, content, headers):
        self.content, self.headers = content, headers

    def raise_for_status(self):
        pass


def test_http_range_file_reads_hdf5_attributes_without_the_whole_file():
    session = _FileSession(SDR)
    with HttpRangeFile("https://example.invalid/x.h5", session, block=4096) as raw, h5py.File(raw, "r") as h5:
        flag = h5["Data_Products/VIIRS-DNB-SDR/VIIRS-DNB-SDR_Gran_0"].attrs["N_Day_Night_Flag"]
    assert flag.ravel()[0] == b"Night"
    assert session.requests * 4096 < len(session.data)
    raw = HttpRangeFile("u", _FileSession(SDR))
    raw.seek(-10, io.SEEK_END)
    assert raw.read() == open(SDR, "rb").read()[-10:]
    assert os.path.getsize(SDR) == raw.tell()
