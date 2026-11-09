import datetime
import json
import zipfile

import h5py
import numpy
import pytest

import satimg
from satimg import SceneGeometry
from satimg.connectors.viirs_connector import ViirsDNBConnector
from satimg.connectors.connector_factory import get_connector
from satimg.connectors._credentials import CredentialsError
from satimg.prediction.viirs_boat_detector import ViirsBoatDetector
from satimg.products.viirs.dnb.viirs_dnb import ViirsDNB
from satimg.products.swath_nodes import node_columns, scan_edge_rows
from satimg.products.viirs.dnb.viirs_dnb_parsing import FILL_BELOW, SCAN_ROWS, granule_key, read_geolocation
from satimg.readers.thumbnail import THUMBNAIL_SIZE
from satimg.registry import resolve
from satimg.transform.swath_transformer import SwathTransformer
from tests import minify_viirs as fx
from tests.conftest import _PRODUCTS
from tests.test_helper import h5_reads

NAME = "VJ102DNB.A2026305.1630.021.2026305232544"
FOLDER = _PRODUCTS / NAME
L1B = FOLDER / f"{NAME}.nc"
GEO = next(FOLDER.glob("VJ103DNB*.nc")) if FOLDER.exists() else None

pytestmark = pytest.mark.skipif(not FOLDER.exists(), reason="VIIRS test granule missing")


@pytest.fixture(scope="module")
def product():
    with satimg.open(str(FOLDER)) as p:
        yield p


@pytest.fixture(scope="module")
def truth():
    with h5py.File(GEO) as g:
        return g["geolocation_data/latitude"][()].astype(float), g["geolocation_data/longitude"][()].astype(float)


def _metres(a, b):
    a, b = numpy.deg2rad(a), numpy.deg2rad(b)
    h = numpy.sin((b[:, 0] - a[:, 0]) / 2) ** 2 + numpy.cos(a[:, 0]) * numpy.cos(b[:, 0]) * numpy.sin((b[:, 1] - a[:, 1]) / 2) ** 2
    return 2 * 6_371_000 * numpy.arcsin(numpy.sqrt(h))


# -- names ----------------------------------------------------------------------
@pytest.mark.parametrize("name", [
    f"{NAME}.nc",
    NAME,  # a folder / zip of it
    "VJ102DNB_NRT.A2026309.1654.021.2026309185751.nc",
    "VJ202DNB_NRT.A2026309.1600.021.2026309173950.nc",
    "VNP02DNB.A2026305.1630.002.2026305232544.nc",
])
def test_resolves(name):
    assert resolve(name) is ViirsDNB


def test_pairs_by_granule_not_production_time():
    assert granule_key("VNP02DNB.A2026305.1630.002.2026305232544.nc") == granule_key("VNP03DNB.A2026305.1630.002.2026305223724.nc")
    assert granule_key("VJ102DNB_NRT.A2026309.1654.021.2026309185751.nc") == granule_key("VJ103DNB_NRT.A2026309.1654.021.2026309181114.nc")
    assert granule_key("VNP02DNB.A2026305.1630.002.x.nc") != granule_key("VNP02DNB_NRT.A2026305.1630.002.nc")
    assert granule_key("VNP02DNB.A2026305.1630.002.x.nc") != granule_key("VJ103DNB.A2026305.1630.002.x.nc")


# -- product ----------------------------------------------------------------------
@pytest.mark.parametrize("how", ["folder", "bare", "zip"])
def test_opens_every_layout(how, tmp_path):
    if how == "folder":
        path = FOLDER
    elif how == "bare":
        path = L1B
    else:
        path = tmp_path / f"{NAME}.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
            for f in (L1B, GEO):
                z.write(f, f"{NAME}/{f.name}")
    with satimg.open(str(path)) as p:
        assert isinstance(p, ViirsDNB)
        assert p.raw.values[0, fx.BOATS[0][0], fx.BOATS[0][1]] > 50e-9


def test_shape_time_attrs(product):
    assert product.shape == (1, fx.ROWS, fx.COLS)
    assert list(product.raw.band.values) == ["DNB"]
    assert product.dtype == numpy.float32
    assert product.timestamp == datetime.datetime(2026, 11, 1, 16, 30, 48)
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
    # 60 nW on the 0.1 - 100 nW log stretch: ~90% up; the sea is near the
    # bottom, but not clipped black
    assert v[fx.BOATS[0]] > 150
    assert 0 < numpy.median(v) < 100
    assert v[fx.FILL] == 0
    assert product.thumbnail().size == THUMBNAIL_SIZE


def test_geolocation_is_the_scan_edge_rows(truth):
    # from #33: the rows read are exactly the file's scan-edge rows, scaled
    lat, lon = truth
    with h5py.File(GEO) as g:
        geolocation = read_geolocation(g)
    rows = scan_edge_rows(lat.shape[0], SCAN_ROWS)
    assert geolocation["shape"] == lat.shape
    numpy.testing.assert_array_equal(geolocation["rows"], rows)
    for got, full in ((geolocation["lat"], lat), (geolocation["lon"], lon)):
        numpy.testing.assert_array_equal(got, numpy.where(full[rows] <= FILL_BELOW, numpy.nan, full[rows]))


def test_open_reads_no_full_resolution_array():
    """Opening reads only the geolocation's scan-edge rows and a sparse
    sample -- never a full-resolution variable, nor any radiance."""
    with h5_reads() as reads, satimg.open(str(FOLDER)) as p:
        assert p.height > 0
    big = [r for r in reads if r[2] >= 1000]
    assert big and all(n <= size / 4 for _, n, size in big), big
    assert not [r for r in reads if "observation_data" in r[0]]


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
    geometry = SceneGeometry.from_dict(stored)
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
def _cmr_entry(name, day_night="NIGHT"):
    base = "https://data.laadsdaac.earthdatacloud.nasa.gov/prod-lads"
    return {
        "time_start": "2026-11-01T16:30:00.000Z",
        "day_night_flag": day_night,
        "links": [
            {"href": f"s3://prod-lads/{name}"},
            {"href": f"{base}/{name.split('.')[0]}/{name}"},
            {"href": f"https://ladsweb.modaps.eosdis.nasa.gov/opendap/{name}.html"},
        ],
    }


def test_connector_pairs_nrt_granules_from_both_satellites(monkeypatch):
    from satimg.connectors import earthdata

    feeds = {
        "VJ102DNB_NRT": [_cmr_entry("VJ102DNB_NRT.A2026309.1654.021.2026309185751.nc"),
                         _cmr_entry("VJ102DNB_NRT.A2026309.1700.021.2026309190000.nc")],
        "VJ103DNB_NRT": [_cmr_entry("VJ103DNB_NRT.A2026309.1654.021.2026309181114.nc")],  # 1700 has no GEO yet
        "VJ202DNB_NRT": [_cmr_entry("VJ202DNB_NRT.A2026309.1600.021.2026309173950.nc")],
        "VJ203DNB_NRT": [_cmr_entry("VJ203DNB_NRT.A2026309.1600.021.2026309173033.nc")],
    }
    calls = []

    class Response:
        def __init__(self, body):
            self.body = body

        def raise_for_status(self):
            pass

        def json(self):
            return self.body

    def get(url, params, timeout):
        calls.append(params)
        return Response({"feed": {"entry": feeds[params["short_name"]]}})

    monkeypatch.setattr(earthdata.requests, "get", get)
    aoi = {"type": "Polygon", "coordinates": [[[123, 33], [135, 33], [135, 40], [123, 33]]]}
    items = ViirsDNBConnector().search(
        geojson=aoi, start_datetime=datetime.datetime(2026, 11, 5, 15), stop_datetime=datetime.datetime(2026, 11, 5, 18),
        day_night="night",
    )
    assert sorted(i["id"] for i in items) == [
        "VJ102DNB_NRT.A2026309.1654.021.2026309185751", "VJ202DNB_NRT.A2026309.1600.021.2026309173950",
    ]
    assert {frozenset(i["assets"]) for i in items} == {
        frozenset({"VJ102DNB_NRT", "VJ103DNB_NRT"}), frozenset({"VJ202DNB_NRT", "VJ203DNB_NRT"}),
    }
    assert calls[0]["bounding_box"] == "123,33,135,40"
    assert calls[0]["temporal"] == "2026-11-05T15:00:00Z,2026-11-05T18:00:00Z"
    assert calls[0]["day_night_flag"] == "night"
    assert all(resolve(i["id"]) is ViirsDNB for i in items)


def test_connector_satellites_archive_and_login(monkeypatch):
    for var in ("EARTHDATA_TOKEN", "EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"):
        monkeypatch.delenv(var, raising=False)
    assert get_connector("viirs-dnb").short_names == ("VJ102DNB_NRT", "VJ103DNB_NRT", "VJ202DNB_NRT", "VJ203DNB_NRT")
    assert get_connector("viirs-dnb", satellite="noaa-21").short_names == ("VJ202DNB_NRT", "VJ203DNB_NRT")
    assert get_connector("viirs-dnb", satellite="noaa-20", nrt=False).short_names == ("VJ102DNB", "VJ103DNB")
    with pytest.raises(ValueError, match="unknown satellite"):
        ViirsDNBConnector("snpp")
    with pytest.raises(CredentialsError):
        ViirsDNBConnector()._get_session()
    assert ViirsDNBConnector(token="t")._get_session().headers["Authorization"] == "Bearer t"
