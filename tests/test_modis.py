import datetime
import json
import zipfile

import numpy
import pytest

import satimg
from satimg import SceneGeometry
from satimg.connectors import ModisConnector, get_connector
from satimg.products import ModisProduct
from satimg.readers.thumbnail import THUMBNAIL_SIZE
from satimg.registry import resolve
from satimg.transform.swath import SwathTransformer
from tests import modis_fixture as fx
from tests.test_helper import h5_reads


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    d = tmp_path_factory.mktemp("modis")
    fx.build(d / fx.TERRA)
    fx.build_geolocation(d / fx.TERRA_GEO)
    fx.build(d / fx.AQUA)  # no MYD03 beside it
    return d


@pytest.fixture(scope="module")
def terra(folder):
    with satimg.open(str(folder / fx.TERRA)) as p:
        yield p


def _metres(a, b):
    a, b = numpy.deg2rad(a), numpy.deg2rad(b)
    h = numpy.sin((b[:, 0] - a[:, 0]) / 2) ** 2 + numpy.cos(a[:, 0]) * numpy.cos(b[:, 0]) * numpy.sin((b[:, 1] - a[:, 1]) / 2) ** 2
    return 2 * 6_371_000 * numpy.arcsin(numpy.sqrt(h))


@pytest.mark.parametrize("name", [fx.TERRA, fx.AQUA, fx.TERRA[:-3], "MYD02QKM_NRT.A2026305.1250.007.nc"])
def test_resolves(name):
    assert resolve(name) is ModisProduct


def test_terra_and_aqua(folder):
    for name, satellite in ((fx.TERRA, "terra"), (fx.AQUA, "aqua")):
        with satimg.open(str(folder / name)) as p:
            assert p.metadata.attrs["satellite"] == satellite
    with satimg.open(str(folder / fx.AQUA)) as p:
        assert p.timestamp == datetime.datetime(2026, 11, 1, 12, 52, 30)


def test_angles_come_from_the_geolocation_file(terra, folder):
    m = terra.metadata
    assert set(m.fields) == {"solar_zenith", "solar_azimuth", "satellite_zenith", "satellite_azimuth"}
    assert m.solar_zenith.at((60, 128)) == pytest.approx(fx.SOLAR_ZENITH)
    assert m.satellite_zenith.at((60, 2)) > 55  # the swath edge
    assert m.satellite_zenith.at((60, 128)) < 5  # nadir
    with satimg.open(str(folder / fx.AQUA)) as aqua:  # no MYD03: positions from the L1B, no angles
        assert aqua.metadata.fields == []
        assert not numpy.isnan(aqua.transformer.rowcol_to_latlon((60, 128))).any()


@pytest.mark.parametrize("name", [fx.TERRA, fx.AQUA])
def test_open_reads_no_full_resolution_array(folder, name):
    """Opening reads only the 1 km geolocation's scan-edge rows (and the
    angles there) -- never a whole dataset, nor any 250 m pixels."""
    with h5_reads() as reads, satimg.open(str(folder / name)) as p:
        assert p.height == fx.ROWS
    big = [r for r in reads if r[2] >= 100]
    assert big and all(n <= size / 4 for _, n, size in big), big
    assert not [r for r in reads if "EV_250_RefSB" in r[0]]


def test_collection_6_hdf4_is_refused_clearly(tmp_path):
    path = tmp_path / "MOD02QKM.A2026305.1035.061.2026305170237.hdf"
    path.write_bytes(b"\x0e\x03\x13\x01")  # an HDF4 file's magic
    with pytest.raises(ValueError, match="Collection 6.1"):
        satimg.open(str(path))


def test_raw_is_the_stored_integers(terra):
    assert terra.shape == (2, fx.ROWS, fx.COLS)
    assert list(terra.raw.band.values) == ["B01", "B02"]
    assert list(terra.raw.wavelength_nm.values) == [645.0, 858.5]
    assert terra.dtype == numpy.uint16
    assert terra.raw.sel(band="B02").values[fx.SHIP] == fx.SHIP_DN
    assert (terra.raw.values[:, fx.FILL_ROWS] == 65535).all()
    assert terra.metadata.attrs["day_night"] == "day"


def test_reflectance_and_visual(terra):
    refl = terra.reflectance(terra.raw).values
    assert refl[1][fx.SHIP] == pytest.approx(fx.SHIP_DN * fx.REFLECTANCE_SCALES[1], rel=1e-5)
    assert numpy.isnan(refl[:, fx.FILL_ROWS]).all()
    v = terra.visual.values[0]
    assert v[fx.SHIP] > 200
    assert v[60, 20] < 100
    assert (v[fx.FILL_ROWS] == 0).all()
    assert terra.thumbnail().size == THUMBNAIL_SIZE


def test_patches_and_zip(terra, folder, tmp_path):
    patch = next(iter(terra.patches_at([(fx.SHIP[1], fx.SHIP[0])], 32)))
    assert patch.visual.values[0, 16, 16] > 200
    archive = tmp_path / (fx.TERRA[:-3] + ".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as z:
        for name in (fx.TERRA, fx.TERRA_GEO):
            z.write(folder / name, f"{fx.TERRA[:-3]}/{name}")
    with satimg.open(str(archive)) as p:
        assert p.raw.sel(band="B02").values[fx.SHIP] == fx.SHIP_DN
        assert p.metadata.solar_zenith.at((60, 128)) == pytest.approx(fx.SOLAR_ZENITH)


def test_transformer_places_1km_pixels_with_the_fractional_offset(terra):
    lat, lon = fx.geolocation()
    j, i = numpy.meshgrid(numpy.arange(lat.shape[0]), numpy.arange(lat.shape[1]), indexing="ij")
    # 1 km pixel (j, i): 250 m row 4j + 1.5, col 4i, pixel centres at +0.5
    rc = numpy.column_stack((4 * j.ravel() + 2.0, 4 * i.ravel() + 0.5))
    error = _metres(terra.transformer.rowcol_to_latlon(rc), numpy.column_stack((lat.ravel(), lon.ravel())))
    assert error.max() < 40


def test_transformer_inverse(terra):
    rng = numpy.random.default_rng(1)
    rc = numpy.column_stack((rng.uniform(4, fx.ROWS - 4, 200), rng.uniform(4, fx.COLS - 4, 200)))
    latlon = terra.transformer.rowcol_to_latlon(rc)
    back = terra.transformer.latlon_to_rowcol(latlon)
    assert not numpy.isnan(back).any()
    # overlapping scans may hand back the other scan's pixel -- seeing the same spot
    assert _metres(terra.transformer.rowcol_to_latlon(back), latlon).max() < 1


def test_geometry_round_trips(terra):
    stored = json.loads(json.dumps(terra.geometry.to_dict(), allow_nan=False))
    geometry = SceneGeometry.from_dict(stored)
    assert isinstance(geometry.transformer, SwathTransformer)
    rc = (60.5, 128.5)
    assert _metres(geometry.transformer.rowcol_to_latlon(rc), terra.transformer.rowcol_to_latlon(rc))[0] < 150


# -- connector ------------------------------------------------------------------
def test_connector_searches_terra_and_aqua(monkeypatch):
    from satimg.connectors import earthdata

    def entry(name, start):
        return {"time_start": start, "day_night_flag": "DAY", "links": [
            {"href": f"https://data.laadsdaac.earthdatacloud.nasa.gov/prod-lads/{name.split('.')[0]}/{name}"}]}

    feeds = {
        "MOD02QKM": [entry(fx.TERRA, "2026-11-01T10:35:00.000Z")],
        "MOD03": [entry(fx.TERRA_GEO, "2026-11-01T10:35:00.000Z")],
        "MYD02QKM": [entry(fx.AQUA, "2026-11-01T12:50:00.000Z"),
                     entry("MYD02QKM.A2026305.0950.007.2026306151000.nc", "2026-11-01T09:50:00.000Z"),
                     entry("MYD02QKM.A2026305.1255.007.2026306152000.nc", "2026-11-01T12:55:00.000Z")],
        "MYD03": [entry("MYD03.A2026305.1250.007.2026306150000.nc", "2026-11-01T12:50:00.000Z"),
                  entry("MYD03.A2026305.0950.007.2026306150500.nc", "2026-11-01T09:50:00.000Z")],
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
    items = ModisConnector().search(day_night="day")
    # by time, both satellites; the 12:55 Aqua granule has no MYD03 yet
    assert [i["id"][:8] for i in items] == ["MYD02QKM", "MOD02QKM", "MYD02QKM"]
    assert set(items[1]["assets"]) == {"MOD02QKM", "MOD03"}
    assert {c["short_name"] for c in calls} == {"MOD02QKM", "MOD03", "MYD02QKM", "MYD03"}
    assert all(c["version"] == "7" for c in calls)
    calls.clear()
    assert [i["id"] for i in ModisConnector("terra").search()] == [fx.TERRA[:-3]]
    assert {c["short_name"] for c in calls} == {"MOD02QKM", "MOD03"}


def test_connector_satellites():
    assert get_connector("modis-qkm").short_names == ("MOD02QKM", "MOD03", "MYD02QKM", "MYD03")
    assert get_connector("modis-qkm", satellite="aqua").short_names == ("MYD02QKM", "MYD03")
    with pytest.raises(ValueError, match="unknown satellite"):
        ModisConnector("suomi")
