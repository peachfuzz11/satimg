import pytest

from satimg.connectors._credentials import CredentialsError
from satimg.products.landsat.c2l1.landsat_c2l1_connector import LandsatC2L1Connector
from satimg.products.sentinel2.l1c.sentinel2_l1c_connector import Sentinel2L1CConnector
from satimg.connectors.connector_factory import get_connector
from satimg.connectors._credentials import Credentials


class TestCredentials:
    def test_explicit_args_win(self, monkeypatch):
        monkeypatch.setenv("CDSE_USERNAME", "env-user")
        creds = Credentials.resolve(
            "arg-user", "arg-pw", service="x", env_prefix="CDSE"
        )
        assert (creds.username, creds.password) == ("arg-user", "arg-pw")

    def test_environment_fallback(self, monkeypatch):
        monkeypatch.setenv("CDSE_USERNAME", "env-user")
        monkeypatch.setenv("CDSE_PASSWORD", "env-pw")
        monkeypatch.setenv("CDSE_TOKEN", "env-tok")
        creds = Credentials.resolve(service="x", env_prefix="CDSE")
        assert creds == Credentials("env-user", "env-pw", "env-tok")

    def test_missing_raises_with_hint(self, monkeypatch):
        monkeypatch.delenv("CDSE_USERNAME", raising=False)
        monkeypatch.delenv("CDSE_PASSWORD", raising=False)
        with pytest.raises(CredentialsError, match="CDSE_USERNAME"):
            Credentials.resolve(service="Copernicus", env_prefix="CDSE")


class TestConnectors:
    def test_construction_is_offline_and_credential_free(self):
        # no network, no credentials needed just to build one
        get_connector("sentinel-1-grd")
        get_connector("landsat-c2l1")

    def test_download_without_credentials_raises(self, monkeypatch):
        for var in ("CDSE_USERNAME", "CDSE_PASSWORD"):
            monkeypatch.delenv(var, raising=False)
        with pytest.raises(CredentialsError):
            Sentinel2L1CConnector()._get_session()

    def test_credentials_passed_through_factory(self, monkeypatch):
        for var in ("USGS_USERNAME", "USGS_PASSWORD", "USGS_TOKEN"):
            monkeypatch.delenv(var, raising=False)
        connector = get_connector("landsat-c2l1", username="u", password="p", token="t")
        assert isinstance(connector, LandsatC2L1Connector)
        assert (connector._username, connector._password, connector._token) == ("u", "p", "t")

    def test_unknown_collection(self):
        with pytest.raises(ValueError, match="no connector"):
            get_connector("modis")


class _Response:
    def __init__(self, status=200, body=b"", location=None):
        self.status_code = status
        self.headers = {"Location": location} if location else {}
        self._body = body
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def iter_content(self, chunk_size):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class _Session:
    def __init__(self, responses):
        self._responses = responses
        self.urls = []

    def get(self, url, **kwargs):
        self.urls.append(url)
        return self._responses[url]


def _zip_bytes(members):
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buf.getvalue()


class TestDownloads:
    ITEM = {"id": "S2A_X", "assets": {"Product": {"href": "https://a/x"}}}

    def test_copernicus_follows_redirects_once_and_streams(self, tmp_path, monkeypatch):
        body = _zip_bytes({"S2A_X.SAFE/manifest.safe": b"<xml/>"})
        session = _Session({
            "https://a/x": _Response(302, location="https://b/y"),
            "https://b/y": _Response(200, body),
        })
        connector = Sentinel2L1CConnector()
        monkeypatch.setattr(connector, "_get_session", lambda: session)
        out = tmp_path / "x.zip"
        assert connector.download(self.ITEM, str(out)) == str(out)
        assert out.read_bytes() == body
        assert session.urls == ["https://a/x", "https://b/y"]  # the payload fetched once
        assert not (tmp_path / "x.zip.part").exists()

    def test_copernicus_bad_zip_leaves_nothing(self, tmp_path, monkeypatch):
        import zipfile

        session = _Session({"https://a/x": _Response(200, b"not a zip")})
        connector = Sentinel2L1CConnector()
        monkeypatch.setattr(connector, "_get_session", lambda: session)
        with pytest.raises(zipfile.BadZipFile):
            connector.download(self.ITEM, str(tmp_path / "x.zip"))
        assert list(tmp_path.iterdir()) == []

    def test_landsat_streams_assets_into_a_stored_zip(self, tmp_path, monkeypatch):
        import zipfile

        item = {
            "id": "LC09_X",
            "assets": {
                "red": {"href": "https://u/red.TIF"},
                "MTL.json": {"href": "https://u/mtl.json"},
                "ignored": {"href": "https://u/other"},
            },
        }
        session = _Session({"https://u/red.TIF": _Response(200, b"tif"), "https://u/mtl.json": _Response(200, b"{}")})
        connector = LandsatC2L1Connector()
        monkeypatch.setattr(connector, "_get_session", lambda: session)
        out = tmp_path / "l.zip"
        connector.download(item, str(out))
        with zipfile.ZipFile(out) as archive:
            assert sorted(archive.namelist()) == ["LC09_X/MTL.json", "LC09_X/red.TIF"]
            assert archive.read("LC09_X/red.TIF") == b"tif"
            assert {i.compress_type for i in archive.infolist()} == {zipfile.ZIP_STORED}
        assert not (tmp_path / "l.zip.part").exists()

    def test_landsat_failed_asset_leaves_nothing(self, tmp_path, monkeypatch):
        item = {"id": "LC09_X", "assets": {"red": {"href": "https://u/red.TIF"}}}
        session = _Session({"https://u/red.TIF": _Response(500)})
        connector = LandsatC2L1Connector()
        monkeypatch.setattr(connector, "_get_session", lambda: session)
        with pytest.raises(RuntimeError):
            connector.download(item, str(tmp_path / "l.zip"))
        assert list(tmp_path.iterdir()) == []


class TestASF:
    def test_auth_follows_asf_host_hops_but_not_cloud_storage(self):
        import requests
        from satimg.connectors.nasa_earthdata_connector import EarthdataSession

        def hop(old, new):
            prepared = requests.Request("GET", new, headers={"Authorization": "x"}).prepare()
            response = requests.Response()
            response.request = requests.Request("GET", old).prepare()
            EarthdataSession().rebuild_auth(prepared, response)
            return "Authorization" in prepared.headers

        assert hop("https://datapool.asf.alaska.edu/a.zip", "https://sentinel1.asf.alaska.edu/a.zip")
        assert hop("https://sentinel1.asf.alaska.edu/a.zip", "https://urs.earthdata.nasa.gov/oauth/authorize")
        assert not hop("https://sentinel1.asf.alaska.edu/a.zip", "https://abc.cloudfront.net/a.zip")

    def test_factory_builds_both_asf_connectors_offline(self):
        from satimg.products.nisar.gcov.nisar_gcov_connector import NisarGCOVConnector
        from satimg.products.sentinel1.grd.sentinel1_grd_asf_connector import Sentinel1GRDASFConnector

        assert isinstance(get_connector("sentinel-1-grd-asf"), Sentinel1GRDASFConnector)
        assert isinstance(get_connector("nisar-gcov"), NisarGCOVConnector)

    def test_sentinel1_search_asks_asf_for_iw_grd_hd(self, monkeypatch):
        from satimg.connectors import asf_connector

        seen = {}

        class Reply:
            def raise_for_status(self):
                pass

            def json(self):
                return {"features": [{"properties": {"sceneName": "S1B_X", "url": "https://asf/S1B_X.zip"}}]}

        def get(url, params, timeout):
            seen.update(url=url, params=params)
            return Reply()

        monkeypatch.setattr(asf_connector.requests, "get", get)
        items = get_connector("sentinel-1-grd-asf").search(ids=["S1B_X"], platform="S1B")
        assert seen["url"] == asf_connector.SEARCH_URL
        assert seen["params"] == {"dataset": "SENTINEL-1", "processingLevel": "GRD_HD", "output": "geojson",
                                  "beamMode": "IW", "platform": "S1B", "granule_list": "S1B_X"}
        assert [(i["id"], i["assets"]["data"]["href"]) for i in items] == [("S1B_X", "https://asf/S1B_X.zip")]

    def test_nisar_keeps_its_dataset_and_level(self, monkeypatch):
        from satimg.connectors import asf_connector

        seen = {}

        class Reply:
            def raise_for_status(self):
                pass

            def json(self):
                return {"features": []}

        monkeypatch.setattr(asf_connector.requests, "get", lambda url, params, timeout: seen.update(params=params) or Reply())
        get_connector("nisar-gcov").search()
        assert (seen["params"]["dataset"], seen["params"]["processingLevel"]) == ("NISAR", "GCOV")

    def test_download_needs_a_login(self, monkeypatch):
        for var in ("EARTHDATA_TOKEN", "EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"):
            monkeypatch.delenv(var, raising=False)
        with pytest.raises(CredentialsError):
            get_connector("sentinel-1-grd-asf")._get_session()

    def test_download_streams_to_the_file_and_leaves_nothing_on_failure(self, tmp_path, monkeypatch):
        from satimg.connectors import asf_connector

        connector = get_connector("sentinel-1-grd-asf", token="t")
        item = {"id": "S1B_X", "assets": {"data": {"href": "https://asf/S1B_X.zip"}}}
        monkeypatch.setattr(asf_connector, "stream", lambda session, url, f: f.write(b"zip"))
        out = tmp_path / "x.zip"
        assert connector.download(item, str(out)) == str(out)
        assert out.read_bytes() == b"zip"
        assert not (tmp_path / "x.zip.part").exists()

        def fail(session, url, f):
            f.write(b"par")
            raise RuntimeError("boom")

        monkeypatch.setattr(asf_connector, "stream", fail)
        with pytest.raises(RuntimeError):
            connector.download(item, str(tmp_path / "y.zip"))
        assert sorted(p.name for p in tmp_path.iterdir()) == ["x.zip"]
