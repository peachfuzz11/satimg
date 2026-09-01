import pytest

from satimg.connectors import (
    CredentialsError,
    LandsatConnector,
    Sentinel2Connector,
    get_connector,
)
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
            Sentinel2Connector()._get_session()

    def test_credentials_passed_through_factory(self, monkeypatch):
        for var in ("USGS_USERNAME", "USGS_PASSWORD", "USGS_TOKEN"):
            monkeypatch.delenv(var, raising=False)
        connector = get_connector("landsat-c2l1", username="u", password="p", token="t")
        assert isinstance(connector, LandsatConnector)
        assert (connector._username, connector._password, connector._token) == ("u", "p", "t")

    def test_unknown_collection(self):
        with pytest.raises(ValueError, match="no connector"):
            get_connector("modis")
