"""The library logs under the ``satimg`` namespace and installs no handler of
its own. These tests pin that contract and a few high-value records."""

import logging

import pytest

import satimg  # noqa: F401  (import triggers the NullHandler registration)
from satimg.connectors import _auth
from satimg.connectors._credentials import Credentials
from satimg.registry import resolve


def test_root_logger_has_only_a_null_handler():
    handlers = logging.getLogger("satimg").handlers
    assert len(handlers) == 1
    assert isinstance(handlers[0], logging.NullHandler)


class _FakeResponse:
    text = "<html><body>no form here</body></html>"

    def raise_for_status(self):
        pass


class _FakeSession:
    """Just enough of ``requests.Session`` for ``_submit_login_form``."""

    def __init__(self):
        self.calls = []

    def get(self, *args, **kwargs):
        self.calls.append(("get", args, kwargs))
        return _FakeResponse()

    def post(self, *args, **kwargs):
        self.calls.append(("post", args, kwargs))
        return _FakeResponse()


def test_login_form_missing_warns(caplog):
    caplog.set_level(logging.WARNING, logger="satimg")
    _auth._submit_login_form(_FakeSession(), "user", "pw")
    assert any(
        r.levelno == logging.WARNING and "login form not found" in r.message
        for r in caplog.records
    )


def test_credential_source_is_logged_explicit(caplog):
    caplog.set_level(logging.DEBUG, logger="satimg")
    Credentials.resolve("arg-user", "arg-pw", service="Copernicus", env_prefix="CDSE")
    assert any(
        "using explicit credentials for Copernicus" in r.message for r in caplog.records
    )


def test_credential_source_is_logged_from_env(caplog, monkeypatch):
    monkeypatch.setenv("CDSE_USERNAME", "env-user")
    monkeypatch.setenv("CDSE_PASSWORD", "env-pw")
    caplog.set_level(logging.DEBUG, logger="satimg")
    Credentials.resolve(service="Copernicus", env_prefix="CDSE")
    assert any(
        "from CDSE_* environment variables" in r.message for r in caplog.records
    )
    # the values themselves must never be logged
    assert not any("env-pw" in r.message for r in caplog.records)


def test_registry_resolution_is_logged(caplog):
    caplog.set_level(logging.DEBUG, logger="satimg")
    resolve("S2A_MSIL1C_20220114T103401_N0301_R108_T33UUB_20220114T123457.SAFE")
    assert any(
        "Sentinel2L1CProduct" in r.message and "pattern" in r.message
        for r in caplog.records
    )
