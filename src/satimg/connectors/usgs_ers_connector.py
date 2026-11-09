"""USGS connectors: a USGS ERS (EarthExplorer) account -> an authenticated
STAC session.

Downloads need a USGS EarthExplorer / ERS account. Pass it explicitly
(``username=``, ``password=``, optionally ``token=``) or set
``USGS_USERNAME`` / ``USGS_PASSWORD`` (and ``USGS_TOKEN``) in the environment.

Login-form handling adapted from the ``landsatlook-stac`` package.
"""

from __future__ import annotations

import logging
import time

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry

from satimg.connectors._credentials import Credentials
from satimg.connectors.connector import STACConnector

logger = logging.getLogger(__name__)

ERS_LOGIN_URL = "https://ers.cr.usgs.gov/login/"
USER_AGENT = "satimg (+python-requests)"


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s


def _submit_login_form(s: requests.Session, username: str, password: str) -> None:
    # Load login page to capture hidden inputs
    r = s.get(ERS_LOGIN_URL, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    form = soup.find("form")
    if form is None:
        logger.warning(
            "ERS login form not found at %s; falling back to direct POST",
            ERS_LOGIN_URL,
        )
        # Try direct post as fallback
        data = {"username": username, "password": password}
        r = s.post(ERS_LOGIN_URL, data=data, timeout=30, allow_redirects=True)
        r.raise_for_status()
        return

    data = {}
    for inp in form.find_all("input"):
        name = inp.get("name")
        if not name:
            continue
        data[name] = inp.get("value", "")
    data["username"] = username
    data["password"] = password

    action = form.get("action") or ERS_LOGIN_URL
    if action.startswith("/"):
        action = requests.compat.urljoin(ERS_LOGIN_URL, action)

    r = s.post(action, data=data, timeout=30, allow_redirects=True)
    r.raise_for_status()


def ers_login(username: str, password: str, token: str | None = None) -> requests.Session:
    """
    Log in to ERS and return an authenticated session.

    Args:
        username (str): USGS ERS username.
        password (str): USGS ERS password.
        token (str, optional): Optional bearer token for authorization header.

    Returns:
        requests.Session: Authenticated session with ERS cookies.

    Note:
        If token is provided and accepted by the host, it is attached as
        Authorization header.
    """
    s = make_session()
    logger.debug("submitting ERS login form for %s", username)
    _submit_login_form(s, username, password)
    # Heuristic settle
    time.sleep(0.3)
    if token:
        s.headers.update({"Authorization": f"Bearer {token}"})
    logger.debug("ERS login complete for %s", username)
    return s


class UsgsErsConnector(STACConnector):
    """A USGS STAC catalogue whose downloads log in with a USGS ERS account."""

    ENV_PREFIX = "USGS"
    SERVICE = "USGS EarthExplorer"

    def __init__(self, username: str | None = None, password: str | None = None, token: str | None = None):
        self._username = username
        self._password = password
        self._token = token

    def _get_session(self, *args, **kwargs) -> requests.Session:
        creds = Credentials.resolve(
            self._username,
            self._password,
            self._token,
            service=self.SERVICE,
            env_prefix=self.ENV_PREFIX,
        )
        logger.debug("logging in to %s as %s", self.SERVICE, creds.username)
        return ers_login(creds.username, creds.password, creds.token)
