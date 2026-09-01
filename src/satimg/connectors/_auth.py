"""USGS ERS (EarthExplorer) form login -> authenticated ``requests.Session``.

Login-form handling adapted from the ``landsatlook-stac`` package.
"""

from __future__ import annotations

import time

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry

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
    _submit_login_form(s, username, password)
    # Heuristic settle
    time.sleep(0.3)
    if token:
        s.headers.update({"Authorization": f"Bearer {token}"})
    return s
