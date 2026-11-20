"""NISAR connector: search the ASF DAAC, download from NASA Earthdata.

``search`` is anonymous (the ASF search API). ``download`` needs a free NASA
Earthdata Login: an EDL bearer token, or a username and password. Pass them
explicitly::

    NisarGCOVConnector(token="eyJ...")
    NisarGCOVConnector(username="me", password="...")

or set ``EARTHDATA_TOKEN`` -- or ``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD``
-- in the environment.

Unlike the Copernicus / USGS connectors, ``download`` writes the product's
own single ``.h5`` file, not a zip: ``satimg.open`` reads it as it is.
"""

from __future__ import annotations

from satimg.connectors.asf_connector import SEARCH_URL, ASFConnector, _wkt  # noqa: F401  (re-exported)

import requests  # noqa: F401  (kept importable here: tests patch ``requests.get`` through this module)


class NisarGCOVConnector(ASFConnector):
    """NISAR L-band products from the ASF DAAC; GCOV by default (pass
    ``processing_level=`` for another, e.g. ``"RSLC"`` -- though only GCOV
    opens as a product)."""

    collection = "nisar-gcov"
    DATASET = "NISAR"
    PROCESSING_LEVEL = "GCOV"
