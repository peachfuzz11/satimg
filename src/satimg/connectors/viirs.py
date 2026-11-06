"""VIIRS Day/Night Band L1B connector: NASA's ``VNP02DNB`` + ``VNP03DNB``
(or the NOAA-20 / -21 ``VJ1`` / ``VJ2`` pairs) from LAADS DAAC.

Search is anonymous, through NASA's CMR -- catalogued, so a footprint or
night-only filter costs nothing; download needs an Earthdata Login (see
:mod:`satimg.connectors.earthdata`) and writes the granule's two files into
one zip ``satimg.open`` reads::

    connector = get_connector("viirs-dnb")                     # SNPP; satellite="noaa-20" / "noaa-21"
    items = connector.search(geojson=aoi, start_datetime=t0, stop_datetime=t1, day_night="night")
    connector.download(items[0], "/data/granule.zip")
"""

from __future__ import annotations

from satimg.connectors.earthdata import CMRConnector

#: satellite -> (L1B short name, geolocation short name)
SHORT_NAMES = {
    "snpp": ("VNP02DNB", "VNP03DNB"),
    "noaa-20": ("VJ102DNB", "VJ103DNB"),
    "noaa-21": ("VJ202DNB", "VJ203DNB"),
}


class ViirsDNBConnector(CMRConnector):
    """VIIRS DNB L1B granules (6 minutes, ``V??02DNB`` + its ``V??03DNB``)
    from one satellite, SNPP by default. ``day_night="night"`` keeps the
    granules boat lights show in."""

    collection = "viirs-dnb"

    def __init__(
        self,
        satellite: str = "snpp",
        username: str | None = None,
        password: str | None = None,
        token: str | None = None,
    ):
        try:
            self.short_names = SHORT_NAMES[satellite]
        except KeyError:
            raise ValueError(f"unknown satellite {satellite!r}; known: {sorted(SHORT_NAMES)}") from None
        super().__init__(username, password, token)
        self.satellite = satellite
