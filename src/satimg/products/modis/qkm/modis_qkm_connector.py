"""MODIS connector: Collection 7 L1B 250 m granules -- ``MOD02QKM`` (Terra)
and ``MYD02QKM`` (Aqua), each with its ``MOD03`` / ``MYD03`` geolocation --
from LAADS DAAC.

Search is anonymous, through NASA's CMR; download needs an Earthdata Login
(see :mod:`satimg.connectors.nasa_earthdata_connector`) and writes the granule's two ``.nc``
files into one zip ``satimg.open`` reads::

    connector = get_connector("modis-qkm")                   # Terra and Aqua; satellite="terra" / "aqua"
    items = connector.search(geojson=aoi, start_datetime=t0, stop_datetime=t1, day_night="day")
    connector.download(items[0], "/data/granule.zip")

Only Collection 7 (netCDF4) is searched: 6.1 is HDF4, which satimg does not
read.
"""

from __future__ import annotations

from satimg.connectors.nasa_earthdata_connector import CMRConnector

#: satellite -> (250 m L1B, geolocation) short names
SHORT_NAMES = {"terra": ("MOD02QKM", "MOD03"), "aqua": ("MYD02QKM", "MYD03")}


class ModisQKMConnector(CMRConnector):
    """MODIS 250 m L1B granules from Terra and Aqua (``satellite="both"``,
    the default) or one of them. Only day-mode granules carry 250 m data,
    so pass ``day_night="day"``."""

    collection = "modis-qkm"
    version = "7"

    def __init__(
        self,
        satellite: str = "both",
        username: str | None = None,
        password: str | None = None,
        token: str | None = None,
    ):
        if satellite == "both":
            self._satellites = tuple(SHORT_NAMES)
        elif satellite in SHORT_NAMES:
            self._satellites = (satellite,)
        else:
            raise ValueError(f"unknown satellite {satellite!r}; known: both, {', '.join(SHORT_NAMES)}")
        super().__init__(username, password, token)
        self.satellite = satellite
        self.short_names = tuple(name for s in self._satellites for name in SHORT_NAMES[s])

    def _groups(self) -> list[tuple[str, ...]]:
        # Terra and Aqua are alternatives, each granule an L1B + geolocation pair
        return [SHORT_NAMES[s] for s in self._satellites]
