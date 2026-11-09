"""VIIRS Day/Night Band connector: NASA's near-real-time L1B from NOAA-20
(``VJ102DNB_NRT`` + ``VJ103DNB_NRT``) and NOAA-21 (``VJ202DNB_NRT`` +
``VJ203DNB_NRT``), from LANCE, a few hours after acquisition.

Search is anonymous, through NASA's CMR -- catalogued, so a footprint or
night-only filter costs nothing; download needs an Earthdata Login (see
:mod:`satimg.connectors.nasa_earthdata_connector`) and writes each granule's radiance and
geolocation files into one zip ``satimg.open`` reads::

    connector = get_connector("viirs-dnb")                     # NOAA-20 and NOAA-21; satellite= for one
    items = connector.search(geojson=aoi, start_datetime=t0, stop_datetime=t1, day_night="night")
    connector.download(items[0], "/data/granule.zip")

LANCE keeps near-real-time files for a limited time; ``nrt=False`` searches
the standard LAADS archive of the same granules (``VJ102DNB`` / ``VJ202DNB``)
instead, for older dates.
"""

from __future__ import annotations

from satimg.connectors.nasa_earthdata_connector import CMRConnector

#: satellite -> (radiance, geolocation) short-name stems
SHORT_NAMES = {
    "noaa-20": ("VJ102DNB", "VJ103DNB"),
    "noaa-21": ("VJ202DNB", "VJ203DNB"),
}


class ViirsDNBConnector(CMRConnector):
    """VIIRS DNB L1B granule pairs (6 minutes, radiance + geolocation) from
    NOAA-20 and NOAA-21 (``satellite="both"``, the default) or one of them,
    near-real-time unless ``nrt=False``. ``day_night="night"`` keeps the
    granules boat lights show in."""

    collection = "viirs-dnb"

    def __init__(
        self,
        satellite: str = "both",
        nrt: bool = True,
        username: str | None = None,
        password: str | None = None,
        token: str | None = None,
    ):
        if satellite == "both":
            satellites = tuple(SHORT_NAMES)
        elif satellite in SHORT_NAMES:
            satellites = (satellite,)
        else:
            raise ValueError(f"unknown satellite {satellite!r}; known: both, {', '.join(SHORT_NAMES)}")
        super().__init__(username, password, token)
        self.satellite = satellite
        self.nrt = nrt
        suffix = "_NRT" if nrt else ""
        self._pairs = [tuple(f"{stem}{suffix}" for stem in SHORT_NAMES[s]) for s in satellites]
        self.short_names = tuple(name for pair in self._pairs for name in pair)

    def _groups(self) -> list[tuple[str, ...]]:
        # NOAA-20 and NOAA-21 are alternatives, each granule a radiance + geolocation pair
        return self._pairs
