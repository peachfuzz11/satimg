"""Sentinel-3 OLCI EFR from the Copernicus Data Space."""

from __future__ import annotations

from satimg.connectors.copernicus_dataspace_connector import (
    CopernicusDataspaceConnector,
)


class Sentinel3OLCIConnector(CopernicusDataspaceConnector):
    """Sentinel-3 OLCI Level-1B full resolution (``OL_1_EFR___``), A and B.
    ``timeliness="nrt"`` (the default) searches the near-real-time products,
    out within hours; ``"ntc"`` the consolidated non-time-critical ones, out
    within days."""

    COLLECTIONS = {"nrt": "sentinel-3-olci-1-efr-nrt", "ntc": "sentinel-3-olci-1-efr-ntc"}

    def __init__(self, username: str | None = None, password: str | None = None, timeliness: str = "nrt"):
        super().__init__(username, password)
        try:
            self.collection = self.COLLECTIONS[timeliness]
        except KeyError:
            raise ValueError(f"unknown timeliness {timeliness!r}; known: {sorted(self.COLLECTIONS)}") from None
        self.timeliness = timeliness
