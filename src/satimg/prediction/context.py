"""Where detections lie: on land, at a wind turbine or offshore platform, in a port or
a harbor -- the :attr:`~satimg.prediction.types.Detection.tags` a detector sets."""

from __future__ import annotations

from functools import cache

import numpy

from satimg.landmask import LandMask
from satimg.platforms import PlatformMask
from satimg.ports import HarborMask, PortMask
from satimg.windfarms import WindfarmMask

ON_LAND = "on_land"
TURBINE = "turbine"
PLATFORM = "platform"
IN_PORT = "in_port"
IN_HARBOR = "in_harbor"

# one instance each (the wind farms one per year), so the detector's tiles share them
_land = cache(LandMask)
_windfarms = cache(WindfarmMask)
_platforms = cache(PlatformMask)
_ports = cache(PortMask)
_harbors = cache(HarborMask)


def context_tags(lat, lon, year: int | None = None) -> list[list[str]]:
    """The tags of each of the matching ``lat``/``lon`` points. A wind farm built
    after ``year`` doesn't count (any farm does when ``year`` is None), and a harbor
    wins over a port around it."""
    lat = numpy.asarray(lat, dtype=float).ravel()
    lon = numpy.asarray(lon, dtype=float).ravel()
    in_harbor = _harbors().contains(lat, lon)
    masks = [
        (ON_LAND, _land().contains(lat, lon)),
        (TURBINE, _windfarms(year).contains(lat, lon)),
        (PLATFORM, _platforms().contains(lat, lon)),
        (IN_PORT, _ports().contains(lat, lon) & ~in_harbor),
        (IN_HARBOR, in_harbor),
    ]
    return [[tag for tag, mask in masks if mask[i]] for i in range(lat.size)]
