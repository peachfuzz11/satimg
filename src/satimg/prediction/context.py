"""Where detections lie: on land, at a wind turbine or offshore platform, in a port or
a harbor -- the :attr:`~satimg.prediction.types.Detection.tags` a detector sets."""

from __future__ import annotations

from functools import cache

import numpy

from satimg.landmask import LandMask
from satimg.platforms import PlatformMask
from satimg.ports import HARBOR, PORT, PortMask
from satimg.windfarms import WindfarmMask

ON_LAND = "on_land"
TURBINE = "turbine"
PLATFORM = "platform"
IN_PORT = "in_port"
IN_HARBOR = "in_harbor"

# each mask loads its data once per process
_land = cache(LandMask)
_windfarms = cache(WindfarmMask)
_platforms = cache(PlatformMask)
_ports = cache(PortMask)


def context_tags(lon, lat, year: int | None = None) -> list[list[str]]:
    """The tags of each of the matching ``lon``/``lat`` points. A wind farm built
    after ``year`` doesn't count (any farm does when ``year`` is None)."""
    lon = numpy.asarray(lon, dtype=float).ravel()
    lat = numpy.asarray(lat, dtype=float).ravel()
    kinds = _ports().kinds_many(lon, lat)
    masks = [
        (ON_LAND, _land().contains_many(lon, lat)),
        (TURBINE, _windfarms().contains_many(lon, lat, year=year)),
        (PLATFORM, _platforms().contains_many(lon, lat)),
        (IN_PORT, kinds == PORT),
        (IN_HARBOR, kinds == HARBOR),
    ]
    return [[tag for tag, mask in masks if mask[i]] for i in range(lon.size)]
