"""Where detections lie: on land, at a wind turbine, offshore platform or fish farm, in
a port or a harbor -- the :attr:`~satimg.prediction.types.Detection.tags` a detector sets."""

from __future__ import annotations

from functools import cache

import numpy

from satimg.fishfarms import FishFarmMask
from satimg.landmask import LandMask
from satimg.platforms import PlatformMask
from satimg.ports import HarborMask, PortMask
from satimg.windfarms import WindfarmMask

ON_LAND = "on_land"
TURBINE = "turbine"
PLATFORM = "platform"
FISHERY = "fishery"
IN_PORT = "in_port"
IN_HARBOR = "in_harbor"
# in context_tags' column order
_TAGS = (ON_LAND, TURBINE, PLATFORM, FISHERY, IN_PORT, IN_HARBOR)

# one instance each (the wind farms one per year), so the detector's tiles share them
_land = cache(LandMask)
_windfarms = cache(WindfarmMask)
_platforms = cache(PlatformMask)
_fish_farms = cache(FishFarmMask)
_ports = cache(PortMask)
_harbors = cache(HarborMask)


def context_tags(lat, lon, year: int | None = None) -> list[list[str]]:
    """The tags of each of the matching ``lat``/``lon`` points. A wind farm built
    after ``year`` doesn't count (any farm does when ``year`` is None), and a harbor
    wins over a port around it."""
    lat = numpy.asarray(lat, dtype=float).ravel()
    lon = numpy.asarray(lon, dtype=float).ravel()
    in_harbor = _harbors().contains(lat, lon)
    masks = numpy.stack([
        _land().contains(lat, lon),
        _windfarms(year).contains(lat, lon),
        _platforms().contains(lat, lon),
        _fish_farms().contains(lat, lon),
        _ports().contains(lat, lon) & ~in_harbor,
        in_harbor,
    ], axis=1)
    # plain lists: indexing a numpy array of strings row by row is ~50x slower
    return [[tag for tag, inside in zip(_TAGS, row) if inside] for row in masks.tolist()]
