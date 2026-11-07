"""Point masks over geographic data: every mask answers
``mask.contains(lat, lon)``.

- :class:`LandMask` -- land, by ``roaring_landmask``
- :class:`PortMask` / :class:`HarborMask` -- commercial ports, small-boat harbors
- :class:`PlatformMask` -- offshore platforms
- :class:`WindfarmMask` -- offshore wind farms
- :class:`FishFarmMask` -- fish farms
"""

from satimg.masks.base import Mask
from satimg.masks.fishfarms import FishFarmMask
from satimg.masks.land import LandMask
from satimg.masks.platforms import PlatformMask
from satimg.masks.ports import HarborMask, PortMask, features_in_bbox
from satimg.masks.windfarms import WindfarmMask

__all__ = ["Mask", "LandMask", "PortMask", "HarborMask", "PlatformMask", "WindfarmMask", "FishFarmMask", "features_in_bbox"]
