from enum import Enum


class SatProductType(Enum):
    SENTINEL_1_IW = "Sentinel-1-IW"
    SENTINEL_1_EW = "Sentinel-1-EW"
    SENTINEL_2_L1C = "Sentinel-2-L1C"
    LANDSAT = "Landsat"
