from satproducts.products.sentinel.sentinel_connector import SentinelConnector


class Sentinel1EWConnector(SentinelConnector):
    collection = "SENTINEL-1"
    product_type = "EW_GRDM_1S-COG"
