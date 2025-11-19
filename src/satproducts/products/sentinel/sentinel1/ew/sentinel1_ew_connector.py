from satproducts.products.sentinel.sentinel_connector import SentinelConnector


class Sentinel1EWConnector(SentinelConnector):
    collection = "sentinel-1-grd"
    product_type = "EW_GRDM_1S"
