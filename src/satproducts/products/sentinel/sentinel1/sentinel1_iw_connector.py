from satproducts.products.sentinel.sentinel_connector import SentinelConnector


class Sentinel1IWConnector(SentinelConnector):
    collection = "SENTINEL-1"
    product_type = "IW_GRDH_1S-COG"
