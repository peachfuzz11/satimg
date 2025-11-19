from satproducts.products.sentinel.sentinel_connector import SentinelConnector


class Sentinel1IWConnector(SentinelConnector):
    collection = "sentinel-1-grd"
    product_type = "IW_GRDH_1S"
