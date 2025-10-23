from satproducts.products.sentinel.sentinel_connector import SentinelConnector


class Sentinel2Connector(SentinelConnector):
    collection = "SENTINEL-2"
    product_type = "S2MSI1C"
