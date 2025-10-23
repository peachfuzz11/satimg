from satproducts.products.sentinel.sentinel2.sentinel2_connector import Sentinel2Connector
from tests.satproducts.products.connector_test import ConnectorTest


class Sentinel2ConnectorTest(ConnectorTest):
    PRODUCT_ID = "S2A_MSIL1C_20220114T103401_N0510_R108_T33UUB_20240429T202900.SAFE"
    CONNECTOR = Sentinel2Connector
