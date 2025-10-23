from satproducts.products.sentinel.sentinel1.sentinel1_iw_connector import Sentinel1IWConnector
from tests.satproducts.products.connector_test import ConnectorTest


class Sentinel1IWConnectorTest(ConnectorTest):
    PRODUCT_ID = "S1C_IW_GRDH_1SDV_20251023T054003_20251023T054028_004685_009421_03A5_COG.SAFE"
    CONNECTOR = Sentinel1IWConnector
