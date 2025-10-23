from satproducts.products.sentinel.sentinel1.sentinel1_ew_connector import Sentinel1EWConnector
from tests.satproducts.products.connector_test import ConnectorTest


class Sentinel1EWConnectorTest(ConnectorTest):
    PRODUCT_ID = "S1C_EW_GRDM_1SDH_20251023T071648_20251023T071724_004686_009425_9F0F_COG.SAFE"
    CONNECTOR = Sentinel1EWConnector
