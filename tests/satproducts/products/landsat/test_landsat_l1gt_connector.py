from satproducts.products.landsat.landsat_l1gt_connector import LandsatL1GTConnector
from tests.satproducts.products.connector_test import ConnectorTest


class LandsatL1GTConnectorTest(ConnectorTest):
    PRODUCT_ID = "LC09_L1TP_194022_20251013_20251013_02_T1"
    CONNECTOR = LandsatL1GTConnector
