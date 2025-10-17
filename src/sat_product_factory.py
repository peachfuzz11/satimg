import typing
from pathlib import Path

from handlers.iceye_handler import IceyeHandler
from handlers.landsat_handler import LandsatHandler
from handlers.rcm_handler import RCMHandler
from handlers.sentinel_handler import SentinelHandler
from handlers.synspective_handler import SynspectiveHandler
from handlers.umbra_handler import UmbraHandler
from products.base.image.raster_product import RasterProduct


class SatProductFactory:
    def __init__(self, product_path: typing.Union[Path, str]):
        self._product_path = product_path
        self._handler_chain = SentinelHandler( IceyeHandler(RCMHandler(SynspectiveHandler(LandsatHandler(UmbraHandler())))))

    def create(self) -> RasterProduct:
        product = self._handler_chain.handle(self._product_path)
        return product
