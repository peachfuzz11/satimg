"""Pixel <-> geographic coordinate conversion.

One :class:`Transformer` interface, one module per way a product is
georeferenced:

- :class:`AffineTransformer` -- a map-projected raster's geotransform + CRS
- :class:`GCPTransformer` -- ground control points (Sentinel-1)
- :class:`SwathTransformer` -- a sensor-geometry swath's per-pixel
  geolocation (VIIRS, MODIS, OLCI)

``Transformer.from_dict`` rebuilds any of them from its ``to_dict``.
"""

from satimg.transform.affine import AffineTransformer
from satimg.transform.base import Coords, Transformer, as_n2
from satimg.transform.gcp import GCPTransformer
from satimg.transform.swath import SwathTransformer

__all__ = ["Transformer", "AffineTransformer", "GCPTransformer", "SwathTransformer", "Coords", "as_n2"]
