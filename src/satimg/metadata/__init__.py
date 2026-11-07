"""Per-pixel ancillary metadata (incidence angle, sun/view angles, ...).

Every sensor ships geometry that varies across the scene but is stored on a
*coarse* grid -- Sentinel-1's ``geolocationGridPoint`` list, Sentinel-2's 23x23
angle grids, Landsat's decimated angle rasters. :class:`Metadata` exposes those as
named :class:`Field` s that interpolate to any pixel, the same way
:class:`~satimg.transform.Transformer` converts coordinates::

    m = product.metadata
    m.fields                       # ['incidence_angle', 'slant_range_time', ...]
    m.incidence_angle.at((row, col))   # -> float, bilinear on the coarse grid
    m.sample((row, col))           # -> {field: value} for every field
    m.corners()                    # -> {field: {'top_left': ..., ..., 'center': ...}}
    m.attrs                        # scalar scene-level metadata

Each field also materialises as a lazy full-grid ``(y, x)``
:class:`xarray.DataArray` (``m.incidence_angle.grid``), which is what backs
``patch.meta`` when walking a product in windows::

    for patch in product.patches(512):
        patch.meta.incidence_angle     # lazy (h, w) DataArray for this window
        patch.meta.at((y, x))          # -> {field: value} at a patch pixel
"""

from satimg.metadata.collection import Metadata
from satimg.metadata.field import CORNER_KEYS, Field
from satimg.metadata.grids import bilinear, fill_nan_nearest, grid_from_points, regular_axis
from satimg.metadata.patch import PatchMeta

__all__ = [
    "Metadata", "Field", "PatchMeta", "CORNER_KEYS",
    "bilinear", "fill_nan_nearest", "grid_from_points", "regular_axis",
]
