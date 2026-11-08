"""A product's bands: :class:`~satimg.readers.raster.Raster` s stacked into
one labelled ``(band, y, x)`` array."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import xarray

from satimg.readers.raster import CHUNK_PX, Raster, Tile, tile_xy
from satimg.tiling import as_band_yx, label_bands


@dataclass(frozen=True)
class BandStack:
    """``rasters`` stacked along ``band`` and named ``names`` (one per band
    across all of them, in order), with a ``wavelength_nm`` coordinate when
    ``wavelengths_nm`` is given.

    With ``match`` (an index into ``rasters``) bands of other resolutions
    are nearest-neighbour resampled onto that raster's grid (Sentinel-2's
    10 / 20 / 60 m bands, Landsat's 30 m bands onto the 15 m pan grid).
    """

    rasters: Sequence[Raster]
    names: Sequence[str]
    wavelengths_nm: Sequence[float] | None = None
    match: int | None = None

    @property
    def overviews(self) -> list[int]:
        """The decimation factors every raster has besides 1, ascending."""
        common = set.intersection(*(set(r.overviews) for r in self.rasters))
        return sorted(common)

    def open(self, tile: Tile = CHUNK_PX, overview: int = 1) -> xarray.DataArray:
        """Every band, lazy, dask-chunked to ``tile`` (all bands in one
        chunk), at ``overview``. ``close()`` shuts every raster it opened.

        With ``match``, rasters are chunked at the fixed :data:`CHUNK_PX` --
        not ``tile`` -- for the resampling: ``reindex_like``'s
        nearest-neighbour lookup builds a dask task per output chunk, so
        aligning directly at a small ``tile`` (a 64 px patch loop) can turn
        one array into hundreds of thousands of chunks and take minutes just
        to build the graph. The result is rechunked to ``tile`` in one final
        step instead, which reads nothing.
        """
        opened = [r.open(tile if self.match is None else CHUNK_PX, overview) for r in self.rasters]
        if self.match is None:
            merged = opened[0] if len(opened) == 1 else xarray.concat(opened, dim="band")
        else:
            target = opened[self.match]
            merged = xarray.concat([b.reindex_like(target, method="nearest") for b in opened], dim="band")
            tw, th = tile_xy(tile)
            merged = merged.chunk({"band": -1, "x": tw, "y": th})
        da = label_bands(as_band_yx(merged), self.names)
        if self.wavelengths_nm is not None:
            da = da.assign_coords(wavelength_nm=("band", list(self.wavelengths_nm)))
        da.set_close(lambda: [band.close() for band in opened])
        return da
