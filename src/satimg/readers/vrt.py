"""In-memory GDAL VRTs that georeference rasters GDAL cannot on its own."""

from __future__ import annotations

from typing import TYPE_CHECKING
from xml.sax.saxutils import escape

import numpy
from rasterio.control import GroundControlPoint

if TYPE_CHECKING:  # pragma: no cover
    from satimg.transform.transformer import Transformer


def gcps(transformer: Transformer, shape: tuple[int, int], per_side: int = 5) -> list[GroundControlPoint]:
    """A ``per_side`` x ``per_side`` grid of ground control points across a
    ``shape`` image, placed by ``transformer`` -- just enough for GDAL to see
    a swath raster (VIIRS, OLCI) as georeferenced in :func:`vrt`; satimg's
    own transformer is what actually maps it."""
    height, width = shape
    rows = numpy.linspace(0, height, per_side)
    cols = numpy.linspace(0, width, per_side)
    rr, cc = (a.ravel() for a in numpy.meshgrid(rows, cols, indexing="ij"))
    latlon = transformer.rowcol_to_latlon(numpy.column_stack((rr, cc)))
    return [GroundControlPoint(row=r, col=c, x=lon, y=lat) for r, c, (lat, lon) in zip(rr, cc, latlon)]


def vrt(
    path: str, shape: tuple[int, int], *, transform=None, crs=None, gcps=None, dtype: str = "Float32", nodata="nan"
) -> str:
    """``path`` -- a ``float32`` raster GDAL opens but cannot georeference,
    e.g. an HDF5 dataset -- wrapped in an in-memory VRT that carries the
    georeferencing satimg knows instead: an affine ``transform`` and ``crs``,
    or a list of ``rasterio`` ``GroundControlPoint`` s (WGS84). So it opens
    georeferenced rather than as a bare array that every lazy read warns
    about, and with an affine transform ``x`` / ``y`` are real map
    coordinates."""
    height, width = shape
    georef = ""
    if transform is not None:
        georef = (
            f"<SRS>{escape(crs.to_wkt())}</SRS>"
            f"<GeoTransform>{', '.join(repr(v) for v in transform.to_gdal())}</GeoTransform>"
        )
    elif gcps:
        points = "".join(
            f'<GCP Id="{i}" Pixel="{g.col}" Line="{g.row}" X="{g.x}" Y="{g.y}"/>' for i, g in enumerate(gcps)
        )
        georef = f'<GCPList Projection="EPSG:4326">{points}</GCPList>'
    return (
        f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">{georef}'
        f'<VRTRasterBand dataType="{dtype}" band="1"><NoDataValue>{nodata}</NoDataValue>'
        f'<SimpleSource><SourceFilename relativeToVRT="0">{escape(path, {chr(34): "&quot;"})}</SourceFilename>'
        f"<SourceBand>1</SourceBand></SimpleSource></VRTRasterBand></VRTDataset>"
    )
