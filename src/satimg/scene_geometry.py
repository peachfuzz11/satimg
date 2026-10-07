"""A product's scene geometry, detached from the product files.

:class:`SceneGeometry` holds what it takes to map pixels to coordinates and
sample per-pixel metadata -- the :class:`~satimg.transform.Transformer`, the
image shape, the coarse metadata grids and the scene-wide attrs -- and
round-trips through a plain JSON-serialisable dict. An app can store it while
the product is open and keep using it after the product file is gone::

    with satimg.open(path) as product:
        stored = product.geometry.to_dict()

    geometry = SceneGeometry.from_dict(stored)    # the right subclass, e.g. Sentinel1Geometry
    geometry.transformer.latlon_to_rowcol((lat, lon))
    geometry.metadata.incidence_angle.at((row, col))
"""

from __future__ import annotations

import math

import numpy
from rasterio.crs import CRS
from rasterio.transform import Affine

from satimg.metadata import Field, Metadata
from satimg.transform import GCPTransformer, Transformer


def _nan_to_none(values) -> list:
    """Nested lists with ``NaN`` as ``None`` -- strict JSON (and PostgreSQL's
    ``jsonb``) has no ``NaN``."""
    return [
        _nan_to_none(v) if isinstance(v, list) else (None if math.isnan(v) else v)
        for v in values
    ]


def _none_to_nan(values) -> numpy.ndarray:
    return numpy.array(values, dtype=float)  # None -> nan


class SceneGeometry:
    """Pixel <-> lat/lon conversion plus per-pixel metadata of one scene.

    ``fields`` maps a field name to ``(rows, cols, values, units)`` -- the
    coarse grid a :class:`~satimg.metadata.Field` interpolates -- and ``shape``
    is the full image's ``(height, width)``.
    """

    #: the ``"type"`` :meth:`to_dict` writes and :meth:`from_dict` dispatches on
    type_name = "scene"
    _types: dict[str, type[SceneGeometry]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        SceneGeometry._types[cls.type_name] = cls

    def __init__(self, transformer: Transformer, fields: dict, attrs: dict, shape: tuple[int, int]):
        self._transformer = transformer
        self._fields = {
            name: (
                numpy.asarray(rows, dtype=float),
                numpy.asarray(cols, dtype=float),
                numpy.asarray(values, dtype=float),
                units,
            )
            for name, (rows, cols, values, units) in fields.items()
        }
        self._attrs = dict(attrs)
        self._shape = (int(shape[0]), int(shape[1]))
        self._metadata = None

    @classmethod
    def from_metadata(
        cls, transformer: Transformer, metadata: Metadata, shape: tuple[int, int]
    ) -> SceneGeometry:
        """Capture an open product's ``transformer`` / ``metadata`` / shape."""
        fields = {
            name: (f._rows, f._cols, f._values, f.units)
            for name, f in ((name, metadata[name]) for name in metadata)
        }
        return cls(transformer, fields, metadata.attrs, shape)

    # -- serialisation --------------------------------------------------
    def to_dict(self) -> dict:
        """A JSON-serialisable dict (no ``NaN``) that :meth:`from_dict` turns
        back into an equal geometry."""
        return {
            "type": self.type_name,
            "shape": list(self._shape),
            "transformer": _transformer_to_dict(self._transformer),
            "fields": {
                name: {
                    "rows": rows.tolist(),
                    "cols": cols.tolist(),
                    "values": _nan_to_none(values.tolist()),
                    "units": units,
                }
                for name, (rows, cols, values, units) in self._fields.items()
            },
            "attrs": dict(self._attrs),
        }

    @classmethod
    def from_dict(cls, data: dict) -> SceneGeometry:
        """Rebuild a geometry from :meth:`to_dict`, as the subclass it was
        saved from (whichever class this is called on)."""
        sub = SceneGeometry._types[data.get("type", SceneGeometry.type_name)]
        return sub._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> SceneGeometry:
        fields = {
            name: (f["rows"], f["cols"], _none_to_nan(f["values"]), f["units"])
            for name, f in data["fields"].items()
        }
        return cls(
            _transformer_from_dict(data["transformer"]), fields, data["attrs"], tuple(data["shape"])
        )

    # -- shape / transform / metadata -----------------------------------
    @property
    def height(self) -> int:
        return self._shape[0]

    @property
    def width(self) -> int:
        return self._shape[1]

    @property
    def transformer(self) -> Transformer:
        return self._transformer

    @property
    def metadata(self) -> Metadata:
        if self._metadata is None:
            self._metadata = Metadata(
                {
                    name: Field(rows, cols, values, name=name, units=units, shape=self._shape)
                    for name, (rows, cols, values, units) in self._fields.items()
                },
                self._attrs,
            )
        return self._metadata


SceneGeometry._types[SceneGeometry.type_name] = SceneGeometry


def _transformer_to_dict(transformer: Transformer) -> dict:
    if isinstance(transformer, GCPTransformer):
        raise TypeError("a GCPTransformer is stored by its owning geometry (see Sentinel1Geometry)")
    return {"transform": list(transformer._transform)[:6], "crs": transformer._crs.to_wkt()}


def _transformer_from_dict(data: dict) -> Transformer:
    return Transformer(Affine(*data["transform"]), CRS.from_wkt(data["crs"]))


def coarsen(metadata: Metadata, shape: tuple[int, int], size: int) -> dict:
    """``metadata``'s fields resampled (bilinearly) onto an at most
    ``size`` x ``size`` grid spanning the full ``shape`` -- a
    :class:`SceneGeometry` ``fields`` dict, for products whose own grids are
    too fine to store. Smooth fields (viewing / sun angles) lose next to
    nothing."""
    height, width = shape
    rows = numpy.linspace(0, height - 1, min(size, height))
    cols = numpy.linspace(0, width - 1, min(size, width))
    rr, cc = numpy.meshgrid(rows, cols, indexing="ij")
    points = numpy.column_stack([rr.ravel(), cc.ravel()])
    fields = {}
    for name in metadata:
        field = metadata[name]
        values = numpy.asarray(field.at(points)).reshape(len(rows), len(cols))
        fields[name] = (rows, cols, values, field.units)
    return fields
