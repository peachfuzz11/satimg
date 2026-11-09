"""Rebuilding a stored :class:`~satimg.transform.transformer.Transformer` of any kind."""

from __future__ import annotations

from satimg.transform.affine_transformer import AffineTransformer
from satimg.transform.gcp_transformer import GCPTransformer
from satimg.transform.swath_transformer import SwathTransformer
from satimg.transform.transformer import Transformer


def load_transformer(data: dict) -> Transformer:
    """The transformer whose :meth:`~Transformer.to_dict` gave ``data``,
    of whichever kind it was."""
    if "transform" in data:
        return AffineTransformer.from_dict(data)
    if "gcps" in data:
        return GCPTransformer.from_dict(data)
    if "swath" in data:
        return SwathTransformer.from_dict(data)
    raise ValueError(f"not a stored transformer: keys {sorted(data)}")
