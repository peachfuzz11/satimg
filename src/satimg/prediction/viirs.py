"""Boat lights in VIIRS Day/Night Band imagery.

At night, lit boats -- squid jiggers and other light-luring fisheries above
all, but any vessel with deck lights -- show in the DNB as spikes: a pixel or
two brighter than the sea around them. Hulls are far below the ~750 m
resolution, so this is not the image-model :class:`~satimg.prediction.Detector`
but a spike detector in the spirit of NOAA's VIIRS Boat Detection (VBD;
Elvidge et al. 2015, *Remote Sensing* 7(3)) -- not a reimplementation of it:

* the background is the median of the 5 x 5 window, its noise that window's
  robust spread (MAD), with :attr:`ViirsBoatConfig.noise_floor` under it;
* a light is a local maximum standing :attr:`~ViirsBoatConfig.min_contrast`
  above the background, :attr:`~ViirsBoatConfig.min_z` noise-levels clear of
  it, with a spike height index ``1 - background / peak`` of at least
  :attr:`~ViirsBoatConfig.min_shi` -- which a bump on moonlit cloud is not;
* an energetic-particle hit (common in the South Atlantic Anomaly) is a
  spike whose neighbours get almost none of its light -- a real light
  spreads over the next pixels through the optics -- and is dropped;
* columns far noisier than the rest (the DNB's outermost aggregation zone)
  and pixels not dark enough (solar zenith under
  :attr:`~ViirsBoatConfig.min_solar_zenith`) are skipped.

Every light is returned, tagged with where it lies like any detection
(:mod:`satimg.prediction.context`) -- on land is a town, at a platform
likely a gas flare -- so filter on the tags. Thresholds were set by eye on
NOAA-20 granules over the Sea of Japan / Yellow Sea and off Peru (November
2026); expect to tune them, and see the limitations in the README.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy
from numpy.lib.stride_tricks import sliding_window_view

from satimg.prediction.context import context_tags
from satimg.prediction.types import BBox, Coordinate, Detection, Label

logger = logging.getLogger(__name__)

#: nW / (cm^2 sr) per the SDR's W / (cm^2 sr)
NANOWATTS = 1e9


@dataclass
class ViirsBoatConfig:
    #: least peak-minus-background radiance, nW / (cm^2 sr)
    min_contrast: float = 1.0
    #: least contrast, in units of the local noise
    min_z: float = 8.0
    #: least spike height index, ``1 - background / peak``
    min_shi: float = 0.5
    #: below this a spike's brightest neighbour holds too little of its
    #: excess for a real (optically blurred) light: a particle hit
    min_spill: float = 0.1
    #: noise never assumed below this, nW / (cm^2 sr) -- the DNB's own floor
    noise_floor: float = 0.3
    #: pixels with the sun higher than this (degrees from zenith) are skipped
    min_solar_zenith: float = 100.0
    #: a column whose row-to-row noise is this many times the median is skipped
    noisy_column_factor: float = 3.0
    label: str = "boat"


def noisy_columns(radiance: numpy.ndarray, factor: float) -> numpy.ndarray:
    """Columns whose median row-to-row change is over ``factor`` times the
    whole image's, or exactly zero (stuck) -- the swath-edge columns that are
    all noise."""
    spread = numpy.nanmedian(numpy.abs(numpy.diff(radiance, axis=0)), axis=0)
    return (spread > factor * numpy.nanmedian(spread)) | (spread == 0) | numpy.isnan(spread)


def find_spikes(radiance: numpy.ndarray, cfg: ViirsBoatConfig, block: int = 128) -> dict:
    """Spike features over a whole ``(rows, cols)`` radiance image in nW /
    (cm^2 sr): ``{"rows", "cols", "z", "shi", "contrast", "particle"}`` for
    every local maximum passing the contrast / z / SHI tests (particle hits
    included, flagged). Computed ``block`` rows at a time to bound memory."""
    height, width = radiance.shape
    x = numpy.nan_to_num(radiance.astype(numpy.float32))
    padded = numpy.pad(x, 2, mode="reflect")
    out = {k: [] for k in ("rows", "cols", "z", "shi", "contrast", "particle")}
    for top in range(0, height, block):
        stop = min(top + block, height)
        win = sliding_window_view(padded[top:stop + 4], (5, 5)).reshape(stop - top, width, 25)
        centre = x[top:stop]
        background = numpy.median(win, axis=2)
        mad = 1.4826 * numpy.median(numpy.abs(win - background[:, :, None]), axis=2)
        ring = numpy.delete(win.reshape(stop - top, width, 5, 5)[:, :, 1:4, 1:4].reshape(stop - top, width, 9), 4, axis=2)
        contrast = centre - background
        z = contrast / numpy.hypot(mad, cfg.noise_floor)
        shi = 1 - background / numpy.maximum(centre, 1e-6)
        hit = (
            (centre >= ring.max(axis=2)) & (contrast >= cfg.min_contrast)
            & (z >= cfg.min_z) & (shi >= cfg.min_shi)
        )
        r, c = numpy.nonzero(hit)
        spill = (ring[r, c] - background[r, c, None]).max(axis=1) / contrast[r, c]
        out["rows"].append(r + top)
        out["cols"].append(c)
        out["z"].append(z[r, c])
        out["shi"].append(shi[r, c])
        out["contrast"].append(contrast[r, c])
        out["particle"].append(spill < cfg.min_spill)
    return {k: numpy.concatenate(v) for k, v in out.items()}


class ViirsBoatDetector:
    """Boat lights over a :class:`~satimg.products.viirs.ViirsDNBProduct`;
    the same shape of API as :class:`~satimg.prediction.Detector`."""

    def __init__(self, product, **overrides):
        self._product = product
        self._config = ViirsBoatConfig(**overrides)

    def detect(self, **overrides) -> list[Detection]:
        cfg = ViirsBoatConfig(**{**self._config.__dict__, **overrides})
        product = self._product
        if product.metadata.attrs.get("day_night") == "day":
            logger.warning("%s is a daytime granule: no boat lights to find", product)
            return []
        radiance = product.raw.values[0] * NANOWATTS
        spikes = find_spikes(radiance, cfg)
        keep = ~spikes["particle"]
        keep &= ~noisy_columns(radiance, cfg.noisy_column_factor)[spikes["cols"]]
        rows, cols = spikes["rows"] + 0.5, spikes["cols"] + 0.5
        keep &= product.metadata.solar_zenith.at(numpy.column_stack((rows, cols))) >= cfg.min_solar_zenith
        logger.info(
            "%s: %d spike(s), %d particle hit(s) dropped, %d light(s)",
            product, len(rows), int(spikes["particle"].sum()), int(keep.sum()),
        )
        rows, cols, shi = rows[keep], cols[keep], spikes["shi"][keep]
        if not len(rows):
            return []
        latlon = product.transformer.rowcol_to_latlon(numpy.column_stack((rows, cols)))
        timestamp = product.timestamp
        tags = context_tags(latlon[:, 0], latlon[:, 1], year=timestamp.year if timestamp else None)
        return [
            Detection(
                BBox(c - 1.5, r - 1.5, c + 1.5, r + 1.5),
                Label(cfg.label, float(min(s, 1.0))),
                Coordinate(lat=float(lat), lon=float(lon)),
                t,
            )
            for r, c, s, (lat, lon), t in zip(rows, cols, shi, latlon, tags)
        ]
