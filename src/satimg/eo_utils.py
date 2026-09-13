"""EO/optical-specific per-pixel geometry equations -- sibling to
:mod:`satimg.sar_utils`.

Today that's just the trivial case: Sentinel-2 and Landsat rasters are
map-projected and north-up, so a compass heading needs no rotation to land in
the image's own pixel frame (contrast :func:`satimg.sar_utils.heading_in_image`,
which derives that rotation from the platform's own heading for imagery still
in native sensor geometry). This is also where future per-pixel corrections
that only make sense for pushbroom/framing optical sensors -- e.g. view-angle
or band-acquisition-timing effects, drawing on the ``view_zenith``/
``view_azimuth`` metadata Sentinel-2 and Landsat already carry -- will land.
"""

from __future__ import annotations

import numpy


def heading_in_image(heading_deg):
    """A compass heading (degrees clockwise from true north) is already the
    direction within a map-projected, north-up raster's own pixel frame --
    no rotation needed. Returns degrees wrapped to ``[0, 360)``.
    """
    return numpy.asarray(heading_deg, dtype=float) % 360.0
