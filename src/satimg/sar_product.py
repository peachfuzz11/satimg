"""The interface every SAR product shares."""

from __future__ import annotations

from satimg.product import Product
from satimg.sar_orbit_geometry import SAROrbitGeometry


class SARProduct(Product):
    """A SAR :class:`~satimg.product.Product`: the moving-target Doppler
    correction and heading conversions, the same on every SAR product
    (Sentinel-1, NISAR, Umbra, Capella, ICEYE), all delegated to its
    :class:`~satimg.sar_orbit_geometry.SAROrbitGeometry`.

    A subclass sets ``_geometry`` in its ``__init__``: the scene's
    transformer, metadata fields and attrs -- ``attrs["orbit"]`` the sensor
    orbit -- and shape.

    The usual use: a target whose speed and heading are known (AIS, say)
    is displaced along track in the image by its own Doppler shift;
    :meth:`correct_position` turns where it was seen into where it was.
    """

    _geometry: SAROrbitGeometry

    @property
    def geometry(self) -> SAROrbitGeometry:
        """As :attr:`Product.geometry`, as a
        :class:`~satimg.sar_orbit_geometry.SAROrbitGeometry`: stored with
        ``to_dict()``, it keeps the methods below after the file is gone."""
        return self._geometry

    @property
    def height(self) -> int:
        return self._geometry.height

    @property
    def width(self) -> int:
        return self._geometry.width

    @property
    def transformer(self):
        return self._geometry.transformer

    def heading_to_los(self, rowcol, heading_deg):
        """See :meth:`satimg.sar_orbit_geometry.SAROrbitGeometry.heading_to_los`."""
        return self._geometry.heading_to_los(rowcol, heading_deg)

    def heading_in_image(self, rowcol, heading_deg):
        """See :meth:`satimg.sar_orbit_geometry.SAROrbitGeometry.heading_in_image`."""
        return self._geometry.heading_in_image(rowcol, heading_deg)

    def heading_from_image(self, rowcol, image_heading_deg):
        """See :meth:`satimg.sar_orbit_geometry.SAROrbitGeometry.heading_from_image`."""
        return self._geometry.heading_from_image(rowcol, image_heading_deg)

    def doppler_azimuth_shift_m(self, rowcol, speed: float, heading_deg: float):
        """See :meth:`satimg.sar_orbit_geometry.SAROrbitGeometry.doppler_azimuth_shift_m`."""
        return self._geometry.doppler_azimuth_shift_m(rowcol, speed, heading_deg)

    def doppler_pixel_shift(self, rowcol, speed: float, heading_deg: float):
        """See :meth:`satimg.sar_orbit_geometry.SAROrbitGeometry.doppler_pixel_shift`."""
        return self._geometry.doppler_pixel_shift(rowcol, speed, heading_deg)

    def correct_position(self, lat: float, lon: float, speed: float, heading_deg: float):
        """See :meth:`satimg.sar_orbit_geometry.SAROrbitGeometry.correct_position`."""
        return self._geometry.correct_position(lat, lon, speed, heading_deg)
