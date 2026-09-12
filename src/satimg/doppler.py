"""Geometry for the SAR moving-target azimuth-shift effect.

A ground target with a non-zero velocity relative to the radar's line of sight (LOS)
is displaced from its true position along the *azimuth* axis of a focused SAR image
-- the effect behind a ship appearing offset from its own wake. The functions here
are pure geometry, taking already-extracted scalars, so they're reusable and
testable independently of any product class; see
:meth:`satimg.products.sentinel1.Sentinel1Product.doppler_azimuth_shift` and
:meth:`~satimg.products.sentinel1.Sentinel1Product.heading_to_los` for the
product-facing entry points.
"""

from __future__ import annotations

import numpy

#: metres/second
SPEED_OF_LIGHT = 299_792_458.0


def heading_to_los(heading_deg, platform_heading_deg):
    """Convert a compass heading (degrees clockwise from true north) into the
    object's bearing relative to the radar's ground-projected line of sight.

    Sentinel-1 is always right-looking, so the LOS ground direction (pointing away
    from the satellite, i.e. towards increasing ground range) sits 90 degrees
    clockwise of the platform's own ground-track heading ``platform_heading_deg``.

    Returns degrees wrapped to ``[-180, 180)``: ``0`` means the object is moving
    directly away from the satellite (ground range increasing), ``-180`` directly
    towards it, and ``+-90`` means it's moving purely along-track -- which produces
    no azimuth shift at all (see :func:`azimuth_shift_m`).
    """
    look_direction = (numpy.asarray(platform_heading_deg, dtype=float) + 90.0) % 360.0
    heading = numpy.asarray(heading_deg, dtype=float)
    return (heading - look_direction + 180.0) % 360.0 - 180.0


def azimuth_shift_m(
    speed,
    heading_deg,
    platform_heading_deg,
    incidence_angle_deg,
    slant_range_m,
    platform_velocity_mps,
):
    """Azimuth displacement (metres) of a moving target in a focused SAR image.

    A target with radial velocity is indistinguishable, to the azimuth-compression
    matched filter, from a *stationary* target sitting at a shifted azimuth
    position -- that shifted position is what this returns. It depends only on the
    target's ground-range-relative motion, the imaging geometry, and the platform's
    velocity; the radar wavelength cancels out of the derivation entirely, so it
    isn't a parameter here.

    ``speed`` in m/s; ``heading_deg``, ``platform_heading_deg`` and
    ``incidence_angle_deg`` in degrees; ``slant_range_m`` and
    ``platform_velocity_mps`` in metres and m/s (Sentinel-1's orbital speed is
    about 7.4 km/s).

    Sign convention: positive means the target appears shifted towards *later*
    azimuth (acquisition) time -- the direction of increasing row index in a
    Sentinel-1 GRD product -- which happens when the object moves away from the
    satellite (increasing ground range). Purely along-track motion
    (:func:`heading_to_los` == +-90) produces zero shift.
    """
    los_angle = numpy.deg2rad(heading_to_los(heading_deg, platform_heading_deg))
    incidence = numpy.deg2rad(numpy.asarray(incidence_angle_deg, dtype=float))
    v_ground_away = numpy.asarray(speed, dtype=float) * numpy.cos(los_angle)
    v_radial = v_ground_away * numpy.sin(incidence)
    return v_radial * numpy.asarray(slant_range_m, dtype=float) / numpy.asarray(
        platform_velocity_mps, dtype=float
    )
