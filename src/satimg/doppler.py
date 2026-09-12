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

#: Earth's mean rotation rate, radians/second (IERS/WGS84 value).
EARTH_ROTATION_RATE = 7.292_115e-5


def orbit_inclination(position_m, velocity_ecef_mps) -> float:
    """Orbital inclination (degrees) from a single Earth-fixed (ECEF) position and
    velocity state vector, such as the ``<orbit>`` entries in a Sentinel-1
    annotation file.

    Inclination is a property of the orbital plane in *inertial* space, so the
    ECEF velocity must first be converted to inertial by adding back Earth's
    rotation (``v_inertial = v_ecef + omega x r``) before computing the specific
    angular momentum ``h = r x v`` and ``i = arccos(h_z / |h|)``. Skipping that
    correction is a several-degree-sized error -- Earth's rotation contributes
    roughly 7% of a low-Earth-orbit satellite's own speed.
    """
    r = numpy.asarray(position_m, dtype=float)
    v_ecef = numpy.asarray(velocity_ecef_mps, dtype=float)
    omega = numpy.array([0.0, 0.0, EARTH_ROTATION_RATE])
    v_inertial = v_ecef + numpy.cross(omega, r)
    h = numpy.cross(r, v_inertial)
    return float(numpy.rad2deg(numpy.arccos(h[2] / numpy.linalg.norm(h))))


def ground_track_heading(latitude_deg, inclination_deg, ascending):
    """Satellite ground-track heading (degrees clockwise from true north) at a
    given geodetic latitude, for a circular orbit of the given inclination.

    Standard spherical-triangle result at the ascending node:
    ``sin(Az) = cos(inclination) / cos(latitude)`` on the ascending arc; the
    descending arc's heading is the supplementary angle, ``180 - Az``. Returns
    degrees wrapped to ``[-180, 180)``.

    ``ascending`` is truthy for the ascending pass, falsy for descending (a plain
    bool, or a boolean array matching the broadcast shape of ``latitude_deg``).
    Undefined (``nan``) wherever ``|cos(inclination)| > cos(latitude)`` -- a
    latitude beyond the orbit's reach.
    """
    lat = numpy.deg2rad(numpy.asarray(latitude_deg, dtype=float))
    inc = numpy.deg2rad(numpy.asarray(inclination_deg, dtype=float))
    az_ascending = numpy.rad2deg(numpy.arcsin(numpy.cos(inc) / numpy.cos(lat)))
    az = numpy.where(numpy.asarray(ascending), az_ascending, 180.0 - az_ascending)
    return (az + 180.0) % 360.0 - 180.0


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
