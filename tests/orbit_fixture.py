"""Synthetic sensor state vectors for the SAR fixtures: a sensor flying a
straight line, square to its line of sight, that images ``(lat, lon)``
broadside at ``t = 0`` -- so the textbook moving-target shift
``speed * sin(incidence) * slant_range / SPEED`` is exact for it."""

from __future__ import annotations

import numpy

from satimg import sar

#: the sensor's speed, m/s
SPEED = 7_600.0
#: state vector times, seconds from the scene centre
TIMES = numpy.arange(-10.0, 11.0, 1.0)


def states(lat, lon, *, look_bearing, incidence, slant_range, right_looking):
    """``(times, positions, velocities)``: ECEF metres / (m/s) at
    :data:`TIMES`. ``look_bearing`` is the compass bearing from the sensor
    towards the target."""
    axes = sar.enu_axes(lat, lon)[0]
    inc, back = numpy.deg2rad(incidence), numpy.deg2rad(look_bearing + 180.0)
    up_los = numpy.array([numpy.sin(inc) * numpy.sin(back), numpy.sin(inc) * numpy.cos(back), numpy.cos(inc)])
    centre = sar.geodetic_to_ecef(lat, lon)[0] + slant_range * axes.T @ up_los
    flight = numpy.deg2rad(look_bearing - 90.0 if right_looking else look_bearing + 90.0)
    velocity = SPEED * axes.T @ numpy.array([numpy.sin(flight), numpy.cos(flight), 0.0])
    positions = centre + TIMES[:, None] * velocity
    return TIMES, positions, numpy.tile(velocity, (len(TIMES), 1))
