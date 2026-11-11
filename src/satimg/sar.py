"""SAR geometry: the moving-target Doppler azimuth shift, worked out on the
ground from the sensor's Earth-fixed orbit.

A ground target with a non-zero velocity along the radar's line of sight is
displaced from its true position along track in a focused SAR image -- the
effect behind a ship appearing offset from its own wake.
:func:`moving_target_shift` gives that displacement in metres and its compass
bearing from the sensor's position and velocity when it images the target,
for any geometry: either look side, any squint, any pixel frame. The rest are
the WGS 84 / orbit helpers it needs. Pure functions on already-extracted
values; :class:`~satimg.sar_orbit_geometry.SAROrbitGeometry` is what every
SAR product (:class:`~satimg.sar_product.SARProduct`) builds on them.
"""

from __future__ import annotations

import numpy

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


#: WGS 84 semi-major axis (metres) and first eccentricity squared
WGS84_A = 6_378_137.0
WGS84_E2 = 6.694_379_990_14e-3
#: the Earth's gravitational parameter (m^3/s^2) and rotation rate (rad/s)
EARTH_GM = 3.986_004_418e14
EARTH_ROTATION = 7.292_115_0e-5


def geodetic_to_ecef(lat_deg, lon_deg, height_m=0.0) -> numpy.ndarray:
    """WGS 84 geodetic ``lat`` / ``lon`` (degrees) and ellipsoidal height
    (metres) as Earth-centred, Earth-fixed ``(N, 3)`` metres."""
    lat = numpy.deg2rad(numpy.atleast_1d(numpy.asarray(lat_deg, dtype=float)))
    lon = numpy.deg2rad(numpy.atleast_1d(numpy.asarray(lon_deg, dtype=float)))
    h = numpy.asarray(height_m, dtype=float)
    n = WGS84_A / numpy.sqrt(1.0 - WGS84_E2 * numpy.sin(lat) ** 2)
    return numpy.column_stack((
        (n + h) * numpy.cos(lat) * numpy.cos(lon),
        (n + h) * numpy.cos(lat) * numpy.sin(lon),
        (n * (1.0 - WGS84_E2) + h) * numpy.sin(lat),
    ))


def enu_axes(lat_deg, lon_deg) -> numpy.ndarray:
    """The local east, north and up unit vectors at each geodetic
    ``lat`` / ``lon``, in ECEF: an ``(N, 3, 3)`` array whose rows are east,
    north, up -- so ``axes @ v`` turns an ECEF vector into local ENU."""
    lat = numpy.deg2rad(numpy.atleast_1d(numpy.asarray(lat_deg, dtype=float)))
    lon = numpy.deg2rad(numpy.atleast_1d(numpy.asarray(lon_deg, dtype=float)))
    zero = numpy.zeros_like(lat)
    east = numpy.stack((-numpy.sin(lon), numpy.cos(lon), zero), axis=-1)
    north = numpy.stack(
        (-numpy.sin(lat) * numpy.cos(lon), -numpy.sin(lat) * numpy.sin(lon), numpy.cos(lat)), axis=-1
    )
    up = numpy.stack((numpy.cos(lat) * numpy.cos(lon), numpy.cos(lat) * numpy.sin(lon), numpy.sin(lat)), axis=-1)
    return numpy.stack((east, north, up), axis=1)


def interpolate_orbit(times, positions, velocities, t) -> tuple[numpy.ndarray, numpy.ndarray]:
    """The sensor's ``(position, velocity)`` at each time ``t`` (seconds, the
    same reference as ``times``), each ``(N, 3)``: a cubic Hermite spline
    through the state vectors -- ``positions`` / ``velocities`` ``(M, 3)``
    ECEF at ``times`` ``(M,)``, ascending. Times outside ``times`` are
    clamped to its ends."""
    times = numpy.asarray(times, dtype=float)
    positions = numpy.asarray(positions, dtype=float)
    velocities = numpy.asarray(velocities, dtype=float)
    t = numpy.clip(numpy.atleast_1d(numpy.asarray(t, dtype=float)), times[0], times[-1])
    i = numpy.clip(numpy.searchsorted(times, t) - 1, 0, len(times) - 2)
    dt = (times[i + 1] - times[i])[:, None]
    s = (t[:, None] - times[i][:, None]) / dt
    p0, p1 = positions[i], positions[i + 1]
    m0, m1 = velocities[i] * dt, velocities[i + 1] * dt
    s2, s3 = s * s, s * s * s
    position = (2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * m0 + (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * m1
    velocity = (
        (6 * s2 - 6 * s) * p0 + (3 * s2 - 4 * s + 1) * m0 + (-6 * s2 + 6 * s) * p1 + (3 * s2 - 2 * s) * m1
    ) / dt
    return position, velocity


def doppler_cone_cosine(sensor_position, sensor_velocity, target_ecef) -> numpy.ndarray:
    """Cosine of the angle between the sensor's velocity and its line of
    sight to each target, ``(N,)``: ``0`` at zero Doppler (broadside),
    positive while the sensor is still closing on the target."""
    los = numpy.atleast_2d(target_ecef) - numpy.atleast_2d(sensor_position)
    velocity = numpy.atleast_2d(sensor_velocity)
    return numpy.einsum("ij,ij->i", los, velocity) / (
        numpy.linalg.norm(los, axis=1) * numpy.linalg.norm(velocity, axis=1)
    )


def imaging_time(times, positions, velocities, target_ecef, cone_cosine=0.0) -> numpy.ndarray:
    """When the sensor sees each target ``(N, 3)`` at the Doppler cone
    angle whose cosine is ``cone_cosine`` (see :func:`doppler_cone_cosine`):
    the target's zero-Doppler time for ``0``, its aperture-centre time for a
    squinted collect given the scene centre's own cone angle. ``(N,)``
    seconds, on ``times``' reference; Newton's method from the middle of
    the orbit, clamped to its span."""
    target = numpy.atleast_2d(numpy.asarray(target_ecef, dtype=float))
    times = numpy.asarray(times, dtype=float)
    c = numpy.asarray(cone_cosine, dtype=float)
    t = numpy.full(len(target), 0.5 * (times[0] + times[-1]))
    for _ in range(8):
        position, velocity = interpolate_orbit(times, positions, velocities, t)
        los = target - position
        speed = numpy.linalg.norm(velocity, axis=1)
        f = numpy.einsum("ij,ij->i", los, velocity) - c * numpy.linalg.norm(los, axis=1) * speed
        t = numpy.clip(t + f / (speed**2 * (1.0 - c**2)), times[0], times[-1])
    return t


def look_bearing(lat_deg, lon_deg, sensor_position) -> numpy.ndarray:
    """Compass bearing (degrees, ``[0, 360)``) of the radar's ground-projected
    look direction at each target -- from the sensor towards the target,
    i.e. towards increasing ground range -- whichever side it looks."""
    target = geodetic_to_ecef(lat_deg, lon_deg)
    los = numpy.einsum("nij,nj->ni", enu_axes(lat_deg, lon_deg), target - numpy.atleast_2d(sensor_position))
    return numpy.rad2deg(numpy.arctan2(los[:, 0], los[:, 1])) % 360.0


def moving_target_shift(lat_deg, lon_deg, sensor_position, sensor_velocity, speed, heading_deg):
    """Where a moving target appears in a focused SAR image, relative to
    where it is: ``(shift_m, bearing_deg)``, each ``(N,)``.

    ``lat`` / ``lon`` is the target (on the ellipsoid), ``speed`` (m/s) and
    ``heading_deg`` (compass) its ground motion, ``sensor_position`` /
    ``sensor_velocity`` the sensor's ECEF state when it images the target
    (from :func:`interpolate_orbit` at :func:`imaging_time`).

    The image puts a target where a stationary one would have the same
    range and range rate. The target's own motion adds ``-v_t . u`` to its
    range rate (``u`` the unit line of sight, target to sensor); stepping a
    stationary point a distance ``d`` along the ground, square to the line
    of sight (so at constant range), changes its range rate by
    ``-d (g . v_s) / R`` (``g`` that ground direction, ``v_s`` the sensor's
    Earth-fixed velocity, ``R`` the slant range). Equating the two, the
    target appears displaced by ``d = (v_t . u) R / (g . v_s)`` along ``g``.

    ``bearing_deg`` is ``g``'s compass bearing, taken on the side the sensor
    is flying towards; ``shift_m`` is positive along it -- a target closing
    on the sensor is shifted forward, a receding one back. Broadside, with
    the sensor's speed ``V`` all square to the line of sight, this is the
    textbook ``-v_r R / V`` (``v_r`` the target's range rate). Purely
    along-``g`` motion gives no shift. Squint needs no special handling: it
    is in ``u`` and ``v_s`` (``g . v_s`` shrinks, the shift grows).
    """
    axes = enu_axes(lat_deg, lon_deg)
    los = numpy.atleast_2d(sensor_position) - geodetic_to_ecef(lat_deg, lon_deg)
    slant_range = numpy.linalg.norm(los, axis=1)
    u = numpy.einsum("nij,nj->ni", axes, los / slant_range[:, None])
    v_s = numpy.einsum("nij,nj->ni", axes, numpy.atleast_2d(sensor_velocity) * numpy.ones_like(los))
    g = numpy.column_stack((-u[:, 1], u[:, 0]))
    g /= numpy.linalg.norm(g, axis=1)[:, None]
    along = numpy.einsum("ij,ij->i", g, v_s[:, :2])
    g *= numpy.sign(along)[:, None]
    along = numpy.abs(along)
    heading = numpy.deg2rad(numpy.asarray(heading_deg, dtype=float))
    speed = numpy.asarray(speed, dtype=float)
    closing = speed * (numpy.sin(heading) * u[:, 0] + numpy.cos(heading) * u[:, 1])
    shift = closing * slant_range / along
    return shift, numpy.rad2deg(numpy.arctan2(g[:, 0], g[:, 1])) % 360.0


def offset_latlon(lat_deg, lon_deg, distance_m, bearing_deg) -> numpy.ndarray:
    """Each ``lat`` / ``lon`` moved ``distance_m`` along compass
    ``bearing_deg``, as ``(N, 2)`` ``(lat, lon)`` -- on the WGS 84
    ellipsoid's radii of curvature at the step's middle latitude: millimetres
    over the few hundred metres to few kilometres of a Doppler shift."""
    lat = numpy.atleast_1d(numpy.asarray(lat_deg, dtype=float))
    lon = numpy.atleast_1d(numpy.asarray(lon_deg, dtype=float))
    bearing = numpy.deg2rad(numpy.asarray(bearing_deg, dtype=float))
    distance = numpy.asarray(distance_m, dtype=float)
    middle = lat
    for _ in range(2):
        sin2 = numpy.sin(numpy.deg2rad(middle)) ** 2
        meridian = WGS84_A * (1.0 - WGS84_E2) / (1.0 - WGS84_E2 * sin2) ** 1.5
        normal = WGS84_A / numpy.sqrt(1.0 - WGS84_E2 * sin2)
        d_lat = numpy.rad2deg(distance * numpy.cos(bearing) / meridian)
        middle = lat + d_lat / 2
    d_lon = numpy.rad2deg(distance * numpy.sin(bearing) / (normal * numpy.cos(numpy.deg2rad(middle))))
    return numpy.column_stack((lat + d_lat, lon + d_lon))
