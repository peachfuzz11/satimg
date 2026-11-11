"""SAR-specific per-pixel geometry: ground-track heading, line-of-sight bearing,
image-frame heading, and the moving-target Doppler azimuth-shift effect.

A ground target with a non-zero velocity relative to the radar's line of sight (LOS)
is displaced from its true position along the *azimuth* axis of a focused SAR image
-- the effect behind a ship appearing offset from its own wake. ``heading_in_image``
re-expresses a compass heading in the image's own rotated *and mirrored*,
native-sensor-geometry pixel frame -- a SAR-only concern, since a map-projected
optical raster is already north-up. The functions here are pure geometry, taking already-extracted scalars,
so they're reusable and testable independently of any product class; see
:meth:`satimg.products.sentinel1.grd.sentinel1_grd.Sentinel1GRD.doppler_azimuth_shift`,
:meth:`~satimg.products.sentinel1.grd.sentinel1_grd.Sentinel1GRD.heading_to_los`,
:meth:`~satimg.products.sentinel1.grd.sentinel1_grd.Sentinel1GRD.heading_in_image` and
:meth:`~satimg.products.sentinel1.grd.sentinel1_grd.Sentinel1GRD.correct_position` for the
product-facing entry points.

The second half works on the ground instead, from the sensor's Earth-fixed
orbit: :func:`moving_target_shift` gives the displacement in metres and its
compass bearing for any geometry -- either look side, any squint, any pixel
frame -- with the ellipsoid / orbit helpers it needs. That is what
:class:`~satimg.sar_orbit_geometry.SAROrbitGeometry` (Umbra, Capella, ICEYE)
builds on.
"""

from __future__ import annotations

import numpy

#: metres/second
SPEED_OF_LIGHT = 299_792_458.0


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


def heading_in_image(heading_deg, platform_heading_deg):
    """Convert a compass heading (degrees clockwise from true north) into the
    direction it points within a SAR GRD's own pixel frame.

    ``platform_heading_deg`` is the satellite's ground-track heading (see
    :func:`ground_track_heading`). GRD imagery is still in native sensor
    geometry: row increases along that heading (image "down") and column
    increases towards the look direction, ``platform_heading_deg + 90`` for a
    right-looking sensor (image "right"). The raster is therefore a *mirror
    image* of the map -- "right" sits counter-clockwise of "up" -- so the sense
    of every angle reverses, and this is not a plain subtraction of a rotation.
    See :meth:`~satimg.products.sentinel1.grd.sentinel1_grd.Sentinel1GRD.heading_in_image`.

    Returns degrees wrapped to ``[0, 360)`` measured clockwise from image
    "up": ``0``/``360`` means the object points towards the top of the image
    (decreasing row), ``90`` towards the right, ``180`` towards the bottom.
    Drawn as an arrow on the raster, it lies along the ship's heading as it
    appears in the image.
    """
    heading = numpy.asarray(heading_deg, dtype=float)
    platform_heading = numpy.asarray(platform_heading_deg, dtype=float)
    return (platform_heading + 180.0 - heading) % 360.0


def heading_from_image(image_heading_deg, platform_heading_deg):
    """Convert a direction in a SAR GRD's own pixel frame (degrees clockwise
    from image "up", as returned by :func:`heading_in_image`) back into a
    compass heading (degrees clockwise from true north), wrapped to
    ``[0, 360)``.

    The image is a mirror of the map, so the conversion is its own inverse:
    this is the same arithmetic as :func:`heading_in_image`, named for the
    direction it is used in.
    """
    image_heading = numpy.asarray(image_heading_deg, dtype=float)
    platform_heading = numpy.asarray(platform_heading_deg, dtype=float)
    return (platform_heading + 180.0 - image_heading) % 360.0


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
    Sentinel-1 GRD product -- which happens when the object moves *towards* the
    satellite (decreasing ground range); a receding object is shifted towards
    earlier time. (A target with range rate ``v_r`` picks up a Doppler offset
    the focusing reads as a stationary target ``-R * v_r / V`` along-track, so
    the sign is opposite to ``v_r``.) Purely along-track motion
    (:func:`heading_to_los` == +-90) produces zero shift.
    """
    los_angle = numpy.deg2rad(heading_to_los(heading_deg, platform_heading_deg))
    incidence = numpy.deg2rad(numpy.asarray(incidence_angle_deg, dtype=float))
    v_ground_away = numpy.asarray(speed, dtype=float) * numpy.cos(los_angle)
    v_radial = v_ground_away * numpy.sin(incidence)
    return -v_radial * numpy.asarray(slant_range_m, dtype=float) / numpy.asarray(
        platform_velocity_mps, dtype=float
    )


# -- moving targets from the sensor's own orbit ---------------------------------
# Products that ship their sensor's state vectors (or enough to rebuild them)
# don't need a per-product pixel frame: the shift is worked out on the ground,
# in east / north metres, from the sensor's Earth-fixed position and velocity
# at the moment it images the target. See
# :class:`~satimg.sar_orbit_geometry.SAROrbitGeometry`.

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
    on the sensor is shifted forward, a receding one back, as in
    :func:`azimuth_shift_m` (which is this at broadside, zero squint, with
    the platform speed for ``g . v_s``). Purely along-``g`` motion gives no
    shift. Squint needs no special handling: it is in ``u`` and ``v_s``.
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
