"""VIIRS Day/Night Band connector: NOAA's open data on AWS (NODD).

The JPSS satellites' VIIRS DNB SDR granules (``SVDNB_*.h5``, ~10 MB, about 85
seconds of orbit each) and their geolocation (``GDNBO_*.h5``, ~50 MB) are
public in NOAA's ``noaa-nesdis-*-pds`` buckets: no account, no credentials,
for either ``search`` or ``download``::

    connector = get_connector("viirs-dnb")                 # NOAA-20; satellite="snpp" / "noaa-21"
    items = connector.search(geojson=aoi, start_datetime=t0, stop_datetime=t1)
    connector.download(items[0], "/data/granule.zip")

The buckets have no catalogue, so ``search`` lists them by time and, to
filter on ``geojson`` or ``night_only``, reads each candidate's footprint and
day/night flag straight out of its HDF5 attributes with HTTP range requests
-- a few hundred KB per granule, never the whole file. Keep the time window
to hours, not weeks.
"""

from __future__ import annotations

import datetime
import io
import logging
import zipfile
from concurrent.futures import ThreadPoolExecutor
from xml.etree import ElementTree

import numpy
import requests

from satimg.connectors._auth import make_session
from satimg.connectors.base import Connector
from satimg.viirs_utils import granule_key, granule_times

logger = logging.getLogger(__name__)

#: satellite name -> (bucket, file-name platform code)
SATELLITES = {
    "snpp": ("noaa-nesdis-snpp-pds", "npp"),
    "noaa-20": ("noaa-nesdis-n20-pds", "j01"),
    "noaa-21": ("noaa-nesdis-n21-pds", "j02"),
}

_S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
class HttpRangeFile(io.RawIOBase):
    """A read-only, seekable file object over an HTTP URL, fetching
    ``block``-sized pieces by range request as they are read (and keeping
    them) -- enough for ``h5py`` to read an HDF5 file's attributes without
    downloading it."""

    def __init__(self, url: str, session: requests.Session | None = None, block: int = 256 << 10):
        self._url = url
        self._session = session or make_session()
        self._block = block
        self._blocks: dict[int, bytes] = {}
        self._pos = 0
        head = self._session.head(url, timeout=60)
        head.raise_for_status()
        self._size = int(head.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._size}[whence]
        self._pos = max(0, base + offset)
        return self._pos

    def _fetch(self, index: int) -> bytes:
        if index not in self._blocks:
            start = index * self._block
            stop = min(start + self._block, self._size) - 1
            response = self._session.get(self._url, headers={"Range": f"bytes={start}-{stop}"}, timeout=60)
            response.raise_for_status()
            self._blocks[index] = response.content
        return self._blocks[index]

    def readinto(self, buffer) -> int:
        stop = min(self._pos + len(buffer), self._size)
        out = bytearray()
        while self._pos + len(out) < stop:
            at = self._pos + len(out)
            block = self._fetch(at // self._block)
            out += block[at % self._block: at % self._block + (stop - at)]
        buffer[: len(out)] = out
        self._pos += len(out)
        return len(out)


def _closed_ring(lats, lons) -> list[tuple[float, float]]:
    ring = [(float(lon), float(lat)) for lat, lon in zip(lats, lons) if lat > -999 and lon > -999]
    return ring + ring[:1]


def _unwrap(ring, reference: float) -> numpy.ndarray:
    """``ring``'s longitudes moved by whole turns to within 180 degrees of
    ``reference``, so a footprint across the antimeridian stays one piece."""
    pts = numpy.asarray(ring, dtype=float)
    pts[:, 0] = (pts[:, 0] - reference + 180.0) % 360.0 - 180.0 + reference
    return pts


def _inside(point, ring) -> bool:
    x, y = point
    inside = False
    for (x1, y1), (x2, y2) in zip(ring[:-1], ring[1:]):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _segments_cross(p1, p2, q1, q2) -> bool:
    def side(a, b, c):
        return numpy.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    return side(p1, p2, q1) != side(p1, p2, q2) and side(q1, q2, p1) != side(q1, q2, p2)


def rings_intersect(a, b) -> bool:
    """Whether two closed lon/lat rings overlap (one inside the other, or
    edges crossing), working across the antimeridian. Plain planar geometry
    in degrees -- fine for granule-sized footprints away from the poles."""
    a = _unwrap(a, a[0][0])
    b = _unwrap(b, a[0][0])
    if any(_inside(p, b) for p in a[:-1]) or any(_inside(p, a) for p in b[:-1]):
        return True
    return any(
        _segments_cross(a[i], a[i + 1], b[j], b[j + 1]) for i in range(len(a) - 1) for j in range(len(b) - 1)
    )


def _aoi_ring(geojson: dict) -> list[tuple[float, float]]:
    if geojson.get("type") == "FeatureCollection":
        geojson = geojson["features"][0]
    if geojson.get("type") == "Feature":
        geojson = geojson["geometry"]
    if geojson["type"] != "Polygon":
        raise ValueError(f"unsupported geometry for VIIRS search: {geojson['type']}")
    return [tuple(p[:2]) for p in geojson["coordinates"][0]]


class ViirsDNBConnector(Connector):
    """VIIRS DNB SDR + GEO granules from one JPSS satellite's NODD bucket."""

    collection = "viirs-dnb"

    def __init__(self, satellite: str = "noaa-20"):
        try:
            self._bucket, self._platform = SATELLITES[satellite]
        except KeyError:
            raise ValueError(f"unknown satellite {satellite!r}; known: {sorted(SATELLITES)}") from None
        self.satellite = satellite
        self._session = make_session()

    @property
    def _base(self) -> str:
        return f"https://{self._bucket}.s3.amazonaws.com"

    def _list(self, prefix: str) -> list[str]:
        keys, token = [], None
        while True:
            params = {"list-type": "2", "prefix": prefix}
            if token:
                params["continuation-token"] = token
            response = self._session.get(self._base, params=params, timeout=60)
            response.raise_for_status()
            root = ElementTree.fromstring(response.content)
            keys += [k.text for k in root.iterfind("s3:Contents/s3:Key", _S3_NS)]
            token = root.findtext("s3:NextContinuationToken", namespaces=_S3_NS)
            if not token:
                return keys

    def _granules(self, kind: str, start: datetime.datetime, stop: datetime.datetime) -> dict[str, str]:
        """``{granule_key: object key}`` of every ``kind`` file overlapping
        ``[start, stop]``, listed hour by hour."""
        folder = {"SVDNB": "VIIRS-DNB-SDR", "GDNBO": "VIIRS-DNB-GEO"}[kind]
        found = {}
        hour = start.replace(minute=0, second=0, microsecond=0) - datetime.timedelta(hours=1)
        while hour <= stop:
            prefix = f"{folder}/{hour:%Y/%m/%d}/{kind}_{self._platform}_d{hour:%Y%m%d}_t{hour:%H}"
            for key in self._list(prefix):
                name = key.rsplit("/", 1)[-1]
                if not name.endswith(".h5"):
                    continue
                t0, t1 = granule_times(name)
                if t1 >= start and t0 <= stop:
                    found[granule_key(name)] = key
            hour += datetime.timedelta(hours=1)
        return found

    def _attributes(self, url: str) -> dict:
        """The SDR granule's own attributes, read by range requests."""
        import h5py

        # a session per call: this runs on many threads at once
        with make_session() as session, HttpRangeFile(url, session) as raw, h5py.File(raw, "r") as h5:
            gran = h5["Data_Products/VIIRS-DNB-SDR/VIIRS-DNB-SDR_Gran_0"].attrs
            return {
                "day_night": gran["N_Day_Night_Flag"].ravel()[0].decode(),
                "ring": _closed_ring(gran["G-Ring_Latitude"].ravel(), gran["G-Ring_Longitude"].ravel()),
            }

    def search(
        self,
        geojson: dict | None = None,
        start_datetime: datetime.datetime | None = None,
        stop_datetime: datetime.datetime | None = None,
        ids: list[str] | None = None,
        exclude_ids: list[str] | None = None,
        night_only: bool = True,
        **kwargs,
    ) -> list[dict]:
        """Granules from ``start_datetime`` to ``stop_datetime`` (both
        needed: there is no catalogue to search open-ended), overlapping
        ``geojson`` if given, and -- with ``night_only``, the default, since
        boat lights only show at night -- not flagged ``Day``. ``item["id"]``
        is the SDR file's stem; ``item["assets"]`` has the ``sdr`` and
        ``geo`` URLs."""
        if not (start_datetime and stop_datetime):
            raise ValueError("VIIRS DNB search needs both start_datetime and stop_datetime")
        sdrs = self._granules("SVDNB", start_datetime, stop_datetime)
        geos = self._granules("GDNBO", start_datetime, stop_datetime)
        aoi = _aoi_ring(geojson) if geojson is not None else None
        items = []
        for key, sdr in sorted(sdrs.items()):
            name = sdr.rsplit("/", 1)[-1][: -len(".h5")]
            if key not in geos or (ids and name not in ids) or (exclude_ids and name in exclude_ids):
                continue
            items.append({
                "id": name,
                "type": "Feature",
                "geometry": None,
                "properties": {"satellite": self.satellite, "datetime": granule_times(name)[0].isoformat()},
                "assets": {"sdr": {"href": f"{self._base}/{sdr}"}, "geo": {"href": f"{self._base}/{geos[key]}"}},
            })
        if aoi is not None or night_only:
            with ThreadPoolExecutor(max_workers=8) as pool:
                attrs = list(pool.map(self._attributes, [i["assets"]["sdr"]["href"] for i in items]))
            kept = []
            for item, a in zip(items, attrs):
                if night_only and a["day_night"] == "Day":
                    continue
                if aoi is not None and not rings_intersect(a["ring"], aoi):
                    continue
                item["geometry"] = {"type": "Polygon", "coordinates": [a["ring"]]}
                item["properties"]["day_night"] = a["day_night"]
                kept.append(item)
            items = kept
        logger.debug("VIIRS DNB search returned %d granule(s)", len(items))
        return items

    def download(self, item: dict, save_path: str) -> str:
        """Both files into one zip (stored, uncompressed: they are already
        compressed inside), under a folder named for the SDR -- which
        ``satimg.open`` reads zip-native."""
        logger.info("downloading %s -> %s", item["id"], save_path)
        with zipfile.ZipFile(save_path, "w", zipfile.ZIP_STORED) as archive:
            for asset in ("sdr", "geo"):
                url = item["assets"][asset]["href"]
                name = url.rsplit("/", 1)[-1]
                with self._session.get(url, stream=True, timeout=300) as response:
                    response.raise_for_status()
                    with archive.open(f"{item['id']}/{name}", "w", force_zip64=True) as out:
                        for block in response.iter_content(chunk_size=4 << 20):
                            out.write(block)
        logger.info("wrote %s", save_path)
        return save_path
