"""Helpers for :class:`~satimg.connectors.static_stac_connector.StaticSTACConnector`:
STAC datetimes, and the date period a child catalogue names."""

from __future__ import annotations

import datetime
import posixpath
import re

#: a trailing ``YYYY``, ``YYYY-MM`` or ``YYYY-MM-DD``
_PERIOD = re.compile(r"(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?$")


def utc(t: datetime.datetime | None) -> datetime.datetime | None:
    """``t`` timezone-aware; a naive datetime is taken as UTC."""
    if t is None or t.tzinfo is not None:
        return t
    return t.replace(tzinfo=datetime.timezone.utc)


def parse_datetime(text: str) -> datetime.datetime:
    """A STAC ``datetime`` (``...Z`` or ``+00:00``) as an aware datetime."""
    return datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))


def link_period(link: dict) -> tuple[datetime.datetime, datetime.datetime] | None:
    """``[start, end)`` of the date period a child link names in its title
    (``"2025-08"``, ``"Capella Open Data 2025-08"``) or its directory
    (``./2025-08/catalog.json``) -- ``None`` if it names none."""
    directory = posixpath.basename(posixpath.dirname(link["href"]))
    for text in (link.get("title") or "", directory):
        m = _PERIOD.search(text.strip())
        if not m:
            continue
        year, month, day = (int(g) if g else None for g in m.groups())
        utc = datetime.timezone.utc
        if month is None:
            return datetime.datetime(year, 1, 1, tzinfo=utc), datetime.datetime(year + 1, 1, 1, tzinfo=utc)
        if day is None:
            end = datetime.datetime(year + month // 12, month % 12 + 1, 1, tzinfo=utc)
            return datetime.datetime(year, month, 1, tzinfo=utc), end
        start = datetime.datetime(year, month, day, tzinfo=utc)
        return start, start + datetime.timedelta(days=1)
    return None


def overlaps(period, start, stop) -> bool:
    if period is None:
        return True
    return (start is None or period[1] > start) and (stop is None or period[0] <= stop)
