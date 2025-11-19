import abc
import datetime

from pystac_client import Client


class STACConnector(abc.ABC):
    STAC_ENDPOINT = None
    collection = None
    product_type = None

    def __init__(self):
        self._client = Client.open(self.STAC_ENDPOINT)

    def search(self, geojson: dict | None = None,
               start_datetime: datetime.datetime | None = None,
               stop_datetime: datetime.datetime | None = None,
               ids: str | None = None,
               exclude_ids: list[str] | None = None,
               **kwargs):
        dt = None
        if start_datetime and stop_datetime:
            dt = f"{start_datetime.isoformat()}/{stop_datetime.isoformat()}"
        elif start_datetime:
            dt = f"{start_datetime.isoformat()}/"
        elif stop_datetime:
            dt = f"/{stop_datetime.isoformat()}"
        filter = None
        if self.product_type:
            filter = {
                "filter": {
                    "op": "and",
                    "args": [
                        {
                            "op": "=",
                            "args": [
                                {
                                    "property": "product:type"
                                },
                                self.product_type,
                            ]

                        },

                    ]}}
        if ids:
            filter = None
        items = list(self._client.search(
            collections=[self.collection],
            datetime=dt,
            intersects=geojson,
            ids=ids,
            filter=filter,
        ).items())
        items = [i.to_dict() for i in items]
        if exclude_ids:
            items = [i for i in items if i["id"] not in exclude_ids]
        return items

    @abc.abstractmethod
    def download(self, item, save_path):
        pass

    @abc.abstractmethod
    def _get_session(self, *args, **kwargs):
        pass
