import datetime
import os
import zipfile
from pathlib import Path

from pystac_client import Client

from satproducts.products.base.connector import Connector
from satproducts.products.landsat.auth import ers_login


class LandsatL1GTConnector(Connector):
    LANDSATLOOK_STAC = "https://landsatlook.usgs.gov/stac-server"
    collection = "landsat-c2l1"

    def __init__(self):
        super().__init__()
        self._client = Client.open(self.LANDSATLOOK_STAC)

    def _get_session(self, *args, **kwargs):
        session = ers_login(**{
            "username": "DerkaDerk",
            "password": "DerkaDerk-123",
            "token": "4nIvIJmcWBTL1VJVPlN!ZmTghxKxNcAazeP8tzhWs4gw7jL4zcv!4up3fxw@UyPe", })
        return session

    def download(self, item, save_path, *args, **kwargs):
        bands = ['thumbnail', 'MTL.json', 'coastal', 'blue', 'green', 'red', 'nir08', 'swir16', 'swir22', 'pan',
                 'cirrus',
                 'lwir11', 'lwir12', 'qa_pixel', 'qa_radsat', 'ANG.txt', 'VAA', 'VZA', 'SAA', 'SZA']
        session = self._get_session()
        with zipfile.ZipFile(save_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for name, asset in item["assets"].items():
                if name not in bands:
                    continue
                href = asset["href"]
                if Path(name).suffix:
                    suffix = ""
                else:
                    suffix = Path(href).suffix
                name = name + suffix
                r = session.get(href, timeout=120)
                r.raise_for_status()
                zipf.writestr(os.path.join(item["id"], name), r.content)
            if zipf.testzip() is not None:
                raise zipfile.BadZipFile("Invalid zip archive")
        return save_path

    def search(self, geojson: dict | None = None,
               start_datetime: datetime.datetime | None = None,
               stop_datetime: datetime.datetime | None = None,
               **kwargs):
        datetime = None
        if start_datetime and stop_datetime:
            datetime = f"{start_datetime.isoformat()}/{stop_datetime.isoformat()}"
        elif start_datetime:
            datetime = f"{start_datetime.isoformat()}/"
        elif stop_datetime:
            datetime = f"/{stop_datetime.isoformat()}"

        payload = {
            "collections": [self.collection],
            "intersects": geojson,
            "datetime": datetime,
        }

        items = list(self._client.search(
            **payload,
            **kwargs,
        ).items())
        return [i.to_dict() for i in items]
