import os
import zipfile
from pathlib import Path

from satproducts.common.stac_connector import STACConnector
from satproducts.products.landsat.auth import ers_login


class LandsatL1GTConnector(STACConnector):
    STAC_ENDPOINT = "https://landsatlook.usgs.gov/stac-server"
    collection = "landsat-c2l1"

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
