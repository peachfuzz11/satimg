import abc
import io
import zipfile

import requests

from satproducts.common.stac_connector import STACConnector


class SentinelConnector(STACConnector, abc.ABC):
    STAC_ENDPOINT = "https://stac.dataspace.copernicus.eu/v1/"

    def download(self, item, save_path):
        url = item['assets']['PRODUCT']['href']
        session = self._get_session()
        response = session.get(url, allow_redirects=False)
        while response.status_code in (301, 302, 303, 307):
            url = response.headers['Location']
            response = session.get(url, allow_redirects=False)

        file_response = session.get(url, verify=True, allow_redirects=True)

        with zipfile.ZipFile(io.BytesIO(file_response.content)) as zip_file:
            if zip_file.testzip() is not None:
                raise zipfile.BadZipFile("Invalid zip archive")
        with open(save_path, "wb") as f:
            f.write(file_response.content)
        return save_path

    def _get_session(self, *args, **kwargs):
        username = "rib40681@dcobe.com"
        password = "Tordenskjold-123!"
        token_request = {
            "url": "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
            "data": {
                "client_id": "cdse-public",
                "username": username,
                "password": password,
                "grant_type": "password",
            }
        }
        r = requests.post(**token_request)
        r.raise_for_status()
        access_token = r.json()["access_token"]
        session = requests.Session()
        session.headers.update({'Authorization': f'Bearer {access_token}'})
        return session
