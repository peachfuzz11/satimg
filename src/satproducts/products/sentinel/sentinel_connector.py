import io
import zipfile

import requests

from satproducts.products.base.connector import Connector


class SentinelConnector(Connector):
    search_url = "https://catalogue.dataspace.copernicus.eu/stac/search"
    collection = None
    product_type = None

    def search(self, geojson=None, start_datetime=None, stop_datetime=None, *args, **kwargs):
        payload = self._get_payload(geojson=geojson, start_datetime=start_datetime, stop_datetime=stop_datetime, *args,
                                    **kwargs)
        response = requests.post(self.search_url, json=payload)
        data = response.json()
        items = data.get("features", [])
        items.extend(self._get_next(data))
        return items

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

    def _get_payload(self, geojson=None, start_datetime=None, stop_datetime=None, ids=None,
                     exclude_ids=None, limit=1000,
                     *args, **kwargs):
        builder = SentinelPayloadBuilder(self.collection, self.product_type, limit=limit)
        if start_datetime:
            builder.with_start_time(start_datetime)
        if stop_datetime:
            builder.with_stop_time(stop_datetime)
        if geojson:
            builder.with_geojson(geojson)
        if exclude_ids:
            builder.exclude_ids(exclude_ids)
        if ids:
            builder.include_ids(ids)
        return builder.build()

    def _get_next(self, data):
        items = []
        while True:
            next_links = [l for l in data.get("links", []) if l.get("rel") == "next"]
            if not next_links:
                break
            next_url = next_links[0]["href"]
            response = requests.get(next_url)
            data = response.json()
            items.extend(data.get("features", []))
        return items

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


class SentinelPayloadBuilder:
    def __init__(self, collection, product_type, limit=1000):
        self.collection = collection
        self.product_type = product_type
        self.limit = limit
        self.filters = []

    def with_start_time(self, start_timestamp):
        self.filters.append({
            "op": "t_after",
            "args": [{"property": "datetime"}, {"timestamp": start_timestamp.isoformat()}]
        })
        return self

    def with_stop_time(self, stop_timestamp):
        self.filters.append({
            "op": "t_before",
            "args": [{"property": "datetime"}, {"timestamp": stop_timestamp.isoformat()}]
        })
        return self

    def with_geojson(self, geojson):
        self.filters.append({
            "op": "s_overlaps",
            "args": [
                {
                    "property": "geometry"
                },
                geojson,

            ]
        })
        return self

    def exclude_ids(self, ids):
        self.filters.append({
            "op": "not",
            "args": [{
                "op": "in",
                "args": [{"property": "id"}, ids]
            }]
        })
        return self

    def include_ids(self, ids):
        self.filters.append({
            "op": "in",
            "args": [
                {"property": "id"},
                ids
            ]
        })
        return self

    def build(self):
        return {
            "limit": self.limit,
            "filter-lang": "cql2-json",
            "filter": {
                "op": "and",
                "args": [
                    {
                        "op": "=",
                        "args": [{"property": "collection"}, self.collection]
                    },
                    {
                        "op": "=",
                        "args": [{"property": "productType"}, self.product_type]
                    },
                    *self.filters
                ]
            }
        }
