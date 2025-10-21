import os
from datetime import datetime
from xml.etree import ElementTree

import rioxarray
import xarray

from satproducts.common.utils import os_utils


def get_manifest_data(path_to_file):
    meta_data_file = os_utils.find_file(path_to_file, "manifest.safe")
    tree = ElementTree.parse(meta_data_file)
    root = tree.getroot()
    namespaces = {
        'safe': 'http://www.esa.int/safe/sentinel-1.0',
        'gml': 'http://www.opengis.net/gml'
    }
    start_time_str = root.find('.//safe:startTime', namespaces).text
    start_time = datetime.strptime(start_time_str, "%Y-%m-%dT%H:%M:%S.%f")
    stop_time_str = root.find('.//safe:stopTime', namespaces).text
    stop_time = datetime.strptime(stop_time_str, "%Y-%m-%dT%H:%M:%S.%f")
    timestamp = (stop_time - start_time) / 2 + start_time
    coordinates_str = root.find('.//gml:coordinates', namespaces).text
    coordinates_list = coordinates_str.split()
    geo_coordinates = []
    for coordinate in coordinates_list:
        lat, lon = map(float, coordinate.split(','))
        geo_coordinates.append((lon, lat))
    geojson_data = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": geo_coordinates
        }
    }
    data = {
        "timestamp": timestamp,
        "footprint": geojson_data
    }
    return data


def open_dataarrays(product_path):
    file_path = os.path.join(product_path, "measurement")
    file_list = sorted([os.path.join(file_path, f) for f in os.listdir(file_path)], key=lambda f: f[::-1])
    das = [rioxarray.open_rasterio(f).chunk() for f in file_list]
    return xarray.concat(das, dim="band")
