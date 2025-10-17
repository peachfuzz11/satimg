import os
from datetime import datetime
from xml.etree import ElementTree

import xarray

from common.utils import os_utils


def get_mtd_tl_data(path_to_file):
    meta_data_file = os_utils.find_file(path_to_file, "MTD_TL.xml")
    tree = ElementTree.parse(meta_data_file)
    root = tree.getroot()
    sensing_time = root.find('.//SENSING_TIME').text
    datetime_obj = datetime.strptime(sensing_time, "%Y-%m-%dT%H:%M:%S.%fZ")
    data = {
        "timestamp": datetime_obj
    }
    return data


def get_mtd_msil1c_data(path_to_file):
    meta_data_file = os_utils.find_file(path_to_file, "MTD_MSIL1C.xml")
    tree = ElementTree.parse(meta_data_file)
    root = tree.getroot()
    # Extract band information from the XML
    bands = []
    for i, elem in enumerate(root.iter("IMAGE_FILE")):
        band_id = i
        file_path = os.path.join(path_to_file, elem.text) + ".jp2"
        resolution = None
        physical_band = None
        for spectral_info in root.findall(".//Spectral_Information"):
            if int(band_id) == int(spectral_info.get("bandId")):
                resolution = spectral_info.find("RESOLUTION").text
                physical_band = spectral_info.get("physicalBand")
        band_meta = {"band_id": str(band_id), "file_path": file_path, "resolution": resolution,
                     "physical_band": physical_band}
        bands.append(band_meta)
    ext_pos_list = root.find('.//EXT_POS_LIST').text
    coordinates = [float(coord) for coord in ext_pos_list.split()]
    geo_coordinates = [(coordinates[i + 1], coordinates[i]) for i in range(0, len(coordinates), 2)]
    geojson_data = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": geo_coordinates
        }
    }
    data = {
        "bands": bands,
        "footprint": geojson_data
    }
    return data


def open_dataarrays(bands):
    das = [xarray.open_dataarray(b["file_path"]).chunk() for b in bands if b["physical_band"] is not None]
    return das


def reindex_and_concatenate_dataarrays(dataarrays):
    target_da = dataarrays[1]
    reindexed_das = [da.reindex_like(target_da, method="nearest").chunk() for da in dataarrays]
    return xarray.concat(reindexed_das, dim="band")
