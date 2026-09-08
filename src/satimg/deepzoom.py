"""Write a Deep Zoom Image (DZI) pyramid from a :class:`~satimg.raster.Raster`.

    from satimg import open
    from satimg.deepzoom import DeepZoom

    DeepZoom(tile_size=512).build(open(path).visual, "/tmp/scene")

produces ``/tmp/scene.dzi`` + ``/tmp/scene_files/`` viewable with OpenSeadragon.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import xml.etree.ElementTree as ElementTree

import numpy
import PIL.Image

from satimg.raster import Raster

logger = logging.getLogger(__name__)

_NS = "http://schemas.microsoft.com/deepzoom/2008"


class DeepZoom:
    def __init__(self, tile_size: int = 512, image_format: str = "png"):
        self.tile_size = tile_size
        self.image_format = image_format

    def build(self, raster: Raster, out_path: str) -> str:
        files_dir = out_path + "_files"
        dzi_path = out_path + ".dzi"
        shutil.rmtree(files_dir, ignore_errors=True)
        os.makedirs(files_dir, exist_ok=True)

        data = raster.hwc()
        width, height = raster.width, raster.height
        logger.info(
            "building Deep Zoom pyramid: %dx%d tile=%d -> %s",
            width,
            height,
            self.tile_size,
            out_path,
        )
        self._write_descriptor(dzi_path, width, height)

        for level, scale in _pyramid(width, height):
            logger.debug("level %d (scale %d)", level, scale)
            level_dir = os.path.join(files_dir, str(level))
            os.makedirs(level_dir, exist_ok=True)
            step = slice(None, None, scale)
            level_data = data.isel(x=step, y=step).fillna(0)
            chunks = {"x": min(self.tile_size, level_data.sizes["x"]),
                      "y": min(self.tile_size, level_data.sizes["y"])}
            if "band" in level_data.dims:
                chunks["band"] = -1
            blocks = level_data.chunk(chunks).persist().data

            def save(block, block_info=None, _dir=level_dir):
                row, col = block_info[0]["chunk-location"][:2]
                path = os.path.join(_dir, f"{col}_{row}.{self.image_format}")
                PIL.Image.fromarray(numpy.asarray(block)).save(path)
                return block

            blocks.map_blocks(save, dtype="uint8").compute(scheduler="threads")
        logger.info("Deep Zoom pyramid written: %s", dzi_path)
        return dzi_path

    def _write_descriptor(self, dzi_path: str, width: int, height: int) -> None:
        image = ElementTree.Element(
            "Image",
            TileSize=str(self.tile_size),
            Overlap="0",
            Format=self.image_format,
            xmlns=_NS,
        )
        ElementTree.SubElement(image, "Size", Width=str(width), Height=str(height))
        ElementTree.ElementTree(image).write(dzi_path, encoding="utf-8", xml_declaration=True)


def _pyramid(width: int, height: int):
    max_level = math.ceil(math.log2(max(width, height)))
    for level in range(max_level + 1):
        yield level, 2 ** (max_level - level)
