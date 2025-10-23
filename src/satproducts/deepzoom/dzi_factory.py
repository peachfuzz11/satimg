import math
import os
import shutil
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import PIL.Image

from satproducts.products.base.product import Product


class DZIFactory:
    def __init__(self, out_path: str, tile_size: int = 512, image_format: str = "png"):
        self._out_path = out_path
        self._file_name = Path(out_path).name
        self._tile_size = tile_size
        self._image_format = image_format

        self._dzi_files_path = os.path.join(self._out_path + "_files")
        self._dzi_path = os.path.join(self._out_path + ".dzi")
        self._osd_path = os.path.join(self._out_path + ".html")

    def create(self, product: Product):
        os.makedirs(self._dzi_files_path, exist_ok=True)
        xr = product.viewable()
        width, height, channels = product.width, product.height, product.channels

        shutil.rmtree(self._dzi_files_path)
        self._create_dzi_descriptor(width, height)

        for level, scale in _build_pyramid(width, height):
            level_dir = os.path.join(self._dzi_files_path, str(level))
            os.makedirs(level_dir, exist_ok=True)
            s = slice(None, None, scale)
            resized_xr = xr.isel(**{"x": s, "y": s}).fillna(0)
            chunk = {'x': min(self._tile_size, resized_xr.sizes["x"]), 'y': min(self._tile_size, resized_xr.sizes["y"])}
            if resized_xr.sizes["band"] == 1:
                resized_xr = resized_xr.isel(band=0)
            else:
                chunk["band"] = -1
            resized_xr = resized_xr.chunk(chunk).data

            # Function to save each tile block
            def save_tile(block, block_info=None):
                i, j = block_info[0]['chunk-location'][:2]
                img_path = os.path.join(level_dir, f"{j}_{i}.{self._image_format}")
                _save_image(img_path, block, format=self._image_format)
                return block

            # Map save function across chunks
            resized_xr.map_blocks(save_tile, dtype="uint8").compute(scheduler='threads')
        return self._dzi_path

    def _create_dzi_descriptor(self, width, height):
        image = ElementTree.Element(
            "Image",
            TileSize=str(self._tile_size),
            Overlap="0",
            Format=self._image_format,
            xmlns="http://schemas.microsoft.com/deepzoom/2008",
        )
        size = ElementTree.SubElement(
            image, "Size", Width=str(width), Height=str(height)
        )
        tree = ElementTree.ElementTree(image)
        tree.write(self._dzi_path, encoding="utf-8", xml_declaration=True)

    def _create_osd_html(self):
        html_content = (
                """
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OpenSeadragon Viewer</title>
        <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/2.4.2/openseadragon.min.js"></script>
    </head>
    <body>
        <div id="openseadragon1" style="width: 100%; height: 90vh;"></div>
        <script type="text/javascript">
            var viewer = OpenSeadragon({
                id: "openseadragon1",
                prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/2.4.2/images/",
                tileSources: \"""" + Path(self._dzi_path).name + """",
                maxZoomPixelRatio: Infinity,
        });
    </script>
</body>
</html>
            """
        )
        with open(self._osd_path, "w") as f:
            f.write(html_content)


def _build_pyramid(width, height):
    max_dimension = max(width, height)
    max_level = math.ceil(math.log2(max_dimension))
    for level in range(max_level + 1):
        scale = 2 ** (max_level - level)
        yield level, scale


def _save_image(path, array, **kwargs):
    image = PIL.Image.fromarray(array)
    image.save(path, **kwargs)
