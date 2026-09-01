# satimg

A uniform, tile-friendly wrapper over satellite imagery. Point it at a product
directory and you get the same interface regardless of sensor: three views on the
pixels and one clean way to walk them in windows without ever losing track of
where a window came from.

Supported products: Sentinel-1 GRD (IW / EW), Sentinel-2 L1C, Landsat C2 L1.

## Concepts

| Object | What it is |
| --- | --- |
| `Product` | A product directory. Exposes `.raw`, `.rgb`, `.gray`, `.patches(...)`, metadata. |
| `Raster` | A lazy `(band, y, x)` array. Reads nothing until you ask for values. |
| `Window` | An immutable pixel rectangle (`col`, `row`, `width`, `height`). Pure geometry. |
| `Grid` | Lays `Window`s over an image with a step and an edge policy. |
| `Patch` | A `Window` bound to the `Raster` it came from. `.array` (lazy) / `.values` (numpy) + geo helpers. |

### Three views on the pixels

```python
import satimg

product = satimg.open("/data/S2A_MSIL1C_20220114T103401_..._T33UUB_....SAFE")

product.raw    # every native band, merged onto one grid, native dtype   -> Raster
product.rgb    # 3-band uint8 true-colour visualisation                  -> Raster
product.gray   # 1-band uint8 visualisation                              -> Raster
```

All three are lazy `Raster`s over the same grid, so a `Window` means the same
thing in each.

### Walking the image in patches

```python
for patch in product.patches(512, overlap=64, kind="rgb"):
    tile = patch.values            # np.ndarray, (3, 512, 512), uint8
    x0, y0 = patch.col, patch.row  # exact offset in the full image
    lat, lon = patch.center_latlon # where the patch sits on Earth
    ...
```

`for patch in product` is shorthand for `product.patches()` (raw bands,
512 px, no overlap).

`kind=` picks the view (`"raw"` / `"rgb"` / `"gray"`). `batch=n` yields lists of
`n` patches instead of one at a time.

### Edge handling

`product.patches(size, edge=...)` decides what happens at the right / bottom border:

| `edge` | behaviour | use for |
| --- | --- | --- |
| `"trim"` (default) | last row/col of patches is smaller — you get exactly the pixels that exist | analysis, lossless re-assembly |
| `"pad"` | every patch is exactly `size`, overhang zero-filled | batched model inference |
| `"skip"` | drop partial patches, keep only full interior tiles | training-set extraction |

Indices are always conserved: `patch.window` tells you where the tile belongs,
`patch.window.clip(product.width, product.height)` gives its valid region.

### Mapping results back

```python
patch = next(product.patches(1024, kind="rgb"))
boxes = detect(patch.values)            # boxes in patch pixels
boxes[:, [0, 2]] += patch.window.col    # -> full-image pixels
boxes[:, [1, 3]] += patch.window.row
latlon = product.transformer.rowcol_to_latlon(boxes[:, [1, 0]])
```

## Working with a single window

```python
from satimg import Window

win = Window(col=4096, row=2048, width=1024, height=1024)
chip = product.raw.read(win)     # lazy xarray.DataArray, dims (band, y, x)
chip = product.raw.values(win)   # numpy, zero-padded if the window overhangs
```

## Rasters compose

```python
ndwi = product.raw.map(lambda d: (d.isel(band=2) - d.isel(band=7))
                                 / (d.isel(band=2) + d.isel(band=7)),
                       name="ndwi")
for patch in ndwi.patches(512):
    ...
```

## Downloading

`search` is anonymous; `download` needs a (free) account for the archive.
Pass credentials to the connector, or set environment variables
(`CDSE_USERNAME` / `CDSE_PASSWORD` for Copernicus, `USGS_USERNAME` /
`USGS_PASSWORD` / optional `USGS_TOKEN` for USGS).

```python
from satimg import get_connector

connector = get_connector("sentinel-2-l1c")                     # search only
items = connector.search(geojson=aoi, start_datetime=start, stop_datetime=stop)

connector = get_connector("sentinel-2-l1c",
                          username="me@example.com", password="...")
connector.download(items[0], "/data/scene.zip")

with satimg.open_zip("/data/scene.zip") as product:
    ...
```

## Deep Zoom

```python
from satimg.deepzoom import DeepZoom

DeepZoom(tile_size=512).build(product.rgb, "/tmp/scene")  # -> /tmp/scene.dzi
```

## Adding a product

Subclass `Product`, implement `raw`, `_render_rgb`, and the metadata properties,
and register a filename pattern:

```python
from satimg.registry import register

@register(r"^MY_SENSOR_\d{8}.*$")
class MySensorProduct(Product):
    ...
```

Importing `satimg.products` registers the built-ins.

## Development

```sh
uv sync
uv run pytest                      # fast unit tests + product integration tests
uv run pytest tests/test_geometry.py tests/test_raster.py   # just the fast ones
```

Integration tests run against minified scenes under `tests/data/` — real
georeferencing and metadata, zeroed pixels.
