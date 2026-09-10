# satimg

A uniform, tile-friendly wrapper over satellite imagery. Point it at a product
directory and you get the same interface regardless of sensor: two views on the
pixels and one clean way to walk them in windows without ever losing track of
where a window came from.

Supported products: Sentinel-1 GRD (IW / EW), Sentinel-2 L1C, Landsat C2 L1.

## Concepts

| Object | What it is |
| --- | --- |
| `Product` | A product directory. Exposes `.raw`, `.visual` (DataArrays), `.patches(...)`, `.width/.height/.bands`, metadata. |
| `Window` | An immutable pixel rectangle (`col`, `row`, `width`, `height`). Pure geometry. |
| `Grid` | Lays `Window`s over an image with a step and an edge policy. |
| `Patch` | A `Window` bound to the `DataArray` it came from. `.array` (lazy) / `.values` (numpy) + geo helpers. |

`.raw` and `.visual` are plain lazy `xarray.DataArray`s, dims `(band, y, x)` —
use them with the full xarray API directly.

### Two views on the pixels

```python
import satimg

product = satimg.open("/data/S2A_MSIL1C_20220114T103401_..._T33UUB_....SAFE")

product.raw     # every native band, merged onto one grid, native dtype  -> xarray.DataArray
product.visual  # uint8 visualisation: 3-band true colour for optical    -> xarray.DataArray
                # sensors (Sentinel-2, Landsat), 1-band greyscale for SAR
                # (Sentinel-1)
```

Both are lazy `DataArray`s over the **same grid**, so a `Window` means the same
thing in each. The `band` axis is labelled, so select by name:

```python
product.raw.sel(band="red")        # Landsat / (SWIR etc.)  ; "B04" for Sentinel-2, "VV" for Sentinel-1
product.raw.band.values            # e.g. ['coastal', 'blue', 'green', 'red', ...]
product.raw.wavelength_nm          # Sentinel-2 only: central wavelength per band
```

### Walking the image in patches

```python
for patch in product.patches(512, overlap=64):
    tile = patch.values            # np.ndarray, (band, 512, 512), native dtype
    x0, y0 = patch.col, patch.row  # exact offset in the full image
    lat, lon = patch.center_latlon # where the patch sits on Earth
    ...
```

`for patch in product` is shorthand for `product.patches()` (raw bands,
512 px, no overlap). Patches from `product.patches()` walk `raw` and expose
`.raw` / `.visual` / `.meta` for that window (see
[Reading raw + visual + metadata together](#reading-raw--visual--metadata-together));
to walk any *other* array (a derived index) use
`satimg.patches(da, 512, transformer=product.transformer)`.

`batch=n` yields lists of `n` patches instead of one at a time.

Tiling the whole image? Call `persist()` first — it reads `raw` and `visual`
into memory once and closes the underlying rasters, so every window is an
in-memory slice instead of a fresh GDAL read:

```python
product = satimg.open(path).persist()   # or .persist("visual") for just that view
for patch in product.patches(512):
    ...
```

A product releases its rasters when its `with` block exits, so looping over
many products never leaks file handles:

```python
with satimg.open(path) as product:
    ...
```

### Edge handling

`product.patches(size, edge=...)` decides what happens at the right / bottom border:

| `edge` | behaviour | use for |
| --- | --- | --- |
| `"pad"` (default) | every patch is exactly `size`, overhang zero-filled | batched model inference |
| `"trim"` | last row/col of patches is smaller — you get exactly the pixels that exist | analysis, lossless re-assembly |
| `"skip"` | drop partial patches, keep only full interior tiles | training-set extraction |

Indices are always conserved: `patch.window` tells you where the tile belongs,
`patch.window.clip(product.width, product.height)` gives its valid region.

### Patches at specific points

When you already know where to look — detections, sample sites, a sparse set of
tiles — `patches_at` visits only those points instead of sweeping the whole
image. Each `(col, row)` is the **centre** of its patch, and every patch is
exactly `size` (overhang zero-filled, so points near an edge still work):

```python
points = [(4096, 2048), (10500, 512), (0, 0)]
for patch in product.patches_at(points, 512):
    tile = patch.values             # (band, 512, 512), centred on the point
    lat, lon = patch.center_latlon  # ~ the point you asked for
```

`batch=n` works the same as for `patches()`. `satimg.windows_at(points, size)`
gives the bare `Window`s if you don't need a patch bound.

### Reading raw + visual + metadata together

A patch from `product.patches()` / `product.patches_at()` is **product-bound**:
one window, every view, plus metadata at three levels.

```python
scene = product.metadata.attrs                 # scene-level scalars, constant per product
                                               # e.g. {"mean_sun_zenith": 77.5, ...}

for p in product.patches_at(points, 512):
    raw = p.raw                                # xarray.DataArray (band, 512, 512), band-labelled
    red = p.raw.sel(band="red").values         # np.ndarray (512, 512)   ("B04" on Sentinel-2)
    nir = p.raw.sel(band="nir08").values       #                          ("B08" on Sentinel-2)
    rgb = p.visual.values                       # np.ndarray (3, 512, 512) uint8, same window

    angles = p.meta.sample()                    # {"sun_zenith": 78.1, "view_zenith": 4.9, ...}
                                               #   -> per-pixel fields, bilinear at the patch centre
    sza_grid = p.meta["sun_zenith"].values      # np.ndarray (512, 512) — the field over the patch

    lat, lon = p.center_latlon                  # where the patch sits on Earth
```

`.raw` / `.visual` / `.meta` need the patch to come from `product.patches*()`;
a bare `satimg.patches(da, ...)` patch only has `.array` / `.values`.

### Mapping results back

```python
patch = next(satimg.patches(product.visual, 1024, transformer=product.transformer))
boxes = detect(patch.values)            # boxes in patch pixels
boxes[:, [0, 2]] += patch.window.col    # -> full-image pixels
boxes[:, [1, 3]] += patch.window.row
latlon = product.transformer.rowcol_to_latlon(boxes[:, [1, 0]])
```

## Working with a single window

```python
from satimg import Window

win = Window(col=4096, row=2048, width=1024, height=1024)
chip = satimg.read_window(product.raw, win)   # lazy xarray.DataArray, (band, y, x)
chip.values                                   # numpy, zero-padded if the window overhangs
```

## Views compose

`.raw` / `.visual` are xarray, so derive new arrays with the xarray API and tile
them the same way — a bare `(y, x)` array is fine:

```python
b = product.raw                                  # Sentinel-2
green, nir = b.sel(band="B03"), b.sel(band="B08")
ndwi = (green - nir) / (green + nir)
for patch in satimg.patches(ndwi, 512):
    ...
```

## Per-pixel metadata

Every sensor ships scene geometry that varies across the image — Sentinel-1's
incidence / elevation angles, Sentinel-2's and Landsat's sun / viewing angles —
stored on a coarse grid. `product.metadata` exposes those as named fields that
bilinearly interpolate to any pixel:

```python
m = product.metadata
m.fields                          # ['sun_zenith', 'sun_azimuth', 'view_zenith', ...]
m.sun_zenith.at((row, col))       # -> float, bilinear on the coarse grid
m.sample((row, col))              # -> {field: value} for every field
m.sun_zenith.corners()            # -> {'top_left', 'top_right', 'bottom_left', 'bottom_right', 'center'}
m.corners()                       # -> {field: {'top_left': ..., ..., 'center': ...}}  (whole grid)
m.attrs                           # scalar scene-level metadata (mean angles, ...)
```

| Product | Fields (all `degrees` unless noted) |
| --- | --- |
| Sentinel-1 | `incidence_angle`, `elevation_angle`, `slant_range_time` (seconds), `height` (metres) |
| Sentinel-2 | `sun_zenith`, `sun_azimuth`, `view_zenith`, `view_azimuth` |
| Landsat | `sun_zenith`, `sun_azimuth`, `view_zenith`, `view_azimuth` |

This is per-pixel geometry (varies across the scene). Per-*band* identity lives
on `product.raw` instead — the labelled `band` coordinate, plus `wavelength_nm`
for Sentinel-2. Scene-wide scalars are `product.metadata.attrs`.

Each field also materialises as a lazy full-grid `(y, x)` `xarray.DataArray`
(`m.sun_zenith.grid`), and patches from `product.patches()` carry a matching lazy
view over their window:

```python
for patch in product.patches(512):
    patch.meta.sun_zenith          # lazy (512, 512) DataArray for this window
    patch.meta.at((y, x))          # -> {field: value} at a patch-local pixel
    patch.meta.sample()            # -> {field: value} at the patch centre
    patch.meta.corners()           # -> {field: {'top_left': ..., ..., 'center': ...}}
```

`corners()` samples five points (the four corners and the centre) without
materialising the patch grid — a cheap way to see how a field varies across a
tile. `corners()[field]["center"]` equals `sample()[field]`.

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
    product.persist()          # keep the pixels once the temp dir is gone
    ...
```

`open_zip` extracts to a temp dir that is removed — along with the product's
open rasters — when the block exits. `persist()` the views you still need
before then; a lazy view can't be rebuilt afterwards.

## Deep Zoom

```python
from satimg.deepzoom import DeepZoom

DeepZoom(tile_size=512).build(product.visual, "/tmp/scene")  # -> /tmp/scene.dzi
```

## Adding a product

Subclass `Product`, implement the abstract hooks (`raw`, `_render_visual` — both
return an `xarray.DataArray`; normalise dims with `satimg.as_band_yx(...)` and
name the band axis with `satimg.label_bands(da, [...])` — `transformer`,
`timestamp`, `footprint`, `thumbnail`), optionally override `_read_metadata` to
expose per-pixel metadata, and register a filename pattern:

```python
from satimg.registry import register

@register(r"^MY_SENSOR_\d{8}.*$")
class MySensorProduct(Product):
    ...
```

Importing `satimg.products` registers the built-ins.

## Logging

`satimg` logs under the `satimg.*` logger namespace and never installs a handler
of its own. Opt in from your application:

```python
import logging

logging.basicConfig(level=logging.INFO)              # your app configures handlers
logging.getLogger("satimg").setLevel(logging.DEBUG)  # DEBUG for per-tile / per-request detail
```

`INFO` covers milestones (downloads, Deep Zoom builds, detection runs); `DEBUG`
adds STAC queries, redirect hops, per-tile counts and product resolution.
Credentials are never logged.

## Development

```sh
uv sync
uv run pytest                      # fast unit tests + product integration tests
uv run pytest tests/test_geometry.py tests/test_tiling.py   # just the fast ones
```

Integration tests run against minified scenes under `tests/data/` — real
georeferencing and metadata, zeroed pixels.
