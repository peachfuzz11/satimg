# satimg

A uniform, tile-friendly wrapper over satellite imagery. Point it at a product
directory and you get the same interface regardless of sensor: two views on the
pixels and one clean way to walk them in windows without ever losing track of
where a window came from.

Supported products: Sentinel-1 GRD (IW / EW), Sentinel-2 L1C, Landsat C2 L1,
NISAR L2 GCOV, VIIRS Day/Night Band (NOAA SDR and NASA L1B).

## Concepts

| Object | What it is |
| --- | --- |
| `Product` | A product directory or zip archive. Exposes `.raw`, `.visual` (DataArrays), `.patches(...)`, `.width/.height/.bands`, metadata. |
| `Window` | An immutable pixel rectangle (`col`, `row`, `width`, `height`). Pure geometry. |
| `Grid` | Lays `Window`s over an image with a step and an edge policy. |
| `Patch` | One window's `raw` / `visual` / `meta`, already sliced (lazy `DataArray`s / `PatchMeta`). |

`.raw` and `.visual` are plain lazy `xarray.DataArray`s, dims `(band, y, x)` —
use them with the full xarray API directly.

## Opening a product

`satimg.open` takes a product directory or a zip archive — detected by
content, not by extension — and returns the matching `Product` either way.

A **directory** gives you full access, same as ever:

```python
import satimg

product = satimg.open("/data/S2A_MSIL1C_20220114T103401_..._T33UUB_....SAFE")

product.raw                    # -> xarray.DataArray
for patch in product.patches(512):
    ...
```

A **zip archive** is read zip-native, straight off the archive — nothing is
ever extracted to disk, and there's no temp dir to release, so a `with`
block isn't needed just for a one-off read. Everything works, pixels
included: metadata files are read with `zipfile`, rasters by GDAL's
`/vsizip/`, which seeks inside the archive:

```python
product = satimg.open("/data/S2A_MSIL1C_20220114T103401_..._T33UUB_....SAFE.zip")

product.timestamp
product.metadata
for patch in product.patches(512):
    ...
```

Copernicus downloads, and the zips the Landsat connector writes, store their
rasters uncompressed (they are compressed already), and GDAL reads those at
plain file speed. A deflated archive works too, but every seek re-inflates from an
earlier point; for heavy pixel work on one, extract it first with
`satimg.open_zip`:

```python
with satimg.open_zip("/data/scene.zip") as product:
    for patch in product.patches(512):
        ...
```

This returns a directory-backed product exactly like `satimg.open` on a
directory would. It extracts to a temp dir that is removed — along with the
product's open rasters — when the block exits; read what you need inside
the block, since a lazy view can't be rebuilt afterwards. Pass `dest=` to
control where that temp dir is created.

A product's open rasters (and, for a zip-native one, its open archive) are
released when its `with` block exits, so use `with satimg.open(path) as
product:` — zipped or not — if you loop over many products, to release
handles promptly rather than waiting on garbage collection.

### Two views on the pixels

```python
import satimg

product = satimg.open("/data/S2A_MSIL1C_20220114T103401_..._T33UUB_....SAFE")

product.raw     # every native band, merged onto one grid, native dtype  -> xarray.DataArray
product.visual  # uint8 visualisation: 3-band true colour for optical    -> xarray.DataArray
                # sensors (Sentinel-2, Landsat), 1-band greyscale for SAR
                # (Sentinel-1, NISAR)
```

Both are lazy `DataArray`s over the **same grid**, so a `Window` means the same
thing in each.

`visual_overview(factor)` is the visual decimated `factor` times, read from the
overviews the product's own rasters carry (Sentinel-1 GRD COGs have 2x to 64x),
so a zoomed-out view or a thumbnail reads a fraction of the pixels.
`visual_overviews` lists the factors on offer, starting with 1 (`visual`
itself):

```python
product.visual_overviews       # e.g. [1, 2, 4, 8, 16, 32, 64]
product.visual_overview(64)    # the whole scene in a few hundred pixels a side
``` The `band` axis is labelled, so select by name:

```python
product.raw.sel(band="red")        # Landsat / (SWIR etc.)  ; "B04" for Sentinel-2, "VV" for Sentinel-1
product.raw.band.values            # e.g. ['coastal', 'blue', 'green', 'red', ...]
product.raw.wavelength_nm          # Sentinel-2 only: central wavelength per band
```

### Walking the image in patches

```python
for patch in product.patches(512, overlap=64):
    tile = patch.raw.values        # np.ndarray, (band, 512, 512), native dtype
    x0, y0 = patch.col, patch.row  # exact offset in the full image
    ...
```

`for patch in product` is shorthand for `product.patches()` (raw bands,
512 px, no overlap). Every patch from `product.patches()` / `patches_at()`
already carries `.raw` / `.visual` / `.meta` for that window (see
[Reading raw + visual + metadata together](#reading-raw--visual--metadata-together)) --
`product.patches()` / `patches_at()` are the only iterators in the library.

`product.patches()` opens `raw` and `visual` chunked to exactly the window
size before iterating, so each patch read decodes just the one tile it covers.
`patches_at()`, whose windows sit wherever the points are, reads them from the
default 1024 px chunks instead: one patch at a time, as a web request would,
costs a fraction of a second.

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
    tile = patch.raw.values         # (band, 512, 512), centred on the point
```

`satimg.windows_at(points, size)` gives the bare `Window`s if you don't need a
patch bound.

### Reading raw + visual + metadata together

Every patch from `product.patches()` / `product.patches_at()` already carries
its own window's raw, visual and metadata -- one window, every view.

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
```

### Mapping results back

```python
patch = next(iter(product.patches(1024)))
boxes = detect(patch.visual.values)     # boxes in patch pixels
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
| VIIRS DNB | `satellite_zenith`, `satellite_azimuth`, `solar_zenith`, `solar_azimuth`, `lunar_zenith`, `lunar_azimuth` |
| NISAR GCOV | `incidence_angle`, `elevation_angle`, `slant_range` (metres), `los_east` / `los_north` / `along_track_east` / `along_track_north` (unit vectors), `ground_track_velocity` (m/s) |

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

### Geometry without the product

Every product's `geometry` -- its transformer, image shape, coarse metadata
grids and scene attrs -- is a small dict that survives strict JSON (e.g. a
PostgreSQL `jsonb` column; `NaN` is stored as `null`). Store it while the
product is open and keep converting coordinates and sampling metadata after the
file is gone. For Sentinel-1, load it as a `Sentinel1Geometry` to also get the
SAR geometry (Doppler shift, headings):

```python
with satimg.open("S1A_IW_GRDH_....SAFE.zip") as product:
    stored = product.geometry.to_dict()      # ~30-80 KB

geometry = satimg.Sentinel1Geometry.from_dict(stored)   # SceneGeometry for other products
geometry.transformer.latlon_to_rowcol((lat, lon))
geometry.metadata.incidence_angle.at((row, col))
geometry.doppler_azimuth_shift((row, col), speed_mps, heading_deg)   # Sentinel-1 only
```

For NISAR, load it as a `NisarGeometry` the same way; for VIIRS, as a
`ViirsGeometry` (its swath transformer stored to within 250 m, ~350 KB).

Landsat's angle fields and NISAR's radar-grid fields are stored resampled to 32 x 32.

## NISAR

A NISAR L2 GCOV granule is a single `.h5` file; `satimg.open` takes it as it
is (or a zip holding it):

```python
product = satimg.open("NISAR_L2_PR_GCOV_..._001.h5")

product.raw.band.values    # ['HHHH', 'HVHV'] -- frequency A diagonal terms, linear gamma-0 power, NaN off-swath
product.visual             # 1-band dB stretch, like Sentinel-1
```

GCOV is geocoded -- a north-up UTM (or polar stereographic) grid -- not a
radar-geometry GRD, so it is neither rotated into the flight direction nor
mirrored. A moving ship's Doppler displacement still runs along the ground
track, which crosses the image at an angle; NISAR also looks *left*. Both are
read from the product's own line-of-sight and along-track vectors, so:

```python
product.doppler_azimuth_shift_m(rowcol, speed_mps, heading_deg)  # metres along track
product.doppler_pixel_shift(rowcol, speed_mps, heading_deg)      # (d_row, d_col)
product.correct_position(lat, lon, speed_mps, heading_deg)       # same as Sentinel-1
product.heading_in_image(rowcol, heading_deg)                    # grid convergence only, no mirroring
```

The detection / characterisation models were trained on Sentinel-1; NISAR's
`visual` is stretched so open sea is dark and ships bright in the same way,
but L-band at 10-20 m is a different look -- expect to fine-tune before
trusting them on it. NISAR also images the open ocean only sparsely; coasts,
straits and sea ice are where its coverage is.

## VIIRS Day/Night Band

The JPSS satellites' VIIRS Day/Night Band sees light at night down to a few
nW / (cm^2 sr): towns, gas flares -- and lit boats, above all the squid
jiggers and other light-luring fleets, at ~750 m, every night, worldwide.
It comes in two formats, both read the same way -- the data file with its
geolocation file next to it, or a folder / zip holding both:

| Format | Files | Granule | From |
| --- | --- | --- | --- |
| NOAA SDR (`ViirsSDRProduct`) | `SVDNB_*.h5` + `GDNBO_*.h5` | ~85 s, 768 x 4064 | NOAA's open AWS buckets, no account |
| NASA L1B (`ViirsL1BProduct`) | `V??02DNB.*.nc` + `V??03DNB.*.nc` | 6 min, ~3232 x 4064 | LAADS DAAC, Earthdata login |

```python
product = satimg.open("SVDNB_j01_d20261101_t1630064_..._oebc_ops.h5")   # or "VJ102DNB.A2026305.1630.021....nc"

product.raw            # radiance, W / (cm^2 sr), float32 -- x 1e9 for nW
product.visual         # log stretch, moonlit cloud to boat lights
product.metadata.attrs["moon_illumination"]   # 0-1: how bright the background is
```

The image is in sensor (swath) geometry, not map projected: `transformer` is a
`SwathTransformer` over the GEO file's per-pixel positions, scan by scan, to
within 50 m. Boats show up as their lights, a pixel or two each, so they are
found by a spike detector rather than the image model:

```python
from satimg.prediction import ViirsBoatDetector

lights = ViirsBoatDetector(product).detect()          # every light, tagged
boats = [d for d in lights if "on_land" not in d.tags and "platform" not in d.tags]
```

It finds local spikes well clear of the local background and its noise,
drops energetic-particle hits (single pixels, frequent in the South Atlantic
Anomaly) and the noisy swath-edge columns, and skips anything not dark
enough. In the spirit of NOAA's VIIRS Boat Detection, not a reimplementation;
thresholds were set on a few NOAA-20 granules. It does not see unlit vessels,
sees through thin cloud only (the light is blurred, often lost), can be
fooled by lightning and moonlit cloud edges, and the moon's glint raises the
floor -- treat it as a lead, not a census.

`get_connector("viirs-dnb")` (or `satellite="snpp"` / `"noaa-21"`) reads NOAA's
open buckets on AWS -- no account for search or download:

```python
connector = get_connector("viirs-dnb")
items = connector.search(geojson=aoi, start_datetime=t0, stop_datetime=t1)   # night only by default
connector.download(items[0], "/data/granule.zip")
```

There is no catalogue, so search lists granules by time and reads each one's
footprint out of its HDF5 by range requests: keep the window to hours.

`get_connector("viirs-dnb-l1b")` fetches the NASA L1B pairs instead: CMR
search (anonymous, catalogued, `day_night="night"`), download with an
Earthdata login (`EARTHDATA_TOKEN` etc., as for NISAR).

## Downloading

`search` is anonymous; `download` needs a (free) account for the archive.
Pass credentials to the connector, or set environment variables
(`CDSE_USERNAME` / `CDSE_PASSWORD` for Copernicus, `USGS_USERNAME` /
`USGS_PASSWORD` / optional `USGS_TOKEN` for USGS, `EARTHDATA_TOKEN` -- or
`EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD` -- for NISAR via NASA Earthdata).

```python
from satimg import get_connector

connector = get_connector("sentinel-2-l1c")                     # search only
items = connector.search(geojson=aoi, start_datetime=start, stop_datetime=stop)

connector = get_connector("sentinel-2-l1c",
                          username="me@example.com", password="...")
connector.download(items[0], "/data/scene.zip")
```

`get_connector("nisar-gcov")` searches the ASF DAAC; its `download` writes the
granule's own `.h5` rather than a zip. `get_connector("viirs-dnb")` needs no
account at all (see [VIIRS](#viirs-daynight-band)). Every other `download`
gives you a zip archive — see
[Opening a product](#opening-a-product) for reading its metadata straight
off the archive, or extracting it for pixel access.

## Land, wind farms, ports, platforms and fish farms

Every mask answers `mask.contains(lat, lon)` for matching `lat`/`lon` points
(scalars or arrays of any shape), giving booleans of that shape:

```python
from satimg.fishfarms import FishFarmMask
from satimg.landmask import LandMask
from satimg.platforms import PlatformMask
from satimg.ports import HarborMask, PortMask, features_in_bbox
from satimg.windfarms import WindfarmMask

LandMask().contains(lat, lon)              # on land
WindfarmMask(year=2025).contains(lat, lon)  # in a wind farm built by 2025 (any farm without a year)
PortMask().contains(lat, lon)              # in a commercial port
HarborMask().contains(lat, lon)            # in a marina or other small-boat harbor
PlatformMask().contains(lat, lon)          # at an offshore platform
FishFarmMask().contains(lat, lon)          # at an offshore fish farm

features_in_bbox(min_lon, min_lat, max_lon, max_lat)  # the port and harbor areas, e.g. for a map
```

Each data file loads once per process (the ports take ~0.5 s), whatever the
number of masks. Points a detection's centre would never quite hit are widened:
an offshore turbine is a 50 m disc, a platform a 200 m disc, a fish farm a 250 m
disc and a berth a 150 m disc, and a port is grown by 50 m to take in the ships
at berth. Berths and oil terminals by the water count as ports.

`Detector` tags every detection with where it lies, in `Detection.tags`:
`on_land`, `turbine`, `platform`, `fishery`, `in_port` and `in_harbor` (a ship at a quay can
be both `on_land` and `in_port`). No detection is dropped for where it lies;
filter on the tags. `satimg.prediction.context_tags(lat,
lon, year=None)` gives the same tags for any points.

The data is rebuilt by `scripts/build_ports.py`, `scripts/build_windfarms.py`,
`scripts/build_platforms.py` and `scripts/build_fishfarms.py`.

Data: ports, harbors, berths, oil terminals, platforms, and wind and fish farms
outside EMODnet's © OpenStreetMap contributors, available under the
[ODbL](https://opendatacommons.org/licenses/odbl/); European wind and fish farms from
[EMODnet Human Activities](https://emodnet.ec.europa.eu/en/human-activities).

## Deep Zoom

```python
from satimg.deepzoom import DeepZoom

DeepZoom(tile_size=512).build(product.visual, "/tmp/scene")  # -> /tmp/scene.dzi
```

## Adding a product

Subclass `Product`, implement the abstract hooks (`_open_raw(tile, overview)`,
`_render_visual(raw, tile)` — both return an `xarray.DataArray` dask-chunked to
`tile`; normalise dims with `satimg.as_band_yx(...)` and name the band axis
with `satimg.label_bands(da, [...])` — `transformer`, `timestamp`,
`footprint`, `thumbnail`), optionally override `_read_metadata` to expose
per-pixel metadata, and register a filename pattern. For `visual_overview`,
`_open_raw` also takes an `overview` factor (pass it on to `merge_bands`) and
`_visual_paths()` names the rasters the visual is read from:

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
