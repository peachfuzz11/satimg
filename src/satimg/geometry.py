"""Pixel-space geometry: a :class:`Window` region and a :class:`Grid` that lays
windows over an image.

Coordinates are pixel indices with the origin at the top-left. ``col`` runs along
the image ``x`` axis, ``row`` along ``y``. This is the ``(x, y, w, h)`` convention
used by rasterio windows, kept deliberately explicit here as ``col/row``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

EdgeMode = str  # "trim" | "pad" | "skip"
_EDGE_MODES = ("trim", "pad", "skip")


@dataclass(frozen=True, slots=True)
class Window:
    """An axis-aligned rectangle of pixels.

    A window is pure geometry -- it carries no data. It always remembers where it
    sits (``col``/``row``) so results computed on a patch can be mapped straight
    back onto the full image or into geographic space.
    """

    col: int
    row: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width < 0 or self.height < 0:
            raise ValueError(f"negative window size: {self.width}x{self.height}")

    # -- derived geometry -------------------------------------------------
    @property
    def col_end(self) -> int:
        return self.col + self.width

    @property
    def row_end(self) -> int:
        return self.row + self.height

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """``(col, row, col_end, row_end)`` -- i.e. ``(x1, y1, x2, y2)``."""
        return self.col, self.row, self.col_end, self.row_end

    @property
    def shape(self) -> tuple[int, int]:
        """``(height, width)`` -- numpy axis order."""
        return self.height, self.width

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """``(col, row)`` of the window centre."""
        return self.col + self.width / 2, self.row + self.height / 2

    # -- indexing helpers ----------------------------------------------
    def isel(self) -> dict[str, slice]:
        """Kwargs for :meth:`xarray.DataArray.isel` / ``__getitem__``."""
        return {"x": slice(self.col, self.col_end), "y": slice(self.row, self.row_end)}

    def slices(self) -> tuple[slice, slice]:
        """``(y_slice, x_slice)`` for indexing a numpy ``(..., y, x)`` array."""
        return slice(self.row, self.row_end), slice(self.col, self.col_end)

    # -- transforms ---------------------------------------------------
    def clip(self, width: int, height: int) -> "Window":
        """Intersect with the extent ``[0, width) x [0, height)``."""
        col = min(max(self.col, 0), width)
        row = min(max(self.row, 0), height)
        return Window(
            col,
            row,
            min(self.col_end, width) - col,
            min(self.row_end, height) - row,
        )

    def pad(self, margin: int) -> "Window":
        """Grow the window by ``margin`` pixels on every side."""
        return Window(
            self.col - margin,
            self.row - margin,
            self.width + 2 * margin,
            self.height + 2 * margin,
        )

    def shift(self, dcol: int = 0, drow: int = 0) -> "Window":
        return Window(self.col + dcol, self.row + drow, self.width, self.height)

    def __contains__(self, point: tuple[float, float]) -> bool:
        col, row = point
        return self.col <= col < self.col_end and self.row <= row < self.row_end


def _axis_starts(length: int, size: int, stride: int) -> list[int]:
    if length <= 0:
        return []
    if size >= length:
        return [0]
    n = math.ceil((length - size) / stride) + 1
    return [i * stride for i in range(n)]


@dataclass(frozen=True, slots=True)
class Grid:
    """Lays a regular grid of :class:`Window` over a ``width x height`` image.

    ``size`` and ``overlap`` may be a single int (square) or a ``(w, h)`` pair.
    The step between windows is ``size - overlap``.

    ``edge`` decides what happens to windows that run past the image border:

    * ``"trim"`` -- clip them to the image (last row/col of patches is smaller).
    * ``"pad"``  -- keep them full size; the reader zero-fills the overhang so
      every patch has identical shape (what you want for batched inference).
    * ``"skip"`` -- drop them, keeping only windows fully inside the image.
    """

    width: int
    height: int
    size: int | tuple[int, int]
    overlap: int | tuple[int, int] = 0
    edge: EdgeMode = "trim"

    def __post_init__(self) -> None:
        if self.edge not in _EDGE_MODES:
            raise ValueError(f"edge must be one of {_EDGE_MODES}, got {self.edge!r}")
        w, h = self._size
        ox, oy = self._overlap
        if w <= 0 or h <= 0:
            raise ValueError("size must be positive")
        if ox >= w or oy >= h:
            raise ValueError("overlap must be smaller than size")

    @property
    def _size(self) -> tuple[int, int]:
        return self.size if isinstance(self.size, tuple) else (self.size, self.size)

    @property
    def _overlap(self) -> tuple[int, int]:
        return self.overlap if isinstance(self.overlap, tuple) else (self.overlap, self.overlap)

    def _windows(self) -> Iterator[Window]:
        (sw, sh), (ox, oy) = self._size, self._overlap
        for row in _axis_starts(self.height, sh, sh - oy):
            for col in _axis_starts(self.width, sw, sw - ox):
                win = Window(col, row, sw, sh)
                if self.edge == "pad":
                    yield win
                elif self.edge == "skip":
                    if win.col_end <= self.width and win.row_end <= self.height:
                        yield win
                else:  # trim
                    yield win.clip(self.width, self.height)

    def __iter__(self) -> Iterator[Window]:
        return self._windows()

    def __len__(self) -> int:
        return sum(1 for _ in self._windows())

    def batched(self, n: int) -> Iterator[list[Window]]:
        """Yield windows in lists of up to ``n`` (last list may be shorter)."""
        if n <= 0:
            raise ValueError("batch size must be positive")
        batch: list[Window] = []
        for win in self._windows():
            batch.append(win)
            if len(batch) == n:
                yield batch
                batch = []
        if batch:
            yield batch
