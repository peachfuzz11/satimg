import pytest

from satimg.geometry import Grid, Window, windows_at


class TestWindow:
    def test_bounds_and_shape(self):
        w = Window(10, 20, 100, 50)
        assert w.bounds == (10, 20, 110, 70)
        assert w.shape == (50, 100)
        assert w.area == 5000
        assert w.center == (60, 45)

    def test_isel_and_slices(self):
        w = Window(10, 20, 100, 50)
        assert w.isel() == {"x": slice(10, 110), "y": slice(20, 70)}
        ys, xs = w.slices()
        assert (ys, xs) == (slice(20, 70), slice(10, 110))

    def test_clip_inside(self):
        assert Window(0, 0, 100, 100).clip(512, 512) == Window(0, 0, 100, 100)

    def test_clip_overhang(self):
        assert Window(450, 480, 100, 100).clip(512, 512) == Window(450, 480, 62, 32)

    def test_clip_fully_outside(self):
        clipped = Window(600, 600, 100, 100).clip(512, 512)
        assert clipped.area == 0

    def test_pad_and_shift(self):
        assert Window(10, 10, 20, 20).pad(5) == Window(5, 5, 30, 30)
        assert Window(10, 10, 20, 20).shift(3, -4) == Window(13, 6, 20, 20)

    def test_contains(self):
        w = Window(0, 0, 10, 10)
        assert (5, 5) in w
        assert (10, 10) not in w

    def test_negative_size_rejected(self):
        with pytest.raises(ValueError):
            Window(0, 0, -1, 10)


class TestGrid:
    def test_default_edge_is_pad(self):
        windows = list(Grid(1000, 800, 256))
        assert all(w.width == 256 and w.height == 256 for w in windows)
        assert windows[-1] == Window(768, 768, 256, 256)

    def test_trim_covers_every_pixel(self):
        grid = Grid(1000, 800, 256, edge="trim")
        windows = list(grid)
        assert len(windows) == len(grid) == 4 * 4
        assert windows[-1].bounds == (768, 768, 1000, 800)
        assert all(w.col_end <= 1000 and w.row_end <= 800 for w in windows)

    def test_pad_keeps_full_size(self):
        windows = list(Grid(1000, 800, 256, edge="pad"))
        assert all(w.width == 256 and w.height == 256 for w in windows)
        assert windows[-1] == Window(768, 768, 256, 256)

    def test_skip_drops_partial_tiles(self):
        windows = list(Grid(1000, 800, 256, edge="skip"))
        assert all(w.col_end <= 1000 and w.row_end <= 800 for w in windows)
        assert len(windows) == 3 * 3

    def test_overlap_changes_stride(self):
        windows = list(Grid(1024, 512, 256, overlap=64, edge="pad"))
        cols = sorted({w.col for w in windows})
        assert cols[:2] == [0, 192]  # stride = 256 - 64

    def test_rectangular_size_and_overlap(self):
        grid = Grid(600, 400, (300, 200), overlap=(100, 0), edge="skip")
        windows = list(grid)
        assert {w.shape for w in windows} == {(200, 300)}
        assert sorted({w.col for w in windows})[:2] == [0, 200]

    def test_batched(self):
        grid = Grid(1000, 1000, 256, edge="pad")  # 16 windows
        batches = list(grid.batched(5))
        assert [len(b) for b in batches] == [5, 5, 5, 1]

    def test_bad_params(self):
        with pytest.raises(ValueError):
            Grid(100, 100, 64, overlap=64)
        with pytest.raises(ValueError):
            Grid(100, 100, 64, edge="bogus")


class TestWindowsAt:
    def test_centres_each_window_on_its_point(self):
        windows = list(windows_at([(100, 50), (300, 300)], 64))
        assert windows[0] == Window(68, 18, 64, 64)   # 100 - 32, 50 - 32
        assert windows[1] == Window(268, 268, 64, 64)
        assert all(w.center == (p[0], p[1]) for w, p in
                   zip(windows, [(100, 50), (300, 300)]))

    def test_rectangular_size(self):
        (w,) = list(windows_at([(100, 100)], (40, 20)))
        assert w == Window(80, 90, 40, 20)

    def test_point_near_edge_still_full_size(self):
        (w,) = list(windows_at([(5, 5)], 64))
        assert (w.width, w.height) == (64, 64)
        assert w.col < 0 and w.row < 0  # overhangs the origin; reader zero-fills

    def test_odd_size_rounds(self):
        (w,) = list(windows_at([(10, 10)], 5))
        assert w == Window(8, 8, 5, 5)  # round(10 - 2.5) == 8

    def test_empty_points_yields_nothing(self):
        assert list(windows_at([], 64)) == []

    def test_bad_size_rejected(self):
        with pytest.raises(ValueError):
            list(windows_at([(0, 0)], 0))
