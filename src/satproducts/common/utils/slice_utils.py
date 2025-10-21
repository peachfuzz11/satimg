import typing
from itertools import islice


def slices(width: int, height: int, width_step: int, height_step: int = None):
    """Slice data into parts of width_step/height_step."""
    if height_step is None:
        height_step = width_step
    for i in range(0, width, width_step):
        for j in range(0, height, height_step):
            yield i, j, width_step, height_step


def patches(patches: typing.Tuple[int, int], width_step: int, height_step: int = None):
    """Slice data into parts of width_step/height_step."""
    if height_step is None:
        height_step = width_step
    for i, j in patches:
        yield i - width_step // 2, j - height_step // 2, i + width_step // 2, j + width_step // 2


def safe_slice(width: int, height: int, width_step: int, height_step: int = None):
    """Slice data into parts of width_step/height_step. The edges may be shorter."""
    for i, j, w, h in slices(width, height, width_step, height_step):
        w = min(i + width_step, width) - i
        h = min(j + height_step, height) - j
        yield i, j, w, h


def batched(iterable, batch_size):
    """Batch data into lists of length n. The last batch may be shorter."""
    it = iter(iterable)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            return
        yield batch
