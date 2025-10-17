class Slice:

    def __init__(self, i: int, j: int, width: int, height: int):
        self._i = i
        self._j = j
        self._width = width
        self._height = height

    @property
    def i(self):
        return self._i

    @property
    def j(self):
        return self._j

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def __repr__(self):
        return f"{type(self)} i:{self.i} j:{self.j} width:{self.width} height:{self.height}"
