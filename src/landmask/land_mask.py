from roaring_landmask import RoaringLandmask


class LandMask:

    def __init__(self):
        self.l = RoaringLandmask.new()

    def mask(self, lat, lon):
        on_land = self.l.contains_many(lon.ravel(), lat.ravel())
        return on_land

    def mask_grid(self, latlon):
        """NxMx2"""
        return self.mask(latlon[..., 0].ravel(), latlon[..., 1].ravel()).reshape(latlon.shape[:2])

    def mask_slice(self, image_slice):
        pass
