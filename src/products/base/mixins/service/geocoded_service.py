def get_bounds_from_footprint(footprint):
    lons, lats = zip(*footprint["geometry"]["coordinates"])
    return max(lons), min(lons), max(lats), min(lats)