from . import maps


def get_iw_latlon(polygon, resolution=5):
    if isinstance(polygon, tuple):  # nothing to do
        return polygon

    # resolution is the number of point per km
    height = int(maps.get_distance(polygon[0, 1], polygon[0, 0], polygon[1, 1], polygon[1, 0]) * resolution)
    width = int(maps.get_distance(polygon[0, 1], polygon[0, 0], polygon[-1, 1], polygon[-1, 0]) * resolution)
    owiLat, owiLon = maps.grid_from_polygon(polygon, (height, width))
    return owiLat, owiLon


def getter_polygon_from_key(sensor_operational_mode='IW', polarisation_mode='VV'):
    raise DeprecationWarning
