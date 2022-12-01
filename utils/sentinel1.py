from zipfile import ZipFile
import numpy as np

from utils.map import get_distance, grid_from_polygon


def get_iw_latlon(polygon, resolution=5):
    # resolution is the number of point per km
    height = int(get_distance(polygon[0,1], polygon[0,0], polygon[1,1], polygon[1,0])*5)
    width = int(get_distance(polygon[0,1], polygon[0,0], polygon[-1,1], polygon[-1,0])*5)
    owiLat, owiLon = grid_from_polygon(polygon, (height, width))
    return owiLat, owiLon
    
def getter_polygon_from_key(sensoroperationalmode='IW', polarisationmode='VV', verbose=1):
    products = {}

    short_name = f"listing_[{sensoroperationalmode}][{polarisationmode}]"
    with ZipFile(f'res/{short_name}.zip') as zip_archive:
        with zip_archive.open(short_name + '.txt') as file:
            for i, line in enumerate(file):
                line = line.decode("utf-8") 
                splitted_line = line.split()
                filename = splitted_line[1]
                orbitdirection = splitted_line[2]
                polygon = np.array([e.split(',')[::-1] for e in line.split('POLYGON')[1][1:-2].split()]).astype('float')[-1:0:-1]

                key = filename.split('_')[4].lower()
                products[key] = (filename, polygon, orbitdirection)
    
    def get_polygon_from_key(key):
        return products[key]
    
    return get_polygon_from_key
