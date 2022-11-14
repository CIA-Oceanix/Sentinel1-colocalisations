import numpy as np
import utm
import PIL.Image
import sys
from zipfile import ZipFile
from datetime import datetime
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

def get_distance(source_latitude, source_longitude, target_latitude, target_longitude):
    EARTH_RADIUS = 6371

    d_lat = np.radians(target_latitude - source_latitude)
    d_lon = np.radians(target_longitude - source_longitude)
    a = (np.sin(d_lat / 2.) * np.sin(d_lat / 2.) +
         np.cos(np.radians(source_latitude)) * np.cos(np.radians(target_latitude)) *
         np.sin(d_lon / 2.) * np.sin(d_lon / 2.))
    c = 2. * np.arctan2(np.sqrt(a), np.sqrt(1. - a))
    d = EARTH_RADIUS * c
    return d

def dms2dd(degrees, minutes, seconds, direction):
    dd = float(degrees) + float(minutes)/60 + float(seconds)/(60*60);
    if direction == 'W' or direction == 'S':
        dd *= -1
    return dd

def grid_from_polygon(polygon, shape):
    utm_easting, utm_northing, ZONE_NUMBER, ZONE_LETTER = utm.from_latlon(polygon[:,1], polygon[:,0])
    utm_polygon = np.stack((utm_easting, utm_northing), axis=1)
    
    easting_grid = np.zeros(shape)
    northing_grid = np.zeros(shape)
    
    v1 = utm_polygon[1]
    v2 = utm_polygon[2]
    easting_grid[0] = np.arange(v1[1], v2[1], (v2[1]-v1[1])/easting_grid.shape[1])[:easting_grid.shape[1]]
    northing_grid[0] = np.arange(v1[0], v2[0], (v2[0]-v1[0])/northing_grid.shape[1])[:northing_grid.shape[1]]
    
    v1 = utm_polygon[0]
    v2 = utm_polygon[3]
    easting_grid[-1] = np.arange(v1[1], v2[1], (v2[1]-v1[1])/easting_grid.shape[1])[:easting_grid.shape[1]]
    northing_grid[-1] = np.arange(v1[0], v2[0], (v2[0]-v1[0])/northing_grid.shape[1])[:northing_grid.shape[1]]
    for i in range(easting_grid.shape[1]):
        v1 = easting_grid[0,i]
        v2 = easting_grid[-1,i]
        easting_grid[:, i] = np.arange(v1, v2, (v2-v1)/easting_grid.shape[0])[:easting_grid.shape[0]]
        
        v1 = northing_grid[0,i]
        v2 = northing_grid[-1,i]
        northing_grid[:, i] = np.arange(v1, v2, (v2-v1)/northing_grid.shape[0])[:northing_grid.shape[0]]
 
    lat_grid, lon_grid = utm.to_latlon(northing_grid, easting_grid, ZONE_NUMBER, ZONE_LETTER, strict=False)
    return lat_grid, lon_grid

    
def get_iw_latlon(polygon=None, metadata_filename=None, shape=None):
    if metadata_filename is not None:
        metadata = np.load(metadata_filename)
        return metadata['owiLat'], metadata['owiLon']
    else:
        if shape is None:
            height = int(get_distance(polygon[0,1], polygon[0,0], polygon[1,1], polygon[1,0])*5)
            width = int(get_distance(polygon[0,1], polygon[0,0], polygon[-1,1], polygon[-1,0])*5)
            return grid_from_polygon(polygon, (height, width))
        return grid_from_polygon(polygon, shape)
    

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

def ini_map(lats, lons, zoom=4, stride=1):
    min_lat = np.min(lats)
    min_lon = np.min(lons)
    max_lat = np.max(lats)
    max_lon = np.max(lons)
    
    lon_0 = (max_lon + min_lon) / 2
    lat_0 = (max_lat + min_lat) / 2
    
    delta_lon = max_lon - min_lon
    delta_lat = max_lon - min_lon
    
    m = Basemap(
        projection = "lcc",
        resolution = 'f',
        llcrnrlon = lon_0 - delta_lon/zoom, 
        llcrnrlat = lat_0 - delta_lat/zoom, 
        urcrnrlon = lon_0 + delta_lon/zoom, 
        urcrnrlat = lat_0 + delta_lat/zoom,
        lon_0 = lon_0,
        lat_0 = lat_0,
        lon_1 = lon_0 + delta_lon/zoom,
        lat_1 = lat_0 + delta_lat/zoom
    )
    m.drawcoastlines()
    m.drawmapboundary()
    
    parallels = np.arange(min_lat//stride*stride, max_lat//stride*stride+stride, stride)
    meridians = np.arange(min_lon//stride*stride, max_lon//stride*stride+stride, stride)
    m.drawparallels(parallels, labels=[False,True,True,False])
    m.drawmeridians(meridians, labels=[True,False,False,True])
    return m

def png_to_gif(input_filenames, output_filename):
    imgs = (PIL.Image.open(filename) for filename in input_filenames)
    img = next(imgs)
    img.save(fp=output_filename, format='GIF', append_images=imgs, save_all=True, duration=200, loop=0)

def log_print(s, f=print, format='%Y-%m-%d %H:%M:%S'):
    f(f"{datetime.now().strftime(format)}\t{s}")

def r_print(s):
    sys.stdout.write(f'\r{s}')


def plot_polygon(polygon, m):
    if polygon is not None:
        x, y = m(polygon[:,0], polygon[:,1])
        plt.plot(x, y, color="black", linestyle='--')
        plt.plot(x[[0,-1]], y[[0,-1]], color="black", linestyle='--')
