import os
import utm
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from utils.misc import platform_cmap_args

    
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

def grid_from_polygon(polygon, shape):
    utm_easting, utm_northing, ZONE_NUMBER, ZONE_LETTER = utm.from_latlon(polygon[:,1], polygon[:,0])
    utm_polygon = np.stack((utm_easting, utm_northing), axis=1)
    
    easting_grid = np.zeros(shape)
    northing_grid = np.zeros(shape)
    
    v1 = utm_polygon[1]
    v2 = utm_polygon[2]
    
    easting_grid[0]  = v1[1] if v1[1] == v2[1] else np.arange(v1[1], v2[1], (v2[1]-v1[1])/easting_grid.shape[1])[:easting_grid.shape[1]]
    northing_grid[0] = v1[0] if v1[0] == v2[0] else np.arange(v1[0], v2[0], (v2[0]-v1[0])/northing_grid.shape[1])[:northing_grid.shape[1]]
    
    v1 = utm_polygon[0]
    v2 = utm_polygon[3]
    easting_grid[-1]  = v1[1] if v1[1] == v2[1] else np.arange(v1[1], v2[1], (v2[1]-v1[1])/easting_grid.shape[1])[:easting_grid.shape[1]]
    northing_grid[-1] = v1[0] if v1[0] == v2[0] else np.arange(v1[0], v2[0], (v2[0]-v1[0])/northing_grid.shape[1])[:northing_grid.shape[1]]
    for i in range(easting_grid.shape[1]):
        v1 = easting_grid[0,i]
        v2 = easting_grid[-1,i]
        easting_grid[:, i] = v1 if v1 == v2 else np.arange(v1, v2, (v2-v1)/easting_grid.shape[0])[:easting_grid.shape[0]]
        
        v1 = northing_grid[0,i]
        v2 = northing_grid[-1,i]
        northing_grid[:, i] = v1 if v1 == v2 else np.arange(v1, v2, (v2-v1)/northing_grid.shape[0])[:northing_grid.shape[0]]
 
    lat_grid, lon_grid = utm.to_latlon(northing_grid, easting_grid, ZONE_NUMBER, ZONE_LETTER, strict=False)
    return lat_grid, lon_grid

    
def plot_polygon(polygon, m):
    if polygon is not None:
        x, y = m(polygon[:,0], polygon[:,1])
        plt.plot(x, y, color="black", linestyle='--')
        plt.plot(x[[0,-1]], y[[0,-1]], color="black", linestyle='--')

        
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


def plot_on_map(platform, channel, data, platform_lat, platform_lon, lat_grid, lon_grid, filename, m=None, polygon=None, suptitle=None):
    if m is None:
        plt.figure(figsize=(12,12))
        m = ini_map(lat_grid, lon_grid)

    kwargs, norm, colorbar_postprocess = platform_cmap_args(platform, channel)
    colormesh = m.pcolormesh(platform_lon, platform_lat, data, norm=norm, latlon=True, shading='auto', **kwargs)
    
    colorbar = plt.colorbar(fraction=0.046, pad=0.04, orientation='horizontal')
    if colorbar_postprocess is not None: colorbar_postprocess(colorbar)

    plot_polygon(polygon, m)

    if suptitle is not None:
        plt.suptitle(suptitle)
    plt.tight_layout()
    os.makedirs(os.path.split(filename)[0], exist_ok=True)
    plt.savefig(filename)

    colorbar.remove()
    colormesh.remove()
    return m
