import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from utils.misc import platform_cmap_args


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
        
    cmap, vmin, vmax = platform_cmap_args(platform, channel)
    
    colormesh = m.pcolormesh(platform_lon, platform_lat, data, latlon=True, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    colorbar = plt.colorbar(fraction=0.046, pad=0.04, orientation='horizontal')

    plot_polygon(polygon, m)

    if suptitle is not None:
        plt.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(filename)

    colorbar.remove()
    colormesh.remove()
    return m
