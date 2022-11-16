import os
import utm
import PIL.Image
import numpy as np
from datetime import datetime
from scipy.interpolate import griddata

from utils.map import plot_on_map
from utils.requests import download_files
from utils.misc import platform_cmap_args
from utils.read import read_from_files_per_platform
from utils.misc import log_print

from check_args import GOES_SERIE, HIMAWARI_SERIE, NEXRAD_BASIS, SATELLITE_PLATFORMS

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

def trim(data, platform_lat, platform_lon, owi_lat, owi_lon):
    def arg2d(array, f=np.nanargmin):
        return  np.unravel_index(f(array), array.shape)
        
    x1, y1 = arg2d(get_distance(platform_lat, platform_lon, np.nanmax(owi_lat), np.nanmax(owi_lon)))
    x2, y2 = arg2d(get_distance(platform_lat, platform_lon, np.nanmin(owi_lat), np.nanmin(owi_lon)))
    
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    platform_lat = platform_lat[x1:x2,y1:y2]
    platform_lon = platform_lon[x1:x2,y1:y2]
    data = data[x1:x2,y1:y2]
    return data, platform_lat, platform_lon


def reproject(platform, data, platform_lat, platform_lon, owi_lat, owi_lon):
    if platform in SATELLITE_PLATFORMS:
        data, platform_lat, platform_lon = trim(data, platform_lat, platform_lon, owi_lat, owi_lon)

    platform_lat[np.isnan(platform_lat)] = 0
    platform_lon[np.isnan(platform_lon)] = 0
    data[np.isnan(data)] = 0

    new_data = griddata(
        np.stack((platform_lat.flatten(), platform_lon.flatten()), axis=1),
        data.flatten(),
        np.stack((owi_lat.flatten(), owi_lon.flatten()), axis=1)
    ).reshape(owi_lat.shape).astype('float')
    

    """import matplotlib.pyplot as plt
    plt.figure(figsize=(16,4))
    plt.subplot(141)
    plt.imshow(platform_lat)
    plt.subplot(142)
    plt.imshow(platform_lon)
    plt.subplot(143)
    plt.imshow(data)
    plt.subplot(144)
    plt.imshow(new_data)
    plt.show()"""

    
    return new_data

def save_reprojection(platform, channel, data, filename):
    cmap, vmin, vmax = platform_cmap_args(platform, channel)

    new_data = np.clip((data-vmin)/(vmax-vmin), 0, 1)
    new_data = cmap(new_data)
    new_data = (new_data * 255).astype(np.uint8)

    os.makedirs(os.path.split(filename)[0], exist_ok=True)
    PIL.Image.fromarray(new_data).save(filename + ".png")
    np.savez_compressed(filename + ".npz", data)


def increased_grid(polygon, km_per_pixel=1, delta_factor=1):
    min_lat = np.min(polygon[:,1])
    max_lat = np.max(polygon[:,1])

    min_lon = np.min(polygon[:,0])
    max_lon = np.max(polygon[:,0])

    delta_lon = max_lon - min_lon
    delta_lat = max_lat - min_lat

    frame_min_lat = min_lat - delta_lat*delta_factor
    frame_max_lat = max_lat + delta_lat*delta_factor
    frame_min_lon = min_lon - delta_lon*delta_factor
    frame_max_lon = max_lon + delta_lon*delta_factor

    height = int(get_distance(frame_min_lat, frame_min_lon, frame_max_lat, frame_min_lon)/km_per_pixel)
    width =  int(max(
        get_distance(frame_min_lat, frame_min_lon, frame_min_lat, frame_max_lon),
        get_distance(frame_max_lat, frame_min_lon, frame_max_lat, frame_max_lon),
    )/km_per_pixel)
        
    frame_polygon = np.array(
        [
            [frame_min_lon, frame_min_lat],
            [frame_min_lon, frame_max_lat],
            [frame_max_lon, frame_max_lat],
            [frame_max_lon, frame_min_lat]
        ]
    )

    return grid_from_polygon(frame_polygon, (height, width))
    
def generate_gif(iw_polygon, channel, urls_per_platforms, gif_filename, verbose):
    def png_to_gif(input_filenames, output_filename):
        imgs = (PIL.Image.open(filename) for filename in input_filenames)
        img = next(imgs)
        img.save(fp=output_filename, format='GIF', append_images=imgs, save_all=True, duration=200, loop=0)
        
    lat_grid, lon_grid = increased_grid(iw_polygon, km_per_pixel=2, delta_factor=2)
    filenames_per_platform = download_files(urls_per_platforms, closest=False)
    m = None

    if verbose: log_print(f"Generate .png")
    for platform in filenames_per_platform:
        png_filenames = []
        for filenames in filenames_per_platform[platform]:
            platform_lat, platform_lon, data = read_from_files_per_platform(filenames, platform, channel)
            if platform in SATELLITE_PLATFORMS:
                data, platform_lat, platform_lon = trim(data, platform_lat, platform_lon, lat_grid, lon_grid)

            folder = os.path.split(filenames[0])[0]
            suptitle = datetime.strptime(os.path.split(folder)[1], '%Y%m%dt%H%M%S').strftime('%Y-%m-%d %H:%M:%S')
            filename = folder + f".{channel}.png"
            
            m = plot_on_map(platform, channel, data, platform_lat, platform_lon, lat_grid, lon_grid, filename, m=m, polygon=iw_polygon, suptitle=suptitle)
            png_filenames.append(filename)

        png_to_gif(png_filenames, gif_filename)
