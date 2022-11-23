import os
import PIL.Image
import numpy as np
from datetime import datetime
from scipy.interpolate import griddata

from utils.map import plot_on_map
from utils.requests import download_files
from utils.misc import platform_cmap_args
from utils.read import read_from_files_per_platform
from utils.misc import log_print
from utils.map import grid_from_polygon, get_distance

from check_args import GOES_SERIE, HIMAWARI_SERIE, NEXRAD_BASIS, SATELLITE_PLATFORMS



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
    return new_data

def save_reprojection(platform, channel, data, filename):
    print(f"{platform=} {channel=}")
    kwargs = platform_cmap_args(platform, channel)[0]
    vmin = kwargs.get("vmin", np.nanmin(data))
    vmax = kwargs.get("vmax", np.nanmax(data))

    new_data = np.clip((data-vmin)/(vmax-vmin), 0, 1)
    new_data = kwargs["cmap"](new_data)
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
    
def generate_gif(iw_polygon, channel, urls_per_platforms, gif_filename, verbose, read_function, download=True):
    def png_to_gif(input_filenames, output_filename):
        imgs = (PIL.Image.open(filename) for filename in input_filenames)
        img = next(imgs)

        os.makedirs(os.path.split(output_filename)[0], exist_ok=True)
        img.save(fp=output_filename, format='GIF', append_images=imgs, save_all=True, duration=200, loop=0)
        
    lat_grid, lon_grid = increased_grid(iw_polygon, km_per_pixel=2, delta_factor=2)
    filenames_per_platform = download_files(urls_per_platforms, closest=False) if download else urls_per_platforms
    m = None

    if verbose: log_print(f"Generate .png")
    for platform in filenames_per_platform:
        png_filenames = []
        for date, filenames in filenames_per_platform[platform].items():
            platform_lat, platform_lon, data = read_function(filenames, platform, channel)
            if platform in SATELLITE_PLATFORMS:
                data, platform_lat, platform_lon = trim(data, platform_lat, platform_lon, lat_grid, lon_grid)

            folder = os.path.split(filenames[0])[0]
            suptitle = date.strftime('%Y-%m-%d %H:%M:%S')
            datestr = date.strftime('%Y%m%dt%H%M%S')
            filename = folder + f".{datestr}.{channel}.png"
            
            m = plot_on_map(platform, channel, data, platform_lat, platform_lon, lat_grid, lon_grid, filename, m=m, polygon=iw_polygon, suptitle=suptitle)
            png_filenames.append(filename)

        png_to_gif(png_filenames, gif_filename)
