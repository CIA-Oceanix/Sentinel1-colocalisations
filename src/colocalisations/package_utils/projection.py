import os

import PIL.Image
import numpy as np
from scipy.interpolate import griddata

from . import check_args
from . import download
from . import maps
from . import misc


def trim(data, platform_lat, platform_lon, owi_lat, owi_lon):
    def arg2d(array, f=np.nanargmin):
        return np.unravel_index(f(array), array.shape)

    x1, y1 = arg2d(maps.get_distance(platform_lat, platform_lon, np.nanmax(owi_lat), np.nanmax(owi_lon)))
    x2, y2 = arg2d(maps.get_distance(platform_lat, platform_lon, np.nanmin(owi_lat), np.nanmin(owi_lon)))
    x3, y3 = arg2d(maps.get_distance(platform_lat, platform_lon, np.nanmax(owi_lat), np.nanmin(owi_lon)))
    x4, y4 = arg2d(maps.get_distance(platform_lat, platform_lon, np.nanmin(owi_lat), np.nanmax(owi_lon)))

    x1, x2 = min((x1, x2, x3, x4)), max((x1, x2, x3, x4))
    y1, y2 = min((y1, y2, y3, y4)), max((y1, y2, y3, y4))

    platform_lat = platform_lat[x1:x2, y1:y2]
    platform_lon = platform_lon[x1:x2, y1:y2]
    data = data[x1:x2, y1:y2]
    return data, platform_lat, platform_lon


def reproject(platform, data, platform_lat, platform_lon, owi_lat, owi_lon):
    if platform in check_args.SATELLITE_PLATFORMS['any'] or platform == 'ERA':
        data, platform_lat, platform_lon = trim(data, platform_lat, platform_lon, owi_lat, owi_lon)

    platform_lat[np.isnan(platform_lat)] = 0
    platform_lon[np.isnan(platform_lon)] = 0
    data[np.isnan(data)] = 0

    new_data = griddata(
        np.stack((platform_lat.flatten(), platform_lon.flatten()), axis=1),
        data.flatten(),
        np.stack((owi_lat.flatten(), owi_lon.flatten()), axis=1),
        method='nearest'
    ).reshape(owi_lat.shape).astype('float')
    return new_data


def save_reprojection(platform, channel, data, filename):
    kwargs = misc.platform_cmap_args(platform, channel)[0]
    vmin = kwargs.get("vmin", np.nanmin(data))
    vmax = kwargs.get("vmax", np.nanmax(data))

    new_data = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    new_data = kwargs["cmap"](new_data)
    new_data = (new_data * 255).astype(np.uint8)

    os.makedirs(os.path.split(filename)[0], exist_ok=True)
    PIL.Image.fromarray(new_data).save(filename + ".png")
    np.savez_compressed(filename + ".npz", data)


def increased_grid(polygon, km_per_pixel=1, delta_factor=1):
    lats, lons = misc.lat_lon_from_polygon(polygon)

    min_lat = np.min(lats)
    max_lat = np.max(lats)
    min_lon = np.min(lons)
    max_lon = np.max(lons)

    delta_lon = max_lon - min_lon
    delta_lat = max_lat - min_lat

    frame_min_lat = min_lat - delta_lat * delta_factor
    frame_max_lat = max_lat + delta_lat * delta_factor
    frame_min_lon = min_lon - delta_lon * delta_factor
    frame_max_lon = max_lon + delta_lon * delta_factor

    height = int(maps.get_distance(frame_min_lat, frame_min_lon, frame_max_lat, frame_min_lon) / km_per_pixel)
    width = int(max(
        maps.get_distance(frame_min_lat, frame_min_lon, frame_min_lat, frame_max_lon),
        maps.get_distance(frame_max_lat, frame_min_lon, frame_max_lat, frame_max_lon),
    ) / km_per_pixel)

    frame_polygon = np.array(
        [
            [frame_min_lon, frame_min_lat],
            [frame_min_lon, frame_max_lat],
            [frame_max_lon, frame_max_lat],
            [frame_max_lon, frame_min_lat]
        ]
    )

    return maps.grid_from_polygon(frame_polygon, (height, width))


def generate_gif(iw_polygon, channel, urls_per_platforms, gif_filename, verbose, read_function, download_asked=True,
                 delta_factor=None):
    def png_to_gif(input_filenames, output_filename):
        imgs = (PIL.Image.open(filename) for filename in input_filenames)
        img = next(imgs)

        os.makedirs(os.path.split(output_filename)[0], exist_ok=True)
        img.save(fp=output_filename, format='GIF', append_images=imgs, save_all=True, duration=200, loop=0)

    lat_grid, lon_grid = increased_grid(iw_polygon, km_per_pixel=2, delta_factor=delta_factor)
    filenames_per_platform = download.download_files(urls_per_platforms,
                                                     closest=False) if download_asked else urls_per_platforms
    m = None

    misc.log_print(f"Generate .png", 2, verbose)
    for platform in filenames_per_platform:
        png_filenames = []
        for date, filenames in filenames_per_platform[platform].items():
            platform_lat, platform_lon, data = read_function(filenames, platform, channel, requested_date=date)
            if platform in check_args.SATELLITE_PLATFORMS['any'] or platform == 'ERA5':
                data, platform_lat, platform_lon = trim(data, platform_lat, platform_lon, lat_grid, lon_grid)

            folder = os.path.split(filenames[0])[0]
            suptitle = date.strftime('%Y-%m-%d %H:%M:%S')
            datestr = date.strftime('%Y%m%dt%H%M%S')
            filename = folder + f"/{datestr}.{channel}.png"
            m = maps.plot_on_map(platform, channel, data, platform_lat, platform_lon, lat_grid, lon_grid, filename, m=m,
                                 polygon=iw_polygon, suptitle=suptitle)
            png_filenames.append(filename)

        png_to_gif(png_filenames, gif_filename)
