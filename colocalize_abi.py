import shutup; shutup.please()


import os
import fire
import PIL.Image
import numpy as np
from urllib import request
from netCDF4 import Dataset
from functools import lru_cache
from xml.etree import ElementTree
from scipy.interpolate import griddata
from datetime import datetime, timedelta

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap 
matplotlib.use('agg')

from cpt import loadCPT
cpt_cmap = LinearSegmentedColormap('cpt', loadCPT('IR4AVHRR6.cpt'))

from utils import get_distance, getter_polygon_from_key, ini_map, png_to_gif, log_print, get_iw_latlon, plot_polygon
from goes_utils import increased_grid
from requests_utils import routing

GOES_SERIE = ["goes16", "goes17", "goes18"]
HIMAWARI_SERIE = ["himawari8", "himawari9"]
SATELLITES = GOES_SERIE + HIMAWARI_SERIE

def get_bucket_url(satellite, channel, date):
    def himawari_bucket_date(date):
        return f"AHI-L2-FLDK-ISatSS/{date.year}/{date.strftime('%m')}/{date.day}/{date.hour:02}{int(date.minute/10)}0/OR_HFD-020-B12-M1{channel}"

    def goes_bucket_date(date):
        return f"ABI-L2-MCMIPF/{date.year}/{date.strftime('%j')}/{date.hour:02}"
        
    url = f"https://noaa-{satellite}.s3.amazonaws.com/?prefix="
    if satellite in HIMAWARI_SERIE:
        url += himawari_bucket_date(date)
    elif satellite in GOES_SERIE:
        url += goes_bucket_date(date)
    return url


def get_bucket_urls(channel, iw_datetime, max_delta_minutes=10, minutes_delta=10, satellites=SATELLITES):
    minutes_deltas = range(-max_delta_minutes, max_delta_minutes+1, minutes_delta)
    dates = [iw_datetime + timedelta(minutes=x) for x in minutes_deltas]
    
    urls = {}
    for satellite in satellites:
        urls[satellite] = {}
        for date in dates:
            urls[satellite][date] = get_bucket_url(satellite, channel, date)
    return urls

@lru_cache(maxsize=2**16)
def bucket_to_urls(bucket_url):
    urls = []
    req = request.urlopen(bucket_url)
    tree = ElementTree.parse(req)
    for elem in tree.iter():
        if elem.tag.endswith('Key'):
            urls.append(elem.text)
    return urls


def get_nc_urls(channel, iw_datetime, bucket_urls_per_satellite):
    urls_per_satellite = {}
    closest_datetime = {}
    for satellite in bucket_urls_per_satellite:
        urls_per_satellite[satellite] = {}
        url_base = f"https://noaa-{satellite}.s3.amazonaws.com/"

        for date, bucket_url in bucket_urls_per_satellite[satellite].items():
            urls = bucket_to_urls(bucket_url)

            closest_urls = {}
            for url in urls:
                date_string = url.split('_')[-2][1:-1]
                url_datetime = datetime.strptime(date_string, '%Y%j%H%M%S')

                current_timedelta = abs(url_datetime - date)
                if (not closest_urls) or smallest_timedelta >= current_timedelta:
                    closest_urls[current_timedelta] = closest_urls.get(current_timedelta, []) + [url]
                    smallest_timedelta = current_timedelta
            if closest_urls:
                urls_per_satellite[satellite][date] = [url_base + url for url in closest_urls[smallest_timedelta]]
                
    urls_per_satellite = {
        key: value
        for key, value in urls_per_satellite.items()
        if value
    }
    return urls_per_satellite


def download_files(urls_per_satellites, closest=False):
    # Assume the closest_file is in the middle of the dic

    routing_args = []
    filenames = {}
    for satellite in urls_per_satellites:
        dates = list(urls_per_satellites[satellite].keys())
        if closest:
            dates = [dates[int(len(dates)/2)]]

        filenames[satellite] = []
        for date in dates:
            filenames[satellite].append([])
            folder = f".temp/{satellite}/{date.strftime('%Y%m%dt%H%M%S')}/"
            for url in urls_per_satellites[satellite][date]:
                filenames[satellite][-1].append(folder + os.path.split(url)[1])
                routing_args.append((url, folder))

    routing(routing_args)
    return filenames
            
def read_from_ncs_per_satellite(filenames, satellite, channel):
    def latlon_from_nc_per_satellite(filename, satellite, channel):
        def latlon_from_nc(x, y, lon_origin, H, r_eq, r_pol):
            x, y = np.meshgrid(x, y)

            lambda_0 = (lon_origin*np.pi)/180.0  
            a_var = np.power(np.sin(x),2.0) + (np.power(np.cos(x),2.0)*(np.power(np.cos(y),2.0)+(((r_eq*r_eq)/(r_pol*r_pol))*np.power(np.sin(y),2.0))))
            b_var = -2.0*H*np.cos(x)*np.cos(y)
            c_var = (H**2.0)-(r_eq**2.0)
            r_s = (-1.0*b_var - np.sqrt((b_var**2)-(4.0*a_var*c_var)))/(2.0*a_var)
            s_x = r_s*np.cos(x)*np.cos(y)
            s_y = - r_s*np.sin(x)
            s_z = r_s*np.cos(x)*np.sin(y)

            abi_lat = (180.0/np.pi)*(np.arctan(((r_eq*r_eq)/(r_pol*r_pol))*((s_z/np.sqrt(((H-s_x)*(H-s_x))+(s_y*s_y))))))
            abi_lon = (lambda_0 - np.arctan(s_y/(H-s_x)))*(180.0/np.pi)
            return abi_lat, abi_lon

        with Dataset(filename) as dataset:
            if satellite in GOES_SERIE:
                projection_info = dataset['goes_imager_projection']
                abi_lat, abi_lon = latlon_from_nc(
                    dataset['x'][:],
                    dataset['y'][:],
                    projection_info.longitude_of_projection_origin,
                    projection_info.perspective_point_height+projection_info.semi_major_axis,
                    projection_info.semi_major_axis,
                    projection_info.semi_minor_axis
                )
                data = dataset["CMI_" + channel][:]
                
            elif satellite in HIMAWARI_SERIE:
                projection_info = dataset['fixedgrid_projection']
                abi_lat, abi_lon = latlon_from_nc(
                    dataset['x'][:] / 10**6,
                    dataset['y'][:] / 10**6,
                    projection_info.longitude_of_projection_origin,
                    projection_info.perspective_point_height+projection_info.semi_major,
                    projection_info.semi_major,
                    projection_info.semi_minor
                )
                data = dataset['Sectorized_CMI'][:]
        return abi_lat, abi_lon, data

    if satellite in GOES_SERIE:
        abi_lat, abi_lon, data = latlon_from_nc_per_satellite(filenames[0], satellite, channel)
        abi_lat = abi_lat.filled(0)
        abi_lon = abi_lon.filled(0)
        data = data.filled(0)

    elif satellite in HIMAWARI_SERIE:
        n = 550  # size of each tile

        abi_lat = np.zeros((10*n,10*n))
        abi_lon = np.zeros((10*n,10*n))
        data = np.zeros((10*n,10*n))
        for i, filename in enumerate(filenames):
            cell_pad = 0
            if i >= 0: cell_pad += 2
            if i >= 6: cell_pad += 3
            if i >= 14: cell_pad += 1
            if i >= 74: cell_pad += 1
            if i >= 82: cell_pad += 3

            tile_lat, tile_lon, tile_data = latlon_from_nc_per_satellite(filename, satellite, channel)
            y = (i+cell_pad)%10
            x = (i+cell_pad)//10
            
            abi_lat[n*x:n*(x+1), n*y:n*(y+1)] = tile_lat
            abi_lon[n*x:n*(x+1), n*y:n*(y+1)] = tile_lon
            data[n*x:n*(x+1), n*y:n*(y+1)] = tile_data

    return abi_lat, abi_lon, data


def get_closest_satellite(closest_filenames_per_satellite, iw_polygon, channel):
    mean_iw_lat = np.mean(iw_polygon[:,1])
    mean_iw_lon = np.mean(iw_polygon[:,0])
            
    closest_satellite = None

    res = {}
    for satellite, filenames in closest_filenames_per_satellite.items():
        abi_lat, abi_lon, data = read_from_ncs_per_satellite(filenames, satellite, channel)
        res[satellite] = abi_lat, abi_lon, data
            
        mean_satellite_lat = np.mean(abi_lat)
        mean_satellite_lon = np.mean(abi_lon)
            
        distance_to_center = get_distance(mean_iw_lat, mean_iw_lon, mean_satellite_lat, mean_satellite_lon)
        if closest_satellite is None or distance_to_center < smallest_distance:
            smallest_distance = distance_to_center
            closest_satellite = satellite
    return closest_satellite, res[closest_satellite]


def trim(data, abi_lat, abi_lon, owi_lat, owi_lon):
    def arg2d(array, f=np.argmin):
        return  np.unravel_index(f(array), array.shape)
    # Trim useless data to accelerate griddata
    x1, y1 = arg2d(get_distance(abi_lat, abi_lon, owi_lat.max(), owi_lon.max()))
    x2, y2 = arg2d(get_distance(abi_lat, abi_lon, owi_lat.min(), owi_lon.min()))
    
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    abi_lat = abi_lat[x1:x2,y1:y2]
    abi_lon = abi_lon[x1:x2,y1:y2]
    data = data[x1:x2,y1:y2]
    return data, abi_lat, abi_lon


def reproject(data, abi_lat, abi_lon, owi_lat, owi_lon):
    data, abi_lat, abi_lon = trim(data, abi_lat, abi_lon, owi_lat, owi_lon)
    
    new_data = griddata(
        np.stack((abi_lat.flatten(), abi_lon.flatten()), axis=1),
        data.flatten(),
        np.stack((owi_lat.flatten(), owi_lon.flatten()), axis=1)
    ).reshape(owi_lat.shape).astype('float')
    return data

def save_reprojection(data, filename):
    new_data = np.clip((data-170)/208, 0, 1)
    new_data = cpt_cmap(new_data)
    new_data = (new_data * 255).astype(np.uint8)

    os.makedirs(os.path.split(filename)[0], exist_ok=True)
    PIL.Image.fromarray(new_data).save(filename + ".png")
    np.savez_compressed(filename + ".npz", data)


def plot_on_map(data, abi_lat, abi_lon, lat_grid, lon_grid, filename, m=None, polygon=None, suptitle=None):
    if m is None:
        plt.figure(figsize=(12,12))
        m = ini_map(lat_grid, lon_grid)
        
    colormesh = m.pcolormesh(abi_lon, abi_lat, data, latlon=True, cmap=cpt_cmap, vmin=170, vmax=378, shading='auto')
    colorbar = plt.colorbar(fraction=0.046, pad=0.04, orientation='horizontal')

    plot_polygon(polygon, m)

    if suptitle is not None:
        plt.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(filename)

    colorbar.remove()
    colormesh.remove()
    return m

    
def generate_gif(iw_polygon, closest_satellite, channel, urls_per_satellites, gif_filename):
    lat_grid, lon_grid = increased_grid(iw_polygon, km_per_pixel=2, delta_factor=2)
    
    filenames_per_satellite = download_files(urls_per_satellites, closest=False)

    png_filenames = []
    m = None
    for filenames in filenames_per_satellite[closest_satellite]:
        abi_lat, abi_lon, data = read_from_ncs_per_satellite(filenames, closest_satellite, channel)
        data, abi_lat, abi_lon = trim(data, abi_lat, abi_lon, lat_grid, lon_grid)

        filename = os.path.split(filenames[0])[0]
        suptitle = datetime.strptime(os.path.split(filename)[1], '%Y%m%dt%H%M%S').strftime('%Y-%m-%d %H:%M:%S')
        m = plot_on_map(data, abi_lat, abi_lon, lat_grid, lon_grid, filename + f".{channel}.png", m=m, polygon=iw_polygon, suptitle=suptitle)
        png_filenames.append(filename)

    png_to_gif(filenames, gif_filename)
    
    
def main(key="20210913t092920", channel="C14", shape=None, metadata_filename=None, verbose=1, sensoroperationalmode="IW", satellites = SATELLITES, gif=True):
    if verbose: log_print(f"Build {sensoroperationalmode} getter")
    getter = getter_polygon_from_key()

    iw_datetime = datetime.strptime(key, '%Y%m%dt%H%M%S')
    iw_filename, iw_polygon = getter(key)[:2]

    if verbose: log_print("Retrieve .nc urls")
    bucket_urls_per_satellite = get_bucket_urls(channel, iw_datetime, satellites=satellites)
    urls_per_satellites = get_nc_urls(channel, iw_datetime, bucket_urls_per_satellite)

    if verbose: log_print("Download closest file to determine closest satellite")
    closest_filenames_per_satellite = download_files(urls_per_satellites, closest=True)
    closest_filenames_per_satellite = {key: value[0] for key, value in closest_filenames_per_satellite.items()}

    closest_satellite, (abi_lat, abi_lon, data) = get_closest_satellite(closest_filenames_per_satellite, iw_polygon, channel)
    urls_per_satellites = {key: value for key, value in urls_per_satellites.items() if key == closest_satellite}

    owi_lat, owi_lon = get_iw_latlon(polygon=iw_polygon, metadata_filename=metadata_filename, shape=shape)

    if verbose: log_print("Project on S1 lat/lon grid")
    data = reproject(data, abi_lat, abi_lon, owi_lat, owi_lon)
    save_reprojection(data, f'outputs/{key}/{key}_{channel}.npz')

    if gif:
        if verbose: log_print("Generate .gif")
        generate_gif(iw_polygon, closest_satellite, channel, urls_per_satellites, f'outputs/{key}/{key}_{channel}.gif')
    
    
if __name__ == "__main__":
    fire.Fire(main)

