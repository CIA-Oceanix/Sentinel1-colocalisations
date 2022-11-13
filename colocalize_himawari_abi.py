import shutup; shutup.please()

from datetime import datetime, timedelta 
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
np.seterr(all="ignore")


import os
import PIL.Image
import sys
import fire
from netCDF4 import Dataset
import shutil
from xml.etree import ElementTree
from urllib import request

from scipy.interpolate import griddata

shutil.rmtree('.temp', ignore_errors=True)
os.makedirs('.temp', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

matplotlib.use('agg')

from utils import get_distance, dms2dd, grid_from_polygon, getter_polygon_from_key, ini_map, png_to_gif
from utils import log_print, r_print, get_iw_latlon, plot_polygon
from requests_utils import routing, download_file

from goes_utils import get_goes_hour_urls, get_close_urls, increased_grid
from cpt import loadCPT
cpt_cmap = LinearSegmentedColormap('cpt', loadCPT('IR4AVHRR6.cpt'))

SATELLITES = ('himawari8', 'himawari9')
CHANNELS = ("C13", "C14")


def list_bucket_urls(iw_datetime, delta_minutes, verbose=1):
    if verbose: log_print(f'Generate bucket urls')

    ts = [iw_datetime + timedelta(minutes=x) for x in range(-delta_minutes, delta_minutes+1, 10)]
    
    urls = {}
    for satellite in SATELLITES:
        url_basis = f"https://noaa-{satellite}.s3.amazonaws.com/?prefix=AHI-L2-FLDK-ISatSS"
        for t in ts:
            url = url_basis + f"/{t.year}/{t.strftime('%m')}/{t.day}/{t.hour:02}{int(t.minute/10)}0"
            urls[satellite] = urls.get(satellite, []) + [url] 
    return urls


def bucket_url_to_file_url(iw_datetime, bucket_urls):
    closest_url_index = {}
    file_per_satellite = [0 for satellite in SATELLITES]
    
    smallest_timedelta = {}
    
    file_urls = {}
    for i_satellite, (satellite, urls) in enumerate(bucket_urls.items()):
        file_urls[satellite] = [[[] for url in urls] for channel in CHANNELS]
        for i_url, url in enumerate(urls):

            for i_channel, channel in enumerate(CHANNELS):
                req = request.urlopen(url + "/OR_HFD-020-B12-M1" + channel)
                tree = ElementTree.parse(req)
                for elem in tree.iter():
                    if elem.tag.endswith('Key'):
                        elem = elem.text
                        key_datetime = datetime.strptime(elem.split('_')[-2][1:-1], '%Y%j%H%M%S')
                        current_timedelta = abs(key_datetime - iw_datetime)

                        file_urls[satellite][i_channel][i_url].append(f"https://noaa-{satellite}.s3.amazonaws.com/{elem}")
                        file_per_satellite[i_satellite] += 1
                            
                        if satellite not in smallest_timedelta or current_timedelta < smallest_timedelta[satellite]:
                            closest_url_index[satellite] = i_url
                            smallest_timedelta[satellite] = current_timedelta
                            
    closest_satellite = SATELLITES[np.argmax(file_per_satellite)]
    return closest_satellite, file_urls[closest_satellite], closest_url_index[closest_satellite]


def latlon_from_goes_dataset(x_coordinate_1d, y_coordinate_1d, projection_info):
    x_coordinate_1d = x_coordinate_1d / 10**6
    y_coordinate_1d = y_coordinate_1d / 10**6

    lon_origin = projection_info.longitude_of_projection_origin
    H = projection_info.perspective_point_height+projection_info.semi_major
    r_eq = projection_info.semi_major
    r_pol = projection_info.semi_minor

    x_coordinate_2d, y_coordinate_2d = np.meshgrid(x_coordinate_1d, y_coordinate_1d)

    lambda_0 = (lon_origin*np.pi)/180.0  
    a_var = np.power(np.sin(x_coordinate_2d),2.0) + (np.power(np.cos(x_coordinate_2d),2.0)*(np.power(np.cos(y_coordinate_2d),2.0)+(((r_eq*r_eq)/(r_pol*r_pol))*np.power(np.sin(y_coordinate_2d),2.0))))
    b_var = -2.0*H*np.cos(x_coordinate_2d)*np.cos(y_coordinate_2d)
    c_var = (H**2.0)-(r_eq**2.0)
    r_s = (-1.0*b_var - np.sqrt((b_var**2)-(4.0*a_var*c_var)))/(2.0*a_var)
    s_x = r_s*np.cos(x_coordinate_2d)*np.cos(y_coordinate_2d)
    s_y = - r_s*np.sin(x_coordinate_2d)
    s_z = r_s*np.cos(x_coordinate_2d)*np.sin(y_coordinate_2d)

    abi_lat = (180.0/np.pi)*(np.arctan(((r_eq*r_eq)/(r_pol*r_pol))*((s_z/np.sqrt(((H-s_x)*(H-s_x))+(s_y*s_y))))))
    abi_lon = (lambda_0 - np.arctan(s_y/(H-s_x)))*(180.0/np.pi)
    return abi_lat, abi_lon


def get_abi_channels(folder, lat_min, lat_max, lon_min, lon_max):
    def arg2d(array, f=np.argmin):
        return  np.unravel_index(f(array), array.shape)
        
    filenames = [
        os.path.join(folder, filename)
        for filename in os.listdir(folder)
        if filename.endswith('.nc')
    ]

    n = 550
    array = np.zeros((10*n,10*n,3))
    for i, filename in enumerate(filenames):
        with Dataset(filename) as dataset:
            lat, lon = latlon_from_goes_dataset(dataset['x'][:], dataset['y'][:], dataset['fixedgrid_projection'])
            data = dataset['Sectorized_CMI'][:]

        cell_pad = 0
        if i >= 0: cell_pad += 2
        if i >= 6: cell_pad += 3
        if i >= 14: cell_pad += 1
        if i >= 74: cell_pad += 1
        if i >= 82: cell_pad += 3

        y = (i+cell_pad)%10
        x = (i+cell_pad)//10
        
        array[n*x:n*(x+1), n*y:n*(y+1), 0] = lat
        array[n*x:n*(x+1), n*y:n*(y+1), 1] = lon
        array[n*x:n*(x+1), n*y:n*(y+1), 2] = data
        
    x1, y1 = arg2d(get_distance(array[:,:,0], array[:,:,1], lat_max, lon_max))
    x2, y2 = arg2d(get_distance(array[:,:,0], array[:,:,1], lat_min, lon_min))
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    lats = array[x1:x2,y1:y2,0]
    lons = array[x1:x2,y1:y2,1]
    data = array[x1:x2,y1:y2,2]
    return data, lats, lons

def project_abi_on_iw_lat_lon(key, channel, folder, shape, owiLat, owiLon):
    data, lats, lons = get_abi_channels(folder, owiLat.min(), owiLat.max(), owiLon.min(), owiLon.max())

    new_data = griddata(
        np.stack((lats.flatten(), lons.flatten()), axis=1),
        data.flatten(),
        np.stack((owiLat.flatten(), owiLon.flatten()), axis=1)
    ).reshape(shape).astype('float')

    np.savez_compressed(f'outputs/{key}/{key}_{channel}.npz', new_data)

    new_data = np.clip((new_data-170)/208, 0, 1)
    new_data = cpt_cmap(new_data)
    new_data = (new_data * 255).astype(np.uint8)

    new_filename = f'outputs/{key}/{key}_{channel}.png'
    PIL.Image.fromarray(new_data).save(new_filename)

def abi_nc_to_png(channel, folder, lat_grid, lon_grid, m=None, polygon=None):
    filename = os.listdir(folder)[0]
    suptitle = datetime.strptime(filename.split('_')[-2][1:-1], '%Y%j%H%M%S').strftime('%Y-%m-%d %H:%M:%S')

    if m is None:
        plt.figure(figsize=(12,12))
        m = ini_map(lat_grid, lon_grid)

    new_filenames = {}
    
    data, lats, lons = get_abi_channels(folder, lat_grid.min(), lat_grid.max(), lon_grid.min(), lon_grid.max())

    colormesh = m.pcolormesh(lons, lats, data, latlon=True, cmap=cpt_cmap, vmin=170, vmax=378, shading='auto')
    colorbar = plt.colorbar(fraction=0.046, pad=0.04, orientation='horizontal')

    plot_polygon(polygon, m)

    plt.suptitle(suptitle)
    plt.title(channel)
    plt.tight_layout()

    new_filename = os.path.join(folder, filename + f".{channel}.png")
    plt.savefig(new_filename)

    colorbar.remove()
    colormesh.remove()

    return new_filename, m

def maps_from_iw_key(key, delta_minutes=90, verbose=1, shape=None, metadata_filename=None, sensoroperationalmode='IW', generate_gif=True):

    start = datetime.now()
    
    if verbose: log_print("Loading IW metadata")
    getter = getter_polygon_from_key(sensoroperationalmode=sensoroperationalmode)
    if verbose: log_print("Loading OK")

    key = key.lower()
    iw_datetime = datetime.strptime(key, '%Y%m%dt%H%M%S')
    iw_filename, polygon, orbit = getter(key)

    if verbose: log_print("Choose satellite")
    bucket_urls = list_bucket_urls(iw_datetime, delta_minutes=delta_minutes)
    closest_satellite, file_urls, closest_url_index = bucket_url_to_file_url(iw_datetime, bucket_urls)
    if verbose: log_print(f"Closest satellite: {closest_satellite}")

    # -----------------------------------------------------------------
    if verbose: log_print('Project on IW grid to generate .npz and .png')
    if verbose: log_print('Download .nc for projection')
    routing_args = [
        (url, f".temp/{channel}/{closest_url_index}") 
        for (i_channel, channel) in enumerate(CHANNELS) 
        for url in file_urls[i_channel][closest_url_index]
    ]
    routing(routing_args)

    os.makedirs('outputs/' + key, exist_ok=True)
    owiLat, owiLon = get_iw_latlon(polygon=polygon, metadata_filename=metadata_filename, shape=shape)
    if owiLat is not None:
        for (i_channel, channel) in enumerate(CHANNELS):
            if verbose: log_print(f'Project channel: {channel}')
            folder = f".temp/{channel}/{closest_url_index}"
            project_abi_on_iw_lat_lon(key, channel, folder, owiLat.shape, owiLat, owiLon)
        if verbose: log_print('Projection done')
    else: log_print('Projection failed')
    # -----------------------------------------------------------------

    if generate_gif:
        if verbose: log_print('Generate .gif')
        if verbose: log_print('Download .nc for .gif')
        routing_args = [
            (url, f".temp/{channel}/{i_urls}") 
            for (i_channel, channel) in enumerate(CHANNELS) 
            for i_urls, urls in enumerate(file_urls[i_channel])
            for url in urls
        ]
        routing(routing_args)

        lat_grid, lon_grid = increased_grid(polygon, km_per_pixel=2, delta_factor=2)

        m = None
        for (i_channel, channel) in enumerate(CHANNELS):
            if verbose: log_print(f'Channel: {channel}')
            png_filenames = []
            n = len(file_urls[i_channel])
            for url_index in range(n):
                if verbose: log_print(f'Generate {url_index+1}/{n} .png', f=r_print if url_index else print)
                folder = f".temp/{channel}/{url_index}"
                png_filename, m = abi_nc_to_png(channel, folder,  lat_grid, lon_grid, polygon=polygon, m=m)
                png_filenames.append(png_filename)
                
            png_to_gif(png_filenames, f'outputs/{key}/{key}_{channel}.gif')

    if verbose: print('Executed in', (datetime(1,1,1) + (datetime.now()-start)).strftime('%H hours, %M minutes, %S seconds'))

    
if __name__ == "__main__":
    fire.Fire(maps_from_iw_key)
