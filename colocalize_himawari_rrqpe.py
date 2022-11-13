import shutup; shutup.please()

from datetime import datetime, timedelta 
from matplotlib.patches import Polygon

from urllib import request
from xml.etree import ElementTree
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

from scipy.interpolate import griddata

shutil.rmtree('.temp', ignore_errors=True)
os.makedirs('.temp', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

matplotlib.use('agg')

from utils import get_distance, grid_from_polygon, getter_polygon_from_key, ini_map, png_to_gif
from utils import log_print, r_print, get_iw_latlon, plot_polygon
from requests_utils import routing, download_file

from goes_utils import increased_grid


def maps_from_iw_key(key, delta_minutes=90, verbose=1, shape=None, metadata_filename=None, sensoroperationalmode='IW', generate_gif=True):
    def get_himawari_rrqpe_urls():
        ts = [iw_datetime + timedelta(minutes=x) for x in range(-delta_minutes, delta_minutes+1, 10)]
        
        urls = {}
        for satellite in ('himawari8', 'himawari9'):
            url_basis = f"https://noaa-{satellite}.s3.amazonaws.com/?list-type=2&prefix=AHI-L2-FLDK-RainfallRate%2F"
            for t in ts:
                url = url_basis + f"{t.year}%2F{t.strftime('%m')}%2F{t.day}%2F{t.hour:02}{int(t.minute/10)}0"
                urls[satellite] = urls.get(satellite, []) + [url] 
        return urls

    def get_close_urls():
        maximum_delta = timedelta(minutes=delta_minutes)

        close_urls = {}
        closest_url = {}
        smallest_timedelta = {}
        for satellite, urls in urls_per_satellite.items():
            for url in urls:
                req = request.urlopen(url)
                tree = ElementTree.parse(req)

                for elem in tree.iter():
                    if elem.tag.endswith('Key'):
                        elem = elem.text
                        if not os.path.split(elem)[1].startswith('RRQPE'):continue
                        key_datetime = datetime.strptime(elem.split('_')[3][1:-1], '%Y%m%d%H%M%S')
                        current_timedelta = abs(key_datetime - iw_datetime)
                        
                        url = f"https://noaa-{satellite}.s3.amazonaws.com/{elem}"
                        if current_timedelta < maximum_delta:
                            close_urls[satellite] =  close_urls.get(satellite, []) + [url]
                        if satellite not in smallest_timedelta or current_timedelta < smallest_timedelta[satellite]:
                            closest_url[satellite] = url
                            smallest_timedelta[satellite] = current_timedelta
        satellite = sorted([(satellite, len(urls)) for satellite, urls in close_urls.items()])[-1][0]
        return close_urls[satellite], closest_url[satellite]

    def read_nc(filename):
        def arg2d(array, f=np.argmin):
            return  np.unravel_index(f(array), array.shape)
            
        with Dataset(filename) as dataset:
            lats = dataset['Latitude'][:]
            lons = dataset['Longitude'][:]
            data = dataset['RRQPE'][:]
            
        x1, y1 = arg2d(get_distance(lats, lons, lat_grid.max(), lon_grid.max()))
        x2, y2 = arg2d(get_distance(lats, lons, lat_grid.min(), lon_grid.min()))
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        lats = lats[x1:x2,y1:y2].filled(0)
        lons = lons[x1:x2,y1:y2].filled(0)
        data = data[x1:x2,y1:y2]
                
        return data, lats, lons


    def nc_to_png(filename, lat_grid, lon_grid, m=None, polygon=None):
        suptitle = datetime.strptime(filename.split('_')[-3][1:-1], '%Y%m%d%H%M%S').strftime('%Y-%m-%d %H:%M:%S')
        data, lats, lons = read_nc(filename)

        if m is None:
            plt.figure(figsize=(12,12))
            m = ini_map(lat_grid, lon_grid)
            
        plot_polygon(polygon, m)
        
        colormesh = m.pcolormesh(lons, lats, data, latlon=True, cmap='turbo', vmin=0, vmax=50, shading='auto')
        colorbar = plt.colorbar(fraction=0.046, pad=0.04, orientation='horizontal')
        plt.suptitle(suptitle)
        plt.title(f"AHI Rainfall Rate [mm/h]")
        plt.tight_layout()

        new_filename = filename + ".RRQPE.png"
        plt.savefig(new_filename)

        colorbar.remove()
        colormesh.remove()
        return new_filename, m

        
    def project_rrqpe_iw_lat_lon(filename, shape):
        data, lats, lons = read_nc(filename)

        new_data = griddata(
            np.stack((lats.flatten(), lons.flatten()), axis=1),
            data.flatten(),
            np.stack((owiLat.flatten(), owiLon.flatten()), axis=1)
        ).reshape(shape).astype('float')

        np.savez_compressed(f'outputs/{key}/{key}_rrqpe.npz', new_data)

        new_data = np.clip(new_data, 0, 40)/40
        new_data = plt.get_cmap('turbo')(new_data)
        new_data = (new_data * 255).astype(np.uint8)

        new_filename = f'outputs/{key}/{key}_rrqpe.png'
        PIL.Image.fromarray(new_data).save(new_filename)
        return new_filename
    
    start = datetime.now()
        
    if verbose: log_print("Loading IW metadata")
    getter = getter_polygon_from_key(sensoroperationalmode=sensoroperationalmode)
    if verbose: log_print("Loading OK")
    
    key = key.lower()
    iw_datetime = datetime.strptime(key, '%Y%m%dt%H%M%S')
    iw_filename, polygon, orbit = getter(key)
    
    if verbose: log_print('Retrieve urls')
    urls_per_satellite = get_himawari_rrqpe_urls()
    close_urls, closest_url = get_close_urls()
    download_file((closest_url, ".temp/"))
    lat_grid, lon_grid = increased_grid(polygon, km_per_pixel=2, delta_factor=2)

    os.makedirs('outputs/' + key, exist_ok=True)
    if verbose: log_print('Reproject in .npz and .png')
    download_file
    owiLat, owiLon = get_iw_latlon(polygon=polygon, metadata_filename=metadata_filename, shape=shape)
    if owiLat is not None:
        new_filename = project_rrqpe_iw_lat_lon(f".temp/{os.path.split(closest_url)[1]}", owiLat.shape)
        if verbose: log_print('Projection done')
    else: log_print('Projection failed')

    if generate_gif:
        routing_args = [(url, ".temp/") for url in close_urls]
        routing(routing_args)
        filenames = [f".temp/{os.path.split(url)[1]}" for url in close_urls]
        
        m = None
        for i, filename in enumerate(filenames):
            if verbose: log_print(f'Generating {i+1}/{len(filenames)} .png', f=r_print if i else print)
            filenames[i], m = nc_to_png(filename,  lat_grid, lon_grid, polygon=polygon, m=m)
        if verbose: print()
        
        if verbose: log_print('Generating .gif')
        png_to_gif(filenames, f'outputs/{key}/{key}_rrqpe.gif')
    
    if verbose: print('Executed in', (datetime(1,1,1) + (datetime.now()-start)).strftime('%H hours, %M minutes, %S seconds'))

if __name__ == "__main__":
    fire.Fire(maps_from_iw_key)
