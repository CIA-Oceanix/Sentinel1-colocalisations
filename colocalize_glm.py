import shutup; shutup.please()

from datetime import datetime, timedelta 
from matplotlib.patches import Polygon
from urllib import request
from xml.etree import ElementTree
import cv2
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL.Image
import sys
import fire
from netCDF4 import Dataset
import shutil


shutil.rmtree('.temp', ignore_errors=True)
os.makedirs('.temp', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

matplotlib.use('agg')

from utils import get_distance, dms2dd, grid_from_polygon, getter_polygon_from_key, ini_map, png_to_gif
from utils import log_print, r_print
from requests_utils import routing, download_file

def get_lightning_map(min_lat, max_lat, min_lon, max_lon, filenames):
    height = int(get_distance(min_lat, min_lon, max_lat, min_lon)/10)
    width =  int(max(
        get_distance(min_lat, min_lon, min_lat, max_lon),
        get_distance(max_lat, min_lon, max_lat, max_lon),
    )/10)
    
    polygon = np.array(
        [
            [min_lon, min_lat],
            [min_lon, max_lat],
            [max_lon, max_lat],
            [max_lon, min_lat]
        ]
    )
    
    lat_grid, lon_grid = grid_from_polygon(polygon, (height, width))
    data = np.zeros(lat_grid.shape)
    
    for filename in filenames:
        with Dataset(filename) as dataset:
            event_lat = dataset['event_lat'][:]
            event_lon = dataset['event_lon'][:]
            event_energy = dataset['event_energy']
            event_energy = (event_energy[:]*event_energy.scale_factor) + event_energy.add_offset
            
        validity = np.logical_and(
            np.logical_and(min_lat < event_lat, event_lat < max_lat),
            np.logical_and(min_lon < event_lon, event_lon < max_lon)
        )

        event_lat = event_lat[validity]
        event_lon = event_lon[validity]
        event_energy = event_energy[validity]
        
        event_lat = ((event_lat - min_lat)/(max_lat - min_lat)*data.shape[1]).astype(int)
        event_lon = ((event_lon - min_lon)/(max_lon - min_lon)*data.shape[0]).astype(int)
        
        np.add.at(data, (event_lon, event_lat), event_energy)
    data[data==0] = np.nan
    return data, lat_grid, lon_grid
    

def maps_from_iw_key(key, delta_minutes=90, verbose=1, shape=None, metadata_filename=None, sensoroperationalmode='IW'):
    def get_glm_hour_urls():
        if verbose: log_print(f'Retrieving urls')

        ts = [iw_datetime + timedelta(minutes=x) for x in range(-delta_minutes, delta_minutes+1, 60)]
        
        urls = {}
        for satellite in ('goes16', 'goes17', 'goes18'):
            url_basis = f"https://noaa-{satellite}.s3.amazonaws.com/?list-type=2"

            for t in ts:
                url = url_basis + f"&prefix=GLM-L2-LCFA%2F{t.year}%2F{t.strftime('%j')}%2F{t.hour:02}"
                urls[satellite] = urls.get(satellite, []) + [url] 
        return urls

    def get_close_urls():
        maximum_delta = timedelta(minutes=delta_minutes)

        close_urls = {}
        closest_url = {}
        smallest_timedelta = {}
        for satellite, hour_urls in hour_urls_per_satellite.items():
            for hour_url in hour_urls:
                req = request.urlopen(hour_url)
                tree = ElementTree.parse(req)

                for elem in tree.iter():
                    if elem.tag.endswith('Key'):
                        key = elem.text
                        key_datetime = datetime.strptime(elem.text.split('_')[3][1:-1], '%Y%j%H%M%S')
                        current_timedelta = abs(key_datetime - iw_datetime)
                        
                        url = f"https://noaa-{satellite}.s3.amazonaws.com/{elem.text}"
                        if current_timedelta < maximum_delta:
                            close_urls[satellite] =  close_urls.get(satellite, []) + [url]
                        if satellite not in smallest_timedelta or current_timedelta < smallest_timedelta[satellite]:
                            closest_url[satellite] = url
                            smallest_timedelta[satellite] = current_timedelta
        return close_urls, closest_url

    def get_closest_satellite():
        mean_iw_lat = np.mean(polygon[:,1])
        mean_iw_lon = np.mean(polygon[:,0])
                
        closest_satellite = None
        for satellite, url in closest_url.items():
            folder = ".temp/" + satellite
            os.makedirs(folder, exist_ok=True)
            download_file(url, folder)
            
            with Dataset(os.path.join(folder, os.listdir(folder)[0])) as dataset:
                mean_satellite_lat = np.mean(dataset['lat_field_of_view_bounds'][:])
                mean_satellite_lon = np.mean(dataset['lat_field_of_view_bounds'][:])
                
            distance_to_center = get_distance(mean_iw_lat, mean_iw_lon, mean_satellite_lat, mean_satellite_lon)
            if closest_satellite is None or distance_to_center < smallest_distance:
                smallest_distance = distance_to_center
                closest_satellite = satellite
        return closest_satellite

    def get_lightning_maps(stride=15):
        min_lat = np.min(polygon[:,1])
        max_lat = np.max(polygon[:,1])

        min_lon = np.min(polygon[:,0])
        max_lon = np.max(polygon[:,0])

        delta_lon = max_lon - min_lon
        delta_lat = max_lat - min_lat

        frame_min_lat = min_lat - delta_lat*2
        frame_max_lat = max_lat + delta_lat*2
        frame_min_lon = min_lon - delta_lon*2
        frame_max_lon = max_lon + delta_lon*2
        
        lightning_maps = []
        new_filenames = []
        for i in range(len(filenames)//stride):
            current_filenames = filenames[i*stride:(i+1)*stride]
            
            lightning_map, lat_grid, lon_grid = get_lightning_map(frame_min_lat, frame_max_lat, frame_min_lon, frame_max_lon, current_filenames)
            lightning_maps.append(lightning_map)
            new_filenames.append(current_filenames[len(current_filenames)//2])
        return lightning_maps, new_filenames, lat_grid, lon_grid

    def lightning_map_to_png():
        colormesh = m.pcolormesh(lon_grid, lat_grid, lightning_map, latlon=True, cmap="turbo", vmin=0, vmax=vmax, shading='auto')
        colorbar = plt.colorbar(fraction=0.046, pad=0.04, orientation='horizontal')
        
        if polygon is not None:
            x, y = m(polygon[:,0], polygon[:,1])
            plt.plot(x, y, color="black", linestyle='--')
            plt.plot(x[[0,-1]], y[[0,-1]], color="black", linestyle='--')
        
        plt.suptitle(datetime.strptime(new_filename.split('_')[-3][1:-1], '%Y%j%H%M%S').strftime('%Y-%m-%d %H:%M:%S'))
        plt.title('Event accumulated energy [J]')
        plt.tight_layout()
        
        plt.savefig(new_filename + ".png")

        colorbar.remove()
        colormesh.remove()
    
    if verbose: log_print("Loading IW metadata")
    getter = getter_polygon_from_key(sensoroperationalmode=sensoroperationalmode)

    key = key.lower()
    iw_datetime = datetime.strptime(key, '%Y%m%dt%H%M%S')
    iw_filename, polygon, orbit = getter(key)

    hour_urls_per_satellite = get_glm_hour_urls()
    close_urls, closest_url = get_close_urls()
    closest_satellite = get_closest_satellite()

    routing_args = [(url, ".temp/" + closest_satellite) for url in close_urls[closest_satellite]]
    routing(routing_args)
    filenames = [f".temp/{closest_satellite}/{os.path.split(url)[1]}" for url in close_urls[closest_satellite]]

    lightning_maps, new_filenames, lat_grid, lon_grid = get_lightning_maps()
    vmax = max([np.nanmax(lightning_map) for lightning_map in lightning_maps])

    if verbose: log_print('Initialize map')
    plt.figure(figsize=(12,12))
    m = ini_map(lat_grid, lon_grid)

    if verbose: log_print(f'Generating .png')
    for new_filename, lightning_map in zip(new_filenames, lightning_maps):
        lightning_map_to_png()

    if verbose: log_print('Generating .gif')
    gif_filename = f'outputs/{key}/{key}_GLM.gif'
    os.makedirs(os.path.split(gif_filename)[0], exist_ok=True)
    png_to_gif([filename+".png" for filename in new_filenames], gif_filename)
    
if __name__ == "__main__":
    fire.Fire(maps_from_iw_key)
