# python colocalize_nexrad.py 20170108t015819

import shutup; shutup.please()

from datetime import datetime, timedelta 
from matplotlib.patches import Polygon
from urllib import request
from xml.etree import ElementTree
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL.Image
import pyart
import sys
import fire
import shutil

from scipy.interpolate import griddata

shutil.rmtree('.temp', ignore_errors=True)
os.makedirs('.temp', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

matplotlib.use('agg')

from utils import get_distance, dms2dd, grid_from_polygon, getter_polygon_from_key, ini_map, png_to_gif
from utils import log_print, r_print, get_iw_latlon

def get_nexrad_stations():
    nexrad_stations = {}
    with open('res/nexrad_stations.txt', 'r') as file:
        for line in file.readlines()[1:]:
            line = line.split('\t')
                
            station_id = line[1]
            lat, lon = line[3].split('/')
            
            lat = dms2dd(lat[:2], lat[2:4], lat[4:6], 'N')
            lon = dms2dd(lon[1:4], lon[4:6], lon[6:8], 'W')
                    
            nexrad_stations[station_id] = {"lat": lat, "lon": lon}
    return nexrad_stations

def ar2v_to_png(filename, m=None, polygon=None, field_name='reflectivity'):
    radar = pyart.io.read_nexrad_archive(filename)
    lats, lons, alts = radar.get_gate_lat_lon_alt(sweep=0)
    data = radar.get_field(sweep=0, field_name='reflectivity')


    if m is None:
        plt.figure(figsize=(12,12))
        m = ini_map(lats, lons)
        
    colormesh = m.pcolormesh(lons, lats, data, latlon=True, cmap="turbo", vmin=0, vmax=40, shading='auto')
    colorbar = plt.colorbar(fraction=0.046, pad=0.04, orientation='horizontal')
    
    if polygon is not None:
        x, y = m(polygon[:,0], polygon[:,1])
        plt.plot(x, y, color="black", linestyle='--')
        plt.plot(x[[0,-1]], y[[0,-1]], color="black", linestyle='--')
    
    plt.suptitle(filename)
    plt.title(field_name)
    plt.tight_layout()
    
    new_filename = filename + ".png"
    plt.savefig(new_filename)

    colorbar.remove()
    colormesh.remove()
    
    return new_filename, m

def maps_from_iw_key(key, delta_minutes=30, verbose=1, shape=None, metadata_filename=None, sensoroperationalmode='IW'):
    def get_closest_station():
        nexrad_stations = get_nexrad_stations()
        
        mean_iw_lat = np.mean(polygon[:,1])
        mean_iw_lon = np.mean(polygon[:,0])
        closest_station_distance = np.inf
        for station, latlon in nexrad_stations.items(): # could be paralelized with np, but too lazy
            station_distance = get_distance(mean_iw_lat, mean_iw_lon, latlon['lat'], latlon['lon'])
            if station_distance < closest_station_distance:
                closest_station_distance = station_distance
                closest_station = station
                
        return closest_station, closest_station_distance
        
    
    def get_nexrad_day_urls():
        t1 = iw_datetime - timedelta(minutes=delta_minutes)
        t2 = iw_datetime + timedelta(minutes=delta_minutes)

        url_basis = "https://noaa-nexrad-level2.s3.amazonaws.com/?list-type=2&delimiter=%2F"
        url1 = url_basis + f"&prefix={t1.year}%2F{t1.month:02}%2F{t1.day:02}%2F{closest_station}%2F"
        url2 = url_basis + f"&prefix={t2.year}%2F{t2.month:02}%2F{t2.day:02}%2F{closest_station}%2F"
        
        urls = [url1] 
        if url2 != url1: urls += [url2] 
        return urls
    
    def get_hour_urls():
        maximum_delta = timedelta(minutes=delta_minutes)
        
        hour_urls = []
        closest_url = None
        for day_url in day_urls:
            req = request.urlopen(day_url)
            tree = ElementTree.parse(req)
            for elem in tree.iter():
                if elem.tag.endswith('Key'):
                    if not elem.text.endswith('V06'): continue
                    key_datetime = datetime.strptime(os.path.split(elem.text)[1][4:-4], '%Y%m%d_%H%M%S')
                    
                    current_timedelta = abs(key_datetime - iw_datetime)
                    
                    url = f"https://noaa-nexrad-level2.s3.amazonaws.com/{elem.text}"
                    if current_timedelta < maximum_delta:
                        hour_urls.append(url)
                    if closest_url is None or current_timedelta < smallest_timedelta:
                        closest_url = url
                        smallest_timedelta = current_timedelta
                        
        return hour_urls, closest_url
    
    def download_nexrad_data():
        new_filenames = [os.path.split(url)[1] for url in hour_urls]
        for filename in os.listdir('.temp'): 
            if os.path.splitext(filename)[0] not in new_filenames:
                os.remove(os.path.join('.temp', filename))
        
        new_filenames = []
        for i, url in enumerate(hour_urls):
            if verbose: log_print(f'Downloading {i+1}/{len(hour_urls)} files', f=r_print if i else print)
            new_filename = f".temp/{os.path.split(url)[1]}.ar2v"
            new_filenames.append(new_filename)
            
            if os.path.exists(new_filename): continue
            req = request.urlopen(url).read()
            with open(new_filename, "wb") as file:
                file.write(req)
        print()
                
        return new_filenames
    
    def project_nexrad_on_iw_lat_lon(shape):
        radar = pyart.io.read_nexrad_archive(f".temp/{os.path.split(closest_url)[1]}.ar2v")
        
        lats, lons, alts = radar.get_gate_lat_lon_alt(sweep=0)
        data = radar.get_field(sweep=0, field_name='reflectivity')

        new_data = griddata(
            np.stack((lats.flatten(), lons.flatten()), axis=1),
            data.flatten(),
            np.stack((owiLat.flatten(), owiLon.flatten()), axis=1)
        ).reshape(shape).astype('float')

        np.savez_compressed(f'outputs/{key}/{key}_NEXRAD.npz', new_data)

        new_data = np.clip(new_data, 0, 40)/40
        new_data = plt.get_cmap('turbo')(new_data)
        new_data = (new_data * 255).astype(np.uint8)

        new_filename = f'outputs/{key}/{key}_NEXRAD.png'
        PIL.Image.fromarray(new_data).save(new_filename)
        return new_filename

    start = datetime.now()
        
    if verbose: log_print("Loading IW metadata")
    getter = getter_polygon_from_key(sensoroperationalmode=sensoroperationalmode)
    if verbose: log_print("Loading OK")
    
    key = key.lower()
    iw_datetime = datetime.strptime(key, '%Y%m%dt%H%M%S')
    iw_filename, polygon, orbit = getter(key)
    
    if verbose: log_print("Search NEXRAD closest station")
    closest_station, closest_station_distance = get_closest_station()
    if verbose: log_print(f"{closest_station} at {closest_station_distance:01} km")
    if closest_station_distance > 500 : return
    
    if verbose: log_print('Retrieve urls')
    day_urls = get_nexrad_day_urls()
    hour_urls, closest_url = get_hour_urls()
    new_filenames = download_nexrad_data()

    m = None
    for i, filename in enumerate(new_filenames):
        if verbose: log_print(f'Generating {i+1}/{len(new_filenames)} .png', f=r_print if i else print)
        new_filenames[i], m = ar2v_to_png(filename, polygon=polygon, m=m)
    if verbose: print()
    
    if verbose: log_print('Generating .gif')
    os.makedirs('outputs/' + key, exist_ok=True)
    png_to_gif(new_filenames, f'outputs/{key}/{key}_NEXRAD.gif')
    
    if verbose: log_print('Reproject in .npz and .png')
    owiLat, owiLon = get_iw_latlon(polygon=polygon, metadata_filename=metadata_filename, shape=shape)
    if owiLat is not None:
        new_filename = project_nexrad_on_iw_lat_lon(owiLat.shape)
        if verbose: log_print('Done')
    if verbose: print('Executed in', (datetime(1,1,1) + (datetime.now()-start)).strftime('%H hours, %M minutes, %S seconds'))

if __name__ == "__main__":
    fire.Fire(maps_from_iw_key)
