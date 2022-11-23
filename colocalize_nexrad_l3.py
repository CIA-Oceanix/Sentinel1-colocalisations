import shutup; shutup.please()


import os
import fire
import shutil
import tarfile

import numpy as np
np.seterr(all="ignore")

from datetime import datetime, timedelta
from functools import lru_cache

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')

from metpy.cbook import get_test_data
from metpy.io import Level3File
import pyproj
WGS84 = pyproj.Geod(ellps='WGS84')

from utils.misc import log_print
from utils.requests import download_files
from utils.sentinel1 import getter_polygon_from_key, get_iw_latlon
from utils.closest_data import get_closest_nexrad_station
from utils.projection import save_reprojection, reproject, generate_gif
from utils.nexrad_l3 import read_melting_layer

from check_args import get_keys


shutil.rmtree('.temp', ignore_errors=True)
os.makedirs('.temp', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# 'DPR': 'Digital Instantaneous Precipitation Rate'
# 'NYQ': 'Base Reflectivity Data Array'
# 'NXQ': 'Base Reflectivity Data Array',
# 'NHI': 'Hail Index'
# 'NZM': 'Melting Layer'
# 'NXH': 'Digital Hydrometeor Classification'
# 'HHC': 'Hybrid Hydrometeor Classification'

@lru_cache(maxsize=2**16)
def cached_command(command):
    return os.popen(command).readlines()

def get_urls(channel, date):
    urls = []
    prefix = f"{date.year}/{date.month:02}/{date.day:02}/{channel}"
    lines = cached_command('gsutil -q ls -l gs://gcp-public-data-nexrad-l3/' + prefix)
    for line in lines:
        line = line.replace('\n', '')
        if line.endswith('.tar.gz'):
            urls.append(line.split()[-1])
    return urls

def get_bucket_urls(channel, iw_datetime, max_timedelta, time_step):
    time_steps = range(-max_timedelta, max_timedelta+1, time_step)
    dates = [iw_datetime + timedelta(minutes=x) for x in time_steps]
    
    urls = {channel: {}}
    for date in dates:
        urls[channel][date] = get_urls(channel, date)
    return urls

def untar(filenames_per_platform, channel):
    new_filenames_per_platform = {}
    for platform in filenames_per_platform:
        extracted = {}
        new_filenames_per_platform[platform] = {}
        for date, filenames in filenames_per_platform[platform].items():
            new_filenames_per_platform[platform][date] = []
            for filename in filenames:
                if filename in extracted:
                    new_filenames_per_platform[platform][date] += new_filenames_per_platform[platform][extracted[filename]]
                    continue
                with tarfile.open(filename) as file:
                    folder = os.path.split(filename)[0]
                    for compressed_filename in file.getnames():
                        if compressed_filename.split('_')[-2] == channel:
                            new_filename = os.path.join(folder, compressed_filename)
                            if os.path.exists(new_filename): continue
                            file.extract(compressed_filename, folder)
                            new_filenames_per_platform[platform][date].append(new_filename)
                extracted[filename] = date
                #os.remove(filename)
    return new_filenames_per_platform


def read(filenames, platform=None, channel=None):
    radar =  Level3File(filenames[0])

    if channel[:3] in ["N0M", "N1M", "N2M", "N3M"]:
        return read_melting_layer(radar)
        
    datadict = radar.sym_block[0][0]
        
    if 'latitude' in datadict:
        width = 0.25
        lat = datadict['latitude']
        lon = datadict['longitude']
        radials = datadict['components'].radials
        
        azimuths = np.array([radial.azimuth for radial in radials])
        ranges = np.array([i*width for i in range(radials[0].num_bins)])
        data = np.array([radial.data for radial in radials])
    else:
        lon = radar.lon
        lat = radar.lat 
        
        azimuths = np.array(datadict['start_az'] + [datadict['end_az'][-1]])
        data = radar.map_data(datadict['data'])
        if channel[:3] in ['DPR']: data = data / 1000 * 25.4 # milipouce/h to mm/h

        
        ranges = np.linspace(0, radar.max_range, data.shape[-1] + 1)

    range_grid, azimuth_grid = np.meshgrid(ranges, azimuths)
    lons, lats, _ = WGS84.fwd(
        np.ones(azimuth_grid.shape)*lon,
        np.ones(azimuth_grid.shape)*lat,
        azimuth_grid,
        range_grid*1000
    )
    if lons.shape[0] == data.shape[0] +1:
        lats = lats[:-1]
        lons = lons[:-1]
    if lons.shape[1] == data.shape[1] +1:
        lats = lats[:,:-1]
        lons = lons[:,:-1]
    return lats, lons, data

def restrict_filenames_from_date(filenames_per_platform):
    for platform in filenames_per_platform:
        for date, filenames in filenames_per_platform[platform].items():
            smallest_timedelta = None
            for filename in filenames:
                filename_date = datetime.strptime(filename.split('_')[-1], '%Y%m%d%H%M')
                current_timedelta = abs(filename_date - date)
                if smallest_timedelta is None or current_timedelta < smallest_timedelta:
                    filenames_per_platform[platform][date] = [filename]
                    smallest_timedelta = current_timedelta
    return filenames_per_platform
    
def main(key, channel='DPR', verbose=1, sensoroperationalmode="IW", max_timedelta=90, time_step=5, gif=True):
    keys = get_keys(key)

    if verbose: log_print(f"Build {sensoroperationalmode} getter")
    getter = getter_polygon_from_key(sensoroperationalmode)

    for key in keys:
        key = key.lower()
        iw_polygon = getter(key)[1]
        s1_time = datetime.strptime(key, '%Y%m%dt%H%M%S')
        owi_lat, owi_lon = get_iw_latlon(polygon=iw_polygon)
        
        if verbose: log_print(f"Retrieve NEXRAD colocalizations")
        closest_station = get_closest_nexrad_station(iw_polygon)
        channel += closest_station[1:]
        if verbose: log_print(f"Closest station is {closest_station}")

        if verbose: log_print(f"Downloading")
        urls_per_platforms = get_bucket_urls(closest_station, s1_time, max_timedelta, time_step)
        filenames_per_platform = download_files(urls_per_platforms, closest=False)
        
        if verbose: log_print("Extracting")
        filenames_per_platform = untar(filenames_per_platform, channel)
        if not filenames_per_platform[closest_station]:
            if verbose: log_print(f"Station {closest_station} has no data for channel {channel} at {s1_time}")
            return
            
        close_filenames_per_platform = restrict_filenames_from_date(filenames_per_platform)
        
        if verbose: log_print("Project on S1 lat/lon grid")
        closest_date = sorted([(abs(s1_time - date), date) for date in filenames_per_platform[closest_station]])[0][1]
        lats, lons, data = read(filenames_per_platform[closest_station][closest_date], channel=channel)
        closest_file_data = reproject(closest_station, data, lats, lons, owi_lat, owi_lon)

        os.makedirs('outputs/' + key, exist_ok=True)
        save_reprojection(closest_station, channel, closest_file_data, f'outputs/{key}/{key}_{channel}')
        
        if gif:
            if verbose: log_print(".gif generation is asked")
            generate_gif(iw_polygon, channel, filenames_per_platform, f'outputs/{key}/{key}_{channel}.gif', verbose, read, download=False)
        if verbose: log_print("Done")

if __name__ == "__main__":
    fire.Fire(main)
