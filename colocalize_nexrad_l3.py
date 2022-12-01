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
from utils.sentinel1 import get_iw_latlon
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


def read(filenames, platform=None, channel=None, requested_date=None):
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
    
def main(
    sentinel1_key = None,
    sentinel1_keys_filename = None,
    requests_filename = None,
    channel = 'DPR',
    sensor_operational_mode = None,
    platform_key = None,
    max_timedelta = None,
    time_step = 5,
    create_gif = None,
    verbose = None):
        
    keys, channel, verbose, sensor_operational_mode, platforms, create_gif, max_timedelta, time_step = check_args(
        sentinel1_key = sentinel1_key,
        sentinel1_keys_filename = sentinel1_keys_filename,
        requests_filename = requests_filename,
        channel = channel,
        sensor_operational_mode = sensor_operational_mode,
        platform_key = platform_key,
        max_timedelta = max_timedelta,
        time_step = time_step,
        create_gif = create_gif,
        verbose = verbose
    )
    
    for i, (filename, requested_date, polygon) in enumerate(keys):
        if verbose: log_print(f"Request {i+1}/{len(keys)}: {filename}")
        projection_lats, projection_lons = get_iw_latlon(polygon=polygon)
        
        if verbose: log_print(f"Retrieve NEXRAD colocalizations")
        closest_station = get_closest_nexrad_station(polygon)
        channel += closest_station[1:]
        if verbose: log_print(f"Closest station is {closest_station}")

        if verbose: log_print(f"Downloading")
        urls_per_platforms = get_bucket_urls(closest_station, requested_date, max_timedelta, time_step)
        filenames_per_platform = download_files(urls_per_platforms, closest=False)
        
        if verbose: log_print("Extracting")
        filenames_per_platform = untar(filenames_per_platform, channel)
        if not filenames_per_platform[closest_station]:
            if verbose: log_print(f"Station {closest_station} has no data for channel {channel} at {requested_date}")
            return
            
        close_filenames_per_platform = restrict_filenames_from_date(filenames_per_platform)
        
        if verbose: log_print("Project on S1 lat/lon grid")
        closest_date = sorted([(abs(requested_date - date), date) for date in filenames_per_platform[closest_station]])[0][1]
        lats, lons, data = read(filenames_per_platform[closest_station][closest_date], channel=channel)
        closest_file_data = reproject(closest_station, data, lats, lons, projection_lats, projection_lons)

        os.makedirs('outputs/' + key, exist_ok=True)
        save_reprojection(closest_station, channel, closest_file_data, f'outputs/{filename}/{filename}_{channel}')
        
        if create_gif:
            if verbose: log_print(".gif generation is asked")
            generate_gif(polygon, channel, filenames_per_platform, f'outputs/{filename}/{filename}_{channel}.gif', verbose, read, download=False)
    if verbose: log_print("Done")

if __name__ == "__main__":
    fire.Fire(main)
