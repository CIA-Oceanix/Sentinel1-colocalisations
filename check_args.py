import os
import numpy as np
from datetime import datetime

GOES_SERIE = ["goes16", "goes17", "goes18"]
HIMAWARI_SERIE = ["himawari8", "himawari9"]
NEXRAD_BASIS = ["nexrad-level2"]
ERA5_PLATFORMS = ['era5']
SATELLITE_PLATFORMS = GOES_SERIE + HIMAWARI_SERIE

ABI_CHANNELS = ['C13', 'C14'] # C01 to C16 are available, but would need other cmap
RRQPEF_CHANNELS = ['RRQPEF']
GLM_CHANNELS = ['GLM']
NEXRAD_L3_CHANNELS = [ 'DPR', 'N0Q', 'N0Q', 'N0M', 'N0H', 'HHC', 'N0Z']

ERA5_CHANNELS = ['northward_wind_at_10_metres', 'eastward_wind_at_10_metres']

with open('res/nexrad_stations.txt', 'r') as file:
    NEXRAD_CHANNELS = [line.split('\t')[1] for line in file.readlines()[1:]]

def get_keys(key):
    assert os.path.exists(key)
    with open(key, 'r') as file:
        lines = file.readlines()
        keys = [line.replace('\n', 'r') for line in lines]
    return keys
    

def check_args(
    sentinel1_key = None,
    sentinel1_keys_filename = None,
    requests_filename = None,
    channel = None,
    sensor_operational_mode = None,
    platform_key = None,
    max_timedelta = None,
    time_step = None,
    create_gif=None,
    verbose=None,
    delta_factor=None
    ):
    from utils.sentinel1 import getter_polygon_from_key
    from utils.misc import log_print

    # Set default values
    if verbose is None: verbose = 1
    if requests_filename is None and sensor_operational_mode is None: sensor_operational_mode = 'IW'
    
    if max_timedelta is None: max_timedelta= 90
    if time_step is None: time_step= 10
    if delta_factor is None: delta_factor=2
    

    # Choose plateform
    if platform_key == 'nexrad':
        platforms = NEXRAD_BASIS
    elif platform_key in ['abi', "rrqpef"]:
        platforms = SATELLITE_PLATFORMS
    elif platform_key in ['glm']:
        platforms = GOES_SERIE
    elif platform_key == 'era5':
        platforms = ERA5_PLATFORMS
    elif channel in NEXRAD_L3_CHANNELS:
        platforms = 'nexrad-level3'
    else:
        raise ValueError

    if platform_key == "glm": channel = "GLM"
    if platform_key == "rrqpef": channel = "RRQPEF"

    # Format the requests
    requests = []
    if sentinel1_key or sentinel1_keys_filename:
        if verbose: log_print(f"Build {sensor_operational_mode} getter")
        getter = getter_polygon_from_key(sensor_operational_mode)

        keys = get_keys(sentinel1_keys_filename) if sentinel1_keys_filename else [sentinel1_key]
        for key in keys:
            iw_filename, iw_polygon = getter(key)[:2]
            iw_datetime = datetime.strptime(key, '%Y%m%dt%H%M%S')
            requests.append((key, iw_datetime, iw_polygon))
    else:
        lines = get_keys(requests_filename)
        for line in lines:
            key, requested_datetime_string, lat1, lon1, lat2, lon2, lat3, lon3, lat4, lon4 = line.split('\t')
            requested_datetime = datetime.strptime(requested_datetime_string, '%Y%m%dt%H%M%S')
            polygon = np.array([
                [lon1, lat1],
                [lon2, lat2],
                [lon3, lat3],
                [lon4, lat4],
            ]).astype('float')
            requests.append((key, requested_datetime, polygon))
            

    return requests, channel, verbose, sensor_operational_mode, platforms, create_gif, max_timedelta, time_step, delta_factor
    
