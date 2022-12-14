import os
from datetime import datetime

import numpy as np

SATELLITE_PLATFORMS = {
    'GOES': ["goes16", "goes17", "goes18"],
    'HIMAWARI': ["himawari8", "himawari9"],
    'SEVIRIS': ["EO:EUM:DAT:MSG:HRSEVIRI", "EO:EUM:DAT:MSG:HRSEVIRI-IODC"]
}
SATELLITE_PLATFORMS['any'] = [platform for l in SATELLITE_PLATFORMS.values() for platform in l]
SATELLITE_PLATFORMS['GLM'] = SATELLITE_PLATFORMS['GOES']
SATELLITE_PLATFORMS['ABI'] = SATELLITE_PLATFORMS['GOES'] + SATELLITE_PLATFORMS['HIMAWARI']
SATELLITE_PLATFORMS['RRQPEF'] = SATELLITE_PLATFORMS['GOES'] + SATELLITE_PLATFORMS['HIMAWARI']


CHANNELS = {
    'ABI': ['C13', 'C14'],
    'RRQPEF': 'RRQPEF',
    'GLM': 'GLM',
    'ERA5': ['northward_wind_at_10_metres', 'eastward_wind_at_10_metres'],
    'NEXRAD_L2': "nexrad-level2",
    'NEXRAD_L3': ['DPR', 'N0Q', 'N0Q', 'N0M', 'N0H', 'HHC', 'N0Z'],
    'SEVIRIS': "HRSEVIRI"
}

def get_keys(key):
    assert os.path.exists(key)
    with open(key, 'r') as file:
        lines = file.readlines()
        keys = [line.replace('\n', '') for line in lines]
    return keys


def check_args(
        sentinel1_key=None,
        sentinel1_keys_filename=None,
        requests_filename=None,
        channel=None,
        sensor_operational_mode=None,
        data=None,
        max_timedelta=None,
        time_step=None,
        create_gif=None,
        verbose=None,
        delta_factor=None
):
    from utils.sentinel1 import getter_polygon_from_key
    from utils.misc import log_print

    # Set default values
    if verbose is None: verbose = 2
    if requests_filename is None and sensor_operational_mode is None: sensor_operational_mode = 'IW'

    if max_timedelta is None: max_timedelta = 90
    if time_step is None: time_step = 10
    if delta_factor is None: delta_factor = 2

    # Choose plateform/channels
    if data in ['ABI', "RRQPEF", 'GLM']:
        platforms = SATELLITE_PLATFORMS[data]
    else:
        platforms = [data]
    if isinstance(CHANNELS[data], list):
        assert channel in CHANNELS[data]
    else:
        channel = CHANNELS[data]


    # Format the requests
    requests = []
    if sentinel1_key or sentinel1_keys_filename:
        log_print(f"Build {sensor_operational_mode} getter", 2, verbose)
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

    return requests, channel, verbose, platforms, create_gif, max_timedelta, time_step, delta_factor
