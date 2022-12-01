import shutup; shutup.please()

import os
import fire
import shutil
import numpy as np
np.seterr(all="ignore")

import matplotlib
#matplotlib.use('agg')

shutil.rmtree('.temp', ignore_errors=True)
os.makedirs('.temp', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

from utils.sentinel1 import get_iw_latlon
from utils.closest_data import get_closest_filenames
from utils.read import read_from_files_per_platform
from utils.projection import reproject, save_reprojection, generate_gif
from utils.misc import log_print

from check_args import check_args, GOES_SERIE, HIMAWARI_SERIE, NEXRAD_BASIS, SATELLITE_PLATFORMS


def main(
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
    delta_factor=None):
        
    keys, channel, verbose, sensor_operational_mode, platforms, create_gif, max_timedelta, time_step, delta_factor = check_args(
        sentinel1_key = sentinel1_key,
        sentinel1_keys_filename = sentinel1_keys_filename,
        requests_filename = requests_filename,
        channel = channel,
        sensor_operational_mode = sensor_operational_mode,
        platform_key = platform_key,
        max_timedelta = max_timedelta,
        time_step = time_step,
        create_gif = create_gif,
        verbose = verbose,
        delta_factor = delta_factor
    )
    

    for i, (filename, requested_date, polygon) in enumerate(keys):
        if verbose: log_print(f"Request {i+1}/{len(keys)}: {filename}")
        projection_lats, projection_lons = get_iw_latlon(polygon=polygon)
            
        if verbose > 1: log_print("Retrieve files urls")
        channel, platform, urls_per_platforms, (platform_lat, platform_lon, closest_file_data) = get_closest_filenames(channel, polygon, requested_date, max_timedelta, time_step, platforms)
        if verbose > 1: log_print(f"Selected plateform is {platform} with channel {channel}")

        if verbose > 1: log_print("Project on S1 lat/lon grid")
        closest_file_data = reproject(platform, closest_file_data, platform_lat, platform_lon, projection_lats, projection_lons)
        save_reprojection(platform, channel,closest_file_data, f'outputs/{filename}/{filename}_{channel}')

        if create_gif:
            if verbose: log_print(".gif generation is asked")
            generate_gif(polygon, channel, urls_per_platforms, f'outputs/{filename}/{filename}_{channel}.gif', verbose, read_from_files_per_platform, requested_date=requested_date, delta_factor=delta_factor)
    if verbose: log_print("Done")
    
    
if __name__ == "__main__":
    fire.Fire(main)

