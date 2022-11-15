import shutup; shutup.please()

import os
import fire
import shutil
import numpy as np
np.seterr(all="ignore")

from datetime import datetime

import matplotlib
matplotlib.use('agg')


shutil.rmtree('.temp', ignore_errors=True)
os.makedirs('.temp', exist_ok=True)
os.makedirs('outputs', exist_ok=True)


from utils.requests import download_files
from utils.sentinel1 import getter_polygon_from_key, get_iw_latlon
from utils.closest_data import get_closest_filenames
from utils.read import read_from_files_per_platform
from utils.projection import reproject, save_reprojection, generate_gif
from utils.misc import log_print

from check_args import check_args, GOES_SERIE, HIMAWARI_SERIE, NEXRAD_BASIS, SATELLITE_PLATFORMS


def main(key=None, channel=None, shape=None, metadata_filename=None, sensoroperationalmode=None, platform_key = None, max_timedelta=None, time_step=None, gif=True, verbose=None):

    keys, channel, shape, metadata_filename, verbose, sensoroperationalmode, platforms, gif, max_timedelta, time_step = check_args(
        key=key, channel=channel, shape=shape, metadata_filename=metadata_filename, verbose=sensoroperationalmode, sensoroperationalmode=sensoroperationalmode,
        platform_key=platform_key, gif=gif, max_timedelta=max_timedelta, time_step=time_step
    )
    
    if verbose: log_print(f"Build {sensoroperationalmode} getter")
    getter = getter_polygon_from_key()

    for key in keys:
        if verbose: log_print(f"Run on key: {key}")
        iw_datetime = datetime.strptime(key, '%Y%m%dt%H%M%S')
        iw_filename, iw_polygon = getter(key)[:2]
        owi_lat, owi_lon = get_iw_latlon(polygon=iw_polygon, metadata_filename=metadata_filename, shape=shape)

        if verbose: log_print("Retrieve files urls")
        platform, urls_per_platforms, (platform_lat, platform_lon, closest_file_data) = get_closest_filenames(channel, iw_polygon, iw_datetime, max_timedelta, time_step, platforms)

        if verbose: log_print("Project on S1 lat/lon grid")
        closest_file_data = reproject(platform, closest_file_data, platform_lat, platform_lon, owi_lat, owi_lon)
        save_reprojection(platform, closest_file_data, f'outputs/{key}/{key}_{channel}')

        if gif:
            if verbose: log_print("Generate .gif")
            generate_gif(iw_polygon, channel, urls_per_platforms, f'outputs/{key}/{key}_{channel}.gif')
    
    
if __name__ == "__main__":
    fire.Fire(main)

