import os

import fire
import numpy as np

np.seterr(all="ignore")

import matplotlib

matplotlib.use('agg')

from colocalisations.package_utils.sentinel1 import get_iw_latlon
from colocalisations.package_utils.closest_data import get_closest_filenames
from colocalisations.package_utils.read_from import read_from_files_per_platform
from colocalisations.package_utils.projection import reproject, save_reprojection, generate_gif
from colocalisations.package_utils.misc import log_print
from colocalisations.package_utils.check_args import check_args


def main(
        pattern=None,
        lat_key=None,
        lon_key=None,
        time_key=None,
        sentinel1_key=None,
        sentinel1_keys_filename=None,
        requests_filename=None,
        channel=None,
        sensor_operational_mode=None,
        data=None,
        max_timedelta=None,
        time_step=None,
        create_gif=False,
        verbose=None,
        delta_factor=None,
        continue_on_error=False,
        output_folder='outputs'
):
    keys, channel, verbose, platforms, create_gif, max_timedelta, time_step, delta_factor = check_args(
        pattern=pattern,
        lat_key=lat_key,
        lon_key=lon_key,
        time_key=time_key,
        sentinel1_key=sentinel1_key,
        sentinel1_keys_filename=sentinel1_keys_filename,
        requests_filename=requests_filename,
        channel=channel,
        sensor_operational_mode=sensor_operational_mode,
        data=data,
        max_timedelta=max_timedelta,
        time_step=time_step,
        create_gif=create_gif,
        verbose=verbose,
        delta_factor=delta_factor
    )

    for i, (filename, requested_date, polygon) in enumerate(keys):
        try:
            log_print(f"Request {i + 1}/{len(keys)}: {filename}", 1, verbose)
            projection_lats, projection_lons = get_iw_latlon(polygon=polygon)

            log_print("Retrieve files urls", 2, verbose)
            channel, platform, urls_per_platforms, (
                platform_lat, platform_lon, closest_file_data) = get_closest_filenames(
                channel, polygon, requested_date, max_timedelta, time_step, platforms)
            log_print(f"Selected platform is {platform} with channel {channel}", 2, verbose)

            log_print("Project on S1 lat/lon grid", 2, verbose)
            closest_file_data = reproject(platform, closest_file_data, platform_lat, platform_lon, projection_lats,
                                          projection_lons)
            data = {f"data": closest_file_data, 'lats': platform_lat, 'lons': platform_lon}
            new_filename = f'{output_folder}/{filename}/{filename}_{channel}'
            new_filename = save_reprojection(platform, channel, data, new_filename)
            log_print(f"Saved in `{new_filename}`", 2, verbose)
            if create_gif:
                log_print(".gif generation is asked", 2, verbose)
                gif_filename = f'{output_folder}/{filename}/{filename}_{channel}.gif'
                generate_gif(polygon, channel, urls_per_platforms, gif_filename, verbose,
                             read_from_files_per_platform, delta_factor=delta_factor)
                log_print(f"Saved in `{gif_filename}`", 2, verbose)

        except Exception as e:
            log_print(f'Exception on request {filename}: {e}', 0, verbose)
            if continue_on_error:
                continue
            else:
                raise e
    log_print("Done", 1, verbose)


if __name__ == "__main__":
    fire.Fire(main)
