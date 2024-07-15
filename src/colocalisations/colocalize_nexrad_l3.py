import os
import tarfile

import fire
import numpy as np

np.seterr(all="ignore")

from datetime import datetime, timedelta
from functools import lru_cache

import matplotlib

matplotlib.use('agg')

from metpy.io import Level3File
import pyproj

WGS84 = pyproj.Geod(ellps='WGS84')

from colocalisations.package_utils.misc import log_print
from colocalisations.package_utils import download_files
from colocalisations.package_utils.sentinel1 import get_iw_latlon
from colocalisations.package_utils.closest_data import get_closest_nexrad_station
from colocalisations.package_utils.projection import save_reprojection, reproject, generate_gif
from colocalisations.package_utils import read_melting_layer
from colocalisations.package_utils.check_args import check_args

# shutil.rmtree('.temp', ignore_errors=True)
os.makedirs('.temp', exist_ok=True)
os.makedirs('outputs', exist_ok=True)


# 'DPR': 'Digital Instantaneous Precipitation Rate'
# 'NYQ': 'Base Reflectivity Data Array'
# 'NXQ': 'Base Reflectivity Data Array',
# 'NHI': 'Hail Index'
# 'NZM': 'Melting Layer'
# 'NXH': 'Digital Hydrometeor Classification'
# 'HHC': 'Hybrid Hydrometeor Classification'

@lru_cache(maxsize=2 ** 16)
def cached_command(command):
    return os.popen(command).readlines()


def get_urls(channel, date):
    urls = []
    prefix = f"{date.year}/{date.month:02}/{date.day:02}/{channel}"
    lines = cached_command('gsutil -q ls -l gs://gcp-public-data-nexrad-l3/' + prefix)
    for line in lines:
        line = line.replace('\n', '')
        if line.endswith('.tar.gz') or line.endswith('.Z'):
            urls.append(line.split()[-1])
    return urls


def get_bucket_urls(channel, iw_datetime, max_timedelta, time_step):
    time_steps = range(-max_timedelta, max_timedelta + 1, time_step)
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
                    new_filenames_per_platform[platform][date] += extracted[filename]
                    continue

                with tarfile.open(filename) as file:
                    folder = os.path.split(filename)[0]
                    for i, compressed_filename in enumerate(file.getnames()):
                        if compressed_filename.split('_')[-2] == channel:
                            new_filename = os.path.join(folder, compressed_filename)
                            if not os.path.exists(new_filename):
                                file.extract(compressed_filename, folder)
                            new_filenames_per_platform[platform][date].append(new_filename)
                extracted[filename] = new_filenames_per_platform[platform][date]
                os.remove(filename)
    return new_filenames_per_platform


def read(filenames, platform=None, channel=None, requested_date=None):
    smallest_timedelta = None
    for filename in filenames:
        filename_datetime = datetime.strptime(filename.split('_')[-1], '%Y%m%d%H%M')
        current_timedelta = abs(requested_date - filename_datetime)
        if smallest_timedelta is None or current_timedelta < smallest_timedelta:
            closest_filename = filename
            smallest_timedelta = current_timedelta

    radar = Level3File(closest_filename)

    if channel[:3] in ["N0M", "N1M", "N2M", "N3M"]:
        return read_melting_layer(radar)

    datadict = radar.sym_block[0][0]

    if 'latitude' in datadict:
        width = 0.25
        lat = datadict['latitude']
        lon = datadict['longitude']
        radials = datadict['components'].radials

        azimuths = np.array([radial.azimuth for radial in radials])
        ranges = np.array([i * width for i in range(radials[0].num_bins)])
        data = np.array([radial.data for radial in radials])
    else:
        lon = radar.lon
        lat = radar.lat

        azimuths = np.array(datadict['start_az'] + [datadict['end_az'][-1]])
        data = radar.map_data(datadict['data'])
        if channel[:3] in ['DPR']: data = data / 1000 * 25.4  # milipouce/h to mm/h

        ranges = np.linspace(0, radar.max_range, data.shape[-1] + 1)

    range_grid, azimuth_grid = np.meshgrid(ranges, azimuths)
    lons, lats, _ = WGS84.fwd(
        np.ones(azimuth_grid.shape) * lon,
        np.ones(azimuth_grid.shape) * lat,
        azimuth_grid,
        range_grid * 1000
    )

    if lons.shape[0] == data.shape[0] + 1:
        lats = lats[:-1]
        lons = lons[:-1]
    if lons.shape[1] == data.shape[1] + 1:
        lats = lats[:, :-1]
        lons = lons[:, :-1]
    return lats, lons, data


def main(
        pattern=None,
        lat_key=None,
        lon_key=None,
        time_key=None,
        sentinel1_key=None,
        sentinel1_keys_filename=None,
        requests_filename=None,
        sensor_operational_mode=None,
        data='NEXRAD_L3',
        channel=None,
        max_timedelta=None,
        time_step=5,
        create_gif=False,
        verbose=None,
        delta_factor=None,
        output_folder = 'outputs'
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
        log_print(f"Request {i + 1}/{len(keys)}: {filename}", 1, verbose)
        projection_lats, projection_lons = get_iw_latlon(polygon=polygon)

        log_print(f"Retrieve NEXRAD colocalizations", 2, verbose)
        closest_station = get_closest_nexrad_station(polygon)
        long_channel = channel + closest_station[1:]
        log_print(f"Closest station is {closest_station}", 2, verbose)

        log_print(f"Downloading", 2, verbose)
        urls_per_platforms = get_bucket_urls(closest_station, requested_date, max_timedelta, time_step)
        filenames_per_platform = download_files(urls_per_platforms, closest=False)

        log_print("Extracting", 2, verbose)
        filenames_per_platform = untar(filenames_per_platform, long_channel)
        if not filenames_per_platform[closest_station]:
            log_print(f"Station {closest_station} has no data for channel {channel} at {requested_date}", 1, verbose)
            continue

        log_print("Project on S1 lat/lon grid", 2, verbose)
        closest_date = \
            sorted([(abs(requested_date - date), date) for date in filenames_per_platform[closest_station]])[0][1]
        lats, lons, data = read(filenames_per_platform[closest_station][closest_date], channel=long_channel,
                                requested_date=closest_date)
        closest_file_data = reproject(closest_station, data, lats, lons, projection_lats, projection_lons)
        data = {f"data": closest_file_data, 'lats': projection_lats, 'lons': projection_lons}

        os.makedirs('outputs/' + filename, exist_ok=True)
        new_filename = f'{output_folder}/{filename}/{filename}_{channel}'
        new_filename = save_reprojection(closest_station, long_channel, data, new_filename)
        log_print(f"Saved in `{new_filename}`", 2, verbose)

        if create_gif:
            log_print(".gif generation is asked", 2, verbose)
            gif_filename = f'outputs/{filename}/{filename}_{channel}.gif'
            generate_gif(polygon, channel, filenames_per_platform, gif_filename,
                         verbose, read, download_asked=False, delta_factor=delta_factor)
        log_print(f"Saved in _`{new_filename}`", 2, verbose)

    log_print("Done", 1, verbose)


if __name__ == "__main__":
    fire.Fire(main)
