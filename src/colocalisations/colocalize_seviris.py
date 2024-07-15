import os
import shutil
import zipfile
from datetime import datetime, timedelta

import fire
import numpy as np

np.seterr(all="ignore")

import eumdac
from satpy import Scene

from colocalisations.package_utils.sentinel1 import get_iw_latlon
from colocalisations.package_utils.projection import reproject, save_reprojection, generate_gif
from colocalisations.package_utils.misc import log_print, lat_lon_from_polygon
from colocalisations.package_utils.check_args import check_args

# shutil.rmtree('.temp', ignore_errors=True)
os.makedirs('.temp', exist_ok=True)
os.makedirs('outputs', exist_ok=True)


def read_products(filenames, platform=None, channel=None, requested_date=None):
    scene = Scene(reader="seviri_l1b_native", filenames=filenames)
    scene.load([10.8], calibration=['brightness_temperature'])
    scene = scene['IR_108']
    lons, lats = scene.attrs['area'].get_lonlats()
    return lats, lons, scene.values


def download_product(product):
    with product.open() as source:
        new_filename = os.path.splitext(source.name)[0]
        long_new_filename = ".temp/" + new_filename + ".nat"
        if not os.path.exists(long_new_filename):
            with open(f".temp/{source.name}", mode='wb') as file:
                shutil.copyfileobj(source, file)

            with zipfile.ZipFile(".temp/" + new_filename + ".zip", "r") as zip_ref:
                zip_ref.extract(os.path.split(long_new_filename)[1], ".temp")

    return long_new_filename


def get_product_by_date(platform, requested_datetime, max_timedelta, time_step):
    credentials = (os.environ['EUMETSAT_CONSUMER_KEY'], os.environ['EUMETSAT_CONSUMER_SECRET'])
    token = eumdac.AccessToken(credentials)
    collection = eumdac.DataStore(token).get_collection(platform)

    products = {platform: {}}
    time_steps = range(-max_timedelta, max_timedelta + 1, time_step)
    for date in [requested_datetime + timedelta(minutes=x) for x in time_steps]:
        start = date - timedelta(minutes=time_step / 2)
        end = date + timedelta(minutes=time_step / 2)

        smallest_timedelta = None
        for product in collection.search(dtstart=start, dtend=end):
            observation_date = product.sensing_start + (product.sensing_end - product.sensing_start) / 2
            current_timedelta = abs(date - observation_date)
            if smallest_timedelta is None or current_timedelta < smallest_timedelta:
                products[platform][date] = product
    return products


def get_closest_products(polygon, requested_date, max_timedelta, time_step):
    lats, lons = lat_lon_from_polygon(polygon)
    mean_iw_lon = np.mean(lons)

    platform = None
    if 22.75 < mean_iw_lon < 120 and requested_date > datetime(year=2017, month=2, day=2):
        platform = "EO:EUM:DAT:MSG:HRSEVIRI-IODC"
    if -30 < mean_iw_lon < 22.75:  platform = "EO:EUM:DAT:MSG:HRSEVIRI"
    if platform is None:
        raise ValueError

    products = get_product_by_date(platform, requested_date, max_timedelta, time_step)
    closest_product = sorted(
        (abs(key - requested_date), value)
        for key, value in products[platform].items()
    )[0][1]
    return platform, products, closest_product


def main(
        pattern=None,
        lat_key=None,
        lon_key=None,
        time_key=None,
        sentinel1_key=None,
        sentinel1_keys_filename=None,
        requests_filename='seviris.txt',
        channel=None,
        sensor_operational_mode='IW',
        data='SEVIRIS',
        max_timedelta=90,
        time_step=15,
        create_gif=False,
        verbose=None,
        delta_factor=None,
        continue_on_error=False,
        output_folder="outputs"
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

            log_print("Retrieve products", 2, verbose)
            platform, products, closest_product = get_closest_products(polygon, requested_date, max_timedelta,
                                                                       time_step)

            log_print(f"Read closest data from {platform}", 2, verbose)
            closest_filename = download_product(closest_product)
            platform_lat, platform_lon, closest_file_data = read_products([closest_filename])

            log_print("Project on S1 lat/lon grid", 2, verbose)
            closest_file_data = reproject(platform, closest_file_data, platform_lat, platform_lon, projection_lats,
                                          projection_lons)
            data = {f"data": closest_file_data, 'lats': projection_lats, 'lons': projection_lons}

            new_filename = f'{output_folder}/{filename}/{filename}_{platform.split(":")[1]}'
            new_filename = save_reprojection(platform, channel, data, new_filename)
            log_print(f"Saved in `{new_filename}`", 2, verbose)

            if create_gif:
                log_print(".gif generation is asked", 2, verbose)
                for date in products[platform]:
                    products[platform][date] = [download_product(products[platform][date])]
                gif_filename = f'outputs/{filename}/{filename}_{platform.split(":")[1]}.gif'
                generate_gif(
                    polygon, channel, products, gif_filename, verbose, read_products, delta_factor=delta_factor,
                    download_asked=False
                )
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
