import shutup;

shutup.please()

import os
import fire
import zipfile
import requests
import numpy as np

np.seterr(all="ignore")

from netCDF4 import Dataset
from scipy.interpolate import griddata
from datetime import datetime, timedelta

import matplotlib

matplotlib.use('agg')

from utils.misc import log_print
from utils.requests import routing
from utils.projection import increased_grid, save_reprojection, trim
from utils.map import plot_on_map
from utils.sentinel1 import getter_polygon_from_key, get_iw_latlon

import logins

from check_args import get_keys

# shutil.rmtree('.temp', ignore_errors=True)
os.makedirs('.temp', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

PLATFORM = "Sentinel3"
CHANNEL = "CHL_OC4ME"


def get_token():
    url = 'https://identity.cloudferro.com/auth/realms/DIAS/protocol/openid-connect/token'
    payload = {
        "username": logins.SENTINEL_USERNAME,
        "password": logins.SENTINEL_PASSWORD,
        "client_id": "CLOUDFERRO_PUBLIC",
        "grant_type": "password"
    }

    r = requests.post(url, data=payload)
    return r.json()['access_token']


def get_download_args(key, polygon):
    start_date = (datetime.strptime(key, '%Y%m%dt%H%M%S') - timedelta(hours=24)).strftime('%Y-%m-%dT%H:%M:%SZ').replace(
        ':', '%3A')
    completion_date = (datetime.strptime(key, '%Y%m%dt%H%M%S') + timedelta(hours=24)).strftime(
        '%Y-%m-%dT%H:%M:%SZ').replace(':', '%3A')

    geometry = f"POLYGON(({polygon[0, 0]}+{polygon[0, 1]}%2C{polygon[1, 0]}+{polygon[1, 1]}%2C{polygon[2, 0]}+" \
               + f"{polygon[2, 1]}%2C{polygon[3, 0]}+{polygon[3, 1]}%2C{polygon[0, 0]}+{polygon[0, 1]}))"
    url = "https://finder.creodias.eu/resto/api/collections/Sentinel3/search.json?maxRecords=10&" \
          + f"startDate={start_date}&completionDate={completion_date}&instrument=OL&processingLevel=LEVEL2&productType=WFR&" \
          + f"geometry={geometry}&sortParam=startDate&sortOrder=descending&status=all&dataset=ESA-DATASET"

    r = requests.get(url)
    products = r.json()['features']

    args = []
    for product in products:
        args.append((
            os.path.split(product['properties']['productIdentifier'])[1] + '.zip',
            product['properties']['services']['download']['url'] + '?token=' + get_token()
        ))

    return args


def unzip(filename):
    root, short_filename = os.path.split(filename)
    key = short_filename.split('_')[7]

    new_folder = os.path.join(root, key)
    os.makedirs(new_folder, exist_ok=True)

    try:
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extract(short_filename[:-4] + "/chl_oc4me.nc", new_folder)
            zip_ref.extract(short_filename[:-4] + "/geo_coordinates.nc", new_folder)
    except zipfile.BadZipFile:
        print('BadZipFile on', short_filename)

    """try: os.remove(filename)
    except PermissionError: pass"""

    return os.path.join(new_folder, short_filename[:-4])


def draw_s3_on_map(folder, iw_polygon, lat_grid, lon_grid, m, filename):
    with Dataset(folder + "/chl_oc4me.nc") as dataset:
        data = dataset[CHANNEL]
        data = 10 ** (data[:] * data.scale_factor + data.add_offset)
        data = data.filled(np.nan)

    with Dataset(folder + "/geo_coordinates.nc") as dataset:
        platform_lat = dataset['latitude'][:]
        platform_lon = dataset['longitude'][:]

    m = plot_on_map(PLATFORM, CHANNEL, data, platform_lat, platform_lon, lat_grid, lon_grid, filename, m=m,
                    polygon=iw_polygon)
    return m, (platform_lat, platform_lon, data)


def project_s3_data(data, iw_polygon):
    owi_lat, owi_lon = get_iw_latlon(polygon=iw_polygon)

    new_data = np.empty(owi_lat.shape)
    new_data.fill(np.nan)

    indexes = [i for (delta, i) in sorted([[delta, i] for i, (delta, _, _, _) in enumerate(data)])]
    for i in indexes[::-1]:
        platform_lat, platform_lon, partial_data = data[i][1:]
        partial_data, platform_lat, platform_lon = trim(partial_data, platform_lat, platform_lon, owi_lat, owi_lon)

        new_partial_data = griddata(
            np.stack((platform_lat.flatten(), platform_lon.flatten()), axis=1),
            partial_data.flatten(),
            np.stack((owi_lat.flatten(), owi_lon.flatten()), axis=1)
        ).reshape(owi_lat.shape).astype('float')
        new_data[~np.isnan(new_partial_data)] = new_partial_data[~np.isnan(new_partial_data)]
    return new_data


def main(key, verbose=1, sensor_operational_mode="IW"):
    keys = get_keys(key)

    log_print(f"Build {sensor_operational_mode} getter", 2, verbose)
    getter = getter_polygon_from_key(sensor_operational_mode)

    for i, key in enumerate(keys):
        log_print(f"Request {i + 1}/{len(keys)}: {key}", 1, verbose)

        key = key.lower()
        iw_polygon = getter(key)[1]
        s1_time = datetime.strptime(key, '%Y%m%dt%H%M%S')

        log_print(f"Retrieve Sentinel3 collocations", 2, verbose)
        collocations = get_download_args(key, iw_polygon)
        lat_grid, lon_grid = increased_grid(iw_polygon, km_per_pixel=1, delta_factor=2)

        log_print(f"Download Sentinel3 collocations", 2, verbose)
        args = [(url, f".temp/{key}//{filename}") for filename, url in collocations]

        routing(args, thread_limit=4)

        log_print(f"Unzip Sentinel3 collocations", 2, verbose)
        new_folders = [unzip(filename) for url, filename in args]

        log_print(f"Draw Sentinel3 around the Sentinel1 observation", 2, verbose)
        m = None

        datas = []
        for folder in new_folders:
            s3_time = datetime.strptime(os.path.split(folder)[1].split('_')[7].lower(), '%Y%m%dt%H%M%S')
            filename = f"outputs/{os.path.split(folder[6:])[0]}.png"

            m, (platform_lat, platform_lon, data) = draw_s3_on_map(folder, iw_polygon, lat_grid, lon_grid, m, filename)
            datas.append((abs(s1_time - s3_time), platform_lat, platform_lon, data))

        log_print(f"Project Sentinel3 on Sentinel1 grid", 2, verbose)
        data = project_s3_data(datas, iw_polygon)
        save_reprojection(PLATFORM, CHANNEL, data, f"outputs/{key}/{key}_S3")
    log_print(f"Done", 1, verbose)


if __name__ == "__main__":
    fire.Fire(main)
