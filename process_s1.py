import os
import shutil
import zipfile

import fire
import requests

import logins
from check_args import get_keys
from utils.deep_learning import apply_on_keys
from utils.misc import log_print
from utils.rt import safe_to_tiff
from utils.sentinel1 import getter_polygon_from_key

#shutil.rmtree('.temp', ignore_errors=True)
os.makedirs('.temp', exist_ok=True)
os.makedirs('outputs', exist_ok=True)


def download_grd_from_asf(filename, folder=".temp"):
    new_filename = os.path.join(folder, filename)
    if not os.path.exists(new_filename):
        url = f"https://sentinel1.asf.alaska.edu/GRD_HD/S{filename[2]}/{filename}"
        with requests.Session() as session:
            session.auth = (logins.ASF_USERNAME, logins.ASF_PASSWORD)
            request = session.request('get', url)
            raw = session.get(request.url, auth=session.auth, stream=True).raw
            with open(new_filename, 'wb') as file:
                shutil.copyfileobj(raw, file)
    return new_filename


def main(key, model, verbose=1):
    keys = get_keys(key)

    log_print(f"Build IW getter", 2, verbose)
    getter = getter_polygon_from_key('IW')

    log_print("Search the zipnames using the getter", 2, verbose)
    zipnames = [getter(key)[0] + '.zip' for key in keys]

    log_print("Download the zipnames using ASF", 2, verbose)
    zip_filenames = [download_grd_from_asf(zipname) for zipname in zipnames]

    log_print("Unzip", 2, verbose)
    unzipped_folders = []
    for filename in zip_filenames:
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(".temp")
        unzipped_folders.append(filename.replace('.zip', '.SAFE'))

    log_print("Generate the .tiff", 2, verbose)
    tiff_filenames = []
    for folder in unzipped_folders:
        tiff_filenames += safe_to_tiff(folder, vh=False)

    log_print("Run the models", 2, verbose)
    apply_on_keys(tiff_filenames, getter, model)
    for filename in tiff_filenames:
        key = os.path.split(filename)[1].split('-')[4]
        new_filename = f"outputs/{key}.tiff"
        shutil.copyfile(filename, new_filename)

    log_print("Done", 1, verbose)


if __name__ == '__main__':
    fire.Fire(main)
