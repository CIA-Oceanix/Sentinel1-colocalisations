import os
import sys
import fire
import shutil
import zipfile
from datetime import datetime, timedelta

import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

from check_args import get_keys
from utils.misc import log_print
from utils.rt import safe_to_tiff
from utils.deep_learning import apply_on_keys
from utils.sentinel1 import getter_polygon_from_key

#shutil.rmtree('.temp', ignore_errors=True)
os.makedirs('.temp', exist_ok=True)
os.makedirs('outputs', exist_ok=True)


def download_grd_from_asf(filename, username, password, folder=".temp"):
    new_filename = os.path.join(folder, filename)
    if not os.path.exists(new_filename):
        url = f"https://sentinel1.asf.alaska.edu/GRD_HD/S{filename[2]}/{filename}"
        with requests.Session() as session:
            session.auth = (username, password)
            request = session.request('get', url)
            raw = session.get(request.url, auth=session.auth, stream=True).raw
            with open(new_filename, 'wb') as file:
                shutil.copyfileobj(raw, file)
    return new_filename
        

def main(asf_username, asf_password, key, model, verbose=1):
    keys = get_keys(key)
    
    if verbose: log_print(f"Build IW getter")
    getter = getter_polygon_from_key('IW')
    

    if verbose: log_print("Search the zipnames using the getter")
    zipnames = [getter(key)[0] + '.zip' for key in keys]

    if verbose: log_print("Download the zipnames using ASF")
    zip_filenames = [download_grd_from_asf(zipname, asf_username, asf_password) for zipname in zipnames]

    if verbose: log_print("Unzip")
    unzipped_folders = []
    for filename in zip_filenames:
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(".temp")
        unzipped_folders.append(filename.replace('.zip', '.SAFE'))

    if verbose: log_print("Generate the .tiff")
    tiff_filenames = []
    for folder in unzipped_folders:
        tiff_filenames += safe_to_tiff(folder, vh=False)

    if verbose: log_print("Run the models")
    deep_learning_outputs = apply_on_keys(tiff_filenames, getter, model)
    for filename in tiff_filenames:
        key = os.path.split(filename)[1].split('-')[4]
        new_filename = f"outputs/{key}.tiff"
        shutil.copyfile(filename, new_filename)

    if verbose: log_print("Done")

    
if __name__ == '__main__':
    fire.Fire(main)
