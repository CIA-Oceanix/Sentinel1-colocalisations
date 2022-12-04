import os
import shutil
import threading
import time
from threading import Thread

import requests

from utils.misc import log_print, r_print

HEADERS = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko)'
                         ' Chrome/23.0.1271.64 Safari/537.11'
           }


def request_stream(url, headers=HEADERS, retry=50):
    ok = 0
    while ok < retry:
        try:
            r = requests.get(url, stream=True, headers=headers)
            if r.status_code == 200:
                return r.raw
            elif r.status_code == 429:  # spam
                time.sleep(1)
        except Exception as e:
            print(url)
            raise e
        ok += 1


def download_file(url, folder=None):
    if folder is None:
        url, folder = url

    filename = os.path.join(folder, url[6:].replace('/', '_').split('?')[0])

    if os.path.exists(filename): return filename
    os.makedirs(os.path.split(filename)[0], exist_ok=True)

    if url.startswith('gs:'):
        os.system(f"gsutil cp {url} {folder}")
        os.rename(os.path.join(folder, os.path.split(url)[1]), filename)
    else:
        raw = request_stream(url)
        with open(filename, 'wb') as file:
            shutil.copyfileobj(raw, file)
    return filename


def routing(args, thread_limit=10, single_wait=0, verbose=1, time_limit=3600, time_sleep=10):
    args = list(set(args))
    thread_limit += threading.active_count()

    threads = []
    n = len(args)
    try:
        for i, arg in enumerate(args):
            while threading.active_count() > thread_limit:
                time.sleep(0.01)
            threads.append(Thread(target=download_file, args=arg))
            threads[-1].start()

            log_print(f'Downloading {i + 1}/{n}', 2, verbose, f=r_print)
            time.sleep(single_wait)
        time.sleep(time_sleep)
    except Exception as e:
        raise e
    if verbose > 1: print()

    [thread.join(timeout=time_limit) for thread in threads]


def download_files(urls_per_platforms, closest=False):
    routing_args = []
    filenames = {}
    for platform in urls_per_platforms:
        dates = list(urls_per_platforms[platform].keys())
        if closest:
            # Assume the closest_file is in the middle of the dic
            dates = [dates[int(len(dates) / 2)]]

        filenames[platform] = {}
        for date in dates:
            filenames[platform][date] = []
            folder = f".temp/{platform}/"
            for url in urls_per_platforms[platform][date]:
                filenames[platform][date].append(folder + url[6:].replace('/', '_'))
                routing_args.append((url, folder))

    routing(routing_args)
    return filenames
