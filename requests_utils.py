import requests
import time
import threading
from threading import Thread
import shutil
import os
from datetime import datetime

from utils import log_print, r_print

HEADERS = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko)'
                         ' Chrome/23.0.1271.64 Safari/537.11'
}

def request_stream(url, headers=HEADERS, retry=5):
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
    filename = os.path.join(folder, os.path.split(url)[1])
    if os.path.exists(filename): return
    
    raw = request_stream(url)
    with open(filename, 'wb') as file:
        shutil.copyfileobj(raw, file)
    return filename


def routing(args, thread_limit=20, single_wait=0, verbose=1, time_limit=10, time_sleep=10):
    thread_limit += threading.active_count()
    begin = datetime.now()
    
    p = 0.0
    threads = []
    n = len(args)
    try:
        for i, arg in enumerate(args):
            while threading.active_count() > thread_limit:
                time.sleep(0.01)
            threads.append(Thread(target=download_file, args=arg))
            threads[-1].start()
            
            if verbose and p != int(i / n * 1000):
                log_print(f'Downloading {i+1}/{n}', f=r_print)
                
                p = int(i / n * 1000)
            time.sleep(single_wait)
        time.sleep(time_sleep)
    except Exception as e:
        raise e
    if verbose: print()

    threads = [thread.join(timeout=time_limit) for thread in threads]
    return