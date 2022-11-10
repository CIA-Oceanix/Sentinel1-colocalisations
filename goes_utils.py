import numpy as np
from datetime import datetime, timedelta
from xml.etree import ElementTree

from urllib import request
from utils import log_print, get_distance, grid_from_polygon
from requests_utils import download_file

channel = 'GLM-L2-LCFA'

def get_goes_hour_urls(channel, iw_datetime, delta_minutes=90, verbose=1):
    if verbose: log_print(f'Retrieving urls')

    ts = [iw_datetime + timedelta(minutes=x) for x in range(-delta_minutes, delta_minutes+1, 60)]
    
    urls = {}
    for satellite in ('goes16', 'goes17', 'goes18'):
        url_basis = f"https://noaa-{satellite}.s3.amazonaws.com/?list-type=2"

        for t in ts:
            url = url_basis + f"&prefix={channel}%2F{t.year}%2F{t.strftime('%j')}%2F{t.hour:02}"
            urls[satellite] = urls.get(satellite, []) + [url] 
    return urls
    
def get_close_urls(iw_datetime, hour_urls_per_satellite, delta_minutes=90):
    maximum_delta = timedelta(minutes=delta_minutes)

    close_urls = {}
    closest_url = {}
    smallest_timedelta = {}
    for satellite, hour_urls in hour_urls_per_satellite.items():
        for hour_url in hour_urls:
            req = request.urlopen(hour_url)
            tree = ElementTree.parse(req)

            for elem in tree.iter():
                if elem.tag.endswith('Key'):
                    key = elem.text
                    key_datetime = datetime.strptime(elem.text.split('_')[3][1:-1], '%Y%j%H%M%S')
                    current_timedelta = abs(key_datetime - iw_datetime)
                    
                    url = f"https://noaa-{satellite}.s3.amazonaws.com/{elem.text}"
                    if current_timedelta < maximum_delta:
                        close_urls[satellite] =  close_urls.get(satellite, []) + [url]
                    if satellite not in smallest_timedelta or current_timedelta < smallest_timedelta[satellite]:
                        closest_url[satellite] = url
                        smallest_timedelta[satellite] = current_timedelta
    return close_urls, closest_url


def increased_grid(polygon, km_per_pixel=1, delta_factor=1):
    min_lat = np.min(polygon[:,1])
    max_lat = np.max(polygon[:,1])

    min_lon = np.min(polygon[:,0])
    max_lon = np.max(polygon[:,0])

    delta_lon = max_lon - min_lon
    delta_lat = max_lat - min_lat

    frame_min_lat = min_lat - delta_lat*delta_factor
    frame_max_lat = max_lat + delta_lat*delta_factor
    frame_min_lon = min_lon - delta_lon*delta_factor
    frame_max_lon = max_lon + delta_lon*delta_factor

    height = int(get_distance(frame_min_lat, frame_min_lon, frame_max_lat, frame_min_lon)/km_per_pixel)
    width =  int(max(
        get_distance(frame_min_lat, frame_min_lon, frame_min_lat, frame_max_lon),
        get_distance(frame_max_lat, frame_min_lon, frame_max_lat, frame_max_lon),
    )/km_per_pixel)
        
    frame_polygon = np.array(
        [
            [frame_min_lon, frame_min_lat],
            [frame_min_lon, frame_max_lat],
            [frame_max_lon, frame_max_lat],
            [frame_max_lon, frame_min_lat]
        ]
    )

    return grid_from_polygon(frame_polygon, (height, width))
