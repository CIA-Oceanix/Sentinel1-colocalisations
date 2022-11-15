import os
import numpy as np

from urllib import request
from functools import lru_cache
from xml.etree import ElementTree
from datetime import datetime, timedelta

from utils.requests import download_files
from utils.projection import get_distance
from utils.read import read_from_files_per_platform

from check_args import GOES_SERIE, HIMAWARI_SERIE, NEXRAD_BASIS, SATELLITE_PLATFORMS

def get_bucket_url(platform, channel, date):
    def himawari_bucket_date(platform, channel, date):
        return f"AHI-L2-FLDK-ISatSS/{date.year}/{date.strftime('%m')}/{date.day}/{date.hour:02}{int(date.minute/10)}0/OR_HFD-020-B12-M1{channel}"

    def goes_bucket_date(platform, channel, date):
        return f"ABI-L2-MCMIPF/{date.year}/{date.strftime('%j')}/{date.hour:02}"

    def nexrad_bucket_date(platform, channel, date):
        return f"{date.year}/{date.month:02}/{date.day:02}/{channel}"
        
    url = f"https://noaa-{platform}.s3.amazonaws.com/?prefix="
    if platform in HIMAWARI_SERIE:
        function = himawari_bucket_date
    elif platform in GOES_SERIE:
        function = goes_bucket_date
    elif platform in NEXRAD_BASIS:
        function = nexrad_bucket_date
    url += function(platform, channel, date)
    return url


def get_bucket_urls(channel, iw_datetime, max_timedelta, time_step, platforms):
    time_steps = range(-max_timedelta, max_timedelta+1, time_step)
    dates = [iw_datetime + timedelta(minutes=x) for x in time_steps]
    
    urls = {}
    for platform in platforms:
        urls[platform] = {}
        for date in dates:
            urls[platform][date] = get_bucket_url(platform, channel, date)
    return urls


@lru_cache(maxsize=2**16)
def bucket_to_urls(bucket_url):
    urls = []
    req = request.urlopen(bucket_url)
    tree = ElementTree.parse(req)
    for elem in tree.iter():
        if elem.tag.endswith('Key'):
            urls.append(elem.text)
    return urls


def get_file_urls(channel, iw_datetime, bucket_urls_per_platform):
    urls_per_platform = {}
    closest_datetime = {}
    for platform in bucket_urls_per_platform:
        urls_per_platform[platform] = {}
        url_base = f"https://noaa-{platform}.s3.amazonaws.com/"

        for date, bucket_url in bucket_urls_per_platform[platform].items():
            urls = bucket_to_urls(bucket_url)

            closest_urls = {}
            for url in urls:
                if platform in SATELLITE_PLATFORMS:
                    date_string = url.split('_')[-2][1:-1]
                    url_datetime = datetime.strptime(date_string, '%Y%j%H%M%S')
                elif platform in NEXRAD_BASIS:
                    if not url.endswith('_V06'): continue
                    date_string = os.path.split(url)[1][4:-4]
                    url_datetime = datetime.strptime(date_string, '%Y%m%d_%H%M%S')
                    

                current_timedelta = abs(url_datetime - date)
                if (not closest_urls) or smallest_timedelta >= current_timedelta:
                    closest_urls[current_timedelta] = closest_urls.get(current_timedelta, []) + [url]
                    smallest_timedelta = current_timedelta
            if closest_urls:
                urls_per_platform[platform][date] = [url_base + url for url in closest_urls[smallest_timedelta]]
                
    urls_per_platform = {
        key: value
        for key, value in urls_per_platform.items()
        if value
    }
    return urls_per_platform


def get_closest_platform(closest_filenames_per_platform, iw_polygon, channel):
    mean_iw_lat = np.mean(iw_polygon[:,1])
    mean_iw_lon = np.mean(iw_polygon[:,0])
            
    closest_platform = None

    res = {}
    for platform, filenames in closest_filenames_per_platform.items():
        platform_lat, platform_lon, data = read_from_files_per_platform(filenames, platform, channel)
        res[platform] = platform_lat, platform_lon, data
            
        mean_platform_lat = np.mean(platform_lat)
        mean_platform_lon = np.mean(platform_lon)
            
        distance_to_center = get_distance(mean_iw_lat, mean_iw_lon, mean_platform_lat, mean_platform_lon)
        if closest_platform is None or distance_to_center < smallest_distance:
            smallest_distance = distance_to_center
            closest_platform = platform
    return closest_platform, res[closest_platform]


def get_closest_nexrad_station(polygon):
    def get_nexrad_stations():
        def dms2dd(degrees, minutes, seconds, direction):
            dd = float(degrees) + float(minutes)/60 + float(seconds)/(60*60);
            if direction == 'W' or direction == 'S':
                dd *= -1
            return dd
            
        nexrad_stations = {}
        with open('res/nexrad_stations.txt', 'r') as file:
            for line in file.readlines()[1:]:
                line = line.split('\t')
                    
                station_id = line[1]
                lat, lon = line[3].split('/')
                
                lat = dms2dd(lat[:2], lat[2:4], lat[4:6], 'N')
                lon = dms2dd(lon[1:4], lon[4:6], lon[6:8], 'W')
                        
                nexrad_stations[station_id] = {"lat": lat, "lon": lon}
        return nexrad_stations
    
    nexrad_stations = get_nexrad_stations()
    
    mean_iw_lat = np.mean(polygon[:,1])
    mean_iw_lon = np.mean(polygon[:,0])
    closest_station_distance = np.inf
    for station, latlon in nexrad_stations.items(): # could be parallelized with np, but too lazy
        station_distance = get_distance(mean_iw_lat, mean_iw_lon, latlon['lat'], latlon['lon'])
        if station_distance < closest_station_distance:
            closest_station_distance = station_distance
            closest_station = station
            
    return closest_station

        
def get_closest_filenames(channel, iw_polygon, iw_datetime, max_timedelta, time_step, platforms):
    if platforms == NEXRAD_BASIS:
        channel = get_closest_nexrad_station(iw_polygon)
    
    bucket_urls_per_platform = get_bucket_urls(channel, iw_datetime, max_timedelta=max_timedelta, time_step=time_step, platforms=platforms)
    urls_per_platforms = get_file_urls(channel, iw_datetime, bucket_urls_per_platform)

    closest_filenames_per_platform = download_files(urls_per_platforms, closest=True)
    closest_filenames_per_platform = {key: value[0] for key, value in closest_filenames_per_platform.items()}

    closest_platform, (platform_lat, platform_lon, data) = get_closest_platform(closest_filenames_per_platform, iw_polygon, channel)
    urls_per_platforms = {key: value for key, value in urls_per_platforms.items() if key == closest_platform}
    return closest_platform, urls_per_platforms, (platform_lat, platform_lon, data)
