import os
import numpy as np

from urllib import request
from functools import lru_cache
from xml.etree import ElementTree
from datetime import datetime, timedelta

from utils.requests import download_files
from utils.projection import get_distance
from utils.read import read_from_files_per_platform

from check_args import GOES_SERIE, HIMAWARI_SERIE, NEXRAD_BASIS, SATELLITE_PLATFORMS, ERA5_PLATFORMS
from check_args import ABI_CHANNELS, RRQPEF_CHANNELS, NEXRAD_CHANNELS, GLM_CHANNELS

def get_bucket_url(platform, channel, date):
    if platform in ERA5_PLATFORMS:
        url_basis = "https://era5-pds.s3.amazonaws.com/"
        prefix = f"{date.year}/{date.strftime('%m')}/data/{channel}"
    else:
        url_basis = f"https://noaa-{platform}.s3.amazonaws.com/"
        if platform in HIMAWARI_SERIE:
            date_string = f"{date.year}/{date.strftime('%m')}/{date.day}/{date.hour:02}{int(date.minute/10)}0"
            if channel in ABI_CHANNELS:
                prefix = f"AHI-L2-FLDK-ISatSS/{date_string}/OR_HFD-020-B12-M1{channel}"
            elif channel in RRQPEF_CHANNELS:
                prefix = f"AHI-L2-FLDK-RainfallRate/{date_string}"
        elif platform in GOES_SERIE:
            date_string = f"{date.year}/{date.strftime('%j')}/{date.hour:02}"
            if channel in ABI_CHANNELS:
                prefix = f"ABI-L2-MCMIPF/{date_string}"
            elif channel in RRQPEF_CHANNELS:
                prefix = f"ABI-L2-RRQPEF/{date_string}"
            elif channel in GLM_CHANNELS:
                prefix = f"GLM-L2-LCFA/{date_string}"
        elif platform in NEXRAD_BASIS:
            prefix = f"{date.year}/{date.month:02}/{date.day:02}/{channel}"
    return url_basis, url_basis + "?prefix=" + prefix


def get_bucket_urls(channel, iw_datetime, max_timedelta, time_step, platforms):
    time_steps = range(-max_timedelta, max_timedelta+1, time_step)
    dates = [iw_datetime + timedelta(minutes=x) for x in time_steps]
    
    urls = {}
    url_basis = {}
    for platform in platforms:
        urls[platform] = {}
        for date in dates:
            url_basis[platform], urls[platform][date] = get_bucket_url(platform, channel, date)
    return url_basis, urls


@lru_cache(maxsize=2**16)
def bucket_to_urls(bucket_url):
    urls = []
    req = request.urlopen(bucket_url)
    tree = ElementTree.parse(req)
    for elem in tree.iter():
        if elem.tag.endswith('Key'):
            urls.append(elem.text)
    return urls


def get_file_urls(channel, iw_datetime, bucket_urls_per_platform, time_step, url_basis):
    urls_per_platform = {}
    closest_datetime = {}

    time_step = timedelta(minutes=time_step)
    for platform in bucket_urls_per_platform:
        urls_per_platform[platform] = {}
        platform_basis = url_basis[platform]

        for date, bucket_url in bucket_urls_per_platform[platform].items():
            urls = bucket_to_urls(bucket_url)

            smallest_timedelta = None
            closest_urls = {}
            for i_url, url in enumerate(urls):
                try:
                    if platform in HIMAWARI_SERIE and channel in RRQPEF_CHANNELS:
                        date_string = url.split('_')[-2][1:-1]
                        url_datetime = datetime.strptime(date_string, '%Y%m%d%H%M%S')
                    elif platform in GOES_SERIE and channel in RRQPEF_CHANNELS:
                        date_string = url.split('_')[-3][1:-1]
                        url_datetime = datetime.strptime(date_string, '%Y%j%H%M%S')
                    elif platform in SATELLITE_PLATFORMS and channel in ABI_CHANNELS + GLM_CHANNELS:
                        date_string = url.split('_')[-2][1:-1]
                        url_datetime = datetime.strptime(date_string, '%Y%j%H%M%S')
                    elif platform in NEXRAD_BASIS:
                        if not url.endswith('_V06'): continue
                        date_string = os.path.split(url)[1][4:-4]
                        url_datetime = datetime.strptime(date_string, '%Y%m%d_%H%M%S')
                    elif platform in ERA5_PLATFORMS:
                        url_datetime = datetime(year=int(url[:4]), month=int(url[5:7]), day=iw_datetime.day)
                except ValueError: continue  # it means that some unsupported file was in the list
                        
                current_timedelta = abs(url_datetime - date)
                
                closest_urls[current_timedelta] = closest_urls.get(current_timedelta, []) + [platform_basis + url]
                urls_per_platform[platform][date] = urls_per_platform[platform].get(date, []) + [platform_basis + url]
                if smallest_timedelta is None or current_timedelta < smallest_timedelta:
                    smallest_timedelta = current_timedelta
            if smallest_timedelta is not None:
                if channel in ABI_CHANNELS + RRQPEF_CHANNELS + NEXRAD_CHANNELS:
                    urls_per_platform[platform][date] = closest_urls[smallest_timedelta]
                   
    urls_per_platform = {
        date: urls
        for date, urls in urls_per_platform.items()
        if urls
    }
    return urls_per_platform


def get_closest_platform(closest_filenames_per_platform, iw_polygon, channel, requested_date=None):
    mean_iw_lat = np.mean(iw_polygon[:,1])
    mean_iw_lon = np.mean(iw_polygon[:,0])
 
    closest_platform = None

    res = {}
    for platform, filenames in closest_filenames_per_platform.items():
        platform_lat, platform_lon, data = read_from_files_per_platform(filenames, platform, channel, requested_date=requested_date)
        res[platform] = platform_lat, platform_lon, data

        mean_platform_lat = np.nanmean(platform_lat)
        mean_platform_lon = np.nanmean(platform_lon)

        distance_to_center = get_distance(mean_iw_lat, mean_iw_lon, mean_platform_lat, mean_platform_lon)
        if closest_platform is None or distance_to_center < smallest_distance:
            smallest_distance = distance_to_center
            closest_platform = platform
    return closest_platform, res[closest_platform]


def get_closest_nexrad_station(polygon, blacklist=['KVTX']):
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
                if station_id in blacklist: continue
                lat, lon = line[3].split('/')
                
                lat = dms2dd(lat[:2], lat[2:4], lat[4:6], 'N')
                lon = dms2dd(lon[1:4], lon[4:6], lon[6:8], 'W')
                
                nexrad_stations[station_id] = {"lat": lat, "lon": lon}
        return nexrad_stations
    
    nexrad_stations = get_nexrad_stations()
    
    mean_iw_lat = np.mean(polygon[:,1])
    mean_iw_lon = np.mean(polygon[:,0])
    closest_station_distance = np.inf
    for station, latlon in nexrad_stations.items():
        station_distance = get_distance(mean_iw_lat, mean_iw_lon, latlon['lat'], latlon['lon'])
        if station_distance < closest_station_distance:
            closest_station_distance = station_distance
            closest_station = station
            
    return closest_station

        
def get_closest_filenames(channel, iw_polygon, iw_datetime, max_timedelta, time_step, platforms):
    if platforms == NEXRAD_BASIS:
        channel = get_closest_nexrad_station(iw_polygon)
    
    url_basis, bucket_urls_per_platform = get_bucket_urls(channel, iw_datetime, max_timedelta=max_timedelta, time_step=time_step, platforms=platforms)
    urls_per_platforms = get_file_urls(channel, iw_datetime, bucket_urls_per_platform, time_step, url_basis)
    closest_filenames_per_platform = download_files(urls_per_platforms, closest=True)
    
    for plaftorm, datedic in closest_filenames_per_platform.items():
        closest_date = sorted([(abs(iw_datetime - date), date) for date in datedic])[0][1]
        closest_filenames_per_platform[plaftorm] = {closest_date: datedic[closest_date]}

    closest_platform, (platform_lat, platform_lon, data) = get_closest_platform(closest_filenames_per_platform, iw_polygon, channel, requested_date=iw_datetime)
    urls_per_platforms = {key: value for key, value in urls_per_platforms.items() if key == closest_platform}
    return channel, closest_platform, urls_per_platforms, (platform_lat, platform_lon, data)
