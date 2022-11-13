import shutup; shutup.please()

from datetime import datetime, timedelta 
from matplotlib.patches import Polygon

from urllib import request
from xml.etree import ElementTree
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
np.seterr(all="ignore")

import os
import PIL.Image
import sys
import fire
from netCDF4 import Dataset
import shutil

from scipy.interpolate import griddata

shutil.rmtree('.temp', ignore_errors=True)
os.makedirs('.temp', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

matplotlib.use('agg')

from utils import get_distance, grid_from_polygon, getter_polygon_from_key, ini_map, png_to_gif
from utils import log_print, r_print, get_iw_latlon, plot_polygon
from requests_utils import routing, download_file

from goes_utils import get_goes_hour_urls, get_close_urls, increased_grid


def maps_from_iw_key(key, delta_minutes=90, verbose=1, shape=None, metadata_filename=None, sensoroperationalmode='IW', generate_gif=True):
    def latlon_from_goes_dataset(dataset):
        x_coordinate_1d = dataset['x'][:]  # E/W scanning angle in radians
        y_coordinate_1d = dataset['y'][:]  # N/S elevation angle in radians
        projection_info = dataset['goes_imager_projection']
        lon_origin = projection_info.longitude_of_projection_origin
        H = projection_info.perspective_point_height+projection_info.semi_major_axis
        r_eq = projection_info.semi_major_axis
        r_pol = projection_info.semi_minor_axis

        # Create 2D coordinate matrices from 1D coordinate vectors
        x_coordinate_2d, y_coordinate_2d = np.meshgrid(x_coordinate_1d, y_coordinate_1d)

        # Equations to calculate latitude and longitude
        lambda_0 = (lon_origin*np.pi)/180.0  
        a_var = np.power(np.sin(x_coordinate_2d),2.0) + (np.power(np.cos(x_coordinate_2d),2.0)*(np.power(np.cos(y_coordinate_2d),2.0)+(((r_eq*r_eq)/(r_pol*r_pol))*np.power(np.sin(y_coordinate_2d),2.0))))
        b_var = -2.0*H*np.cos(x_coordinate_2d)*np.cos(y_coordinate_2d)
        c_var = (H**2.0)-(r_eq**2.0)
        r_s = (-1.0*b_var - np.sqrt((b_var**2)-(4.0*a_var*c_var)))/(2.0*a_var)
        s_x = r_s*np.cos(x_coordinate_2d)*np.cos(y_coordinate_2d)
        s_y = - r_s*np.sin(x_coordinate_2d)
        s_z = r_s*np.cos(x_coordinate_2d)*np.sin(y_coordinate_2d)

        # Ignore numpy errors for sqrt of negative number; occurs for GOES-16 ABI CONUS sector data
        np.seterr(all='ignore')

        abi_lat = (180.0/np.pi)*(np.arctan(((r_eq*r_eq)/(r_pol*r_pol))*((s_z/np.sqrt(((H-s_x)*(H-s_x))+(s_y*s_y))))))
        abi_lon = (lambda_0 - np.arctan(s_y/(H-s_x)))*(180.0/np.pi)
        return abi_lat, abi_lon

    def get_closest_satellite():
        mean_iw_lat = np.mean(polygon[:,1])
        mean_iw_lon = np.mean(polygon[:,0])
                
        closest_satellite = None
        for satellite, url in closest_url.items():
            folder = ".temp/" + satellite
            os.makedirs(folder, exist_ok=True)
            download_file(url, folder)
            
            with Dataset(os.path.join(folder, os.listdir(folder)[0])) as dataset:
                lats, lons = latlon_from_goes_dataset(dataset)
                
            mean_satellite_lat = np.mean(lats)
            mean_satellite_lon = np.mean(lons)
                
            distance_to_center = get_distance(mean_iw_lat, mean_iw_lon, mean_satellite_lat, mean_satellite_lon)
            if closest_satellite is None or distance_to_center < smallest_distance:
                smallest_distance = distance_to_center
                closest_satellite = satellite
        return closest_satellite

    def read_nc(filename):
        def arg2d(array, f=np.argmin):
            return  np.unravel_index(f(array), array.shape)
            
        with Dataset(filename) as dataset:
            data = dataset['RRQPE'][:]

            lats, lons = latlon_from_goes_dataset(dataset)
            
        x1, y1 = arg2d(get_distance(lats, lons, lat_grid.max(), lon_grid.max()))
        x2, y2 = arg2d(get_distance(lats, lons, lat_grid.min(), lon_grid.min()))
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        lats = lats[x1:x2,y1:y2].filled(0)
        lons = lons[x1:x2,y1:y2].filled(0)
        data = data[x1:x2,y1:y2]
                
        return data, lats, lons


    def nc_to_png(filename, lat_grid, lon_grid, m=None, polygon=None):
        suptitle = datetime.strptime(filename.split('_')[-3][1:-1], '%Y%j%H%M%S').strftime('%Y-%m-%d %H:%M:%S')
        data, lats, lons = read_nc(filename)

        if m is None:
            plt.figure(figsize=(12,12))
            m = ini_map(lat_grid, lon_grid)
            
        plot_polygon(polygon, m)
        
        colormesh = m.pcolormesh(lons, lats, data, latlon=True, cmap='turbo', vmin=0, vmax=50, shading='auto')
        colorbar = plt.colorbar(fraction=0.046, pad=0.04, orientation='horizontal')
        plt.suptitle(suptitle)
        plt.title(f"AHI Rainfall Rate [mm/h]")
        plt.tight_layout()

        new_filename = filename + ".RRQPE.png"
        plt.savefig(new_filename)

        colorbar.remove()
        colormesh.remove()
        return new_filename, m

        
    def project_rrqpe_iw_lat_lon(filename, shape):
        data, lats, lons = read_nc(filename)

        new_data = griddata(
            np.stack((lats.flatten(), lons.flatten()), axis=1),
            data.flatten(),
            np.stack((owiLat.flatten(), owiLon.flatten()), axis=1)
        ).reshape(shape).astype('float')

        np.savez_compressed(f'outputs/{key}/{key}_rrqpe.npz', new_data)

        new_data = np.clip(new_data, 0, 40)/40
        new_data = plt.get_cmap('turbo')(new_data)
        new_data = (new_data * 255).astype(np.uint8)

        new_filename = f'outputs/{key}/{key}_rrqpe.png'
        PIL.Image.fromarray(new_data).save(new_filename)
        return new_filename
    
    start = datetime.now()
    
    if verbose: log_print("Loading IW metadata")
    getter = getter_polygon_from_key(sensoroperationalmode=sensoroperationalmode)
    if verbose: log_print("Loading OK")

    key = key.lower()
    iw_datetime = datetime.strptime(key, '%Y%m%dt%H%M%S')
    iw_filename, polygon, orbit = getter(key)

    if verbose: log_print("Choose satellite")
    hour_urls_per_satellite = get_goes_hour_urls('ABI-L2-RRQPEF', iw_datetime, delta_minutes=delta_minutes, verbose=verbose)
    close_urls, closest_url = get_close_urls(iw_datetime, hour_urls_per_satellite, delta_minutes=delta_minutes)
    closest_satellite = get_closest_satellite()
    close_urls = close_urls[closest_satellite]
    closest_url = closest_url[closest_satellite]
    if verbose: log_print(f"Closest satellite: {closest_satellite}")

    
    lat_grid, lon_grid = increased_grid(polygon, km_per_pixel=2, delta_factor=2)

    os.makedirs('outputs/' + key, exist_ok=True)
    if verbose: log_print('Reproject in .npz and .png')
    download_file
    owiLat, owiLon = get_iw_latlon(polygon=polygon, metadata_filename=metadata_filename, shape=shape)
    if owiLat is not None:
        new_filename = project_rrqpe_iw_lat_lon(f".temp/{closest_satellite}/{os.path.split(closest_url)[1]}", owiLat.shape)
        if verbose: log_print('Projection done')
    else: log_print('Projection failed')

    if generate_gif:
        routing_args = [(url, ".temp/") for url in close_urls]
        routing(routing_args)
        filenames = [f".temp/{os.path.split(url)[1]}" for url in close_urls]
        
        m = None
        for i, filename in enumerate(filenames):
            if verbose: log_print(f'Generating {i+1}/{len(filenames)} .png', f=r_print if i else print)
            filenames[i], m = nc_to_png(filename,  lat_grid, lon_grid, polygon=polygon, m=m)
        if verbose: print()
        
        if verbose: log_print('Generating .gif')
        png_to_gif(filenames, f'outputs/{key}/{key}_rrqpe.gif')
    
    if verbose: print('Executed in', (datetime(1,1,1) + (datetime.now()-start)).strftime('%H hours, %M minutes, %S seconds'))

if __name__ == "__main__":
    fire.Fire(maps_from_iw_key)
