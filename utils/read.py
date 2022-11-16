import numpy as np
from netCDF4 import Dataset

from check_args import GOES_SERIE, HIMAWARI_SERIE, NEXRAD_BASIS, ABI_CHANNELS, RRQPEF_CHANNELS, GLM_CHANNELS


def accumulate_lightning_maps(filenames):
    lat_grid = None
    
    data = np.zeros(lat_grid.shape)

    for filename in filenames:
        with Dataset(filename) as dataset:
            event_lat = dataset['event_lat'][:]
            event_lon = dataset['event_lon'][:]
            event_energy = dataset['event_energy']
            event_energy = (event_energy[:]*event_energy.scale_factor) + event_energy.add_offset

            if lat_grid is None:
                min_lat, max_lat = dataset['lat_field_of_view_bounds']
                min_lon, max_lon = dataset['lon_field_of_view_bounds']
                shape = ((max_lat-min_lat)//10, (max_lon-min_lon)//10)
                polygon = [[min_lat, min_lon], [max_lat, min_lon], [max_lat, max_lon], [min_lat, max_lat]]
                lat_grid, lon_grid = grid_from_polygon(polygon, shape)
                print(f"{lat_grid.max()=}, {lat_grid.max()=}, {lon_grid.min()=}, {lon_grid.max()=}")

        validity = np.logical_and(
            np.logical_and(min_lat < event_lat, event_lat < max_lat),
            np.logical_and(min_lon < event_lon, event_lon < max_lon)
        )
        
        event_lat = event_lat[validity]
        event_lon = event_lon[validity]
        event_energy = event_energy[validity]
        
        event_lat = ((event_lat - min_lat)/(max_lat - min_lat)*data.shape[1]).astype(int)
        event_lon = ((event_lon - min_lon)/(max_lon - min_lon)*data.shape[0]).astype(int)
        np.add.at(data, (event_lon, event_lat), event_energy)
        
    data[data==0] = np.nan
    return data, lat_grid, lon_grid

    
def read_from_files_per_platform(filenames, platform, channel):
    def latlon_from_file(filename, platform, channel):
        def latlon_from_abi_file(x, y, lon_origin, H, r_eq, r_pol):
            x, y = np.meshgrid(x, y)

            lambda_0 = (lon_origin*np.pi)/180.0  
            a_var = np.power(np.sin(x),2.0) + (np.power(np.cos(x),2.0)*(np.power(np.cos(y),2.0)+(((r_eq*r_eq)/(r_pol*r_pol))*np.power(np.sin(y),2.0))))
            b_var = -2.0*H*np.cos(x)*np.cos(y)
            c_var = (H**2.0)-(r_eq**2.0)
            r_s = (-1.0*b_var - np.sqrt((b_var**2)-(4.0*a_var*c_var)))/(2.0*a_var)
            s_x = r_s*np.cos(x)*np.cos(y)
            s_y = - r_s*np.sin(x)
            s_z = r_s*np.cos(x)*np.sin(y)

            platform_lat = (180.0/np.pi)*(np.arctan(((r_eq*r_eq)/(r_pol*r_pol))*((s_z/np.sqrt(((H-s_x)*(H-s_x))+(s_y*s_y))))))
            platform_lon = (lambda_0 - np.arctan(s_y/(H-s_x)))*(180.0/np.pi)
            return platform_lat, platform_lon

        with Dataset(filename) as dataset:
            if platform in GOES_SERIE:
                projection_info = dataset['goes_imager_projection']
                platform_lat, platform_lon = latlon_from_abi_file(
                    dataset['x'][:],
                    dataset['y'][:],
                    projection_info.longitude_of_projection_origin,
                    projection_info.perspective_point_height+projection_info.semi_major_axis,
                    projection_info.semi_major_axis,
                    projection_info.semi_minor_axis
                )
                if channel in ABI_CHANNELS:
                    data = dataset["CMI_" + channel][:]
                if channel in RRQPEF_CHANNELS:
                    data = dataset["RRQPE"][:]
                
            elif platform in HIMAWARI_SERIE:
                if channel in ABI_CHANNELS:
                    data = dataset['Sectorized_CMI'][:]
                    projection_info = dataset['fixedgrid_projection']
                    platform_lat, platform_lon = latlon_from_abi_file(
                        dataset['x'][:] / 10**6,
                        dataset['y'][:] / 10**6,
                        projection_info.longitude_of_projection_origin,
                        projection_info.perspective_point_height+projection_info.semi_major,
                        projection_info.semi_major,
                        projection_info.semi_minor
                    )
                elif channel in RRQPEF_CHANNELS:
                    data = dataset['RRQPE'][:]
                    platform_lat = dataset['Latitude'][:]
                    platform_lon = dataset['Longitude'][:]
                    
        return platform_lat, platform_lon, data

    if platform in GOES_SERIE:
        if channel in GLM_CHANNELS:
            data, lat_grid, lon_grid = accumulate_lightning_maps(filenames)
        else:
            platform_lat, platform_lon, data = latlon_from_file(filenames[0], platform, channel)
            platform_lat = platform_lat.filled(np.nan)
            platform_lon = platform_lon.filled(np.nan)
            data = data.filled(np.nan)

    elif platform in HIMAWARI_SERIE:
        if channel in ABI_CHANNELS:
            n = 550  # size of each tile

            platform_lat = np.zeros((10*n,10*n))
            platform_lon = np.zeros((10*n,10*n))
            data = np.zeros((10*n,10*n))
            for i, filename in enumerate(filenames):
                cell_pad = 0
                if i >= 0: cell_pad += 2
                if i >= 6: cell_pad += 3
                if i >= 14: cell_pad += 1
                if i >= 74: cell_pad += 1
                if i >= 82: cell_pad += 3

                tile_lat, tile_lon, tile_data = latlon_from_file(filename, platform, channel)
                y = (i+cell_pad)%10
                x = (i+cell_pad)//10
                
                platform_lat[n*x:n*(x+1), n*y:n*(y+1)] = tile_lat
                platform_lon[n*x:n*(x+1), n*y:n*(y+1)] = tile_lon
                data[n*x:n*(x+1), n*y:n*(y+1)] = tile_data
        elif channel in RRQPEF_CHANNELS:
            platform_lat, platform_lon, data = latlon_from_file(filenames[0], platform, channel)

            """import matplotlib.pyplot as plt
            plt.figure(figsize=(12,4))
            plt.subplot(131)
            plt.imshow(platform_lat)
            plt.subplot(132)
            plt.imshow(platform_lon)
            plt.subplot(133)
            plt.imshow(data)
            plt.show()"""

    elif platform in NEXRAD_BASIS:
        import pyart # /!\ Import here because this is a special library
        assert len(filenames) == 1
        filename = filenames[0]
        radar = pyart.io.read_nexrad_archive(filename)
        
        platform_lat, platform_lon, alts = radar.get_gate_lat_lon_alt(sweep=0)
        data = radar.get_field(sweep=0, field_name='reflectivity').astype('float')
        
    return platform_lat, platform_lon, data
