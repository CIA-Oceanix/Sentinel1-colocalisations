import numpy as np
from skimage.draw import polygon

def read_melting_layer(radar, shape=2000, resolution=4):
    lon = radar.lon
    lat = radar.lat
    
    data = np.zeros((shape, shape))
    for i, entry in enumerate(radar.sym_block[0]):
        
        if not i%2: continue
        vectors = np.array(entry['vectors'])*resolution + shape//2
        rr, cc = polygon(vectors[:,0], vectors[:,1], data.shape)
        data[rr, cc] +=1 #= i//2

    data = 4-data
    data[data == 4] = np.nan

    lats = np.full((shape, shape), lat)
    lons = np.full((shape, shape), lon)

    indices = (np.indices((shape, shape)) - shape//2)/resolution

    r_earth = 6378
    new_lats = lats + (indices[1] / r_earth) * (180 / np.pi)
    new_lons = lons + (indices[0] / r_earth) * (180 / np.pi) / np.cos(new_lats * np.pi / 180)
    return new_lats, new_lons, data
