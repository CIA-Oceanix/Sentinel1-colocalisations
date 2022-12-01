import os
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import fire
import json

import PIL.Image
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session


from global_land_mask import globe
import MightyMosaic.MightyMosaic
get_mosaic = lambda x, y: MightyMosaic.MightyMosaic.from_array(x, y, fill_mode = 'reflect', overlap_factor=2)

from utils.sentinel1 import get_iw_latlon

def rainfall_rgb(im):
    colors = np.array(((1,0,0), (1,1,0), (0,1,1)))
    new_im = np.zeros((im.shape[0], im.shape[1], 3), float)
    for i in range(3): new_im[im[:,:,i] > 0.5] = colors[i]
    return new_im
    
MODELS = {
     "NEXRAD": {
         "resize_factor": 2, 
         "input_lambda": lambda x: x/(2**16-1)*2 * 3 / 5.35, 
         "output_lambda": rainfall_rgb
     },
     "BiologicalSlicks": {
         "resize_factor": 1, 
         "input_lambda": lambda x: x/(2**16-1)*2,
         "output_lambda": lambda x: x
     },
     "ConvectiveCells": {
         "resize_factor": 4, 
         "input_lambda": lambda x: x/(2**16-1)*2,
         "output_lambda": lambda x: x
     }
}

def get_apply_model_on_wideswath(model_key, batch_size=4):
    clear_session()
    model = load_model(f"models/{model_key}.h5", compile=False)

    input_lambda = MODELS[model_key]['input_lambda']
    output_lambda = MODELS[model_key]['output_lambda']
    resize_factor = MODELS[model_key]['resize_factor']
    
    def apply_model_on_wideswath(filename):
        array = np.array(PIL.Image.open(filename), float)
        array = input_lambda(array)
        
        mosaic = get_mosaic(array, model.input_shape[1:3])
        
        prediction = mosaic.apply(model.predict, batch_size=batch_size).get_fusion()
        prediction = output_lambda(prediction)
        
        return prediction
        
    return apply_model_on_wideswath

    
def apply_on_keys(filenames, getter, model_key):
    output_filenames = []
    
    apply_model = get_apply_model_on_wideswath(model_key)
    for tiff_filename in filenames:
        key = os.path.split(tiff_filename)[1].split('-')[4]
        output_filename = f"outputs/{key}/DL_{model_key}.png"

        output = apply_model(tiff_filename)
        
        if output.shape[2] == 1:  output = output[:,:,[0,0,0,0]]
        if output.shape[2] == 3:  output = output[:,:,[0,1,2,0]]

        polygon = getter(key)[1]
        lat_grid, lon_grid = get_iw_latlon(polygon=polygon, shape=output.shape[:2])
        mask = globe.is_land(lat_grid, lon_grid)

        output[:,:,3] = np.where(mask > 0.5, 0, 1)
        output = (np.clip(output, 0, 1)*(2**8-1)).astype('uint8')

        os.makedirs(os.path.split(output_filename)[0], exist_ok=True)
        PIL.Image.fromarray(output).save(output_filename)
    return output_filenames
    
