import numpy as np
import tqdm
import PIL.Image
from cv2 import resize
import glob
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session


def get_apply_multiscale_model_on_wideswath(model_key):
    clear_session()
    model = load_model(f"res/models/{model_key}.h5", compile=False)
    model.summary()

    input_layers = [layer for layer in model.layers if 'input' in layer.name]
    n = len(input_layers)
    s = input_layers[0].input_shape[0][1]

    def apply_prediction(out_, input_array, args):
        predictions = model.predict(input_array)[0]
        for ab, prediction in zip(args, predictions):
            if ab is not None:
                a, b = ab
                out_[a:a + s, b:b + s][s // 4:-s // 4, s // 4:-s // 4] = prediction[s // 4:-s // 4, s // 4:-s // 4, 0]
        return out_

    def process_filename(filename, batch_size=32):
        im = np.array(PIL.Image.open(filename)) / (2 ** 16 - 1)
        shapes = [(im.shape[0] // (2 ** i), im.shape[1] // (2 ** i)) for i in range(n)]
        images = [resize(im, shape[::-1]) for shape in shapes]

        images = [np.pad(im, ((s // 2, s // 2), (s // 2, s // 2)), mode="reflect") for im in images]
        out_ = np.zeros(images[0].shape)

        input_array = [np.zeros((batch_size, s, s, 1)) for _ in range(n)]
        k = 0
        args = [None for _ in range(batch_size)]
        for x in range(0, im.shape[0], s // 2):
            for y in range(0, im.shape[1], s // 2):
                args[k % batch_size] = (x, y)
                for i in range(n):
                    current_x = x // (2 ** i)
                    current_y = y // (2 ** i)

                    input_array[i][k % batch_size, :, :, 0] = images[i][current_x: current_x + s,
                                                              current_y: current_y + s]

                if k % batch_size == batch_size - 1:
                    out_ = apply_prediction(out_, input_array, args)
                    args = [None for _ in range(batch_size)]
                k += 1

        out_ = apply_prediction(out_, input_array, args)[s // 2:-s // 2, s // 2:-s // 2]
        return out_

    return process_filename
