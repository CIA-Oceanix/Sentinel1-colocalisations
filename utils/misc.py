import sys
from datetime import datetime
import matplotlib.pyplot as plt

from utils.cpt import cpt_cmap

from check_args import GOES_SERIE, HIMAWARI_SERIE, NEXRAD_BASIS, SATELLITE_PLATFORMS, ABI_CHANNELS, RRQPEF_CHANNELS, NEXRAD_CHANNELS, GLM_CHANNELS


def log_print(s, f=print, format='%Y-%m-%d %H:%M:%S'):
    f(f"{datetime.now().strftime(format)}\t{s}")

def r_print(s):
    sys.stdout.write(f'\r{s}')

def platform_cmap_args(platform, channel):
    vmin=None
    vmax=None
    cmap=plt.get_cmap('turbo')
    if channel in ABI_CHANNELS:
        vmin=170
        vmax=378
        cmap=cpt_cmap
    elif channel in GLM_CHANNELS:
        vmin=1
        vmax=100
        cmap = plt.get_cmap('hot')
        cmap.set_under(color=(0.5,0.5,0.5))
    elif channel in NEXRAD_CHANNELS + RRQPEF_CHANNELS:
        vmin=0
        vmax=40
    return cmap, vmin, vmax
