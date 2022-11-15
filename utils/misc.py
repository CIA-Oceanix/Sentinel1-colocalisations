import sys
from datetime import datetime
import matplotlib.pyplot as plt

from utils.cpt import cpt_cmap

from check_args import GOES_SERIE, HIMAWARI_SERIE, NEXRAD_BASIS, SATELLITE_PLATFORMS


def log_print(s, f=print, format='%Y-%m-%d %H:%M:%S'):
    f(f"{datetime.now().strftime(format)}\t{s}")

def r_print(s):
    sys.stdout.write(f'\r{s}')

def platform_cmap_args(platform):
    if platform in SATELLITE_PLATFORMS:
        vmin=170
        vmax=378
        cmap=cpt_cmap
    elif platform in NEXRAD_BASIS:
        vmin=0
        vmax=40
        cmap=plt.get_cmap('turbo')
    return cmap, vmin, vmax
