import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from check_args import CHANNELS, SATELLITE_PLATFORMS
from utils.cpt import cpt_cmap, getHHC_cmap


def log_print(s, verbose_min, verbose, f=print, format='%Y-%m-%d %H:%M:%S'):
    if verbose >= verbose_min:
        f(f"{datetime.now().strftime(format)}\t{s}")


def r_print(s):
    sys.stdout.write(f'\r{s}')


def platform_cmap_args(platform, channel):
    kargs = {"cmap": plt.get_cmap('turbo')}
    colobar_postprocess = None
    norm = None
    if channel in CHANNELS['ABI'] or platform in SATELLITE_PLATFORMS['SEVIRIS']:
        kargs['vmin'] = 170
        kargs['vmax'] = 378
        kargs['cmap'] = cpt_cmap
    elif channel == 'GLM':
        kargs['vmin'] = 1
        kargs['vmax'] = 100
        kargs['cmap'] = plt.get_cmap('hot')
        kargs['cmap'].set_under(color=(0.5, 0.5, 0.5))
    elif channel in ["nexrad-level2", 'RRQPEF'] or channel[:3] in ['N0Z', 'N0Q']:
        kargs['vmin'] = 0
        kargs['vmax'] = 60
    elif channel[:3] == 'DPR':
        kargs['vmin'] = 0
        kargs['vmax'] = 100
    elif channel[:3] == 'NZM':
        kargs['vmin'] = 0
        kargs['vmax'] = 4
    elif channel[:3] in ['HHC', 'N0H']:
        cmap, cmap_legend = getHHC_cmap()

        def colobar_postprocess(colorbar, cmap_legend=cmap_legend):
            colorbar.set_ticks(np.arange(0.5, len(cmap_legend), 1))
            colorbar.set_ticklabels(cmap_legend)
            colorbar.ax.tick_params(rotation=45)

        kargs['vmin'] = 0
        kargs['vmax'] = 15
        kargs['cmap'] = cmap
    elif 'wind_at_10_metres' in channel:
        kargs['vmin'] = -20
        kargs['vmax'] = 20
        kargs['cmap'] = plt.get_cmap('coolwarm')

    return kargs, norm, colobar_postprocess
