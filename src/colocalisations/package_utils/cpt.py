import colorsys
import os.path

import matplotlib as mpl
import numpy as np


def loadCPT(colormap):
    folder = os.path.join(os.path.dirname(os.path.dirname(__file__)))
    filename = os.path.join(folder, "res/colormap", colormap)
    try:
        f = open(filename)
    except:
        print("File ", filename, "not found")
        return None

    lines = f.readlines()

    f.close()

    x = np.array([])
    r = np.array([])
    g = np.array([])
    b = np.array([])

    colorModel = 'RGB'

    for l in lines:
        ls = l.split()
        if l[0] == '#':
            if ls[-1] == 'HSV':
                colorModel = 'HSV'
                continue
            else:
                continue
        if ls[0] == 'B' or ls[0] == 'F' or ls[0] == 'N':
            pass
        else:
            x = np.append(x, float(ls[0]))
            r = np.append(r, float(ls[1]))
            g = np.append(g, float(ls[2]))
            b = np.append(b, float(ls[3]))
            xtemp = float(ls[4])
            rtemp = float(ls[5])
            gtemp = float(ls[6])
            btemp = float(ls[7])

        x = np.append(x, xtemp)
        r = np.append(r, rtemp)
        g = np.append(g, gtemp)
        b = np.append(b, btemp)

    if colorModel == 'HSV':
        for i in range(r.shape[0]):
            rr, gg, bb = colorsys.hsv_to_rgb(r[i] / 360., g[i], b[i])
        r[i] = rr
        g[i] = gg
        b[i] = bb

    if colorModel == 'RGB':
        r = r / 255.0
        g = g / 255.0
        b = b / 255.0

    xNorm = (x - x[0]) / (x[-1] - x[0])

    red = []
    blue = []
    green = []

    for i in range(len(x)):
        red.append([xNorm[i], r[i], r[i]])
        green.append([xNorm[i], g[i], g[i]])
        blue.append([xNorm[i], b[i], b[i]])

    colorDict = {'red': red, 'green': green, 'blue': blue}

    return colorDict


cpt_cmap = mpl.colors.LinearSegmentedColormap('cpt', loadCPT('IR4AVHRR6.cpt'))


def getHHC_cmap():
    cmaplist = np.array([
        (0, 0, 0),
        (38, 11, 104),
        (64, 38, 141),
        (145, 0, 0),
        (255, 0, 0),
        (161, 76, 198),
        (255, 0, 255),
        (255, 200, 0),
        (255, 255, 0),
        (0, 100, 0),
        (0, 255, 0),
        (0, 235, 235),
        (0, 0, 255),
        (100, 110, 100),
        (150, 150, 150),
    ]) / 255

    labels = [
        'Giant Hall', 'Large Hall', 'Hall', 'Heavy Rain', 'Rain', 'Big Drops', 'Freezing Rain', 'FR/IP',
        'Ice Piellets', 'Graupel', 'Wet Snow', 'Dry Snow', 'Crystals', 'Birds/Insects', 'Ground Clutter']
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Hydrometeor Classes', cmaplist, 15)
    return cmap, labels


def getDPR_cmap():
    # https://unidata.github.io/python-gallery/examples/Precipitation_Map.html
    clevs = [0, 1, 2.5, 5, 7.5, 10, 15, 20, 30, 40, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 750]

    cmap_data = [
        (1.0, 1.0, 1.0),
        (0.3137255012989044, 0.8156862854957581, 0.8156862854957581),
        (0.0, 1.0, 1.0),
        (0.0, 0.8784313797950745, 0.501960813999176),
        (0.0, 0.7529411911964417, 0.0),
        (0.501960813999176, 0.8784313797950745, 0.0),
        (1.0, 1.0, 0.0),
        (1.0, 0.6274510025978088, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.125490203499794, 0.501960813999176),
        (0.9411764740943909, 0.250980406999588, 1.0),
        (0.501960813999176, 0.125490203499794, 1.0),
        (0.250980406999588, 0.250980406999588, 1.0),
        (0.125490203499794, 0.125490203499794, 0.501960813999176),
        (0.125490203499794, 0.125490203499794, 0.125490203499794),
        (0.501960813999176, 0.501960813999176, 0.501960813999176),
        (0.8784313797950745, 0.8784313797950745, 0.8784313797950745),
        (0.9333333373069763, 0.8313725590705872, 0.7372549176216125),
        (0.8549019694328308, 0.6509804129600525, 0.47058823704719543),
        (0.6274510025978088, 0.42352941632270813, 0.23529411852359772),
        (0.4000000059604645, 0.20000000298023224, 0.0)
    ]
    cmap = mpl.colors.ListedColormap(cmap_data, 'Rain Rate')
    norm = mpl.colors.BoundaryNorm(clevs, cmap.N)
    return cmap, norm
