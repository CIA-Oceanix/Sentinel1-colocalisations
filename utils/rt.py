import os
import cv2
import sys
import pyproj
import argparse
import PIL.Image
import numpy as np
import xarray as xr
from osgeo import gdal
import lxml.etree as etree
from skimage.transform import resize
from osgeo import gdal, gdal_array, osr

import netCDF4
from netCDF4 import Dataset
from scipy.interpolate import interp1d, bisplrep, bisplev, LinearNDInterpolator
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure

def create_grid(dataset):
    i_indices, j_indices = np.meshgrid(np.arange(dataset.RasterXSize),
                                       np.arange(dataset.RasterYSize),
                                       indexing='ij')
    i_indices, j_indices = np.rot90(i_indices), np.rot90(j_indices, 3)
    return i_indices, j_indices

def pixel_to_world(geotransform, x, y):
    xoff, a, b, yoff, d, e = geotransform
    lon_mat = a * x + b * y + xoff
    lat_mat = d * x + e * y + yoff
    return lon_mat, lat_mat

def transform_coords(dataset, epsg, x, y):
    source = dataset.GetGCPSpatialRef()
    source_pyproj = source.ExportToProj4() 
    pyproj_transformer = pyproj.Transformer.from_crs(source_pyproj, 'epsg:{}'.format(epsg), always_xy=True)
    return pyproj_transformer.transform(x, y)

def grid_pixel_to_world(gdal_ds):
    geotransform = gdal.GCPsToGeoTransform(gdal_ds.GetGCPs())
    i_indices, j_indices = create_grid(gdal_ds)
    
    world_x, world_y = pixel_to_world(geotransform, i_indices, j_indices)
    lon, lat = transform_coords(gdal_ds, 4326, world_x, world_y)
    return lon, lat, i_indices, j_indices


def cmod5n_rt(theta):
    """
    Compute cmod5n
    """
    # INPUT : theta : incidence angle [deg]
    # OUTPUT : sig = cmod5n(theta,10m/s,45°)
    THETM = 40.
    THETHR = 25.

    X = (theta - THETM) / THETHR
    XX = X*X
    A0 = -0.6878 - 0.7957 * X + 0.3380 * XX - 0.1728 * np.multiply(X, XX)
    A1 = 0.0040 * X
    S = 1.103 + 0.159 * X
    GAM = 6.7329 + 2.7713 * X - 2.2885 * XX
    a3 = 1. / (1. + np.exp(-S))
    B0 = (a3 ** GAM) * 10. ** (A0 + 10. * A1)
    B1 = 0.066 * (0.5 + X - np.tanh(4. * (X + 0.4422)))
    B1 = 0.045 * (1. + X) - B1
    B1 = B1 / 1.0133265099270379
    sig = B0 * (1.0 + B1 * 2 ** (-0.5)) ** 1.6
    return sig

# ____________________________----------------_____________________________
# ____________________________Sar Image Reader_____________________________
class S1ImageReader():
    """
    Class to handle S1 imagery reading
    """
    def __init__(self, root_dir, relative_path, filename, mode):
        self.relative_path = relative_path
        self.absolute_path = os.path.join(root_dir, relative_path)
        self.filename = self.handle_filename(filename)
        self.tiff_file = None
        self.calib_file = None
        self.noise_file = None
        self.annotation_file = None
        self.dataset = None
        self.mode = mode

    def files_finder(self):
        """
        Utils to build pathes for S1A images
        """
        tiff_filename = '{}.tiff'.format(self.filename)
        calib_filename = 'calibration-{}.xml'.format(self.filename)
        noise_filename = 'noise-{}.xml'.format(self.filename)
        annotation_filename = '{}.xml'.format(self.filename)
        
        self.tiff_file = os.path.join(self.absolute_path, 'measurement', tiff_filename)
        self.calib_file = os.path.join(self.absolute_path, 'annotation', 'calibration', calib_filename)
        self.noise_file = os.path.join(self.absolute_path, 'annotation', 'calibration', noise_filename)
        self.annotation_file = os.path.join(self.absolute_path, 'annotation', annotation_filename)

    def read_ds(self):
        """
        Util to read dataset with GDAL
        """
        try:
            self.dataset = gdal.Open(self.tiff_file)
        except Exception as error:
            sys.exit()

    def get_image(self):
        """
        Read image from gdal dataset
        """
        image = self.dataset.ReadAsArray(buf_type=gdal_array.flip_code(np.float32))
        return image

    @staticmethod
    def handle_filename(filename):
        """
        Utils to handle filename from SAFE directories
        """
        extension = next((suffix for suffix in (".tiff", ".xml")
                          if filename.endswith(suffix)), None)
        if extension is not None:
            filename = filename.replace(extension, '')
        return filename

# ____________________________-------------------_____________________________
# ____________________________Sar Image Processor_____________________________

class S1ImageProcessor(S1ImageReader):
    """
    Class to process S1 imagery
    """
    def __init__(self, root_dir, relative_path, filename, mode):
        super().__init__(root_dir, relative_path, filename, mode)
        self.ny, self.nx = None, None
        self.xc, self.yc = None, None
        self.Yc, self.Xc = None, None
        self.sar_mask = None
        self.lat = None
        self.lon = None
        self.image = None

    def get_data(self):
        self.image = self.get_image()
        self.image = np.abs(self.image) ** 2
        self.image = self.image.astype(np.float32)
        self.ny, self.nx = self.image.shape

        if self.mode.upper() in ['IW']: ## assume we are in grdh
            self.xc = np.arange(self.nx)[4::10] # 100 m/px
            self.yc = np.arange(self.ny)[4::10]
        elif self.mode.upper() in ['EW']:
            self.xc = np.arange(self.nx)[1::3]  # 120 m/px
            self.yc = np.arange(self.ny)[1::3]
            
        self.Xc, self.Yc = np.meshgrid(self.xc, self.yc, indexing='xy')

    def calibration_dn_to_s0(self):
        root = etree.parse(self.calib_file).getroot()
        
        tab_s0 = []
        tab_line, tab_pix = [], []
        try:
            for nv in root.xpath('.//calibrationVectorList/calibrationVector'):
                l = float(nv.xpath('.//line')[0].text)
                p = [float(e_) for e_ in nv.xpath('.//pixel')[0].text.split(' ')]
                s0 = [float(e_) for e_ in nv.xpath('.//sigmaNought')[0].text.split(' ')]
                # New + : to perform Spline interpolation (not the Linear)
                tab_line += [l] * len(p)
                tab_pix += p
                tab_s0 += s0
        except Exception as Exp:
            sys.exit()
        return tab_line, tab_pix, tab_s0

    def incidence_angle_reading(self):
        """
        Function that allows to read incidence angle
        """
        root = etree.parse(self.annotation_file).getroot()
        tab_line = []
        tab_pix = []
        tab_inc = []
        try:
            for nv in root.xpath('geolocationGrid/geolocationGridPointList/geolocationGridPoint'):
                l = float(nv.xpath('.//line')[0].text)
                p = float(nv.xpath('.//pixel')[0].text)
                inc = float(nv.xpath('.//incidenceAngle')[0].text)
                tab_line.append(l)
                tab_pix.append(p)
                tab_inc.append(inc)
        except Exception as Exp:
            sys.exit()

        return bisplrep(x=tab_pix, y=tab_line, z=tab_inc,
                        xb=0, xe=max(tab_pix), yb=0, ye=max(tab_line),
                        kx=1, ky=1)

    def get_noise_28(self):
        # _____________________________
        # For IPF 2.8 version and lower
        # -----------------------------
        tab_s0_noise = []
        tab_pts_noise = []
        rt_xpath = './/noiseVectorList/noiseVector'
        rv_xpath_Lut = './/noiseLut'
        try:
            for nv in self.root.xpath(rt_xpath):
                l = float(nv.xpath('.//line')[0].text)
                p = [float(e_) for e_ in nv.xpath('.//pixel')[0].text.split(' ')]
                s0 = [float(e_) for e_ in nv.xpath(rv_xpath_Lut)[0].text.split(' ')]
                tab_pts_noise += zip(p, [l] * len(p))
                tab_s0_noise += s0
            tck_lut_noise = LinearNDInterpolator(tab_pts_noise, np.array(tab_s0_noise))
            noise = tck_lut_noise(self.Xc, self.Yc)
        except Exception as Exp:
            sys.exit()
        return noise

    def get_noise_29(self):
        # _____________________________
        # For IPF 2.9 version and upper
        # -----------------------------

        # start registering Azimuth vectors+interpolator creation
        tabs = dict()
        block_noiseAz = dict()
        for nv in self.root.xpath('.//noiseAzimuthVectorList/noiseAzimuthVector'):

            _sw = nv.xpath('swath')[0].text
            _fy = int(nv.xpath('firstAzimuthLine')[0].text)
            _ly = int(nv.xpath('lastAzimuthLine')[0].text)
            _fx = int(nv.xpath('firstRangeSample')[0].text)
            _lx = int(nv.xpath('lastRangeSample')[0].text)

            line = [int(e_) for e_ in nv.xpath('line')[0].text.split(' ')]
            val = [float(e_) for e_ in nv.xpath('noiseAzimuthLut')[0].text.split(' ')]
            if len(line) == 1:
                line = np.arange(_fy, _ly + 1)
                val = np.ones_like(line) * val
                if len(line) == 1:
                    continue
            faz = interp1d(line, val, kind='linear', fill_value='extrapolate')
            tabs[(_sw, _fy, _fx, _ly, _lx)] = faz

        block_noiseAz = tabs

        # Start registering range vector
        tab_s0 = []
        pix = []
        line = []
        tck_lut_noise = dict()
        for nv in self.root.xpath('.//noiseRangeVectorList/noiseRangeVector'):
            l = int(nv.xpath('.//line')[0].text)
            p = [int(e_) for e_ in nv.xpath('.//pixel')[0].text.split(' ')]
            s0 = [float(e_) for e_ in nv.xpath('.//noiseRangeLut')[0].text.split(' ')]
            pix += p
            line += [l] * len(p)
            tab_s0 += s0

        tck_lut_noise = pix, line, tab_s0

        # creation of 2d interpolator for range vectors (one interpolator by subswath)
        InterpnoiseRgT = dict()
        lu = np.array(np.array(tck_lut_noise))
        
        for sw in np.unique([tab[0] for tab in block_noiseAz]):  # for each subswath
            # azhimuth block belonging to the considered subswath (ordering by first line):
            _blocktocon = sorted([tab for tab in tabs if tab[0] == sw], key=lambda l: l[1])
            idx = False
            ltoadd = []
            ptoadd = []
            ztoadd = []
            for (itr, _b) in enumerate(_blocktocon):
                (_sw, _fy, _fx, _ly, _lx) = _b
                idx2 = ((lu[0] >= _fx) * (lu[0] <= _lx) * (lu[1] >= _fy) * (lu[1] <= _ly) * (lu[2] != 0))
                idx += idx2
            ymin = min(lu[1, idx])
            ymax = max(lu[1, idx])
            xmin = min(lu[0, idx])
            xmax = max(lu[0, idx])
            ltoadd = np.repeat([_blocktocon[0][1], _blocktocon[-1][3]], 100)
            ptoadd = np.tile(np.linspace(xmin, xmax, 100), 2)
            idx4 = (lu[1, idx] == ymin)
            ztoadd = (interp1d(lu[0, idx][idx4],
                               lu[2, idx][idx4],
                               kind='linear',
                               fill_value='extrapolate'))(np.linspace(xmin, xmax, 100))
            idx4 = (lu[1, idx] == ymax)
            ztoadd = np.append(ztoadd, interp1d(lu[0, idx][idx4],
                                                lu[2, idx][idx4],
                                                kind='linear',
                                                fill_value='extrapolate')(np.linspace(xmin, xmax, 100)))

            InterpnoiseRgT[sw] = LinearNDInterpolator(list(zip(np.append(lu[0, idx], ptoadd),
                                                      np.append(lu[1, idx], ltoadd))),
                                                      np.append(lu[2, idx], ztoadd))

        block_noiseRg = InterpnoiseRgT
        Azline = self.yc
        Rgline = self.xc
        noise  = np.zeros((self.yc.shape[0], self.xc.shape[0]), dtype=np.float32)

        if len(block_noiseRg) == 0:
            # we have at least one 2d interpolator for range vector since at least 1 subswath
            raise NameError('Non Conformed block_noiseAz')

        if len(block_noiseAz) == 0:
            # we expect variation on azimuth otherwise th product is corrupted
            raise NameError('Non Conformed block_noiseAz')
        for _tab in block_noiseAz:
            idAz = (Azline >= _tab[1]) & (Azline <= _tab[3])
            if not idAz.any():
                continue
            tckAz = block_noiseAz[_tab]
            idRg  = (Rgline >= _tab[2]) & (Rgline <= _tab[4])
            faz   = np.outer(tckAz(Azline[idAz]), np.ones_like(Rgline[idRg]).T)

            InterpnoiseRgT = block_noiseRg[_tab[0]]
            ID = (self.Xc >= _tab[2]) & (self.Xc <= _tab[4]) & (self.Yc >= _tab[1]) & (self.Yc <= _tab[3])
            noiseRg = InterpnoiseRgT(self.Xc[ID], self.Yc[ID])
            noise[ID] = noiseRg * faz.flatten()

        return noise.astype(np.float32)

    def get_noise(self):
        """
        Get Noise file and parse it
        """
        self.root = etree.parse(self.noise_file).getroot()
        
        if self.root.xpath('.//noiseVectorList/noiseVector'):
            noise = self.get_noise_28()
        elif self.root.xpath('.//noiseRangeVectorList/noiseRangeVector'):
            noise = self.get_noise_29()
        else:
            noise=0
        return noise

    def get_geotransform(self):
        """
        Get geotransforms from GCPs
        """
        try :
            geotransform = gdal.GCPsToGeoTransform(self.dataset.GetGCPs())
            yy, xx = np.meshgrid(self.yc, self.xc)
            X = geotransform[0] + geotransform[1] * xx + geotransform[2] * yy
            Y = geotransform[3] + geotransform[4] * xx + geotransform[5] * yy
            x_min = np.nanmin(X)
            x_max = np.nanmax(X)
            y_min = np.nanmin(Y)
            y_max = np.nanmax(Y)
            bbox = [x_min, y_min, x_max, y_max]
        except Exception as Exp :
            sys.exit()
        return geotransform, bbox, X, Y


    def process_image(self):
        """
        Function to apply all the processing steps from raw S1 Image to processed
        image
        """
        tab_line, tab_pix, tab_s0 = self.calibration_dn_to_s0()
        tck_lut_s0 = bisplrep(x=tab_pix, y=tab_line, z=tab_s0,
                              xb=0, xe=max(tab_pix),
                              yb=0, ye=max(tab_line),
                              kx=1, ky=1)

        s0_interp = bisplev(self.xc, self.yc, tck_lut_s0).T

        self.image = resize(self.image, (self.yc.shape[0], self.xc.shape[0]), anti_aliasing=True)
        self.image = self.image /(s0_interp**2).astype(np.float32, copy=False)

        try:
            noise = self.get_noise()
            nesz = noise/(s0_interp**2)
            self.image = np.maximum(self.image-nesz, 0)
            del noise, nesz, s0_interp, tck_lut_s0
        except:
            pass

        inc_interpolator = self.incidence_angle_reading()
        tab_inc = bisplev(self.xc, self.yc, inc_interpolator).T.astype(np.float32)
        _nrcs = cmod5n_rt(tab_inc)
        self.image = self.image/_nrcs

        self.image = np.minimum(self.image, (6 if '-vv-' in self.filename else 3))
        self.image = np.round((2**16-1)*self.image/(6 if '-vv-' in self.filename else 3)).astype('uint16')

        _, _, self.lon, self.lat = self.get_geotransform()

        return self.image


def safe_to_tiff(safe_dir, vh=True):
    filenames = os.listdir(os.path.join(safe_dir, 'measurement'))

    output_dir = os.path.split(safe_dir)[0]
    os.makedirs(output_dir, exist_ok=True)
    new_filenames = []
    for filename in filenames:
        polarization = os.path.split(filename)[1].split('-')[3]
        if polarization == 'vh' and not vh: continue

        mode = os.path.split(filename)[1].split('-')[1]
        s1_image_processor = S1ImageProcessor(safe_dir, '', filename, mode)
        s1_image_processor.files_finder()
        s1_image_processor.read_ds()
        s1_image_processor.get_data()
        s1_image_processor.process_image()

        new_filename = f"{output_dir}/{filename}"
        PIL.Image.fromarray(s1_image_processor.image).save(new_filename)
        new_filenames.append(new_filename)
    return new_filenames
