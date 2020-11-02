# TODO remove future etc
# TODO replace GladImageReader with rasterio or equivalent
# TODO replace GladVectorReader with ?? - something that reads to pandas?



from __future__ import print_function
from __future__ import division
from builtins import zip
from builtins import str
from builtins import range
from builtins import object
from past.utils import old_div
import sys
import warnings, logging
import gdal
import ogr
import numpy as np
import osr
import matplotlib.pyplot as pyplot
from enum import Enum
from rasterio.rio.options import nodata_opt
from scipy import stats as stats
import scipy.signal as signal
from matplotlib import patches
from sklearn import linear_model, metrics
from sklearn.model_selection import cross_val_predict, cross_validate

import collections
from collections import OrderedDict
from sklearn.preprocessing import PolynomialFeatures
from skimage.feature import greycomatrix, greycoprops
from skimage import data
import geopandas as gpd, pandas as pd

from PIL import Image
from PIL import ImageDraw
import os
import numpy as np
import rasterio
import re
import rasterio
from rasterio.features import sieve
from rasterio.windows import Window
from rasterio.mask import raster_geometry_mask

if sys.version_info.major == 3:
    from sklearn.metrics import make_scorer
else:
    from sklearn.metrics.scorer import make_scorer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def nanentropy(x, axis=None):
    """
    x is assumed to be an (nsignals, nsamples) array containing integers between
    0 and n_unique_vals
    """
    if axis is None:
        # quantise x
        nbins = 100
        x = x[~np.isnan(x)]
        if len(x) < 10:
            return 0.
        x = x-x.min()
        x = np.int32(np.round(np.float32(nbins*x)/x.max()))

        value, counts = np.unique(x, return_counts=True)
        p = np.array([count/float(x.size) for count in counts])
        # p = np.array([np.size(x[x == i]) / (1.0 * x.size) for i in np.unique(x)])
        # compute Shannon entropy in bits
        return -np.sum(p * np.log2(p))
    else:        # hack for 2D slices of 3D array (on a rolling_window 3D array)
        along_axis = np.setdiff1d(range(0, len(x.shape)), axis)[0]
        e = np.zeros((x.shape[along_axis]))
        for slice in range(x.shape[along_axis]):
            if along_axis == 0:
                xa = x[slice,:,:]
            elif along_axis == 1:
                xa = x[:,slice,:]
            elif along_axis == 2:
                xa = x[:,:,slice]
            e[slice] = nanentropy(xa)
        return e


def scatter_y_actual_vs_pred(y, pred, scores, xlabel='Measured AGC (t C ha$^{-1}$)', ylabel='Predicted AGC (t C ha$^{-1}$)'):
    df = pd.DataFrame({xlabel: y, ylabel: pred})
    scatter_ds(df, do_regress=False)
    # scatter_plot(y, pred, xlabel=xlabel, ylabel=ylabel, do_regress = False)

    mn = np.min([y, pred])
    mx = np.max([y, pred])
    h, = pyplot.plot([0, mx], [0, mx], 'k--', lw=2, zorder=-1, label='1:1')
    pyplot.xlim(0, mx)
    pyplot.ylim(0, mx)
    pyplot.text(26, 5, str.format('$R^2$ = {0:.2f}', scores['R2_stacked']),
               fontdict={'size': 11})
    pyplot.text(26, 2, str.format('RMSE = {0:.2f} t C ha{1}',np.abs(scores['test_-RMSE']).mean(),'$^{-1}$'),
               fontdict={'size': 11})
    pyplot.show()
    pyplot.tight_layout()
    pyplot.legend([h], ['1:1'], frameon=False)
    pyplot.pause(0.1)


def scatter_ds(data, x_col=None, y_col=None, class_col=None, label_col=None, thumbnail_col=None, do_regress=True,
               x_label=None, y_label=None, xfn=lambda x: x, yfn=lambda y: y):
    ims = 20.       # scale factor for thumbnails
    if x_col is None:
        x_col = data.columns[0]
    if y_col is None:
        y_col = data.columns[1]

    x = xfn(data[x_col])
    y = yfn(data[y_col])

    if class_col is None:
        data['class_col'] = np.zeros((data.shape[0],1))
        class_col = 'class_col'

    # TO DO: either remove thumbnail option or refactor
    # TO DO: sort classes
    # if 'Intact' in classes and 'Moderate' in classes and 'Degraded' in classes and classes.size==3:
    #     classes = np.array(['Degraded', 'Moderate', 'Intact'])
    classes = np.array([class_name for class_name, class_group in data.groupby(by=class_col)])
    n_classes =  len(classes)
    if n_classes == 1:
        colours = [(0., 0., 0.)]
    else:
        colours = ['tab:orange', 'g', 'r', 'b', 'y', 'k', 'm']

    xlim = [x.min(), x.max()]
    ylim = [y.min(), y.max()]
    xd = np.diff(xlim)[0]
    yd = np.diff(ylim)[0]

    pyplot.axis('tight')
    pyplot.axis(xlim + ylim)
    ax = pyplot.gca()
    handles = [0] * n_classes

    for class_i, (class_label, class_data) in enumerate(data.groupby(by=class_col)):
        colour = colours[class_i]
        if thumbnail_col is None:
            pyplot.plot(xfn(class_data[x_col]), yfn(class_data[y_col]), markerfacecolor=colour, marker='.', label=class_label, linestyle='None',
                       markeredgecolor=colour, markersize=5)
        for rowi, row in class_data.iterrows():
            xx = xfn(row[x_col])
            yy = yfn(row[y_col])
            if label_col is not None:   # add a text label for each point
                pyplot.text(xx - .0015, yy - .0015, row[label_col],
                           fontdict={'size': 9, 'color': colour, 'weight': 'bold'})

            if thumbnail_col is not None:   # add a thumbnail for each point
                imbuf = np.array(row[thumbnail_col])
                band_idx = [0, 1, 2]
                if imbuf.shape[2] == 8:  # wv3
                    band_idx = [4, 2, 1]
                elif imbuf.shape[2] >= 8:  # wv3 + pan
                    band_idx = [5, 3, 2]

                extent = [xx - xd/(2 * ims), xx + xd/(2 * ims), yy - yd/(2 * ims), yy + yd/(2 * ims)]
                pyplot.imshow(imbuf[:, :, band_idx], extent=extent, aspect='auto')
                handles[class_i] = ax.add_patch(patches.Rectangle((extent[0], extent[2]), xd/ims, yd/ims,
                                                fill=False, edgecolor=colour, linewidth=2.))

    if do_regress:  # find and display regression error stats
        (slope, intercept, r, p, stde) = stats.linregress(x, y)
        scores, predicted = FeatureSelector.score_model(x[:,None], y[:,None], model=linear_model.LinearRegression(),
                                                        find_predicted=True, cv=len(x), print_scores=False, score_fn=None)

        pyplot.text((xlim[0] + xd * 0.7), (ylim[0] + yd * 0.05), '$R^2$ = {0:.2f}'.format(np.round(scores['R2_stacked'], 2)),
                   fontdict={'size': 12})
        yr = np.array(xlim)*slope + intercept
        pyplot.plot(xlim, yr, 'k--', lw=2, zorder=-1)

        yhat = x * slope + intercept
        rmse = np.sqrt(np.mean((y - yhat) ** 2))

        logger.info('RMSE: {0:.4f}'.format(rmse))
        logger.info('RMSE LOOCV: {0:.4f}'.format(-scores['test_-RMSE'].mean()))
        logger.info('R^2:  {0:.4f}'.format(r ** 2))
        logger.info('R^2 stacked: {0:.4f}'.format(scores['R2_stacked']))
        logger.info('P (slope=0): {0:.4f}'.format(p))
        logger.info('Slope: {0:.4f}'.format(slope))
        logger.info('Std error of slope: {0:.4f}'.format(stde))
    else:
        r = np.nan
        rmse = np.nan

    if x_label is not None:
        pyplot.xlabel(x_label, fontdict={'size': 12})
    else:
        pyplot.xlabel(x_col[-1], fontdict={'size': 12})

    if y_label is not None:
        pyplot.ylabel(y_label, fontdict={'size': 12})
    else:
        pyplot.ylabel(y_col[-1], fontdict={'size': 12})

    if n_classes > 1:
        if not thumbnail_col is None:
            pyplot.legend(handles, classes, fontsize=12)
        else:
            pyplot.legend(classes, fontsize=12)
    pyplot.show()
    return r ** 2, rmse

class GroundClass(Enum):
    Ground = 1
    DPlant = 2
    LPlant = 3
    Shadow = 4

class PatchFeatureExtractor():

    def __init__(self, num_bands=9):
        self.pan_bands, self.band_dict = ImageFeatureExtractor.get_band_info(num_bands)
        # self.feat_keys = []
        self.fn_dict = {}
        return


    def generate_fn_dict(self):
        if len(self.fn_dict) > 0:
            return

        # inner band ratios
        self.inner_dict = OrderedDict()
        self.inner_dict['pan/1'] = lambda pan, bands: pan
        # self.inner_dict['1/pan'] = lambda pan, bands: 1/pan   # TO DO check = does log10 invalidate?
        for num_key in list(self.band_dict.keys()):
            self.inner_dict['{0}/pan'.format(num_key)] = lambda pan, bands, num_key=num_key: bands[self.band_dict[num_key], :, :] / pan
            self.inner_dict['pan/{0}'.format(num_key)] = lambda pan, bands, num_key=num_key: pan / bands[self.band_dict[num_key], :, :]
            self.inner_dict['{0}/1'.format(num_key)] = lambda pan, bands, num_key=num_key: bands[self.band_dict[num_key], :, :]
            # self.inner_dict['1/{0}'.format(num_key)] = lambda pan, bands, num_key=num_key: 1/bands[self.band_dict[num_key], :, :] # TO DO check = does log10 invalidate?
            for den_key in list(self.band_dict.keys()):
                if not num_key == den_key:
                    self.inner_dict['{0}/{1}'.format(num_key, den_key)] = lambda pan, bands, num_key=num_key, den_key=den_key: bands[self.band_dict[num_key], :, :] / bands[self.band_dict[den_key], :, :]

        # inner veg indices
        SAVI_L = 0.05
        # these vals for MODIS from https://en.wikipedia.org/wiki/Enhanced_vegetation_index
        # L = 1.; C1 = 6; C2 = 7.5; G = 2.5
        nir_keys = [key for key in list(self.band_dict.keys()) if ('NIR' in key) or ('RE' in key)]
        for nir_key in nir_keys:
            post_fix = '' if nir_key == 'NIR' else '_{0}'.format(nir_key)
            self.inner_dict['NDVI' + post_fix] = lambda pan, bands, nir_key=nir_key: (bands[self.band_dict[nir_key], :, :] - bands[self.band_dict['R'], :, :]) / \
                                          (bands[self.band_dict[nir_key], :, :] + bands[self.band_dict['R'], :, :])
            self.inner_dict['SAVI' + post_fix] = lambda pan, bands, nir_key=nir_key: (1 + SAVI_L) * (bands[self.band_dict[nir_key], :, :] - bands[self.band_dict['R'], :, :]) / \
                                          (SAVI_L + bands[self.band_dict[nir_key], :, :] + bands[self.band_dict['R'], :, :])

        self.win_dict = OrderedDict({'mean': np.mean, 'std': np.std, 'entropy': nanentropy})
        self.scale_dict = OrderedDict({'log': np.log10, 'sqr': lambda x: np.power(x, 2), 'sqrt': np.sqrt})
        # mean, std, entropy
        for inner_key, inner_fn in self.inner_dict.items():
            for win_key, win_fn in self.win_dict.items():
                fn_key = '({0}({1}))'.format(win_key, inner_key)
                fn = lambda pan, bands, win_fn=win_fn, inner_fn=inner_fn: win_fn(inner_fn(pan, bands))
                self.fn_dict[fn_key] = fn
                if win_key == 'mean':
                    for scale_key, scale_fn in self.scale_dict.items():
                        fn_key = '{0}({1}({2}))'.format(scale_key, win_key, inner_key)
                        fn = lambda pan, bands, scale_fn=scale_fn, win_fn=win_fn, inner_fn=inner_fn: scale_fn(win_fn(inner_fn(pan, bands)))
                        self.fn_dict[fn_key] = fn

        # **2, sqrt, log (with x and 1/x feats - do we need both **2 and sqrt?)




# class to extract features from polygons in an raster
class ImageFeatureExtractor(object):
    def __init__(self, image_reader=rasterio.io.DatasetReader, plot_feat_dict={}, plot_data_gdf=gpd.GeoDataFrame()):
        ''' Class
        Parameters
        ----------
        image_reader
        plot_feat_dict
        plot_data_gdf
        '''
        self.image_reader = image_reader
        self.plot_data_gdf = plot_data_gdf
        self.im_plot_data_gdf = gpd.GeoDataFrame()


    @staticmethod
    def get_band_info(num_bands=9):
        if num_bands == 8:  # assume Wv3
            # pan_bands = np.array([1, 2, 3, 4, 5, 6])     # this needs some experimentation and may be better using the actual pan info
            pan_bands = [1, 2, 4, 5]
            band_dict = OrderedDict([('C', 0), ('B', 1), ('G', 2), ('Y', 3), ('R', 4), ('RE', 5), ('NIR', 6), ('NIR2', 7)])
            # band_dict = OrderedDict(
            #     [('B', 1), ('G', 2), ('Y', 3), ('R', 4), ('RE', 5), ('NIR', 6), ('NIR2', 7)])       # exclude C which is noisy
        elif num_bands == 9:  # assume Wv3 ms + pan in band 0
            # pan_bands = np.array([0])
            pan_bands = [2, 3, 5, 6]
            band_dict = OrderedDict(
                [('C', 1), ('B', 2), ('G', 3), ('Y', 4), ('R', 5), ('RE', 6), ('NIR', 7), ('NIR2', 8)])
            # band_dict = OrderedDict(
            #     [('B', 2), ('G', 3), ('Y', 4), ('R', 5), ('RE', 6), ('NIR', 7), ('NIR2', 8)])       # exclude C which is noisy
        else:  # assume NGI
            # pan_bands = np.array([0, 1, 2, 3])
            pan_bands = [0, 1, 2, 3]
            band_dict = OrderedDict([('R', 0), ('G', 1), ('B', 2), ('NIR', 3)])
        return pan_bands, band_dict

    @staticmethod
    def extract_patch_ms_features_ex(imbuf, mask=None, per_pixel=False, include_texture=True):
        feat_dict = OrderedDict()

        # note that ms bands have been scaled and likely will not sum back to approx pan, so it is better to use the actual pan
        pan_bands, band_dict = ImageFeatureExtractor.get_band_info(imbuf.shape[2])

        if mask is None:
            mask = np.all(imbuf>0, axis=2)  #np.ones(imbuf.shape[:2])
        mask = np.bool8(mask)
        # mask = mask & np.all(imbuf > 0, axis=2) & np.all(imbuf[:,:,], axis=2)
        imbuf_mask = np.ndarray(shape=(np.int32(mask.sum()), imbuf.shape[2]), dtype=np.float64)
        for i in range(0, imbuf.shape[2]):
            band = imbuf[:, :, i]
            imbuf_mask[:, i] = np.float64(band[mask])               # TODO HACK - check / 5000.  # 5000 is scale for MODIS / XCALIB

        # construct basic per-pixel features

        feat_dict['pan'] = np.mean(imbuf_mask[:, pan_bands], 1)
        # feat_dict['int'] = np.sum(imbuf_mask[:, np.array([2, 3, 5, 6])], 1)     #hack - remove
        if True:
            # constuct features of all possible band/pan ratios
            for key in list(band_dict.keys()):
                feat_dict[key] = imbuf_mask[:, band_dict[key]]
            poly_feat_dict = OrderedDict()
            for key1 in list(feat_dict.keys()):
                poly_feat_dict[key1] = feat_dict[key1]
                for key2 in list(feat_dict.keys()):
                    if not key2 == key1:
                        new_key = '{0}/{1}'.format(key1, key2)
                        poly_feat_dict[new_key] = old_div(feat_dict[key1],feat_dict[key2])
            # poly_feat_dict = feat_dict
        else:
            feat_dict['1/pan'] = 1. / feat_dict['pan']
            for key in list(band_dict.keys()):
                feat_dict[key] = imbuf_mask[:, band_dict[key]]
                feat_dict['1/{0}'.format(key)] = 1./imbuf_mask[:, band_dict[key]]   # invert features to get band ratios with polynomial combinations

            feat_array = np.array(list(feat_dict.values())).transpose()
            poly_feats = PolynomialFeatures(degree=2, interaction_only=False)
            poly_feat_array = np.float64(poly_feats.fit_transform(feat_array))
            poly_feat_names = np.array(poly_feats.get_feature_names())
            my_poly_feat_names = []
            for feat_name in poly_feat_names:       # messy way of re-constructing meaningful feature names
                my_feat_name = ''
                if feat_name == '1':
                    my_feat_name = '1'
                else:
                    ns = np.array(feat_name.split('x'))
                    ns = ns[ns!='']
                    for nss in ns:
                        nsss = np.array(nss.split('^'))
                        my_feat_name += '({0})'.format(list(feat_dict.keys())[int(nsss[0])])
                        if nsss.size > 1:
                            my_feat_name += '^' + nsss[1]
                my_poly_feat_names.append(my_feat_name)
                # print feat_name
                # print my_feat_name

            # remove constant features (x * 1/x)
            const_idx = np.all(np.round(poly_feat_array, 6)==1., axis=0)
            poly_feat_array = poly_feat_array[:, ~const_idx]
            poly_feat_names = poly_feat_names[~const_idx]
            my_poly_feat_names = np.array(my_poly_feat_names)[~const_idx]
            poly_feat_dict = OrderedDict(list(zip(my_poly_feat_names, poly_feat_array.transpose())))

        # add NDVI and SAVI here, to avoid making ratios of them
        SAVI_L = 0.05
        # these vals for MODIS from https://en.wikipedia.org/wiki/Enhanced_vegetation_index
        L = 1.; C1 = 6; C2 = 7.5; G = 2.5
        nir_keys = [key for key in list(band_dict.keys()) if ('NIR' in key) or ('RE' in key)]
        for i, nir_key in enumerate(nir_keys):
            nir = imbuf_mask[:, band_dict[nir_key]]
            r = imbuf_mask[:, band_dict['R']]
            ndvi = old_div((nir - r),(nir + r))
            savi = old_div((1 + SAVI_L) * (nir - r),(SAVI_L + nir + r))
            # evi = G * (nir - r) / (L + nir + C1*r - C2*imbuf_mask[:, band_dict['B']])
            # evi[np.isnan(evi)] = 0; evi[np.isinf(evi)] = 0
            post_fix = '' if nir_key == 'NIR' else '_{0}'.format(nir_key)
            poly_feat_dict['NDVI' + post_fix] = ndvi
            poly_feat_dict['SAVI' + post_fix] = savi
            # poly_feat_dict['EVI' + post_fix] = evi

        if 'NIR2' in band_dict and False:
            nir2 = imbuf_mask[:, band_dict['NIR2']]
            nir = imbuf_mask[:, band_dict['NIR']]
            poly_feat_dict['NDWI'] = old_div((nir2 - nir), (nir2 + nir))

        # poly_feat_dict['NDVI'] = (imbuf_mask[:, band_dict['NIR']] - imbuf_mask[:, band_dict['R']]) / (imbuf_mask[:, band_dict['NIR']] + imbuf_mask[:, band_dict['R']])
        # poly_feat_dict['SAVI'] = (1 + L) * (imbuf_mask[:, band_dict['NIR']] - imbuf_mask[:, band_dict['R']]) / (L + imbuf_mask[:, band_dict['NIR']] + imbuf_mask[:, band_dict['R']])
        # poly_feat_dict['EVI'] = G * (imbuf_mask[:, band_dict['NIR']] - imbuf_mask[:, band_dict['R']]) / \
        #                         (L + imbuf_mask[:, band_dict['NIR']] + C1*imbuf_mask[:, band_dict['R']] - C2*imbuf_mask[:, band_dict['B']])
        if not per_pixel:   # find mean, std dev and ?? features over the plot
            if True:
                plot_feat_dict = OrderedDict()
                for feat_key, feat_vect in poly_feat_dict.items():
                    plot_feat_dict[feat_key] = feat_vect.mean()
                    # note: here some fns are applied to per plot feats and not to per pixel feats (based on theory that only band ratios are necessary/valid per pixel)
                    plot_feat_dict['({0})^2'.format(feat_key)] = (feat_vect.mean()**2)

                    if include_texture:
                        plot_feat_dict['Std({0})'.format(feat_key)] = feat_vect.std()
                        plot_feat_dict['Entropy({0})'.format(feat_key)] = nanentropy(feat_vect)

                    if ('VI' in feat_key) or ('ND' in feat_key):        # avoid -ve logs / sqrts
                        plot_feat_dict['1+Log({0})'.format(feat_key)] = np.log10(1. + feat_vect.mean())
                        plot_feat_dict['1+({0})^.5'.format(feat_key)] = np.sqrt(1. + feat_vect.mean())
                    else:
                        plot_feat_dict['Log({0})'.format(feat_key)] = np.log10(feat_vect.mean())
                        plot_feat_dict['({0})^.5'.format(feat_key)] = np.sqrt(feat_vect.mean())
                    # plot_feat_dict['({0})^.5'.format(feat_key)] = np.sqrt(feat_vect.mean())

                    if False: # GLCM features
                        # find pan and quantise to nlevels levels
                        nlevels = 10
                        max_dn = 4000
                        pan_band = np.mean(imbuf[:,:,pan_bands], axis=2)
                        if True:   # global clip normalise
                            pan_band[pan_band < 0] = 0
                            pan_band[pan_band > max_dn] = max_dn
                            pan_band = np.uint8(old_div(((nlevels - 1) * pan_band), max_dn))
                        else:       # local min max normalise
                            pan_band = pan_band - pan_band.min()
                            pan_band = np.uint8(old_div(((nlevels-1)*pan_band),pan_band.max()))
                        cm = greycomatrix(pan_band, distances=[2, 8], angles=[0, old_div(np.pi, 2)], levels=nlevels, symmetric=True, normed=False)
                        for prop_key, prop in {'corr':'correlation', 'diss':'dissimilarity', 'hom':'homogeneity'}.items():
                            d = greycoprops(cm, prop)
                            dd = d.mean(axis=1)
                            plot_feat_dict['{0}(GLCM)'.format(prop_key)] = dd.mean()
                            plot_feat_dict['{0}(GLCM)[0]'.format(prop_key)] = dd[0]
                            plot_feat_dict['{0}(GLCM)[-1]'.format(prop_key)] = dd[-1]
            else:
                plot_feat_dict = OrderedDict()
                for feat_key, feat_vect in poly_feat_dict.items():
                    plot_feat_dict['Mean({0})'.format(feat_key)] = feat_vect.mean()
                    if ('NDVI' in feat_key) or ('SAVI' in feat_key):
                        plot_feat_dict['1+Log({0})'.format(feat_key)] = np.log10(1.+feat_vect.mean())
                    else:
                        plot_feat_dict['Log({0})'.format(feat_key)] = np.log10(feat_vect.mean())
                    plot_feat_dict['Std({0})'.format(feat_key)] = feat_vect.std()
            return plot_feat_dict
        else:
            return poly_feat_dict


    # per_pixel = True, patch_fn = su.ImPlotFeatureExtractor.extract_patch_ms_features
    def extract_all_features(self, per_pixel=False, patch_fn=extract_patch_ms_features_ex):
        # geotransform = ds.GetGeoTransform()
        # transform = osr.CoordinateTransformation(self.plot_feat_dict['spatial_ref'], gdal.osr.SpatialReference(self.image_reader.crs.to_string()))
        # transform = osr.CoordinateTransformation(gdal.osr.SpatialReference(self.plot_data_gdf.crs.to_wkt()),
        #                                          gdal.osr.SpatialReference(self.image_reader.crs.to_wkt()))

        self.plot_data_gdf = self.plot_data_gdf.to_crs(self.image_reader.crs)
        self.plot_data_gdf = self.plot_data_gdf.set_index('ID').sort_index()

        # self.im_data_df = pd.DataFrame()
        im_plot_data_dict = {}

        im_plot_count = 0
        max_thumbnail_vals = np.zeros((self.image_reader.count))

        # loop through plot polygons
        for plot_id, plot in self.plot_data_gdf.iterrows():
            # convert polygon to mask with rio
            plot_mask, plot_transform, plot_window = raster_geometry_mask(self.image_reader, [plot['geometry']], crop=True,
                                                                     all_touched=False)
            plot_cnrs_pixel =  np.fliplr(np.array(plot_window.toranges()).transpose())
            plot_mask = ~plot_mask

            # check plot window lies inside image
            # TODO - is there a rio shortcut for this?
            if not (np.all(plot_cnrs_pixel >= 0) and np.all(plot_cnrs_pixel[:, 0] < self.image_reader.width) \
                    and np.all(plot_cnrs_pixel[:, 1] < self.image_reader.height)):  # and plot.has_key('Yc') and plot['Yc'] > 0.:
                logger.warning(f'Excluding plot {plot["ID"]} - outside image extent')
                continue

            im_buf = self.image_reader.read(window=plot_window)     # read plot ROI from image
            im_buf = np.moveaxis(im_buf, 0, 2)  # TODO get rid of this somehow eg change all imbuf axis orderings to bands first

            if np.all(im_buf == 0) and not patch_fn == self.extract_patch_clf_features:
                logger.warning(f'Excluding plot {plot["ID"]} - all pixels are zero')
                continue

            # amend plot_mask to exclude NAN and -ve pixels
            if False: #patch_fn == self.extract_patch_clf_features:
                plot_mask = plot_mask & np.all(~np.isnan(im_buf), axis=2)
            else:
                plot_mask = plot_mask & np.all(im_buf > 0, axis=2) & np.all(~np.isnan(im_buf), axis=2)

            im_feat_dict = patch_fn(im_buf.copy(), mask=plot_mask, per_pixel=per_pixel)    # extract image features for this plot

            im_data_dict = im_feat_dict.copy()      # copy plot data into feat dict
            for k, v in plot.items():
                im_data_dict[k] = v

            # calculate versions of ABC and AGC with actual polygon area, rather than theoretical plot sizes
            if 'Abc' in plot and 'LitterCHa' in plot:
                litterCHa = np.max([plot['LitterCHa'], 0.])
                abc = np.max([plot['Abc'], 0.])
                im_data_dict['AbcHa2'] = abc * (100. ** 2) /  plot['geometry'].area
                im_data_dict['AgcHa2'] = litterCHa + im_data_dict['AbcHa2']

            # create and store plot thumbnail
            thumbnail = np.float32(im_buf.copy())
            thumbnail[~plot_mask] = 0.
            im_data_dict['thumbnail'] = thumbnail

            im_plot_data_dict[plot_id] = im_data_dict

            # store max thumbnail vals for scaling later
            max_val = np.percentile(thumbnail, 98., axis=(0,1))
            max_thumbnail_vals[max_val > max_thumbnail_vals] = max_val[max_val > max_thumbnail_vals]
            im_plot_count += 1

            log_dict = {'ABC': 'Abc' in plot, 'Num zero pixels': (im_buf == 0).sum(), 'Num -ve pixels': (im_buf < 0).sum(),
                'Num nan pixels': np.isnan(im_buf).sum()}
            logger.info(', '.join([f'Plot {plot_id}'] + ['{}: {}'.format(k, v) for k, v in log_dict.items()]))

        logger.info('Processed {0} plots'.format(im_plot_count))

        # scale thumbnails for display
        for im_data_dict in im_plot_data_dict.values():
            thumbnail = im_data_dict['thumbnail']
            for b in range(0, self.image_reader.count):
                thumbnail[:, :, b] /= max_thumbnail_vals[b]
                thumbnail[:, :, b][thumbnail[:, :, b] > 1.] = 1.
            im_data_dict['thumbnail'] = thumbnail

        # create MultiIndex column labels that separate features from other data
        data_labels = ['feats']*len(im_feat_dict) + ['data']*(len(im_data_dict) - len(im_feat_dict))
        columns = pd.MultiIndex.from_arrays([data_labels, list(im_data_dict.keys())], names=['high','low'])

        self.im_plot_data_gdf = gpd.GeoDataFrame.from_dict(im_plot_data_dict, orient='index')
        self.im_plot_data_gdf.columns = columns
        self.im_plot_data_gdf[('data','ID')] = self.im_plot_data_gdf.index

        return self.im_plot_data_gdf

class FeatureSelector(object):
    def __init__(self):
        return
    @staticmethod
    def forward_selection(feat_df, y, max_num_feats=0, model=linear_model.LinearRegression(),
                          score_fn=None, cv=None):

        if max_num_feats == 0:
            max_num_feats = feat_df.shape[1]
        selected_feats_df = gpd.GeoDataFrame(index=feat_df.index)   # remember order items are added
        selected_scores = []
        available_feats_df = feat_df.copy()

        logger.info('Forward selection: ')
        if score_fn is None:
            logger.info('Using negative RMSE score')
        else:
            logger.info('Using user score')
        while selected_feats_df.shape[1] < max_num_feats:
            best_score = -np.inf
            best_feat = []
            for feat_key, feat_vec in available_feats_df.iteritems():
                test_feats_df = pd.concat((selected_feats_df, feat_vec), axis=1, ignore_index=False) # list(selected_feats.values()) + [feat_vec]
                scores, predicted = FeatureSelector.score_model(test_feats_df, y, model=model,
                                                                score_fn=score_fn, cv=cv, find_predicted=False)
                if score_fn is None:
                    score = -np.sqrt((scores['test_-RMSE']**2).mean())       # NB not mean(sum(RMSE))
                else:
                    score = scores['test_user'].mean()

                if score > best_score:
                    best_score = score
                    best_feat = list(feat_vec)
                    best_key = feat_key
            selected_feats_df[best_key] = best_feat
            selected_scores.append(best_score)
            available_feats_df.pop(best_key)
            logger.info('Feature {0} of {1}: {2}, Score: {3:.1f}'.format(selected_feats_df.shape[1], max_num_feats, best_key, best_score))
        # logger.info(' ')
        selected_scores = np.array(selected_scores)
        selected_feat_keys = selected_feats_df.columns
        best_selected_feat_keys = selected_feat_keys[:np.argmax(selected_scores) + 1]
        logger.info('Best score: {0}'.format(selected_scores.max()))
        logger.info('Num feats at best score: {0}'.format(np.argmax(selected_scores) + 1))
        logger.info('Feat keys at best score: {0}'.format(best_selected_feat_keys))

        return selected_feats_df, selected_scores


    @staticmethod
    def ranking(feat_df, y, model=linear_model.LinearRegression(), score_fn=None, cv=None):
        # X, feat_keys_mod, y = self.get_feat_array(y_key=y_feat_key)

        logger.info('Ranking: ')
        if score_fn is None:
            logger.info('Using negative RMSE score')
        else:
            logger.info('Using user score')

        feat_scores = []
        for i, (feat_key, feat_vec) in enumerate(feat_df.iteritems()):
            logger.info('Scoring feature {0} of {1}'.format(i+1, feat_df.shape[1]))

            scores, predicted = FeatureSelector.score_model(pd.DataFrame(feat_vec), y, model=model, score_fn=score_fn, cv=cv, find_predicted=False)

            if score_fn == None:
                score = -np.sqrt((scores['test_-RMSE']**2).mean())
            else:
                score = scores['test_user'].mean()
            feat_scores.append(score)

        feat_scores = np.array(feat_scores)

        logger.info('Best score: {0}'.format(feat_scores.max()))
        logger.info('Best feat: {0}'.format(feat_df.columns[np.argmax(feat_scores)]))
        return feat_scores

    @staticmethod
    def score_model(X, y, model=linear_model.LinearRegression(), score_fn=None,
                    cv=None, find_predicted=True, print_scores=False):

        predicted = None
        if cv is None:
            cv = len(y)        # Leave one out

        if score_fn is not None:
            scoring = {#'R2': make_scorer(metrics.r2_score),        # R2 in cross validation is suspect
                       '-RMSE': make_scorer(lambda y, pred: -np.sqrt(metrics.mean_squared_error(y, pred))),
                       'user': make_scorer(score_fn)}
        else:
            scoring = {'-RMSE': make_scorer(lambda y, pred: -np.sqrt(metrics.mean_squared_error(y, pred)))}

        # TO DO: this does k-fold.  We should try stratified k-fold with degradation strata.
        scores = cross_validate(model, X, y, scoring=scoring, cv=cv, n_jobs=-1)

        if print_scores:
            rmse_ci = np.percentile(-scores['test_-RMSE'], [5, 95])
            # r2_ci = np.percentile(-scores['test_R2'], [5, 95])
            logger.info('RMSE mean: {0:.4f}, std: {1:.4f}, 5-95%: {2:.4f} - {3:.4f}'.format(-scores['test_-RMSE'].mean(),
                    scores['test_-RMSE'].std(), rmse_ci[0], rmse_ci[1]))
            # logger.info('R2 mean: {0:.4f}, std: {1:.4f}, 5-95%: {2:.4f} - {3:.4f}'.format(scores['test_R2'].mean(), scores['test_R2'].std(),
            #                                                                               r2_ci[0], r2_ci[1]))
        if find_predicted:
            predicted = cross_val_predict(model, X, y, cv=cv)  #)
            scores['R2_stacked'] = metrics.r2_score(y, predicted)   # Also suspect for validation, but better than cross validated R2
            if print_scores:
                logger.info('R2 (stacked): {0:.4f}'.format(scores['R2_stacked']))
        return scores, predicted




# params: calib_plots from >1 data set
#       model_data_plots from >1 data set
#       (for now the above are the same thing)
#       1 data_set is specified as fitted one, the rest are tests, this can also be done sort of cross-validated
#       a model spec i.e. feature indices and model type
#       num calib plots to use


class ApplyLinearModel(object):
    def __init__(self, in_file_name='', out_file_name='', model=linear_model.LinearRegression, model_keys=[],
                 feat_ex_fn=ImageFeatureExtractor.extract_patch_ms_features_ex, num_bands=9, save_feats=False):
        self.in_file_name = in_file_name
        self.out_file_name = out_file_name
        self.model = model
        self.model_keys = model_keys
        self.feat_ex_fn = feat_ex_fn
        self.pan_bands = []
        self.band_dict = {}
        self.save_feats = save_feats
        self.nodata = np.nan

        input_ds = None
        output_ds = None

    # def __del__(self):
    #     self.Close()

    # Contruct fn ptrs for each feature from strings
    # (so we don't have to find the entire feature library)
    # Examples:
    #     'Log(R/pan)',
    #     'Std(pan/NIR2)',
    #     'Log(B)',
    #     'Entropy(NIR/Y)',
    #     '(NIR/Y)^2',
    # def ConstructFeatEx(self, model_keys=[], num_bands=9):
    @staticmethod
    def construct_feat_ex_fn(model_key='', pan_bands=None, band_dict=None):
        win_fn_list = []
        inner_str_list = []
        inner_fn_list = []
        # find outer fn
        ks = re.split('\(|\)', model_key.lower())
        if len(ks) == 1:    # no brackets
            inner_str = ks[0]
            win_fn = lambda x: np.nanmean(x, axis=(0, 2))
        else:
            inner_str = ks[1]
            # outer fn
            if not ks[0]:  # nothing before (
                if ks[-1]:  # raise to power
                    kp = re.split('\^', ks[-1])
                    win_fn = lambda x: np.power(np.nanmean(x, axis=(0,2)), eval(kp[1]))
                else:  # straight mean
                    win_fn = lambda x: np.nanmean(x, axis=(0,2))
            elif ks[0] == 'std':
                win_fn = lambda x: np.nanstd(x, axis=(0,2))
            elif ks[0] == 'entropy':
                win_fn = lambda x: nanentropy(x, axis=(0,2)).astype('float32')
            elif ks[0] == 'log':
                win_fn = lambda x: np.log10(np.nanmean(x, axis=(0,2)))

        # find inner fn i.e. band ratio
        iss = re.split('/', inner_str)
        if len(iss) > 1:
            if iss[0] == 'pan':
                inner_fn = lambda pan, x: pan / x[band_dict[iss[1].upper()], :, :]
            elif iss[1] == 'pan':
                inner_fn = lambda pan, x: x[band_dict[iss[0].upper()], :, :] / pan
            else:
                inner_fn = lambda pan, x: x[band_dict[iss[0].upper()], :, :] / x[band_dict[iss[1].upper()], :, :]
        elif 'savi' in inner_str:   # TO DO: add in underscore options
            if 'nir2' in inner_str:
                inner_fn = lambda pan, x: (1 + 0.05) * (x[band_dict['NIR2'], :, :] - x[band_dict['R'], :, :]) / (
                            0.05 + x[band_dict['NIR2'], :, :] + x[band_dict['R'], :, :])
            elif 're' in inner_str:
                inner_fn = lambda pan, x: (1 + 0.05) * (x[band_dict['RE'], :, :] - x[band_dict['R'], :, :]) / (0.05 + x[band_dict['RE'], :, :] + x[band_dict['R'], :, :])
            else:
                inner_fn = lambda pan, x: (1 + 0.05) * (x[band_dict['NIR'], :, :] - x[band_dict['R'], :, :]) / (0.05 + x[band_dict['NIR'], :, :] + x[band_dict['R'], :, :])

        elif 'ndvi' in inner_str:
            if 'nir2' in inner_str:
                inner_fn = lambda pan, x: (x[band_dict['NIR2'], :, :] - x[band_dict['R'], :, :]) / (x[band_dict['NIR2'], :, :] + x[band_dict['R'], :, :])
            elif 're' in inner_str:
                inner_fn = lambda pan, x: (x[band_dict['RE'], :, :] - x[band_dict['R'], :, :]) / (x[band_dict['RE'], :, :] + x[band_dict['R'], :, :])
            else:
                inner_fn = lambda pan, x: (x[band_dict['NIR'], :, :] - x[band_dict['R'], :, :]) / (x[band_dict['NIR'], :, :] + x[band_dict['R'], :, :])
        else:
            if iss[0] == 'pan':
                inner_fn = lambda pan, x: pan
            else:
                inner_fn = lambda pan, x: x[band_dict[iss[0].upper()], :, :]

        return inner_str, win_fn, inner_fn

    @staticmethod
    def construct_feat_ex_fns(model_keys=[], num_bands=9):
        import re
        pan_bands, band_dict = mdl.ImageFeatureExtractor.get_band_info(num_bands)

        win_fn_list = []
        inner_str_list = []
        inner_fn_list = []
        for model_key in model_keys:
            # find outer fn
            inner_str, win_fn, inner_fn = ApplyLinearModel.construct_feat_ex_fn(model_key, pan_bands=pan_bands, band_dict=band_dict)
            win_fn_list.append(win_fn)
            inner_str_list.append(inner_str)
            inner_fn_list.append(inner_fn)
        # inner fn
        return win_fn_list, inner_fn_list

    # @staticmethod
    # def rolling_window(a, window, step_size=1):
    #     shape = a.shape[:-1] + (a.shape[-1] - window + 1 - step_size + 1, window)
    #     strides = a.strides + (a.strides[-1] * step_size,)
    #     return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides, writeable=False)

    @staticmethod
    def rolling_window(a, window, step_size=1):
        shape = a.shape[:-1] + (int(1 + (a.shape[-1] - window) / step_size), window)
        strides = a.strides[:-1] + (step_size * a.strides[-1], a.strides[-1])
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides, writeable=False)

    def feat_ex(self, im_buf=[]):
        return

    def create(self, win_size=(1, 1), step_size=(1, 1)):
        with rasterio.Env():
            with rasterio.open(self.in_file_name, 'r') as in_ds:
                if self.save_feats:
                    out_bands = len(self.model_keys) + 1
                else:
                    out_bands = 1
                out_profile = in_ds.profile
                out_size = np.floor([1 + (in_ds.width - win_size[0])/step_size[0], 1 + (in_ds.height - win_size[1])/step_size[1]]).astype('int32')
                out_profile.update(dtype=rasterio.float32, count=out_bands, compress='deflate', driver='GTiff', width=out_size[0], height=out_size[1], nodata=self.nodata)

                out_profile['transform'] = out_profile['transform']*rasterio.Affine.scale(step_size[0])
                if (out_size[0]/out_profile['blockxsize'] < 10) | (out_size[1]/out_profile['blockysize'] < 10):
                    out_profile['tiled'] = False
                    out_profile['blockxsize'] = out_size[0]
                    out_profile['blockysize'] = 1
                # out_transform = np.array(rasterio.Affine(out_profile['transform']).to_gdal())
                # out_transform[0] = ['transform'][4] * step_size[0]
                # out_transform[4] = ['transform'][4] * step_size[1]
                # to do: find geotransform based on out width/height

                # self.win_fn_list, self.band_ratio_list = AgcMap.construct_feat_ex_fns(self.model_keys, num_bands=in_ds.count)

                with rasterio.open(self.out_file_name, 'w', **out_profile) as out_ds:
                    pan_bands, band_dict = ImageFeatureExtractor.get_band_info(in_ds.count)
                    win_off = np.floor(np.array(win_size)/(2*step_size[0])).astype('int32')
                    prog_update = 10
                    for cy in range(0, in_ds.height - win_size[1] + 1, step_size[1]):     #12031): #
                        # read row of windows into mem and slide the win on this rather than the actual file
                        # NB rasterio index is x,y, numpy index is row, col (i.e. y,x)

                        # in_buf = np.zeros((win_size[1], in_ds.width), dtype=in_ds.dtypes[0])
                        in_win = Window(0, cy, out_size[0]*win_size[0], win_size[1])
                        bands = list(range(1, in_ds.count + 1))
                        in_buf = in_ds.read(bands, window=in_win).astype(rasterio.float32)  # NB bands along first dim

                        # TO DO:  deal with -ve vals i.e. make all nans, mask out or something
                        pan = in_buf[pan_bands, :, :].mean(axis=0)
                        in_nan_mask = np.any(in_buf <= 0, axis=0) | np.any(pan == 0, axis=0)  # this is overly conservative but neater/faster
                        # agc_buf = self.model.intercept_ * np.ones((1, in_ds.width - win_size[0] + 1), dtype=out_ds.dtypes[0])
                        agc_buf = self.model.intercept_ * np.ones((out_size[0]), dtype=out_ds.dtypes[0])
                        out_win = Window(win_off[0], int(cy/step_size[0]) + win_off[1], out_size[0], 1)
                        # for i, (win_fn, band_ratio_fn) in enumerate(zip(self.win_fn_list, self.band_ratio_list)):
                        for i, model_key in enumerate(self.model_keys):
                            inner_str, win_fn, band_ratio_fn = ApplyLinearModel.construct_feat_ex_fn(model_key, pan_bands=pan_bands,
                                                                                                     band_dict=band_dict)
                            band_ratio = band_ratio_fn(pan, in_buf)
                            band_ratio[in_nan_mask] = np.nan           # to exclude from window stats/fns
                            feat_buf = win_fn(ApplyLinearModel.rolling_window(band_ratio, win_size[0], step_size=step_size[0])) * self.model.coef_[i]
                            agc_buf += feat_buf
                            if i==0:
                                first_feat_buf = feat_buf
                            if self.save_feats:
                                if feat_buf.size == 1:
                                    feat_buf = np.zeros((1, out_size[0]), dtype=out_ds.dtypes[0])
                                else:
                                    feat_buf[np.isinf(feat_buf) | np.isnan(feat_buf)] = 0
                                out_ds.write(feat_buf.reshape(1,-1), indexes=2+i, window=out_win)

                        if True:    # lower limit of 0, and set noisy/invalid pixels to nodata, so that they can be filled in a post-processing step
                            nodata_mask = np.isinf(agc_buf) | np.isnan(agc_buf) # | (agc_buf > 70)        # 55/70 comes from looking ta the histogram
                            agc_buf[nodata_mask] = self.nodata
                            # agc_buf = sig.medfilt(agc_buf, kernel_size=3).astype(np.float32)
                            # agc_buf[(agc_buf < 0)] = 0
                        else:   # inelegant hack for suspect values
                            nodata_mask = np.isinf(agc_buf) | np.isnan(agc_buf) | (agc_buf < 0) | (agc_buf > 100)
                            agc_buf[nodata_mask] = first_feat_buf[nodata_mask]
                        # out_win = Window(win_off[0], cy + win_off[1], in_ds.width - win_size[0] + 1, 1)
                        out_ds.write(agc_buf.reshape(1,-1), indexes=1, window=out_win)

                        if np.floor(old_div(100*cy, in_ds.height)) > prog_update:
                            print('.', end=' ')
                            prog_update += 10
                            # break
                    print(' ')

    def post_proc(self):
        from rasterio.windows import Window
        from rasterio import fill
        with rasterio.Env():
            with rasterio.open(self.out_file_name, 'r') as in_ds:
                out_profile = in_ds.profile
                out_profile.update(count=1)
                split_ext = os.path.splitext(self.out_file_name)
                out_file_name = '{0}_postproc{1}'.format(split_ext[0], split_ext[1])
                with rasterio.open(out_file_name, 'w', **out_profile) as out_ds:

                    if (not out_profile['tiled']) or (np.prod(in_ds.shape) < 10e6):
                        in_windows = enumerate([Window(0,0,in_ds.width, in_ds.height)])        # read whole raster at once
                    else:
                        in_windows = in_ds.block_windows(1)                         # read in blocks
                    # in_masks = in_ds.read_masks()
                    # sieved_msk = sieve(in_mask, size=500)
                    for ji, block_win in in_windows:
                        in_block = in_ds.read(1, window=block_win, masked=True)

                        in_block[in_block < 0] = 0
                        in_block.mask = (in_block.mask.astype(np.bool) | (in_block > 95) | (in_block < 0)).astype(rasterio.uint8)

                        in_mask = in_block.mask.copy()
                        sieved_msk = sieve(in_mask.astype(rasterio.uint8), size=2000)

                        out_block = fill.fillnodata(in_block, mask=None, max_search_distance=20, smoothing_iterations=1)
                        out_block[sieved_msk.astype(np.bool)] = self.nodata
                        # out_block = signal.medfilt2d(in_block, kernel_size=3)
                        out_ds.write(out_block, indexes=1, window=block_win)
                        # out_ds.write_mask(~sieved_msk, window=block_win)

                # with rasterio.open(out_file_name, 'r+', **out_profile) as out_ds:
                #     tmp_masks = out_ds.read_masks()