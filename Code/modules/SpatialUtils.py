from __future__ import print_function
from __future__ import division
from builtins import zip
from builtins import str
from builtins import range
from builtins import object
from past.utils import old_div
import sys
import gdal
import ogr
import numpy as np
import osr
import pylab
from rasterio.rio.options import nodata_opt
from scipy import stats as stats
import scipy.signal as signal
from matplotlib import patches
from sklearn import linear_model, metrics
from sklearn.model_selection import cross_val_predict, cross_validate
import collections
from collections import OrderedDict
from sklearn.preprocessing import PolynomialFeatures
# from pandas import DataFrame as pd
from skimage.feature import greycomatrix, greycoprops
from skimage import data
import matplotlib.pyplot as pyplot

# Python Imaging Library imports
from PIL import Image
from PIL import ImageDraw
import os
import numpy as np
import rasterio
import re
from rasterio.features import sieve


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
    else:        # hack for 2D slices of 3D array (on a RollingWindow 3D array)
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


def entropy(x, along_axis=None):
    """
    x is assumed to be an (nsignals, nsamples) array containing integers between
    0 and n_unique_vals
    """
    # quantise x
    nbins = 100
    x = x.flatten()
    x = x-x.min()
    x = np.int32(np.round(old_div((nbins*x),x.max())))

    p = np.array([old_div(np.size(x[x == i]), (1.0 * x.size)) for i in np.unique(x)])
    # compute Shannon entropy in bits
    return -np.sum(p * np.log2(p))

def scatter_ds(data, x_col=None, y_col=None, class_col=None, label_col=None, thumbnail_col=None, do_regress=True,
               x_label=None, y_label=None, xfn=lambda x: x, yfn=lambda y: y):
    """

    Parameters
    ----------
    data
    x_col
    y_col
    class_col
    label_col
    thumbnail_col
    do_regress
    x_label
    y_label
    xfn
    yfn

    Returns
    -------

    """
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
        colours = ['g', 'tab:orange', 'r', 'b', 'y', 'k', 'm']

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
            pylab.plot(xfn(class_data[x_col]), yfn(class_data[y_col]), markerfacecolor=colour, marker='.', label=class_label, linestyle='None',
                       markeredgecolor=colour)
        for rowi, row in class_data.iterrows():
            xx = xfn(row[x_col])
            yy = yfn(row[y_col])
            if label_col is not None:   # add a text label for each point
                pylab.text(xx - .0015, yy - .0015, row[label_col],
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
                handles[class_i] = ax.add_patch(patches.Rectangle(extent[0], extent[2], xd/ims, yd/ims,
                                                fill=False, edgecolor=colour, linewidth=2.))

    if do_regress:  # find and display regression error stats
        (slope, intercept, r, p, stde) = stats.linregress(x, y)
        scores, predicted = FeatureSelector.score_model(x[:,None], y[:,None], model=linear_model.LinearRegression(),
                                                        find_predicted=True, cv=len(x), print_scores=True)

        pylab.text((xlim[0] + xd * 0.7), (ylim[0] + yd * 0.05), '$R^2$ = {0:.2f}'.format(np.round(scores['R2_stacked'], 2)),
                   fontdict={'size': 12})
        yr = np.array(xlim)*slope + intercept
        pyplot.plot(xlim, yr, 'k--', lw=2, zorder=-1)

        yhat = x * slope + intercept
        rmse = np.sqrt(np.mean((y - yhat) ** 2))

        print('RMSE = {0:.4f}'.format(rmse))
        print('LOOCV RMSE = {0:.4f}'.format(np.sqrt(-scores['test_user'].mean())))
        print('R^2  = {0:.4f}'.format(r ** 2))
        print('Stacked R^2  = {0:.4f}'.format(scores['R2_stacked']))
        print('P (slope=0) = {0:f}'.format(p))
        print('Slope = {0:.4f}'.format(slope))
        print('Std error of slope = {0:.4f}'.format(stde))
    else:
        r = np.nan
        rmse = np.nan

    if x_label is not None:
        pyplot.xlabel(x_label, fontdict={'size': 12})
    else:
        pylab.xlabel(x_col, fontdict={'size': 12})

    if y_label is not None:
        pyplot.ylabel(y_label, fontdict={'size': 12})
    else:
        pylab.xlabel(y_col, fontdict={'size': 12})

    if n_classes > 1:
        if not thumbnail_col is None:
            pylab.legend(handles, classes, fontsize=12)
        else:
            pylab.legend(classes, fontsize=12)
    return r ** 2, rmse


def scatter_plot(x, y, class_labels=None, labels=None, thumbnails=None, do_regress=True, xlabel=None, ylabel=None,
                 xfn=lambda x: x, yfn=lambda y: y):
    # x = np.array([xfn(plot[x_feat_key]) for plot in im_feat_dict.values()])
    if type(x[0]) is np.ndarray:  # this is pixel data and requires concat to flatten it
        cfn = lambda x: np.hstack(x)[::5]
        thumbnails = None
    else:
        cfn = lambda x: x

    # if xfn is not None:
    #     x = xfn(x)
    # y = np.array([yfn(plot[y_feat_key]) for plot in im_feat_dict.values()])
    # if type(x[0]) is np.ndarray:
    #     ycfn = lambda x: np.concatenate(x)
    # else:
    #     ycfn = lambda x: x

    # if yfn is not None:
    #     y = yfn(y)

    # if show_class_labels == True:
    #     class_labels = np.array([plot[class_key] for plot in im_feat_dict.values()])
    # else:
    #     class_labels = np.zeros(x.__len__())
    # if show_thumbnails == True:
    #     thumbnails = np.array([plot['thumbnail'] for plot in im_feat_dict.values()])
    #
    # if show_labels == True:
    #     labels = np.array([plot['ID'] for plot in im_feat_dict.values()])
    if class_labels is None:
        class_labels = np.zeros(x.__len__())
    classes = np.unique(class_labels)
    if 'Intact' in classes and 'Moderate' in classes and 'Degraded' in classes and classes.size==3:
        classes = np.array(['Degraded', 'Moderate', 'Intact'])

    if classes.__len__() == 1:
        # colours = [u'#1f77b4']
        colours = [(0., 0., 0.)]
    else:
        # colours = ['r', 'm', 'b', 'g', 'y', 'k', 'o']
        colours = ['r', 'tab:orange', 'g', 'b', 'y', 'k', 'm']

    ylim = [np.min(cfn(y)), np.max(cfn(y))]
    xlim = [np.min(cfn(x)), np.max(cfn(x))]
    xd = np.diff(xlim)[0]
    yd = np.diff(ylim)[0]

    # pylab.figure()
    pylab.axis('tight')
    pylab.axis(np.concatenate([xlim, ylim]))
    # pylab.hold('on')
    ax = pylab.gca()
    handles = np.zeros(classes.size).tolist()
    #

    for ci, (class_label, colour) in enumerate(zip(classes, colours[:classes.__len__()])):
        class_idx = class_labels == class_label
        if thumbnails is None:
            pylab.plot(cfn(x[class_idx]), cfn(y[class_idx]), markerfacecolor=colour, marker='.', label=class_label, linestyle='None',
                       markeredgecolor=colour)

        for xyi, (xx, yy) in enumerate(zip(x[class_idx], y[class_idx])):  # , np.array(plot_names)[class_idx]):
            if type(xx) is np.ndarray:
                xx = xx[0]
            if type(yy) is np.ndarray:
                yy = yy[0]
            if not labels is None:
                pylab.text(xx - .0015, yy - .0015, np.array(labels)[class_idx][xyi],
                           fontdict={'size': 9, 'color': colour, 'weight': 'bold'})

            if not thumbnails is None:
                imbuf = np.array(thumbnails)[class_idx][xyi]
                band_idx = [0, 1, 2]
                if imbuf.shape[2] == 8:  # guess wv3
                    band_idx = [4, 2, 1]

                ims = 20.
                extent = [xx - old_div(xd, (2 * ims)), xx + old_div(xd, (2 * ims)), yy - old_div(yd, (2 * ims)), yy + old_div(yd, (2 * ims))]
                # pylab.imshow(imbuf[:, :, :3], extent=extent, aspect='auto')  # zorder=-1,
                pylab.imshow(imbuf[:, :, band_idx], extent=extent, aspect='auto')  # zorder=-1,
                handles[ci] = ax.add_patch(
                    patches.Rectangle((xx - old_div(xd, (2 * ims)), yy - old_div(yd, (2 * ims))), old_div(xd, ims), old_div(yd, ims), fill=False,
                                      edgecolor=colour, linewidth=2.))
                # pylab.plot(mPixels[::step], dRawPixels[::step], color='k', marker='.', linestyle='', markersize=.5)
        if do_regress and classes.__len__() > 1 and False:
            (slope, intercept, r, p, stde) = stats.linregress(cfn(x[class_idx]), cfn(y[class_idx]))
            pylab.text(xlim[0] + xd * 0.7, ylim[0] + yd * 0.05 * (ci + 2),
                       '{1}: $R^2$ = {0:.2f}'.format(np.round(r ** 2, 2), classes[ci]),
                       fontdict={'size': 10, 'color': colour})


    if do_regress:
        (slope, intercept, r, p, stde) = stats.linregress(cfn(x), cfn(y))
        scores, predicted = FeatureSelector.score_model(cfn(x)[:,None], cfn(y)[:,None], model=linear_model.LinearRegression(), find_predicted=True, cv=len(x), print_scores=True)
        r2_stacked = scores['R2_stacked']
        rmse = np.abs(scores['test_user']).mean()

        pylab.text((xlim[0] + xd * 0.7), (ylim[0] + yd * 0.05), '$R^2$ = {0:.2f}'.format(np.round(scores['R2_stacked'], 2)),
                   fontdict={'size': 12})
        print('RMSE^2 = {0:.4f}'.format(rmse))
        print('R^2 = {0:.4f}'.format(r ** 2))
        print('P (slope=0) = {0:f}'.format(p))
        print('Slope = {0:.4f}'.format(slope))
        print('Std error of slope = {0:.4f}'.format(stde))
        yhat = cfn(x) * slope + intercept
        rmse = np.sqrt(np.mean((cfn(y) - yhat) ** 2))
        print('RMS error = {0:.4f}'.format(rmse))

        yr = [0, 0]
        yr[0] = xlim[0]*slope + intercept
        yr[1] = xlim[1]*slope + intercept
        pylab.plot(xlim, yr, 'k--', lw=2, zorder=-1)
    else:
        r = np.nan
        rmse = np.nan

    if xlabel is not None:
        pylab.xlabel(xlabel, fontdict={'size': 12})
    # else:
    #     pylab.xlabel(x_feat_key, fontdict={'size': 12})
    if ylabel is not None:
        pylab.ylabel(ylabel, fontdict={'size': 12})
    # else:
    #     pylab.ylabel(y_feat_key, fontdict={'size': 12})
    # pylab.ylabel(yf)
    # pylab.grid()
    if classes.size > 1:
        if not thumbnails is None:
            pylab.legend(handles, classes, fontsize=12)
        else:
            pylab.legend(classes, fontsize=12)
    return r ** 2, rmse

class GdalImageReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        if not os.path.exists(file_name):
            raise Exception("File does not exist: {0}".format(file_name))
        self.ds = None
        self.__open()

    def __open(self):
        self.ds = gdal.OpenEx(self.file_name, gdal.OF_RASTER)
        if self.ds is None:
            raise Exception("Could not open {0}".format(self.file_name))

        print('Driver: {0}'.format(self.ds.GetDriver().LongName))
        self.width = self.ds.RasterXSize
        self.height = self.ds.RasterYSize
        self.num_bands = self.ds.RasterCount
        print('Size: {0} x {1} x {2} (width x height x bands)'.format(self.ds.RasterXSize, self.ds.RasterYSize, self.ds.RasterCount))
        print('Projection: {0}'.format(self.ds.GetProjection()))
        self.spatial_ref = osr.SpatialReference(self.ds.GetProjection())
        self.geotransform = self.ds.GetGeoTransform()
        if not self.geotransform is None:
            self.origin = np.array([self.geotransform[0], self.geotransform[3]])
            print('Origin = ({0}, {1})'.format(self.geotransform[0], self.geotransform[3]))
            print('Pixel Size =  = ({0}, {1})'.format(self.geotransform[0], self.geotransform[3]))
            print('Pixel Size = ({0}, {1})'.format(self.geotransform[1], self.geotransform[5]))
            self.pixel_size = np.array([self.geotransform[1], self.geotransform[5]])
        else:
            self.origin = np.array([0, 0])
            self.pixel_size = np.array([1, 1])
        self.gdal_dtype = self.ds.GetRasterBand(1).DataType
        if self.gdal_dtype == gdal.GDT_UInt16:
            self.dtype = np.uint16
        elif self.gdal_dtype == gdal.GDT_Int16:
            self.dtype = np.int16
        if self.gdal_dtype == gdal.GDT_Float32:
            self.dtype = np.float32
        elif self.gdal_dtype == gdal.GDT_Float64:
            self.dtype = np.float64
        else:
            self.dtype = np.float32

        self.block_size = self.ds.GetRasterBand(1).GetBlockSize()
        self.image_array = None

    def cleanup(self):
        self.image_array = None
        self.ds = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    def __del__(self):
        self.cleanup()

    def world_to_pixel(self, x, y):
        row = old_div((y - self.origin[1]),self.pixel_size[1])     # row
        col =  old_div((x - self.origin[0]),self.pixel_size[0])    # col
        return (col, row)

    def pixel_to_world(self, col, row):
        y = row * self.pixel_size[1] + self.origin[1]
        x = col * self.pixel_size[0] + self.origin[0]
        return (col, row)

    def read_image(self):
        # the below orders pixels by band, row, col but we want band as last dimension
        # self.image_array = self.ds.ReadAsArray(buf_type=self.gdal_dtype)
        self.image_array = self.read_image_roi()
        return self.image_array

    def read_image_roi(self, col_range=None, row_range=None, band_range=None):
        if row_range is None:
            row_range = [0, self.height]
        if col_range is None:
            col_range = [0, self.width]
        if band_range is None:
            band_range = [0, self.num_bands]

        # check ranges
        for drange, drange_max in zip([row_range, col_range, band_range], [self.height, self.width, self.num_bands]):
            drange[0] = np.maximum(0, drange[0])
            drange[1] = np.minimum(drange_max, drange[1])

        image_roi = np.zeros((np.diff(row_range)[0], np.diff(col_range)[0], np.diff(band_range)[0]), dtype=self.dtype)
        for bi in range(band_range[0], band_range[1]):
            image_roi[:, :, bi] = self.ds.GetRasterBand(bi + 1).ReadAsArray(int(col_range[0]), int(row_range[0]), int(np.diff(col_range)[0]),
                                                                 int(np.diff(row_range)[0]), buf_type=self.gdal_dtype)
        return image_roi

class GdalVectorReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        if not os.path.exists(file_name):
            raise Exception("File does not exist: {0}".format(file_name))
        self.ds = None
        self.__open()

    def __open(self):
        self.ds = gdal.OpenEx(self.file_name, gdal.OF_VECTOR)
        if self.ds is None:
            raise Exception("Could not open {0}".format(self.file_name))
        self.num_layers = self.ds.GetLayerCount()
        self.layer_dict = {}

    def cleanup(self):
        self.layer_dict = None
        self.ds = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def __del__(self):
        self.cleanup()

    def read(self, id_field = None):
        self.ds.ResetReading()
        for li in range(0, self.num_layers):
            layer = self.ds.GetLayerByIndex(li)
            # self.layers.append(layer)
            layer.ResetReading()
            feat_defn = layer.GetLayerDefn()
            feat_dict = {}
            print('Reading feats in layer: {0}'.format(layer.GetName()))
            for feati, feat in enumerate(layer):
                # print '.',
                fdict = {}
                for fieldi in range(feat_defn.GetFieldCount()):
                    field_defn = feat_defn.GetFieldDefn(fieldi)
                    fdict[field_defn.GetName()] = feat.GetField(fieldi)

                    # find id field if it exists otherwise set to feature index
                    if id_field is None:
                        idFields = ['id', 'ID', 'Id', 'Name', 'name']
                        id = None
                        for idField in idFields:
                            if idField in list(fdict.keys()):
                                id = fdict[idField]
                                break
                    else:
                        id = fdict[id_field]

                    if id is None:
                        id = str(feati)
                        fdict['ID'] = id

                    if False:
                        id = fdict['ID']
                        if id[0] == 'S' or id[:3] == 'TCH':
                            fdict['DegrClass'] = 'Severe'
                        elif id[0] == 'M':
                            fdict['DegrClass'] = 'Moderate'
                        elif id[0] == 'P' or id[:3] == 'INT':
                            fdict['DegrClass'] = 'Pristine'
                        else:
                            fdict['DegrClass'] = '?'

                geom = feat.GetGeometryRef()
                fdict['geom'] = geom.Clone()
                if geom is not None and (geom.GetGeometryType() == ogr.wkbPolygon):
                    print("%s Polygon with %d points"  % (id, geom.GetGeometryRef(0).GetPointCount()))
                    fdict['points'] = geom.GetGeometryRef(0).GetPoints()[:-1]
                    # pixCnr = []
                    # for point in f['points']:
                    #     pixCnr.append(World2Pixel(geotransform, point[0], point[1]))
                elif geom is not None and (geom.GetGeometryType() == ogr.wkbPoint or geom.GetGeometryType() == ogr.wkbPoint25D or geom.GetGeometryType() == ogr.wkbPointZM):
                    print("%s Point (%.6f, %.6f)" % (id, geom.GetX(), geom.GetY()))
                    fdict['X'] = geom.GetX()
                    fdict['Y'] = geom.GetY()
                    if False:    #'GNSS_Heigh' in fdict.keys():
                        fdict['Z'] = fdict['GNSS_Heigh']  # ? - should be able to get this from geom
                    else:
                        fdict['Z'] = geom.GetZ()  # this has been xformed from elippsoidal to Sa Geoid 2010
                    # f['ID'] = f['Datafile'][:-4] + str(f['Point_ID'])
                else:
                    print("unknown geometry/n")
                feat_dict[id] = fdict
                # print ' '

            self.layer_dict[layer.GetName()]={'feat_dict':feat_dict, 'spatial_ref':layer.GetSpatialRef()}
        return self.layer_dict

from enum import Enum

class GroundClass(Enum):
    Ground = 1
    DPlant = 2
    LPlant = 3
    Shadow = 4

# class to extract features from polygons in an raster
class ImPlotFeatureExtractor(object):
    def __init__(self, image_reader=GdalImageReader, plot_feat_dict={}):
        self.image_reader = image_reader
        self.plot_feat_dict = plot_feat_dict
        self.im_feat_dict = {}
        self.im_feat_count = 0

    # 1st im channel is dtm, 2nd im channel is dsm
    @staticmethod
    def extract_patch_clf_features(imbuf, mask, per_pixel=False):
        if mask is None:
            mask = np.any(imbuf>0, axis=2)  #np.ones(imbuf.shape[:2])
        mask = np.bool8(mask)

        imbuf_mask = np.ndarray(shape=(np.int32(mask.sum()), imbuf.shape[2]), dtype=np.float64)
        for i in range(0, imbuf.shape[2]):
            band = imbuf[:, :, i]
            imbuf_mask[:, i] = np.int32(band[mask])
        # imbuf_mask[:, 3] = imbuf_mask[:,  3]/2.
        # wv3 bands
        ground_classes = OrderedDict([('Ground', 1), ('DPlant', 2), ('LPlant', 3), ('Shadow', 4)])
        feat_dict = {}
        for cl_key, cl_num in ground_classes.items():
            feat_dict['{0}Cover'.format(cl_key)] = old_div(100*(imbuf_mask == cl_num).sum(),mask.sum())
        feat_dict['VegCover'] = old_div(100*((imbuf_mask == ground_classes['DPlant']) | (imbuf_mask == ground_classes['LPlant'])).sum(),mask.sum())
        return feat_dict

    @staticmethod
    # 1st im channel is dtm, 2nd im channel is dsm
    def extract_patch_dem_features(imbuf, mask=None, per_pixel=False):
        if mask is None:
            mask = np.any(imbuf>0, axis=2)  #np.ones(imbuf.shape[:2])
        mask = np.bool8(mask)

        imbuf_mask = np.ndarray(shape=(np.int32(mask.sum()), imbuf.shape[2]), dtype=np.float64)
        for i in range(0, imbuf.shape[2]):
            band = imbuf[:, :, i]
            imbuf_mask[:, i] = np.float64(band[mask])
        # imbuf_mask[:, 3] = imbuf_mask[:,  3]/2.
        # wv3 bands
        dtm = imbuf_mask[:, 0]
        dsm = imbuf_mask[:, 1]
        veg_height = dtm-dsm

        feat_dict = {}

        feat_dict['sum(veg.hgt)'] = veg_height.sum()
        feat_dict['mean(veg.hgt)'] = veg_height.mean()
        feat_dict['max(veg.hgt)'] = veg_height.max()
        feat_dict['min(veg.hgt)'] = veg_height.min()
        feat_dict['std(veg.hgt)'] = veg_height.std()
        if imbuf_mask.shape[1] == 3:
            tri = imbuf_mask[:, 2]
            feat_dict['mean(tri)'] = tri.mean()
            feat_dict['std(tri)'] = tri.mean()

        return feat_dict

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
        pan_bands, band_dict = ImPlotFeatureExtractor.get_band_info(imbuf.shape[2])

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
                    plot_feat_dict['({0})^2'.format(feat_key)] = feat_vect.mean()**2

                    if include_texture:
                        plot_feat_dict['Std({0})'.format(feat_key)] = feat_vect.std()
                        plot_feat_dict['Entropy({0})'.format(feat_key)] = entropy(feat_vect)

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


    @staticmethod
    def extract_patch_ms_features(imbuf, mask=None, per_pixel=False):
        feat_dict = {}
        L = 0.05
        if imbuf.shape[2] == 8:     # assume Wv3
            b_i = 1
            g_i = 2
            r_i = 4
            ir_i = 5
        else:                       # assume NGI
            b_i = 2
            g_i = 1
            r_i = 0
            ir_i = 3

        if mask is None:
            mask = np.any(imbuf>0, axis=2)  #np.ones(imbuf.shape[:2])
        mask = np.bool8(mask)
        # mask = mask & np.all(imbuf > 0, axis=2) & np.all(imbuf[:,:,], axis=2)
        imbuf_mask = np.ndarray(shape=(np.int32(mask.sum()), imbuf.shape[2]), dtype=np.float64)
        for i in range(0, imbuf.shape[2]):
            band = imbuf[:, :, i]
            imbuf_mask[:, i] = np.float64(band[mask])               # TODO HACK - check / 5000.  # 5000 is scale for MODIS / XCALIB
            # imbuf_mask[:, 3] = imbuf_mask[:,  3]/2.
            # wv3 bands
            # if np.any(imbuf_mask<0):
            #     print 'imbuf_mask < 0'
            # print np.where(imbuf_mask<0)
        # wv 3 channels
        s = np.sum(imbuf_mask[:, [b_i, g_i, r_i, ir_i]], 1)  # NNB only sum r,g,b as ir confuses things in g_n
        #  s = np.sum(imbuf_mask[:,:4], 1)   # ??? check this
        cn = old_div(imbuf_mask, np.tile(s[:, None], (1, imbuf_mask.shape[1])))

        ndvi = old_div((imbuf_mask[:, ir_i] - imbuf_mask[:, r_i]), (imbuf_mask[:, ir_i] + imbuf_mask[:, r_i]))
        ir_rat = old_div(imbuf_mask[:, ir_i], imbuf_mask[:, r_i])
        savi = old_div((1 + L) * (imbuf_mask[:, ir_i] - imbuf_mask[:, r_i]), (L + imbuf_mask[:, ir_i] + imbuf_mask[:, r_i]))
        feat_dict = {}
        if per_pixel:
            fn = lambda x: x
        else:
            fn = lambda x: x.mean()
            feat_dict['i_std'] = (old_div(s, imbuf_mask.shape[1])).std()
            feat_dict['NDVI_std'] = ndvi.std()

        feat_dict['R'] = fn(imbuf_mask[:, r_i])
        feat_dict['G'] = fn(imbuf_mask[:, g_i])
        feat_dict['B'] = fn(imbuf_mask[:, b_i])
        feat_dict['IR'] = fn(imbuf_mask[:, ir_i])
        feat_dict['r_n'] = fn(cn[:, r_i])
        feat_dict['g_n'] = fn(cn[:, g_i])
        feat_dict['b_n'] = fn(cn[:, b_i])
        feat_dict['ir_n'] = fn(cn[:, ir_i])
        feat_dict['NDVI'] = fn(ndvi)
        feat_dict['SAVI'] = fn(savi)
        feat_dict['ir_rat'] = fn(ir_rat)
        feat_dict['i'] = fn((old_div(s, imbuf_mask.shape[1])))

        return feat_dict

    # per_pixel = True, patch_fn = su.ImPlotFeatureExtractor.extract_patch_ms_features
    def extract_all_features(self, per_pixel=False, patch_fn=extract_patch_ms_features):
        # geotransform = ds.GetGeoTransform()
        transform = osr.CoordinateTransformation(self.plot_feat_dict['spatial_ref'], self.image_reader.spatial_ref)
        self.im_feat_count = 0
        self.im_feat_dict = {}
        # plotTagcDict = {}
        # class_labels = ['Pristine', 'Moderate', 'Severe']
        max_im_vals = np.zeros((self.image_reader.num_bands))
        for plot in list(self.plot_feat_dict['feat_dict'].values()):
            # transform plot corners into ds pixel space
            # plotCnrsWorld = plot['points']
            # if plot['ID'] == 'PV11':        #hack
            #     continue

            plot_cnrs_pixel = []
            for cnr in plot['points']:
                point = ogr.Geometry(ogr.wkbPoint)
                point.AddPoint(cnr[0], cnr[1])
                point.Transform(transform)              # xform into image projection
                (pixel, line) = self.image_reader.world_to_pixel(point.GetX(), point.GetY())
                plot_cnrs_pixel.append([pixel, line])

            plot_cnrs_pixel = np.array(plot_cnrs_pixel)
            # if all the points fall inside the image
            if np.all(plot_cnrs_pixel >= 0) and np.all(plot_cnrs_pixel[:, 0] < self.image_reader.width) \
                    and np.all(plot_cnrs_pixel[:, 1] < self.image_reader.height):  # and plot.has_key('Yc') and plot['Yc'] > 0.:

                # get rectangular window extents
                ul_cnr = np.int32(np.floor(np.min(plot_cnrs_pixel, 0)))
                lr_cnr = np.int32(np.ceil(np.max(plot_cnrs_pixel, 0)))
                plot_size_pixel = np.int32(lr_cnr - ul_cnr) + 1

                # make a mask for this plot
                img = Image.fromarray(np.zeros((plot_size_pixel[1], plot_size_pixel[0])))

                # Draw a rotated rectangle on the image.
                draw = ImageDraw.Draw(img)
                # rect = get_rect(x=120, y=80, width=100, height=40, angle=30.0)
                draw.polygon([tuple(np.round(p - ul_cnr)) for p in plot_cnrs_pixel], fill=1)
                # Convert the Image data to a numpy array.
                plot_mask = np.bool8(np.asarray(img))

                # if plot.has_key('Yc') and plot['Yc'] > 0:
                #     plot['YcPp'] = plot['Yc'] / plot_mask.sum()         # the average per pixel in the mask
                #     # plot['YcPm2'] = plot['Yc'] / (plot['Size'] ** 2)    # the average per m2 in the theoretical plot size

                # extrap cs to to per ha based on actual plot size
                # if 'Yc' in plot and 'LitterHa' in plot:
                if 'Abc' in plot and 'LitterHa' in plot:
                    litterHa = np.max([plot['LitterHa'], 0.])
                    abc = np.max([plot['Abc'], 0.])
                    plot_area_m2_ = plot_mask.sum()*(np.prod(np.abs(self.image_reader.pixel_size)))
                    plot_geom = plot['geom'].Clone()
                    plot_geom.Transform(transform)
                    plot_area_m2 = plot_geom.GetArea()
                    plot['AbcHa2'] = (old_div(abc * (100.**2), plot_area_m2))
                    plot['AgcHa2'] = litterHa + plot['AbcHa2']

                # else:
                #      print '%s - no yc' % (plot['ID'])

                # extract image patch with mask
                imbuf = self.image_reader.read_image_roi(col_range=[ul_cnr[0], lr_cnr[0]+1], row_range=[ul_cnr[1], lr_cnr[1]+1])
                # imbuf = np.zeros((plot_size_pixel[1], plot_size_pixel[0], self.image_reader.num_bands), dtype=float)
                # for b in range(0, self.image_reader.num_bands):
                #     imbuf[:, :, b] = self.image_reader. ds.GetRasterBand(b).ReadAsArray(ul_cnr[0], ul_cnr[1], plot_size_pixel[0],
                #                                                          plot_size_pixel[1])

                # imbuf[:, :, 3] = imbuf[:, :, 3] / 2  # hack for NGI XCALIB
                if np.all(imbuf == 0) and not patch_fn == self.extract_patch_clf_features:
                    print('Plot {0} has image data == zero, ommitting'.format(plot['ID']))
                    continue
                # for b in range(0, 4):
                #     imbuf[:, :, b] = imbuf[:, :, b] / max_im_vals_[b]
                if not plot_mask.shape == imbuf.shape[0:2]:
                    print("error - mask and buf different sizes")
                    raise Exception("error - mask and buf different sizes")

                # ignore <=0 pixels (sometimes pan sharp outputs <= 0)
                if patch_fn == self.extract_patch_clf_features:
                    plot_mask = plot_mask & np.all(~np.isnan(imbuf), axis=2)
                else:
                    plot_mask = plot_mask & np.all(imbuf > 0, axis=2) & np.all(~np.isnan(imbuf), axis=2)

                feat_dict = patch_fn(imbuf.copy(), mask=plot_mask, per_pixel=per_pixel)

                # copy across other plot fields into plot_dict
                plot_dict = OrderedDict(plot)
                plot_dict['feats'] = feat_dict
                # for f in plot.keys():
                #     feat_dict[f] = plot[f]

                plot_dict['thumbnail'] = np.float32(imbuf.copy())
                plot_dict['thumbnail'][~plot_mask] = 0.
                if not plot_mask.shape == plot_dict['thumbnail'].shape[0:2]:
                    print("error - mask and thumbnail different sizes")
                    raise Exception("error - mask and thumbnail different sizes")

                self.im_feat_dict[plot['ID']] = plot_dict
                # plotTagcDict[plot['PLOT']] = csGtDict[plot['PLOT']]['TAGC']

                # store max thumbnail vals for scaling later
                tmp = np.reshape(plot_dict['thumbnail'], (np.prod(plot_size_pixel), self.image_reader.num_bands))
                # max_tmp = tmp.max(axis=0)
                max_tmp = np.percentile(tmp, 98., axis=0)
                max_im_vals[max_tmp > max_im_vals] = max_tmp[max_tmp > max_im_vals]
                # print plot['PLOT']
                self.im_feat_count += 1

                log_msg = 'complete'
                if 'Abc' not in plot or plot['Abc'] <= 0:
                    log_msg += ', no Abc field'
                if np.any(imbuf == 0):
                    log_msg += ', np.any(imbuf == 0) count {0}'.format(np.sum(imbuf == 0))
                if np.any(imbuf < 0):
                    log_msg += ', np.any(imbuf < 0) count {0}'.format(np.sum(imbuf < 0))
                if np.any(np.isnan(imbuf)):
                    log_msg += ', np.any(np.isnan(imbuf)) count {0}'.format(np.sum(np.any(np.isnan(imbuf))))

                print("Plot {0}: {1}".format(plot['ID'], log_msg))
            else:
                print("Plot {0}: cannot include (outside image extent or no Abc data)".format(plot['ID']))

        print('Found features for {0} polygons'.format(self.im_feat_count))

        # scale thumbnails
        for k, v in self.im_feat_dict.items():
            # print "{0} - scaling thumbnail".format(v['ID'])
            thumb = v['thumbnail']
            # max_im_vals[1] = max_im_vals[1]
            for b in range(0, self.image_reader.num_bands):
                thumb[:, :, b] = old_div(thumb[:, :, b], max_im_vals[b])
                thumb[:, :, b][thumb[:, :, b] > 1.] = 1.
                thumb[:, :, b] = thumb[:, :, b]
            # thumb[:, :, 0] = thumb[:, :, 0] / 1.5
            # thumb[:, :, 1] = thumb[:, :, 1] * 1.2
            # thumb[:, :, 1][thumb[:, :, 1] > 1.] = 1.
            self.im_feat_dict[k]['thumbnail'] = thumb   #[:, :, [4, 2, 1]]

        return self.im_feat_dict


    # get array of non-lin function features
    def get_feat_array(self, feat_keys=None, y_feat_key=None, include_nonlin=True):
        if feat_keys is None:  # get all feats
            feat_keys = ['i', 'r_n', 'g_n', 'b_n', 'ir_n', 'ir_rat', 'SAVI', 'NDVI', 'i_std', 'NDVI_std']
            available_feat_keys = list(self.im_feat_dict.values())[0]['feats'].keys()
            feat_keys = np.intersect1d(feat_keys, available_feat_keys).tolist()

        y = []
        if y_feat_key is not None:
            y = np.hstack([np.tile(plot[y_feat_key], plot['feats'][feat_keys[0]].size) for plot in list(self.im_feat_dict.values())])

        X = []
        feat_keys_mod = []
        for feat_key in feat_keys:
            f = np.hstack([plot['feats'][feat_key] for plot in list(self.im_feat_dict.values())])
            X.append(f)
            feat_keys_mod.append(feat_key)

        if include_nonlin:
            for feat_key in feat_keys:
                f = np.hstack([plot['feats'][feat_key] for plot in list(self.im_feat_dict.values())])
                if feat_key == 'SAVI' or feat_key == 'NDVI':
                    f += 1.
                X.append(np.log10(f))
                feat_keys_mod.append('log10({0})'.format(feat_key))
            for feat_key in feat_keys:
                f = np.hstack([plot['feats'][feat_key] for plot in list(self.im_feat_dict.values())])
                X.append(f ** 2)
                feat_keys_mod.append(('{0}^2'.formatfeat_key))
            for feat_key in feat_keys:
                f = np.hstack([plot['feats'][feat_key] for plot in list(self.im_feat_dict.values())])
                if feat_key == 'SAVI' or feat_key == 'NDVI':
                    f += 1.
                X.append(np.sqrt(f))
                feat_keys_mod.append('{0}^.5'.format(feat_key))

        id = np.array([plot['ID'] for plot in list(self.im_feat_dict.values())])
        feat_keys_mod = np.array(feat_keys_mod)
        X = np.array(X).transpose()
        print('feat_array nan sum: {0}'.format(np.isnan(X).sum()))
        print('feat_array nan feats: ' + str(feat_keys_mod[np.any(np.isnan(X), axis=0)]))
        # print 'feat_array nan plots: ' + str(id[np.any(np.isnan(X), axis=1)])
        print('feat_array inf sum: {0}'.format((X == np.inf).sum()))
        print('feat_array inf feats: ' + str(feat_keys_mod[np.any((X == np.inf), axis=0)]))
        # print 'feat_array inf plots: ' + str(id[np.any((X == np.inf), axis=1)])
        return X, y, feat_keys_mod

    # this version goes with extract_patch_ms_features_ex
    def get_feat_array_ex(self, y_feat_key=None, feat_keys=None):
        if self.im_feat_dict is None or len(self.im_feat_dict) == 0:
            raise Exception('No features')

        if feat_keys is None:  # get all feats
            feat_keys = list(list(self.im_feat_dict.values())[0]['feats'].keys())

        y = []
        if y_feat_key is not None:
            y = np.hstack([np.tile(plot[y_feat_key], plot['feats'][feat_keys[0]].size) for plot in list(self.im_feat_dict.values())])

        X = []
        for feat_key in feat_keys:
            f = np.hstack([plot['feats'][feat_key] for plot in list(self.im_feat_dict.values())])
            X.append(f)

        id = np.array([plot['ID'] for plot in list(self.im_feat_dict.values())])
        feat_keys = np.array(feat_keys)
        X = np.array(X).transpose()
        print('feat_array nan sum: {0}'.format(np.isnan(X).sum()))
        print('feat_array nan feats: ' + str(feat_keys[np.any(np.isnan(X), axis=0)]))
        # print 'feat_array nan plots: ' + str(id[np.any(np.isnan(X), axis=1)])
        print('feat_array inf sum: {0}'.format((X == np.inf).sum()))
        print('feat_array inf feats: ' + str(feat_keys[np.any((X == np.inf), axis=0)]))
        # print 'feat_array inf plots: ' + str(id[np.any((X == np.inf), axis=1)])
        return X, y, feat_keys


    @staticmethod
    def sscatter_plot_(im_feat_dict, x_feat_key='NDVI', y_feat_key='', class_key='', show_labels=True, show_class_labels=True,
                     show_thumbnails=False, do_regress=True, xlabel=None, ylabel=None, xfn=lambda x: x, yfn=lambda y: y):
        x = np.array([xfn(plot['feats'][x_feat_key]) for plot in list(im_feat_dict.values())])
        if type(x[0]) is np.ndarray:        # this is pixel data and requires concat to flatten it
            cfn = lambda x: np.hstack(x)[::5]
            show_thumbnails = False
        else:
            cfn = lambda x: x

        # if xfn is not None:
        #     x = xfn(x)
        y = np.array([yfn(plot[y_feat_key]) for plot in list(im_feat_dict.values())])
        # if type(x[0]) is np.ndarray:
        #     ycfn = lambda x: np.concatenate(x)
        # else:
        #     ycfn = lambda x: x

        # if yfn is not None:
        #     y = yfn(y)

        if show_class_labels == True:
            class_labels = np.array([plot[class_key] for plot in list(im_feat_dict.values())])
        else:
            class_labels = np.zeros(x.__len__())
        if show_thumbnails == True:
            thumbnails = np.array([plot['thumbnail'] for plot in list(im_feat_dict.values())])

        if show_labels == True:
            labels = np.array([plot['ID'] for plot in list(im_feat_dict.values())])

        classes = np.unique(class_labels)
        colours = ['r', 'm', 'b', 'g', 'y', 'k', 'o']

        ylim = [np.min(cfn(y)), np.max(cfn(y))]
        xlim = [np.min(cfn(x)), np.max(cfn(x))]
        xd = np.diff(xlim)[0]
        yd = np.diff(ylim)[0]

        pylab.figure()
        pylab.axis(np.concatenate([xlim, ylim]))
        # pylab.hold('on')
        ax = pylab.gca()
        handles = np.zeros(classes.size).tolist()
        #

        for ci, (class_label, colour) in enumerate(zip(classes, colours[:classes.__len__()])):
            class_idx = class_labels == class_label
            if not show_thumbnails:
                pylab.plot(cfn(x[class_idx]), cfn(y[class_idx]), colour + 'o', label=class_label, markeredgecolor=(0, 0, 0))

            for xyi, (xx, yy) in enumerate(zip(x[class_idx], y[class_idx])):  # , np.array(plot_names)[class_idx]):
                if type(xx) is np.ndarray:
                    xx = xx[0]
                if type(yy) is np.ndarray:
                    yy = yy[0]
                if show_labels:
                    pylab.text(xx - .0015, yy - .0015, np.array(labels)[class_idx][xyi],
                               fontdict={'size': 9, 'color': colour, 'weight': 'bold'})

                if show_thumbnails:
                    imbuf = np.array(thumbnails)[class_idx][xyi]
                    band_idx = [0, 1, 2]
                    if imbuf.shape[2] == 8:  # guess wv3
                        band_idx = [4, 2, 1]

                    ims = 20.
                    extent = [xx - old_div(xd, (2 * ims)), xx + old_div(xd, (2 * ims)), yy - old_div(yd, (2 * ims)), yy + old_div(yd, (2 * ims))]
                    #pylab.imshow(imbuf[:, :, :3], extent=extent, aspect='auto')  # zorder=-1,
                    pylab.imshow(imbuf[:,:,band_idx], extent=extent, aspect='auto')  # zorder=-1,
                    handles[ci] = ax.add_patch(
                        patches.Rectangle((xx - old_div(xd, (2 * ims)), yy - old_div(yd, (2 * ims))), old_div(xd, ims), old_div(yd, ims), fill=False,
                                          edgecolor=colour, linewidth=2.))
                    # pylab.plot(mPixels[::step], dRawPixels[::step], color='k', marker='.', linestyle='', markersize=.5)
            if do_regress and classes.__len__() > 1 and False:
                (slope, intercept, r, p, stde) = stats.linregress(cfn(x[class_idx]), cfn(y[class_idx]))
                pylab.text(xlim[0] + xd * 0.7, ylim[0] + yd * 0.05 * (ci + 2),
                           '{1}: $R^2$ = {0:.2f}'.format(np.round(r ** 2, 2), classes[ci]),
                           fontdict={'size': 10, 'color': colour})

        if do_regress:
            (slope, intercept, r, p, stde) = stats.linregress(cfn(x), cfn(y))
            pylab.text((xlim[0] + xd * 0.7), (ylim[0] + yd * 0.05), '$R^2$ = {0:.2f}'.format(np.round(r ** 2, 2)),
                       fontdict={'size': 12})
            print('R^2 = {0:.4f}'.format(r ** 2))
            print('P (slope=0) = {0:f}'.format(p))
            print('Slope = {0:.4f}'.format(slope))
            print('Std error of slope = {0:.4f}'.format(stde))
            yhat = cfn(x)*slope + intercept
            rmse = np.sqrt(np.mean((cfn(y) - yhat) ** 2))
            print('RMS error = {0:.4f}'.format(rmse))
        else:
            r = np.nan
            rmse = np.nan

        if xlabel is not None:
            pylab.xlabel(xlabel, fontdict={'size': 12})
        else:
            pylab.xlabel(x_feat_key, fontdict={'size': 12})
        if ylabel is not None:
            pylab.ylabel(ylabel, fontdict={'size': 12})
        else:
            pylab.ylabel(y_feat_key, fontdict={'size': 12})
        # pylab.ylabel(yf)
        pylab.grid()
        if classes.size > 0:
            if show_thumbnails:
                pylab.legend(handles, classes, fontsize=12)
            else:
                pylab.legend(classes, fontsize=12)
        return r**2, rmse

    @staticmethod
    def sscatter_plot(im_feat_dict, x_feat_key='NDVI', y_feat_key='', class_key='', show_labels=True, show_class_labels=True,
                     show_thumbnails=False, do_regress=True, xlabel=None, ylabel=None, xfn=lambda x: x, yfn=lambda y: y):
        x = np.array([xfn(plot['feats'][x_feat_key]) for plot in list(im_feat_dict.values())])
        y = np.array([yfn(plot[y_feat_key]) for plot in list(im_feat_dict.values())])

        if show_class_labels == True:
            class_labels = np.array([plot[class_key] for plot in list(im_feat_dict.values())])
        else:
            class_labels = None

        if show_thumbnails == True:
            thumbnails = np.array([plot['thumbnail'] for plot in list(im_feat_dict.values())])
        else:
            thumbnails = None

        if show_labels == True:
            labels = np.array([plot['ID'] for plot in list(im_feat_dict.values())])
        else:
            labels = None

        if xlabel is None:
            xlabel = x_feat_key
        if ylabel is None:
            ylabel = y_feat_key
        # pylab.ylabel(yf)
        return scatter_plot(x, y, class_labels=class_labels, labels=labels, thumbnails=thumbnails,
                                                   do_regress=do_regress, xlabel=xlabel, ylabel=ylabel, xfn=xfn, yfn=yfn)


    def scatter_plot(self, x_feat_key='NDVI', y_feat_key='', class_key='', show_labels=True, show_class_labels=True,
                     show_thumbnails=False, do_regress=True, xlabel=None, ylabel=None, xfn=lambda x: x, yfn=lambda y: y):
        return ImPlotFeatureExtractor.sscatter_plot(self.im_feat_dict, x_feat_key=x_feat_key, y_feat_key=y_feat_key, class_key=class_key, show_labels=show_labels, show_class_labels=show_class_labels,
                     show_thumbnails=show_thumbnails, do_regress=do_regress, xlabel=xlabel, ylabel=ylabel, xfn=xfn, yfn=yfn)

class FeatureSelector(object):
    def __init__(self):
        return

    @staticmethod
    def forward_selection(X, y, feat_keys=None, max_num_feats=0, model=linear_model.LinearRegression(),
                          score_fn=lambda y, pred: -1*np.sqrt(metrics.mean_squared_error(y,pred)), cv=None):
        # X, feat_keys_mod, y = self.get_feat_array(y_key=y_feat_key)
        if feat_keys is None:
            feat_keys = [str(i) for i in range(0, X.shape[1])]
        feat_list = X.transpose().tolist()
        feat_dict = dict(list(zip(feat_keys, feat_list)))
        if max_num_feats == 0:
            max_num_feats = X.shape[1]
        selected_feats = collections.OrderedDict()   # remember order items are added
        selected_scores = []
        available_feats = feat_dict

        print('Forward selection: ', end=' ')
        while len(selected_feats) < max_num_feats:
            best_score = -np.inf
            best_feat = []
            for feat_key, feat_vec in available_feats.items():
                test_feats = list(selected_feats.values()) + [feat_vec]
                scores, predicted = FeatureSelector.score_model(np.array(test_feats).transpose(), y, model=model, score_fn=score_fn, cv=cv, find_predicted=False)
                score = scores['test_user'].mean()
                if score > best_score:
                    best_score = score
                    best_feat = list(feat_vec)
                    best_key = feat_key
            selected_feats[best_key] = best_feat
            selected_scores.append(best_score)
            available_feats.pop(best_key)
            print(best_key + ', ', end=' ')
        print(' ')
        selected_scores = np.array(selected_scores)
        selected_feat_keys = list(selected_feats.keys())
        best_selected_feat_keys = selected_feat_keys[:np.argmax(selected_scores) + 1]
        print('Best score: {0}'.format(selected_scores.max()))
        print('Num feats at best score: {0}'.format(np.argmax(selected_scores) + 1))
        print('Feat keys at best score: {0}'.format(best_selected_feat_keys))

        return np.array(list(selected_feats.values())).transpose(), selected_scores, selected_feat_keys

    @staticmethod
    def ranking(X, y, feat_keys=None, model=linear_model.LinearRegression(),
                score_fn=lambda y, pred: -1*np.sqrt(metrics.mean_squared_error(y,pred)), cv=None):
        # X, feat_keys_mod, y = self.get_feat_array(y_key=y_feat_key)
        if feat_keys is None:
            feat_keys = [str(i) for i in range(0, X.shape[1])]
        feat_list = X.transpose().tolist()
        feat_dict = OrderedDict(list(zip(feat_keys, feat_list)))
        feat_scores = []
        for i, (feat_key, feat_vect) in enumerate(feat_dict.items()):
            if False:
                score = score_fn(y, np.array(feat_vect).reshape(-1, 1))
            else:
                scores, predicted = FeatureSelector.score_model(np.array(feat_vect).reshape(-1, 1), y.reshape(-1, 1), model=model,
                                                           score_fn=score_fn, cv=cv, find_predicted=True)
                score = scores['test_user'].mean()
                # score = scores['R2_stacked']        #hack
            feat_scores.append(score)
            if i%20==0:
                print('.', end=' ')
        print(' ')
        feat_scores = np.array(feat_scores)

        print('Best score: {0}'.format(feat_scores.max()))
        print('Best feat: {0}'.format(feat_keys[np.argmax(feat_scores)]))
        return feat_scores

    @staticmethod
    def score_model(X, y, model=linear_model.LinearRegression(), score_fn=lambda y, pred: -1*np.sqrt(metrics.mean_squared_error(y,pred)),
                    cv=None, find_predicted=True, print_scores=False):

        # X = np.array(feat_list).transpose()
        # y = np.array([plot[y_feat_key] for plot in self.im_feat_dict.values()])
        # y = np.hstack([np.tile(plot[y_feat_key], plot[feat_keys[0]].size) for plot in self.im_feat_dict.values()])
        predicted = None
        if sys.version_info.major == 3:
            from sklearn.metrics import make_scorer
        else:
            from sklearn.metrics.scorer import make_scorer
        if cv is None:
            cv = y.__len__()        # Leave one out

        scoring = {'R2': make_scorer(metrics.r2_score),
                   '-RMSE': make_scorer(lambda y, pred: -np.sqrt(metrics.mean_squared_error(y, pred))),
                   'user': make_scorer(score_fn)}
        scores = cross_validate(model, X, y, scoring=scoring, cv=cv)
        if print_scores:
            rmse_ci = np.percentile(-scores['test_-RMSE'], [5, 95])
            r2_ci = np.percentile(-scores['test_R2'], [5, 95])
            print('RMSE: {0:.4f} ({1:.4f})'.format(-scores['test_-RMSE'].mean(), scores['test_-RMSE'].std()))
            print('RMSE 5-95%: {0:.4f} - {1:.4f}'.format(rmse_ci[0], rmse_ci[1]))
            print('R2 (average over folds): {0:.4f} ({1:.4f})'.format(scores['test_R2'].mean(), scores['test_R2'].std()))
            print('R2 5-95%: {0:.4f} - {1:.4f}'.format(r2_ci[0], r2_ci[1]))
        if find_predicted:
            predicted = cross_val_predict(model, X, y, cv=cv)  #)
            scores['R2_stacked'] = metrics.r2_score(y, predicted)   # DO NOT USE FOR VALIDATION
            if print_scores:
                print('R2 (stacked): {0:.4f}'.format(scores['R2_stacked']))
        return scores, predicted
        # score = {}
        # score = score_fn(y, predicted)
        # score['r2'] = metrics.r2_score(y, predicted)
        # score['rmse'] = np.sqrt(metrics.mean_squared_error(y, predicted))
        # score['mse'] = (-scores['test_neg_mean_squared_error'])
        # score['rms'] = np.sqrt(-scores['test_neg_mean_squared_error'])


    # assumes X is 2D and scaled 0-1, clf is a trained classifier
    @staticmethod
    def plot_clf(clf, X, y, feat_keys=None, class_labels=None):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from matplotlib.colors import ListedColormap

        if X.shape[1] > 2:
            raise Exception('X.shape[1] > 2')
        if X.shape[1] > 2:
            raise Exception('X.shape[1] > 2')
        h = 0.002
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # just plot the dataset first
        cm = plt.cm.Set2
        plt.figure()
        ax = plt.subplot(1, 1, 1)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        # if hasattr(clf, "decision_function"):
        #     Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        # else:
        #     # Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        # ax.contourf(xx, yy, Z, cmap=cm, alpha=1, levels=[0,1,2,3,4,5])
        ax.imshow(Z, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap=cm, aspect='auto')

        # Plot the training points
        s = 100
        colours = cm(np.linspace(0., 1., np.unique(y).size))  #make same colours as contourf
        h = []
        for i, class_id in enumerate(np.unique(y)):
            class_idx = y == class_id
            h.append(ax.scatter(X[class_idx, 0], X[class_idx, 1], color=colours[i], edgecolor='black', s=20))

        # ax.scatter(X[::s, 0], X[::s, 1], c=y[::s], cmap=cm, edgecolor='black', s=15)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        if feat_keys is not None:
            ax.set_xlabel(feat_keys[0])
            ax.set_ylabel(feat_keys[1])

        # from matplotlib.lines import
        # custom_lines = [Line2D([0], [0], color=cm(0.), lw=4),
        #                 Line2D([0], [0], color=cm(.5), lw=4),
        #                 Line2D([0], [0], color=cm(1.), lw=4)]

        # fig, ax = plt.subplots()
        # lines = ax.plot(data)
        if class_labels is not None:
            ax.legend(h, class_labels)
        # ax.set_xticks(())
        # ax.set_yticks(())


# params: calib_plots from >1 data set
#       model_data_plots from >1 data set
#       (for now the above are the same thing)
#       1 data_set is specified as fitted one, the rest are tests, this can also be done sort of cross-validated
#       a model spec i.e. feature indices and model type
#       num calib plots to use

class ModelCalibrationTest(object):
    # def __init__(self, plot_featdict_list=[], y_key='', calib_featdict_list=[], feat_keys='', model_feat_keys=['r_n'], model=linear_model.LinearRegression()):
    #     self.plot_data_list = plot_data_list
    #     self.calib_data_list = calib_data_list
    #     self.feat_keys = feat_keys
    #     self.model_feat_idx = model_feat_idx
    #     self.model = model
    #     self.y = y
    #     self.fitted_models = []

    def __init__(self, plot_data_list=[], y=[], strata=None, calib_data_list=[], feat_keys='', model_feat_idx=[0], model=linear_model.LinearRegression):
        self.plot_data_list = plot_data_list
        self.calib_data_list = calib_data_list
        self.feat_keys = feat_keys
        self.model_feat_idx = model_feat_idx
        self.model = model
        self.y = y
        self.fitted_models = []
        self.strata = strata
        self.model_scores_array = None
        self.calib_scores_array = None

    def BootStrapCalibration(self, fit_model, fit_plot_data, fit_calib_data, test_plot_data, test_calib_data,
                                     n_bootstraps=10, n_calib_plots=10):
        r2_model = np.zeros((n_bootstraps, 1))
        rmse_model = np.zeros((n_bootstraps, 1))

        r2_calib = np.zeros((n_bootstraps, self.model_feat_idx.__len__()))
        rmse_calib = np.zeros((n_bootstraps, self.model_feat_idx.__len__()))
        # TO DO: make a sub-function
        # sample with bootstrapping the calib plots, fit calib models, apply to plot_data and test
        for bi in range(0, n_bootstraps):
            if self.strata is None:
                calib_plot_idx = np.random.permutation(test_plot_data.__len__())[:n_calib_plots]
            else:
                calib_plot_idx = []
                strata_list = np.unique(self.strata)
                for strata_i in strata_list:
                    strata_idx = np.int32(np.where(strata_i == self.strata)[0])
                    calib_plot_idx += np.random.permutation(strata_idx)[
                                      :np.round(old_div(n_calib_plots, strata_list.__len__()))].tolist()

                calib_plot_idx = np.array(calib_plot_idx)
            calib_feats = []
            # loop through features in model_feat_idx and calibrate each one
            calib_feats = np.zeros((test_calib_data.shape[0], self.model_feat_idx.__len__()))
            for fi, feat_idx in enumerate(self.model_feat_idx):
                calib_model = linear_model.LinearRegression()
                calib_model.fit(test_calib_data[calib_plot_idx, feat_idx].reshape(-1, 1),
                                fit_calib_data[calib_plot_idx, feat_idx].reshape(-1, 1))
                calib_feat = calib_model.predict(test_plot_data[:, feat_idx].reshape(-1, 1))
                r2_calib[bi, fi] = metrics.r2_score(fit_plot_data[:, feat_idx], calib_feat)
                rmse_calib[bi, fi] = np.sqrt(
                    metrics.mean_squared_error(fit_plot_data[:, feat_idx], calib_feat))
                calib_feats[:, fi] = calib_feat.flatten()

            # calib_feats = np.array(calib_feats).transpose()
            predicted = fit_model.predict(calib_feats)
            r2_model[bi] = metrics.r2_score(self.y, predicted)
            rmse_model[bi] = np.sqrt(metrics.mean_squared_error(self.y, predicted))

        model_scores = {'r2':r2_model, 'rmse': rmse_model}
        calib_scores = {'r2':r2_calib, 'rmse': rmse_calib}
        return model_scores, calib_scores

    def TestCalibration(self, n_bootstraps=10, n_calib_plots=10):
        np.set_printoptions(precision=4)
        self.fitted_models = []
        for plot_data in self.plot_data_list:
            fit_model = self.model()
            fit_model.fit(plot_data[:, self.model_feat_idx], self.y)
            self.fitted_models.append(fit_model)

        model_idx = list(range(0, self.fitted_models.__len__()))
        # loop over different images/models to fit to
        self.model_scores_array = np.zeros((self.fitted_models.__len__(), self.fitted_models.__len__()), dtype=object)
        self.calib_scores_array = np.zeros((self.fitted_models.__len__(), self.fitted_models.__len__()), dtype=object)

        for fmi in model_idx:
            fit_model = self.fitted_models[fmi]
            test_model_idx = np.setdiff1d(model_idx, fmi)
            fit_calib_data = self.calib_data_list[fmi]
            fit_plot_data = self.plot_data_list[fmi]
            # loop over the remaining images/models to test calibration on
            for tmi in test_model_idx:
                test_calib_data = self.calib_data_list[tmi]
                test_plot_data = self.plot_data_list[tmi]

                model_scores, calib_scores = self.BootStrapCalibration(fit_model, fit_plot_data, fit_calib_data, test_plot_data,
                                                           test_calib_data, n_bootstraps=n_bootstraps, n_calib_plots=n_calib_plots)
                self.model_scores_array[fmi, tmi] = {'mean(r2)': model_scores['r2'].mean(), 'std(r2)': model_scores['r2'].std(),
                                                'mean(rmse)': model_scores['rmse'].mean(), 'std(rmse)': model_scores['rmse'].std()}
                self.calib_scores_array[fmi, tmi] = {'mean(r2)': calib_scores['r2'].mean(axis=0), 'std(r2)': calib_scores['r2'].std(axis=0),
                                                'mean(rmse)': calib_scores['rmse'].mean(axis=0), 'std(rmse)': calib_scores['rmse'].std(axis=0)}
                print('Model scores (fit model {0}, calib model {1})'.format(fmi, tmi))
                print('mean(R^2): {0:.4f}'.format(model_scores['r2'].mean()))
                print('std(R^2): {0:.4f}'.format(model_scores['r2'].std()))
                print('mean(RMSE): {0:.4f}'.format(model_scores['rmse'].mean()))
                print('std(RMSE): {0:.4f}'.format(model_scores['rmse'].std()))
                print(' ')
                print('Calib scores (fit model {0}, calib model {1})'.format(fmi, tmi))
                print('mean(R^2): {0}'.format(calib_scores['r2'].mean(axis=0)))
                print('std(R^2): {0}'.format(calib_scores['r2'].std(axis=0)))
                print('mean(RMSE): {0}'.format(calib_scores['rmse'].mean(axis=0)))
                print('std(RMSE): {0}'.format(calib_scores['rmse'].std(axis=0)))
                print(' ')
        return self.model_scores_array, self.calib_scores_array

    def PrintScores(self):
        for scores_array, label in zip([self.model_scores_array, self.calib_scores_array], ['Model', 'Calib']):
            for key in ['mean(r2)', 'std(r2)', 'mean(rmse)', 'std(rmse)']:
                print('{0} {1}:'.format(label, key))
                score_array = np.zeros(scores_array.shape)
                for ri in range(scores_array.shape[0]):
                    for ci in range(scores_array.shape[1]):
                        if scores_array[ri, ci] is None:
                            score_array[ri, ci] = 0.
                        else:
                            score_array[ri, ci] = scores_array[ri, ci][key]
                print(score_array)
                overall_mean = np.diag(np.flipud(score_array)).mean()
                print('Overall mean({0} {1}): {2:0.4f}'.format(label, key, overall_mean))
                print(' ')

        return


class ModelCalibrationTestEx(object):
    # def __init__(self, plot_featdict_list=[], y_key='', calib_featdict_list=[], feat_keys='', model_feat_keys=['r_n'], model=linear_model.LinearRegression()):
    #     self.plot_data_list = plot_data_list
    #     self.calib_data_list = calib_data_list
    #     self.feat_keys = feat_keys
    #     self.model_feat_idx = model_feat_idx
    #     self.model = model
    #     self.y = y
    #     self.fitted_models = []

    def __init__(self, plot_data_list=[], y=[], strata=None, calib_data_list=[], model=linear_model.LinearRegression):
        self.plot_data_list = plot_data_list
        self.calib_data_list = calib_data_list
        self.model = model
        self.y = y
        self.fitted_models = []
        self.strata = strata
        self.model_scores_array = None
        self.calib_scores_array = None

    def BootStrapCalibration(self, fit_model, fit_plot_data, fit_calib_data, test_plot_data, test_calib_data,
                                     n_bootstraps=10, n_calib_plots=10):
        r2_model = np.zeros((n_bootstraps, 1))
        rmse_model = np.zeros((n_bootstraps, 1))

        r2_calib = np.zeros((n_bootstraps, test_calib_data.shape[1]))
        rmse_calib = np.zeros((n_bootstraps, test_calib_data.shape[1]))
        # TO DO: make a sub-function
        # sample with bootstrapping the calib plots, fit calib models, apply to plot_data and test
        for bi in range(0, n_bootstraps):
            if self.strata is None:
                calib_plot_idx = np.random.permutation(test_plot_data.__len__())[:n_calib_plots]
            else:
                calib_plot_idx = []
                strata_list = np.unique(self.strata)
                for strata_i in strata_list:
                    strata_idx = np.int32(np.where(strata_i == self.strata)[0])
                    calib_plot_idx += np.random.permutation(strata_idx)[
                                      :np.round(old_div(n_calib_plots, strata_list.__len__()))].tolist()

                calib_plot_idx = np.array(calib_plot_idx)
            # test_plot_idx = np.setdiff1d(np.arange(0, len(self.y)), calib_plot_idx)   # exclude the calib plots
            test_plot_idx = np.arange(0, len(self.y))   # include the calib plots
            calib_feats = []
            # loop through features in model_feat_idx and calibrate each one
            # calib_feats = np.zeros((test_calib_data.shape[0], test_calib_data.shape[1]))
            calib_feats = np.zeros((len(test_plot_idx), test_calib_data.shape[1]))
            for fi in range(0, test_calib_data.shape[1]):
                calib_model = linear_model.LinearRegression()
                calib_model.fit(test_calib_data[calib_plot_idx, fi].reshape(-1, 1),
                                fit_calib_data[calib_plot_idx, fi].reshape(-1, 1))

                calib_feat = calib_model.predict(test_plot_data[test_plot_idx, fi].reshape(-1, 1))
                r2_calib[bi, fi] = metrics.r2_score(fit_plot_data[test_plot_idx, fi], calib_feat)
                rmse_calib[bi, fi] = np.sqrt(
                    metrics.mean_squared_error(fit_plot_data[test_plot_idx, fi], calib_feat))
                calib_feats[:, fi] = calib_feat.flatten()

            # calib_feats = np.array(calib_feats).transpose()
            predicted = fit_model.predict(calib_feats)
            r2_model[bi] = metrics.r2_score(self.y[test_plot_idx], predicted)
            rmse_model[bi] = np.sqrt(metrics.mean_squared_error(self.y[test_plot_idx], predicted))

        model_scores = {'r2':r2_model, 'rmse': rmse_model}
        calib_scores = {'r2':r2_calib, 'rmse': rmse_calib}
        return model_scores, calib_scores

    def TestCalibration(self, n_bootstraps=10, n_calib_plots=10):
        np.set_printoptions(precision=4)
        self.fitted_models = []
        for plot_data in self.plot_data_list:
            fit_model = self.model()
            fit_model.fit(plot_data, self.y)
            self.fitted_models.append(fit_model)

        model_idx = list(range(0, self.fitted_models.__len__()))
        # loop over different images/models to fit to
        self.model_scores_array = np.zeros((self.fitted_models.__len__(), self.fitted_models.__len__()), dtype=object)
        self.calib_scores_array = np.zeros((self.fitted_models.__len__(), self.fitted_models.__len__()), dtype=object)

        for fmi in model_idx:
            fit_model = self.fitted_models[fmi]
            test_model_idx = np.setdiff1d(model_idx, fmi)
            fit_calib_data = self.calib_data_list[fmi]
            fit_plot_data = self.plot_data_list[fmi]
            # loop over the remaining images/models to test calibration on
            for tmi in test_model_idx:
                test_calib_data = self.calib_data_list[tmi]
                test_plot_data = self.plot_data_list[tmi]

                model_scores, calib_scores = self.BootStrapCalibration(fit_model, fit_plot_data, fit_calib_data, test_plot_data,
                                                           test_calib_data, n_bootstraps=n_bootstraps, n_calib_plots=n_calib_plots)
                self.model_scores_array[fmi, tmi] = {'mean(r2)': model_scores['r2'].mean(), 'std(r2)': model_scores['r2'].std(),
                                                'mean(rmse)': model_scores['rmse'].mean(), 'std(rmse)': model_scores['rmse'].std()}
                self.calib_scores_array[fmi, tmi] = {'mean(r2)': calib_scores['r2'].mean(axis=0), 'std(r2)': calib_scores['r2'].std(axis=0),
                                                'mean(rmse)': calib_scores['rmse'].mean(axis=0), 'std(rmse)': calib_scores['rmse'].std(axis=0)}
                print('Model scores (fit model {0}, calib model {1})'.format(fmi, tmi))
                print('mean(R^2): {0:.4f}'.format(model_scores['r2'].mean()))
                print('std(R^2): {0:.4f}'.format(model_scores['r2'].std()))
                print('mean(RMSE): {0:.4f}'.format(model_scores['rmse'].mean()))
                print('std(RMSE): {0:.4f}'.format(model_scores['rmse'].std()))
                print(' ')
                print('Calib scores (fit model {0}, calib model {1})'.format(fmi, tmi))
                print('mean(R^2): {0}'.format(calib_scores['r2'].mean(axis=0)))
                print('std(R^2): {0}'.format(calib_scores['r2'].std(axis=0)))
                print('mean(RMSE): {0}'.format(calib_scores['rmse'].mean(axis=0)))
                print('std(RMSE): {0}'.format(calib_scores['rmse'].std(axis=0)))
                print(' ')
        return self.model_scores_array, self.calib_scores_array

    def PrintScores(self):
        for scores_array, label in zip([self.model_scores_array, self.calib_scores_array], ['Model', 'Calib']):
            for key in ['mean(r2)', 'std(r2)', 'mean(rmse)', 'std(rmse)']:
                print('{0} {1}:'.format(label, key))
                score_array = np.zeros(scores_array.shape)
                for ri in range(scores_array.shape[0]):
                    for ci in range(scores_array.shape[1]):
                        if scores_array[ri, ci] is None or type(scores_array[ri, ci]) is int:
                            score_array[ri, ci] = 0.
                        else:
                            score_array[ri, ci] = scores_array[ri, ci][key]
                print(score_array)
                overall_mean = np.diag(np.flipud(score_array)).mean()
                print('Overall mean({0} {1}): {2:0.4f}'.format(label, key, overall_mean))
                print(' ')

        return

class AgcMap(object):
    def __init__(self, in_file_name='', out_file_name='', model=linear_model.LinearRegression, model_keys=[],
                 feat_ex_fn=ImPlotFeatureExtractor.extract_patch_ms_features_ex, num_bands=9, save_feats=False):
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
    def ConstructFeatExFn(model_key='', pan_bands=None, band_dict=None):
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
    def ConstructFeatExFns(model_keys=[], num_b=9):
        import re
        pan_bands, band_dict = ImPlotFeatureExtractor.get_band_info(num_bands)

        win_fn_list = []
        inner_str_list = []
        inner_fn_list = []
        for model_key in model_keys:
            # find outer fn
            inner_str, win_fn, inner_fn = AgcMap.ConstructFeatExFn(model_key, pan_bands=pan_bands, band_dict=band_dict)
            win_fn_list.append(win_fn)
            inner_str_list.append(inner_str)
            inner_fn_list.append(inner_fn)
        # inner fn
        return win_fn_list, inner_fn_list

    # @staticmethod
    # def RollingWindow(a, window, step_size=1):
    #     shape = a.shape[:-1] + (a.shape[-1] - window + 1 - step_size + 1, window)
    #     strides = a.strides + (a.strides[-1] * step_size,)
    #     return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides, writeable=False)

    @staticmethod
    def RollingWindow(a, window, step_size=1):
        shape = a.shape[:-1] + (int(1 + (a.shape[-1] - window) / step_size), window)
        strides = a.strides[:-1] + (step_size * a.strides[-1], a.strides[-1])
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides, writeable=False)

    def FeatEx(self, im_buf=[]):
        return

    def Create(self, win_size=(1, 1), step_size=(1, 1)):
        from rasterio.windows import Window
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

                # self.win_fn_list, self.band_ratio_list = AgcMap.ConstructFeatExFns(self.model_keys, num_bands=in_ds.count)

                with rasterio.open(self.out_file_name, 'w', **out_profile) as out_ds:
                    pan_bands, band_dict = ImPlotFeatureExtractor.get_band_info(in_ds.count)
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
                            inner_str, win_fn, band_ratio_fn = AgcMap.ConstructFeatExFn(model_key, pan_bands=pan_bands,
                                                                                        band_dict=band_dict)
                            band_ratio = band_ratio_fn(pan, in_buf)
                            band_ratio[in_nan_mask] = np.nan           # to exclude from window stats/fns
                            feat_buf = win_fn(AgcMap.RollingWindow(band_ratio, win_size[0], step_size=step_size[0])) * self.model.coef_[i]
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

    def PostProc(self):
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